import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, Optional
import math


class PolarTransform(nn.Module):
    """
    Polar coordinate transformation module

    Converts a satellite image (Cartesian coordinates) into a polar coordinate representation, simulating a ground-level view

    Principle:
    - Satellite image: top-down view, the center is the capture location
    - Polar transformation: rearranges the surrounding scene by azimuth angle and radial distance
    - Result: a viewpoint similar to a ground-level panorama

    Input: satellite_image [B, 3, H, W]
    Output: polar_image [B, 3, H_polar, W_polar]

    Parameters:
        - output_size: polar image size (height, width)
          - height: corresponds to radial distance
          - width: corresponds to azimuth angle (usually 360 degrees)
        - center: transformation center point (x, y), defaults to the image center
        - input_size: input image size (H, W), used to precompute the grid (recommended to set to the actual dataset size)
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (128, 256),
        center: Optional[Tuple[float, float]] = None,
        max_radius: Optional[float] = None,
        input_size: Tuple[int, int] = (128, 128)  # ← new: input image size
    ):
        super().__init__()

        self.output_size = output_size
        self.center = center
        self.input_size = input_size  # ← store the input size

        # Maximum radial distance (computed from the input size)
        if max_radius is None:
            H, W = input_size
            center_x = W / 2.0 if center is None else center[0]
            center_y = H / 2.0 if center is None else center[1]
            max_radius = math.sqrt(center_x**2 + center_y**2)

        self.max_radius = max_radius

        # Precompute the grid at initialization (optimization: avoids runtime computation)
        H, W = input_size
        grid = self._create_polar_grid(H, W, 1, device='cpu')  # compute on CPU first
        self.register_buffer('grid', grid)  # register as a buffer (moved to the correct device automatically)

        print(f"[PolarTransform] Precomputed grid: input_size={input_size} -> output_size={output_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: satellite image [B, C, H, W]

        Returns:
            polar_image: polar image [B, C, H_polar, W_polar]
        """
        B, C, H, W = x.shape

        # Validate the input size (optional, for debugging)
        if (H, W) != self.input_size:
            raise ValueError(
                f"Input size mismatch! Expected {self.input_size}, got {(H, W)}. "
                f"Please set the correct input_size parameter when initializing PolarTransform."
            )

        # Expand the grid to the current batch size (zero-copy, very fast)
        if self.grid.shape[0] != B:
            grid_expanded = self.grid.expand(B, -1, -1, -1)  # [B, H_polar, W_polar, 2]
        else:
            grid_expanded = self.grid

        # Apply bilinear interpolation for the polar transformation
        # grid_sample expects input [B, C, H, W] and grid [B, H, W, 2]
        polar_image = F.grid_sample(
            x,
            grid_expanded,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return polar_image

    def _create_polar_grid(
        self,
        H: int,
        W: int,
        B: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create the polar coordinate grid

        Args:
            H: input height
            W: input width
            B: batch size
            device: compute device

        Returns:
            grid: sampling grid [B, H_polar, W_polar, 2]
        """
        H_polar, W_polar = self.output_size

        # Determine the transformation center
        if self.center is None:
            center_x = W / 2.0
            center_y = H / 2.0
        else:
            center_x, center_y = self.center

        # Generate radial and angular coordinates
        # Radial coordinates: from 0 to max_radius
        r = torch.linspace(0, self.max_radius if self.max_radius else min(H, W) / 2, H_polar, device=device)

        # Angular coordinates: from 0 to 2π (corresponding to 360 degrees)
        theta = torch.linspace(0, 2 * math.pi, W_polar, device=device)

        # Create the mesh grid
        r_grid, theta_grid = torch.meshgrid(r, theta, indexing='ij')

        # Polar to Cartesian coordinates
        # x = r * cos(theta) + center_x
        # y = r * sin(theta) + center_y
        x = r_grid * torch.cos(theta_grid) + center_x
        y = r_grid * torch.sin(theta_grid) + center_y

        # Normalize to [-1, 1] (required by grid_sample)
        x_norm = 2 * x / (W - 1) - 1
        y_norm = 2 * y / (H - 1) - 1

        # Assemble the grid [B, H_polar, W_polar, 2]
        # the last dimension of the grid holds (x, y) coordinates
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        return grid


class PolarTransformSVIPredictor(nn.Module):
    """
    SVI prediction model based on polar coordinate transformation

    Architecture:
    1. Polar transformation: satellite image → ground-view image
    2. ResNet50 backbone: feature extraction
    3. Classification head: predicts building, sky, and vegetation fractions

    Data flow:
    satellite_rgb [B, 3, H, W]
        ↓
    PolarTransform → polar_image [B, 3, H_polar, W_polar]
        ↓
    ResNet50 → features [B, 2048]
        ↓
    ClassificationHead → predictions [B, 3]
    """

    def __init__(
        self,
        # Polar transformation parameters
        polar_output_size: Tuple[int, int] = (128, 256),
        input_size: Tuple[int, int] = (128, 128),  # ← new: input image size
        # ResNet parameters
        pretrained: bool = False,
        freeze_backbone: bool = False,
        # Prediction head parameters
        hidden_dims: list = [512, 256],
        output_dim: int = 3,
        dropout_p: float = 0.1
    ):
        super().__init__()

        # Module 1: polar transformation (precomputed grid)
        self.polar_transform = PolarTransform(
            output_size=polar_output_size,
            input_size=input_size  # ← pass the input size
        )

        # Module 2: ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)

        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        backbone_output_dim = 2048

        # Freeze the backbone (optional)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Module 3: classification head
        layers = []
        input_dim = backbone_output_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Sigmoid())  # ensure the output is in [0, 1]

        self.prediction_head = nn.Sequential(*layers)

        # Save the configuration
        self.polar_output_size = polar_output_size
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: input dict, must contain:
                - 'rgb' or 'rsi': RGB satellite image [B, 3, H, W]

        Returns:
            predictions: predictions [B, 3] (vegetation, sky, building)
        """
        # Extract the RGB satellite image
        rgb_key = None
        for key in ['rsi', 'rgb']:
            if key in inputs:
                rgb_key = key
                break

        if rgb_key is None:
            raise KeyError("Missing 'rsi' or 'rgb' input")

        satellite_rgb = inputs[rgb_key]

        # Module 1: polar transformation
        polar_image = self.polar_transform(satellite_rgb)

        # Module 2: ResNet50 feature extraction
        features = self.backbone(polar_image)  # [B, 2048, 1, 1]
        features = features.flatten(1)  # [B, 2048]

        # Module 3: prediction
        predictions = self.prediction_head(features)  # [B, 3]

        return predictions

    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'PolarTransformSVIPredictor',
            'polar_output_size': self.polar_output_size,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'architecture': {
                'module1': 'PolarTransform',
                'module2': 'ResNet50',
                'module3': 'MLP Prediction Head'
            }
        }


def create_polar_transform_model(config: Dict) -> nn.Module:
    """
    Factory function: create the polar transformation model

    Args:
        config: configuration dict, containing:
            - polar_output_size: polar image size (height, width)
            - input_size: input image size (height, width), default (128, 128)
            - pretrained: whether to use pretrained weights
            - freeze_backbone: whether to freeze the backbone
            - hidden_dims: list of hidden layer dimensions
            - output_dim: output dimension
            - dropout_p: dropout probability

    Returns:
        model: PolarTransformSVIPredictor instance

    Example:
        config = {
            'polar_output_size': [128, 256],
            'input_size': [128, 128],
            'pretrained': True,
            'freeze_backbone': False,
            'hidden_dims': [512, 256],
            'output_dim': 3,
            'dropout_p': 0.1
        }
        model = create_polar_transform_model(config)
    """
    model_config = config.get('model', {})
    data_config = config.get('data_for_model', config.get('data', {}))

    # Get the input size (prefer the model config, then target_size from the data config)
    input_size = model_config.get('input_size', None)
    if input_size is None and 'target_size' in data_config:
        input_size = tuple(data_config['target_size'])
    if input_size is None:
        input_size = (128, 128)  # default

    model = PolarTransformSVIPredictor(
        polar_output_size=tuple(model_config.get('polar_output_size', [128, 256])),
        input_size=tuple(input_size),  # ← pass the input size
        pretrained=model_config.get('pretrained', False),
        freeze_backbone=model_config.get('freeze_backbone', False),
        hidden_dims=model_config.get('head', {}).get('hidden_dims', [512, 256]),
        output_dim=model_config.get('head', {}).get('output_dim', 3),
        dropout_p=model_config.get('head', {}).get('dropout_p', 0.1)
    )

    return model
