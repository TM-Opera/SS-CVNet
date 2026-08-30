"""
SVI prediction model - concise version
Supports ResNet and ViT backbones with simple concatenation fusion and an MLP prediction head
"""

import warnings
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple

from .modules import ClassificationHead
from .ss_cvnet import create_sscvnet
from .polar_transform_model import PolarTransformSVIPredictor, create_polar_transform_model



class CrossViewResNetAdapter(nn.Module):
    """
    CrossViewResNet adapter

    Converts the standard input format (inputs dict) into the format required by CrossViewResNet
    Uses the pre-generated M_geo mapping matrix
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_config = config.get('model', {})

        # Create the CrossViewResNet model
        self.model = create_cross_view_resnet(
            pano_in_channels=model_config.get('pano_in_channels', 1),
            sat_in_channels=model_config.get('sat_in_channels', 3),
            pano_resolution=tuple(model_config.get('pano_resolution', [128, 256])),
            sat_resolution=tuple(model_config.get('sat_resolution', [128, 128])),
            feature_channels=model_config.get('feature_channels', 64),
            num_classes=model_config.get('num_classes', 3),
            dropout_p=model_config.get('dropout_p', 0.1)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass

        Args:
            inputs: input dict, must contain:
                - 'generated_masks': containing 'semantic_mask' and 'mapping_matrix'
                - 'rgb' or 'rsi': RGB satellite image

        Returns:
            predictions: predictions [B, 3]
        """
        # Extract the RGB satellite image
        rgb_key = None
        for key in ['rsi', 'rgb']:
            if key in inputs:
                rgb_key = key
                break

        if rgb_key is None:
            raise KeyError("Missing 'rsi' or 'rgb' input")

        sat_rgb = inputs[rgb_key]

        # Extract the mask data
        if 'generated_masks' not in inputs:
            raise KeyError("Missing 'generated_masks' input")

        generated_masks = inputs['generated_masks']
        semantic_mask = generated_masks['semantic_mask']  # [B, H_pano, W_pano]
        M_geo = generated_masks['mapping_matrix']  # [B, H_pano, W_pano, 2]

        # Prepare the panoramic structure map (using the semantic mask)
        pano_structure = semantic_mask.unsqueeze(1).float()  # [B, 1, H_pano, W_pano]

        # CrossViewResNet forward pass
        predictions = self.model(
            pano_structure=pano_structure,
            sat_rgb=sat_rgb,
            M_geo=M_geo
        )

        return predictions


class CrossViewResNetOptimizedAdapter(nn.Module):
    """
    CrossViewResNetOptimized adapter

    Converts the standard input format (inputs dict) into the format required by CrossViewResNetOptimized
    Uses the pre-generated M_geo mapping matrix
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_config = config.get('model', {})

        # Create the CrossViewResNetOptimized model
        self.model = create_cross_view_resnet_optimized(
            pano_in_channels=model_config.get('pano_in_channels', 1),
            sat_in_channels=model_config.get('sat_in_channels', 3),
            pano_resolution=tuple(model_config.get('pano_resolution', [128, 256])),
            sat_resolution=tuple(model_config.get('sat_resolution', [128, 128])),
            feature_channels=model_config.get('feature_channels', 64),
            num_classes=model_config.get('num_classes', 3),
            dropout_p=model_config.get('dropout_p', 0.1)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass

        Args:
            inputs: input dict, must contain:
                - 'generated_masks': containing 'semantic_mask' and 'mapping_matrix'
                - 'rgb' or 'rsi': RGB satellite image

        Returns:
            predictions: predictions [B, 3]
        """
        # Extract the RGB satellite image
        rgb_key = None
        for key in ['rsi', 'rgb']:
            if key in inputs:
                rgb_key = key
                break

        if rgb_key is None:
            raise KeyError("Missing 'rsi' or 'rgb' input")

        sat_rgb = inputs[rgb_key]

        # Extract the mask data
        if 'generated_masks' not in inputs:
            raise KeyError("Missing 'generated_masks' input")

        generated_masks = inputs['generated_masks']
        semantic_mask = generated_masks['semantic_mask']  # [B, H_pano, W_pano]
        M_geo = generated_masks['mapping_matrix']  # [B, H_pano, W_pano, 2]

        # Prepare the panoramic structure map (using the semantic mask)
        pano_structure = semantic_mask.unsqueeze(1).float()  # [B, 1, H_pano, W_pano]

        # CrossViewResNetOptimized forward pass
        predictions = self.model(
            pano_structure=pano_structure,
            sat_rgb=sat_rgb,
            M_geo=M_geo
        )

        return predictions


class EDFCAdapter(nn.Module):
    """
    EDFC model adapter

    Converts the standard input format (inputs dict) into the format required by EDFC
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_config = config.get('model', {})

        # Create the EDFC model
        self.model = create_edfc(
            num_classes=model_config.get('num_classes', 3),
            lambda_weight=model_config.get('dsa_lambda', 0.8),
            dropout_p=model_config.get('dropout_p', 0.2)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass

        Args:
            inputs: input dict, must contain:
                - 'rgb' or 'rsi': RGB satellite image
                - 'building_height': building height map
                - 'tree_height': tree height map

        Returns:
            predictions: predictions [B, 3]
        """
        # Extract the RGB satellite image
        rgb_key = None
        for key in ['rsi', 'rgb']:
            if key in inputs:
                rgb_key = key
                break

        if rgb_key is None:
            raise KeyError("Missing 'rsi' or 'rgb' input")

        rgb = inputs[rgb_key]

        # Extract the height maps
        if 'building_height' not in inputs:
            raise KeyError("Missing 'building_height' input")
        if 'tree_height' not in inputs:
            raise KeyError("Missing 'tree_height' input")

        building_height = inputs['building_height']
        tree_height = inputs['tree_height']

        # EDFC forward pass
        predictions = self.model(
            rgb=rgb,
            building_height=building_height,
            tree_height=tree_height
        )

        return predictions


# Filter torchvision deprecated-parameter warnings
warnings.filterwarnings('ignore', message='The parameter .pretrained. is deprecated.*')
warnings.filterwarnings('ignore', message='Arguments other than a weight enum.*')

class BackboneFactory:
    """Backbone factory"""

    @staticmethod
    def create_backbone(
        backbone_type: str,
        variant: str,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224
    ) -> nn.Module:
        """Create a backbone network

        Args:
            backbone_type: backbone type ('resnet', 'vit', 'swin', or 'convnext')
            variant: specific variant
            pretrained: whether to use pretrained weights
            in_channels: number of input channels
            img_size: input image size (required by ViT/Swin, default 224)

        Returns:
            backbone model
        """
        if backbone_type == 'resnet':
            # ResNet family
            resnet_variants = {
                'resnet18': (models.resnet18, models.ResNet18_Weights),
                'resnet34': (models.resnet34, models.ResNet34_Weights),
                'resnet50': (models.resnet50, models.ResNet50_Weights),
                'resnet101': (models.resnet101, models.ResNet101_Weights),
                'resnet152': (models.resnet152, models.ResNet152_Weights)
            }

            if variant not in resnet_variants:
                raise ValueError(f"Unsupported ResNet variant: {variant}")

            # Use the new weights API (torchvision 0.13+)
            model_fn, weights_class = resnet_variants[variant]
            # Select weights according to the pretrained parameter
            if pretrained:
                weights = weights_class.DEFAULT
            else:
                weights = None
            backbone = model_fn(weights=weights)

            # Modify the first convolution layer to accommodate different input channel counts
            if in_channels != 3:
                original_conv1 = backbone.conv1
                backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                # Reinitialize the weights
                nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

            # Get the output feature dimension
            backbone.output_dim = backbone.fc.in_features

            # Remove the final fully connected layer
            backbone.fc = nn.Identity()

            return backbone

        elif backbone_type == 'vit':
            # Vision Transformer family
            try:
                import timm
            except ImportError:
                raise ImportError("Using ViT requires the timm library: pip install timm")

            # Alias mapping
            vit_alias_map = {
                'vit-t': 'vit_tiny_patch16_224',
                'vit-s': 'vit_small_patch16_224',
                'vit-b': 'vit_base_patch16_224',
                'vit-l': 'vit_large_patch16_224'
            }

            # Convert alias names to full names
            actual_variant = vit_alias_map.get(variant, variant)

            vit_variants = [
                'vit_tiny_patch16_224',
                'vit_small_patch16_224',
                'vit_base_patch16_224',
                'vit_large_patch16_224'
            ]

            if actual_variant not in vit_variants:
                raise ValueError(f"Unsupported ViT variant: {variant}. Supported variants: {', '.join(list(vit_alias_map.keys()))}")

            # Create the ViT model
            try:
                backbone = timm.create_model(
                    actual_variant,
                    pretrained=pretrained,
                    num_classes=0,
                    in_chans=in_channels,
                    img_size=img_size  # use the given image size
                )
            except (RuntimeError, OSError) as e:
                if pretrained and ('network' in str(e).lower() or 'unreachable' in str(e).lower()):
                    print(f"[Warning] Unable to download pretrained weights (network error); falling back to random initialization")
                    backbone = timm.create_model(
                        actual_variant,
                        pretrained=False,
                        num_classes=0,
                        in_chans=in_channels,
                        img_size=img_size
                    )
                else:
                    raise
            backbone.output_dim = backbone.embed_dim

            return backbone

        elif backbone_type == 'swin':
            # Swin Transformer family
            try:
                import timm
            except ImportError:
                raise ImportError("Using Swin Transformer requires the timm library: pip install timm")

            # Alias mapping
            swin_alias_map = {
                'swin-t': 'swin_tiny_patch4_window7_224',
                'swin-s': 'swin_small_patch4_window7_224',
                'swin-b': 'swin_base_patch4_window7_224',
                'swin-l': 'swin_large_patch4_window7_224'
            }

            # Convert alias names to full names
            actual_variant = swin_alias_map.get(variant, variant)

            swin_variants = [
                'swin_tiny_patch4_window7_224',
                'swin_small_patch4_window7_224',
                'swin_base_patch4_window7_224',
                'swin_base_patch4_window12_384',
                'swin_large_patch4_window7_224',
                'swin_large_patch4_window12_384'
            ]

            if actual_variant not in swin_variants:
                raise ValueError(f"Unsupported Swin Transformer variant: {variant}. Supported variants: {', '.join(list(swin_alias_map.keys()))}")

            # Create the Swin Transformer model
            try:
                backbone = timm.create_model(
                    actual_variant,
                    pretrained=pretrained,
                    num_classes=0,
                    in_chans=in_channels,
                    img_size=img_size  # use the given image size
                )
            except (RuntimeError, OSError) as e:
                if pretrained and ('network' in str(e).lower() or 'unreachable' in str(e).lower()):
                    print(f"[Warning] Unable to download pretrained weights (network error); falling back to random initialization")
                    backbone = timm.create_model(
                        actual_variant,
                        pretrained=False,
                        num_classes=0,
                        in_chans=in_channels,
                        img_size=img_size
                    )
                else:
                    raise
            # Output feature dimension of Swin Transformer
            backbone.output_dim = backbone.num_features  # usually equals embed_dim

            return backbone

        elif backbone_type == 'convnext':
            # ConvNeXt family
            try:
                import timm
            except ImportError:
                raise ImportError("Using ConvNeXt requires the timm library: pip install timm")

            # Alias mapping
            convnext_alias_map = {
                'convnext-t': 'convnext_tiny',
                'convnext-s': 'convnext_small',
                'convnext-b': 'convnext_base',
                'convnext-l': 'convnext_large'
            }

            # Convert alias names to full names
            actual_variant = convnext_alias_map.get(variant, variant)

            convnext_variants = [
                'convnext_tiny',
                'convnext_small',
                'convnext_base',
                'convnext_large',
                'convnextv2_tiny',
                'convnextv2_small',
                'convnextv2_base',
                'convnextv2_large'
            ]

            if actual_variant not in convnext_variants:
                raise ValueError(f"Unsupported ConvNeXt variant: {variant}. Supported variants: {', '.join(list(convnext_alias_map.keys()))}")

            # Create the ConvNeXt model
            try:
                backbone = timm.create_model(
                    actual_variant,
                    pretrained=pretrained,
                    num_classes=0,
                    in_chans=in_channels
                )
            except (RuntimeError, OSError) as e:
                if pretrained and ('network' in str(e).lower() or 'unreachable' in str(e).lower()):
                    print(f"[Warning] Unable to download pretrained weights (network error); falling back to random initialization")
                    backbone = timm.create_model(
                        actual_variant,
                        pretrained=False,
                        num_classes=0,
                        in_chans=in_channels
                    )
                else:
                    raise
            # Output feature dimension of ConvNeXt
            backbone.output_dim = backbone.num_features  # usually equals embed_dim

            return backbone

        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")


class SVIPredictor(nn.Module):
    """Street view image prediction model - concise version

    Supports multimodal inputs with simple concatenation fusion
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: configuration dict, containing:
                - modalities: list of modalities to use
                - backbone: backbone configuration
                - head: prediction head configuration
                - pretrained: whether to load pretrained weights
                - freeze_backbone: whether to freeze the backbone
        """
        super(SVIPredictor, self).__init__()

        self.modalities = config.get('modalities', ['rsi'])
        self.config = config

        # Get the pretrained and freeze configurations
        pretrained = config.get('pretrained', False)
        freeze_backbone = config.get('freeze_backbone', False)

        # Get the target image size (for ViT/Swin)
        target_size = config.get('data_for_model', {}).get('target_size', [224, 224])
        if isinstance(target_size, list):
            img_size = target_size[0]  # assume square
        else:
            img_size = 224

        # Create a backbone for each modality
        self.backbones = nn.ModuleDict()
        self.feature_dims = []

        for modality in self.modalities:
            # Set the input channel count according to the modality type
            # Supported modality names: 'rgb'/'rsi' → 3 channels, 'building_height'/'building'/'tree_height'/'tree' → 1 channel
            if modality in ['rsi', 'rgb']:
                in_channels = 3  # RGB image
            else:  # building_height, tree_height
                in_channels = 1  # single-channel height map

            backbone = BackboneFactory.create_backbone(
                backbone_type=config['backbone_type'],
                variant=config['backbone_variant'],
                pretrained=pretrained,
                in_channels=in_channels,
                img_size=img_size
            )

            # Freeze the backbone parameters
            if freeze_backbone:
                for param in backbone.parameters():
                    param.requires_grad = False

            self.backbones[modality] = backbone
            self.feature_dims.append(backbone.output_dim)

        # Compute the total feature dimension (after concatenation)
        total_dim = sum(self.feature_dims)

        # Create the prediction head
        head_config = config.get('head', {})
        self.prediction_head = ClassificationHead(
            input_dim=total_dim,
            hidden_dim=head_config.get('hidden_dim', 256),  # fix: use hidden_dim instead of hidden_dims
            output_dim=head_config.get('output_dim', 3),
            dropout_p=head_config.get('dropout_p', 0.1)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass

        Args:
            inputs: input dict, keys are modality names, values are tensors [B, C, H, W]

        Returns:
            predictions [B, 3] (vegetation, sky, building)
        """
        # Extract features of each modality
        features_list = []
        for modality in self.modalities:
            if modality not in inputs:
                # Missing modality, use zero features
                backbone = self.backbones[modality]
                dummy_input = torch.zeros(
                    inputs[list(inputs.keys())[0]].size(0),
                    3 if modality == 'rsi' else 1,
                    224, 224,
                    device=inputs[list(inputs.keys())[0]].device
                )
                feature = backbone(dummy_input)
            else:
                feature = self.backbones[modality](inputs[modality])

            # Global average pooling
            if feature.dim() == 4:  # [B, C, H, W]
                feature = nn.functional.adaptive_avg_pool2d(feature, (1, 1))
                feature = feature.view(feature.size(0), -1)  # [B, C]

            features_list.append(feature)

        # Concatenate all features
        concatenated = torch.cat(features_list, dim=1)  # [B, total_dim]

        # Predict
        predictions = self.prediction_head(concatenated)

        return predictions

    def get_trainable_params_info(self) -> Dict:
        """Get trainable parameter information

        Returns:
            dict containing the total, trainable, and frozen parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Count parameters of each module
        backbone_params = 0
        backbone_trainable = 0
        for backbone in self.backbones.values():
            for param in backbone.parameters():
                backbone_params += param.numel()
                if param.requires_grad:
                    backbone_trainable += param.numel()

        head_params = sum(p.numel() for p in self.prediction_head.parameters())
        head_trainable = sum(p.numel() for p in self.prediction_head.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'backbone_params': backbone_params,
            'backbone_trainable': backbone_trainable,
            'head_params': head_params,
            'head_trainable': head_trainable
        }


def create_model(config: Dict) -> nn.Module:
    """Create the SVI prediction model (supports multiple architectures)

    Args:
        config: configuration dict, containing:
            - architecture_type: architecture type ('simple' | 'sscvnet' | 'crossview_resnet' | 'polar_transform')
            - physics_aware.enabled: whether to enable physics-aware weighting (optional)
            - exp.version: physics-aware model version (v1.0 or v1.1)
            - other configurations...

    Returns:
        model instance

    Supported architecture types:
    1. simple: simple concatenation fusion (baseline model, backward compatible)
    2. sscvnet: SS-CVNet VIFE depth-aware fusion
       - Supports ablation options (semantic embedding, height map, visibility mask, cross-view alignment)
       - Uses pre-computed mask data (generated_masks)
    3. crossview_resnet: CrossViewResNet cross-view ResNet architecture
       - Based on paper arXiv:2408.14765 Section 3.3
       - Uses cross-attention + weight matrix + ResNet50
    4. edfc: EDFC dual-stream fusion network (RGB + DSM)
       - Based on paper https://www.mdpi.com/2072-4292/14/5/1294
       - Uses the DSA depth-aware self-attention module
       - Dual-branch ResNet50 architecture
    5. polar_transform: polar transformation cross-view prediction model
       - Converts the satellite image to a ground view via polar transformation
       - Uses ResNet50 to predict building, sky, and vegetation fractions
       - Based on the polar transformation method of the SAFA paper
    6. physics_aware: physics-aware weighted pooling (when physics_aware.enabled=true)
       - v1.0: fixed physical parameters
       - v1.1: learnable physical parameters
    """
    # Get the architecture type (compatible with config['architecture_type'] and config['model']['architecture_type'])
    architecture_type = config.get('architecture_type') or config.get('model', {}).get('architecture_type', 'simple')

    # Check whether physics-aware weighting is enabled
    if config.get('physics_aware', {}).get('enabled', False):
        # Use the physics-aware model
        import sys
        from pathlib import Path

        # Select the model folder according to the version
        version = config.get('exp', {}).get('version', 'v1.0')
        if version == 'v1.1':
            physics_aware_path = Path(__file__).parent / 'physics_aware__v1_1'
            module_name = 'models.physics_aware__v1_1.architecture'
        else:
            physics_aware_path = Path(__file__).parent / 'physics_aware__v1_0'
            module_name = 'models.physics_aware__v1_0.architecture'

        # Execute the Python file directly and get the class
        with open(physics_aware_path / 'architecture.py', 'r') as f:
            code = f.read()
        # Execute the code in a temporary namespace
        temp_namespace = {'__name__': module_name}
        exec(code, temp_namespace)
        return temp_namespace['PhysicsAwareSVIPredictor'](config)

    # Config conversion: sscvnet needs the full config (including config['model'] and config['device'])
    # other architectures need the flattened model_config (i.e. config['model'] plus modalities and exp)
    if architecture_type == 'sscvnet':
        # sscvnet needs the full config
        model_config = config
    else:
        # Other architectures use the flattened model_config
        model_config = config.get('model', {}).copy()
        # Get modalities and exp from the top level (if present)
        if 'modalities' in config:
            model_config['modalities'] = config['modalities']
        if 'exp' in config:
            model_config['exp'] = config['exp']

    if architecture_type == 'simple':
        # Simple concatenation fusion (baseline model, backward compatible)
        return SVIPredictor(model_config)

    elif architecture_type == 'sscvnet':
        # SS-CVNet VIFE depth-aware fusion
        return create_sscvnet(model_config)

    elif architecture_type == 'crossview_resnet':
        # CrossViewResNet cross-view ResNet architecture
        # needs the full config (including device)
        # uses an adapter to handle the input format conversion
        return CrossViewResNetAdapter(config)

    elif architecture_type == 'crossview_resnet_optimized':
        # CrossViewResNetOptimized cross-view ResNet architecture
        # needs the full config (including device)
        # uses an adapter to handle the input format conversion
        return CrossViewResNetOptimizedAdapter(config)

    elif architecture_type == 'edfc':
        # EDFC dual-stream fusion network (RGB + DSM)
        # needs the full config
        return EDFCAdapter(config)

    elif architecture_type == 'polar_transform':
        # Polar transformation cross-view prediction model
        # needs the full config
        return create_polar_transform_model(config)

    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
