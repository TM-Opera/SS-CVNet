"""
SS-CVNet: depth-aware fusion network for street view indicator prediction

1. Module 1: semantic embedding encoder
   - Input: pre-computed pano_semantic [B, H_pano, W_pano]
   - Output: semantic_features [B, 64, H_pano, W_pano]
   - Converts discrete semantic labels (4 classes) into continuous feature representations

2. Module 2: VIFE depth-aware self-attention multimodal texture encoding
   - Fusion: RGB + V_mask_sat + building_height + tree_height
   - Uses: models/VIFE.py
   - Output: texture_features [B, 64, H_sat, W_sat]

3. Module 3: feature-level cross-view alignment
   - M_geo transformation + gated fusion
   - Uses: CrossViewAlignmentModule from models/modules.py
   - Output: unified_features [B, 64, H_pano, W_pano]

4. Module 4: deep feature extraction
   - Uses: BackboneEncoder from models/modules.py
   - Output: deep_features [B, out_features]

5. Module 5: classification prediction
   - MLP: [B, out_features] → [B, 3] (BVI, GVI, SVF)

Data flow:
Input (RGB + building height + tree height + pre-computed masks)
  ↓
Module 1: semantic embedding → semantic_features [B, 64, H_pano, W_pano]
  ↓
Module 2: VIFE fusion → texture_features [B, 64, H_sat, W_sat]
  ↓
Module 3: M_geo alignment + gated fusion → unified_features [B, 64, H_pano, W_pano]
  ↓
Module 4: backbone encoding → deep_features [B, out_features]
  ↓
Module 5: classification → (BVI, GVI, SVF)

Important changes:
- No longer includes real-time geometric computation modules
- Uses pre-computed mask data (generated_masks)
- Must be provided in inputs['generated_masks']:
    - 'semantic_mask': [B, H_pano, W_pano] semantic labels (0-3)
    - 'mapping_matrix': [B, H_pano, W_pano, 2] mapping matrix
    - 'visibility_mask': [B, H_sat, W_sat] visibility mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import modules
from .VIFE import VIFE
from .modules import CrossViewAlignmentModule, BackboneEncoder  # imported from modules.py
from .projection_transformer import ProjectionTransformer


# =============================================================================
# Module 1: semantic embedding encoder (new)
# =============================================================================

class SemanticEmbeddingEncoder(nn.Module):
    """
    Semantic embedding encoder

    Converts a discrete semantic label map (4 classes) into a continuous feature representation

    Input:
    - pano_semantic: [B, 1, H_pano, W_pano] or [B, H_pano, W_pano]
      value range: {0, 1, 2, 3} corresponding to {sky, ground, building, tree}

    Output:
    - semantic_features: [B, 64, H_pano, W_pano]
      semantic feature representation
    """

    def __init__(
        self,
        num_classes: int = 4,
        out_channels: int = 64
    ):
        """
        Args:
            num_classes: number of semantic classes (default 4)
            out_channels: number of output feature channels (default 64)
        """
        super().__init__()

        self.num_classes = num_classes
        self.out_channels = out_channels

        # Embedding layer: discrete labels → continuous vectors
        self.embedding = nn.Embedding(num_classes, out_channels)

        # 3×3 convolution for spatial context extraction
        self.spatial_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True
        )

        # 1×1 convolution for feature adjustment
        self.adjust_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=True
        )

        # Activation function
        self.activation = nn.GELU()

    def forward(self, pano_semantic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pano_semantic: [B, 1, H, W] or [B, H, W]

        Returns:
            semantic_features: [B, 64, H, W]
        """
        # Ensure the input shape is correct
        if pano_semantic.dim() == 3:
            pano_semantic = pano_semantic.unsqueeze(1)  # [B, 1, H, W]

        B, _, H, W = pano_semantic.shape

        # Ensure an integer type (required by Embedding)
        if pano_semantic.dtype != torch.long:
            pano_semantic = pano_semantic.long()

        # Remove the channel dimension and flatten: [B, H, W]
        pano_semantic = pano_semantic.squeeze(1)

        # Embedding: [B, H, W] → [B, H, W, 64]
        semantic_embed = self.embedding(pano_semantic)

        # Transpose: [B, H, W, 64] → [B, 64, H, W]
        semantic_embed = semantic_embed.permute(0, 3, 1, 2)

        # 3×3 convolution for spatial context extraction
        semantic_features = self.spatial_conv(semantic_embed)
        semantic_features = self.activation(semantic_features)

        # 1×1 convolution adjustment
        semantic_features = self.adjust_conv(semantic_features)

        return semantic_features


# =============================================================================
# Module 2: VIFE depth-aware self-attention texture encoder (uses VIFE)
# =============================================================================

class VIFETextureEncoder(nn.Module):
    """
    VIFE depth-aware self-attention texture encoder

    Fuses RGB, visibility mask, building height, and tree height information

    Input:
    - rgb: [B, 3, H_sat, W_sat] RGB image
    - v_mask_sat: [B, 1, H_sat, W_sat] visibility mask (optional)
    - building_height: [B, 1, H_sat, W_sat] building height (optional)
    - tree_height: [B, 1, H_sat, W_sat] tree height (optional)

    Output:
    - texture_features: [B, 64, H_sat, W_sat]

    Ablation options:
    - use_height_map: whether to use height maps (building_height + tree_height)
    - use_visibility_mask: whether to use the visibility mask
    - use_center_circle_mask: use a center circular mask instead of the visibility mask
    - use_se_attention: use SEBlock spatial attention instead of the visibility mask
    """

    def __init__(
        self,
        in_channels_rgb: int = 3,
        in_channels_dsm: int = 1,
        out_channels: int = 64,
        lambda_weight: float = 0.8,
        use_window_attention: bool = True,
        window_size: int = 4,
        use_height_map: bool = True,
        use_visibility_mask: bool = True,
        use_center_circle_mask: bool = False,
        circle_mask_radius: int = 50,
        use_se_attention: bool = False,
        se_reduction: int = 16
    ):
        """
        Args:
            in_channels_rgb: number of RGB input channels
            in_channels_dsm: number of DSM input channels
            out_channels: number of output feature channels
            lambda_weight: depth similarity weight coefficient
            use_window_attention: whether to use window attention
            window_size: window size
            use_height_map: whether to use height maps (ablation option)
            use_visibility_mask: whether to use the visibility mask (ablation option)
            use_center_circle_mask: use a center circular mask instead (ablation option)
            circle_mask_radius: circular mask radius (pixels, default 50)
            use_se_attention: use SEBlock attention (ablation option)
            se_reduction: SEBlock reduction ratio (default 16)
        """
        super().__init__()

        self.out_channels = out_channels
        self.use_height_map = use_height_map
        self.use_visibility_mask = use_visibility_mask
        self.use_center_circle_mask = use_center_circle_mask
        self.circle_mask_radius = circle_mask_radius
        self.use_se_attention = use_se_attention

        if not use_height_map:
            print("VIFETextureEncoder initialized WITHOUT height map (building_height and tree_height will be ignored)")
            self.lambda_weight = 0.0  # when not using height maps, the depth similarity weight is set to 0
        else:
            print("VIFETextureEncoder initialized WITH height map (building_height and tree_height will be used)")
            self.lambda_weight = lambda_weight

        if use_center_circle_mask:
            print(f"VIFETextureEncoder will use CENTER CIRCLE mask (radius={circle_mask_radius}px) instead of visibility mask")
            self.use_visibility_mask = False  # the circular mask and the visibility mask are mutually exclusive

        if use_se_attention:
            print(f"VIFETextureEncoder will use SEBLOCK attention instead of visibility mask")

        # VIFE module (already includes RGB feature extraction)
        self.vife = VIFE(
            in_channels_rgb=in_channels_rgb,
            in_channels_dsm=in_channels_dsm,
            out_channels=out_channels,
            lambda_weight=self.lambda_weight,
            use_window_attention=use_window_attention,
            window_size=window_size,
            use_se_attention=use_se_attention,
            se_reduction=se_reduction
        )

    def forward(
        self,
        rgb: torch.Tensor,
        v_mask_sat: torch.Tensor,
        building_height: torch.Tensor,
        tree_height: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W]
            v_mask_sat: [B, 1, H, W] or [B, H, W]
            building_height: [B, 1, H, W]
            tree_height: [B, 1, H, W]

        Returns:
            texture_features: [B, 64, H, W]
        """
        B, _, H, W = rgb.shape
        device = rgb.device

        # Fuse the DSM: take the maximum
        dsm = torch.max(building_height, tree_height)  # [B, 1, H, W]

        # Handle the mask (ablation options)
        if self.use_center_circle_mask:
            # Generate the center circular mask
            v_mask_sat = self._create_center_circle_mask(B, H, W, device)
        elif self.use_visibility_mask:
            # Use the pre-computed visibility mask
            if v_mask_sat.dim() == 3:
                v_mask_sat = v_mask_sat.unsqueeze(1)
        elif self.use_se_attention:
            # Use SEBlock, no mask needed (an all-ones mask is passed; actually replaced by SEBlock)
            v_mask_sat = torch.ones(B, 1, H, W, device=device)
        else:
            # No mask used; create an all-ones mask (meaning fully visible)
            v_mask_sat = torch.ones(B, 1, H, W, device=device)

        # VIFE processing
        texture_features = self.vife(rgb, dsm, v_mask_sat)

        return texture_features

    def _create_center_circle_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create the center circular mask

        Args:
            batch_size: batch size
            height: image height
            width: image width
            device: device

        Returns:
            mask: [B, 1, H, W] center circular mask
        """
        # Create the coordinate grid
        y = torch.arange(height, device=device)
        x = torch.arange(width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Compute the distance to the center
        center_y = height // 2
        center_x = width // 2
        distance = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

        # Create the circular mask (1 inside the circle, 0 outside)
        mask = (distance <= self.circle_mask_radius).float()

        # Add batch and channel dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask = mask.expand(batch_size, 1, height, width)  # [B, 1, H, W]

        return mask


# =============================================================================
# Module 5: classification prediction head
# =============================================================================

class ClassificationHead(nn.Module):
    """
    Classification prediction head (Module 5)

    Input: F_high [B, 512] (ResNet18) or [B, 2048] (ResNet50)
    Output: predictions [B, 3] (BVI, GVI, SVF)

    Network structure:
    Dropout(0.1)
        ↓
    Linear(input_dim→256) + ReLU
        ↓
    Dropout(0.1)
        ↓
    Linear(256→3) + Sigmoid
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 3,
        dropout_p: float = 0.1
    ):
        """
        Args:
            input_dim: input feature dimension (default 512 corresponds to ResNet50)
            hidden_dim: hidden layer dimension (default 256)
            output_dim: output dimension (default 3: BVI, GVI, SVF)
            dropout_p: dropout probability (default 0.1)
        """
        super(ClassificationHead, self).__init__()

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        logger.debug(
            f"ClassificationHead initialized: "
            f"{input_dim}→{hidden_dim}→{output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: input features [B, input_dim]

        Returns:
            predictions: predictions [B, 3] (BVI, GVI, SVF), each value in [0, 1]
        """
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        #x = self.sigmoid(x)

        return x


# =============================================================================
# SS-CVNet full network
# =============================================================================

class SSCVNet(nn.Module):
    """
    SS-CVNet: VIFE depth-aware fusion

    Core innovations:
    1. Module 1: semantic embedding encoding (receives pre-computed pano_semantic)
    2. Module 2: VIFE depth-aware self-attention texture encoding
    3. Module 3: feature-level cross-view alignment
    4. Module 4: ResNet/ConvNeXt deep feature extraction (BackboneEncoder)
    5. Module 5: classification prediction

    Input:
    - 'rgb' or 'rsi': [B, 3, H_sat, W_sat]
    - 'building_height': [B, 1, H_sat, W_sat]
    - 'tree_height': [B, 1, H_sat, W_sat]
    - 'generated_masks': pre-computed mask data (required), containing:
        - 'semantic_mask': [B, H_pano, W_pano] semantic labels (0-3)
        - 'mapping_matrix': [B, H_pano, W_pano, 2] mapping matrix
        - 'visibility_mask': [B, H_sat, W_sat] visibility mask

    Output:
    - predictions: [B, 3] (BVI, GVI, SVF)

    Notes:
    - No longer includes real-time geometric computation modules
    - Uses pre-computed mask data (generated by generate_masks.py)
    - Ensure inputs['generated_masks'] contains all required masks
    """

    def __init__(
        self,
        # Module 1 parameters (semantic embedding encoder)
        semantic_num_classes: int = 4,
        semantic_out_channels: int = 64,
        use_semantic_embedding: bool = True,  # ablation option: whether to use semantic embedding
        # Module 2 parameters
        vife_lambda_weight: float = 0.8,
        vife_use_window_attention: bool = True,
        vife_window_size: int = 4,
        use_height_map: bool = True,  # ablation option: whether to use height maps
        use_visibility_mask: bool = True,  # ablation option: whether to use the visibility mask
        use_center_circle_mask: bool = False,  # ablation option: use a center circular mask
        circle_mask_radius: int = 50,  # circular mask radius (pixels)
        use_se_attention: bool = False,  # ablation option: use SEBlock attention
        se_reduction: int = 16,  # SEBlock reduction ratio
        # Module 3 parameters
        use_cross_view_alignment: bool = True,  # ablation option: whether to use cross-view alignment
        use_polar_transform: bool = False,  # ablation option: use polar transformation
        polar_output_size: tuple = (128, 256),  # polar output size
        use_learnable_matrix: bool = False,  # ablation option: use a learnable matrix
        sat_resolution: tuple = (128, 128),  # satellite image resolution
        # Module 4 parameters
        backbone_type: str = 'resnet50',
        # Fusion strategy (Section 4.3.3), applied to the Module 3 fusion when
        # use_cross_view_alignment=True: 'gate' (default) | 'rsc' | 'cma' | 'ssi'
        fusion_strategy: str = 'gate',
        # Module 5 parameters
        head_hidden_dim: int = 512,
        dropout_p: float = 0.1,
        # General parameters
        device: str = 'cuda'
    ):
        super().__init__()

        self.fusion_strategy = fusion_strategy
        self.device = device
        self.use_semantic_embedding = use_semantic_embedding
        self.use_cross_view_alignment = use_cross_view_alignment
        self.use_polar_transform = use_polar_transform
        self.use_learnable_matrix = use_learnable_matrix
        self.semantic_out_channels = semantic_out_channels
        self.polar_output_size = polar_output_size

        # Module 1: semantic embedding encoder (receives pre-computed pano_semantic)
        self.semantic_encoder = SemanticEmbeddingEncoder(
            num_classes=semantic_num_classes,
            out_channels=semantic_out_channels
        )

        # Module 2: VIFE texture encoder (with ablation options)
        self.module2 = VIFETextureEncoder(
            in_channels_rgb=3,
            in_channels_dsm=1,
            out_channels=64,
            lambda_weight=vife_lambda_weight,
            use_window_attention=vife_use_window_attention,
            window_size=vife_window_size,
            use_height_map=use_height_map,
            use_visibility_mask=use_visibility_mask,
            use_center_circle_mask=use_center_circle_mask,
            circle_mask_radius=circle_mask_radius,
            use_se_attention=use_se_attention,
            se_reduction=se_reduction
        )

        # Create different module structures depending on whether cross-view alignment is used
        if self.use_cross_view_alignment:
            # Module 3: feature alignment (implementation in modules.py);
            # the fusion_strategy ablation (Section 4.3.3) applies to its internal fusion
            self.module3 = CrossViewAlignmentModule(
                feature_channels=64,
                use_polar_transform=use_polar_transform,
                polar_output_size=polar_output_size,
                use_learnable_matrix=use_learnable_matrix,
                sat_resolution=sat_resolution,
                fusion_strategy=fusion_strategy
            )

            # Module 4: single BackboneEncoder
            self.module4_semantic = None  # not needed
            self.module4_texture = None   # not needed

            self.module4 = BackboneEncoder(
                backbone_type=backbone_type,
                in_channels=64
            )

            # Input dimension of Module 5
            module5_input_dim = self.module4.out_features

        else:
            # Without cross-view alignment: create module4 for semantic and texture separately,
            # then fuse via gating

            # Module 4: two independent BackboneEncoders
            self.module4_semantic = BackboneEncoder(
                backbone_type=backbone_type,
                in_channels=64
            )
            self.module4_texture = BackboneEncoder(
                backbone_type=backbone_type,
                in_channels=64
            )

            self.module4 = None  # no unified module4

            # Gated fusion module
            gate_input_dim = self.module4_semantic.out_features + self.module4_texture.out_features
            self.gate_fusion = nn.Sequential(
                nn.Linear(gate_input_dim, gate_input_dim // 2),
                nn.ReLU(),
                nn.Linear(gate_input_dim // 2, 2),  # weights of the two branches
                nn.Softmax(dim=-1)
            )

            # Input dimension of Module 5
            module5_input_dim = self.module4_semantic.out_features

        # Module 5: classification prediction
        self.module5 = ClassificationHead(
            input_dim=module5_input_dim,
            hidden_dim=head_hidden_dim,
            output_dim=3,
            dropout_p=dropout_p
        )

        # Record the configuration
        self.config = {
            'use_semantic_embedding': use_semantic_embedding,
            'use_height_map': use_height_map,
            'use_visibility_mask': use_visibility_mask,
            'use_center_circle_mask': use_center_circle_mask,
            'use_se_attention': use_se_attention,
            'use_cross_view_alignment': use_cross_view_alignment,
            'use_polar_transform': use_polar_transform,
            'use_learnable_matrix': use_learnable_matrix,
            'fusion_strategy': fusion_strategy,
            'backbone_type': backbone_type,
        }

        # Print the ablation configuration in detail
        logger.info("="*80)
        logger.info("SS-CVNet model initialization completed")
        logger.info("="*80)
        logger.info("Architecture configuration:")
        logger.info(f"  Backbone: {backbone_type}")
        logger.info(f"  Module 4 output dimension: {module5_input_dim}")
        logger.info("")

        logger.info("Ablation option configuration:")
        logger.info(f"  Module 1 - semantic embedding:   {'✓ enabled' if use_semantic_embedding else '✗ disabled'}")
        logger.info(f"  Module 2 - height map fusion:    {'✓ enabled' if use_height_map else '✗ disabled'}")
        logger.info(f"  Module 2 - visibility mask:      {'✓ enabled' if use_visibility_mask else '✗ disabled'}")
        if use_center_circle_mask:
            logger.info(f"  Module 2 - center circular mask: ✓ enabled (radius={circle_mask_radius}px)")
        if use_se_attention:
            logger.info(f"  Module 2 - SEBlock attention:    ✓ enabled (reduction={se_reduction})")
        logger.info(f"  Module 3 - cross-view alignment: {'✓ enabled' if use_cross_view_alignment else '✗ disabled (dual-branch gated fusion)'}")
        if use_cross_view_alignment:
            logger.info(f"  Module 3 - fusion strategy:      {fusion_strategy}")
            if use_polar_transform:
                logger.info(f"  Module 3 - polar transformation: ✓ enabled (output size={polar_output_size})")
            if use_learnable_matrix:
                logger.info(f"  Module 3 - learnable matrix:     ✓ enabled (resolution={sat_resolution})")
        logger.info("")

        logger.info("Module 2 (VIFE) configuration:")
        logger.info(f"  λ weight (depth similarity):     {vife_lambda_weight}")
        logger.info(f"  Window attention:                {'enabled' if vife_use_window_attention else 'disabled'}")
        logger.info(f"  Window size:                     {vife_window_size}×{vife_window_size}")
        logger.info("")

        logger.info("Module 5 (classification head) configuration:")
        logger.info(f"  Hidden layer dimension:          {head_hidden_dim}")
        logger.info(f"  Dropout probability:             {dropout_p}")
        logger.info("")

        # Show the total parameter count of the model
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model parameters:")
        logger.info(f"  Total parameters:                {total_params:,}")
        logger.info(f"  Trainable parameters:            {trainable_params:,}")
        logger.info("="*80)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: input dict, must contain:
                - 'rgb' or 'rsi': [B, 3, H_sat, W_sat]
                - 'building_height': [B, 1, H_sat, W_sat]
                - 'tree_height': [B, 1, H_sat, W_sat]
                - 'generated_masks': pre-computed mask data (required), containing:
                    - 'semantic_mask': [B, H_pano, W_pano] semantic labels (0-3)
                    - 'mapping_matrix': [B, H_pano, W_pano, 2] mapping matrix
                    - 'visibility_mask': [B, H_sat, W_sat] visibility mask

        Returns:
            predictions: [B, 3] (BVI, GVI, SVF)
        """
        # Extract the inputs
        rgb_key = None
        for key in ['rsi', 'rgb']:
            if key in inputs:
                rgb_key = key
                break

        if rgb_key is None:
            raise KeyError("Missing 'rsi' or 'rgb' input")

        rgb = inputs[rgb_key]
        building_height = inputs['building_height']
        tree_height = inputs['tree_height']

        # Extract data from the pre-generated masks
        if 'generated_masks' not in inputs:
            raise KeyError(
                "Missing 'generated_masks'. Please use pre-generated masks.\n"
                "Run: python generate_masks.py --cities <city>"
            )

        generated_masks = inputs['generated_masks']
        pano_semantic = generated_masks.get('semantic_mask', None)
        M_geo = generated_masks.get('mapping_matrix', None)
        V_mask_sat = generated_masks.get('visibility_mask', None)

        if pano_semantic is None:
            raise KeyError("Missing 'semantic_mask' in generated_masks")
        if M_geo is None:
            raise KeyError("Missing 'mapping_matrix' in generated_masks")
        if V_mask_sat is None:
            raise KeyError("Missing 'visibility_mask' in generated_masks")

        device = rgb.device

        # Module 1: semantic embedding encoding (ablation option)
        if self.use_semantic_embedding:
            semantic_features = self.semantic_encoder(pano_semantic)  # [B, 64, H_pano, W_pano]
        else:
            # Without semantic embedding, create all-zero semantic_features
            B = pano_semantic.shape[0]
            H_pano = pano_semantic.shape[1]
            W_pano = pano_semantic.shape[2]
            semantic_features = torch.zeros(B, self.semantic_out_channels, H_pano, W_pano, device=device)

        # Module 2: VIFE texture encoding
        texture_features = self.module2(rgb, V_mask_sat, building_height, tree_height)  # [B, 64, H_sat, W_sat]

        # Use different processing flows depending on whether cross-view alignment is used
        if self.use_cross_view_alignment:
            # Module 3: feature alignment (implementation in modules.py)
            # output: [B, 64, H_pano, W_pano]
            unified_features = self.module3(semantic_features, texture_features, M_geo)

            # View transformation (multi_view corresponds to the four-direction perspective projection)
            #if self.perspective == 'fisheye':
                #print("Converting to fisheye shape.")
                #unified_features = self.transformer(unified_features, mode='fisheye')
            #if self.perspective == 'multi_view':
                #print("Converting to perspective shape (multi_view).")

            # Module 4: backbone deep feature extraction
            deep_features = self.module4(unified_features)  # [B, out_features]

        else:
            # Without cross-view alignment: process semantic and texture separately, then fuse via gating
            # extract deep features separately
            deep_semantic = self.module4_semantic(semantic_features)  # [B, out_features]
            deep_texture = self.module4_texture(texture_features)    # [B, out_features]

            # Concatenate the features of the two branches
            combined_features = torch.cat([deep_semantic, deep_texture], dim=1)  # [B, 2*out_features]

            # Compute the gating weights
            gate_weights = self.gate_fusion(combined_features)  # [B, 2]

            # Weighted fusion
            gate_semantic = gate_weights[:, 0:1]  # [B, 1]
            gate_texture = gate_weights[:, 1:2]   # [B, 1]

            deep_features = gate_semantic * deep_semantic + gate_texture * deep_texture  # [B, out_features]


        # Module 5: classification prediction
        predictions = self.module5(deep_features)  # [B, 3]

        return predictions

    def get_config(self) -> dict:
        """
        Get the model configuration information

        Returns:
            config: model configuration dict
        """
        return self.config.copy()


# =============================================================================
# Factory function
# =============================================================================

def create_sscvnet(config: dict) -> SSCVNet:
    """
    Create the SS-CVNet model from the configuration

    Args:
        config: configuration dict, containing:
            - model: model configuration
            - device: device

    Returns:
        model: SSCVNet instance

    Notes:
        The model no longer needs the geometric parameters of Module 1 because pre-computed
        mask data is used. Ensure inputs['generated_masks'] contains:
            - 'semantic_mask': [B, H_pano, W_pano]
            - 'mapping_matrix': [B, H_pano, W_pano, 2]
            - 'visibility_mask': [B, H_sat, W_sat]

    Ablation option configuration example:
        config = {
            'model': {
                'ablation': {
                    'use_semantic_embedding': True,
                    'use_height_map': True,
                    'use_visibility_mask': True,
                    'use_center_circle_mask': False,
                    'circle_mask_radius': 50,
                    'use_se_attention': False,
                    'use_cross_view_alignment': True,
                    'use_polar_transform': False,
                    'polar_output_size': [128, 256],
                    'use_learnable_matrix': False,
                    'sat_resolution': [128, 128]
                },
                'module2': {
                    'lambda_weight': 0.8,
                    'use_window_attention': True,
                    'window_size': 4,
                    'se_reduction': 16
                },
                'module4': {
                    'backbone_type': 'resnet50',
                    'num_groups': 32
                },
                'module5': {
                    'hidden_dim': 512,
                    'dropout_p': 0.1
                }
            },
            'device': 'cuda'
        }
    """
    model_config = config.get('model', {})
    ablation_config = model_config.get('ablation', {})
    device = config.get('device', 'cuda')

    # Print the ablation configuration summary
    if ablation_config:
        print("")
        print("="*80)
        print("Loading SS-CVNet ablation options from the config file")
        print("="*80)

        # Show enabled features
        enabled_features = []
        disabled_features = []

        if ablation_config.get('use_semantic_embedding', True):
            enabled_features.append("Semantic embedding encoding")
        else:
            disabled_features.append("Semantic embedding encoding")

        if ablation_config.get('use_height_map', True):
            enabled_features.append("Height map fusion")
        else:
            disabled_features.append("Height map fusion")

        if ablation_config.get('use_visibility_mask', True):
            enabled_features.append("Visibility mask")
        else:
            disabled_features.append("Visibility mask")

        if ablation_config.get('use_center_circle_mask', False):
            radius = ablation_config.get('circle_mask_radius', 50)
            enabled_features.append(f"Center circular mask (radius={radius}px)")

        if ablation_config.get('use_se_attention', False):
            reduction = ablation_config.get('se_reduction', 16)
            enabled_features.append(f"SEBlock attention (reduction={reduction})")

        if ablation_config.get('use_cross_view_alignment', True):
            enabled_features.append("Cross-view alignment")
            enabled_features.append(f"Fusion strategy ({ablation_config.get('fusion_strategy', 'gate')})")
            if ablation_config.get('use_polar_transform', False):
                polar_size = ablation_config.get('polar_output_size', [128, 256])
                enabled_features.append(f"Polar transformation ({polar_size})")
            if ablation_config.get('use_learnable_matrix', False):
                sat_res = ablation_config.get('sat_resolution', [128, 128])
                enabled_features.append(f"Learnable matrix ({sat_res})")
        else:
            disabled_features.append("Cross-view alignment (dual-branch gated fusion)")

        print("Enabled features:")
        for feature in enabled_features:
            print(f"  ✓ {feature}")

        if disabled_features:
            print("")
            print("Disabled features:")
            for feature in disabled_features:
                print(f"  ✗ {feature}")

        print("="*80)
        print("")

    # Create the model
    model = SSCVNet(
        # Module 1 parameters (semantic embedding encoder)
        semantic_num_classes=4,
        semantic_out_channels=64,
        use_semantic_embedding=ablation_config.get('use_semantic_embedding', True),
        # Module 2 parameters
        vife_lambda_weight=model_config.get('module2', {}).get('lambda_weight', 0.8),
        vife_use_window_attention=model_config.get('module2', {}).get('use_window_attention', True),
        vife_window_size=model_config.get('module2', {}).get('window_size', 4),
        use_height_map=ablation_config.get('use_height_map', True),
        use_visibility_mask=ablation_config.get('use_visibility_mask', True),
        use_center_circle_mask=ablation_config.get('use_center_circle_mask', False),
        circle_mask_radius=ablation_config.get('circle_mask_radius', 50),
        use_se_attention=ablation_config.get('use_se_attention', False),
        se_reduction=model_config.get('module2', {}).get('se_reduction', 16),
        # Module 3 parameters
        use_cross_view_alignment=ablation_config.get('use_cross_view_alignment', True),
        use_polar_transform=ablation_config.get('use_polar_transform', False),
        polar_output_size=tuple(ablation_config.get('polar_output_size', [128, 256])),
        use_learnable_matrix=ablation_config.get('use_learnable_matrix', False),
        sat_resolution=tuple(ablation_config.get('sat_resolution', [128, 128])),
        # Module 4 parameters
        backbone_type=model_config.get('module4', {}).get('backbone_type', 'resnet50'),
        # Fusion strategy (used when use_cross_view_alignment=False)
        fusion_strategy=ablation_config.get('fusion_strategy', 'gate'),
        # Module 5 parameters
        head_hidden_dim=model_config.get('module5', {}).get('hidden_dim', 512),
        dropout_p=model_config.get('module5', {}).get('dropout_p', 0.1),
        device=device
    )

    return model
