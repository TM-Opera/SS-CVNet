import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, Optional
import logging

# Filter torchvision deprecated-parameter warnings
warnings.filterwarnings('ignore', message='The parameter .pretrained. is deprecated.*')
warnings.filterwarnings('ignore', message='Arguments other than a weight enum.*')

logger = logging.getLogger(__name__)


# =============================================================================
# RMSNorm: root mean square normalization layer
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the root mean square
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # Normalize and apply the learnable weight
        return x / rms * self.weight


# =============================================================================
# Module 2: dual-branch feature encoder (DBFE)
# =============================================================================

class GELU(nn.Module):
    """GELU activation function

    Gaussian Error Linear Unit
    Has smoother gradients than ReLU and performs better in modern networks
    """
    def forward(self, x):
        return F.gelu(x)


class SemanticEmbeddingEncoder(nn.Module):
    """
    Semantic embedding encoder (Branch A)

    Function: converts discrete semantic labels into continuous vector representations

    Input:
    - pano_semantic [B, H_pano, W_pano] or [B, 1, H_pano, W_pano]
      discrete class labels: 0, 1, 2, 3
      0: sky region
      1: building region
      2: tree region
      3: ground region

    Output: semantic_features [B, 64, H_pano, W_pano]

    Characteristics:
    - Keeps the spatial resolution unchanged (H_pano × W_pano)
    - Converts discrete classes into a continuous 64-dimensional vector space
    - Learnable semantic representation (optimized during network training)
    """

    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 64,
        use_act: bool = True  # ← activation enabled by default
    ):
        """
        Args:
            num_classes: number of semantic classes (default 4)
            embed_dim: embedding dimension (default 64)
            use_act: whether to use an activation function (default True)
        """
        super().__init__()

        # Save the embedding dimension (for convenient external access)
        self.embed_dim = embed_dim

        # Embedding layer: maps discrete classes into a continuous vector space
        self.embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=embed_dim
        )

        # ReLU activation (required, adds non-linearity)
        self.use_act = use_act
        if use_act:
            self.act = GELU()


    def forward(self, pano_semantic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            pano_semantic: semantic label map [B, H_pano, W_pano] or [B, 1, H_pano, W_pano]
                          value range: integers 0-3

        Returns:
            semantic_features: semantic embedding features [B, 64, H_pano, W_pano]
        """
        # Handle the input shape
        if pano_semantic.dim() == 4 and pano_semantic.size(1) == 1:
            # [B, 1, H, W] → [B, H, W]
            pano_semantic = pano_semantic.squeeze(1)

        # Ensure the input is an integer type (required by Embedding)
        if pano_semantic.dtype != torch.long:
            pano_semantic = pano_semantic.long()

        # Ensure the value range is within [0, num_classes-1]
        # clamp prevents out-of-range values (should not happen in theory)
        pano_semantic = torch.clamp(pano_semantic, 0, 3)

        # Embedding lookup
        # [B, H, W] → [B, H, W, 64]
        semantic_embedded = self.embedding(pano_semantic)

        # Convert to the format commonly used by convolution layers
        # [B, H, W, 64] → [B, 64, H, W]
        semantic_features = semantic_embedded.permute(0, 3, 1, 2)

        # Optional GELU activation
        if self.use_act:
            semantic_features = self.act(semantic_features)

        return semantic_features


class TextureEncoder(nn.Module):
    """
    Texture encoder (Branch B)

    Input:
    - rgb_sat [B, 3, H_sat, W_sat]: satellite RGB image
    - V_mask_sat [B, H_sat, W_sat]: visibility mask (optional)

    Output: texture_features_sat [B, 64, H_sat, W_sat]

    Network structure:
    Satellite RGB [B, 3, H_sat, W_sat]
        ↓
    (optional) RGB × V_mask_sat (visibility attention filtering)
        ↓
    Conv 3×3, padding=1 (3→32) + ReLU
        ↓
    Conv 1×1 (32→64) + ReLU
        ↓
    Output [B, 64, H_sat, W_sat]

    Characteristics:
    - The first layer uses a 3×3 convolution to capture local texture features
    - The second layer uses a 1×1 convolution to expand the channel count
    - Keeps the spatial resolution unchanged (H_sat × W_sat)
    - Extracts real texture details (color/material/pattern)
    - padding=1 ensures the output size equals the input size
    """

    def __init__(
        self,
        in_channels: int = 3,
        intermediate_channels: int = 32,
        out_channels: int = 64,
        use_act: bool = True,
        apply_visibility_mask: bool = True
    ):
        """
        Args:
            in_channels: number of input channels (default 3: RGB)
            intermediate_channels: number of intermediate channels (default 32)
            out_channels: number of output channels (default 64)
            use_act: whether to use GELU activation (default True)
            apply_visibility_mask: whether to apply the visibility mask (default True)
        """
        super().__init__()

        # Layer 1: 3×3 convolution to expand channels and capture local texture features
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        # Layer 2: 1×1 convolution to further expand channels
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_act = use_act
        if use_act:
            self.act1 = GELU()
            self.act2 = GELU()

        self.apply_visibility_mask = apply_visibility_mask

        logger.debug(
            f"TextureEncoder initialized: "
            f"{in_channels}→{intermediate_channels}→{out_channels}, "
            f"visibility_mask={apply_visibility_mask}"
        )

    def forward(
        self,
        rgb_sat: torch.Tensor,
        V_mask_sat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            rgb_sat: satellite RGB image [B, 3, H_sat, W_sat]
            V_mask_sat: optional visibility mask [B, H_sat, W_sat], range [0, 1]

        Returns:
            texture_features: texture features [B, 64, H_sat, W_sat]
        """
        x = rgb_sat

        # Layer 1: 1×1 convolution
        x = self.conv1(x)
        x = self.bn1(x)
        if self.use_act:
            x = self.act1(x)

        # Layer 2: 1×1 convolution
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.act2(x)

        # Optional: visibility spatial attention filtering
        if V_mask_sat is not None and self.apply_visibility_mask:
            # V_mask_sat: [B, H_sat, W_sat] → [B, 1, H_sat, W_sat]
            V_mask_expanded = V_mask_sat.unsqueeze(1)
            V_mask_binary = (V_mask_expanded >= 0.3).float()
            x = x * V_mask_binary

        return x


class DualBranchFeatureEncoder(nn.Module):
    """
    Dual-branch feature encoder (DBFE, Module 2)

    Integrates two encoding branches:
    - Branch A: semantic embedding encoder
    - Branch B: texture encoder

    Input:
    - pano_semantic [B, H_pano, W_pano] or [B, 1, H_pano, W_pano]: semantic labels
    - rgb_sat [B, 3, H_sat, W_sat]: satellite RGB image
    - V_mask_sat [B, H_sat, W_sat]: visibility mask (optional)

    Output:
    - semantic_features [B, 64, H_pano, W_pano]: semantic features
    - texture_features_sat [B, 64, H_sat, W_sat]: satellite texture features
    """

    def __init__(
        self,
        semantic_channels: int = 64,
        texture_channels: int = 64,
        texture_intermediate_channels: int = 32,
        num_classes: int = 4,
        apply_visibility_mask: bool = True,
        use_semantic_mask: bool = True
    ):
        """
        Args:
            semantic_channels: number of semantic feature channels (default 64)
            texture_channels: number of texture feature channels (default 64)
            texture_intermediate_channels: number of intermediate texture encoder channels (default 32)
            num_classes: number of semantic classes (default 4)
            apply_visibility_mask: whether to apply the visibility mask (default True)
            use_semantic_mask: whether to use the semantic mask (default True)
                                when False, all-zero vectors are used as fill
        """
        super().__init__()

        self.use_semantic_mask = use_semantic_mask

        # Branch A: semantic embedding encoder
        self.semantic_encoder = SemanticEmbeddingEncoder(
            num_classes=num_classes,
            embed_dim=semantic_channels
        )

        # Branch B: texture encoder
        self.texture_encoder = TextureEncoder(
            in_channels=3,
            intermediate_channels=texture_intermediate_channels,
            out_channels=texture_channels,
            apply_visibility_mask=apply_visibility_mask
        )

        logger.info(
            f"DualBranchFeatureEncoder (DBFE) initialized: "
            f"semantic={semantic_channels}, texture={texture_channels}, "
            f"intermediate={texture_intermediate_channels}, visibility_mask={apply_visibility_mask}, "
            f"use_semantic_mask={use_semantic_mask}"
        )

    def forward(
        self,
        pano_semantic: torch.Tensor,
        rgb_sat: torch.Tensor,
        V_mask_sat: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            pano_semantic: semantic labels [B, H_pano, W_pano] or [B, 1, H_pano, W_pano]
            rgb_sat: satellite RGB image [B, 3, H_sat, W_sat]
            V_mask_sat: optional visibility mask [B, H_sat, W_sat]

        Returns:
            dict: {
                'semantic_features': [B, 64, H_pano, W_pano],
                'texture_features_sat': [B, 64, H_sat, W_sat]
            }
        """
        # Branch A: extract semantic embedding features
        if self.use_semantic_mask:
            # Use the real semantic mask
            semantic_features = self.semantic_encoder(pano_semantic)
        else:
            # Without the semantic mask, fill with all-zero vectors
            # get the device
            device = rgb_sat.device

            # Determine the shape of the semantic mask
            # if pano_semantic is provided, use its shape; otherwise infer from the satellite image shape
            if pano_semantic is not None:
                if pano_semantic.dim() == 3:
                    B, H_pano, W_pano = pano_semantic.shape
                else:
                    B = pano_semantic.shape[0]
                    # default to 256x512 as the panorama size
                    H_pano, W_pano = 256, 512
            else:
                B = rgb_sat.shape[0]
                H_pano, W_pano = 256, 512  # default panorama size

            # Create an all-zero tensor (same spatial resolution as the semantic features)
            # semantic_encoder output [B, semantic_channels, H_pano, W_pano]
            semantic_features = torch.zeros(
                (B, self.semantic_encoder.embed_dim, H_pano, W_pano),
                dtype=rgb_sat.dtype,
                device=device
            )

        # Branch B: extract texture features
        texture_features_sat = self.texture_encoder(rgb_sat, V_mask_sat)

        return {
            'semantic_features': semantic_features,
            'texture_features_sat': texture_features_sat
        }

    def get_output_info(self) -> Dict:
        """
        Get output information

        Returns:
            dict: shape and dtype information of output tensors
        """
        return {
            'semantic_features': {
                'shape': '(None, 64, H_pano, W_pano)',
                'description': 'semantic embedding features, converting discrete classes into continuous vector representations'
            },
            'texture_features_sat': {
                'shape': '(None, 64, H_sat, W_sat)',
                'description': 'satellite texture features, extracting real texture details'
            }
        }


def create_dual_branch_encoder(
    semantic_channels: int = 64,
    texture_channels: int = 64,
    texture_intermediate_channels: int = 32,
    num_classes: int = 4,
    apply_visibility_mask: bool = True
) -> DualBranchFeatureEncoder:
    """
    Factory function: create the dual-branch feature encoder

    Args:
        semantic_channels: number of semantic feature channels
        texture_channels: number of texture feature channels
        texture_intermediate_channels: number of intermediate texture encoder channels
        num_classes: number of semantic classes
        apply_visibility_mask: whether to apply the visibility mask

    Returns:
        DualBranchFeatureEncoder instance
    """
    encoder = DualBranchFeatureEncoder(
        semantic_channels=semantic_channels,
        texture_channels=texture_channels,
        texture_intermediate_channels=texture_intermediate_channels,
        num_classes=num_classes,
        apply_visibility_mask=apply_visibility_mask
    )

    return encoder


# =============================================================================
# Module 3: feature-level cross-view alignment module
# =============================================================================

class CrossViewAlignmentModule(nn.Module):
    """
    Feature-level cross-view alignment module (Module 3)

    Function: applies the M_geo transformation at the feature level, avoiding physical texture errors

    Input:
    - semantic_features [B, 64, H_pano, W_pano]: semantic features
    - texture_features_sat [B, 64, H_sat, W_sat]: satellite texture features
    - M_geo [B, H_pano, W_pano, 2]: correspondence matrix (floating-point coordinates)

    Output:
    - unified_features [B, 64, H_pano, W_pano]: primary features of the unified view

    Core operations:
    1. Feature-level M_geo transformation (bilinear interpolation)
    2. Feature fusion (gating mechanism)

    Ablation options:
    - use_polar_transform: use polar transformation instead of M_geo
    - use_learnable_matrix: use a learnable matrix instead of the physically computed M_geo
    """

    def __init__(
        self,
        feature_channels: int = 64,
        use_polar_transform: bool = False,
        polar_output_size: tuple = (128, 256),
        use_learnable_matrix: bool = False,
        sat_resolution: tuple = (128, 128),
        fusion_strategy: str = 'gate'
    ):
        """
        Args:
            feature_channels: number of feature channels (default 64)
            use_polar_transform: use polar transformation (ablation option, default False)
            polar_output_size: polar output size (H, W) (default 128×256)
            use_learnable_matrix: use a learnable matrix (ablation option, default False)
            sat_resolution: satellite image resolution (H_sat, W_sat) (default 128×128)
        """
        super().__init__()

        self.feature_channels = feature_channels
        self.use_polar_transform = use_polar_transform
        self.use_learnable_matrix = use_learnable_matrix
        self.sat_resolution = sat_resolution

        # Add LayerNorm to standardize the feature ranges, improving fusion quality
        self.norm_semantic = nn.LayerNorm(feature_channels)
        self.norm_texture = nn.LayerNorm(feature_channels)

        # Feature fusion module (Section 4.3.3): all variants fuse the semantic features
        # with the warped texture features while keeping the output shape [B, C, H, W]
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == 'gate':
            # Gated fusion (default): global average pooling + an MLP computes
            # per-channel gating weights
            self.gate_network = nn.Sequential(
                nn.Linear(feature_channels * 2, feature_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feature_channels // 2, feature_channels),
                nn.Sigmoid()  # outputs gating weights in [0, 1]
            )
            self.fusion_proj = None
            self.cross_attn = None
            self.attn_window_size = None
        elif fusion_strategy == 'rsc':
            # RSC (Raw-Semantic Concat): concatenate the semantic and texture features
            # and project back with a 1×1 conv (a per-pixel linear layer)
            self.fusion_proj = nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1)
            self.gate_network = None
            self.cross_attn = None
            self.attn_window_size = None
        elif fusion_strategy == 'ssi':
            # SSI (Static Semantic-Infill): element-wise addition of the two feature maps
            # followed by a 1×1 conv
            self.fusion_proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
            self.gate_network = None
            self.cross_attn = None
            self.attn_window_size = None
        elif fusion_strategy == 'cma':
            # CMA (Cross-Modal Attention): window-based multi-head cross-attention with the
            # semantic features as query and the warped texture features as key/value
            self.attn_window_size = 8  # 8×8 windows keep the attention cost tractable
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feature_channels,
                num_heads=8,
                batch_first=True
            )
            self.gate_network = None
            self.fusion_proj = None
        else:
            raise ValueError(
                f"Unknown fusion strategy: {fusion_strategy}. "
                f"Supported strategies: 'gate', 'rsc', 'cma', 'ssi'"
            )

        # Ablation option 1: polar transformation
        if use_polar_transform:
            from .polar_transform_model import PolarTransform
            self.polar_transform = PolarTransform(
                output_size=polar_output_size,
                center=None,  # use the image center
                max_radius=None
            )
            logger.info(f"CrossViewAlignmentModule initialized with Polar Transform: output_size={polar_output_size}")

        # Ablation option 2: learnable matrix
        if use_learnable_matrix:
            # Create the learnable mapping matrix
            # initialize with values similar to M_geo (a mapping outward from the center)
            H_pano, W_pano = polar_output_size
            H_sat, W_sat = sat_resolution

            # Create the initial grid: a radial mapping outward from the center
            y_coords = torch.linspace(0, H_sat - 1, H_pano)
            x_coords = torch.linspace(0, W_sat - 1, W_pano)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Initialize as a mapping outward from the center
            center_y, center_x = H_sat / 2, W_sat / 2
            initial_matrix = torch.stack([
                grid_x,
                grid_y
            ], dim=-1)  # [H_pano, W_pano, 2]

            # Register as a learnable parameter
            self.learned_M_geo = nn.Parameter(
                initial_matrix.float(),
                requires_grad=True
            )
            logger.info(f"CrossViewAlignmentModule initialized with Learnable Matrix: shape={initial_matrix.shape}")

        logger.debug(
            f"CrossViewAlignmentModule initialized: channels={feature_channels}, "
            f"polar_transform={use_polar_transform}, learnable_matrix={use_learnable_matrix}, "
            f"with LayerNorm and Gated Fusion"
        )

    def forward(
        self,
        semantic_features: torch.Tensor,
        texture_features_sat: torch.Tensor,
        M_geo: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            semantic_features: semantic features [B, 64, H_pano, W_pano]
            texture_features_sat: satellite texture features [B, 64, H_sat, W_sat]
            M_geo: correspondence matrix [B, H_pano, W_pano, 2] (when the learnable matrix is not used)

        Returns:
            unified_features: primary features of the unified view [B, 64, H_pano, W_pano]
        """
        B, C, H_sat, W_sat = texture_features_sat.shape
        _, _, H_pano, W_pano = semantic_features.shape

        # Step 1: feature-level transformation (selected by ablation option)
        if self.use_polar_transform:
            # Ablation option 1: use polar transformation
            # first apply the polar transformation to the texture features
            # texture_features_sat: [B, C, H_sat, W_sat]
            texture_features_pano = self.apply_polar_transform(texture_features_sat)
            # [B, C, H_pano, W_pano]
        elif self.use_learnable_matrix:
            # Ablation option 2: use the learnable matrix
            # expand learned_M_geo to the batch dimension
            M_learned = self.learned_M_geo.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H_pano, W_pano, 2]
            texture_features_pano = self.apply_m_geo_transform(
                texture_features_sat, M_learned, (H_sat, W_sat)
            )
        else:
            # Original version: use the physically computed M_geo
            texture_features_pano = self.apply_m_geo_transform(
                texture_features_sat, M_geo, (H_sat, W_sat)
            )

        # Step 2: feature standardization (LayerNorm)
        # convert [B, C, H, W] → [B, H, W, C] for LayerNorm
        semantic_features_norm = self.norm_semantic(semantic_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        texture_features_norm = self.norm_texture(texture_features_pano.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Step 3: feature fusion (strategy selected by the Section 4.3.3 ablation)
        if self.fusion_strategy == 'rsc':
            # RSC (Raw-Semantic Concat): concatenate + single 1×1 conv projection
            combined = torch.cat([semantic_features_norm, texture_features_norm], dim=1)  # [B, 2C, H, W]
            unified_features = self.fusion_proj(combined)  # [B, C, H, W]
        elif self.fusion_strategy == 'ssi':
            # SSI (Static Semantic-Infill): element-wise addition + single 1×1 conv projection
            unified_features = self.fusion_proj(semantic_features_norm + texture_features_norm)  # [B, C, H, W]
        elif self.fusion_strategy == 'cma':
            # CMA (Cross-Modal Attention): window-based multi-head cross-attention
            unified_features = self._window_cross_attention(semantic_features_norm, texture_features_norm)  # [B, C, H, W]
        else:
            # Gated fusion (default)
            # 3.1 Compute the global feature descriptors (global average pooling)
            semantic_global = semantic_features_norm.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
            texture_global = texture_features_norm.mean(dim=[2, 3], keepdim=True)    # [B, C, 1, 1]

            # 3.2 Concatenate the global features
            global_features = torch.cat([semantic_global, texture_global], dim=1)  # [B, 2C, 1, 1]

            # 3.3 Compute the weights via the gating network
            gate_weights = self.gate_network(global_features.squeeze(-1).squeeze(-1))  # [B, C]
            gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

            # 3.4 Apply the gating weights for adaptive fusion
            # gate_weights controls the blending ratio of semantic and texture
            unified_features = gate_weights * semantic_features_norm + (1 - gate_weights) * texture_features_norm

        # [B, 64, H_pano, W_pano]

        return unified_features

    def _window_cross_attention(
        self,
        semantic: torch.Tensor,
        texture: torch.Tensor
    ) -> torch.Tensor:
        """
        Window-based multi-head cross-attention fusion (CMA)

        The feature maps are partitioned into ws×ws windows; within each window the
        semantic features attend to the warped texture features. Windowing avoids the
        prohibitive O(L²) cost of full attention over 128×256 tokens.

        Args:
            semantic: [B, C, H, W] (query)
            texture: [B, C, H, W] (key/value)

        Returns:
            fused: [B, C, H, W]
        """
        B, C, H, W = semantic.shape
        ws = self.attn_window_size
        if H % ws != 0 or W % ws != 0:
            raise ValueError(
                f"CMA fusion requires H and W divisible by the window size {ws}, "
                f"got H={H}, W={W}"
            )

        # Partition into windows: [B, C, H, W] → [B*num_windows, ws*ws, C]
        q = semantic.view(B, C, H // ws, ws, W // ws, ws) \
                    .permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        kv = texture.view(B, C, H // ws, ws, W // ws, ws) \
                    .permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)

        out, _ = self.cross_attn(q, kv, kv)  # [B*num_windows, ws*ws, C]

        # Reverse the partition: [B*num_windows, ws*ws, C] → [B, C, H, W]
        out = out.view(B, H // ws, W // ws, ws, ws, C) \
                 .permute(0, 1, 3, 2, 4, 5).reshape(B, C, H, W)

        return out

    def apply_m_geo_transform(
        self,
        satellite_features: torch.Tensor,
        M_geo: torch.Tensor,
        sat_resolution: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Feature-level M_geo transformation (bilinear interpolation)

        Args:
            satellite_features: satellite-view features [B, C, H_sat, W_sat]
            M_geo: correspondence matrix [B, H_pano, W_pano, 2]
            sat_resolution: satellite image resolution (H_sat, W_sat)

        Returns:
            pano_features: panorama-view features [B, C, H_pano, W_pano]
        """
        B, C, H_sat, W_sat = satellite_features.shape
        _, H_pano, W_pano, _ = M_geo.shape

        # Normalize M_geo to [-1, 1] (required by F.grid_sample)
        # normalize the x coordinates
        x_coords = M_geo[..., 0]  # [B, H_pano, W_pano]
        x_norm = (x_coords / (W_sat - 1)) * 2 - 1

        # normalize the y coordinates
        y_coords = M_geo[..., 1]  # [B, H_pano, W_pano]
        y_norm = (y_coords / (H_sat - 1)) * 2 - 1

        # Assemble the grid: [B, H_pano, W_pano, 2]
        grid = torch.stack([x_norm, y_norm], dim=-1)

        # Handle invalid coordinates (-1 denotes the sky region)
        # create the valid mask
        valid_mask = (M_geo[..., 0] >= 0) & (M_geo[..., 1] >= 0)
        valid_mask = valid_mask.unsqueeze(1).float()  # [B, 1, H_pano, W_pano]

        # Bilinear interpolation sampling
        pano_features = F.grid_sample(
            satellite_features,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        # [B, C, H_pano, W_pano]

        # Apply the valid mask (zero out invalid regions)
        pano_features = pano_features * valid_mask

        return pano_features

    def apply_polar_transform(
        self,
        satellite_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Polar transformation (ablation option 1)

        Converts satellite-view features into panorama-view features

        Args:
            satellite_features: satellite-view features [B, C, H_sat, W_sat]

        Returns:
            pano_features: panorama-view features [B, C, H_pano, W_pano]
        """
        # Use the PolarTransform module
        # the features first need to be converted into an image-like format
        # satellite_features: [B, C, H_sat, W_sat]

        pano_features = self.polar_transform(satellite_features)

        return pano_features


# =============================================================================
# Module 4: deep feature extraction module (modified ResNet)
# =============================================================================

class ModifiedResNet(nn.Module):
    """
    Modified ResNet (Module 4)

    Modifications:
    1. First convolution layer: input channels changed from 3 to 64
    2. Kaiming initialization: weight initialization adapted to multi-channel inputs
    3. Learnable downsampling layer: uses Conv2d(64,64,3,stride=2) instead of the fixed MaxPool
    4. Adaptive pooling: unifies the output to 1×1

    Notes:
    - No pretrained weights are used; all parameters are trained from scratch
    - Uses learnable convolution downsampling, which is more suitable for semantic embedding features than MaxPool
    - The downsampling layer can learn the optimal feature aggregation, balancing detail preservation and computational efficiency

    Input: F_primary [B, 64, H_pano, W_pano]
    Output: F_high [B, 512]
    """

    def __init__(
        self,
        resnet_variant: str = 'resnet18',
        in_channels: int = 64,
        use_learnable_downsample: bool = True
    ):
        """
        Args:
            resnet_variant: ResNet variant (resnet18/resnet34/resnet50)
            in_channels: number of input channels (default 64)
            use_learnable_downsample: whether to use a learnable downsampling layer (default True)
        """
        super().__init__()

        self.resnet_variant = resnet_variant
        self.in_channels = in_channels
        self.use_learnable_downsample = use_learnable_downsample

        # Create a standard ResNet (no pretrained weights)
        # uses the new weights API (torchvision 0.13+)
        if resnet_variant == 'resnet18':
            resnet = models.resnet18()
            self.out_features = 512
        elif resnet_variant == 'resnet34':
            resnet = models.resnet34()
            self.out_features = 512
        elif resnet_variant == 'resnet50':
            resnet = models.resnet50()
            self.out_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")

        # Modify the first convolution layer to accommodate 64-channel input
        self.conv1 = nn.Conv2d(
            in_channels, 64,  # 64→64
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Improved initialization strategy: Xavier initialization, more suitable for the 64→64 case
        nn.init.xavier_normal_(self.conv1.weight, gain=1.0)

        # Learnable downsampling layer (replaces MaxPool)
        if use_learnable_downsample:
            self.downsample = nn.Conv2d(
                64, 64,              # keeps the channel count unchanged
                kernel_size=3,        # 3×3 convolution kernel
                stride=2,             # 2× downsampling
                padding=1,            # keeps boundary alignment
                bias=False            # no bias (BN follows)
            )
            self.bn_downsample = nn.BatchNorm2d(64)

            # Use Kaiming initialization, paired with the ReLU activation
            nn.init.kaiming_normal_(
                self.downsample.weight,
                mode='fan_out',
                nonlinearity='relu'
            )

        # Keep the other ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Adaptive average pooling
        self.avgpool = resnet.avgpool

        logger.info(
            f"ModifiedResNet initialized: "
            f"variant={resnet_variant}, in_channels={in_channels}, "
            f"out_features={self.out_features}, pretrained=False, "
            f"learnable_downsample={use_learnable_downsample}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: input features [B, 64, H, W], where H, W are usually 224

        Returns:
            features: high-level features [B, out_features]

        """
        # First convolution layer (modified)
        x = self.conv1(x)       # [B, 64, H, W] → [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)

        # Learnable downsampling (replaces MaxPool)
        if self.use_learnable_downsample:
            x = self.downsample(x)           # [B, 64, H/2, W/2] → [B, 64, H/4, W/4]
            x = self.bn_downsample(x)
            x = self.relu(x)

        # ResNet layers
        x = self.layer1(x)      # size unchanged
        x = self.layer2(x)      # size halved
        x = self.layer3(x)      # size halved
        x = self.layer4(x)      # size halved

        # Adaptive average pooling
        x = self.avgpool(x)     # → [B, out_features, 1, 1]

        # Flatten
        x = torch.flatten(x, 1)  # → [B, out_features]

        return x


class BackboneEncoder(nn.Module):
    """
    General backbone encoder (Module 4)

    Supports multiple backbone architectures:
    - ResNet family: resnet18, resnet34, resnet50, resnet101
    - ConvNext family: convnext_tiny, convnext_small, convnext_base
    - ViT family: vit_tiny, vit_small, vit_base
    - Swin family: swin_tiny, swin_small, swin_base

    Modifications:
    1. Adapts the input channels: from the default 3 channels to in_channels (default 64)
    2. Supports optional pretrained weights
    3. Adaptive pooling: unifies the output to 1×1
    4. Handles different input sizes (especially ViT which requires a fixed size)

    Input: features [B, in_channels, H, W]
    Output: features [B, out_features]
    """

    def __init__(
        self,
        backbone_type: str = 'resnet18',
        in_channels: int = 64,
        pretrained: bool = False,
        img_size: int = 224
    ):
        """
        Args:
            backbone_type: backbone type
                - ResNet: resnet18, resnet34, resnet50, resnet101
                - ConvNext: convnext_tiny, convnext_small, convnext_base
                - ViT: vit_tiny, vit_small, vit_base
                - Swin: swin_tiny, swin_small, swin_base
            in_channels: number of input channels (default 64)
            pretrained: whether to use pretrained weights (default False)
            img_size: input image size (default 224, required by ViT/Swin)
        """
        super().__init__()

        self.backbone_type = backbone_type
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.img_size = img_size

        # Create the model according to the backbone type
        if backbone_type.startswith('resnet'):
            # ResNet family
            self.model, self.out_features = self._create_resnet(
                backbone_type, in_channels, pretrained
            )
            self.model_family = 'resnet'
        elif backbone_type.startswith('convnext'):
            # ConvNext family (uses the timm library)
            self.model, self.out_features = self._create_convnext(
                backbone_type, in_channels, pretrained
            )
            self.model_family = 'convnext'
        elif backbone_type.startswith('vit'):
            # ViT family (uses the timm library)
            self.model, self.out_features = self._create_vit(
                backbone_type, in_channels, pretrained, img_size
            )
            self.model_family = 'vit'
        elif backbone_type.startswith('swin'):
            # Swin family (uses the timm library)
            self.model, self.out_features = self._create_swin(
                backbone_type, in_channels, pretrained, img_size
            )
            self.model_family = 'swin'
        else:
            raise ValueError(
                f"Unsupported backbone type: {backbone_type}. "
                f"Supported types: ResNet (resnet18/34/50/101), "
                f"ConvNext (convnext_tiny/small/base), "
                f"ViT (vit_tiny/small/base), "
                f"Swin (swin_tiny/small/base)"
            )

        logger.info(
            f"BackboneEncoder initialized: "
            f"backbone={backbone_type}, in_channels={in_channels}, "
            f"out_features={self.out_features}, pretrained={pretrained}, "
            f"img_size={img_size}, family={self.model_family}"
        )

    def _create_resnet(self, variant: str, in_channels: int, pretrained: bool):
        """Create a ResNet backbone"""
        import torchvision.models as models

        if variant == 'resnet18':
            resnet = models.resnet18(weights=None if not pretrained else 'IMAGENET1K_V1')
            out_features = 512
        elif variant == 'resnet34':
            resnet = models.resnet34(weights=None if not pretrained else 'IMAGENET1K_V1')
            out_features = 512
        elif variant == 'resnet50':
            resnet = models.resnet50(weights=None if not pretrained else 'IMAGENET1K_V1')
            out_features = 2048
        elif variant == 'resnet101':
            resnet = models.resnet101(weights=None if not pretrained else 'IMAGENET1K_V1')
            out_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        # Modify the first convolution layer to accommodate the in_channels input
        resnet.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Xavier initialization (when pretrained weights are not used)
        if not pretrained:
            nn.init.xavier_normal_(resnet.conv1.weight, gain=1.0)

        # Use the remaining ResNet layers
        return resnet, out_features

    def _create_convnext(self, variant: str, in_channels: int, pretrained: bool):
        """Create a ConvNext backbone (uses the timm library)"""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "The timm library is required to use ConvNext. "
                "Please run: pip install timm"
            )

        # ConvNext model mapping
        convnext_models = {
            'convnext_tiny': 'convnext_tiny',
            'convnext_small': 'convnext_small',
            'convnext_base': 'convnext_base',
            'convnext_large': 'convnext_large'
        }

        if variant not in convnext_models:
            raise ValueError(f"Unsupported ConvNext variant: {variant}")

        # Create the model
        model_name = convnext_models[variant]
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels
        )

        # Get the output feature dimension
        out_features_map = {
            'convnext_tiny': 768,
            'convnext_small': 768,
            'convnext_base': 1024,
            'convnext_large': 1536
        }
        out_features = out_features_map.get(variant, 768)

        return model, out_features

    def _create_vit(self, variant: str, in_channels: int, pretrained: bool, img_size: int):
        """Create a Vision Transformer backbone (uses the timm library)"""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "The timm library is required to use ViT. "
                "Please run: pip install timm"
            )

        # ViT model mapping
        vit_models = {
            'vit_tiny': 'vit_tiny_patch16_224',
            'vit_small': 'vit_small_patch16_224',
            'vit_base': 'vit_base_patch16_224',
            'vit_large': 'vit_large_patch16_224'
        }

        if variant not in vit_models:
            raise ValueError(f"Unsupported ViT variant: {variant}")

        # Create the model
        model_name = vit_models[variant]
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=img_size,
            num_classes=0  # remove the classification head, extract features only
        )

        # Get the output feature dimension
        out_features_map = {
            'vit_tiny': 192,
            'vit_small': 384,
            'vit_base': 768,
            'vit_large': 1024
        }
        out_features = out_features_map.get(variant, 768)

        # ViT requires a fixed input size; add adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((img_size, img_size))

        return model, out_features

    def _create_swin(self, variant: str, in_channels: int, pretrained: bool, img_size: int):
        """Create a Swin Transformer backbone (uses the timm library)"""
        try:
            import timm
        except ImportError:
            raise ImportError(
                "The timm library is required to use Swin Transformer. "
                "Please run: pip install timm"
            )

        # Swin model mapping
        swin_models = {
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_small': 'swin_small_patch4_window7_224',
            'swin_base': 'swin_base_patch4_window7_224',
            'swin_large': 'swin_large_patch4_window7_224'
        }

        if variant not in swin_models:
            raise ValueError(f"Unsupported Swin variant: {variant}")

        # Create the model
        model_name = swin_models[variant]
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # remove the classification head
            img_size=img_size  # set the input image size
        )

        # Get the output feature dimension
        out_features_map = {
            'swin_tiny': 768,
            'swin_small': 768,
            'swin_base': 1024,
            'swin_large': 1536
        }
        out_features = out_features_map.get(variant, 768)

        # Swin also requires a fixed input size; add adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((img_size, img_size))

        return model, out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: input features [B, in_channels, H, W]

        Returns:
            features: high-level features [B, out_features]
        """
        if self.model_family == 'resnet':
            # ResNet forward pass
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)

        elif self.model_family == 'convnext':
            # ConvNext forward pass
            features = self.model.forward_features(x)
            x = self.model.head.global_pool(features)
            x = self.model.head.flatten(x)

        elif self.model_family == 'vit':
            # ViT forward pass
            # adaptive pooling to the fixed size (if needed)
            if x.shape[2:] != (self.img_size, self.img_size):
                x = self.adaptive_pool(x)

            # the timm ViT model's forward_features returns token embeddings
            features = self.model.forward_features(x)  # [B, num_tokens+1, embed_dim]

            # take the CLS token (the first token)
            x = features[:, 0]  # [B, embed_dim]

        elif self.model_family == 'swin':
            # Swin Transformer forward pass
            # Swin also requires a fixed input size; adaptively pool to img_size
            if x.shape[2:] != (self.img_size, self.img_size):
                x = self.adaptive_pool(x)

            features = self.model.forward_features(x)  # [B, H', W', C]

            # Global average pooling
            x = F.adaptive_avg_pool2d(features.permute(0, 3, 1, 2), 1)  # [B, C, 1, 1]
            x = torch.flatten(x, 1)  # [B, C]

        return x

    def get_output_dim(self) -> int:
        """Get the output feature dimension"""
        return self.out_features


# Keep the ModifiedResNet alias for backward compatibility
# ModifiedResNet is now implemented via BackboneEncoder
ModifiedResNet = BackboneEncoder


# =============================================================================
# Module 5: classification prediction module
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
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 3,
        dropout_p: float = 0.1
    ):
        """
        Args:
            input_dim: input feature dimension (default 512 corresponds to ResNet18)
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
        x = self.sigmoid(x)

        return x
