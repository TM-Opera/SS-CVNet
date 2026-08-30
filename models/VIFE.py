import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import RMSNorm
import logging

logger = logging.getLogger(__name__)


class GELU(nn.Module):
    """GELU activation function

    Gaussian Error Linear Unit
    Has smoother gradients than ReLU and performs better in modern networks
    """
    def forward(self, x):
        return F.gelu(x)


# =============================================================================
# SEBlock: Squeeze-and-Excitation attention module
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block

    Based on the paper:
    Squeeze-and-Excitation Networks
    Hu, Shen, Sun (CVPR 2018)
    https://arxiv.org/abs/1709.01507

    Input: features [B, C, H, W]
    Output: attention weights [B, C, 1, 1] (per-channel scaling factors)

    Characteristics:
    1. Global average pooling: compresses the spatial dimensions to 1×1
    2. Two fully connected layers: dimensionality reduction → expansion (reduction ratio is usually reduction=16)
    3. Sigmoid activation: produces channel attention weights in the [0, 1] range
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16
    ):
        """
        Args:
            channels: number of input feature channels
            reduction: reduction ratio (default 16)
                intermediate dimension = channels // reduction
        """
        super().__init__()

        self.channels = channels
        self.reduction = reduction

        # Global average pooling: [B, C, H, W] → [B, C, 1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Two fully connected layers (dimensionality reduction → expansion)
        reduced_channels = max(channels // reduction, 1)  # ensure at least 1

        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features [B, C, H, W]

        Returns:
            attention: channel attention weights [B, C, 1, 1]
        """
        B, C, H, W = x.shape

        # Squeeze: global average pooling
        # [B, C, H, W] → [B, C, 1, 1]
        squeeze = self.global_avg_pool(x)

        # Excitation: two fully connected layers
        # [B, C, 1, 1] → [B, C]
        squeeze = squeeze.view(B, C)

        # [B, C] → [B, C//reduction]
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)

        # [B, C//reduction] → [B, C]
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        # [B, C] → [B, C, 1, 1]
        attention = excitation.view(B, C, 1, 1)

        return attention


# =============================================================================
# VIFE: improved VIFE module (with ablation options)
# =============================================================================

class VIFE(nn.Module):
    """
    Input:
    - rgb: RGB remote sensing data [B, 3, H, W]
    - dsm: DSM height data [B, 1, H, W]
    - vis_mask: visibility mask [B, 1, H, W]

    Output:
    - output: fused features [B, 64, H, W]

    Characteristics:
    1. Feature extraction convolution block (3×3 + 1×1, mapped to 64 dimensions)
    2. Independent 1×1 convolutions for Q, K, V
    3. Single-head self-attention (not multi-head)
    4. Visibility mask as spatial attention (element-wise multiplication)

    Ablation options:
    - use_se_attention: use SEBlock to generate channel attention weights
    - use_center_circle_mask: use a center circular mask (handled in VIFETextureEncoder)
    """

    def __init__(
        self,
        in_channels_rgb: int = 3,
        in_channels_dsm: int = 1,
        out_channels: int = 64,
        lambda_weight: float = 0.8,
        use_window_attention: bool = True,
        window_size: int = 4,
        use_se_attention: bool = False,
        se_reduction: int = 16
    ):
        """
        Args:
            in_channels_rgb: number of RGB input channels (default 3)
            in_channels_dsm: number of DSM input channels (default 1)
            out_channels: number of output feature channels (default 64)
            lambda_weight: depth similarity weight coefficient λ (default 0.8)
            use_window_attention: whether to use window attention (default True)
                True: window attention, low GPU memory usage (recommended)
                False: global attention, high GPU memory usage
            window_size: window size (default 4)
                Recommended: 4 (divides 128) | 8 (larger receptive field)
                Note: the input size must be divisible by window_size
            use_se_attention: use SEBlock channel attention (ablation option, default False)
            se_reduction: SEBlock reduction ratio (default 16)
        """
        super().__init__()

        self.in_channels_rgb = in_channels_rgb
        self.in_channels_dsm = in_channels_dsm
        self.out_channels = out_channels
        self.lambda_weight = lambda_weight
        self.use_window_attention = use_window_attention
        self.window_size = window_size
        self.use_se_attention = use_se_attention

        # LayerNorm layers
        self.norm_rgb = nn.LayerNorm(out_channels)
        self.norm_dsm = nn.LayerNorm(in_channels_dsm)

        # Q, K RMSNorm layers
        self.q_norm = RMSNorm(out_channels)
        self.k_norm = RMSNorm(out_channels)

        # SEBlock (ablation option)
        if self.use_se_attention:
            self.se_block = SEBlock(
                channels=out_channels,
                reduction=se_reduction
            )
            logger.info(f"VIFE initialized with SEBlock attention (reduction={se_reduction})")

        # =====================================================================
        # Feature extraction convolution block
        # Input: RGB [B, 3, H, W]
        # Output: features [B, 64, H, W]
        # =====================================================================
        self.feature_extractor = nn.Sequential(
            # 3×3 convolution for spatial feature extraction
            nn.Conv2d(
                in_channels_rgb,
                64,  # intermediate channel count
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False
            ),

            # 1×1 convolution to expand to 64 dimensions
            nn.Conv2d(
                64,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False
            ),
            GELU()
        )

        # =====================================================================
        # Independent 1×1 convolutions for Q, K, V
        # Input: features [B, out_channels, H, W]
        # Output: Q/K/V [B, out_channels, H, W]
        # Uses 1×1 depthwise convolution (groups=out_channels)
        # =====================================================================
        self.q_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            groups=out_channels,  # depthwise convolution
            bias=False
        )

        self.k_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            groups=out_channels,  # depthwise convolution
            bias=False
        )

        self.v_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            groups=out_channels,  # depthwise convolution
            bias=False
        )

        # Scaling factor (single-head attention)
        self.scale = out_channels ** -0.5


    def forward(
        self,
        rgb: torch.Tensor,
        dsm: torch.Tensor,
        vis_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            rgb: RGB remote sensing data [B, 3, H, W]
            dsm: DSM height data [B, 1, H, W]
            vis_mask: visibility mask [B, 1, H, W], range [0, 1]

        Returns:
            output: fused features [B, 64, H, W]
        """
        B, _, H, W = rgb.shape
        N = H * W  # total number of pixels

        # =====================================================================
        # Step 1: feature extraction
        # Expand the RGB image into a 64-dimensional feature space via the convolution block
        # =====================================================================
        feat = self.feature_extractor(rgb)  # [B, 64, H, W]

        B, _, H_feat, W_feat = feat.shape
        N_feat = H_feat * W_feat

        feat_norm = self.norm_rgb(feat.permute(0, 2, 3, 1))  # [B, H, W, 64]
        feat_norm = feat_norm.permute(0, 3, 1, 2)  # [B, 64, H, W]

        # =====================================================================
        # Step 2: compute Q, K, V
        # Uses independent 1×1 convolutions
        # =====================================================================
        q = self.q_conv(feat_norm)  # [B, 64, H, W]
        k = self.k_conv(feat_norm)  # [B, 64, H, W]
        v = self.v_conv(feat_norm)  # [B, 64, H, W]

        # Reshape to single-head format: [B, N_feat, C]
        q = self.q_norm(q.reshape(B, self.out_channels, N_feat).permute(0, 2, 1))  # [B, N_feat, 64]
        k = self.k_norm(k.reshape(B, self.out_channels, N_feat).permute(0, 2, 1))  # [B, N_feat, 64]
        v = (v.reshape(B, self.out_channels, N_feat).permute(0, 2, 1))  # [B, N_feat, 64]

        # =====================================================================
        # Steps 3-7: self-attention computation (global or windowed mode)
        # =====================================================================
        if self.use_window_attention:
            # Windowed mode: compute self-attention within windows (DSM fused automatically)
            dsm = self.norm_dsm(dsm.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 1, H, W]

            # Compute window attention (window partitioning and depth similarity for the DSM handled internally)
            out = self._compute_window_attention(q, k, v, dsm, H_feat, W_feat)  # [B, N_feat, 64]
        else:
            # Global mode: compute global self-attention (high GPU memory usage)
            # [B, N_feat, 64] × [B, 64, N_feat] → [B, N_feat, N_feat]
            color_sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Compute depth similarity
            dsm = self.norm_dsm(dsm.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 1, H, W]
            depth_sim = self._compute_depth_similarity(dsm)  # [B, N_feat, N_feat]
            depth_sim = -self.lambda_weight * depth_sim

            # Composite similarity
            composite_sim = color_sim + depth_sim  # [B, N_feat, N_feat]

            # SoftMax normalization
            attn = F.softmax(composite_sim, dim=-1)  # [B, N_feat, N_feat]

            # Weighted sum
            out = torch.matmul(attn, v)  # [B, N_feat, 64]

        # =====================================================================
        # Step 8: restore shape + residual connection
        # =====================================================================
        out = out.permute(0, 2, 1).reshape(B, self.out_channels, H_feat, W_feat) + feat  # [B, 64, H, W]

        # =====================================================================
        # Step 9: apply the attention mechanism (ablation option)
        # =====================================================================
        if self.use_se_attention:
            # Use SEBlock to generate channel attention weights
            # SEBlock: [B, 64, H, W] → [B, 64, 1, 1]
            se_attention = self.se_block(out)  # [B, 64, 1, 1]
            # Per-channel scaling
            out = out * se_attention  # [B, 64, H, W]
        else:
            # Use the visibility mask as spatial attention
            # vis_mask: [B, 1, H, W]
            # Element-wise multiplication
            out = out * vis_mask  # [B, 64, H, W]

        return out

    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
        Partition the feature map into windows

        Args:
            x: [B, C, H, W]
            window_size: window size

        Returns:
            windows: [num_windows*B, C, window_size, window_size]
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.view(-1, C, window_size, window_size)
        return windows

    def _window_reverse(self, windows: torch.Tensor, H: int, W: int, window_size: int) -> torch.Tensor:
        """
        Merge windows back into the feature map

        Args:
            windows: [num_windows*B, C, window_size, window_size]
            H: original height
            W: original width
            window_size: window size

        Returns:
            x: [B, C, H, W]
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, -1, H, W)
        return x

    def _compute_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dsm: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Compute windowed self-attention (automatically fuses DSM depth similarity)

        Args:
            q: [B, N, C]
            k: [B, N, C]
            v: [B, N, C]
            dsm: [B, 1, H, W] DSM height map (same size as the feature map)
            H: feature map height
            W: feature map width

        Returns:
            out: [B, N, C]
        """
        B, N, C = q.shape

        # Reshape to 2D: [B, C, H, W]
        q = q.permute(0, 2, 1).reshape(B, C, H, W)
        k = k.permute(0, 2, 1).reshape(B, C, H, W)
        v = v.permute(0, 2, 1).reshape(B, C, H, W)

        # Partition into windows
        q_windows = self._window_partition(q, self.window_size)  # [num_windows*B, C, 4, 4]
        k_windows = self._window_partition(k, self.window_size)
        v_windows = self._window_partition(v, self.window_size)
        dsm_windows = self._window_partition(dsm, self.window_size)  # [num_windows*B, 1, 4, 4]

        # Reshape to [num_windows*B, window_size*window_size, C]
        num_windows = q_windows.shape[0]
        window_area = self.window_size * self.window_size

        q_windows = q_windows.view(num_windows, C, window_area).permute(0, 2, 1)
        k_windows = k_windows.view(num_windows, C, window_area).permute(0, 2, 1)
        v_windows = v_windows.view(num_windows, C, window_area).permute(0, 2, 1)

        # Compute color similarity: [num_windows*B, 16, 16]
        color_sim = torch.matmul(q_windows, k_windows.transpose(-2, -1))

        # Compute intra-window depth similarity
        dsm_vec = dsm_windows.view(num_windows, window_area)  # [num_windows*B, 16]
        depth_i = dsm_vec.unsqueeze(2)  # [num_windows*B, 16, 1]
        depth_j = dsm_vec.unsqueeze(1)  # [num_windows*B, 1, 16]
        depth_diff = torch.abs(depth_i - depth_j)  # [num_windows*B, 16, 16]
        depth_sim = -self.lambda_weight * depth_diff  # [num_windows*B, 16, 16]

        # Fuse depth similarity: composite similarity = color similarity + depth similarity
        composite_sim = (color_sim + depth_sim) * self.scale  # [num_windows*B, 16, 16]
        attn = F.softmax(composite_sim, dim=-1)

        # Weighted sum: [num_windows*B, 16, C]
        out_windows = torch.matmul(attn, v_windows)

        # Restore window shape: [num_windows*B, C, 4, 4]
        out_windows = out_windows.permute(0, 2, 1).view(num_windows, C, self.window_size, self.window_size)

        # Merge windows: [B, C, H, W]
        out = self._window_reverse(out_windows, H, W, self.window_size)

        # Reshape to [B, N, C]
        out = out.view(B, C, N).permute(0, 2, 1)

        return out


    def _compute_depth_similarity(self, dsm: torch.Tensor) -> torch.Tensor:
        """
        Compute the depth similarity matrix |di - dj|

        Args:
            dsm: DSM height data [B, 1, H, W]

        Returns:
            depth_diff: depth difference matrix [B, N, N]
        """
        B, _, H, W = dsm.shape
        N = H * W

        # Reshape to a vector: [B, 1, H, W] → [B, N]
        depth_vec = dsm.reshape(B, N)  # [B, N]

        # Compute the depth difference between all pixel pairs
        # [B, N, 1] - [B, 1, N] → [B, N, N]
        depth_i = depth_vec.unsqueeze(2)  # [B, N, 1]
        depth_j = depth_vec.unsqueeze(1)  # [B, 1, N]

        # Absolute value of the depth difference |di - dj|
        depth_diff = torch.abs(depth_i - depth_j)  # [B, N, N]

        return depth_diff


# =============================================================================
# Test DSAModuleV2
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("DSA Test")
    print("=" * 80)

    # Create random inputs
    B, H, W = 2, 32, 32
    rgb = torch.randn(B, 3, H, W)
    dsm = torch.randn(B, 1, H, W)
    vis_mask = torch.rand(B, 1, H, W)  # range [0, 1]

    print(f"\nInput shapes:")
    print(f"  rgb: {rgb.shape}")
    print(f"  dsm: {dsm.shape}")
    print(f"  vis_mask: {vis_mask.shape}")

    # Test DSA
    print("\n" + "-" * 80)
    print("Testing DSA")
    print("-" * 80)

    vife = VIFE(
        in_channels_rgb=3,
        in_channels_dsm=1,
        out_channels=64,
        lambda_weight=0.8
    )

    output = vife(rgb, dsm, vis_mask)
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in vife.parameters()):,}")

    # Gradient test
    print("\n" + "-" * 80)
    print("Gradient test")
    print("-" * 80)

    rgb_grad = rgb.clone().requires_grad_(True)
    dsm_grad = dsm.clone().requires_grad_(True)
    vis_mask_grad = vis_mask.clone().requires_grad_(True)

    output_grad = vife(rgb_grad, dsm_grad, vis_mask_grad)
    loss = output_grad.sum()
    loss.backward()

    print(f"rgb gradient shape: {rgb_grad.grad.shape}")
    print(f"dsm gradient shape: {dsm_grad.grad.shape}")
    print(f"vis_mask gradient shape: {vis_mask_grad.grad.shape}")
    print(f"rgb gradient norm: {rgb_grad.grad.norm().item():.6f}")
    print(f"dsm gradient norm: {dsm_grad.grad.norm().item():.6f}")
    print(f"vis_mask gradient norm: {vis_mask_grad.grad.norm().item():.6f}")

    print("\n" + "=" * 80)
    print("DSAModuleV2 test completed!")
    print("=" * 80)
