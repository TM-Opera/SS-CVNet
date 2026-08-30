import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProjectionTransformer(nn.Module):
    """Panoramic projection transformer (fully GPU-vectorized version)"""

    def __init__(self, device='cuda', in_channels=64):
        """
        Args:
            device: compute device ('cuda' or 'cpu')
            in_channels: number of input image channels, default 64
        """
        super(ProjectionTransformer, self).__init__()
        self.device = device
        self.in_channels = in_channels

    def forward(self, x, mode='fisheye'):
        """
        Forward pass

        Args:
            x: input tensor (B, C, H, W), H=128, W=256
            mode: transformation mode ('fisheye' or 'perspective')

        Returns:
            transformed tensor
        """
        with torch.no_grad():
            if mode == 'fisheye':
                return self.equirect_to_fisheye(x)
            elif mode == 'perspective':
                return self.equirect_to_perspective(x)
            else:
                raise ValueError(f"Unknown mode: {mode}")

    def equirect_to_fisheye(self, x):
        """
        Equirectangular projection -> fisheye projection (upper hemisphere)

        Args:
            x: (B, C, H, W), H=128, W=256

        Returns:
            (B, C, 128, 128) fisheye projection
        """
        B, C, H, W = x.shape
        out_size = 128

        # Create fisheye pixel coordinate grid (origin at center)
        y = torch.linspace(-1, 1, out_size, device=self.device)
        x_coord = torch.linspace(-1, 1, out_size, device=self.device)
        yy, xx = torch.meshgrid(y, x_coord, indexing='ij')  # (128, 128)

        # Polar coordinates
        r = torch.sqrt(xx ** 2 + yy ** 2)
        phi = torch.atan2(xx, -yy)  # azimuth angle [-π, π]

        # Valid region (inside the circle)
        mask = r <= 1.0

        # Zenith angle theta = r * (π/2)
        theta = r * (np.pi / 2)

        # Map to equirectangular coordinates
        eq_row = (theta / np.pi * H).clamp(0, H - 1)
        eq_col = ((phi + np.pi) / (2 * np.pi) * W).clamp(0, W - 1)

        # Normalize to [-1, 1] for grid_sample
        grid_row = eq_row / (H - 1) * 2 - 1
        grid_col = eq_col / (W - 1) * 2 - 1

        # Assemble the grid (1, out_size, out_size, 2)
        grid = torch.stack([grid_col, grid_row], dim=-1).unsqueeze(0)
        grid = grid.repeat(B, 1, 1, 1)  # (B, 128, 128, 2)

        # Bilinear interpolation via grid_sample (fully vectorized)
        output = F.grid_sample(
            x, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )  # (B, C, 128, 128)

        # Apply the mask (set outside-circle pixels to 0)
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).to(x.device)  # (1, 1, 128, 128)
        output = output * mask_expanded

        return output

    def equirect_to_perspective(self, x):
        """
        Equirectangular projection -> four-direction perspective projection

        Args:
            x: (B, C, H, W), H=128, W=256

        Returns:
            (B, C, 128, 512) four-direction concatenation
        """
        B, C, H, W = x.shape
        view_h, view_w = 128, 128
        hfov = vfov = np.pi / 2

        # Yaw angles of the four directions
        yaw_angles = [0.0, np.pi / 2, np.pi, -np.pi / 2]

        views = []
        for yaw in yaw_angles:
            view = self._single_perspective_view(
                x, yaw, hfov, vfov, view_h, view_w
            )
            views.append(view)

        # Horizontal concatenation
        output = torch.cat(views, dim=3)  # (B, C, 128, 512)
        return output

    def _single_perspective_view(self, x, yaw, hfov, vfov, out_h, out_w):
        """
        Perspective projection for a single direction

        Args:
            x: (B, C, H, W)
            yaw: horizontal rotation angle (radians)
            hfov, vfov: field of view
            out_h, out_w: output size

        Returns:
            (B, C, out_h, out_w)
        """
        B, C, H, W = x.shape

        # Focal lengths
        f_h = 1.0 / np.tan(hfov / 2)
        f_v = 1.0 / np.tan(vfov / 2)

        # Normalized pixel coordinates
        v = torch.linspace(-1, 1, out_h, device=self.device)
        u = torch.linspace(-1, 1, out_w, device=self.device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')

        # Camera coordinates
        nx = uu / f_h
        ny = -vv / f_v

        # Rotate to world coordinates
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)

        ray_x = nx * cos_yaw + sin_yaw
        ray_y = ny
        ray_z = -nx * sin_yaw + cos_yaw

        # Convert to spherical coordinates
        phi = torch.atan2(ray_x, ray_z)
        ray_len = torch.sqrt(ray_x ** 2 + ray_y ** 2 + ray_z ** 2)
        lat = torch.asin(ray_y / ray_len)

        # Map to equirectangular coordinates
        eq_col = ((phi + np.pi) / (2 * np.pi) * W).clamp(0, W - 1)
        eq_row = ((np.pi / 2 - lat) / np.pi * H).clamp(0, H - 1)

        # Normalize to [-1, 1] for grid_sample
        grid_row = eq_row / (H - 1) * 2 - 1
        grid_col = eq_col / (W - 1) * 2 - 1

        # Assemble the grid
        grid = torch.stack([grid_col, grid_row], dim=-1).unsqueeze(0)
        grid = grid.repeat(B, 1, 1, 1)  # (B, out_h, out_w, 2)

        # grid_sample
        output = F.grid_sample(
            x, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )  # (B, C, out_h, out_w)

        return output



    def to_fisheye(self, x):
        """
        Convert to fisheye projection

        Args:
            x: (B, C, 128, 256) input tensor

        Returns:
            (B, C, 128, 128) fisheye projection
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if x.dim() == 3:  # (C, H, W) → (1, C, H, W)
            x = x.unsqueeze(0)

        if x.device != torch.device(self.device):
            x = x.to(self.device)

        with torch.no_grad():
            return self.forward(x, mode='fisheye')

    def to_perspective(self, x):
        """
        Convert to four-direction perspective projection

        Args:
            x: (B, C, 128, 256) input tensor

        Returns:
            (B, C, 128, 512) four-direction concatenation
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if x.dim() == 3:  # (C, H, W) → (1, C, H, W)
            x = x.unsqueeze(0)

        if x.device != torch.device(self.device):
            x = x.to(self.device)

        with torch.no_grad():
            return self.forward(x, mode='perspective')

    def to_both(self, x):
        """
        Generate both projections at once

        Args:
            x: (B, C, 128, 256) input tensor

        Returns:
            fisheye: (B, C, 128, 128)
            perspective: (B, C, 128, 512)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.device != torch.device(self.device):
            x = x.to(self.device)

        with torch.no_grad():
            fisheye = self.forward(x, mode='fisheye')
            perspective = self.forward(x, mode='perspective')
            return fisheye, perspective


# =============================================================================
# Usage example
# =============================================================================

def example_usage():
    """Usage example"""
    print("=" * 60)
    print("GPU Panoramic Projection Transformer - Usage Example")
    print("=" * 60)

    # Create the transformer (64 channels by default)
    transformer = ProjectionTransformer(device='cuda', in_channels=64)

    # Example 1: convert from a numpy array
    print("\nExample 1: convert from a numpy array")
    panorama = np.random.rand(64, 128, 256).astype(np.float32)
    fisheye = transformer.to_fisheye(panorama)
    print(f"  Input: {panorama.shape}")
    print(f"  Fisheye output: {fisheye.shape}")

    # Example 2: from a torch tensor
    print("\nExample 2: from a torch tensor")
    panorama_tensor = torch.randn(1, 64, 128, 256).cuda()
    fisheye_tensor = transformer.to_fisheye(panorama_tensor)
    perspective_tensor = transformer.to_perspective(panorama_tensor)
    print(f"  Input: {panorama_tensor.shape}")
    print(f"  Fisheye: {fisheye_tensor.shape}")
    print(f"  Perspective: {perspective_tensor.shape}")

    # Example 3: batch processing
    print("\nExample 3: batch processing")
    batch_panorama = torch.randn(10, 64, 128, 256).cuda()
    fisheye_batch = transformer.to_fisheye(batch_panorama)
    perspective_batch = transformer.to_perspective(batch_panorama)
    print(f"  Input batch: {batch_panorama.shape}")
    print(f"  Fisheye batch: {fisheye_batch.shape}")
    print(f"  Perspective batch: {perspective_batch.shape}")

    # Example 4: generate both projections at once
    print("\nExample 4: generate both projections at once")
    fisheye_4, perspective_4 = transformer.to_both(batch_panorama)
    print(f"  Fisheye: {fisheye_4.shape}")
    print(f"  Perspective: {perspective_4.shape}")

    # Example 5: convert back to numpy
    print("\nExample 5: convert back to numpy")
    fisheye_numpy = fisheye_4[0].cpu().numpy()
    print(f"  numpy shape: {fisheye_numpy.shape}")

    # Example 6: use a different channel count
    print("\nExample 6: use a different channel count")
    transformer_3ch = ProjectionTransformer(device='cuda', in_channels=3)
    panorama_3ch = torch.randn(1, 3, 128, 256).cuda()
    fisheye_3ch = transformer_3ch.to_fisheye(panorama_3ch)
    print(f"  3-channel input: {panorama_3ch.shape}")
    print(f"  3-channel fisheye: {fisheye_3ch.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_usage()
