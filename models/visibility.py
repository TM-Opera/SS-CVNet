"""
Ground observation point visibility analysis module (CUDA vectorization + Bresenham ray tracing + Gaussian blur)

Core logic (from the legacy mask generation script):
1. Observation point: image center (street view position)
   - Observation height = height_map[center_x, center_y] (dynamically obtained from the height map)
2. Target points: every pixel in the image
   - The height of the target point itself is considered
3. Height map preprocessing: Gaussian blur (sigma=1.0, consistent with the legacy mask generation script)
   - Reduces noise and smooths edges
4. Uses the Bresenham algorithm to generate discrete pixel paths
5. Checks occlusion along the path:
   expected_height = observer_h + (target_h - observer_h) * (i / distance)
   if height_map[x, y] > expected_height + threshold:
       return i / distance  # returns the fraction of the path where the first occlusion occurs
6. Visibility = fraction of the path length at the first occlusion

Output: visibility value [0, 1], continuous floating-point precision
    1.0 = fully visible (no occlusion)
    0.x = partially visible (first occlusion at x% of the path)
    0.0 = fully invisible (occluded at the start)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


def create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel

    Args:
        kernel_size: kernel size (odd number)
        sigma: Gaussian standard deviation

    Returns:
        Gaussian kernel [kernel_size, kernel_size]
    """
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


class GroundVisibilityAnalyzer(nn.Module):
    """
    Ground observation point visibility analyzer (CUDA vectorization + Bresenham ray tracing + Gaussian blur + occlusion buffer)

    Key features:
    - Observation height dynamically obtained from the height map (consistent with the legacy mask generation script)
    - Considers the height of the target point itself
    - Uses the Bresenham algorithm to generate discrete pixel paths
    - GPU batch parallel computation of all target points
    - Gaussian blur preprocessing (sigma=1)
    - Occlusion buffer: applies exponential decay to occluded pixels, preserving occluder information

    Occlusion buffer mechanism (scheme 1: local buffer based on occlusion position):
    - Identifies the first occlusion point of each ray
    - Computes the distance from the target pixel to the occlusion point
    - Applies exponential decay: visibility = threshold * exp(-distance / buffer_size)
    - Effect: the boundaries of occluders (buildings/trees) remain visible, showing what object blocks the line of sight
    """

    def __init__(
        self,
        sat_resolution: Tuple[int, int] = (224, 224),
        max_height_meters: float = 200.0,
        camera_height_meters: float = 1.6,  # camera height above ground
        device: str = "cuda",
        threshold: float = 0.5,  # occlusion threshold
        gaussian_sigma: float = 1.0,  # Gaussian blur standard deviation (consistent with the legacy mask generation script)
        apply_gaussian: bool = True,  # whether to apply Gaussian blur
        buffer_size_pixels: int = 40,  # buffer size (pixels, 1 pixel = 1 m)
        use_occlusion_buffer: bool = True  # whether to enable the occlusion buffer
    ):
        super().__init__()
        self.sat_h, self.sat_w = sat_resolution
        self.max_height = max_height_meters
        self.camera_height = camera_height_meters
        self.device = device
        self.threshold = threshold
        self.gaussian_sigma = gaussian_sigma
        self.apply_gaussian = apply_gaussian
        self.buffer_size_pixels = buffer_size_pixels
        self.use_occlusion_buffer = use_occlusion_buffer

        # Pre-create the Gaussian kernel (kernel_size=5, sigma=1.0 corresponds to scipy.ndimage.gaussian_filter sigma=1)
        if self.apply_gaussian:
            kernel_size = 5
            kernel = create_gaussian_kernel(kernel_size, gaussian_sigma)
            # Expand to convolution kernel format [out_channels, in_channels, H, W]
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            # Register as a buffer (moves to the correct device automatically with the model)
            self.register_buffer('gaussian_kernel_buffer', kernel)

        logger.info(
            f"GroundVisibilityAnalyzer initialized: "
            f"resolution={sat_resolution}, threshold={threshold}m, "
            f"device={device}, gaussian_sigma={gaussian_sigma}, "
            f"apply_gaussian={apply_gaussian}, "
            f"buffer_size={buffer_size_pixels}px, "
            f"use_occlusion_buffer={use_occlusion_buffer}"
        )

    @torch.inference_mode()
    def forward(
        self,
        building_height_map: torch.Tensor,
        tree_height_map: torch.Tensor,
        batch_size_targets: int = 8192,  # number of target points processed per batch
    ) -> torch.Tensor:
        """
        Compute the ground observation point visibility mask (Bresenham ray tracing + occlusion buffer)

        Args:
            building_height_map: building height map [B, H, W], in meters
            tree_height_map: tree height map [B, H, W], in meters
            batch_size_targets: number of target points per batch (controls GPU memory usage)

        Returns:
            visibility_map: visibility map [B, H, W], range [0, 1]
                1.0 = fully visible (no occlusion along the path)
                0.0 < x < 1.0 = partially visible (occlusion position fraction + buffer decay)
                0.0 = fully invisible (occluded at the start)

        Notes:
        - When use_occlusion_buffer=True, exponential decay is applied to all pixels not fully visible
        - When use_occlusion_buffer=False, the raw occlusion position fraction is returned (no decay)
        """
        building_height_map = building_height_map.to(self.device)
        tree_height_map = tree_height_map.to(self.device)

        batch_size, H, W = building_height_map.shape
        device = building_height_map.device

        # 1. Merge the height maps (buildings take precedence, trees as a supplement)
        combined_height = torch.maximum(building_height_map, tree_height_map)

        # 2. Apply Gaussian blur (consistent with the legacy mask generation script)
        # original version: total_height = gaussian_filter(total_height, sigma=1)
        if self.apply_gaussian:
            combined_height = self._apply_gaussian_blur(combined_height)

        # 3. Initialize the result: fully invisible (0.0) by default
        visibility_map = torch.zeros(
            (batch_size, H, W),
            dtype=torch.float32,
            device=device
        )

        # 4. Compute the observation point position (image center)
        center_x = W // 2
        center_y = H // 2

        # 5. Generate all target point coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        # Flatten into 1D lists
        target_y = y_coords.flatten()  # [H*W]
        target_x = x_coords.flatten()  # [H*W]

        # Exclude the observation point itself
        mask_center = (target_x == center_x) & (target_y == center_y)
        target_x = target_x[~mask_center]
        target_y = target_y[~mask_center]

        num_targets = target_x.shape[0]
        logger.debug(f"Processing {num_targets} target points")

        # 6. Process target points in batches (avoids GPU memory overflow)
        for batch_start in range(0, num_targets, batch_size_targets):
            batch_end = min(batch_start + batch_size_targets, num_targets)

            # Target points of the current batch
            batch_target_x = target_x[batch_start:batch_end]  # [N]
            batch_target_y = target_y[batch_start:batch_end]  # [N]
            num_batch_targets = batch_end - batch_start

            # 7. Compute the parameters of each ray (vectorized Bresenham)
            dx = (batch_target_x - center_x).abs()  # [N]
            dy = (batch_target_y - center_y).abs()  # [N]
            sx = torch.where(batch_target_x >= center_x, 1, -1)  # [N]
            sy = torch.where(batch_target_y >= center_y, 1, -1)  # [N]
            err = dx - dy  # [N]

            # Compute the maximum length of each ray
            ray_lengths = dx.maximum(dy) + 1  # [N]
            max_ray_length = ray_lengths.max().item()

            # 8. Precompute the path of each ray (batch processing)
            # initialize path coordinates [N, max_length]
            path_x = torch.zeros((num_batch_targets, max_ray_length),
                                dtype=torch.long, device=device)
            path_y = torch.zeros((num_batch_targets, max_ray_length),
                                dtype=torch.long, device=device)

            # CUDA-accelerated Bresenham algorithm
            self._batch_bresenham(
                path_x, path_y,
                center_x, center_y,
                batch_target_x, batch_target_y,
                dx, dy, sx, sy, err,
                ray_lengths, max_ray_length
            )

            # 9. Process each batch sample
            for b in range(batch_size):
                # Get the height map of the current sample (DSM: ground elevation + buildings/trees)
                height_map = combined_height[b]  # [H, W]

                # Get the observation height = ground height at the center + camera height above ground
                # note: height_map should be a DSM (digital surface model) containing ground elevation
                ground_height_at_center = height_map[center_y, center_x]
                observer_height = ground_height_at_center + self.camera_height

                # Get the target point heights (considering target heights, consistent with the legacy mask generation script)
                target_height = height_map[batch_target_y, batch_target_x]  # [N]

                # Get obstacle heights along the path [N, max_length]
                # boundary protection: clamp to the valid range
                path_x_clamped = path_x.clamp(0, W - 1)
                path_y_clamped = path_y.clamp(0, H - 1)
                path_obstacle_heights = height_map[path_y_clamped, path_x_clamped]

                # 10. Compute the expected ray heights (linear interpolation)
                # expected_height = observer_h + (target_h - observer_h) * (i / distance)
                progress = torch.arange(max_ray_length, device=device).float() / \
                          ray_lengths.unsqueeze(1).clamp(min=1)  # [N, max_length]

                expected_heights = observer_height + \
                                  (target_height.unsqueeze(1) - observer_height) * progress  # [N, max_length]

                # 11. Occlusion detection (consistent with the legacy mask generation script)
                # if height_map[x, y] > expected_height + threshold:
                #     return i / distance
                occluded = path_obstacle_heights > (expected_heights + self.threshold)  # [N, max_length]

                # 12. Create the valid mask (exclude out-of-boundary points)
                valid_mask = (
                    (path_x >= 0) & (path_x < W) &
                    (path_y >= 0) & (path_y < H)
                )  # [N, max_length]

                # 13. Find the first occlusion position (record index and coordinates)
                # mark invalid positions as occluded to avoid being selected
                occluded_with_invalid = occluded | ~valid_mask

                # Find the first True along the path dimension
                # append an all-True column to ensure at least one True
                occluded_padded = torch.cat([
                    occluded_with_invalid,
                    torch.ones((num_batch_targets, 1), dtype=torch.bool, device=device)
                ], dim=1)  # [N, max_length+1]

                first_occlusion_idx = occluded_padded.long().argmax(dim=1)  # [N]

                # 14. Compute the visibility score (first occlusion position / path length)
                # consistent with the legacy mask generation script: return i / distance
                ray_visibility = first_occlusion_idx.float() / ray_lengths.clamp(min=1)  # [N]

                # 15. Apply the occlusion buffer (scheme 1: local buffer based on occlusion position)
                if self.use_occlusion_buffer:
                    # Get the coordinates of the first occlusion point
                    # first_occlusion_idx: [N], max_ray_length is a scalar
                    valid_occlusion_mask = first_occlusion_idx < max_ray_length  # [N]

                    # Initialize the occlusion point coordinates (defaults to the target point itself)
                    occlusion_x = batch_target_x.clone()  # [N]
                    occlusion_y = batch_target_y.clone()  # [N]

                    # For valid occlusions, get the occlusion point coordinates
                    if valid_occlusion_mask.any():
                        # Get the path indices of the occlusion points
                        occ_indices = first_occlusion_idx[valid_occlusion_mask]  # [M]
                        # Extract coordinates from the path
                        occ_x = path_x[valid_occlusion_mask, occ_indices]  # [M]
                        occ_y = path_y[valid_occlusion_mask, occ_indices]  # [M]

                        # Update the occlusion point coordinates
                        occlusion_x[valid_occlusion_mask] = occ_x
                        occlusion_y[valid_occlusion_mask] = occ_y

                    # Compute the Euclidean distance from the target point to the occlusion point
                    distance_to_occlusion = torch.sqrt(
                        (batch_target_x.float() - occlusion_x.float()) ** 2 +
                        (batch_target_y.float() - occlusion_y.float()) ** 2
                    )  # [N]

                    # Apply exponential decay: decay is applied to all occluded pixels
                    # new_visibility = original_visibility * exp(-distance / buffer_size)
                    occlusion_mask = ray_visibility < 1.0  # [N] pixels not fully visible

                    if occlusion_mask.any():
                        # Compute the buffered visibility (keep the original visibility and apply decay)
                        attenuation_factor = torch.exp(
                            -distance_to_occlusion[occlusion_mask] / self.buffer_size_pixels
                        )  # [M]

                        # Update the visibility of occluded pixels (take the larger of the original and decayed values)
                        ray_visibility[occlusion_mask] = torch.maximum(
                            ray_visibility[occlusion_mask],
                            ray_visibility[occlusion_mask] * attenuation_factor
                        )

                # 16. Write the results back to visibility_map
                flat_indices = batch_target_y * W + batch_target_x  # [N]
                visibility_map[b].flatten().index_put_(
                    [flat_indices],
                    ray_visibility,
                    accumulate=False
                )

        # 17. The observation point itself is set to visible (special case)
        visibility_map[:, center_y, center_x] = 1.0

        # 18. Ensure the result range is [0, 1]
        visibility_map = visibility_map.clamp(0.0, 1.0)

        return visibility_map

    def _apply_gaussian_blur(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to the height map (corresponds to scipy.ndimage.gaussian_filter(sigma=1))

        Args:
            height_map: height map [B, H, W]

        Returns:
            blurred height map [B, H, W]
        """
        # Expand dimensions to [B, 1, H, W] for convolution
        height_map_expanded = height_map.unsqueeze(1)  # [B, 1, H, W]

        # Get the input device and ensure the Gaussian kernel is on the same device
        input_device = height_map.device
        kernel = self.gaussian_kernel_buffer.to(input_device)

        # Apply convolution (padding preserves the size)
        kernel_size = kernel.shape[-1]
        padding = kernel_size // 2
        blurred = F.conv2d(
            height_map_expanded,
            kernel,
            padding=padding
        )

        # Remove the channel dimension
        return blurred.squeeze(1)  # [B, H, W]

    def _batch_bresenham(
        self,
        path_x: torch.Tensor,  # [N, max_length] output
        path_y: torch.Tensor,  # [N, max_length] output
        center_x: int,
        center_y: int,
        target_x: torch.Tensor,  # [N]
        target_y: torch.Tensor,  # [N]
        dx: torch.Tensor,  # [N]
        dy: torch.Tensor,  # [N]
        sx: torch.Tensor,  # [N]
        sy: torch.Tensor,  # [N]
        err: torch.Tensor,  # [N]
        ray_lengths: torch.Tensor,  # [N]
        max_ray_length: int
    ):
        """
        Batch Bresenham algorithm (CUDA accelerated)

        Generates path coordinates for N rays simultaneously
        """
        N = target_x.shape[0]

        # Initialize current positions
        curr_x = torch.full((N,), center_x, dtype=torch.long, device=self.device)
        curr_y = torch.full((N,), center_y, dtype=torch.long, device=self.device)
        curr_err = err.clone()
        curr_dx = dx.clone()
        curr_dy = dy.clone()

        # Iteratively generate the path (vectorized)
        for step in range(max_ray_length):
            # Record the current position
            valid_mask = step < ray_lengths
            path_x[valid_mask, step] = curr_x[valid_mask]
            path_y[valid_mask, step] = curr_y[valid_mask]

            # Update positions (core of the Bresenham algorithm)
            e2 = 2 * curr_err

            # x-direction step
            move_x = e2 > -curr_dy
            curr_err = torch.where(move_x, curr_err - curr_dy, curr_err)
            curr_x = torch.where(move_x, curr_x + sx, curr_x)

            # y-direction step
            move_y = e2 < curr_dx
            curr_err = torch.where(move_y, curr_err + curr_dx, curr_err)
            curr_y = torch.where(move_y, curr_y + sy, curr_y)

    @torch.inference_mode()
    def forward_with_path_visualization(
        self,
        building_height_map: torch.Tensor,
        tree_height_map: torch.Tensor,
        batch_size_targets: int = 4096,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visibility and return the occlusion heatmap (for visualization and debugging)

        Args:
            building_height_map: building height map [B, H, W]
            tree_height_map: tree height map [B, H, W]
            batch_size_targets: number of target points per batch

        Returns:
            visibility_map: visibility map [B, H, W], range [0, 1]
            occlusion_heatmap: occlusion heatmap [B, H, W], showing the number of occluded rays at each position
        """
        building_height_map = building_height_map.to(self.device)
        tree_height_map = tree_height_map.to(self.device)

        batch_size, H, W = building_height_map.shape
        device = building_height_map.device

        # 1. Merge the height maps
        combined_height = torch.maximum(building_height_map, tree_height_map)

        # 2. Apply Gaussian blur (consistent with the legacy mask generation script)
        if self.apply_gaussian:
            combined_height = self._apply_gaussian_blur(combined_height)

        visibility_map = torch.zeros((batch_size, H, W), dtype=torch.float32, device=device)
        occlusion_heatmap = torch.zeros((batch_size, H, W), dtype=torch.float32, device=device)

        center_x = W // 2
        center_y = H // 2

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        target_y = y_coords.flatten()
        target_x = x_coords.flatten()

        mask_center = (target_x == center_x) & (target_y == center_y)
        target_x = target_x[~mask_center]
        target_y = target_y[~mask_center]

        num_targets = target_x.shape[0]

        for batch_start in range(0, num_targets, batch_size_targets):
            batch_end = min(batch_start + batch_size_targets, num_targets)

            batch_target_x = target_x[batch_start:batch_end]
            batch_target_y = target_y[batch_start:batch_end]
            num_batch_targets = batch_end - batch_start

            dx = (batch_target_x - center_x).abs()
            dy = (batch_target_y - center_y).abs()
            sx = torch.where(batch_target_x >= center_x, 1, -1)
            sy = torch.where(batch_target_y >= center_y, 1, -1)
            err = dx - dy

            ray_lengths = dx.maximum(dy) + 1
            max_ray_length = ray_lengths.max().item()

            path_x = torch.zeros((num_batch_targets, max_ray_length),
                                dtype=torch.long, device=device)
            path_y = torch.zeros((num_batch_targets, max_ray_length),
                                dtype=torch.long, device=device)

            self._batch_bresenham(
                path_x, path_y,
                center_x, center_y,
                batch_target_x, batch_target_y,
                dx, dy, sx, sy, err,
                ray_lengths, max_ray_length
            )

            for b in range(batch_size):
                height_map = combined_height[b]
                observer_height = height_map[center_y, center_x]
                target_height = height_map[batch_target_y, batch_target_x]

                path_x_clamped = path_x.clamp(0, W - 1)
                path_y_clamped = path_y.clamp(0, H - 1)
                path_obstacle_heights = height_map[path_y_clamped, path_x_clamped]

                progress = torch.arange(max_ray_length, device=device).float() / \
                          ray_lengths.unsqueeze(1).clamp(min=1)
                expected_heights = observer_height + \
                                  (target_height.unsqueeze(1) - observer_height) * progress

                occluded = path_obstacle_heights > (expected_heights + self.threshold)

                valid_mask = (
                    (path_x >= 0) & (path_x < W) &
                    (path_y >= 0) & (path_y < H)
                )

                occluded_with_invalid = occluded & valid_mask

                occluded_padded = torch.cat([
                    occluded_with_invalid,
                    torch.zeros((num_batch_targets, 1), dtype=torch.bool, device=device)
                ], dim=1)

                # Simple binary visibility (for the heatmap)
                ray_occluded = occluded_with_invalid.any(dim=1)
                ray_visibility = (~ray_occluded).float()

                flat_indices = batch_target_y * W + batch_target_x
                visibility_map[b].flatten().index_put_(
                    [flat_indices],
                    ray_visibility,
                    accumulate=False
                )

                # Accumulate the occlusion heatmap
                occluded_points = occluded_with_invalid.nonzero(as_tuple=False)
                if occluded_points.shape[0] > 0:
                    occ_ray_idx = occluded_points[:, 0]
                    occ_step_idx = occluded_points[:, 1]

                    occ_x = path_x[occ_ray_idx, occ_step_idx]
                    occ_y = path_y[occ_ray_idx, occ_step_idx]

                    occ_flat_indices = occ_y * W + occ_x

                    values = torch.ones_like(occ_flat_indices, dtype=torch.float32)
                    occlusion_heatmap[b].flatten().index_add_(0, occ_flat_indices, values)

        visibility_map[:, center_y, center_x] = 1.0
        visibility_map = visibility_map.clamp(0.0, 1.0)

        return visibility_map, occlusion_heatmap


def visualize_visibility_result(
    visibility_map: torch.Tensor,
    occlusion_heatmap: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Visualize the visibility analysis results

    Args:
        visibility_map: visibility map [B, H, W] or [H, W]
        occlusion_heatmap: occlusion heatmap [B, H, W] or [H, W] (optional)
        save_path: save path (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert to numpy
    if isinstance(visibility_map, torch.Tensor):
        visibility_np = visibility_map.cpu().numpy()
    else:
        visibility_np = visibility_map

    if occlusion_heatmap is not None:
        if isinstance(occlusion_heatmap, torch.Tensor):
            heatmap_np = occlusion_heatmap.cpu().numpy()
        else:
            heatmap_np = occlusion_heatmap

    # Handle the batch dimension
    if visibility_np.ndim == 3:
        visibility_np = visibility_np[0]
    if occlusion_heatmap is not None and heatmap_np.ndim == 3:
        heatmap_np = heatmap_np[0]

    # Create the figure
    if occlusion_heatmap is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im1 = axes[0].imshow(visibility_np, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0].set_title('Visibility Map (Green=Visible, Red=Occluded)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        im2 = axes[1].imshow(heatmap_np, cmap='hot', vmin=0)
        axes[1].set_title('Occlusion Heatmap (Bright=More Occlusions)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        im = ax.imshow(visibility_np, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title('Visibility Map (Green=Visible, Red=Occluded)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")

    return fig
