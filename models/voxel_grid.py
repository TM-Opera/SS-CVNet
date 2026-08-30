"""
3D voxel grid construction module (CUDA accelerated)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VoxelGridBuilder(nn.Module):
    """
    CUDA-accelerated 3D voxel grid builder

    Converts building/tree height maps into a 3D semantic voxel grid
    """

    def __init__(
        self,
        sat_resolution: Tuple[int, int] = (224, 224),
        voxel_height_layers: int = 64,
        max_height_meters: float = 150.0,
        device: str = "cuda"
    ):
        super().__init__()
        self.sat_h, self.sat_w = sat_resolution
        self.voxel_layers = voxel_height_layers
        self.max_height = max_height_meters
        self.device = device

        # Semantic label definitions
        self.register_buffer("label_sky", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("label_ground", torch.tensor(1, dtype=torch.uint8))
        self.register_buffer("label_building", torch.tensor(2, dtype=torch.uint8))
        self.register_buffer("label_tree", torch.tensor(3, dtype=torch.uint8))

        logger.info(f"VoxelGridBuilder initialized: {sat_resolution}, {voxel_height_layers} layers")

    def forward(
        self,
        building_height_map: torch.Tensor,
        tree_height_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the 3D semantic voxel grid

        Args:
            building_height_map: building height map [B, H, W], in meters
            tree_height_map: tree height map [B, H, W], in meters

        Returns:
            voxel_grid: 3D voxel grid [B, H, W, D], storing semantic labels
        """
        batch_size = building_height_map.shape[0]
        device = building_height_map.device  # Get device dynamically

        # Initialize the voxel grid as ground
        voxel_grid = torch.full(
            (batch_size, self.sat_h, self.sat_w, self.voxel_layers),
            self.label_ground.item(),
            dtype=torch.uint8,
            device=device
        )

        # Normalize heights to voxel layer counts
        building_voxels = torch.clamp(
            (building_height_map / self.max_height * self.voxel_layers).long(),
            0, self.voxel_layers - 1
        )
        tree_voxels = torch.clamp(
            (tree_height_map / self.max_height * self.voxel_layers).long(),
            0, self.voxel_layers - 1
        )

        # CUDA-accelerated voxel filling (vectorized operations)
        voxel_grid = self._fill_voxels_cuda(
            voxel_grid, building_height_map, tree_height_map,
            building_voxels, tree_voxels
        )

        logger.debug(f"Voxel grid shape: {voxel_grid.shape}")
        return voxel_grid

    def _fill_voxels_cuda(
        self,
        voxel_grid: torch.Tensor,
        building_height: torch.Tensor,
        tree_height: torch.Tensor,
        building_voxels: torch.Tensor,
        tree_voxels: torch.Tensor
    ) -> torch.Tensor:
        """
        CUDA-accelerated voxel filling

        Replaces loops with broadcasting and mask operations to fully exploit GPU parallelism
        """
        batch_size, h, w = building_height.shape
        device = voxel_grid.device  # Get device dynamically

        # Create height layer indices [0, 1, ..., D-1]
        layer_indices = torch.arange(self.voxel_layers, device=device).view(1, 1, 1, -1)

        # Building voxel mask: layer index < building height layers
        building_mask = layer_indices < building_voxels.view(batch_size, h, w, 1)

        # Tree voxel mask: layer index < tree height layers
        tree_mask = layer_indices < tree_voxels.view(batch_size, h, w, 1)

        # Fill buildings first (buildings override trees)
        voxel_grid[building_mask] = self.label_building.item()

        # Fill trees (only where there is no building)
        tree_only_mask = tree_mask & ~building_mask
        voxel_grid[tree_only_mask] = self.label_tree.item()

        return voxel_grid
