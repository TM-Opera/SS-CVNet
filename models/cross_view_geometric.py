import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional

from models.voxel_grid import VoxelGridBuilder
from models.ray_casting import CrossViewRayCasting
from models.visibility import GroundVisibilityAnalyzer

logger = logging.getLogger(__name__)


class CrossViewGeometricModule(nn.Module):
    """
    Cross-view geometric correspondence module (Module 1)

    Integrates three submodules:
    1. 3D voxel grid construction (VoxelGridBuilder)
    2. Cross-view ray casting (CrossViewRayCasting)
    3. Ground visibility analysis (GroundVisibilityAnalyzer)

    Outputs:
    - M_geo: mapping matrix [B, H_pano, W_pano, 2], floating-point coordinates
    - V_mask_sat: top-down visibility mask [B, H, W]
    - pano_semantic: panoramic semantic label map [B, H_pano, W_pano]
    """

    def __init__(
        self,
        sat_resolution: Tuple[int, int] = (224, 224),
        pano_resolution: Tuple[int, int] = (1024, 512),
        voxel_height_layers: int = 64,
        max_height_meters: float = 150.0,
        camera_height_meters: float = 1.6,
        scene_coverage_meters: float = 256.0,
        device: str = "cuda",
        sampling_mode: str = "logarithmic",
        log_sampling_min_distance: float = 1.0,
        log_sampling_max_distance: float = 100.0,
        log_sampling_num_steps: int = 50,
        visibility_threshold: float = 0.5,
        visibility_gaussian_sigma: float = 1.0,
        visibility_apply_gaussian: bool = True
    ):
        """
        Args:
            sat_resolution: satellite image resolution (H, W)
            pano_resolution: panorama resolution (H, W)
            voxel_height_layers: number of voxel height layers
            max_height_meters: maximum height (meters)
            camera_height_meters: camera height (meters)
            scene_coverage_meters: scene coverage range (meters)
            device: compute device
            sampling_mode: ray sampling mode ("logarithmic", "uniform", "hybrid")
            log_sampling_min_distance: minimum distance for logarithmic sampling (meters)
            log_sampling_max_distance: maximum distance for logarithmic sampling (meters)
            log_sampling_num_steps: number of logarithmic sampling steps
            visibility_threshold: visibility threshold (meters)
            visibility_gaussian_sigma: Gaussian blur standard deviation
            visibility_apply_gaussian: whether to apply Gaussian blur
        """
        super().__init__()

        self.sat_h, self.sat_w = sat_resolution
        self.pano_h, self.pano_w = pano_resolution
        self.device = device

        # Submodule 1: 3D voxel grid builder
        self.voxel_builder = VoxelGridBuilder(
            sat_resolution=sat_resolution,
            voxel_height_layers=voxel_height_layers,
            max_height_meters=max_height_meters,
            device=device
        )

        # Submodule 2: cross-view ray caster
        self.ray_caster = CrossViewRayCasting(
            sat_resolution=sat_resolution,
            pano_resolution=pano_resolution,
            voxel_height_layers=voxel_height_layers,
            max_height_meters=max_height_meters,
            camera_height_meters=camera_height_meters,
            scene_coverage_meters=scene_coverage_meters,
            device=device,
            sampling_mode=sampling_mode,
            log_sampling_min_distance=log_sampling_min_distance,
            log_sampling_max_distance=log_sampling_max_distance,
            log_sampling_num_steps=log_sampling_num_steps
        )

        # Submodule 3: ground visibility analyzer
        self.visibility_analyzer = GroundVisibilityAnalyzer(
            sat_resolution=sat_resolution,
            max_height_meters=max_height_meters,
            device=device,
            threshold=visibility_threshold,
            gaussian_sigma=visibility_gaussian_sigma,
            apply_gaussian=visibility_apply_gaussian
        )

        logger.info(f"CrossViewGeometricModule initialized:")
        logger.info(f"  Satellite resolution: {sat_resolution}")
        logger.info(f"  Panorama resolution: {pano_resolution}")
        logger.info(f"  Voxel layers: {voxel_height_layers}")
        logger.info(f"  Sampling mode: {sampling_mode}")
        logger.info(f"  Device: {device}")

    def forward(
        self,
        building_height_map: torch.Tensor,
        tree_height_map: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            building_height_map: building height map [B, H, W], in meters
            tree_height_map: tree height map [B, H, W], in meters

        Returns:
            dict: {
                'M_geo': mapping matrix [B, H_pano, W_pano, 2], floating-point coordinates
                'V_mask_sat': top-down visibility mask [B, H, W], range [0, 1]
                'pano_semantic': panoramic semantic label map [B, H_pano, W_pano]
            }
        """
        batch_size = building_height_map.shape[0]
        device = building_height_map.device

        logger.debug(f"CrossViewGeometricModule forward: batch_size={batch_size}")

        # Step 1: build the 3D voxel grid
        logger.debug("  Step 1: building 3D voxel grid...")
        voxel_grid = self.voxel_builder(building_height_map, tree_height_map)
        # voxel_grid: [B, H, W, D]

        # Step 2: ray casting to generate the mapping matrix and panoramic semantic map
        logger.debug("  Step 2: ray casting...")
        ray_casting_results = self.ray_caster(
            voxel_grid,
            building_height_map=building_height_map,
            tree_height_map=tree_height_map,
            return_mapping_matrix=True
        )

        pano_semantic = ray_casting_results['pano_semantic']
        M_geo = ray_casting_results['M_geo']
        # pano_semantic: [B, H_pano, W_pano]
        # M_geo: [B, H_pano, W_pano, 2]

        # Step 3: compute the ground visibility mask
        logger.debug("  Step 3: computing visibility mask...")
        V_mask_sat = self.visibility_analyzer(building_height_map, tree_height_map)
        # V_mask_sat: [B, H, W], range [0, 1]

        logger.debug("  CrossViewGeometricModule forward completed")

        return {
            'M_geo': M_geo,
            'V_mask_sat': V_mask_sat,
            'pano_semantic': pano_semantic
        }

    def get_output_info(self) -> Dict:
        """
        Get output information

        Returns:
            dict: shape and dtype information of output tensors
        """
        return {
            'M_geo': {
                'shape': (None, self.pano_h, self.pano_w, 2),
                'dtype': 'torch.float32',
                'range': '[-1, -1] denotes invalid coordinates (sky)',
                'description': 'floating-point coordinate mapping from panorama pixels to the satellite image'
            },
            'V_mask_sat': {
                'shape': (None, self.sat_h, self.sat_w),
                'dtype': 'torch.float32',
                'range': '[0, 1]',
                'description': 'satellite image visibility mask, 1.0 means fully visible'
            },
            'pano_semantic': {
                'shape': (None, self.pano_h, self.pano_w),
                'dtype': 'torch.uint8',
                'range': '[0, 1, 2, 3]',
                'description': 'panoramic semantic labels: 0=sky, 1=ground, 2=building, 3=tree'
            }
        }


def create_cross_view_geometric_module(
    config: Optional[Dict] = None,
    device: str = "cuda"
) -> CrossViewGeometricModule:
    """
    Factory function: create the cross-view geometric correspondence module

    Args:
        config: configuration dict (optional)
        device: compute device

    Returns:
        CrossViewGeometricModule instance
    """
    if config is None:
        # Default configuration
        config = {
            'sat_resolution': (224, 224),
            'pano_resolution': (256, 512),
            'voxel_height_layers': 96,
            'max_height_meters': 200.0,
            'camera_height_meters': 1.6,
            'scene_coverage_meters': 224.0,
            'sampling_mode': 'logarithmic',
            'log_sampling_num_steps': 50,
            'visibility_threshold': 0.5,
            'visibility_gaussian_sigma': 1.0,
            'visibility_apply_gaussian': True
        }

    module = CrossViewGeometricModule(
        device=device,
        **config
    )

    return module
