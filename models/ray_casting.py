import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict
import logging
import math

logger = logging.getLogger(__name__)


class CrossViewRayCasting(nn.Module):
    """
    Fully vectorized cross-view ray casting

    Optimization: eliminates Python loops and fully exploits GPU parallelism
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
        debug_mode: bool = False,
        sampling_mode: str = "logarithmic",  # "uniform", "logarithmic", or "hybrid"
        log_sampling_min_distance: float = 1.0,  # minimum sampling distance (meters)
        log_sampling_max_distance: float = 100.0,  # maximum sampling distance (meters)
        log_sampling_num_steps: int = 50,  # number of logarithmic sampling points
        hybrid_transition_distance: float = 30.0,  # hybrid sampling transition distance (meters)
        hybrid_near_steps: int = 35,  # number of near-field logarithmic sampling points
        hybrid_far_step_size: float = 2.0  # far-field uniform sampling step size (meters)
    ):
        super().__init__()
        self.sat_h, self.sat_w = sat_resolution
        self.pano_h, self.pano_w = pano_resolution
        self.voxel_layers = voxel_height_layers
        self.max_height = max_height_meters
        self.camera_height = camera_height_meters
        self.scene_coverage = scene_coverage_meters
        self.device = device
        self.debug_mode = debug_mode

        # Sampling mode configuration
        self.sampling_mode = sampling_mode
        self.log_r_min = log_sampling_min_distance
        self.log_r_max = log_sampling_max_distance
        self.log_num_steps = log_sampling_num_steps

        # Hybrid sampling parameters
        self.hybrid_transition = hybrid_transition_distance
        self.hybrid_near_steps = hybrid_near_steps
        self.hybrid_far_step = hybrid_far_step_size

        # Compute the physical voxel size
        self.voxel_size = self.scene_coverage / self.sat_w

        # Precompute angle grids (avoids repeated computation)
        self._precompute_angles()

        # Semantic labels
        self.label_sky = 0
        self.label_ground = 1
        self.label_building = 2
        self.label_tree = 3

        logger.info(
            f"CrossViewRayCasting (Vectorized) initialized: "
            f"{pano_resolution}, sampling_mode={sampling_mode}"
        )

    def _precompute_angles(self):
        """Precompute the elevation and azimuth angles of each panorama pixel (formulas 4-5)"""
        y_pano = torch.arange(self.pano_h, device=self.device).view(-1, 1)
        theta = math.pi / 2 - (y_pano * math.pi / self.pano_h)

        x_pano = torch.arange(self.pano_w, device=self.device).view(1, -1)
        phi = (x_pano * 2 * math.pi / self.pano_w) - math.pi

        self.register_buffer("theta_grid", theta.expand(self.pano_h, self.pano_w))
        self.register_buffer("phi_grid", phi.expand(self.pano_h, self.pano_w))

        # Precompute trigonometric values
        self.register_buffer("cos_theta", torch.cos(self.theta_grid))
        self.register_buffer("sin_theta", torch.sin(self.theta_grid))
        self.register_buffer("cos_phi", torch.cos(self.phi_grid))
        self.register_buffer("sin_phi", torch.sin(self.phi_grid))

    def forward(
        self,
        voxel_grid: torch.Tensor,
        building_height_map: Optional[torch.Tensor] = None,
        tree_height_map: Optional[torch.Tensor] = None,
        return_mapping_matrix: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Execute fully vectorized ray casting to generate the panoramic semantic map and mapping matrix

        Vectorization strategy:
        1. Compute the coordinates and heights of all distance steps at once
        2. Check occlusion for all steps in a batch
        3. Find the first occlusion (minimum step) of each pixel
        4. Determine the semantic label according to the first occlusion type

        Args:
            voxel_grid: 3D voxel grid [B, H, W, D]
            building_height_map: building height map [B, H, W]
            tree_height_map: tree height map [B, H, W]
            return_mapping_matrix: whether to return the mapping matrix M_geo

        Returns:
            If return_mapping_matrix=False:
                pano_semantic: panoramic semantic map [B, H_pano, W_pano]
            If return_mapping_matrix=True:
                dict: {
                    'pano_semantic': panoramic semantic map [B, H_pano, W_pano],
                    'M_geo': mapping matrix [B, H_pano, W_pano, 2]
                }
        """
        batch_size = voxel_grid.shape[0]
        device = voxel_grid.device

        # Execute fully vectorized ray casting
        results = self._ray_casting_vectorized(
            voxel_grid, building_height_map, tree_height_map,
            return_mapping_matrix=return_mapping_matrix
        )

        # Return different formats according to the parameter (backward compatible)
        if return_mapping_matrix:
            return {
                'pano_semantic': results['pano_semantic'],
                'M_geo': results['M_geo']
            }
        else:
            return results['pano_semantic']

    def _ray_casting_vectorized(
        self,
        voxel_grid: torch.Tensor,
        building_height_map: Optional[torch.Tensor],
        tree_height_map: Optional[torch.Tensor],
        return_mapping_matrix: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Core logic of fully vectorized ray casting (chunked processing to reduce GPU memory)

        Key optimizations:
        1. Eliminates Python loops; all distance steps are computed in parallel
        2. Uses chunked processing to reduce GPU memory usage
        3. Batch lookup of the first occlusion ensures result consistency

        Args:
            return_mapping_matrix: whether to return the mapping matrix M_geo

        Returns:
            dict: {
                'pano_semantic': panoramic semantic map [B, H_pano, W_pano],
                'M_geo': mapping matrix [B, H_pano, W_pano, 2] (if return_mapping_matrix=True)
            }
        """
        batch_size = voxel_grid.shape[0]
        device = voxel_grid.device

        # Compute the distance sequence according to the sampling mode
        if self.sampling_mode == "logarithmic":
            # Logarithmic sampling mode
            num_steps = self.log_num_steps
            max_R = self.log_r_max
            # Precompute the logarithmic distance sequence: r_i = r_min * (r_max/r_min)^(i/(N-1))
            ratio = torch.arange(num_steps, device=device).float() / (num_steps - 1)
            R_all = self.log_r_min * torch.pow(
                self.log_r_max / self.log_r_min,
                ratio
            )  # [num_steps]

            if self.debug_mode:
                logger.info(
                    f"Logarithmic sampling mode: r_min={self.log_r_min}m, "
                    f"r_max={self.log_r_max}m, num_steps={num_steps}"
                )

        elif self.sampling_mode == "hybrid":
            # Hybrid sampling mode: near-field logarithmic + far-field uniform
            # Near-field logarithmic sampling (0 - transition_distance)
            near_ratio = torch.arange(
                self.hybrid_near_steps, device=device
            ).float() / (self.hybrid_near_steps - 1)
            R_near = self.log_r_min * torch.pow(
                self.hybrid_transition / self.log_r_min,
                near_ratio
            )  # [hybrid_near_steps]

            # Far-field uniform sampling (transition_distance - max_distance)
            max_R = self.log_r_max
            far_start = self.hybrid_transition + self.hybrid_far_step
            num_far_steps = int((max_R - far_start) / self.hybrid_far_step)
            R_far_start = far_start
            R_far = torch.arange(
                num_far_steps, device=device
            ).float() * self.hybrid_far_step + R_far_start  # [num_far_steps]

            # Merge the near field and far field
            R_all = torch.cat([R_near, R_far])  # [hybrid_near_steps + num_far_steps]
            num_steps = len(R_all)

            if self.debug_mode:
                logger.info(
                    f"Hybrid sampling mode: "
                    f"near-field logarithmic (0-{self.hybrid_transition}m, {self.hybrid_near_steps} points), "
                    f"far-field uniform ({self.hybrid_transition}-{max_R:.0f}m, {num_far_steps} points), "
                    f"{num_steps} points in total"
                )

        else:
            # Uniform sampling mode (original implementation)
            max_R = math.sqrt(2) * self.scene_coverage
            step_R = self.voxel_size
            num_steps = int(max_R / step_R)
            # Uniform distance sequence: r_i = (i+1) * step_R
            R_all = (torch.arange(num_steps, device=device).float() + 1) * step_R  # [num_steps]

            if self.debug_mode:
                logger.info(
                    f"Uniform sampling mode: max_R={max_R:.1f}m, "
                    f"step_R={step_R:.2f}m, num_steps={num_steps}"
                )

        # Chunked processing (reduces GPU memory usage)
        chunk_size = 10

        # Center coordinates
        center_x = self.sat_w / 2.0
        center_y = self.sat_h / 2.0

        # Get the precomputed trigonometric values
        cos_theta = self.cos_theta.to(device)  # [pano_h, pano_w]
        sin_theta = self.sin_theta.to(device)
        cos_phi = self.cos_phi.to(device)
        sin_phi = self.sin_phi.to(device)

        if self.debug_mode:
            logger.info(f"Vectorized ray casting: num_steps={num_steps}, chunk_size={chunk_size}, max_R={max_R:.1f}m")

        # ========== Initialize first-occlusion records ==========
        # first_occlusion_step: [B, pano_h, pano_w]
        # initialized to num_steps (meaning no occlusion found yet)
        first_occlusion_step = torch.full(
            (batch_size, self.pano_h, self.pano_w),
            num_steps,
            dtype=torch.long,
            device=device
        )

        # first_occlusion_type: [B, pano_h, pano_w]
        # initialized to 0 (no occlusion)
        first_occlusion_type = torch.zeros(
            (batch_size, self.pano_h, self.pano_w),
            dtype=torch.uint8,
            device=device
        )

        # first_occlusion_coord: [B, pano_h, pano_w, 2] - top-down coordinates of the first occlusion
        # initialized to -1 denoting invalid coordinates (only used when the mapping matrix is needed)
        if return_mapping_matrix:
            first_occlusion_coord = torch.full(
                (batch_size, self.pano_h, self.pano_w, 2),
                -1.0,
                dtype=torch.float32,
                device=device
            ).contiguous()  # ensure contiguous memory
        else:
            first_occlusion_coord = None

        # ========== Process all distance steps in chunks ==========
        for chunk_start in range(0, num_steps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_steps)
            current_chunk_size = chunk_end - chunk_start

            # ========== Compute the coordinates of the current chunk ==========
            # R_chunk: [chunk_size] - sliced from the precomputed distance sequence
            R_chunk = R_all[chunk_start:chunk_end]  # [chunk_size]

            # R_chunk_expanded: [chunk_size, 1, 1]
            R_chunk_expanded = R_chunk.view(-1, 1, 1)

            # x_sate_chunk, y_sate_chunk: [chunk_size, pano_h, pano_w]
            x_sate_chunk = center_x + R_chunk_expanded * cos_theta * cos_phi
            y_sate_chunk = center_y - R_chunk_expanded * cos_theta * sin_phi

            # Compute the ray height
            # ray_height_chunk: [chunk_size, pano_h, pano_w]
            ray_height_chunk = R_chunk_expanded * sin_theta / (cos_theta + 1e-8) + self.camera_height

            # ========== Expand to the batch dimension ==========
            # [chunk_size, pano_h, pano_w] -> [B, chunk_size, pano_h, pano_w]
            x_sate_chunk = x_sate_chunk.unsqueeze(0).expand(batch_size, -1, -1, -1)
            y_sate_chunk = y_sate_chunk.unsqueeze(0).expand(batch_size, -1, -1, -1)
            ray_height_chunk = ray_height_chunk.unsqueeze(0).expand(batch_size, -1, -1, -1)

            # ========== Round coordinates ==========
            x_idx_chunk = torch.clamp(x_sate_chunk.long(), 0, self.sat_w - 1)
            y_idx_chunk = torch.clamp(y_sate_chunk.long(), 0, self.sat_h - 1)

            # ========== Create the valid region mask ==========
            valid_mask_chunk = (
                (x_sate_chunk >= 0) & (x_sate_chunk < self.sat_w) &
                (y_sate_chunk >= 0) & (y_sate_chunk < self.sat_h)
            )

            # ========== Check occlusion for the current chunk ==========
            occlusion_type_chunk = self._check_occlusion_all_steps(
                x_idx_chunk, y_idx_chunk, ray_height_chunk,
                voxel_grid, building_height_map, tree_height_map,
                valid_mask_chunk
            )  # [B, chunk_size, pano_h, pano_w]

            # ========== Update the first-occlusion records ==========
            # only update pixels where no occlusion has been found yet
            not_found_yet = (first_occlusion_step == num_steps)  # [B, pano_h, pano_w]

            # For each step in the current chunk
            for i in range(current_chunk_size):
                step_idx = chunk_start + i

                # Occlusion type at the current step
                occlusion_at_step = occlusion_type_chunk[:, i, :, :]  # [B, pano_h, pano_w]

                # Top-down coordinates at the current step (floating-point precision)
                x_at_step = x_sate_chunk[:, i, :, :]  # [B, pano_h, pano_w]
                y_at_step = y_sate_chunk[:, i, :, :]  # [B, pano_h, pano_w]

                # Find pixels first occluded at the current step
                # condition: (not found before) and (occluded at the current step)
                first_found_now = not_found_yet & (occlusion_at_step > 0)

                # Update the first-occlusion records
                first_occlusion_step[first_found_now] = step_idx
                first_occlusion_type[first_found_now] = occlusion_at_step[first_found_now]

                # If the mapping matrix is needed, record the coordinates of the first occlusion
                if return_mapping_matrix and first_found_now.any():
                    # Use explicit indices instead of boolean masks
                    indices = torch.where(first_found_now)  # returns (batch_idx, height_idx, width_idx)
                    if len(indices[0]) > 0:
                        # Use explicit indices
                        first_occlusion_coord[indices[0], indices[1], indices[2], 0] = x_at_step[indices[0], indices[1], indices[2]]
                        first_occlusion_coord[indices[0], indices[1], indices[2], 1] = y_at_step[indices[0], indices[1], indices[2]]

                # Update the not-found mask
                not_found_yet = not_found_yet & (occlusion_at_step == 0)

        # ========== Determine semantic labels from the first occlusion ==========
        pano_semantic = self._occlusion_type_to_semantic(first_occlusion_type)

        if self.debug_mode:
            logger.info("Vectorized ray casting completed")

        # ========== Return results ==========
        if return_mapping_matrix:
            return {
                'pano_semantic': pano_semantic,
                'M_geo': first_occlusion_coord
            }
        else:
            return {
                'pano_semantic': pano_semantic
            }

    def _check_occlusion_all_steps(
        self,
        x_idx_all: torch.Tensor,
        y_idx_all: torch.Tensor,
        ray_height_all: torch.Tensor,
        voxel_grid: torch.Tensor,
        building_height_map: Optional[torch.Tensor],
        tree_height_map: Optional[torch.Tensor],
        valid_mask_all: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch check occlusion for all distance steps

        Args:
            x_idx_all: [B, num_steps, pano_h, pano_w]
            y_idx_all: [B, num_steps, pano_h, pano_w]
            ray_height_all: [B, num_steps, pano_h, pano_w]
            valid_mask_all: [B, num_steps, pano_h, pano_w]

        Returns:
            occlusion_type_all: [B, num_steps, pano_h, pano_w]
            0=no occlusion, 1=building, 2=tree, 3=ground
        """
        batch_size, num_steps, pano_h, pano_w = x_idx_all.shape
        device = x_idx_all.device

        # Initialize the occlusion type tensor
        occlusion_type_all = torch.zeros(
            (batch_size, num_steps, pano_h, pano_w),
            dtype=torch.uint8,
            device=device
        )

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1).expand(-1, num_steps, pano_h, pano_w)

        # Check building and tree occlusion
        if building_height_map is not None and tree_height_map is not None:
            # Get building and tree heights
            bld_height = building_height_map[batch_indices, y_idx_all, x_idx_all]
            tree_h = tree_height_map[batch_indices, y_idx_all, x_idx_all]

            # Merge heights (buildings take precedence)
            combined_height = torch.maximum(bld_height, tree_h)

            # Determine whether an object is hit (ray height within the object height range)
            hit_object = (
                valid_mask_all &
                (ray_height_all >= 0) &
                (ray_height_all <= combined_height) &
                (combined_height > 0)
            )

            # Determine building or tree (buildings take precedence)
            is_building = (bld_height >= tree_h) & hit_object
            is_tree = (tree_h > bld_height) & hit_object

            # Mark the occlusion types
            occlusion_type_all[is_building] = 1  # building
            occlusion_type_all[is_tree] = 2  # tree

        # Check ground occlusion (only where there is no object)
        hit_ground = valid_mask_all & (ray_height_all < 0) & (occlusion_type_all == 0)
        occlusion_type_all[hit_ground] = 3  # ground

        return occlusion_type_all

    def _find_first_occlusion(
        self,
        occlusion_type_all: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """
        Find the first occlusion step of each pixel

        Args:
            occlusion_type_all: [B, num_steps, pano_h, pano_w]
            num_steps: total number of steps

        Returns:
            first_occlusion_step: [B, pano_h, pano_w]
            returns num_steps for pixels without occlusion
        """
        # has_occlusion: [B, pano_h, pano_w]
        has_occlusion = (occlusion_type_all > 0).any(dim=1)

        # Find the first non-zero position (first occlusion)
        # argmax returns the index of the first maximum value, i.e. the first True for a bool tensor
        first_occlusion_step = occlusion_type_all.float().argmax(dim=1)  # [B, pano_h, pano_w]

        # Fix pixels without occlusion (argmax returns 0 for an all-zero tensor; should be num_steps)
        first_occlusion_step = torch.where(
            has_occlusion,
            first_occlusion_step,
            torch.tensor(num_steps, dtype=torch.long, device=occlusion_type_all.device)
        )

        return first_occlusion_step

    def _occlusion_type_to_semantic(
        self,
        occlusion_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Map occlusion types to semantic labels

        Args:
            occlusion_type: [B, pano_h, pano_w]
            0=no occlusion, 1=building, 2=tree, 3=ground

        Returns:
            pano_semantic: [B, pano_h, pano_w]
            0=sky, 1=ground, 2=building, 3=tree
        """
        batch_size, pano_h, pano_w = occlusion_type.shape
        device = occlusion_type.device

        # Initialize as sky
        pano_semantic = torch.full(
            (batch_size, pano_h, pano_w),
            self.label_sky,
            dtype=torch.uint8,
            device=device
        )

        # Map occlusion types to semantic labels
        # 1 (building) -> 2 (building)
        pano_semantic[occlusion_type == 1] = self.label_building

        # 2 (tree) -> 3 (tree)
        pano_semantic[occlusion_type == 2] = self.label_tree

        # 3 (ground) -> 1 (ground)
        pano_semantic[occlusion_type == 3] = self.label_ground

        # 0 (no occlusion) -> 0 (sky) [already the default]

        return pano_semantic
