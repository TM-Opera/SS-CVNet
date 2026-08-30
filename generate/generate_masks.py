"""
Optimized mask generation script - uses data preloading and parallel I/O

Main optimizations:
1. Preloads all TIF files into memory (if memory allows)
2. Uses multiple threads to read TIF files in parallel
3. Batched resize operations (GPU-accelerated)
4. Reduces CPU-GPU transfers
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress TIFF warnings (global settings, applied outside threads)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_GSTREAMER_LOGLEVEL'] = '-8'

import argparse
import yaml
from typing import List, Tuple
import warnings
import io
import time
from contextlib import contextmanager

import numpy as np
import torch
import cv2
from tqdm import tqdm

from models.cross_view_geometric import CrossViewGeometricModule

warnings.filterwarnings('ignore')

# Thread-local storage for per-thread output suppression
_thread_local = threading.local()
_devnull_files = []  # tracks all open devnull files for cleanup

def get_thread_devnull():
    """Get the devnull file object of the current thread"""
    if not hasattr(_thread_local, 'devnull') or _thread_local.devnull is None:
        _thread_local.devnull = open(os.devnull, 'w')
        _devnull_files.append(_thread_local.devnull)
    return _thread_local.devnull

@contextmanager
def suppress_all_output_threadsafe():
    """Thread-safe output suppression"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Use the devnull of the current thread
    devnull = get_thread_devnull()

    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # note: do not close the devnull; keep it open for the thread lifetime

def cleanup_thread_resources():
    """Clean up the resources of all threads"""
    global _devnull_files
    for devnull in _devnull_files:
        try:
            devnull.close()
        except:
            pass
    _devnull_files = []

import atexit
atexit.register(cleanup_thread_resources)


def format_time(seconds: float) -> str:
    """Format a time duration for display"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


class OptimizedMaskGenerator:
    """Optimized mask generator"""

    def __init__(self, config_path: str = 'configs/SS-CVNet.yaml', device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load the config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check the config structure
        if 'model' in config and 'module1' in config['model']:
            module1_config = config['model']['module1']
        else:
            # Use the default configuration
            print("Warning: model.module1 not found in the config file; using the default configuration")
            module1_config = {
                'sat_resolution': [128, 128],
                'pano_resolution': [128, 256],
                'voxel_height_layers': 32,
                'max_height_meters': 100.0,
                'camera_height_meters': 1.5,
                'scene_coverage_meters': 100.0,
                'sampling_mode': 'linear',
                'log_sampling_num_steps': 10,
                'visibility_threshold': 0.1,
                'visibility_gaussian_sigma': 0.5,
                'visibility_apply_gaussian': False
            }

        self.sat_resolution = tuple(module1_config['sat_resolution'])
        self.pano_resolution = tuple(module1_config['pano_resolution'])

        # Create the model
        self.module = CrossViewGeometricModule(
            sat_resolution=self.sat_resolution,
            pano_resolution=self.pano_resolution,
            voxel_height_layers=module1_config['voxel_height_layers'],
            max_height_meters=module1_config['max_height_meters'],
            camera_height_meters=module1_config['camera_height_meters'],
            scene_coverage_meters=module1_config['scene_coverage_meters'],
            device=str(self.device),
            sampling_mode=module1_config.get('sampling_mode', 'linear'),
            log_sampling_num_steps=module1_config.get('log_sampling_num_steps', 10),
            visibility_threshold=module1_config.get('visibility_threshold', 0.1),
            visibility_gaussian_sigma=module1_config.get('visibility_gaussian_sigma', 0.5),
            visibility_apply_gaussian=module1_config.get('visibility_apply_gaussian', False)
        )

        self.module.eval()
        print(f"Mask generator using device: {self.device}")

    def load_height_map(self, file_path: Path) -> torch.Tensor:
        """Load a height map (TIF format)"""
        with suppress_all_output_threadsafe():
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"unable to read the image: {file_path}")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height_array = img.astype(np.float32)
        height_map = torch.from_numpy(height_array)

        return height_map

    def load_batch_parallel(self, building_paths: List[Path], tree_paths: List[Path],
                           num_workers: int = 8) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Load a batch of data in parallel"""
        building_heights = [None] * len(building_paths)
        tree_heights = [None] * len(tree_paths)

        def load_single(idx):
            building_height = self.load_height_map(building_paths[idx])
            tree_height = self.load_height_map(tree_paths[idx])
            return idx, building_height, tree_height

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(load_single, i) for i in range(len(building_paths))]
            for future in tqdm(futures, desc="  Loading data", leave=False):
                idx, building_height, tree_height = future.result()
                building_heights[idx] = building_height
                tree_heights[idx] = tree_height

        return building_heights, tree_heights

    def process_batch_optimized(self, building_paths: List[Path], tree_paths: List[Path],
                               num_workers: int = 8) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Optimized batch processing function"""
        batch_size = len(building_paths)

        # 1. Load the data in parallel
        load_start = time.time()
        building_heights, tree_heights = self.load_batch_parallel(building_paths, tree_paths, num_workers)
        load_time = time.time() - load_start

        # 2. Batched resize (on the GPU)
        resize_start = time.time()
        building_heights_resized = []
        tree_heights_resized = []

        # Move all data to the GPU first
        building_heights_gpu = []
        tree_heights_gpu = []

        for bh, th in zip(building_heights, tree_heights):
            if bh.shape != self.sat_resolution:
                # Resize on the GPU
                bh_tensor = bh.unsqueeze(0).unsqueeze(1).to(self.device)
                th_tensor = th.unsqueeze(0).unsqueeze(1).to(self.device)

                bh_resized = torch.nn.functional.interpolate(
                    bh_tensor, size=self.sat_resolution, mode='bilinear', align_corners=False
                ).squeeze().cpu()

                th_resized = torch.nn.functional.interpolate(
                    th_tensor, size=self.sat_resolution, mode='bilinear', align_corners=False
                ).squeeze().cpu()

                building_heights_resized.append(bh_resized)
                tree_heights_resized.append(th_resized)
            else:
                building_heights_resized.append(bh)
                tree_heights_resized.append(th)

        # Stack into batches
        building_heights_batch = torch.stack(building_heights_resized).to(self.device)
        tree_heights_batch = torch.stack(tree_heights_resized).to(self.device)
        resize_time = time.time() - resize_start

        # 3. GPU computation
        compute_start = time.time()
        with torch.no_grad():
            results = self.module(building_heights_batch, tree_heights_batch)
        compute_time = time.time() - compute_start

        # 4. Split the results
        v_mask_sats = []
        pano_semantics = []
        m_geos = []

        for i in range(batch_size):
            v_mask_sats.append(results['V_mask_sat'][i].cpu().numpy())
            pano_semantics.append(results['pano_semantic'][i].cpu().numpy())
            m_geos.append(results['M_geo'][i].cpu().numpy())

        total_time = load_time + resize_time + compute_time

        return v_mask_sats, pano_semantics, m_geos, {
            'load_time': load_time,
            'resize_time': resize_time,
            'compute_time': compute_time,
            'total_time': total_time
        }

    def process_city_optimized(self, feature_root: Path, city_name: str,
                              output_root: Path, skip_existing: bool = True,
                              batch_size: int = 2048, num_io_workers: int = 8,
                              save_visibility: bool = True, save_semantic: bool = True,
                              save_mapping: bool = True) -> dict:
        """Optimized per-city processing function"""
        building_dir = feature_root / city_name / 'building_height'
        tree_dir = feature_root / city_name / 'tree_height'

        # Create the output directories
        visibility_dir = output_root / city_name / 'visibility_mask'
        semantic_dir = output_root / city_name / 'semantic_mask'
        mapping_dir = output_root / city_name / 'mapping_matrix'

        if save_visibility:
            visibility_dir.mkdir(parents=True, exist_ok=True)
        if save_semantic:
            semantic_dir.mkdir(parents=True, exist_ok=True)
        if save_mapping:
            mapping_dir.mkdir(parents=True, exist_ok=True)

        # Collect all building files
        building_files = sorted(list(building_dir.glob('*.tif')) +
                               list(building_dir.glob('*.png')))

        print(f"\nProcessing city: {city_name}")
        print(f"  File count: {len(building_files)}")
        print(f"  Batch size: {batch_size}")
        print(f"  I/O threads: {num_io_workers}")

        start_time = time.time()
        stats = {
            'total': len(building_files),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'batches': 0,
            'times': {'load': 0, 'resize': 0, 'compute': 0}
        }

        # Filter the files that need processing
        files_to_process = []
        for building_path in building_files:
            region_id = building_path.stem
            tree_path = tree_dir / f"{region_id}{building_path.suffix}"

            if not tree_path.exists():
                stats['failed'] += 1
                continue

            # Check whether all required files already exist
            if skip_existing:
                all_exist = True
                if save_visibility and not visibility_dir.joinpath(f"{region_id}.npy").exists():
                    all_exist = False
                if save_semantic and not semantic_dir.joinpath(f"{region_id}.npy").exists():
                    all_exist = False
                if save_mapping and not mapping_dir.joinpath(f"{region_id}.npy").exists():
                    all_exist = False

                if all_exist:
                    stats['skipped'] += 1
                    continue

            files_to_process.append((building_path, tree_path, region_id))

        print(f"  Files to process: {len(files_to_process)}")

        if len(files_to_process) == 0:
            stats['elapsed_time'] = 0.0
            return stats

        # Batch processing
        num_batches = (len(files_to_process) + batch_size - 1) // batch_size
        print(f"  Number of batches: {num_batches}")

        for batch_idx in tqdm(range(num_batches), desc="  Generation progress"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(files_to_process))

            batch_data = files_to_process[start_idx:end_idx]
            building_paths = [item[0] for item in batch_data]
            tree_paths = [item[1] for item in batch_data]
            region_ids = [item[2] for item in batch_data]

            try:
                v_mask_sats, pano_semantics, m_geos, timing = self.process_batch_optimized(
                    building_paths, tree_paths, num_io_workers
                )

                # Save the results
                for i, region_id in enumerate(region_ids):
                    if save_visibility:
                        np.save(visibility_dir / f"{region_id}.npy", v_mask_sats[i])
                    if save_semantic:
                        np.save(semantic_dir / f"{region_id}.npy", pano_semantics[i])
                    if save_mapping:
                        np.save(mapping_dir / f"{region_id}.npy", m_geos[i])

                stats['processed'] += len(region_ids)
                stats['batches'] += 1
                stats['times']['load'] += timing['load_time']
                stats['times']['resize'] += timing['resize_time']
                stats['times']['compute'] += timing['compute_time']

            except Exception as e:
                print(f"    [Error] batch {batch_idx} failed: {e}")
                stats['failed'] += len(region_ids)
                continue

        elapsed_time = time.time() - start_time
        stats['elapsed_time'] = elapsed_time

        # Print detailed timing statistics
        if stats['batches'] > 0:
            avg_load = stats['times']['load'] / stats['batches']
            avg_resize = stats['times']['resize'] / stats['batches']
            avg_compute = stats['times']['compute'] / stats['batches']
            avg_total = (avg_load + avg_resize + avg_compute)

            print(f"  Average time per batch:")
            print(f"    Data loading: {avg_load:.2f}s ({avg_load/avg_total*100:.1f}%)")
            print(f"    Resize:       {avg_resize:.2f}s ({avg_resize/avg_total*100:.1f}%)")
            print(f"    GPU compute:  {avg_compute:.2f}s ({avg_compute/avg_total*100:.1f}%)")
            print(f"    Total:        {avg_total:.2f}s")

        print(f"  Done: processed {stats['processed']}, skipped {stats['skipped']}, failed {stats['failed']}, "
              f"elapsed {format_time(elapsed_time)}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Optimized mask generation script - parallel I/O and GPU acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str, default='configs/SS-CVNet.yaml',
                       help='config file path')
    parser.add_argument('--cities', type=str, nargs='+', default=None,
                       help='list of cities to process')
    parser.add_argument('--all-cities', action='store_true',
                       help='process all cities')
    parser.add_argument('--feature_root', type=str, default='data/processed/patches',
                       help='root directory of the feature data')
    parser.add_argument('--output_root', type=str, default='data/SVData',
                       help='output root directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                       choices=['cuda:0', 'cuda:1', 'cpu'],
                       help='compute device')
    parser.add_argument('--skip_existing', action='store_true',
                       help='skip files that already exist')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='batch size (default: 2048)')
    parser.add_argument('--io_workers', type=int, default=4,
                       help='number of parallel I/O threads (default: 4)')

    parser.add_argument('--save_visibility', action='store_true', default=True)
    parser.add_argument('--no-save_visibility', action='store_false', dest='save_visibility')
    parser.add_argument('--save_semantic', action='store_true', default=True)
    parser.add_argument('--no-save_semantic', action='store_false', dest='save_semantic')
    parser.add_argument('--save_mapping', action='store_true', default=True)
    parser.add_argument('--no-save_mapping', action='store_false', dest='save_mapping')

    args = parser.parse_args()

    # Create the generator
    generator = OptimizedMaskGenerator(config_path=args.config, device=args.device)

    # Process the city list
    if args.all_cities:
        feature_root = Path(args.feature_root)
        if not feature_root.exists():
            print(f"Error: the feature data root directory does not exist: {feature_root}")
            return
        args.cities = sorted([d.name for d in feature_root.iterdir() if d.is_dir()])
        print(f"Automatically collected {len(args.cities)} cities")

    if not args.cities:
        print("Error: please specify --cities or use --all-cities")
        return

    # Process each city
    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)

    print(f"\nStarting cities: {', '.join(args.cities)}")
    print(f"I/O threads: {args.io_workers}")
    print(f"Batch size: {args.batch_size}")

    total_start = time.time()
    all_stats = []

    for city_name in args.cities:
        stats = generator.process_city_optimized(
            feature_root=feature_root,
            city_name=city_name,
            output_root=output_root,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            num_io_workers=args.io_workers,
            save_visibility=args.save_visibility,
            save_semantic=args.save_semantic,
            save_mapping=args.save_mapping
        )
        all_stats.append(stats)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("All cities processed!")
    print(f"Total elapsed: {format_time(total_elapsed)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
