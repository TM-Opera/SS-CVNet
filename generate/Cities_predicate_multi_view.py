"""
SS-CVNet multi-view batch inference - data-sharing optimized version (based on the
single-view validated logic)

Features:
1. Preloads the models of all requested perspectives at startup (resident in GPU memory)
2. Each city's data is loaded only once; the same batches pass through all models serially
3. Results are stored in per-perspective directories: output_dir/{perspective}/{city}_infer.csv
4. Supports resumable execution; completed perspectives are skipped automatically
5. Fully reuses the preprocessing and normalization logic of the single-view version

Usage:
    # Generate all 3 perspectives (recommended)
    python Cities_predicate_multi_view.py --config configs/SS-CVNet.yaml --perspective all

    # Generate specific perspectives only
    python Cities_predicate_multi_view.py --config configs/SS-CVNet.yaml --perspective multi_view fisheye

    # Run in the background
    nohup python Cities_predicate_multi_view.py --config configs/SS-CVNet.yaml \
        --perspective pano fisheye > infer_logs/multi_view_optimized.log 2>&1 &

    nohup python Cities_predicate_multi_view.py --config configs/SS-CVNet.yaml \
            --perspective multi_view > infer_logs/multi_view_optimized.log 2>&1 &
"""

import argparse
import csv
import gc
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.ss_cvnet import create_sscvnet


# =============================================================================
# Dataset class (fully reuses the preprocessing logic of the single-view version)
# =============================================================================

class SharedInferenceDataset(Dataset):
    """
    Multi-view shared inference dataset

    The preprocessing logic is identical to InferenceDataset, ensuring train/inference alignment
    """

    def __init__(
        self,
        sv_data_root: str,
        patches_root: str,
        cities: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (128, 128)
    ):
        self.sv_data_root = Path(sv_data_root)
        self.patches_root = Path(patches_root)
        self.target_size = target_size
        self.samples = self._scan_samples(cities)

        print(f"  [Dataset] cities: {len(set(s['city'] for s in self.samples))} | "
              f"samples: {len(self.samples)}")

    def _scan_samples(self, cities: Optional[List[str]]) -> List[Dict]:
        samples = []
        if cities is None:
            city_dirs = sorted([d for d in self.sv_data_root.iterdir() if d.is_dir()])
        else:
            city_dirs = [self.sv_data_root / c for c in cities
                         if (self.sv_data_root / c).exists()]

        for city_dir in city_dirs:
            city_name = city_dir.name
            required = ['visibility_mask', 'semantic_mask', 'mapping_matrix']
            if not all((city_dir / d).exists() for d in required):
                continue

            patch_city = self.patches_root / city_name
            rgb_dir = patch_city / 'rgb'
            bh_dir = patch_city / 'building_height'
            th_dir = patch_city / 'tree_height'

            if not all([rgb_dir.exists(), bh_dir.exists(), th_dir.exists()]):
                continue

            sample_ids = sorted([f.stem for f in (city_dir / 'visibility_mask').glob('*.npy')])
            for sid in sample_ids:
                samples.append({
                    'city': city_name, 'sample_id': sid,
                    'rgb_path': str(rgb_dir / f'{sid}.tif'),
                    'bh_path': str(bh_dir / f'{sid}.tif'),
                    'th_path': str(th_dir / f'{sid}.tif'),
                    'vis_path': str(city_dir / 'visibility_mask' / f'{sid}.npy'),
                    'sem_path': str(city_dir / 'semantic_mask' / f'{sid}.npy'),
                    'map_path': str(city_dir / 'mapping_matrix' / f'{sid}.npy'),
                })
        return samples

    def __len__(self): return len(self.samples)

    def _load_tiff_as_tensor(self, tiff_path: str) -> Optional[torch.Tensor]:
        try:
            with rasterio.open(tiff_path) as src:
                data = src.read()  # ★ no dtype conversion yet

            if data.size == 0: return None

            original_dtype = data.dtype  # ★ record the original dtype
            data = data.astype(np.float32)

            if original_dtype == np.uint8:
                data = data / 255.0
            elif original_dtype == np.uint16:
                data = data / 65535.0
            else:
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            return torch.from_numpy(data)
        except Exception:
            return None

    def _resize_tensor(self, tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Fully reuses the resize logic of the single-view version"""
        import torch.nn.functional as F
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        C, H, W = tensor.shape
        if (H, W) == target_size:
            return tensor
        tensor = tensor.unsqueeze(0)
        resized = F.interpolate(
            tensor, size=target_size, mode='bilinear', align_corners=False
        )
        return resized.squeeze(0)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        info = self.samples[idx]

        # 1. Load the images (reusing the original logic)
        rgb = self._load_tiff_as_tensor(info['rgb_path'])
        bh = self._load_tiff_as_tensor(info['bh_path'])
        th = self._load_tiff_as_tensor(info['th_path'])
        if any(x is None for x in [rgb, bh, th]): return None

        # 2. Resize (reusing the original logic)
        rgb = self._resize_tensor(rgb, self.target_size)
        bh = self._resize_tensor(bh, self.target_size)
        th = self._resize_tensor(th, self.target_size)

        # 3. RGB normalization (reusing the original ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = (rgb - mean) / std

        # 4. Height map processing (reusing the original NaN handling + max normalization)
        bh = torch.nan_to_num(bh, nan=0.0, posinf=0.0, neginf=0.0)
        th = torch.nan_to_num(th, nan=0.0, posinf=0.0, neginf=0.0)
        if bh.max() > 1.0: bh = bh / bh.max()
        if th.max() > 1.0: th = th / th.max()

        # 5. Load the masks (original sizes kept; handled uniformly by collate_fn)
        vis = torch.from_numpy(np.load(info['vis_path'])).float()
        sem = torch.from_numpy(np.load(info['sem_path'])).long()
        m_geo = torch.from_numpy(np.load(info['map_path'])).float()

        return {
            'rgb': rgb, 'building_height': bh, 'tree_height': th,
            'generated_masks': {
                'visibility_mask': vis,
                'semantic_mask': sem,
                'mapping_matrix': m_geo
            },
            'metadata': {'id': info['sample_id'], 'city': info['city']}
        }


def collate_fn(batch: List[Dict]) -> Optional[Dict]:
    """
    Custom collate function that filters out invalid samples containing None

    Args:
        batch: list of samples

    Returns:
        collated data dict, or None if the whole batch is empty
    """
    # Filter out None samples (samples whose TIFF files failed to load)
    valid_batch = []
    skipped_samples = []
    for item in batch:
        # item itself is None, meaning the sample failed to load
        if item is None:
            continue
        # or any field of the sample is None
        if (item.get('rgb') is None or
            item.get('building_height') is None or
            item.get('tree_height') is None):
            skipped_samples.append(item['metadata']['id'])
        else:
            valid_batch.append(item)

    # If all samples are invalid, return None
    if len(valid_batch) == 0:
        return None

    # Print a warning if samples were skipped
    if len(skipped_samples) > 0:
        print(f"  Batch warning: skipped {len(skipped_samples)} corrupted samples: {skipped_samples[:5]}{'...' if len(skipped_samples) > 5 else ''}")

    # Extract the fields
    rgb = torch.stack([item['rgb'] for item in valid_batch])
    building_height = torch.stack([item['building_height'] for item in valid_batch])
    tree_height = torch.stack([item['tree_height'] for item in valid_batch])

    # Handle the generated masks (sizes may differ; use lists)
    visibility_masks = [item['generated_masks']['visibility_mask'] for item in valid_batch]
    semantic_masks = [item['generated_masks']['semantic_mask'] for item in valid_batch]
    mapping_matrices = [item['generated_masks']['mapping_matrix'] for item in valid_batch]

    # Try stacking; fall back to padding on failure
    try:
        visibility_mask = torch.stack(visibility_masks)
        semantic_mask = torch.stack(semantic_masks)
        mapping_matrix = torch.stack(mapping_matrices)
    except RuntimeError:
        # Inconsistent sizes; padding is required
        max_h = max([m.shape[0] for m in visibility_masks])
        max_w = max([m.shape[1] for m in visibility_masks])

        padded_visibility = []
        padded_semantic = []
        padded_mapping = []

        for i in range(len(valid_batch)):
            h, w = visibility_masks[i].shape

            # Pad the visibility mask
            vis_padded = torch.zeros(max_h, max_w)
            vis_padded[:h, :w] = visibility_masks[i]
            padded_visibility.append(vis_padded)

            # Pad the semantic mask
            sem_padded = torch.zeros(max_h, max_w, dtype=torch.long)
            sem_padded[:h, :w] = semantic_masks[i]
            padded_semantic.append(sem_padded)

            # Pad the mapping matrix
            map_padded = torch.zeros(max_h, max_w, 2)
            map_padded[:h, :w, :] = mapping_matrices[i]
            padded_mapping.append(map_padded)

        visibility_mask = torch.stack(padded_visibility)
        semantic_mask = torch.stack(padded_semantic)
        mapping_matrix = torch.stack(padded_mapping)

    metadata = [item['metadata'] for item in valid_batch]

    return {
        'inputs': {
            'rgb': rgb,
            'building_height': building_height,
            'tree_height': tree_height,
            'generated_masks': {
                'visibility_mask': visibility_mask,
                'semantic_mask': semantic_mask,
                'mapping_matrix': mapping_matrix
            }
        },
        'metadata': metadata
    }



# =============================================================================
# Main function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SS-CVNet multi-view shared-data inference')
    parser.add_argument('--config', type=str, default='configs/SS-CVNet.yaml')
    parser.add_argument('--perspective', type=str, nargs='+', default=['pano', 'fisheye'],
                        choices=['multi_view', 'fisheye', 'pano', 'all'])
    parser.add_argument('--sv_data_root', type=str, default='/workspace/SVData3')
    parser.add_argument('--patches_root', type=str, default='/workspace/patch2')
    parser.add_argument('--output_dir', type=str, default='/workspace2/data/inference/predictions')
    parser.add_argument('--cities', type=str, nargs='+', default=None)
    parser.add_argument('--batch_size', type=int, default=382)
    parser.add_argument('--num_workers', type=int, default=10)  # ★ fewer workers to avoid process explosion
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    print("="*80)
    print("SS-CVNet multi-view shared-data inference (optimized)")
    print("="*80)

    # 1. Config and device
    config_path = Path(args.config)
    if not config_path.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    target_size = tuple(config['data'].get('target_size', [128, 128]))
    print(f"Device: {device} | target size: {target_size}")

    # 2. ★ Preload all models into GPU memory (executed only once)
    perspective_map = {
        'multi_view': 'Parameter/multi_view/sscvnet_multi_view/best_model.pth',
        'fisheye': 'Parameter/fisheye/sscvnet_fisheye/best_model.pth',
        'pano': 'Parameter/pano/sscvnet_pano/best_model.pth',
    }

    perspectives = list(perspective_map.keys()) if 'all' in args.perspective \
        else [p for p in args.perspective if p in perspective_map]

    models = {}
    for p in perspectives:
        mp = Path(perspective_map[p])
        if not mp.exists():
            print(f"  ⚠ model not found, skipping: {p}"); continue

        print(f"[{p}] loading model into GPU memory...", end=" ")
        model = create_sscvnet(config).to(device)
        sd = torch.load(mp, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        model.eval()
        models[p] = model
        print("✓")

    if not models:
        print("Error: no available models"); return

    print(f"\n★ Successfully preloaded {len(models)} models: {list(models.keys())}")
    print(f"  Subsequent inference shares the same data; no repeated IO is needed\n")

    # 3. Determine the city list
    sv_root = Path(args.sv_data_root)
    if args.cities is None:
        all_cities = sorted([d.name for d in sv_root.iterdir() if d.is_dir()])
    else:
        all_cities = [c for c in args.cities if (sv_root / c).exists()]

    if not all_cities:
        print("Error: no valid cities"); return

    print(f"To process: {len(all_cities)} cities × {len(models)} perspectives\n")

    output_root = Path(args.output_dir)
    total_start = time.time()
    summary = []

    # 4. ★ Process city by city: load the data once → serial inference through multiple models
    for i, city in enumerate(all_cities, 1):
        t0 = time.time()
        print(f"[{i}/{len(all_cities)}] {city}", end=" ")

        # Check which perspectives need processing (resumable execution)
        to_process = {}
        skip_count = 0
        for p_name in models:
            csv_path = output_root / p_name / f"{city}_infer.csv"
            if csv_path.exists():
                skip_count += 1
            else:
                to_process[p_name] = models[p_name]

        if skip_count > 0: print(f"({skip_count} perspectives skipped)", end="")
        if not to_process:
            print(" → all done"); continue
        print(f" → to process: {list(to_process.keys())}")

        try:
            # ★ Key: each city's data is loaded only once
            dataset = SharedInferenceDataset(
                sv_data_root=args.sv_data_root,
                patches_root=args.patches_root,
                cities=[city], target_size=target_size
            )
            if len(dataset) == 0:
                print("   no valid samples"); continue

            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=True, collate_fn=collate_fn
            )

            # Prepare result buffers for each perspective to process
            res_buf = {p: [] for p in to_process}
            meta_buf = {p: [] for p in to_process}

            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"  {city}", leave=False)
                for batch in pbar:
                    if batch is None: continue

                    inputs = batch['inputs']
                    for k in ['rgb', 'building_height', 'tree_height']:
                        inputs[k] = inputs[k].to(device)
                    masks = inputs['generated_masks']
                    for mk in masks: masks[mk] = masks[mk].to(device)

                    # ★ The same batch passes through all pending models serially
                    for p_name, model in to_process.items():
                        pred = model(inputs)
                        res_buf[p_name].append(pred.cpu().numpy())
                        meta_buf[p_name].extend(batch['metadata'])

            # Save each perspective's results to its own subdirectory
            for p_name in to_process:
                preds = np.vstack(res_buf[p_name])
                metas = meta_buf[p_name]

                out_dir = output_root / p_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_csv = out_dir / f"{city}_infer.csv"

                with open(out_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id', 'GVI', 'BVI', 'SVF'])
                    sorted_res = sorted(zip(metas, preds), key=lambda x: x[0]['id'])
                    for meta, pred in sorted_res:
                        writer.writerow([meta['id'],
                            f"{pred[0]:.16f}", f"{pred[2]:.16f}", f"{pred[1]:.16f}"])

                n = len(metas)
                summary.append({'city': city, 'perspective': p_name,
                               'status': 'success', 'samples': n})

            del dataset, dataloader, res_buf, meta_buf
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ failed: {e}")
            for p in to_process:
                summary.append({'city': city, 'perspective': p,
                               'status': 'failed', 'error': str(e)})

        elapsed = time.time() - t0
        print(f"  ⏱ {elapsed:.1f}s")

    # 5. Summary
    total_time = time.time() - total_start
    succ = sum(1 for r in summary if r['status']=='success')
    fail = sum(1 for r in summary if r['status']=='failed')
    total_samples = sum(r.get('samples',0) for r in summary)

    print("\n" + "="*80)
    print(f"Completed! ✓{succ} ✗{fail} | samples: {total_samples:,} | elapsed: {total_time/3600:.2f}h")
    print(f"Result directory: {output_root}/{{perspective}}/{{city}}_infer.csv")
    print("="*80)


if __name__ == '__main__':
    main()
