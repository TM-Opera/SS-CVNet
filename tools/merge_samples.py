#!/usr/bin/env python3
"""
Merge the 6 NPY files of each sample into a single NPZ file

Features:
- Merges rgb, building_height, tree_height, visibility_mask, semantic_mask, and
  mapping_matrix into one sample.npz file
- Reduces filesystem calls: 6 → 1
- Improves cache efficiency and loading speed

Input structure:
data2/processed/patches_npy/
├── Tokyo/
│   ├── rgb/sample_001.npy
│   ├── building_height/sample_001.npy
│   ├── tree_height/sample_001.npy
│   ├── visibility_mask/sample_001.npy
│   ├── semantic_mask/sample_001.npy
│   └── mapping_matrix/sample_001.npy

Output structure:
data2/processed/patches_merged/
└── Tokyo/
    └── sample_001.npz  # contains all 6 arrays
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import multiprocessing as mp
import numpy as np


def merge_single_sample(
    sample_id: str,
    city: str,
    input_root: Path,
    output_dir: Path,
    modalities: List[str],
    mask_types: List[str]
) -> bool:
    """
    Merge all NPY files of a single sample into one NPZ file

    Args:
        sample_id: sample ID
        city: city name
        input_root: input root directory
        output_dir: output directory
        modalities: list of image modalities
        mask_types: list of mask types

    Returns:
        whether the merge succeeded
    """
    try:
        data_dict = {}
        success = True

        # Load the image data
        for modality in modalities:
            npy_path = input_root / city / modality / f"{sample_id}.npy"
            if npy_path.exists():
                data_dict[modality] = np.load(npy_path)
            else:
                # If an image is missing, create a zero array
                if modality == 'rgb':
                    data_dict[modality] = np.zeros((3, 128, 128), dtype=np.float32)
                else:
                    data_dict[modality] = np.zeros((1, 128, 128), dtype=np.float32)

        # Load the mask data (if present)
        for mask_type in mask_types:
            mask_path = input_root / city / mask_type / f"{sample_id}.npy"
            if mask_path.exists():
                data_dict[mask_type] = np.load(mask_path)
            else:
                # Mask missing; mark it as None
                data_dict[mask_type] = None

        # Save as an NPZ file
        output_file = output_dir / f"{sample_id}.npz"
        np.savez_compressed(output_file, **data_dict)

        return True

    except Exception as e:
        print(f"\nError processing {city}/{sample_id}: {e}")
        return False


def collect_samples(
    input_root: Path,
    cities: List[str],
    modalities: List[str],
    mask_types: List[str]
) -> List[Tuple[str, str]]:
    """
    Collect all samples that need merging

    Returns:
        sample list: [(sample_id, city), ...]
    """
    samples = []

    for city in cities:
        city_dir = input_root / city
        if not city_dir.exists():
            continue

        # Get the sample list from the first modality directory
        first_modality = modalities[0]
        modality_dir = city_dir / first_modality

        if not modality_dir.exists():
            continue

        # Collect all NPY files
        for npy_file in modality_dir.glob('*.npy'):
            sample_id = npy_file.stem
            samples.append((sample_id, city))

    return samples


def main():
    parser = argparse.ArgumentParser(description='Merge the 6 NPY files of each sample into one NPZ file')
    parser.add_argument(
        '--input',
        type=str,
        default='data2/processed/patches_npy',
        help='input directory (containing the separate NPY files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data2/processed/patches_merged',
        help='output directory (where the merged NPZ files are saved)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='number of parallel worker processes'
    )
    parser.add_argument(
        '--check-existing',
        action='store_true',
        help='check existing files and skip already-processed samples'
    )
    parser.add_argument(
        '--include-masks',
        action='store_true',
        default=True,
        help='whether to include mask data (default True)'
    )

    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    print("=" * 80)
    print("NPY merge tool - 6 files → 1 NPZ file")
    print("=" * 80)
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Parallel workers: {args.workers}")
    print(f"Include masks: {args.include_masks}")
    print("=" * 80)

    # Check the input directory
    if not input_root.exists():
        print(f"Error: the input directory does not exist: {input_root}")
        sys.exit(1)

    # Define the modalities and mask types
    modalities = ['rgb', 'building_height', 'tree_height']
    mask_types = ['visibility_mask', 'semantic_mask', 'mapping_matrix'] if args.include_masks else []

    # Collect all cities
    cities = sorted([d.name for d in input_root.iterdir() if d.is_dir()])
    print(f"\nFound {len(cities)} cities")

    # Collect all samples
    print("\nCollecting samples...")
    samples = collect_samples(input_root, cities, modalities, mask_types)
    total_samples = len(samples)

    if total_samples == 0:
        print("No samples found")
        return

    print(f"Found {total_samples:,} samples")

    # Create the output directories
    for city in cities:
        (output_root / city).mkdir(parents=True, exist_ok=True)

    # Filter existing files
    if args.check_existing:
        original_count = len(samples)
        samples = [(sid, city) for sid, city in samples
                   if not (output_root / city / f"{sid}.npz").exists()]
        skipped = original_count - len(samples)
        if skipped > 0:
            print(f"Skipped {skipped:,} existing files")

    if not samples:
        print("Nothing to process")
        return

    # Start merging
    print(f"\nStarting the merge (using {args.workers} processes)...")
    print("-" * 80)

    # Use a multiprocessing pool
    with mp.Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.starmap(
                merge_single_sample,
                [(sid, city, input_root, output_root / city, modalities, mask_types)
                 for sid, city in samples]
            ),
            total=len(samples),
            desc="Merge progress",
            unit="samples"
        ))

    # Summarize the results
    success_count = sum(results)
    fail_count = len(samples) - success_count

    print("\n" + "=" * 80)
    print("Merge completed!")
    print("=" * 80)
    print(f"Total: {len(samples):,} samples")
    print(f"Succeeded: {success_count:,}")
    print(f"Failed: {fail_count:,}")

    # Compute the output size
    if success_count > 0:
        output_size = sum(
            f.stat().st_size
            for f in output_root.rglob('*.npz')
        ) / (1024 ** 3)
        print(f"\nOutput size: {output_size:.2f} GB")

    # Per-city sample statistics
    print("\nPer-city sample statistics:")
    print("-" * 80)
    for city in sorted(cities):
        city_dir = output_root / city
        if city_dir.exists():
            npz_count = len(list(city_dir.glob('*.npz')))
            print(f"  {city}: {npz_count:,} samples")

    # Show the contents of one example file
    print("\nExample of the merged file structure:")
    print("-" * 80)
    sample_file = None
    for city in cities[:1]:
        city_dir = output_root / city
        if city_dir.exists():
            sample_files = list(city_dir.glob('*.npz'))
            if sample_files:
                sample_file = sample_files[0]
                break

    if sample_file:
        print(f"\nExample file: {sample_file}")
        data = np.load(sample_file)
        print(f"Contained arrays:")
        for key in data.files:
            arr = data[key]
            if arr is None:
                print(f"  - {key}: None")
            else:
                print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}, size={arr.nbytes/1024:.1f}KB")
        data.close()

    print("\nUsage:")
    print("-" * 80)
    print("# Load the merged data")
    print("data = np.load('data2/processed/patches_merged/Tokyo/sample_001.npz')")
    print("rgb = data['rgb']              # RGB image (3, 128, 128)")
    print("visibility = data['visibility_mask']  # visibility mask (128, 128)")
    print("# All 6 arrays are in a single file!")
    print("")
    print("Performance gains:")
    print("  - Filesystem calls: 6 → 1 (83% fewer)")
    print("  - Expected speedup: additional 1.5-2x")


if __name__ == '__main__':
    main()
