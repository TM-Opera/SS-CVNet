"""
Dataset class supporting the merged NPZ format

Usage:
from utils.merged_dataset import MergedDataset, create_merged_dataloaders

# Method 1: use the Dataset directly
dataset = MergedDataset(
    csv_path='data2/processed/complete_mapping_data.csv',
    merged_root='data2/processed/patches_merged',
    modalities=['rgb', 'building_height', 'tree_height'],
    use_masks=True  # this parameter is read from the config file
)

# Method 2: use create_merged_dataloaders (recommended)
train_loader, val_loader = create_merged_dataloaders(
    csv_path='data2/processed/complete_mapping_data.csv',
    merged_root='data2/processed/patches_merged',
    modalities=['rgb', 'building_height', 'tree_height'],
    batch_size=128,
    use_masks=True  # controls uniformly whether the training and validation sets load masks
)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class MergedDataset(Dataset):
    """
    Street view indicator dataset supporting the merged NPZ format

    All data of each sample (6 arrays) is stored in a single NPZ file:
    - rgb: RGB image (3, H, W)
    - building_height: building height (1, H, W)
    - tree_height: tree height (1, H, W)
    - visibility_mask: visibility mask (H, W)
    - semantic_mask: semantic mask (H, W_pano)
    - mapping_matrix: mapping matrix (H_pano, W_pano, 2)
    """

    def __init__(
        self,
        csv_path: str,
        merged_root: str,
        modalities: List[str] = ['rgb', 'building_height', 'tree_height'],
        cities: Optional[List[str]] = None,
        use_masks: bool = True
    ):
        """
        Args:
            csv_path: path to the CSV annotation file
            merged_root: root directory of the merged NPZ files
            modalities: list of image modalities to use
            cities: list of cities to include; None means all cities
            use_masks: whether to use mask data (read from the config file; applies uniformly
                to the training and validation sets)
        """
        self.merged_root = Path(merged_root)
        self.modalities = modalities
        self.use_masks = use_masks

        # Load the CSV data
        self.data = pd.read_csv(csv_path)

        # Filter cities
        if cities is not None:
            if isinstance(cities, str):
                cities = [cities]
            if len(cities) == 1 and cities[0].lower() == 'all':
                cities = None
            elif len(cities) == 1 and cities[0].lower() == 'none':
                cities = None
            elif len(cities) == 1 and cities[0] == []:
                cities = None
            elif len(cities) == 1 and cities[0] == ['None']:
                cities = None
            else:
                self.data = self.data[self.data['city'].isin(cities)]

        self.data = self.data.reset_index(drop=True)

        print(f"Merged dataset loaded")
        print(f"  Root directory: {merged_root}")
        print(f"  Modalities: {modalities}")
        print(f"  Cities: {cities if cities else 'all'}")
        print(f"  Use masks: {'yes' if use_masks else 'no'}")
        print(f"  Sample count: {len(self.data):,}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        sample_id = row.get('id', row.get('region_id', f'sample_{idx}'))
        city = row['city']

        # Load the merged NPZ file
        npz_path = self.merged_root / city / f"{sample_id}.npz"

        try:
            # Load all data at once
            data_dict = np.load(npz_path)

            # Extract the image modalities (use the raw data directly, no resizing)
            inputs = {}
            for modality in self.modalities:
                if modality in data_dict:
                    tensor = torch.from_numpy(data_dict[modality]).float()
                    inputs[modality] = tensor
                else:
                    # If a modality is missing, skip it and log a warning
                    print(f"Warning: modality {modality} does not exist in {city}/{sample_id}")

            # Extract the targets
            targets = torch.tensor([
                row['vegetation'],
                row['sky'],
                row['building']
            ], dtype=torch.float32)

            # Build the result dict
            result = {
                'inputs': inputs,
                'targets': targets,
                'metadata': {
                    'id': sample_id,
                    'city': city,
                    'perspective': row.get('perspective', 'panorama')
                }
            }

            # Extract the mask data (if present and needed)
            if self.use_masks:
                masks = {}
                has_all_masks = True

                for mask_name in ['visibility_mask', 'semantic_mask', 'mapping_matrix']:
                    if mask_name in data_dict and data_dict[mask_name] is not None:
                        mask_tensor = torch.from_numpy(data_dict[mask_name])
                        if mask_name == 'semantic_mask':
                            mask_tensor = mask_tensor.long()  # the semantic mask uses the long type
                        else:
                            mask_tensor = mask_tensor.float()
                        masks[mask_name] = mask_tensor
                    else:
                        has_all_masks = False
                        break

                if has_all_masks and len(masks) == 3:
                    result['generated_masks'] = masks

            data_dict.close()

            return result

        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty data (size unspecified, since the original size is unknown)
            return {
                'inputs': {},
                'targets': torch.zeros(3, dtype=torch.float32),
                'metadata': {'id': sample_id, 'city': city},
                'error': str(e)
            }


def create_merged_dataloaders(
    csv_path: str,
    merged_root: str,
    modalities: List[str],
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: Optional[float] = None,
    cities: Optional[List[str]] = None,
    shuffle: bool = True,
    seed: int = 42,
    use_masks: bool = True
):
    """
    Create data loaders using the merged NPZ format

    Args:
        csv_path: CSV file path
        merged_root: root directory of the merged NPZ files
        modalities: modality list
        batch_size: batch size
        train_split: training set ratio
        val_split: validation set ratio
        cities: city list
        shuffle: whether to shuffle
        seed: random seed
        use_masks: whether to use mask data (read from the config file; applied uniformly to
            the training and validation sets)

    Returns:
        train_loader, val_loader

    Notes:
        the use_masks parameter should be read from the config file; the training and
        validation sets use the same setting
        example: use_masks=config['data']['use_generated_masks']
    """
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    # Create the full dataset
    full_dataset = MergedDataset(
        csv_path=csv_path,
        merged_root=merged_root,
        modalities=modalities,
        cities=None,  # load all cities first
        use_masks=use_masks
    )

    total_size = len(full_dataset)
    indices = list(range(total_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    # Compute the split points
    if val_split is None:
        val_split = 1 - train_split

    train_end = int(train_split * total_size)
    val_start = train_end

    # Create independent datasets (the training and validation sets share the same mask configuration)
    train_dataset = MergedDataset(
        csv_path=csv_path,
        merged_root=merged_root,
        modalities=modalities,
        cities=cities,
        use_masks=use_masks  # read from the config file
    )
    val_dataset = MergedDataset(
        csv_path=csv_path,
        merged_root=merged_root,
        modalities=modalities,
        cities=cities,
        use_masks=use_masks  # read from the config file
    )

    # Split using Subset
    train_subset = Subset(train_dataset, indices[:train_end])
    val_subset = Subset(val_dataset, indices[val_start:])

    # Create the data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    print(f"Data split: train {len(train_subset)} | val {len(val_subset)}")
    return train_loader, val_loader
