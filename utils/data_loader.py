"""
Data loading module - concise version
Supports multi-modal TIF/PNG file loading, using OpenCV to read images (better GeoTIFF support)
Uses torchvision for normalization
Supports data augmentation (during training)
Uses preprocessed images by default (skips resize+crop transforms)
Supports loading pre-generated GeoView mask data
"""

import os
import sys
import io
from contextlib import contextmanager

# Strategy 1: environment variables must be set before importing cv2
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_GSTREAMER_LOGLEVEL'] = '-8'

# Strategy 2: redirect stdout and stderr to /dev/null
@contextmanager
def suppress_all_output():
    """Fully suppress stdout and stderr output"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        devnull.close()

# Now import cv2 (environment variables are set)
import cv2

# Strategy 3: set the OpenCV internal log level
try:
    cv2.setLogLevel(0)  # 0 = LOG_LEVEL_SILENT (fully silent)
except:
    pass

# Other standard imports
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import random
import numpy as np
import warnings

# Suppress Python warnings
warnings.filterwarnings('ignore')


# ImageNet normalization parameters (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def custom_collate_fn(batch):
    """
    Custom collate function for handling batch data containing generated_masks

    Args:
        batch: list of dicts returned by the dataset __getitem__

    Returns:
        merged batch dict
    """
    # Filter out samples that failed to load (empty inputs)
    valid_batch = [item for item in batch if len(item.get('inputs', {})) > 0]

    if len(valid_batch) == 0:
        # The whole batch failed; return an empty batch (with a warning)
        print("Warning: the entire batch failed to load; returning empty data")
        return {
            'inputs': {},
            'targets': torch.zeros(0, 3),
            'metadata': []
        }

    if len(valid_batch) < len(batch):
        print(f"Warning: {len(batch) - len(valid_batch)}/{len(batch)} samples in the batch failed to load and were skipped")

    # Separate the fields
    inputs_list = [item['inputs'] for item in valid_batch]
    targets_list = [item['targets'] for item in valid_batch]
    metadata_list = [item['metadata'] for item in valid_batch]

    # Merge the inputs dict (each modality merged separately)
    batch_inputs = {}
    # Get the modalities present in all samples
    all_modalities = set()
    for inputs in inputs_list:
        all_modalities.update(inputs.keys())

    for modality in all_modalities:
        # Merge only the samples containing this modality
        modality_tensors = [item[modality] for item in inputs_list if modality in item]
        if len(modality_tensors) > 0:
            batch_inputs[modality] = torch.stack(modality_tensors)

    # Merge the targets
    batch_targets = torch.stack(targets_list)

    # Build the result dict
    result = {
        'inputs': batch_inputs,
        'targets': batch_targets,
        'metadata': metadata_list
    }

    # If generated_masks are present, merge them
    first_masks = valid_batch[0].get('generated_masks')
    if first_masks is not None and len(first_masks) > 0:
        # Check whether all valid samples have masks
        has_masks = all('generated_masks' in item and item['generated_masks'] is not None for item in valid_batch)
        if has_masks:
            batch_masks = {
                'visibility_mask': torch.stack([item['generated_masks']['visibility_mask'] for item in valid_batch]),
                'semantic_mask': torch.stack([item['generated_masks']['semantic_mask'] for item in valid_batch]),
                'mapping_matrix': torch.stack([item['generated_masks']['mapping_matrix'] for item in valid_batch])
            }
            result['generated_masks'] = batch_masks

    return result


class SynchronizedTransform:
    """Synchronized transform - ensures all modalities receive the same augmentation"""

    def __init__(
        self,
        target_size: Tuple[int, int],
        is_train: bool = True,
        use_preprocessed: bool = True,
        use_augmentation: bool = False
    ):
        """
        Args:
            target_size: target image size (height, width)
            is_train: whether this is training mode (augmentation used during training)
            use_preprocessed: whether to use preprocessed images (default True, skips resize+crop)
            use_augmentation: whether to use data augmentation (default False, no augmentation)
                              when enabled, only geometric transforms are applied (flips, rotations)
        """
        self.target_size = target_size
        self.is_train = is_train
        self.use_preprocessed = use_preprocessed
        self.use_augmentation = use_augmentation

    def __call__(self, *image_modality_pairs: Tuple[Image.Image, str]) -> List[torch.Tensor]:
        """
        Apply the same transforms to multiple images

        Args:
            *image_modality_pairs: list of (image, modality name) tuples

        Returns:
            list of transformed tensors
        """
        images = [pair[0] for pair in image_modality_pairs]
        modalities = [pair[1] for pair in image_modality_pairs]

        # If using preprocessed images, skip the resize and crop operations
        if not self.use_preprocessed:
            # Two-stage transform:
            # 1. resize all images to 500×500 first (preserve aspect ratio + black padding)
            # 2. then center-crop to the target size
            images = [self._resize(img) for img in images]
        else:
            # Images are already preprocessed; use them directly
            pass

        # Data augmentation: enabled only when use_augmentation=True and is_train=True
        # only geometric transforms are used (flips, rotations)
        if self.use_augmentation and self.is_train:
            # 1. random horizontal flip - 30% probability
            if random.random() > 0.7:
                images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]

            # 2. random vertical flip - 30% probability
            if random.random() > 0.7:
                images = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]

            # 3. random rotation (0, 90, 180, 270 degrees) - 20% probability
            if random.random() > 0.8:
                angle = random.choice([0, 90, 180, 270])
                if angle != 0:
                    images = [img.rotate(angle, expand=False) for img in images]

        # Convert to tensors
        results = []
        for img, modality in zip(images, modalities):
            # Convert to a tensor
            tensor = transforms.ToTensor()(img)

            # Apply normalization only to RGB images (rsi/rgb)
            # single-channel height maps (building_height/tree_height) keep their raw values, no normalization
            if modality in ['rsi', 'rgb']:
                tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
            # building_height and tree_height: not normalized, raw height values are kept

            results.append(tensor)

        return results

    def _resize(self, img: Image.Image) -> Image.Image:
        """
        Resize directly to the target size (aspect ratio not preserved)

        Args:
            img: PIL image

        Returns:
            processed PIL image (original channel count preserved)
        """
        # Get the target size
        crop_w, crop_h = self.target_size  # 224, 224

        # Resize directly to the target size (aspect ratio not preserved)
        img_resized = img.resize((crop_w, crop_h), Image.LANCZOS)

        return img_resized


def _get_all_cities_from_csv(csv_path: str) -> List[str]:
    """
    Get all available cities from the CSV file

    Args:
        csv_path: CSV file path

    Returns:
        city list (deduplicated, sorted)
    """
    try:
        df = pd.read_csv(csv_path)
        if 'city' in df.columns:
            cities = df['city'].unique().tolist()
            cities.sort()  # sort for consistency
            return cities
        else:
            raise ValueError(f"the CSV file has no 'city' column: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"unable to read the CSV file to get the city list: {e}")


class SVIDataset(Dataset):
    """Street view indicator dataset - concise version

    Supports multi-modal PNG/TIF inputs (building_height, rgb, tree_height), using torchvision for image normalization
    Supports synchronized data augmentation (during training)
    Uses preprocessed images by default (skips resize+crop, keeps only augmentation)
    Supports loading pre-generated GeoView mask data

    Supported modality names (mapped automatically to the actual folders):
    - 'rgb' or 'rsi' → 'rgb/' folder (RGB image, ImageNet normalization)
    - 'building_height' or 'building' → 'building_height/' folder (single-channel height map, no normalization)
    - 'tree_height' or 'tree' → 'tree_height/' folder (single-channel height map, no normalization)
    """

    def __init__(
        self,
        csv_path: str,
        feature_root: str,
        modalities: List[str] = ['rsi'],
        cities: Optional[List[str]] = None,
        perspectives: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (224, 224),
        is_train: bool = True,
        use_preprocessed: bool = True,
        use_augmentation: bool = False,
        use_generated_masks: bool = False,
        mask_root: Optional[str] = None,
        use_npz_format: bool = False
    ):
        """
        Args:
            csv_path: path to the CSV annotation file
            feature_root: root directory of the feature files
            modalities: list of modalities to use, e.g. ['rgb', 'building_height', 'tree_height']
                      supported aliases: 'rsi'→'rgb', 'building'→'building_height', 'tree'→'tree_height'
            cities: list of cities to include; None means all cities
            perspectives: list of perspectives to include; None means all perspectives
            target_size: target image size (height, width)
            is_train: whether this is training mode (augmentation used during training)
            use_preprocessed: whether to use preprocessed images (default True)
            use_augmentation: whether to use data augmentation (default False; only geometric
                transforms applied when enabled)
            use_generated_masks: whether to use pre-generated GeoView masks (default False)
            mask_root: root directory of the pre-generated masks (defaults to data2/SVData)
            use_npz_format: whether to use NPZ format data (default False; when enabled, loads
                from the merged NPZ files)
        """
        self.feature_root = Path(feature_root)
        self.modalities = modalities
        self.target_size = target_size
        self.is_train = is_train
        self.use_preprocessed = use_preprocessed
        self.use_augmentation = use_augmentation
        self.use_generated_masks = use_generated_masks
        self.use_npz_format = use_npz_format

        # Mapping from modality names to folder names
        self.modality_folder_map = {
            'rgb': 'rgb',
            'rsi': 'rgb',  # rsi also maps to the rgb folder
            'building_height': 'building_height',
            'building': 'building_height',
            'tree_height': 'tree_height',
            'tree': 'tree_height'
        }

        # Set the mask root directory
        if mask_root is None:
            # use data2/SVData as the mask root directory by default
            self.mask_root = Path('data2/SVData')
        else:
            self.mask_root = Path(mask_root)

        # NPZ-format related settings
        if self.use_npz_format:
            # with the NPZ format, feature_root points directly to the patches_merged directory
            # the mask data is already contained in the NPZ files; no extra mask_root is needed
            print(f"  [NPZ format] NPZ loading enabled; all data is in a single file")
            # the NPZ format is already preprocessed; skip the transforms
            self.use_preprocessed = True

        # Smart modality filtering: automatically optimizes the loaded modalities based on the
        # configuration and architecture type
        # when the config only requires the rsi or rgb modality, the building_height and
        # tree_height maps are not needed
        original_modalities = modalities.copy()
        should_filter_height_maps = False
        filter_reason = ""

        if use_generated_masks:
            filter_reason = "pre-generated masks in use"
        elif len(modalities) == 1 and modalities[0] in ['rsi', 'rgb']:
            should_filter_height_maps = True
            filter_reason = "the config uses only the RGB modality"

        if should_filter_height_maps:
            # Filter out the height map modalities (building_height, tree_height), keeping only rsi/rgb and others
            self.modalities = [m for m in modalities if m not in ['building', 'tree', 'building_height', 'tree_height']]
            if len(self.modalities) < len(original_modalities):
                removed = set(original_modalities) - set(self.modalities)
                print(f"  [Smart modality filtering] {filter_reason}; skipping height map modalities: {', '.join(removed)}")
                print(f"  [Smart modality filtering] actually loaded modalities: {', '.join(self.modalities)}")
        else:
            self.modalities = modalities

        # Load the CSV data
        self.data = pd.read_csv(csv_path)

        # Filter cities and perspectives (supports case-insensitive 'all'/'All'/'ALL')
        # normalize the cities argument: convert a string to a list; 'all' (case-insensitive) becomes None
        if cities is not None:
            # compatible with both string and list formats
            if isinstance(cities, str):
                cities = [cities]
            # check for 'all' (case-insensitive)
            if len(cities) == 1 and cities[0].lower() == 'all':
                cities = None  # use all cities
            # check for 'none' (YAML parses None as the string 'None')
            elif len(cities) == 1 and cities[0].lower() == 'none':
                cities = None  # use all cities
            elif len(cities) == 1 and cities[0] == []:
                cities = None  # YAML empty list
            elif len(cities) == 1 and cities[0] == ['None']:
                cities = None  # YAML None as a string
            else:
                # apply the city filter
                self.data = self.data[self.data['city'].isin(cities)]
        if perspectives is not None and 'perspective' in self.data.columns:
            # compatible with both string and list formats
            if isinstance(perspectives, str):
                perspectives = [perspectives]
            self.data = self.data[self.data['perspective'].isin(perspectives)]

        self.data = self.data.reset_index(drop=True)

        # Create the synchronized transform
        self.sync_transform = SynchronizedTransform(
            target_size, is_train, use_preprocessed,
            use_augmentation
        )

        print(f"Dataset loaded")
        print(f"  Modalities: {modalities}")
        print(f"  Cities: {cities if cities else 'all'}")
        print(f"  Perspectives: {perspectives if perspectives else 'all'}")
        if use_augmentation and is_train:
            print(f"  Data augmentation: enabled (geometric transforms)")
        else:
            print(f"  Data augmentation: disabled")
        print(f"  Use preprocessed images: {'yes' if use_preprocessed else 'no'}")
        print(f"  Use NPZ format: {'yes' if use_npz_format else 'no'}")
        if use_generated_masks:
            print(f"  Use pre-generated masks: yes (root directory: {self.mask_root})")
        else:
            print(f"  Use pre-generated masks: no")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        sample_id = row.get('id', row.get('region_id', f'sample_{idx}'))  # compatible with different column names
        city = row['city']

        # NPZ format: load all data directly from the merged NPZ file
        if self.use_npz_format:
            npz_path = self.feature_root / city / f"{sample_id}.npz"

            try:
                # Load all data at once
                data_dict = np.load(npz_path)

                # Extract the image modalities
                inputs = {}
                for modality in self.modalities:
                    if modality in data_dict:
                        tensor = torch.from_numpy(data_dict[modality]).float()
                        inputs[modality] = tensor
                    else:
                        # If a modality is missing, skip it (the NPZ format needs no zero-tensor padding)
                        pass

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
                if self.use_generated_masks:
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
                print(f"Error loading NPZ file {npz_path}: {e}")
                import traceback
                traceback.print_exc()
                # Return empty data (size unspecified)
                return {
                    'inputs': {},  # empty dict; skip this sample
                    'targets': torch.zeros(3, dtype=torch.float32),
                    'metadata': {'id': sample_id, 'city': city},
                    'error': str(e)
                }

        # Legacy format: load and process from image files
        # collect all existing images (with modality info)
        images_to_transform = []  # stores (image, modality) tuples
        modality_indices = {}  # records the index of each modality in the images list

        for modality in self.modalities:
            # Get the actual folder name (supports modality name mapping)
            folder_name = self.modality_folder_map.get(modality, modality)

            # Detect the file extension automatically (supports .tif and .png)
            file_path = None
            for ext in ['.tif', '.png']:
                test_path = self.feature_root / city / folder_name / f"{sample_id}{ext}"
                if test_path.exists():
                    file_path = test_path
                    break

            if file_path is not None:
                # Load the image with OpenCV (better support for GeoTIFF and various TIFF formats)
                # fully suppress TIFF warning output (stdout and stderr)
                with suppress_all_output():
                    img_np = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

                # OpenCV reads in BGR format; convert to RGB
                if img_np is None:
                    print(f"Warning: unable to load image {file_path}")
                    continue

                if img_np.ndim == 3 and img_np.shape[2] == 3:
                    # BGR -> RGB
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                elif img_np.ndim == 2:
                    # 2D grayscale (H, W); expand to 3 channels
                    # first convert to 3D (H, W, 1), then replicate to 3 channels
                    img_np = np.stack([img_np] * 3, axis=-1)

                # Convert the numpy array to a PIL Image (compatible with the existing transform pipeline)
                if modality in ['rsi', 'rgb']:
                    # RGB image: ensure uint8 format
                    if img_np.dtype != np.uint8:
                        # normalize to 0-255 and convert to uint8
                        img_min, img_max = img_np.min(), img_np.max()
                        if img_max > img_min:
                            img_np = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            img_np = np.zeros_like(img_np, dtype=np.uint8)
                    img = Image.fromarray(img_np, mode='RGB')
                else:
                    # building_height and tree_height: single-channel height maps
                    # need conversion to a single-channel PIL Image
                    if img_np.ndim == 3:
                        # if 3 channels (possibly expanded above), take the first channel
                        img_np = img_np[:, :, 0]

                    # keep the raw height values; convert to uint8 for the PIL Image
                    # assume the height value range is reasonable (0-255 or larger)
                    if img_np.dtype == np.uint16:
                        img = Image.fromarray(img_np.astype(np.uint16), mode='I;16')
                    elif img_np.dtype in [np.float32, np.float64]:
                        # floating-point height map, assume values are in a reasonable range
                        # keep the relative information; normalize to 0-255
                        img_min, img_max = img_np.min(), img_np.max()
                        if img_max > img_min:
                            img_normalized = ((img_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            img_normalized = np.zeros_like(img_np, dtype=np.uint8)
                        img = Image.fromarray(img_normalized, mode='L')
                    else:
                        # uint8 or other
                        img = Image.fromarray(img_np.astype(np.uint8), mode='L')

                images_to_transform.append((img, modality))
                modality_indices[modality] = len(images_to_transform) - 1

        # Apply the synchronized transform (now receiving (image, modality) tuples)
        if images_to_transform:
            transformed_tensors = self.sync_transform(*images_to_transform)

            # Assign the results back to the corresponding modalities
            inputs = {}
            for modality, idx in modality_indices.items():
                inputs[modality] = transformed_tensors[idx]
        else:
            # All modalities missing; use zero tensors
            inputs = {}
            for modality in self.modalities:
                if modality in ['rsi', 'rgb']:
                    # RGB zero tensor (normalized)
                    zeros = torch.zeros(3, *self.target_size)
                    for i in range(3):
                        zeros[i] = (0 - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
                    inputs[modality] = zeros
                else:
                    # Single-channel zero tensor (not normalized; use 0 directly)
                    zeros = torch.zeros(1, *self.target_size)
                    inputs[modality] = zeros

        # Get the targets
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
                'perspective': row.get('perspective', 'panorama')  # compatible with CSVs without a perspective column
            }
        }

        # If pre-generated masks are used, load them
        if self.use_generated_masks:
            masks = self._load_generated_masks(sample_id, city)
            if masks is not None:
                result['generated_masks'] = masks

        return result

    def _load_generated_masks(self, sample_id: str, city: str) -> Optional[Dict]:
        """
        Load the pre-generated GeoView mask data

        Args:
            sample_id: sample ID
            city: city name

        Returns:
            dict containing the three masks, or None if loading fails
        """
        try:
            # Build the file paths
            visibility_path = self.mask_root / city / 'visibility_mask' / f"{sample_id}.npy"
            semantic_path = self.mask_root / city / 'semantic_mask' / f"{sample_id}.npy"
            mapping_path = self.mask_root / city / 'mapping_matrix' / f"{sample_id}.npy"

            # Check whether the files exist
            if not (visibility_path.exists() and semantic_path.exists() and mapping_path.exists()):
                return None

            # Load the npy files
            v_mask = np.load(visibility_path)
            s_mask = np.load(semantic_path)
            m_geo = np.load(mapping_path)

            # Convert to torch tensors
            v_mask_tensor = torch.from_numpy(v_mask).float()  # [H, W]
            s_mask_tensor = torch.from_numpy(s_mask).long()   # [H_pano, W_pano]
            m_geo_tensor = torch.from_numpy(m_geo).float()   # [H_pano, W_pano, 2]

            return {
                'visibility_mask': v_mask_tensor,
                'semantic_mask': s_mask_tensor,
                'mapping_matrix': m_geo_tensor
            }

        except Exception as e:
            # Return None on load failure, falling back to on-the-fly computation
            print(f"Warning: failed to load the pre-generated mask {sample_id}: {str(e)}")
            return None


def create_dataloaders(
    csv_path: str,
    feature_root: str,
    modalities: List[str],
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: Optional[float] = None,
    test_split: Optional[float] = None,
    cities: Optional[List[str]] = None,
    val_cities: Optional[List[str]] = None,
    perspectives: Optional[List[str]] = None,
    target_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    seed: int = 42,
    use_preprocessed: bool = True,
    use_augmentation: bool = False,
    use_generated_masks: bool = False,
    mask_root: Optional[str] = None,
    use_npz_format: bool = False
):
    """Create the training, validation, and test data loaders

    Args:
        csv_path: CSV file path
        feature_root: root directory of the feature files
        modalities: modality list
        batch_size: batch size
        train_split: training set ratio (0-1)
        val_split: validation set ratio (0-1); if None, (1-train_split) is used
        test_split: test set ratio (0-1); if 0 or None, no test set is created
        cities: city list
        val_cities: validation city list (city-split mode)
        perspectives: perspective list
        target_size: target image size
        shuffle: whether to shuffle
        seed: random seed
        use_preprocessed: whether to use preprocessed images
        use_augmentation: whether to use data augmentation (default False; only geometric
            transforms applied when enabled)
        use_generated_masks: whether to use pre-generated GeoView masks (default False)
        mask_root: root directory of the pre-generated masks (defaults to data2/SVData)
        use_npz_format: whether to use NPZ format data (default False; when enabled, loads
            from the merged NPZ files)

    Returns:
        if test_split is 0 or None: train_loader, val_loader
        if test_split > 0: train_loader, val_loader, test_loader
    """

    # =====================================================================
    # Mode selection: split by city vs split by ratio
    # =====================================================================

    # Check whether city-split mode is used
    if val_cities is not None and len(val_cities) > 0:
        # Mode 2: split by city
        print("=" * 80)
        print("Data split mode: city-level split")
        print("=" * 80)

        # Validate val_cities
        if cities is not None and cities != 'all':
            # If a city subset is specified, check whether val_cities is within the subset
            invalid_cities = [c for c in val_cities if c not in cities]
            if invalid_cities:
                raise ValueError(
                    f"val_cities contains invalid cities: {invalid_cities}\n"
                    f"val_cities must be a subset of cities.\n"
                    f"cities = {cities}\n"
                    f"val_cities = {val_cities}"
                )
            train_cities = [c for c in cities if c not in val_cities]
        else:
            # cities = 'all' or None; get all available cities
            all_cities = _get_all_cities_from_csv(csv_path)
            invalid_cities = [c for c in val_cities if c not in all_cities]
            if invalid_cities:
                raise ValueError(
                    f"val_cities contains invalid cities: {invalid_cities}\n"
                    f"available cities: {all_cities}"
                )
            train_cities = [c for c in all_cities if c not in val_cities]

        print(f"Training cities ({len(train_cities)}): {train_cities[:5]}{'...' if len(train_cities) > 5 else ''}")
        print(f"Validation cities ({len(val_cities)}): {val_cities}")
        print(f"Data split: strict city-level split, no data leakage")
        print("=" * 80)

        # Create the training dataset
        train_dataset = SVIDataset(
            csv_path=csv_path,
            feature_root=feature_root,
            modalities=modalities,
            cities=train_cities,
            perspectives=perspectives,
            target_size=target_size,
            is_train=True,
            use_preprocessed=use_preprocessed,
            use_augmentation=use_augmentation,
            use_generated_masks=use_generated_masks,
            mask_root=mask_root,
            use_npz_format=use_npz_format
        )

        # Create the validation dataset
        val_dataset = SVIDataset(
            csv_path=csv_path,
            feature_root=feature_root,
            modalities=modalities,
            cities=val_cities,
            perspectives=perspectives,
            target_size=target_size,
            is_train=False,  # the validation set does not use augmentation
            use_preprocessed=use_preprocessed,
            use_augmentation=False,
            use_generated_masks=use_generated_masks,
            mask_root=mask_root,
            use_npz_format=use_npz_format
        )

        # Create the DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # the validation set is not shuffled
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        print(f"Training set samples: {len(train_dataset)}")
        print(f"Validation set samples: {len(val_dataset)}")
        print("=" * 80 + "\n")

        # Handle the test set
        if test_split is not None and test_split > 0:
            # If a test set is needed, create an empty one
            # or create it from the training cities (optional)
            print(f"Warning: the city-split mode does not support a test set yet; test_split={test_split} is ignored")
            return train_loader, val_loader
        else:
            return train_loader, val_loader

    else:
        # Mode 1: random ratio split (original logic)
        print("=" * 80)
        print("Data split mode: random split")
        print("=" * 80)
        print(f"train_split={train_split}, val_split={val_split if val_split else (1-train_split)}")
        print(f"all cities will be mixed and randomly split by ratio")
        print("=" * 80 + "\n")

        # Generate random indices (using the full dataset)
        import numpy as np
        full_dataset = SVIDataset(
            csv_path=csv_path,
            feature_root=feature_root,
            modalities=modalities,
            cities=cities,
            perspectives=perspectives,
            target_size=target_size,
            is_train=True,  # temporarily True to get all data
            use_preprocessed=use_preprocessed,
            use_augmentation=use_augmentation,
            use_generated_masks=use_generated_masks,
            mask_root=mask_root,
            use_npz_format=use_npz_format
        )

        total_size = len(full_dataset)
        indices = list(range(total_size))

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        # Compute the split points
        # compatible with two calling conventions:
        # 1. only train_split given: the val ratio is (1-train_split), no test set
        # 2. train_split, val_split, test_split given: split by the three ratios
        if val_split is None or test_split is None:
            # legacy logic: only train_split given
            train_end = int(train_split * total_size)
            val_start = train_end
            val_end = total_size
            test_start = total_size
            use_test_split = False
        else:
            # new logic: all three ratios given
            train_end = int(train_split * total_size)
            val_start = train_end
            val_end = val_start + int(val_split * total_size)
            test_start = val_end
            use_test_split = test_split > 0

    # Create independent datasets
    train_dataset = SVIDataset(
        csv_path=csv_path,
        feature_root=feature_root,
        modalities=modalities,
        cities=cities,
        perspectives=perspectives,
        target_size=target_size,
        is_train=True,
        use_preprocessed=use_preprocessed,
        use_augmentation=use_augmentation,
        use_generated_masks=use_generated_masks,
        mask_root=mask_root,
        use_npz_format=use_npz_format
    )
    val_dataset = SVIDataset(
        csv_path=csv_path,
        feature_root=feature_root,
        modalities=modalities,
        cities=cities,
        perspectives=perspectives,
        target_size=target_size,
        is_train=False,
        use_preprocessed=use_preprocessed,
        use_augmentation=False,
        use_generated_masks=use_generated_masks,
        mask_root=mask_root,
        use_npz_format=use_npz_format
    )

    # Split using Subset
    train_subset = Subset(train_dataset, indices[:train_end])
    val_subset = Subset(val_dataset, indices[val_start:val_end])

    # Create the data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,             # fewer workers to reduce resource contention
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,         # lower prefetch factor to reduce memory usage
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,             # fewer workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,         # lower prefetch factor
        collate_fn=custom_collate_fn
    )

    # Print the split information
    if use_test_split:
        test_dataset = SVIDataset(
            csv_path=csv_path,
            feature_root=feature_root,
            modalities=modalities,
            cities=cities,
            perspectives=perspectives,
            target_size=target_size,
            is_train=False,
            use_preprocessed=use_preprocessed,
            use_augmentation=False,
            use_generated_masks=use_generated_masks,
            mask_root=mask_root,
            use_npz_format=use_npz_format
        )
        test_subset = Subset(test_dataset, indices[test_start:])

        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=custom_collate_fn
        )

        print(f"Data split: train {len(train_subset)} | val {len(val_subset)} | test {len(test_subset)}")
        return train_loader, val_loader, test_loader

    else:
        print(f"Data split: train {len(train_subset)} | val {len(val_subset)}")
        return train_loader, val_loader
    return train_loader, val_loader
