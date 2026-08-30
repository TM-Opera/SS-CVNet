"""
Training script - concise version
Uses MSE loss, computes RMSE, R², MAE metrics, recorded per label
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import create_model
from models.loss import create_geoview_loss
from utils.data_loader import create_dataloaders
from utils.logger import TrainingLogger
from utils.seed import set_seed, get_seed


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
    """Compute evaluation metrics

    Args:
        predictions: predictions [N, 3]
        targets: ground truth [N, 3]

    Returns:
        metrics dict
    """
    metrics = {}

    # Compute metrics for each label
    label_names = ['vegetation', 'sky', 'building']
    for i, label in enumerate(label_names):
        pred = predictions[:, i]
        true = targets[:, i]

        metrics[f'{label}_mse'] = mean_squared_error(true, pred)
        metrics[f'{label}_rmse'] = np.sqrt(metrics[f'{label}_mse'])
        metrics[f'{label}_mae'] = mean_absolute_error(true, pred)
        metrics[f'{label}_r2'] = r2_score(true, pred)

    # Compute overall metrics (mean)
    metrics['ALL_mse'] = np.mean([metrics[f'{label}_mse'] for label in label_names])
    metrics['ALL_rmse'] = np.mean([metrics[f'{label}_rmse'] for label in label_names])
    metrics['ALL_mae'] = np.mean([metrics[f'{label}_mae'] for label in label_names])
    metrics['ALL_r2'] = np.mean([metrics[f'{label}_r2'] for label in label_names])

    return metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_geoview_loss: bool = False
) -> Dict:
    """Train for one epoch

    Args:
        model: model
        dataloader: data loader
        criterion: main loss function
        optimizer: optimizer
        device: device
        use_geoview_loss: whether to use the GeoView loss function (default False)

    Returns:
        training metrics dict
    """
    model.train()

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_mse_loss = 0.0
    total_physics_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['inputs']
        targets = batch['targets'].to(device)

        # Move the data to the device
        for modality in inputs.keys():
            inputs[modality] = inputs[modality].to(device)

        # If pre-generated masks are present, move them to the device as well
        if 'generated_masks' in batch and batch['generated_masks'] is not None:
            masks = batch['generated_masks']
            inputs['generated_masks'] = {
                'visibility_mask': masks['visibility_mask'].to(device),
                'semantic_mask': masks['semantic_mask'].to(device),
                'mapping_matrix': masks['mapping_matrix'].to(device)
            }

        # Forward pass
        optimizer.zero_grad()
        predictions = model(inputs)

        # Compute the loss
        if use_geoview_loss:
            # GeoViewNetLoss returns (total_loss, base_loss, physics_loss)
            loss, base_loss, physics_loss = criterion(predictions, targets)
            total_mse_loss += base_loss.item()
            total_physics_loss += physics_loss.item()
        else:
            # MSELoss returns a single loss value
            loss = criterion(predictions, targets)

        # Backward pass (using the total loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    # Compute metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    metrics = compute_metrics(all_predictions, all_targets)

    # Record the MSE loss (for metric comparison)
    if use_geoview_loss:
        metrics['loss'] = total_mse_loss / len(dataloader)
        metrics['total_loss'] = total_loss / len(dataloader)
        metrics['physics_loss'] = total_physics_loss / len(dataloader)
    else:
        metrics['loss'] = total_loss / len(dataloader)

    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_geoview_loss: bool = False
) -> Dict:
    """Validate the model

    Args:
        model: model
        dataloader: data loader
        criterion: loss function
        device: device
        use_geoview_loss: whether to use the GeoView loss function (default False)

    Returns:
        validation metrics dict
    """
    model.eval()

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_mse_loss = 0.0
    total_physics_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs']
            targets = batch['targets'].to(device)

            # Move the data to the device
            for modality in inputs.keys():
                inputs[modality] = inputs[modality].to(device)

            # If pre-generated masks are present, move them to the device as well
            if 'generated_masks' in batch and batch['generated_masks'] is not None:
                masks = batch['generated_masks']
                inputs['generated_masks'] = {
                    'visibility_mask': masks['visibility_mask'].to(device),
                    'semantic_mask': masks['semantic_mask'].to(device),
                    'mapping_matrix': masks['mapping_matrix'].to(device)
                }

            # Forward pass
            predictions = model(inputs)

            # Compute the loss
            if use_geoview_loss:
                # GeoViewNetLoss returns (total_loss, base_loss, physics_loss)
                loss, base_loss, physics_loss = criterion(predictions, targets)
                total_mse_loss += base_loss.item()
                total_physics_loss += physics_loss.item()
            else:
                # MSELoss returns a single loss value
                loss = criterion(predictions, targets)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    metrics = compute_metrics(all_predictions, all_targets)

    # Record the MSE loss (for metric comparison)
    if use_geoview_loss:
        metrics['loss'] = total_mse_loss / len(dataloader)
        metrics['total_loss'] = total_loss / len(dataloader)
        metrics['physics_loss'] = total_physics_loss / len(dataloader)
    else:
        metrics['loss'] = total_loss / len(dataloader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train the SVI prediction model')
    parser.add_argument('--config', type=str, default='configs/template.yaml',
                       help='config file path')
    # General config override arguments (supports nested paths, e.g. model.ablation.use_center_circle_mask)
    parser.add_argument('--set', type=str, action='append', nargs=2,
                       metavar=('KEY', 'VALUE'),
                       help='override any config entry, e.g.: --set model.ablation.use_center_circle_mask true '
                            'or: --set data.batch_size 128 (can be used multiple times)')
    # Config override arguments (backward compatible)
    parser.add_argument('--cities', type=str, nargs='+',
                       help='list of cities to use, e.g.: --cities Chicago Boston (equivalent to --set data.cities)')
    parser.add_argument('--perspectives', type=str, nargs='+',
                       help='[deprecated] list of perspectives to use; only panorama is currently supported')
    parser.add_argument('--modalities', type=str, nargs='+',
                       help='list of modalities to use, e.g.: --modalities rsi building tree')
    parser.add_argument('--version', type=str,
                       help='model version, e.g.: --version v3')
    parser.add_argument('--epochs', type=int,
                       help='number of training epochs')
    parser.add_argument('--lr', type=float,
                       help='learning rate')
    parser.add_argument('--batch_size', type=int,
                       help='batch size')
    parser.add_argument('--device', type=str,
                       help='device, e.g.: cuda:0, cuda:1, cpu')
    parser.add_argument('--early_stopping_patience', type=int,
                       help='early stopping patience')
    args = parser.parse_args()

    # Load the config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Handle the --set arguments (nested config overrides)
    if args.set:
        print('\n[Config override] --set arguments detected:')
        for key_value in args.set:
            key, value = key_value
            print(f'  - {key} = {value}')

            # Parse the nested path
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            last_key = keys[-1]

            # Type inference
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif value.startswith('[') and value.endswith(']'):
                # Parse the list
                try:
                    value = eval(value)
                    if not isinstance(value, list):
                        raise ValueError("failed to parse the value as a list")
                except:
                    value = [value]  # convert a single value to a list
            elif value.startswith("'") and value.endswith("'"):
                value = value.strip("'")

            current[last_key] = value
            print(f'    → config[{key}] = {value}')
        print()

    # Command-line arguments override the config file
    if args.cities:
        config['data']['cities'] = args.cities
        print(f'[Config override] cities: {args.cities}')
    if args.perspectives:
        print(f'[Warning] the --perspectives argument is deprecated; only the panorama perspective is supported')
        print(f'[Config override] perspectives: {args.perspectives} (will be ignored)')
        # no longer set perspectives into the config
    if args.modalities:
        config['data']['modalities'] = args.modalities
        print(f'[Config override] modalities: {args.modalities}')
    if args.version:
        config['exp']['version'] = args.version
        print(f'[Config override] version: {args.version}')
    if args.epochs:
        config['train']['epochs'] = args.epochs
        print(f'[Config override] epochs: {args.epochs}')
    if args.lr:
        config['train']['lr'] = args.lr
        print(f'[Config override] lr: {args.lr}')
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
        print(f'[Config override] batch size: {args.batch_size}')
    if args.device:
        config['device'] = args.device
        print(f'[Config override] device: {args.device}')
    if args.early_stopping_patience:
        config['train']['early_stopping_patience'] = args.early_stopping_patience
        print(f'[Config override] early stopping patience: {args.early_stopping_patience}')

    # Set the random seed for reproducibility
    seed = get_seed(config)
    set_seed(seed)

    # Create the output directory
    exp_name = f"{config['exp']['model_name']}_{config['exp']['version']}"
    perspective = config['data'].get('perspective', 'panorama')
    output_dir = Path(f'Parameter/{perspective}') / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the logger
    logger = TrainingLogger(output_dir)
    logger.info(f"Starting training: {exp_name}, epochs: {config['train']['epochs']}")
    logger.info(f"Random seed: {seed}, batch size: {config['data']['batch_size']}")
    logger.info(f"Early stopping patience: {config['train'].get('early_stopping_patience', 10)}")

    # Set the device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Detect whether this is a GeoView architecture family (geoview, geoview2, sscvnet, crossview_resnet)
    architecture_type = config['model']['architecture_type']
    is_geoview = architecture_type in ['geoview', 'geoview2', 'sscvnet', 'crossview_resnet', 'crossview_resnet_optimized']

    # GeoView architectures use pre-generated masks to speed up training
    use_generated_masks = is_geoview

    if is_geoview:
        logger.info(f"GeoView architecture family detected ({architecture_type}); pre-generated mask loading enabled")


    # Create the data loaders
    # Automatically determine whether to use the NPZ format based on feature_root
    feature_root = config['data']['feature_root']
    use_npz_format = 'merged' in feature_root or 'patches_merged' in feature_root

    # With the NPZ format the data is already preprocessed to 128x128; no target_size/resize needed
    # the regular format requires target_size for resizing
    target_size = tuple(config['data']['target_size']) if not use_npz_format else None
    use_preprocessed = use_npz_format  # NPZ format is already preprocessed; the regular format needs resizing

    train_loader, val_loader = create_dataloaders(
        csv_path=config['data']['csv_path'],
        feature_root=feature_root,
        modalities=config['data']['modalities'],
        batch_size=config['data']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data'].get('val_split'),
        cities=config['data'].get('cities'),
        val_cities=config['data'].get('val_cities'),  # new: split by city
        perspectives=None,  # no longer needed; only panorama
        target_size=target_size,  # None for the NPZ format; required for the regular format
        shuffle=config['data'].get('shuffle', True),
        seed=config.get('seed', 42),
        use_preprocessed=use_preprocessed,  # set automatically according to the data format
        use_augmentation=False,  # no data augmentation
        use_generated_masks=use_generated_masks,  # read from the config
        mask_root=None,  # no longer needed; masks are in the NPZ files
        use_npz_format=use_npz_format  # determined automatically from feature_root
    )

    # Report the perspective and cities in use
    logger.info(f"Perspective: {perspective}")
    cities_str = config['data'].get('cities', 'All (all cities)')
    logger.info(f"Cities: {cities_str}")

    # Report the dataset sizes
    logger.info(f"Training set size: {len(train_loader.dataset)}, validation set size: {len(val_loader.dataset)}")

    # Print the modalities in use
    logger.info(f"Modalities: {config['data']['modalities']}")

    # Create the model (automatically uses data.modalities)
    # create_model converts the config automatically according to the architecture type
    # add modalities, data, and exp to the top level of the config for create_model to use
    config['modalities'] = config['data']['modalities']
    # add the data config (used to get target_size)
    config['data_for_model'] = config['data'].copy()
    # exp is already at the top level of the config; no need to add it
    model = create_model(config).to(device)

    # Get detailed model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # For SVIPredictor, show detailed freezing information
    if hasattr(model, 'get_trainable_params_info'):
        params_info = model.get_trainable_params_info()
        logger.info(f"Model parameters: {params_info['total_params']:,}")
        logger.info(f"  Trainable parameters: {params_info['trainable_params']:,}")
        logger.info(f"  Frozen parameters: {params_info['frozen_params']:,}")
        logger.info(f"  Backbone parameters: {params_info['backbone_params']:,} (trainable: {params_info['backbone_trainable']:,})")
        logger.info(f"  Prediction head parameters: {params_info['head_params']:,} (trainable: {params_info['head_trainable']:,})")

        # Show whether pretrained weights are used
        if model.config.get('pretrained', False):
            logger.info(f"  Pretrained weights: yes")
        if model.config.get('freeze_backbone', False):
            logger.info(f"  Backbone frozen: yes")
    else:
        logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # The model in use
    logger.info(f"Model architecture: {architecture_type}")

    # Backbone model information (if present in the config)
    if 'backbone_variant' in config['model']:
        logger.info(f"Backbone model: {config['model']['backbone_variant']}")
    elif 'resnet_variant' in config['model']:
        # GeoView v5.2 architecture uses resnet_variant
        logger.info(f"ResNet variant: {config['model']['resnet_variant']}")
    elif 'module4' in config['model'] and 'backbone_type' in config['model']['module4']:
        # GeoView v5.3 uses backbone_type
        logger.info(f"ResNet variant: {config['model']['module4']['backbone_type']}")


    # Create the loss function and optimizer
    # Select the loss function according to the architecture type
    architecture_type = config['model']['architecture_type']
    if architecture_type == 'geoview' or architecture_type == 'geoview2' or architecture_type == 'sscvnet':
        # GeoView architectures use GeoViewNetLoss

        # Check whether label-weighted loss is used
        use_weighted_loss = config.get('train', {}).get('use_weighted_loss', False)
        label_weights = config.get('train', {}).get('label_weights', None)

        if use_weighted_loss:
            # Use the label-weighted version
            criterion = create_geoview_loss(
                lambda_physics=0.1,
                with_logging=False,
                label_weights=label_weights
            )
            use_geoview_loss = True
            if label_weights:
                logger.info(f"Loss function: GeoViewNetLoss (weighted MSE + Physics, weights={label_weights})")
            else:
                logger.info("Loss function: GeoViewNetLoss (weighted MSE + Physics, default weights=[1.0, 0.8, 1.5])")
        else:
            # Use the standard version
            criterion = create_geoview_loss(
                lambda_physics=0.1,
                with_logging=False
            )
            use_geoview_loss = True
            logger.info("Loss function: GeoViewNetLoss (MSE + Physics)")
    else:
        # Other architectures use MSELoss
        #criterion = nn.MSELoss()
        criterion = create_geoview_loss(
                lambda_physics=0.1,
                with_logging=False
            )
        use_geoview_loss = True
        logger.info("Loss function: GeoViewNetLoss (MSE + Physics)")

    # Create the optimizer
    lr = config['train']['lr']
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    train_metrics_history = []
    val_metrics_history = []
    best_r2 = -float('inf')
    patience_counter = 0
    patience_limit = config['train'].get('early_stopping_patience', 10)  # early stopping patience (read from the config)

    logger.info(f"Starting the training loop for {config['train']['epochs']} epochs")

    for epoch in range(1, config['train']['epochs'] + 1):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_geoview_loss=use_geoview_loss
        )

        # Validate
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            use_geoview_loss=use_geoview_loss
        )

        # Record the metrics
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        # Print the log
        current_lr = optimizer.param_groups[0]['lr']

        # Build the log message (unified format for GeoView and other architectures)
        log_msg = (
            f"Epoch {epoch} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"R²: All={train_metrics['ALL_r2']:.4f}, Veg={train_metrics['vegetation_r2']:.4f}, Sky={train_metrics['sky_r2']:.4f}, Bld={train_metrics['building_r2']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"R²: All={val_metrics['ALL_r2']:.4f}, Veg={val_metrics['vegetation_r2']:.4f}, Sky={val_metrics['sky_r2']:.4f}, Bld={val_metrics['building_r2']:.4f}"
        )

        logger.info(log_msg)

        # Save the best model
        if val_metrics['ALL_r2'] > best_r2:
            best_r2 = val_metrics['ALL_r2']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            logger.info(f"Best model saved (R²: {best_r2:.4f})")
            patience_counter = 0  # reset the early stopping counter
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= patience_limit:
            logger.info(f"Early stopping triggered! Validation R² has not improved for {patience_limit} epochs")
            break

    # Save the metrics to JSON files
    # convert to a serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    train_metrics_serializable = [
        {k: convert_to_serializable(v) for k, v in metrics.items()}
        for metrics in train_metrics_history
    ]
    val_metrics_serializable = [
        {k: convert_to_serializable(v) for k, v in metrics.items()}
        for metrics in val_metrics_history
    ]

    with open(output_dir / 'train_metrics.json', 'w') as f:
        json.dump(train_metrics_serializable, f, indent=2)

    with open(output_dir / 'val_metrics.json', 'w') as f:
        json.dump(val_metrics_serializable, f, indent=2)

    logger.info(f"Training completed! Best R²: {best_r2:.6f}, model saved to: {output_dir / 'best_model.pth'}")
    logger.close()


if __name__ == '__main__':
    main()
