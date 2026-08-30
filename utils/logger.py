"""
Logging management module
Unified log format management, supporting file, console, and TensorBoard
"""

import os
import json
import logging
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import torch
import numpy as np


class TrainingLogger:
    """Simplified training logger - concise version"""

    def __init__(self, output_dir: Union[str, Path]):
        """
        Args:
            output_dir: output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create the log file
        self.log_file = self.output_dir / 'train.log'

        # Set up the logger
        self.logger = logging.getLogger('SVI_Training')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Set the format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        """Log an INFO-level message"""
        self.logger.info(message)
        # Force-flush the file handler
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

    def close(self):
        """Close the logger"""
        for handler in self.logger.handlers:
            handler.close()


class Logger:
    """Unified log manager"""

    def __init__(
        self,
        output_dir: str,
        config: Dict[str, Any],
        log_file: str = 'train.log',
        use_tensorboard: bool = True
    ):
        """
        Args:
            output_dir: output directory
            config: configuration dict
            log_file: log file name
            use_tensorboard: whether to use TensorBoard
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.use_tensorboard = use_tensorboard

        # Create the required subdirectories
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.tables_dir = self.output_dir / 'tables'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'

        for dir in [self.checkpoints_dir, self.tables_dir, self.logs_dir, self.plots_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        # History: saves the metrics of each epoch
        self.history = {
            'epoch': [],
            'train': [],
            'val': [],
            'lr': []
        }

        # Set up the logger
        self.logger = logging.getLogger('SVI_Training')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(
            self.logs_dir / log_file,
            mode='w',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Set the format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # TensorBoard writer
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.logs_dir / 'tensorboard'))
            except ImportError:
                self.logger.warning("TensorBoard is not installed; skipping TensorBoard logging")
                self.use_tensorboard = False

        # Save the config snapshot
        self.save_config()
        # Save the metadata
        self.save_metadata()

    def info(self, message: str):
        """Log an INFO-level message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a WARNING-level message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an ERROR-level message"""
        self.logger.error(message)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None
    ):
        """Log the metrics of each epoch

        Args:
            epoch: current epoch number
            train_metrics: training metrics dict
            val_metrics: validation metrics dict (optional)
            learning_rate: current learning rate (optional)
        """
        # Save to the history
        self.history['epoch'].append(epoch)
        self.history['train'].append(train_metrics.copy())
        if val_metrics:
            self.history['val'].append(val_metrics.copy())
        else:
            # If there are no validation metrics, append an empty dict
            self.history['val'].append({})
        if learning_rate is not None:
            self.history['lr'].append(learning_rate)
        else:
            self.history['lr'].append(None)

        # Build the log message
        msg_parts = [f"Epoch {epoch}"]

        if train_metrics:
            train_str = ", ".join([f"{k}: {v:.6f}" for k, v in train_metrics.items()])
            msg_parts.append(f"Train - {train_str}")

        if val_metrics:
            val_str = ", ".join([f"{k}: {v:.6f}" for k, v in val_metrics.items()])
            msg_parts.append(f"Val - {val_str}")

        if learning_rate is not None:
            msg_parts.append(f"LR: {learning_rate:.6f}")

        self.info(" | ".join(msg_parts))

        # Log to TensorBoard
        if self.writer is not None:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)

            if val_metrics:
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

            if learning_rate is not None:
                self.writer.add_scalar('train/learning_rate', learning_rate, epoch)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: str = 'checkpoint.pth'
    ):
        """Save a model checkpoint

        Args:
            model: model instance
            optimizer: optimizer
            scheduler: learning rate scheduler (optional)
            epoch: current epoch
            metrics: metrics dict
            is_best: whether this is the best model
            filename: file name
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save the checkpoint
        checkpoint_path = self.checkpoints_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.info(f"Checkpoint saved: {checkpoint_path}")

        # If this is the best model, additionally save it as best.pth
        if is_best:
            best_path = self.checkpoints_dir / 'best.pth'
            shutil.copy(checkpoint_path, best_path)
            self.info(f"Best model saved: {best_path}")

    def save_config(self, filename: str = 'config_snapshot.yaml'):
        """Save a config snapshot"""
        config_path = self.logs_dir / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        self.info(f"Config snapshot saved: {config_path}")

    def save_metadata(self, filename: str = 'meta.json'):
        """Save metadata"""
        meta = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.config.get('seed', 42),
            'device': self.config.get('device', 'unknown'),
            'hostname': os.uname().nodename,
        }

        # Try to collect git information
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            meta['git_commit'] = git_commit
        except:
            meta['git_commit'] = 'unknown'

        meta_path = self.logs_dir / filename
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self.info(f"Metadata saved: {meta_path}")

    def save_metrics(self, metrics: Dict[str, Any], filename: str = 'metrics.json'):
        """Save evaluation metrics

        Args:
            metrics: metrics dict
            filename: file name
        """
        metrics_path = self.tables_dir / filename
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        self.info(f"Metrics saved: {metrics_path}")

    def save_predictions(
        self,
        predictions: Dict[str, Any],
        filename: str = 'preds.npz'
    ):
        """Save prediction results

        Args:
            predictions: prediction results dict, containing y_true, y_pred, etc.
            filename: file name
        """
        preds_path = self.tables_dir / filename
        np.savez(preds_path, **predictions)
        self.info(f"Predictions saved: {preds_path}")

    def close(self):
        """Close the logger"""
        # Save the training history
        self.save_history()

        if self.writer is not None:
            self.writer.close()

        for handler in self.logger.handlers:
            handler.close()

    def save_history(self, filename: str = 'train_history.npz'):
        """Save the training history to an npy file

        Args:
            filename: file name
        """
        history_path = self.tables_dir / filename

        # Convert the history into a saveable format
        save_dict = {
            'epoch': np.array(self.history['epoch']),
            'lr': np.array(self.history['lr'])
        }

        # Save the training and validation metrics
        if self.history['train']:
            # Get all metric names
            metric_names = list(self.history['train'][0].keys())
            for metric_name in metric_names:
                save_dict[f'train_{metric_name}'] = np.array([
                    epoch_metrics.get(metric_name, np.nan)
                    for epoch_metrics in self.history['train']
                ])

        if self.history['val'] and any(self.history['val']):
            # Get all metric names (from the first non-empty validation record)
            metric_names = list(next(
                (m.keys() for m in self.history['val'] if m),
                {}
            ))
            for metric_name in metric_names:
                save_dict[f'val_{metric_name}'] = np.array([
                    epoch_metrics.get(metric_name, np.nan)
                    for epoch_metrics in self.history['val']
                ])

        # Save as an npz file
        np.savez(history_path, **save_dict)
        self.info(f"Training history saved: {history_path}")
