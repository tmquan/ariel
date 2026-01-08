"""
ARIEL: Instance Segmentation with Discriminative Loss

Main entry point for training pixel embedding models for instance segmentation.
Supports CVPPP plant datasets and SNEMI neuron segmentation.

This script uses:
- OmegaConf configuration management
- PyTorch Lightning training abstraction
- MONAI data pipelines
- Discriminative loss for instance embedding

Usage:
    # Training with default SNEMI config
    python code/main.py
    
    # Training with CVPPP config
    python code/main.py --config conf/config_cvppp.yaml
    
    # Override parameters via CLI
    python code/main.py --data.batch_size 8 --training.max_epochs 200
    
    # Fast development run
    python code/main.py --training.fast_dev_run True
"""

import os
import sys
import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set multiprocessing start method to 'spawn' for better compatibility with Python 3.14
# This must be done before importing torch
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import torch

# Enable Tensor Core optimization for supported GPUs (e.g., NVIDIA GB200)
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelSummary,
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from einops import rearrange

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.ariel2d import ArielTransfer2D
from loader.datamodule import (
    SNEMIDataModule,
    CVPPP14DataModule,
    CVPPP15DataModule,
    CVPPP17DataModule,
    CVPPPRLDataModule,
    CombinedCVPPPDataModule,
)
from losses.discriminative import DiscriminativeLoss
from callbacks.tensorboard_image import TensorBoardImageCallback
from utils.clustering import cluster_embeddings, compute_instance_metrics


class ArielLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for ArielTransfer2D model.
    
    Handles:
    - Forward pass with semantic + instance heads
    - Loss computation (discriminative + cross-entropy)
    - Optimization
    - Logging
    """
    
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        loss_config: Dict
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = ArielTransfer2D(model_config=model_config)
        
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        
        # Loss functions
        # Discriminative loss for instance embeddings
        disc_cfg = loss_config.get('discriminative', {})
        self.discriminative_loss = DiscriminativeLoss(
            delta_var=disc_cfg.get('delta_var', 0.5),
            delta_dist=disc_cfg.get('delta_dist', 1.5),
            norm=disc_cfg.get('norm', 2),
            alpha=disc_cfg.get('alpha', 1.0),
            beta=disc_cfg.get('beta', 1.0),
            gamma=disc_cfg.get('gamma', 0.001),
        )
        
        # Cross-entropy loss for semantic segmentation (fg/bg)
        self.semantic_loss = nn.CrossEntropyLoss()
        
        # Loss weights
        self.semantic_weight = loss_config.get('semantic_weight', 1.0)
        self.instance_weight = loss_config.get('instance_weight', 1.0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all outputs."""
        return self.model(x)
    
    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all losses.
        
        Args:
            outputs: Model outputs with 'semantic' and 'embedding'
            labels: Instance segmentation labels [B, 1, H, W] or [B, H, W]
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Semantic loss: foreground (label > 0) vs background (label == 0)
        semantic_logits = outputs['semantic']  # [B, 2, H, W]
        
        # Ensure labels shape: [B, 1, H, W] -> [B, H, W]
        if labels.dim() == 4:
            labels_squeezed = rearrange(labels, 'b 1 h w -> b h w')
        else:
            labels_squeezed = labels
        
        # Create binary mask: 0 = background, 1 = foreground
        semantic_target = (labels_squeezed > 0).long()  # [B, H, W]
        
        loss_semantic = self.semantic_loss(semantic_logits, semantic_target)
        
        # Discriminative loss for instance embeddings
        embedding = outputs['embedding']  # [B, E, H, W]
        
        loss_disc, loss_var, loss_dist, loss_reg = self.discriminative_loss(
            embedding, labels_squeezed
        )
        
        # Total loss
        total_loss = (
            self.semantic_weight * loss_semantic +
            self.instance_weight * loss_disc
        )
        
        loss_dict = {
            'loss': total_loss,
            'loss_semantic': loss_semantic,
            'loss_disc': loss_disc,
            'loss_var': loss_var,
            'loss_dist': loss_dist,
            'loss_reg': loss_reg,
        }
        
        return total_loss, loss_dict
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images)
        
        # Compute losses
        loss, loss_dict = self._compute_losses(outputs, labels)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/loss_semantic', loss_dict['loss_semantic'], on_step=False, on_epoch=True)
        self.log('train/loss_disc', loss_dict['loss_disc'], on_step=False, on_epoch=True)
        self.log('train/loss_var', loss_dict['loss_var'], on_step=False, on_epoch=True)
        self.log('train/loss_dist', loss_dict['loss_dist'], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with clustering-based instance metrics (computed only on first batch)."""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images)
        
        # Compute losses
        loss, loss_dict = self._compute_losses(outputs, labels)
        
        # Prepare labels: [B, 1, H, W] -> [B, H, W]
        if labels.dim() == 4:
            labels_squeezed = rearrange(labels, 'b 1 h w -> b h w')
        else:
            labels_squeezed = labels
        
        # Compute semantic metrics: [B, 2, H, W] -> [B, H, W]
        semantic_pred = outputs['semantic'].argmax(dim=1)
        semantic_target = (labels_squeezed > 0).long()
        
        # IoU for foreground
        intersection = ((semantic_pred == 1) & (semantic_target == 1)).sum().float()
        union = ((semantic_pred == 1) | (semantic_target == 1)).sum().float()
        iou = intersection / (union + 1e-8)
        
        # Log basic metrics for all batches
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/loss_semantic', loss_dict['loss_semantic'], on_step=False, on_epoch=True)
        self.log('val/loss_disc', loss_dict['loss_disc'], on_step=False, on_epoch=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True)
        
        # Clustering is expensive - only compute on first batch
        if batch_idx == 0:
            embedding = outputs['embedding']  # [B, E, H, W]
            foreground_mask = semantic_pred  # [B, H, W]
            
            # Get bandwidth from discriminative loss config
            bandwidth = self.loss_config.get('discriminative', {}).get('delta_var', 0.5)
            
            # Cluster embeddings (only first sample to speed up)
            pred_instances = cluster_embeddings(
                embedding[:1],  # Only first sample
                foreground_mask=foreground_mask[:1],
                method='sklearn_meanshift',
                bandwidth=bandwidth,
                min_cluster_size=50,
                device=self.device,
            )
            
            # Compute instance segmentation metrics
            ari, ami = compute_instance_metrics(pred_instances, labels_squeezed[:1].long())
            
            self.log('val/ari', ari, on_step=False, on_epoch=True)
            self.log('val/ami', ami, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.get('lr', 1e-3),
            weight_decay=self.optimizer_config.get('weight_decay', 1e-4)
        )
        
        scheduler_config = self.optimizer_config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'monitor': 'val/loss'
                }
            }
        
        return optimizer


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file (default: conf/config.yaml)
        overrides: Dictionary of overrides to apply
    
    Returns:
        DictConfig object with merged configuration
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "conf" / "config.yaml"
    
    config_path = Path(config_path)
    
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
    else:
        # Create default config if file doesn't exist
        cfg = create_default_config()
    
    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    return cfg


def create_default_config() -> DictConfig:
    """Create default configuration."""
    return OmegaConf.create({
        'experiment_name': 'ariel_snemi3d',
        'project_name': 'ariel-connectomics',
        'seed': 42,
        'logger': 'tensorboard',
        'log_dir': 'logs',
        'data': {
            'dataset': 'snemi',
            'data_root': 'data',
            'batch_size': 4,
            'num_workers': 4,
            'train_val_split': 0.2,
            'cache_rate': 0.5,
            'pin_memory': True,
            'image_size': None,
            'crop_size': [256, 256],
            'challenge': 'all',
        },
        'model': {
            'net_config': {
                'in_channels': 1,
                'init_filters': 32,
                'feature_dim': 64,
                'emb_dim': 16,
                'dropout': 0.2,
            },
            'optimizer': {
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': {
                    'type': 'cosine',
                    'T_max': 100,
                    'eta_min': 1e-6
                }
            }
        },
        'loss': {
            'discriminative': {
                'delta_var': 0.5,
                'delta_dist': 1.5,
                'norm': 2,
                'alpha': 1.0,
                'beta': 1.0,
                'gamma': 0.001,
            },
            'semantic_weight': 1.0,
            'instance_weight': 1.0,
        },
        'training': {
            'max_epochs': 100,
            'accelerator': 'auto',
            'devices': 1,
            'strategy': 'auto',
            'precision': '32-true',
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'val_check_interval': 1.0,
            'check_val_every_n_epoch': 1,
            'num_sanity_val_steps': 2,
            'log_every_n_steps': 50,
            'enable_progress_bar': True,
            'enable_model_summary': True,
            'deterministic': False,
            'benchmark': True,
            'fast_dev_run': False,
        },
        'callbacks': {
            'checkpoint': {
                'enabled': True,
                'dirpath': 'checkpoints',
                'save_top_k': 3,
                'monitor': 'val/loss',
                'mode': 'min',
                'save_last': True
            },
            'early_stopping': {
                'enabled': False,
                'patience': 20,
                'monitor': 'val/loss',
                'mode': 'min',
            },
            'lr_monitor': {
                'enabled': True
            },
            'tensorboard_image': {
                'enabled': True,
                'log_every_n_steps': 200,
                'log_every_n_epochs': 1,
                'num_samples': 4,
                'log_train': True,
                'log_val': True,
            }
        }
    })


def parse_cli_overrides() -> Tuple[argparse.Namespace, Dict]:
    """Parse CLI arguments into override dictionary."""
    parser = argparse.ArgumentParser(description='ARIEL Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--inference', action='store_true', help='Run in inference mode')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path for inference')
    parser.add_argument('--volume_path', type=str, default=None, help='Volume path for inference')
    
    # Parse known args, rest are config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse override arguments (--key.subkey value format)
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]  # Remove --
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                # Try to parse as number or boolean
                try:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                # Convert dot notation to nested dict
                keys = key.split('.')
                d = overrides
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    return args, overrides


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Create appropriate datamodule based on config."""
    data_cfg = cfg.data
    dataset_type = data_cfg.get('dataset', 'snemi').lower()
    
    # Common args
    common_args = {
        'data_root': data_cfg.get('data_root', 'data'),
        'batch_size': data_cfg.get('batch_size', 4),
        'num_workers': data_cfg.get('num_workers', 4),
        'train_val_split': data_cfg.get('train_val_split', 0.2),
        'cache_rate': data_cfg.get('cache_rate', 0.5),
        'pin_memory': data_cfg.get('pin_memory', True),
    }
    
    # Image size
    image_size = data_cfg.get('image_size', None)
    if image_size is not None:
        common_args['image_size'] = tuple(image_size) if isinstance(image_size, list) else image_size
    
    if dataset_type == 'snemi':
        return SNEMIDataModule(**common_args)
    
    elif dataset_type == 'cvppp14':
        crop_size = data_cfg.get('crop_size', [256, 256])
        return CVPPP14DataModule(
            **common_args,
            crop_size=tuple(crop_size) if isinstance(crop_size, list) else crop_size,
        )
    
    elif dataset_type == 'cvppp15':
        crop_size = data_cfg.get('crop_size', [256, 256])
        challenge = data_cfg.get('challenge', 'all')
        return CVPPP15DataModule(
            **common_args,
            crop_size=tuple(crop_size) if isinstance(crop_size, list) else crop_size,
            challenge=challenge,
        )
    
    elif dataset_type == 'cvppp17':
        crop_size = data_cfg.get('crop_size', [256, 256])
        challenge = data_cfg.get('challenge', 'all')
        return CVPPP17DataModule(
            **common_args,
            crop_size=tuple(crop_size) if isinstance(crop_size, list) else crop_size,
            challenge=challenge,
        )
    
    elif dataset_type in ('cvppp_rl', 'prl', 'ppp_prl'):
        subset = data_cfg.get('subset', 'all')
        return CVPPPRLDataModule(
            **common_args,
            subset=subset,
        )
    
    elif dataset_type in ('cvppp_all', 'combined', 'all'):
        # Combined CVPPP dataset
        data_roots = data_cfg.get('data_roots', {})
        crop_size = data_cfg.get('crop_size', [256, 256])
        return CombinedCVPPPDataModule(
            data_roots=data_roots,
            batch_size=data_cfg.get('batch_size', 4),
            num_workers=data_cfg.get('num_workers', 0),
            train_val_split=data_cfg.get('train_val_split', 0.2),
            cache_rate=data_cfg.get('cache_rate', 1.0),
            pin_memory=data_cfg.get('pin_memory', True),
            image_size=tuple(image_size) if image_size else None,
            crop_size=tuple(crop_size) if isinstance(crop_size, list) else crop_size,
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def setup_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Setup training callbacks from configuration."""
    callbacks = []
    
    callback_cfg = cfg.get('callbacks', {})
    
    # Model Checkpoint
    checkpoint_cfg = callback_cfg.get('checkpoint', {})
    if checkpoint_cfg.get('enabled', True):
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_cfg.get('dirpath', 'checkpoints'),
                filename=checkpoint_cfg.get('filename', 'ariel-{epoch:02d}-{val/loss:.4f}'),
                save_top_k=checkpoint_cfg.get('save_top_k', 3),
                monitor=checkpoint_cfg.get('monitor', 'val/loss'),
                mode=checkpoint_cfg.get('mode', 'min'),
                save_last=checkpoint_cfg.get('save_last', True),
                verbose=checkpoint_cfg.get('verbose', True),
                auto_insert_metric_name=False
            )
        )
    
    # Early Stopping
    early_stopping_cfg = callback_cfg.get('early_stopping', {})
    if early_stopping_cfg.get('enabled', False):
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_cfg.get('monitor', 'val/loss'),
                patience=early_stopping_cfg.get('patience', 20),
                mode=early_stopping_cfg.get('mode', 'min'),
                verbose=early_stopping_cfg.get('verbose', True),
                min_delta=early_stopping_cfg.get('min_delta', 0.0)
            )
        )
    
    # Learning Rate Monitor
    if callback_cfg.get('lr_monitor', {}).get('enabled', True):
        callbacks.append(
            LearningRateMonitor(logging_interval='step')
        )
    
    # TensorBoard Image Callback
    tb_image_cfg = callback_cfg.get('tensorboard_image', {})
    if tb_image_cfg.get('enabled', True):
        callbacks.append(
            TensorBoardImageCallback(
                log_every_n_steps=tb_image_cfg.get('log_every_n_steps', 200),
                log_every_n_epochs=tb_image_cfg.get('log_every_n_epochs', 1),
                num_samples=tb_image_cfg.get('num_samples', 4),
                log_train=tb_image_cfg.get('log_train', True),
                log_val=tb_image_cfg.get('log_val', True),
            )
        )
    
    # Progress Bar
    callbacks.append(RichProgressBar())
    
    # Model Summary
    callbacks.append(ModelSummary(max_depth=2))
    
    return callbacks


def setup_logger(cfg: DictConfig):
    """Setup experiment logger."""
    logger_type = cfg.get('logger', 'tensorboard')
    
    if logger_type == 'tensorboard':
        return TensorBoardLogger(
            save_dir=cfg.get('log_dir', 'logs'),
            name=cfg.get('experiment_name', 'ariel'),
            version=None
        )
    elif logger_type == 'wandb':
        return WandbLogger(
            project=cfg.get('project_name', 'ariel-instance-seg'),
            name=f"{cfg.get('experiment_name', 'ariel')}_{cfg.get('seed', 42)}",
            save_dir=cfg.get('log_dir', 'logs')
        )
    else:
        return True  # Default Lightning logger


def main(cfg: DictConfig):
    """
    Main training entry point.
    
    Args:
        cfg: Configuration object containing all parameters
    """
    # Print configuration
    print("=" * 60)
    print("ARIEL - Instance Segmentation with Discriminative Loss")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    pl.seed_everything(seed, workers=True)
    print(f"\n✓ Random seed set to: {seed}")
    
    # Initialize DataModule
    datamodule = get_datamodule(cfg)
    data_cfg = cfg.data
    
    print(f"\n✓ DataModule initialized:")
    print(f"    Dataset: {data_cfg.get('dataset', 'snemi')}")
    print(f"    Data root: {data_cfg.get('data_root', 'data')}")
    print(f"    Batch size: {data_cfg.get('batch_size', 4)}")
    print(f"    Train/Val split: {data_cfg.get('train_val_split', 0.2)}")
    
    # Initialize Model
    model_cfg = cfg.model
    model = ArielLightningModule(
        model_config=dict(model_cfg.get('net_config', {})),
        optimizer_config=dict(model_cfg.get('optimizer', {})),
        loss_config=dict(cfg.get('loss', {}))
    )
    
    print(f"\n✓ Model initialized:")
    print(f"    Architecture: ArielTransfer2D")
    print(f"    Input channels: {model_cfg.get('net_config', {}).get('in_channels', 1)}")
    print(f"    Init filters: {model_cfg.get('net_config', {}).get('init_filters', 32)}")
    print(f"    Feature dim: {model_cfg.get('net_config', {}).get('feature_dim', 64)}")
    print(f"    Embedding dim: {model_cfg.get('net_config', {}).get('emb_dim', 16)}")
    
    # Print loss config
    loss_cfg = cfg.get('loss', {})
    disc_cfg = loss_cfg.get('discriminative', {})
    print(f"\n✓ Loss configuration:")
    print(f"    Discriminative Loss:")
    print(f"      delta_var: {disc_cfg.get('delta_var', 0.5)}")
    print(f"      delta_dist: {disc_cfg.get('delta_dist', 1.5)}")
    print(f"      alpha (var): {disc_cfg.get('alpha', 1.0)}")
    print(f"      beta (dist): {disc_cfg.get('beta', 1.0)}")
    print(f"      gamma (reg): {disc_cfg.get('gamma', 0.001)}")
    print(f"    Semantic weight: {loss_cfg.get('semantic_weight', 1.0)}")
    print(f"    Instance weight: {loss_cfg.get('instance_weight', 1.0)}")
    
    # Setup Callbacks
    callbacks = setup_callbacks(cfg)
    print(f"\n✓ Callbacks: {len(callbacks)} callbacks registered")
    
    # Setup Logger
    logger = setup_logger(cfg)
    print(f"✓ Logger: {cfg.get('logger', 'tensorboard')}")
    
    # Initialize Trainer
    training_cfg = cfg.training
    trainer = pl.Trainer(
        max_epochs=training_cfg.get('max_epochs', 100),
        accelerator=training_cfg.get('accelerator', 'auto'),
        devices=training_cfg.get('devices', 1),
        strategy=training_cfg.get('strategy', 'auto'),
        precision=training_cfg.get('precision', '32-true'),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=training_cfg.get('log_every_n_steps', 50),
        gradient_clip_val=training_cfg.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=training_cfg.get('accumulate_grad_batches', 1),
        val_check_interval=training_cfg.get('val_check_interval', 1.0),
        check_val_every_n_epoch=training_cfg.get('check_val_every_n_epoch', 1),
        num_sanity_val_steps=training_cfg.get('num_sanity_val_steps', 2),
        enable_progress_bar=training_cfg.get('enable_progress_bar', True),
        enable_model_summary=training_cfg.get('enable_model_summary', True),
        deterministic=training_cfg.get('deterministic', False),
        benchmark=training_cfg.get('benchmark', True),
        fast_dev_run=training_cfg.get('fast_dev_run', False)
    )
    
    print(f"\n✓ Trainer initialized:")
    print(f"    Max epochs: {training_cfg.get('max_epochs', 100)}")
    print(f"    Accelerator: {training_cfg.get('accelerator', 'auto')}")
    print(f"    Devices: {training_cfg.get('devices', 1)}")
    print(f"    Precision: {training_cfg.get('precision', '32-true')}")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise
    
    # Save final checkpoint
    if trainer.global_rank == 0:
        final_path = Path('checkpoints') / 'final_model.ckpt'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_path))
        print(f"\n✓ Final model saved to: {final_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    
    return trainer.callback_metrics


def predict(cfg: DictConfig, checkpoint_path: str, volume_path: str):
    """
    Inference entry point for processing new volumes.
    
    Args:
        cfg: Configuration object
        checkpoint_path: Path to model checkpoint
        volume_path: Path to input volume
    """
    print("=" * 60)
    print("ARIEL - Inference Mode")
    print("=" * 60)
    
    # Load trained model
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
    
    print(f"\n✓ Loading model from: {checkpoint_path}")
    model = ArielLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Setup inference data
    if not volume_path or not os.path.exists(volume_path):
        raise ValueError(f"Invalid volume path: {volume_path}")
    
    print(f"✓ Processing volume: {volume_path}")
    
    # TODO: Implement inference pipeline
    # This would include:
    # - Load image/volume
    # - Run forward pass
    # - Cluster embeddings to instances (mean-shift, HDBSCAN, etc.)
    # - Save results
    
    print("\n⚠ Inference implementation TODO")
    print("  - Load image")
    print("  - Run model forward pass")
    print("  - Cluster embeddings to instances")
    print("  - Save segmentation results")
    
    return None


if __name__ == "__main__":
    # Parse CLI arguments
    args, overrides = parse_cli_overrides()
    
    # Load configuration
    cfg = load_config(args.config, overrides)
    
    # Run training or inference
    if args.inference:
        predict(cfg, args.checkpoint_path, args.volume_path)
    else:
        main(cfg)
