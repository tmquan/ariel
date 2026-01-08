"""
TensorBoard Image Callback for logging training visualizations.

Logs images, embeddings, and predictions during training/validation.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Optional, Dict, Any
from einops import rearrange, repeat

from utils.clustering import cluster_embeddings


def colorize_instances(instance_mask: torch.Tensor, max_instances: int = 50) -> torch.Tensor:
    """
    Convert instance mask to RGB colormap.
    
    Args:
        instance_mask: Instance labels [H, W] with unique integers per instance
        max_instances: Maximum expected instances for color cycling
    
    Returns:
        RGB image [3, H, W] with values in [0, 1]
    """
    # Create a colormap (HSV-based for distinct colors)
    colors = []
    for i in range(max_instances):
        hue = i / max_instances
        # Convert HSV to RGB (simplified)
        if hue < 1/6:
            r, g, b = 1.0, hue * 6, 0.0
        elif hue < 2/6:
            r, g, b = 1.0 - (hue - 1/6) * 6, 1.0, 0.0
        elif hue < 3/6:
            r, g, b = 0.0, 1.0, (hue - 2/6) * 6
        elif hue < 4/6:
            r, g, b = 0.0, 1.0 - (hue - 3/6) * 6, 1.0
        elif hue < 5/6:
            r, g, b = (hue - 4/6) * 6, 0.0, 1.0
        else:
            r, g, b = 1.0, 0.0, 1.0 - (hue - 5/6) * 6
        colors.append([r, g, b])
    
    # [max_instances, 3]
    colors = torch.tensor(colors, device=instance_mask.device, dtype=torch.float32)
    
    # Map instance IDs to colors
    H, W = instance_mask.shape
    rgb = torch.zeros(3, H, W, device=instance_mask.device, dtype=torch.float32)
    
    unique_ids = torch.unique(instance_mask)
    for inst_id in unique_ids:
        if inst_id == 0:  # Background
            continue
        mask = instance_mask == inst_id
        color_idx = (inst_id.item() - 1) % max_instances
        # Assign RGB channels
        rgb[0, mask] = colors[color_idx, 0]
        rgb[1, mask] = colors[color_idx, 1]
        rgb[2, mask] = colors[color_idx, 2]
    
    return rgb


def embedding_to_rgb(embedding: torch.Tensor) -> torch.Tensor:
    """
    Project high-dimensional embedding to RGB using first 3 dimensions.
    
    Args:
        embedding: Embedding tensor [E, H, W]
    
    Returns:
        RGB visualization [3, H, W] with values in [0, 1]
    """
    E, H, W = embedding.shape
    
    # Rearrange to [E, H*W] for processing
    emb_flat = rearrange(embedding, 'e h w -> e (h w)')
    
    # Take first 3 dimensions or pad if E < 3
    if E >= 3:
        rgb_flat = emb_flat[:3]  # [3, H*W]
    else:
        # Pad with zeros: create [3, H*W] tensor
        rgb_flat = torch.zeros(3, H * W, device=embedding.device, dtype=embedding.dtype)
        rgb_flat[:E] = emb_flat
    
    # Normalize each channel to [0, 1]
    rgb_min = rgb_flat.min(dim=1, keepdim=True)[0]  # [3, 1]
    rgb_flat = rgb_flat - rgb_min
    rgb_max = rgb_flat.max(dim=1, keepdim=True)[0].clamp(min=1e-8)  # [3, 1]
    rgb_flat = rgb_flat / rgb_max
    
    # Rearrange back to [3, H, W]
    rgb = rearrange(rgb_flat, 'c (h w) -> c h w', h=H, w=W)
    
    return rgb


class TensorBoardImageCallback(Callback):
    """
    PyTorch Lightning callback for logging images to TensorBoard.
    
    Logs:
    - Input images
    - Ground truth instance masks (colorized)
    - Predicted embeddings (RGB projection)
    - Semantic predictions (foreground/background)
    
    Args:
        log_every_n_steps: Log images every N training steps
        log_every_n_epochs: Log images every N epochs (alternative to steps)
        num_samples: Number of samples to log per batch
        log_train: Whether to log training images
        log_val: Whether to log validation images
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 500,
        log_every_n_epochs: Optional[int] = None,
        num_samples: int = 4,
        log_train: bool = True,
        log_val: bool = True,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.log_train = log_train
        self.log_val = log_val
        
        self._last_train_batch = None
        self._last_val_batch = None
    
    def _should_log_step(self, trainer: pl.Trainer) -> bool:
        """Check if we should log based on step count."""
        if self.log_every_n_epochs is not None:
            return False
        return trainer.global_step % self.log_every_n_steps == 0
    
    def _should_log_epoch(self, trainer: pl.Trainer) -> bool:
        """Check if we should log based on epoch count."""
        if self.log_every_n_epochs is None:
            return False
        return trainer.current_epoch % self.log_every_n_epochs == 0
    
    def _log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        prefix: str
    ):
        """Log images to TensorBoard."""
        if trainer.logger is None:
            return
        
        # Get tensorboard logger
        logger = trainer.logger.experiment
        if not hasattr(logger, 'add_image'):
            return
        
        step = trainer.global_step
        n_samples = min(self.num_samples, batch['image'].shape[0])
        
        for i in range(n_samples):
            # Input image: [C, H, W]
            img = batch['image'][i]
            
            if img.shape[0] == 1:
                # Grayscale to RGB: [1, H, W] -> [3, H, W]
                img = repeat(img, '1 h w -> 3 h w')
            elif img.shape[0] > 3:
                # Take first 3 channels
                img = img[:3]
            
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            logger.add_image(f'{prefix}/input_{i}', img, step)
            
            # Ground truth instance mask
            if 'label' in batch:
                label = batch['label'][i]
                if label.dim() == 3:
                    label = rearrange(label, '1 h w -> h w')
                gt_rgb = colorize_instances(label.long())
                logger.add_image(f'{prefix}/gt_instances_{i}', gt_rgb, step)
            
            # Predicted embedding: [E, H, W]
            if outputs is not None and 'embedding' in outputs:
                emb = outputs['embedding'][i]
                emb_rgb = embedding_to_rgb(emb)
                logger.add_image(f'{prefix}/embedding_{i}', emb_rgb, step)
            
            # Semantic prediction (foreground/background): [2, H, W] or [1, H, W]
            if outputs is not None and 'semantic' in outputs:
                sem = outputs['semantic'][i]
                if sem.shape[0] == 2:
                    # Softmax and take foreground probability: [2, H, W] -> [1, H, W]
                    sem_prob = F.softmax(sem, dim=0)[1:2]
                    sem_pred = outputs['semantic'][i].argmax(dim=0)  # [H, W]
                else:
                    sem_prob = torch.sigmoid(sem)
                    sem_pred = (sem_prob > 0.5).squeeze(0)  # [H, W]
                
                # Convert to RGB (grayscale): [1, H, W] -> [3, H, W]
                sem_rgb = repeat(sem_prob, '1 h w -> 3 h w')
                logger.add_image(f'{prefix}/semantic_pred_{i}', sem_rgb, step)
                
                # Predicted instances via clustering
                if outputs is not None and 'embedding' in outputs:
                    try:
                        # Get embedding for this sample: [E, H, W]
                        emb_single = outputs['embedding'][i:i+1]  # [1, E, H, W]
                        
                        # Get delta_var from pl_module if available
                        delta_var = getattr(pl_module, 'discriminative_loss', None)
                        bandwidth = delta_var.delta_var if delta_var is not None else 0.5
                        
                        # Cluster embeddings using semantic mask
                        pred_instances = cluster_embeddings(
                            emb_single,
                            foreground_mask=sem_pred.unsqueeze(0),  # [1, H, W]
                            method='sklearn_meanshift',
                            bandwidth=bandwidth,
                            device=emb_single.device
                        )
                        
                        # Colorize and log: pred_instances is [1, H, W]
                        pred_rgb = colorize_instances(pred_instances[0].long())
                        logger.add_image(f'{prefix}/pred_instances_{i}', pred_rgb, step)
                    except Exception as e:
                        # Clustering can fail, just skip logging
                        pass
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ):
        """Called at the end of training batch."""
        if not self.log_train:
            return
        
        self._last_train_batch = batch
        
        if self._should_log_step(trainer):
            # Get model predictions
            with torch.no_grad():
                pred_outputs = pl_module(batch['image'])
            
            self._log_images(trainer, pl_module, batch, pred_outputs, 'train')
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Called at the end of validation batch."""
        if not self.log_val:
            return
        
        if batch_idx == 0:
            self._last_val_batch = batch
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """Log validation images at end of epoch."""
        if not self.log_val or self._last_val_batch is None:
            return
        
        if self._should_log_epoch(trainer) or self._should_log_step(trainer):
            batch = self._last_val_batch
            with torch.no_grad():
                pred_outputs = pl_module(batch['image'].to(pl_module.device))
            
            self._log_images(trainer, pl_module, batch, pred_outputs, 'val')
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """Log training images at end of epoch if using epoch-based logging."""
        if not self.log_train or self._last_train_batch is None:
            return
        
        if self._should_log_epoch(trainer):
            batch = self._last_train_batch
            with torch.no_grad():
                pred_outputs = pl_module(batch['image'].to(pl_module.device))
            
            self._log_images(trainer, pl_module, batch, pred_outputs, 'train')
