"""
ARIEL 2D Models for Instance Segmentation.

This module provides:
- ArielTransfer2D: Multi-task model with semantic (fg/bg) + instance embedding heads
- ArielEstimate2D: Affinity estimation network for score prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from monai.networks.nets import SegResNet, ResNet
from monai.networks.nets.resnet import ResNetBlock


class ArielTransfer2D(nn.Module):
    """
    Multi-task segmentation network for instance segmentation.
    
    Architecture:
    - Backbone: SegResNet encoder-decoder
    - Semantic Head: Foreground/Background segmentation (2 classes)
    - Instance Head: Pixel embedding for discriminative clustering
    
    Args:
        model_config: Configuration dictionary with:
            - in_channels: Input channels (default: 1 for grayscale, 3 for RGB)
            - init_filters: Initial conv filters (default: 32)
            - feature_dim: Backbone output features (default: 32)
            - emb_dim: Instance embedding dimension (default: 16)
            - dropout: Dropout probability (default: 0.2)
            - blocks_down: Encoder block depths (default: (1, 2, 2, 4))
            - blocks_up: Decoder block depths (default: (1, 1, 1))
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if model_config is None:
            model_config = {}
        
        self.in_channels = model_config.get('in_channels', 1)
        self.feature_dim = model_config.get('feature_dim', 32)
        self.emb_dim = model_config.get('emb_dim', 16)
        
        # Backbone: SegResNet encoder-decoder
        self.backbone = SegResNet(
            spatial_dims=2,
            init_filters=model_config.get('init_filters', 32),
            in_channels=self.in_channels,
            out_channels=self.feature_dim,
            dropout_prob=model_config.get('dropout', 0.2),
            blocks_down=model_config.get('blocks_down', (1, 2, 2, 4)),
            blocks_up=model_config.get('blocks_up', (1, 1, 1)),
        )
        
        # Semantic Head: Foreground/Background (2 classes)
        self.sem_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 2, 2, kernel_size=1),  # 2 classes: bg, fg
        )
        
        # Instance Embedding Head: High-dimensional pixel embeddings
        self.ins_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, self.emb_dim, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image [B, C, H, W]
        
        Returns:
            Dictionary with:
                - 'features': Backbone features [B, feature_dim, H, W]
                - 'semantic': Semantic logits [B, 2, H, W] (bg, fg)
                - 'embedding': Instance embeddings [B, emb_dim, H, W]
        """
        # Backbone features
        features = self.backbone(x)  # [B, feature_dim, H, W]
        
        # Semantic segmentation (foreground/background)
        semantic = self.sem_head(features)  # [B, 2, H, W]
        
        # Instance embedding
        embedding = self.ins_head(features)  # [B, emb_dim, H, W]
        
        return {
            'features': features,
            'semantic': semantic,
            'embedding': embedding,
        }


class ArielEstimate2D(nn.Module):
    """
    Affinity estimation network for connectomics segmentation.
    
    Takes predicted and ground truth affinity maps as input and outputs
    a score with sigmoid activation.
    
    Uses only MONAI ResNet backbone with sigmoid output.
    
    Args:
        model_config: Configuration dictionary with:
            - in_channels: Number of input channels (default: 2 = 1 pred + 1 gt)
            - out_channels: Output channels (default: 1)
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if model_config is None:
            model_config = {}
        
        # Input: 1 channel (Pred Affinity) + 1 channel (GT Affinity) = 2 channels
        self.in_channels = model_config.get('in_channels', 2)
        self.out_channels = model_config.get('out_channels', 1)
        
        # Backbone: MONAI ResNet with sigmoid output
        self.backbone = ResNet(
            spatial_dims=2,
            n_input_channels=self.in_channels,
            num_classes=self.out_channels,
            block=ResNetBlock,
            layers=(2, 2, 2, 2),  # ResNet18 configuration
            block_inplanes=(64, 128, 256, 512),
        )
        
        # Sigmoid activation
        self.activate = nn.Sigmoid()
    
    def forward(
        self,
        label_pred: torch.Tensor,
        label_true: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            label_pred: Predicted affinity maps [B, 1, H, W]
            label_true: Ground truth affinity maps [B, 1, H, W] (optional during inference)
        
        Returns:
            Score with sigmoid activation [B, out_channels]
        """
        # Concatenate pred and gt affinity along channel dimension
        if label_true is not None:
            x = torch.cat([label_pred, label_true], dim=1)
        else:
            # During inference, if no GT available, use zeros
            x = torch.cat([label_pred, torch.zeros_like(label_pred)], dim=1)
        
        # ResNet backbone
        out = self.backbone(x)
        
        # Sigmoid
        out = self.activate(out)
        
        return out
