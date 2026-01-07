import torch
import torch.nn as nn
from typing import Dict, Any, List
from monai.networks.nets import SegResNet

class Ariel2D(nn.Module):
    def __init__(self, 
        model_config: Dict[str, Any], 
        label_config: Dict[str, int]
    ):
        """
        Args:
            model_config: Configuration for the network
            label_config: Registry for the labels
        """
        super().__init__()
        # Initialize Backbone
        # We use SegResNet as the encoder, mimicking VISTA2D structure
        # Adapted to output feature maps rather than final logits immediately
        self.backbone = SegResNet(
            spatial_dims=2,
            init_filters=model_config.get('init_filters', 32),
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('feature_dim', 32), # Intermediate feature size
            dropout_prob=model_config.get('dropout', 0.2),
            blocks_down=model_config.get('blocks_down', ),
            blocks_up=model_config.get('blocks_up', )
        )
        
         # Head 1: Semantic Segmentation (Mitochondria, Membrane, Synapse)
        # We output C channels where C is number of semantic classes
        self.sem_classes = [k for k in label_config.keys() if k!= 'neuron' and k!= 'background']
        self.sem_head = nn.Conv2d(
            in_channels=model_config.get('feature_dim', 32),
            out_channels=len(self.sem_classes) + 1, # +1 for background
            kernel_size=1
        )
        
        # Head 2: Instance Embedding (Neurons)
        self.emb_dim = model_config.get('emb_dim', 16)
        self.ins_head = nn.Conv2d(
            in_channels=model_config.get('feature_dim', 32),
            out_channels=self.emb_dim,
            kernel_size=1
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x