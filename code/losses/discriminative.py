"""
Discriminative Loss for Instance Segmentation.

Based on "Semantic Instance Segmentation with a Discriminative Loss Function"
by De Brabandere et al. (2017)

The loss encourages:
- L_var: Embeddings of same instance to be close (within delta_var)
- L_dist: Embeddings of different instances to be far apart (beyond delta_dist)
- L_reg: Embeddings to stay close to origin (regularization)
"""

import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange, repeat


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for learning pixel embeddings.
    
    The loss consists of three terms:
    1. Variance term (L_var): Pull embeddings of same instance together
    2. Distance term (L_dist): Push different instance centers apart
    3. Regularization term (L_reg): Keep instance centers near origin
    
    Args:
        delta_var: Margin for variance term (embeddings within same instance)
        delta_dist: Margin for distance term (between different instances)
        norm: Norm type (1 or 2)
        alpha: Weight for variance term
        beta: Weight for distance term
        gamma: Weight for regularization term
        reduction: 'mean' or 'sum' reduction over batch
    """
    
    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        norm: int = 2,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute discriminative loss.
        
        Args:
            embedding: Pixel embeddings [B, E, H, W] where E is embedding dim
            instance_mask: Instance segmentation labels [B, 1, H, W] or [B, H, W]
                          Each unique value > 0 is a different instance, 0 is background
        
        Returns:
            Tuple of (total_loss, L_var, L_dist, L_reg)
        """
        batch_size = embedding.shape[0]
        
        # Ensure instance_mask has correct shape [B, H, W]
        if instance_mask.dim() == 4:
            instance_mask = rearrange(instance_mask, 'b 1 h w -> b h w')
        
        # Rearrange embedding: [B, E, H, W] -> [B, E, H*W] for easier indexing
        emb_flat = rearrange(embedding, 'b e h w -> b e (h w)')
        mask_flat = rearrange(instance_mask, 'b h w -> b (h w)')
        
        # Initialize loss accumulators
        loss_var_total = torch.tensor(0.0, device=embedding.device)
        loss_dist_total = torch.tensor(0.0, device=embedding.device)
        loss_reg_total = torch.tensor(0.0, device=embedding.device)
        
        valid_batches = 0
        
        for b in range(batch_size):
            emb = emb_flat[b]  # [E, N] where N = H*W
            mask = mask_flat[b]  # [N]
            
            # Get unique instance IDs (excluding background 0)
            instance_ids = torch.unique(mask)
            instance_ids = instance_ids[instance_ids > 0]
            
            n_instances = len(instance_ids)
            
            if n_instances == 0:
                # No instances in this sample, skip
                continue
            
            valid_batches += 1
            
            # Compute instance centers and variance loss
            centers = []
            loss_var = torch.tensor(0.0, device=embedding.device)
            
            for inst_id in instance_ids:
                # Get mask for this instance
                inst_mask = (mask == inst_id)
                n_pixels = inst_mask.sum().float()
                
                if n_pixels == 0:
                    continue
                
                # Get embeddings for this instance: [E, N_inst]
                inst_embeddings = emb[:, inst_mask]
                
                # Compute center (mean embedding): [E]
                center = inst_embeddings.mean(dim=1)
                centers.append(center)
                
                # Variance loss: pull embeddings toward center
                # Broadcast center: [E] -> [E, 1] for subtraction
                center_broadcast = rearrange(center, 'e -> e 1')
                distances = torch.norm(inst_embeddings - center_broadcast, p=self.norm, dim=0)
                hinged = torch.clamp(distances - self.delta_var, min=0.0)
                loss_var = loss_var + (hinged ** 2).mean()
            
            if len(centers) > 0:
                loss_var = loss_var / len(centers)
                loss_var_total = loss_var_total + loss_var
            
            # Distance loss: push instance centers apart
            loss_dist = torch.tensor(0.0, device=embedding.device)
            
            if n_instances > 1:
                # Stack centers: [N_inst, E]
                centers_tensor = torch.stack(centers)
                n_pairs = 0
                
                for i in range(n_instances):
                    for j in range(i + 1, n_instances):
                        dist = torch.norm(centers_tensor[i] - centers_tensor[j], p=self.norm)
                        # 2 * delta_dist - ||mu_a - mu_b||_p, clamp to >= 0
                        hinged = torch.clamp(2 * self.delta_dist - dist, min=0.0)
                        loss_dist = loss_dist + hinged ** 2
                        n_pairs += 1
                
                if n_pairs > 0:
                    loss_dist = loss_dist / n_pairs
            
            loss_dist_total = loss_dist_total + loss_dist
            
            # Regularization loss: keep centers near origin
            loss_reg = torch.tensor(0.0, device=embedding.device)
            if len(centers) > 0:
                # Stack centers: [N_inst, E]
                centers_tensor = torch.stack(centers)
                loss_reg = torch.norm(centers_tensor, p=self.norm, dim=1).mean()
            
            loss_reg_total = loss_reg_total + loss_reg
        
        # Average over valid batches
        if valid_batches > 0:
            loss_var_total = loss_var_total / valid_batches
            loss_dist_total = loss_dist_total / valid_batches
            loss_reg_total = loss_reg_total / valid_batches
        
        # Combine losses
        total_loss = (
            self.alpha * loss_var_total +
            self.beta * loss_dist_total +
            self.gamma * loss_reg_total
        )
        
        return total_loss, loss_var_total, loss_dist_total, loss_reg_total
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"delta_var={self.delta_var}, "
            f"delta_dist={self.delta_dist}, "
            f"norm={self.norm}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"gamma={self.gamma})"
        )
