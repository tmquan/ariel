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
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from einops import rearrange, reduce, repeat


class DiscriminativeLoss(nn.Module):
    """
    Discriminative loss for learning pixel embeddings.
    
    The loss consists of three terms:
    1. Variance term (L_var): Pull embeddings of same instance together
    2. Distance term (L_dist): Push different instance centers apart
    3. Regularization term (L_reg): Keep instance centers near origin
    
    Supports both 2D and 3D inputs:
    - 2D: embedding [B, E, H, W], instance_mask [B, 1, H, W] or [B, H, W]
    - 3D: embedding [B, E, D, H, W], instance_mask [B, 1, D, H, W] or [B, D, H, W]
    
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
    
    def _flatten_spatial(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Flatten spatial dimensions for both 2D and 3D inputs.
        
        Args:
            embedding: [B, E, H, W] for 2D or [B, E, D, H, W] for 3D
            instance_mask: [B, 1, H, W] / [B, H, W] for 2D or [B, 1, D, H, W] / [B, D, H, W] for 3D
        
        Returns:
            Tuple of (emb_flat [B, E, N], mask_flat [B, N], is_3d)
        """
        is_3d = embedding.dim() == 5
        
        if is_3d:
            # 3D: [B, E, D, H, W] -> [B, E, N] where N = D*H*W
            emb_flat = rearrange(embedding, 'b e d h w -> b e (d h w)')
            
            # Handle instance_mask shape
            if instance_mask.dim() == 5:
                mask_flat = rearrange(instance_mask, 'b 1 d h w -> b (d h w)')
            else:
                mask_flat = rearrange(instance_mask, 'b d h w -> b (d h w)')
        else:
            # 2D: [B, E, H, W] -> [B, E, N] where N = H*W
            emb_flat = rearrange(embedding, 'b e h w -> b e (h w)')
            
            # Handle instance_mask shape
            if instance_mask.dim() == 4:
                mask_flat = rearrange(instance_mask, 'b 1 h w -> b (h w)')
            else:
                mask_flat = rearrange(instance_mask, 'b h w -> b (h w)')
        
        return emb_flat, mask_flat, is_3d
    
    def _compute_cluster_means(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cluster centers using loop-based approach.
        
        Args:
            emb: Flattened embeddings [E, N]
            inst: Flattened instance mask [N]
            unique_instances: Unique instance IDs (excluding background)
        
        Returns:
            Cluster centers [C, E] where C is number of instances
        """
        centers = []
        
        for inst_id in unique_instances:
            inst_mask = (inst == inst_id)
            if inst_mask.sum() == 0:
                continue
            
            # Get embeddings for this instance: [E, N_inst]
            inst_embeddings = emb[:, inst_mask]
            
            # Compute center (mean embedding): [E]
            center = inst_embeddings.mean(dim=1)
            centers.append(center)
        
        if len(centers) == 0:
            return torch.zeros((0, emb.shape[0]), device=emb.device)
        
        return torch.stack(centers)  # [C, E]
    
    def _variance_loss(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute variance loss: pull embeddings toward their cluster center.
        
        Args:
            emb: Flattened embeddings [E, N]
            inst: Flattened instance mask [N]
            unique_instances: Unique instance IDs
            cluster_centers: Cluster centers [C, E]
        
        Returns:
            Variance loss (scalar)
        """
        num_instances = len(unique_instances)
        
        if num_instances == 0:
            return torch.tensor(0.0, device=emb.device)
        
        loss_var = torch.tensor(0.0, device=emb.device)
        
        for idx, inst_id in enumerate(unique_instances):
            inst_mask = (inst == inst_id)
            if inst_mask.sum() == 0:
                continue
            
            # Get embeddings for this instance: [E, N_inst]
            inst_embeddings = emb[:, inst_mask]
            
            # Get center for this instance
            center = cluster_centers[idx]
            
            # Broadcast center: [E] -> [E, 1]
            center_broadcast = rearrange(center, 'e -> e 1')
            
            # Compute distances from center
            distances = torch.norm(inst_embeddings - center_broadcast, p=self.norm, dim=0)
            
            # Hinge loss: only penalize if distance > delta_var
            hinged = F.relu(distances - self.delta_var) ** 2
            loss_var = loss_var + hinged.mean()
        
        # Average over instances
        return loss_var / num_instances
    
    def _distance_loss(
        self,
        cluster_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance loss: push different instance centers apart.
        
        Args:
            cluster_centers: Cluster centers [C, E]
        
        Returns:
            Distance loss (scalar)
        """
        num_instances = cluster_centers.shape[0]
        
        if num_instances <= 1:
            return torch.tensor(0.0, device=cluster_centers.device)
        
        loss_dist = torch.tensor(0.0, device=cluster_centers.device)
        n_pairs = 0
        
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                dist = torch.norm(cluster_centers[i] - cluster_centers[j], p=self.norm)
                # Hinge loss: penalize if centers are too close
                hinged = F.relu(2 * self.delta_dist - dist) ** 2
                loss_dist = loss_dist + hinged
                n_pairs += 1
        
        if n_pairs > 0:
            loss_dist = loss_dist / n_pairs
        
        return loss_dist
    
    def _regularization_loss(
        self,
        cluster_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regularization loss: keep centers near origin.
        
        Args:
            cluster_centers: Cluster centers [C, E]
        
        Returns:
            Regularization loss (scalar)
        """
        if cluster_centers.shape[0] == 0:
            return torch.tensor(0.0, device=cluster_centers.device)
        
        norms = torch.norm(cluster_centers, p=self.norm, dim=1)
        return norms.mean()
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute discriminative loss.
        
        Args:
            embedding: Pixel embeddings [B, E, H, W] (2D) or [B, E, D, H, W] (3D)
            instance_mask: Instance labels [B, 1, H, W] / [B, H, W] (2D) 
                          or [B, 1, D, H, W] / [B, D, H, W] (3D)
                          Each unique value > 0 is a different instance, 0 is background
        
        Returns:
            Tuple of (total_loss, L_var, L_dist, L_reg)
        """
        batch_size = embedding.shape[0]
        
        # Flatten spatial dimensions (handles both 2D and 3D)
        emb_flat, mask_flat, is_3d = self._flatten_spatial(embedding, instance_mask)
        
        # Initialize loss accumulators
        loss_var_total = torch.tensor(0.0, device=embedding.device)
        loss_dist_total = torch.tensor(0.0, device=embedding.device)
        loss_reg_total = torch.tensor(0.0, device=embedding.device)
        
        valid_batches = 0
        
        for b in range(batch_size):
            emb = emb_flat[b]  # [E, N]
            inst = mask_flat[b]  # [N]
            
            # Get unique instance IDs (excluding background 0)
            unique_instances = torch.unique(inst)
            unique_instances = unique_instances[unique_instances > 0]
            
            if len(unique_instances) == 0:
                continue
            
            valid_batches += 1
            
            # Compute cluster centers
            cluster_centers = self._compute_cluster_means(emb, inst, unique_instances)
            
            # Compute individual losses
            loss_var = self._variance_loss(emb, inst, unique_instances, cluster_centers)
            loss_dist = self._distance_loss(cluster_centers)
            loss_reg = self._regularization_loss(cluster_centers)
            
            loss_var_total = loss_var_total + loss_var
            loss_dist_total = loss_dist_total + loss_dist
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


class DiscriminativeLossVectorized(DiscriminativeLoss):
    """
    Vectorized implementation using einops for better GPU efficiency.
    
    Avoids explicit Python loops where possible using scatter operations
    and einops rearrangements.
    """
    
    def _compute_cluster_means(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor
    ) -> torch.Tensor:
        """Compute cluster means using scatter operations."""
        num_instances = len(unique_instances)
        emb_dim = emb.shape[0]
        
        # Create instance-to-index mapping
        max_inst = int(inst.max().item()) + 1
        inst_to_idx = torch.full((max_inst,), -1, device=emb.device, dtype=torch.long)
        
        for idx, inst_id in enumerate(unique_instances):
            inst_to_idx[inst_id.long()] = idx
        
        # Map each voxel to cluster index
        cluster_indices = inst_to_idx[inst.long()]  # (N,)
        valid_mask = cluster_indices >= 0
        
        if not valid_mask.any():
            return torch.zeros((num_instances, emb_dim), device=emb.device)
        
        valid_emb = emb[:, valid_mask]  # (E, N_valid)
        valid_idx = cluster_indices[valid_mask]  # (N_valid,)
        
        # Compute cluster sums and counts
        cluster_sums = torch.zeros((num_instances, emb_dim), device=emb.device)
        cluster_counts = torch.zeros(num_instances, device=emb.device)
        
        # Scatter add for each embedding dimension
        valid_emb_t = rearrange(valid_emb, 'e n -> n e')
        for e in range(emb_dim):
            cluster_sums[:, e].scatter_add_(0, valid_idx, valid_emb_t[:, e])
        cluster_counts.scatter_add_(0, valid_idx, torch.ones_like(valid_idx, dtype=torch.float))
        
        # Compute means
        cluster_counts = torch.clamp(cluster_counts, min=1)
        cluster_means = cluster_sums / rearrange(cluster_counts, 'c -> c 1')
        
        return cluster_means
    
    def _variance_loss(
        self,
        emb: torch.Tensor,
        inst: torch.Tensor,
        unique_instances: torch.Tensor,
        cluster_means: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized variance loss using scatter operations."""
        num_instances = len(unique_instances)
        
        if num_instances == 0:
            return torch.tensor(0.0, device=emb.device)
        
        # Create instance-to-index mapping
        max_inst = int(inst.max().item()) + 1
        inst_to_idx = torch.full((max_inst,), -1, device=emb.device, dtype=torch.long)
        
        for idx, inst_id in enumerate(unique_instances):
            inst_to_idx[inst_id.long()] = idx
        
        # Map each voxel to cluster index
        cluster_indices = inst_to_idx[inst.long()]  # (N,)
        valid_mask = cluster_indices >= 0
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=emb.device)
        
        # Get valid embeddings and indices
        valid_emb = emb[:, valid_mask]  # (E, N_valid)
        valid_idx = cluster_indices[valid_mask]  # (N_valid,)
        
        # Gather cluster means for each voxel
        # cluster_means: (C, E) -> index by valid_idx
        gathered_means = cluster_means[valid_idx]  # (N_valid, E)
        gathered_means = rearrange(gathered_means, 'n e -> e n')
        
        # Compute distances
        diff = valid_emb - gathered_means  # (E, N_valid)
        distances = torch.norm(diff, p=self.norm, dim=0)  # (N_valid,)
        
        # Hinge loss
        hinged = F.relu(distances - self.delta_var) ** 2
        
        # Compute per-cluster mean using scatter
        cluster_losses = torch.zeros(num_instances, device=emb.device)
        cluster_counts = torch.zeros(num_instances, device=emb.device)
        
        cluster_losses.scatter_add_(0, valid_idx, hinged)
        cluster_counts.scatter_add_(0, valid_idx, torch.ones_like(hinged))
        
        # Avoid division by zero
        cluster_counts = torch.clamp(cluster_counts, min=1)
        
        # Mean per cluster, then mean across clusters
        per_cluster_loss = cluster_losses / cluster_counts
        var_loss = reduce(per_cluster_loss, 'c -> ', 'mean')
        
        return var_loss
    
    def _distance_loss(
        self,
        cluster_means: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized distance loss between cluster centers."""
        num_instances = cluster_means.shape[0]
        
        if num_instances <= 1:
            return torch.tensor(0.0, device=cluster_means.device)
        
        # Pairwise distances using broadcasting
        # cluster_means: (C, E)
        means_i = rearrange(cluster_means, 'c e -> c 1 e')
        means_j = rearrange(cluster_means, 'c e -> 1 c e')
        
        # Pairwise difference: (C, C, E)
        diff = means_i - means_j
        pairwise_dist = torch.norm(diff, p=self.norm, dim=2)  # (C, C)
        
        # Get upper triangle (avoid diagonal and double counting)
        triu_indices = torch.triu_indices(num_instances, num_instances, offset=1, device=cluster_means.device)
        upper_dists = pairwise_dist[triu_indices[0], triu_indices[1]]
        
        # Hinge loss: push apart beyond 2 * delta_dist
        hinged = F.relu(2 * self.delta_dist - upper_dists) ** 2
        
        dist_loss = reduce(hinged, 'n -> ', 'mean')
        
        return dist_loss
    
    def _regularization_loss(
        self,
        cluster_means: torch.Tensor
    ) -> torch.Tensor:
        """Regularization loss to keep centers near origin."""
        if cluster_means.shape[0] == 0:
            return torch.tensor(0.0, device=cluster_means.device)
        
        norms = torch.norm(cluster_means, p=self.norm, dim=1)
        reg_loss = reduce(norms, 'c -> ', 'mean')
        
        return reg_loss
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute discriminative loss using vectorized operations.
        
        Args:
            embedding: Pixel embeddings [B, E, H, W] (2D) or [B, E, D, H, W] (3D)
            instance_mask: Instance labels [B, 1, H, W] / [B, H, W] (2D) 
                          or [B, 1, D, H, W] / [B, D, H, W] (3D)
        
        Returns:
            Tuple of (total_loss, L_var, L_dist, L_reg)
        """
        batch_size = embedding.shape[0]
        
        # Flatten spatial dimensions (handles both 2D and 3D)
        emb_flat, mask_flat, is_3d = self._flatten_spatial(embedding, instance_mask)
        
        # Initialize loss accumulators
        loss_var_total = torch.tensor(0.0, device=embedding.device)
        loss_dist_total = torch.tensor(0.0, device=embedding.device)
        loss_reg_total = torch.tensor(0.0, device=embedding.device)
        
        valid_batches = 0
        
        for b in range(batch_size):
            emb = emb_flat[b]  # [E, N]
            inst = mask_flat[b]  # [N]
            
            # Get unique instances (excluding background)
            unique_instances = torch.unique(inst)
            unique_instances = unique_instances[unique_instances > 0]
            
            if len(unique_instances) == 0:
                continue
            
            valid_batches += 1
            
            # Compute cluster means
            cluster_means = self._compute_cluster_means(emb, inst, unique_instances)
            
            # Compute losses
            loss_var = self._variance_loss(emb, inst, unique_instances, cluster_means)
            loss_dist = self._distance_loss(cluster_means)
            loss_reg = self._regularization_loss(cluster_means)
            
            loss_var_total = loss_var_total + loss_var
            loss_dist_total = loss_dist_total + loss_dist
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


class CombinedInstanceLoss(nn.Module):
    """
    Combined loss for instance segmentation with optional boundary loss.
    
    Uses einops for tensor operations.
    """
    
    def __init__(
        self,
        discriminative_config: Dict,
        use_boundary_loss: bool = False,
        boundary_weight: float = 0.5,
        vectorized: bool = True
    ):
        super().__init__()
        if vectorized:
            self.disc_loss = DiscriminativeLossVectorized(**discriminative_config)
        else:
            self.disc_loss = DiscriminativeLoss(**discriminative_config)
        self.use_boundary_loss = use_boundary_loss
        self.boundary_weight = boundary_weight
        
        if use_boundary_loss:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor,
        boundary_pred: Optional[torch.Tensor] = None,
        boundary_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            embedding: Pixel embeddings [B, E, H, W]
            instance_mask: Instance labels [B, 1, H, W] or [B, H, W]
            boundary_pred: Optional boundary predictions [B, 1, H, W]
            boundary_target: Optional boundary targets [B, 1, H, W]
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        disc_loss, loss_var, loss_dist, loss_reg = self.disc_loss(embedding, instance_mask)
        
        total_loss = disc_loss
        loss_dict = {
            'loss_disc': disc_loss,
            'loss_var': loss_var,
            'loss_dist': loss_dist,
            'loss_reg': loss_reg,
        }
        
        if self.use_boundary_loss and boundary_pred is not None and boundary_target is not None:
            boundary_loss = self.bce_loss(boundary_pred, boundary_target.float())
            total_loss = total_loss + self.boundary_weight * boundary_loss
            loss_dict['loss_boundary'] = boundary_loss
        
        loss_dict['loss_combined'] = total_loss
        
        return total_loss, loss_dict


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for embedding learning.
    
    Alternative to discriminative loss using pairwise comparisons
    with einops for efficient batch operations.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        num_samples: int = 1000
    ):
        super().__init__()
        self.margin = margin
        self.num_samples = num_samples
    
    def forward(
        self,
        embedding: torch.Tensor,
        instance_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contrastive loss via sampled pairs.
        
        Args:
            embedding: (B, E, H, W) for 2D or (B, E, D, H, W) for 3D
            instance_mask: (B, 1, H, W) for 2D or (B, 1, D, H, W) for 3D
        """
        batch_size = embedding.shape[0]
        device = embedding.device
        is_3d = embedding.dim() == 5
        
        total_pos_loss = torch.tensor(0.0, device=device)
        total_neg_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for b in range(batch_size):
            # Flatten spatial dims
            if is_3d:
                emb = rearrange(embedding[b], 'e d h w -> (d h w) e')  # (N, E)
                inst = rearrange(instance_mask[b, 0], 'd h w -> (d h w)')  # (N,)
            else:
                emb = rearrange(embedding[b], 'e h w -> (h w) e')  # (N, E)
                inst = rearrange(instance_mask[b, 0], 'h w -> (h w)')  # (N,)
            
            # Sample pairs
            n_voxels = emb.shape[0]
            if n_voxels < 2:
                continue
            
            # Random pair indices
            idx1 = torch.randint(0, n_voxels, (self.num_samples,), device=device)
            idx2 = torch.randint(0, n_voxels, (self.num_samples,), device=device)
            
            # Get embeddings and labels
            emb1 = emb[idx1]  # (S, E)
            emb2 = emb[idx2]  # (S, E)
            label1 = inst[idx1]
            label2 = inst[idx2]
            
            # Same instance (positive) or different (negative)
            same_instance = (label1 == label2) & (label1 != 0)
            diff_instance = (label1 != label2) & (label1 != 0) & (label2 != 0)
            
            # Pairwise distances
            diff = emb1 - emb2
            distances = torch.norm(diff, dim=1)
            
            # Positive loss: pull together
            if same_instance.any():
                pos_loss = reduce(distances[same_instance] ** 2, 'n -> ', 'mean')
                total_pos_loss = total_pos_loss + pos_loss
            
            # Negative loss: push apart with margin
            if diff_instance.any():
                neg_loss = reduce(
                    F.relu(self.margin - distances[diff_instance]) ** 2,
                    'n -> ', 'mean'
                )
                total_neg_loss = total_neg_loss + neg_loss
            
            valid_batches += 1
        
        if valid_batches > 0:
            total_pos_loss = total_pos_loss / valid_batches
            total_neg_loss = total_neg_loss / valid_batches
        
        total_loss = total_pos_loss + total_neg_loss
        
        return total_loss, {
            'loss_pos': total_pos_loss,
            'loss_neg': total_neg_loss,
            'loss_total': total_loss
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(margin={self.margin}, num_samples={self.num_samples})"
