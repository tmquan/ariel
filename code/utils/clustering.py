"""
Clustering utilities for converting pixel embeddings to instance segmentation.

Provides various clustering methods to group pixels based on their embeddings.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from einops import rearrange


def mean_shift_clustering(
    embedding: np.ndarray,
    bandwidth: float = 0.5,
    max_iter: int = 100,
    convergence_threshold: float = 1e-3,
    min_cluster_size: int = 50,
) -> np.ndarray:
    """
    Mean-shift clustering for pixel embeddings.
    
    Args:
        embedding: Pixel embeddings [N, E] where N is number of pixels
        bandwidth: Kernel bandwidth (related to delta_var in discriminative loss)
        max_iter: Maximum iterations for convergence
        convergence_threshold: Stop when shift is below this
        min_cluster_size: Minimum pixels per cluster
    
    Returns:
        Cluster labels [N] with 0 for background/unassigned
    """
    N, E = embedding.shape
    
    # Initialize: each point starts at its own position
    points = embedding.copy()
    labels = np.zeros(N, dtype=np.int32)
    
    # Mean-shift iterations
    for _ in range(max_iter):
        new_points = np.zeros_like(points)
        
        for i in range(N):
            # Compute distances to all points: [N]
            diff = embedding - points[i]  # [N, E]
            distances = np.linalg.norm(diff, axis=1)  # [N]
            
            # Gaussian kernel weights: [N]
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
            weights = weights / (weights.sum() + 1e-8)
            
            # Shift toward weighted mean: [E]
            new_points[i] = np.sum(embedding * weights[:, np.newaxis], axis=0)
        
        # Check convergence
        shift = np.linalg.norm(new_points - points, axis=1).mean()
        points = new_points
        
        if shift < convergence_threshold:
            break
    
    # Cluster assignment: group converged points
    cluster_centers = []
    cluster_id = 1
    
    for i in range(N):
        assigned = False
        for j, center in enumerate(cluster_centers):
            if np.linalg.norm(points[i] - center) < bandwidth:
                labels[i] = j + 1
                assigned = True
                break
        
        if not assigned:
            cluster_centers.append(points[i])
            labels[i] = cluster_id
            cluster_id += 1
    
    # Filter small clusters
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label > 0 and count < min_cluster_size:
            labels[labels == label] = 0
    
    # Relabel consecutively
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels], dtype=np.int32)
    
    return labels


def cluster_embeddings(
    embedding: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    method: str = 'meanshift',
    bandwidth: float = 0.5,
    min_cluster_size: int = 50,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Cluster pixel embeddings to obtain instance labels.
    
    Args:
        embedding: Pixel embeddings [E, H, W] or [B, E, H, W]
        foreground_mask: Binary mask [H, W] or [B, H, W], only cluster foreground pixels
        method: Clustering method ('meanshift', 'hdbscan', 'sklearn_meanshift')
        bandwidth: Bandwidth for mean-shift clustering
        min_cluster_size: Minimum pixels per cluster
        device: Device for output tensor
    
    Returns:
        Instance labels [H, W] or [B, H, W] with unique integers per instance
    """
    # Handle batch dimension
    if embedding.dim() == 4:
        batch_results = []
        for b in range(embedding.shape[0]):
            fg_mask = foreground_mask[b] if foreground_mask is not None else None
            result = cluster_embeddings(
                embedding[b], fg_mask, method, bandwidth, min_cluster_size, device
            )
            batch_results.append(result)
        return torch.stack(batch_results)
    
    # Single image: [E, H, W]
    E, H, W = embedding.shape
    
    # Rearrange to [H*W, E] for clustering (pixels as samples, embedding as features)
    emb_np = embedding.detach().cpu().numpy()
    emb_flat = rearrange(emb_np, 'e h w -> (h w) e')  # [H*W, E]
    
    # Apply foreground mask if provided
    if foreground_mask is not None:
        # Flatten mask: [H, W] -> [H*W]
        fg_mask_np = rearrange(foreground_mask.detach().cpu().numpy(), 'h w -> (h w)')
        fg_indices = np.where(fg_mask_np > 0)[0]
        
        if len(fg_indices) == 0:
            # No foreground pixels
            return torch.zeros(H, W, dtype=torch.long, device=device)
        
        emb_fg = emb_flat[fg_indices]  # [N_fg, E]
    else:
        fg_indices = np.arange(H * W)
        emb_fg = emb_flat
    
    # Perform clustering
    if method == 'meanshift':
        labels_fg = mean_shift_clustering(
            emb_fg,
            bandwidth=bandwidth,
            min_cluster_size=min_cluster_size,
        )
    elif method == 'sklearn_meanshift':
        try:
            from sklearn.cluster import MeanShift
            clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            labels_fg = clusterer.fit_predict(emb_fg) + 1  # +1 to reserve 0 for background
        except ImportError:
            # Fallback to custom implementation
            labels_fg = mean_shift_clustering(emb_fg, bandwidth=bandwidth)
    elif method == 'hdbscan':
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels_fg = clusterer.fit_predict(emb_fg) + 1  # -1 becomes 0 (background)
            labels_fg = np.maximum(labels_fg, 0)
        except ImportError:
            # Fallback to mean-shift
            labels_fg = mean_shift_clustering(emb_fg, bandwidth=bandwidth)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Create full label map: [H*W]
    labels_flat = np.zeros(H * W, dtype=np.int32)
    labels_flat[fg_indices] = labels_fg
    
    # Reshape back to [H, W]
    labels = labels_flat.reshape(H, W)
    
    return torch.from_numpy(labels).to(device=device, dtype=torch.long)


def compute_instance_metrics(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute instance segmentation metrics.
    
    Args:
        pred_labels: Predicted instance labels [H, W] or [B, H, W]
        true_labels: Ground truth instance labels [H, W] or [B, H, W]
    
    Returns:
        Tuple of (adjusted_rand_score, adjusted_mutual_info_score)
        Both values are clamped to [0, 1] range.
    """
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    
    # Handle batch dimension
    if pred_labels.dim() == 3:
        ari_sum, ami_sum = 0.0, 0.0
        batch_size = pred_labels.shape[0]
        valid_count = 0
        
        for b in range(batch_size):
            # Flatten: [H, W] -> [H*W]
            pred = rearrange(pred_labels[b].detach().cpu(), 'h w -> (h w)').numpy()
            true = rearrange(true_labels[b].detach().cpu(), 'h w -> (h w)').numpy()
            
            # Skip if all background
            if np.all(true == 0) or np.all(pred == 0):
                continue
            
            # Clamp to [0, 1] range
            ari_sum += max(0.0, adjusted_rand_score(true, pred))
            ami_sum += max(0.0, adjusted_mutual_info_score(true, pred))
            valid_count += 1
        
        if valid_count == 0:
            return 0.0, 0.0
        
        return ari_sum / valid_count, ami_sum / valid_count
    
    # Single image: flatten [H, W] -> [H*W]
    pred = rearrange(pred_labels.detach().cpu(), 'h w -> (h w)').numpy()
    true = rearrange(true_labels.detach().cpu(), 'h w -> (h w)').numpy()
    
    if np.all(true == 0) or np.all(pred == 0):
        return 0.0, 0.0
    
    # Clamp to [0, 1] range
    ari = max(0.0, adjusted_rand_score(true, pred))
    ami = max(0.0, adjusted_mutual_info_score(true, pred))
    
    return ari, ami
