# Datasets loader implementation
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import h5py
import tifffile

from monai.data import CacheDataset
from monai.transforms import LoadImaged, Randomizable

__all__ = [
    "SNEMIDataset", 
    "CREMIDataset", 
    "MiCRONSDataset",
]

class SNEMIDataset(CacheDataset, Randomizable):
    """
    SNEMI3D Dataset for neuron segmentation.
    
    Dataset Structure:
    - Resolution: 6x6x30 nm
    - Train: AC4 (100 slices, 6x6x3 um) from Kasthuri dataset
    - Test: AC3 (100 slices, 6x6x3 um) from Kasthuri dataset
    
    Note: The original SNEMI3D description mentions a validation set from
    "last 156 slices of AC3", but the current AC3 volume only has 100 slices.
    For validation, you can use a subset of the training data or download
    additional data from the SNEMI3D challenge website.
    
    Args:
        root_dir: Root directory containing the data files
        split: One of 'train' or 'test' ('valid' uses subset of AC4)
        transform: Optional transforms to apply to the data
        cache_rate: Fraction of dataset to cache in memory (default: 1.0)
        train_val_split: For 'valid' split, fraction of AC4 to use for validation (default: 0.2)
    
    Expected file structure:
        root_dir/
            AC3_inputs.h5 or AC3_inputs.tiff  (100 slices)
            AC3_labels.h5 or AC3_labels.tiff  (100 slices, not used for test)
            AC4_inputs.h5 or AC4_inputs.tiff  (100 slices)
            AC4_labels.h5 or AC4_labels.tiff  (100 slices)
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
    ):
        self.root_dir = root_dir
        self.split = split.lower()
        self.train_val_split = train_val_split
        
        if self.split not in ["train", "valid", "test"]:
            raise ValueError(f"split must be 'train', 'valid', or 'test', got {split}")
        
        # Prepare data list
        data_dicts = self._prepare_data()
        
        # Initialize parent CacheDataset
        super().__init__(
            data=data_dicts,
            transform=transform,
            cache_rate=cache_rate,
        )
    
    def _load_volume(self, filename: str) -> np.ndarray:
        """Load volume from either .h5 or .tiff file."""
        filepath = os.path.join(self.root_dir, filename)
        
        if filename.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                return f['main'][:]
        elif filename.endswith('.tiff') or filename.endswith('.tif'):
            return tifffile.imread(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def _prepare_data(self) -> List[Dict[str, np.ndarray]]:
        """Prepare data dictionaries based on split."""
        data_list = []
        
        if self.split == "train":
            # Train: AC4 (100 slices) - use first (1-train_val_split) fraction
            # Try .tiff first, fallback to .h5
            if os.path.exists(os.path.join(self.root_dir, 'AC4_inputs.tiff')):
                inputs = self._load_volume('AC4_inputs.tiff')
                labels = self._load_volume('AC4_labels.tiff')
            else:
                inputs = self._load_volume('AC4_inputs.h5')
                labels = self._load_volume('AC4_labels.h5')
            
            # Calculate split point
            n_total = inputs.shape[0]
            n_train = int(n_total * (1.0 - self.train_val_split))
            
            # Create a data dict for each training slice
            for i in range(n_train):
                data_list.append({
                    "image": inputs[i],  # Shape: (H, W)
                    "label": labels[i],  # Shape: (H, W)
                    "slice_idx": i,
                    "volume": "AC4"
                })
        
        elif self.split == "test":
            # Test: AC3 (100 slices)
            if os.path.exists(os.path.join(self.root_dir, 'AC3_inputs.tiff')):
                inputs = self._load_volume('AC3_inputs.tiff')
            else:
                inputs = self._load_volume('AC3_inputs.h5')
            
            # Test set has no labels
            for i in range(inputs.shape[0]):
                data_list.append({
                    "image": inputs[i],  # Shape: (H, W)
                    "slice_idx": i,
                    "volume": "AC3"
                })
        
        elif self.split == "valid":
            # Valid: AC4 (100 slices) - use last train_val_split fraction
            if os.path.exists(os.path.join(self.root_dir, 'AC4_inputs.tiff')):
                inputs = self._load_volume('AC4_inputs.tiff')
                labels = self._load_volume('AC4_labels.tiff')
            else:
                inputs = self._load_volume('AC4_inputs.h5')
                labels = self._load_volume('AC4_labels.h5')
            
            # Calculate split point
            n_total = inputs.shape[0]
            n_train = int(n_total * (1.0 - self.train_val_split))
            
            # Use slices from n_train onwards for validation
            for i in range(n_train, n_total):
                data_list.append({
                    "image": inputs[i],  # Shape: (H, W)
                    "label": labels[i],  # Shape: (H, W)
                    "slice_idx": i,
                    "volume": "AC4_valid"
                })
        
        return data_list

class CREMIDataset(CacheDataset, Randomizable):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        super().__init__(root_dir, transform=transform)