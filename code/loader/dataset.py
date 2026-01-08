# Datasets loader implementation
"""
Dataset classes for paired image-label data.

This module provides:
- PairedDataset: Base class for image-label pair datasets
- Specific implementations: CVPPP14, CVPPP15, CVPPP17, CVPPPRL, SNEMI
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import h5py
import tifffile
from PIL import Image

from monai.data import CacheDataset
from monai.transforms import Randomizable


__all__ = [
    "PairedDataset",
    "CVPPP14Dataset",
    "CVPPP15Dataset", 
    "CVPPP17Dataset",
    "CVPPPRLDataset",
    "SNEMIDataset", 
]


class PairedDataset(CacheDataset, Randomizable):
    """
    Base class for paired image-label datasets.
    
    Provides common functionality for loading image-label pairs from a directory.
    Subclasses should implement _get_file_list() to define dataset-specific file discovery.
    
    Args:
        root_dir: Root directory containing the data
        split: Data split ('train', 'valid', 'test')
        transform: Optional transforms to apply
        cache_rate: Fraction of data to cache in memory
        train_val_split: Fraction for validation split
    """
    
    def __init__(
        self, 
        root_dir: str, 
        split: str = "train",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.train_val_split = train_val_split
        
        if self.split not in ["train", "valid", "test"]:
            raise ValueError(f"split must be 'train', 'valid', or 'test', got {split}")
        
        # Get file list and prepare data
        data_dicts = self._prepare_data()
        
        # Initialize parent CacheDataset
        super().__init__(
            data=data_dicts,
            transform=transform,
            cache_rate=cache_rate,
        )
    
    @abstractmethod
    def _get_file_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get list of (image_path, label_path) tuples.
        
        Returns:
            List of tuples: (image_path, label_path) where label_path can be None
        """
        raise NotImplementedError
    
    def _load_image(self, filepath: str) -> np.ndarray:
        """Load image from file."""
        filepath = str(filepath)
        
        if filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                return f['main'][:]
        elif filepath.endswith(('.tiff', '.tif')):
            return tifffile.imread(filepath)
        elif filepath.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(filepath)
            return np.array(img)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    
    def _prepare_data(self) -> List[Dict]:
        """Prepare data dictionaries based on split."""
        file_list = self._get_file_list()
        
        # For train/valid, filter to only include samples WITH labels
        if self.split in ("train", "valid"):
            file_list = [(img, lbl) for img, lbl in file_list 
                         if lbl is not None and os.path.exists(lbl)]
        
        # Apply train/val split
        n_total = len(file_list)
        n_train = int(n_total * (1.0 - self.train_val_split))
        
        if self.split == "train":
            file_list = file_list[:n_train]
        elif self.split == "valid":
            file_list = file_list[n_train:]
        # test uses all files (no split applied in _get_file_list)
        
        data_list = []
        for idx, (img_path, label_path) in enumerate(file_list):
            data_dict = {
                "image": self._load_image(img_path),
                "image_path": str(img_path),
                "idx": idx,
            }
            
            if label_path is not None and os.path.exists(label_path):
                data_dict["label"] = self._load_image(label_path)
                data_dict["label_path"] = str(label_path)
            
            data_list.append(data_dict)
        
        return data_list


class CVPPP14Dataset(PairedDataset):
    """
    CVPPP 2014 Dataset for plant leaf segmentation.
    
    Structure:
        root_dir/
            images/    # A1_plant001.png, A2_plant001.png, ...
            labels/    # Instance segmentation labels
    
    Args:
        root_dir: Path to CVPPP14 directory
        split: 'train', 'valid', or 'test'
        transform: Optional transforms
        cache_rate: Cache fraction (default: 1.0)
        train_val_split: Validation fraction (default: 0.2)
    """
    
    def _get_file_list(self) -> List[Tuple[str, Optional[str]]]:
        images_dir = self.root_dir / "images"
        labels_dir = self.root_dir / "labels"
        
        file_list = []
        for img_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / img_path.name
            file_list.append((str(img_path), str(label_path) if label_path.exists() else None))
        
        return file_list


class CVPPP15Dataset(PairedDataset):
    """
    CVPPP 2015 Dataset for plant leaf segmentation.
    
    Includes both LCC (Leaf Counting Challenge) and LSC (Leaf Segmentation Challenge) data.
    
    Args:
        root_dir: Path to CVPPP15 directory
        split: 'train', 'valid', or 'test'
        challenge: 'all', 'LCC', or 'LSC' (default: 'all')
        transform: Optional transforms
        cache_rate: Cache fraction (default: 1.0)
        train_val_split: Validation fraction (default: 0.2)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        challenge: str = "all",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
    ):
        self.challenge = challenge.upper()
        super().__init__(root_dir, split, transform, cache_rate, train_val_split)
    
    def _get_file_list(self) -> List[Tuple[str, Optional[str]]]:
        images_dir = self.root_dir / "images"
        labels_dir = self.root_dir / "labels"
        
        file_list = []
        for img_path in sorted(images_dir.glob("*.png")):
            # Filter by challenge type if specified
            if self.challenge != "ALL":
                if not img_path.name.startswith(self.challenge):
                    continue
            
            label_path = labels_dir / img_path.name
            file_list.append((str(img_path), str(label_path) if label_path.exists() else None))
        
        return file_list


class CVPPP17Dataset(PairedDataset):
    """
    CVPPP 2017 Dataset for plant leaf segmentation.
    
    Includes both LCC and LSC challenge data.
    
    Args:
        root_dir: Path to CVPPP17 directory
        split: 'train', 'valid', or 'test'
        challenge: 'all', 'LCC', or 'LSC' (default: 'all')
        transform: Optional transforms
        cache_rate: Cache fraction (default: 1.0)
        train_val_split: Validation fraction (default: 0.2)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        challenge: str = "all",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
    ):
        self.challenge = challenge.upper()
        super().__init__(root_dir, split, transform, cache_rate, train_val_split)
    
    def _get_file_list(self) -> List[Tuple[str, Optional[str]]]:
        images_dir = self.root_dir / "images"
        labels_dir = self.root_dir / "labels"
        
        file_list = []
        for img_path in sorted(images_dir.glob("*.png")):
            # Filter by challenge type if specified
            if self.challenge != "ALL":
                if not img_path.name.startswith(self.challenge):
                    continue
            
            label_path = labels_dir / img_path.name
            file_list.append((str(img_path), str(label_path) if label_path.exists() else None))
        
        return file_list


class CVPPPRLDataset(PairedDataset):
    """
    Plant Phenotyping (PRL) Dataset.
    
    Contains Plant and Tray subsets for leaf segmentation.
    
    Args:
        root_dir: Path to PPP_PRL directory
        split: 'train', 'valid', or 'test'
        subset: 'all', 'Plant', or 'Tray' (default: 'all')
        transform: Optional transforms
        cache_rate: Cache fraction (default: 1.0)
        train_val_split: Validation fraction (default: 0.2)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        subset: str = "all",
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        train_val_split: float = 0.2,
    ):
        self.subset = subset
        super().__init__(root_dir, split, transform, cache_rate, train_val_split)
    
    def _get_file_list(self) -> List[Tuple[str, Optional[str]]]:
        images_dir = self.root_dir / "images"
        labels_dir = self.root_dir / "labels"
        
        file_list = []
        for img_path in sorted(images_dir.glob("*.png")):
            # Filter by subset if specified
            if self.subset.lower() != "all":
                if not img_path.name.startswith(self.subset):
                    continue
            
            label_path = labels_dir / img_path.name
            file_list.append((str(img_path), str(label_path) if label_path.exists() else None))
        
        return file_list


class SNEMIDataset(PairedDataset):
    """
    SNEMI3D Dataset for neuron segmentation.
    
    Dataset Structure:
    - Resolution: 6x6x30 nm
    - Train: AC4 (100 slices) from Kasthuri dataset
    - Test: AC3 (100 slices) from Kasthuri dataset
    
    Args:
        root_dir: Root directory containing AC3/AC4 files
        split: 'train', 'valid', or 'test'
        transform: Optional transforms
        cache_rate: Cache fraction (default: 1.0)
        train_val_split: Validation fraction (default: 0.2)
    
    Expected file structure:
        root_dir/
            AC3_inputs.h5 or AC3_inputs.tiff
            AC3_labels.h5 or AC3_labels.tiff
            AC4_inputs.h5 or AC4_inputs.tiff
            AC4_labels.h5 or AC4_labels.tiff
    """
    
    def _load_volume(self, filename: str) -> np.ndarray:
        """Load 3D volume from file."""
        filepath = self.root_dir / filename
        
        if filename.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                return f['main'][:]
        elif filename.endswith(('.tiff', '.tif')):
            return tifffile.imread(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def _get_file_list(self) -> List[Tuple[str, Optional[str]]]:
        """For SNEMI, we don't use file list - override _prepare_data instead."""
        return []
    
    def _prepare_data(self) -> List[Dict]:
        """Prepare SNEMI data dictionaries based on split."""
        data_list = []
        
        if self.split in ["train", "valid"]:
            # Train/Valid: AC4 volume
            if (self.root_dir / 'AC4_inputs.tiff').exists():
                inputs = self._load_volume('AC4_inputs.tiff')
                labels = self._load_volume('AC4_labels.tiff')
            else:
                inputs = self._load_volume('AC4_inputs.h5')
                labels = self._load_volume('AC4_labels.h5')
            
            # Calculate split point
            n_total = inputs.shape[0]
            n_train = int(n_total * (1.0 - self.train_val_split))
            
            if self.split == "train":
                slice_range = range(n_train)
            else:  # valid
                slice_range = range(n_train, n_total)
            
            for i in slice_range:
                data_list.append({
                    "image": inputs[i],
                    "label": labels[i],
                    "slice_idx": i,
                    "volume": "AC4",
                    "idx": i,
                })
        
        elif self.split == "test":
            # Test: AC3 volume (no labels)
            if (self.root_dir / 'AC3_inputs.tiff').exists():
                inputs = self._load_volume('AC3_inputs.tiff')
            else:
                inputs = self._load_volume('AC3_inputs.h5')
            
            for i in range(inputs.shape[0]):
                data_list.append({
                    "image": inputs[i],
                    "slice_idx": i,
                    "volume": "AC3",
                    "idx": i,
                })
        
        return data_list
