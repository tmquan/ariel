"""
Data modules for paired image-label datasets.

This module provides:
- PairedDataModule: Base class for paired data
- Specific implementations: CVPPP14, CVPPP15, CVPPP17, CVPPPRL, SNEMI DataModules
"""

from typing import Optional, Type
from abc import ABC

import torch
import pytorch_lightning as pl
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandSpatialCropd,
    ScaleIntensityd,
    ToTensord,
    Resized,
    Lambdad,
)

from loader.dataset import (
    PairedDataset,
    CVPPP14Dataset,
    CVPPP15Dataset,
    CVPPP17Dataset,
    CVPPPRLDataset,
    SNEMIDataset,
)


def select_rgb_channels(x):
    """Select only RGB channels (first 3) from RGBA images."""
    return x[:3]


__all__ = [
    "PairedDataModule",
    "CVPPP14DataModule",
    "CVPPP15DataModule",
    "CVPPP17DataModule",
    "CVPPPRLDataModule",
    "SNEMIDataModule",
    "CombinedCVPPPDataModule",
]


class PairedDataModule(pl.LightningDataModule, ABC):
    """
    Base PyTorch Lightning DataModule for paired image-label datasets.
    
    Provides common functionality for:
    - Data loading and splitting
    - Transform application
    - DataLoader creation
    
    Subclasses should set `dataset_class` to the appropriate PairedDataset subclass.
    
    Args:
        data_root: Path to the data directory
        batch_size: Batch size for training and validation
        num_workers: Number of worker processes for data loading
        train_val_split: Fraction for validation (default: 0.2)
        cache_rate: Fraction of data to cache in memory (default: 0.5)
        pin_memory: Whether to pin memory for faster GPU transfer
        image_size: Optional image size for resizing (H, W)
    """
    
    dataset_class: Type[PairedDataset] = PairedDataset
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.cache_rate = cache_rate
        self.pin_memory = pin_memory
        self.image_size = image_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _get_dataset_kwargs(self) -> dict:
        """Get additional kwargs for dataset initialization. Override in subclasses."""
        return {}
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        extra_kwargs = self._get_dataset_kwargs()
        
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(
                root_dir=self.data_root,
                split='train',
                train_val_split=self.train_val_split,
                cache_rate=self.cache_rate,
                transform=self.get_train_transforms(),
                **extra_kwargs
            )
            
            self.val_dataset = self.dataset_class(
                root_dir=self.data_root,
                split='valid',
                train_val_split=self.train_val_split,
                cache_rate=1.0,
                transform=self.get_val_transforms(),
                **extra_kwargs
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_class(
                root_dir=self.data_root,
                split='test',
                cache_rate=0.0,
                transform=self.get_val_transforms(),
                **extra_kwargs
            )
    
    def get_train_transforms(self):
        """Get training transforms with augmentation."""
        transforms = [
            EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.extend([
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
            ToTensord(keys=['image', 'label']),
        ])
        
        return Compose(transforms)
    
    def get_val_transforms(self):
        """Get validation transforms without augmentation."""
        transforms = [
            EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.append(ToTensord(keys=['image', 'label']))
        
        return Compose(transforms)
    
    def train_dataloader(self):
        """Create training DataLoader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        """Create validation DataLoader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self):
        """Create test DataLoader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def predict_dataloader(self):
        """Create prediction DataLoader (same as test)."""
        return self.test_dataloader()


class CVPPP14DataModule(PairedDataModule):
    """
    DataModule for CVPPP 2014 plant leaf segmentation dataset.
    
    Args:
        data_root: Path to CVPPP14 directory
        crop_size: Random crop size for training (H, W) (default: (256, 256))
        batch_size: Batch size (default: 4)
        num_workers: Data loading workers (default: 4)
        train_val_split: Validation fraction (default: 0.2)
        cache_rate: Cache fraction (default: 0.5)
        pin_memory: Pin memory (default: True)
        image_size: Optional resize dimensions (H, W)
    """
    
    dataset_class = CVPPP14Dataset
    
    def __init__(
        self,
        data_root: str,
        crop_size: tuple = (256, 256),
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
    ):
        self.crop_size = crop_size
        super().__init__(data_root, batch_size, num_workers, train_val_split, cache_rate, pin_memory, image_size)
    
    def get_train_transforms(self):
        """Get training transforms with random crop for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        # Random spatial crop
        transforms.extend([
            RandSpatialCropd(
                keys=['image', 'label'],
                roi_size=self.crop_size,
                random_size=False,
            ),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
            ToTensord(keys=['image', 'label']),
        ])
        
        return Compose(transforms)
    
    def get_val_transforms(self):
        """Get validation transforms for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.append(ToTensord(keys=['image', 'label']))
        
        return Compose(transforms)


class CVPPP15DataModule(PairedDataModule):
    """
    DataModule for CVPPP 2015 plant leaf segmentation dataset.
    
    Args:
        data_root: Path to CVPPP15 directory
        challenge: 'all', 'LCC', or 'LSC' (default: 'all')
        crop_size: Random crop size for training (H, W) (default: (256, 256))
        batch_size: Batch size (default: 4)
        num_workers: Data loading workers (default: 4)
        train_val_split: Validation fraction (default: 0.2)
        cache_rate: Cache fraction (default: 0.5)
        pin_memory: Pin memory (default: True)
        image_size: Optional resize dimensions (H, W)
    """
    
    dataset_class = CVPPP15Dataset
    
    def __init__(
        self,
        data_root: str,
        challenge: str = "all",
        crop_size: tuple = (256, 256),
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
    ):
        self.challenge = challenge
        self.crop_size = crop_size
        super().__init__(data_root, batch_size, num_workers, train_val_split, cache_rate, pin_memory, image_size)
    
    def _get_dataset_kwargs(self) -> dict:
        return {"challenge": self.challenge}
    
    def get_train_transforms(self):
        """Get training transforms with random crop for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        # Random spatial crop
        transforms.extend([
            RandSpatialCropd(
                keys=['image', 'label'],
                roi_size=self.crop_size,
                random_size=False,
            ),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
            ToTensord(keys=['image', 'label']),
        ])
        
        return Compose(transforms)
    
    def get_val_transforms(self):
        """Get validation transforms for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.append(ToTensord(keys=['image', 'label']))
        
        return Compose(transforms)


class CVPPP17DataModule(PairedDataModule):
    """
    DataModule for CVPPP 2017 plant leaf segmentation dataset.
    
    Args:
        data_root: Path to CVPPP17 directory
        challenge: 'all', 'LCC', or 'LSC' (default: 'all')
        crop_size: Random crop size for training (H, W) (default: (256, 256))
        batch_size: Batch size (default: 4)
        num_workers: Data loading workers (default: 4)
        train_val_split: Validation fraction (default: 0.2)
        cache_rate: Cache fraction (default: 0.5)
        pin_memory: Pin memory (default: True)
        image_size: Optional resize dimensions (H, W)
    """
    
    dataset_class = CVPPP17Dataset
    
    def __init__(
        self,
        data_root: str,
        challenge: str = "all",
        crop_size: tuple = (256, 256),
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
    ):
        self.challenge = challenge
        self.crop_size = crop_size
        super().__init__(data_root, batch_size, num_workers, train_val_split, cache_rate, pin_memory, image_size)
    
    def _get_dataset_kwargs(self) -> dict:
        return {"challenge": self.challenge}
    
    def get_train_transforms(self):
        """Get training transforms with random crop for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        # Random spatial crop
        transforms.extend([
            RandSpatialCropd(
                keys=['image', 'label'],
                roi_size=self.crop_size,
                random_size=False,
            ),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
            ToTensord(keys=['image', 'label']),
        ])
        
        return Compose(transforms)
    
    def get_val_transforms(self):
        """Get validation transforms for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.append(ToTensord(keys=['image', 'label']))
        
        return Compose(transforms)


class CVPPPRLDataModule(PairedDataModule):
    """
    DataModule for Plant Phenotyping (PRL) dataset.
    
    Args:
        data_root: Path to PPP_PRL directory
        subset: 'all', 'Plant', or 'Tray' (default: 'all')
        batch_size: Batch size (default: 4)
        num_workers: Data loading workers (default: 4)
        train_val_split: Validation fraction (default: 0.2)
        cache_rate: Cache fraction (default: 0.5)
        pin_memory: Pin memory (default: True)
        image_size: Optional resize dimensions (H, W)
    """
    
    dataset_class = CVPPPRLDataset
    
    def __init__(
        self,
        data_root: str,
        subset: str = "all",
        batch_size: int = 4,
        num_workers: int = 4,
        train_val_split: float = 0.2,
        cache_rate: float = 0.5,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
    ):
        self.subset = subset
        super().__init__(data_root, batch_size, num_workers, train_val_split, cache_rate, pin_memory, image_size)
    
    def _get_dataset_kwargs(self) -> dict:
        return {"subset": self.subset}
    
    def get_train_transforms(self):
        """Get training transforms for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.extend([
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
            ToTensord(keys=['image', 'label']),
        ])
        
        return Compose(transforms)
    
    def get_val_transforms(self):
        """Get validation transforms for RGB/RGBA plant images."""
        transforms = [
            # RGBA/RGB images: [H, W, 4/3] -> [4/3, H, W], Labels: [H, W] -> [1, H, W]
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            # Select only RGB channels (drop alpha if present): [C, H, W] -> [3, H, W]
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.append(ToTensord(keys=['image', 'label']))
        
        return Compose(transforms)


class SNEMIDataModule(PairedDataModule):
    """
    DataModule for SNEMI3D neuron segmentation dataset.
    
    Args:
        data_root: Path to SNEMI data directory
        batch_size: Batch size (default: 4)
        num_workers: Data loading workers (default: 4)
        train_val_split: Validation fraction (default: 0.2)
        cache_rate: Cache fraction (default: 0.5)
        pin_memory: Pin memory (default: True)
        image_size: Optional resize dimensions (H, W)
    """
    
    dataset_class = SNEMIDataset
    
    def get_train_transforms(self):
        """Get training transforms for grayscale EM images."""
        transforms = [
            EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.extend([
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.7, 1.3)),
            ToTensord(keys=['image', 'label']),
        ])
        
        return Compose(transforms)
    
    def get_val_transforms(self):
        """Get validation transforms for grayscale EM images."""
        transforms = [
            EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        transforms.append(ToTensord(keys=['image', 'label']))
        
        return Compose(transforms)


class CombinedCVPPPDataModule(pl.LightningDataModule):
    """
    Combined DataModule for all CVPPP datasets (14, 15, 17, PRL).
    
    Concatenates all datasets for training on the complete CVPPP benchmark.
    
    Args:
        data_roots: Dict mapping dataset names to their root directories
        batch_size: Batch size for training and validation
        num_workers: Number of worker processes
        train_val_split: Fraction for validation
        cache_rate: Fraction of data to cache
        pin_memory: Whether to pin memory
        image_size: Image size for resizing
        crop_size: Crop size for random cropping
    """
    
    def __init__(
        self,
        data_roots: dict,
        batch_size: int = 4,
        num_workers: int = 0,
        train_val_split: float = 0.2,
        cache_rate: float = 1.0,
        pin_memory: bool = True,
        image_size: Optional[tuple] = None,
        crop_size: tuple = (256, 256),
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.data_roots = data_roots
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.cache_rate = cache_rate
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.crop_size = crop_size
        
        self.train_dataset = None
        self.val_dataset = None
    
    def _get_transforms(self, is_train: bool):
        """Get transforms for RGB/RGBA plant images."""
        transforms = [
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),
            Lambdad(keys=['image'], func=select_rgb_channels),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
        ]
        
        if self.image_size is not None:
            transforms.append(
                Resized(keys=['image', 'label'], spatial_size=self.image_size, mode=['bilinear', 'nearest'])
            )
        
        if is_train:
            transforms.extend([
                RandSpatialCropd(
                    keys=['image', 'label'],
                    roi_size=self.crop_size,
                    random_size=False,
                ),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
                RandRotate90d(keys=['image', 'label'], prob=0.5, spatial_axes=(0, 1)),
                RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
                RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
            ])
        
        transforms.append(ToTensord(keys=['image', 'label']))
        return Compose(transforms)
    
    def setup(self, stage: Optional[str] = None):
        """Setup all datasets."""
        from torch.utils.data import ConcatDataset
        
        train_transform = self._get_transforms(is_train=True)
        val_transform = self._get_transforms(is_train=False)
        
        train_datasets = []
        val_datasets = []
        
        # CVPPP14
        if 'cvppp14' in self.data_roots:
            train_datasets.append(CVPPP14Dataset(
                root_dir=self.data_roots['cvppp14'],
                split='train',
                train_val_split=self.train_val_split,
                cache_rate=self.cache_rate,
                transform=train_transform,
            ))
            val_datasets.append(CVPPP14Dataset(
                root_dir=self.data_roots['cvppp14'],
                split='valid',
                train_val_split=self.train_val_split,
                cache_rate=1.0,
                transform=val_transform,
            ))
        
        # CVPPP15
        if 'cvppp15' in self.data_roots:
            train_datasets.append(CVPPP15Dataset(
                root_dir=self.data_roots['cvppp15'],
                split='train',
                train_val_split=self.train_val_split,
                cache_rate=self.cache_rate,
                transform=train_transform,
                challenge='all',
            ))
            val_datasets.append(CVPPP15Dataset(
                root_dir=self.data_roots['cvppp15'],
                split='valid',
                train_val_split=self.train_val_split,
                cache_rate=1.0,
                transform=val_transform,
                challenge='all',
            ))
        
        # CVPPP17
        if 'cvppp17' in self.data_roots:
            train_datasets.append(CVPPP17Dataset(
                root_dir=self.data_roots['cvppp17'],
                split='train',
                train_val_split=self.train_val_split,
                cache_rate=self.cache_rate,
                transform=train_transform,
                challenge='all',
            ))
            val_datasets.append(CVPPP17Dataset(
                root_dir=self.data_roots['cvppp17'],
                split='valid',
                train_val_split=self.train_val_split,
                cache_rate=1.0,
                transform=val_transform,
                challenge='all',
            ))
        
        # CVPPP PRL
        if 'cvppp_rl' in self.data_roots:
            train_datasets.append(CVPPPRLDataset(
                root_dir=self.data_roots['cvppp_rl'],
                split='train',
                train_val_split=self.train_val_split,
                cache_rate=self.cache_rate,
                transform=train_transform,
            ))
            val_datasets.append(CVPPPRLDataset(
                root_dir=self.data_roots['cvppp_rl'],
                split='valid',
                train_val_split=self.train_val_split,
                cache_rate=1.0,
                transform=val_transform,
            ))
        
        # Concatenate all datasets
        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
        if val_datasets:
            self.val_dataset = ConcatDataset(val_datasets)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
