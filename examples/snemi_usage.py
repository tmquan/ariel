"""
Example usage of SNEMIDataset for training a segmentation model.

This script demonstrates how to:
1. Load the SNEMI3D dataset
2. Apply transforms for data augmentation
3. Create DataLoaders for training
4. Basic training loop structure
"""

import os
import sys

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

import torch
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandAdjustContrastd,
    ScaleIntensityd,
    ToTensord,
)

from loader.datasets import SNEMIDataset


def get_transforms(is_train=True):
    """
    Get transforms for training or validation.
    
    Args:
        is_train: If True, includes augmentation transforms
        
    Returns:
        MONAI Compose transform
    """
    keys = ['image', 'label']
    
    if is_train:
        return Compose([
            EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.3)),
            ToTensord(keys=keys),
        ])
    else:
        return Compose([
            EnsureChannelFirstd(keys=keys, channel_dim='no_channel'),
            ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
            ToTensord(keys=keys),
        ])


def create_dataloaders(data_dir, batch_size=4, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = SNEMIDataset(
        root_dir=data_dir,
        split='train',
        transform=get_transforms(is_train=True),
        cache_rate=0.5,  # Cache 50% of training data
        train_val_split=0.2  # 80/20 train/val split
    )
    
    val_dataset = SNEMIDataset(
        root_dir=data_dir,
        split='valid',
        transform=get_transforms(is_train=False),
        cache_rate=1.0,  # Cache all validation data
        train_val_split=0.2  # Must match train split
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def example_training_loop():
    """
    Example training loop structure.
    
    Note: This is a skeleton showing the structure.
    You need to implement your model, loss, optimizer, etc.
    """
    # Configuration
    data_dir = '/home/nvidia/ariel/data'
    batch_size = 4
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # TODO: Initialize your model
    # model = YourModel().to(device)
    
    # TODO: Initialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # TODO: Initialize loss function
    # criterion = YourLossFunction()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        # model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)  # Shape: (B, 1, H, W)
            labels = batch['label'].to(device)  # Shape: (B, 1, H, W)
            
            # TODO: Forward pass
            # outputs = model(images)
            # loss = criterion(outputs, labels)
            
            # TODO: Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Image shape: {images.shape}")
        
        # Validation phase
        # model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                # TODO: Forward pass
                # outputs = model(images)
                # loss = criterion(outputs, labels)
                # val_loss += loss.item()
        
        # Print epoch statistics
        avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if val_loss > 0 else 0
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print("-" * 50)


def quick_dataset_check():
    """
    Quick check to verify dataset is working.
    """
    data_dir = '/home/nvidia/ariel/data'
    
    print("=" * 60)
    print("Quick Dataset Check")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Load a small sample from each split
    try:
        print("\n1. Loading train split...")
        train_ds = SNEMIDataset(data_dir, split='train', cache_rate=0.0)
        train_sample = train_ds[0]
        print(f"   ✓ Train: {len(train_ds)} samples")
        print(f"     - Image shape: {train_sample['image'].shape}")
        print(f"     - Label shape: {train_sample['label'].shape}")
        
        print("\n2. Loading valid split...")
        valid_ds = SNEMIDataset(data_dir, split='valid', cache_rate=0.0)
        valid_sample = valid_ds[0]
        print(f"   ✓ Valid: {len(valid_ds)} samples")
        print(f"     - Image shape: {valid_sample['image'].shape}")
        print(f"     - Label shape: {valid_sample['label'].shape}")
        
        print("\n3. Loading test split...")
        test_ds = SNEMIDataset(data_dir, split='test', cache_rate=0.0)
        test_sample = test_ds[0]
        print(f"   ✓ Test: {len(test_ds)} samples")
        print(f"     - Image shape: {test_sample['image'].shape}")
        print(f"     - Has labels: {'label' in test_sample}")
        
        print("\n" + "=" * 60)
        print("All checks passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Run quick check
    quick_dataset_check()
    
    # Uncomment to run example training loop structure
    # example_training_loop()

