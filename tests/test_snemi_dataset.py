"""Unit tests for SNEMIDataset."""

import os
import sys
import pytest
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from loader.datasets import SNEMIDataset


class TestSNEMIDataset:
    """Test suite for SNEMIDataset class."""
    
    @pytest.fixture
    def data_dir(self):
        """Fixture for data directory path."""
        return os.path.join(os.path.dirname(__file__), '..', 'data')
    
    def test_train_split(self, data_dir):
        """Test training split loads correctly."""
        dataset = SNEMIDataset(
            root_dir=data_dir,
            split='train',
            cache_rate=0.0,
            train_val_split=0.2
        )
        
        # With 100 slices and 80/20 split, train should have 80 samples
        assert len(dataset) == 80, f"Expected 80 training samples (80% of 100), got {len(dataset)}"
        
        sample = dataset[0]
        assert 'image' in sample, "Sample should contain 'image' key"
        assert 'label' in sample, "Training sample should contain 'label' key"
        assert 'slice_idx' in sample, "Sample should contain 'slice_idx' key"
        assert 'volume' in sample, "Sample should contain 'volume' key"
        
        assert sample['volume'] == 'AC4', f"Training volume should be AC4, got {sample['volume']}"
        assert sample['image'].dtype == np.uint8, f"Image dtype should be uint8, got {sample['image'].dtype}"
        assert len(sample['image'].shape) == 2, f"Image should be 2D, got shape {sample['image'].shape}"
    
    def test_valid_split(self, data_dir):
        """Test validation split loads correctly."""
        dataset = SNEMIDataset(
            root_dir=data_dir,
            split='valid',
            cache_rate=0.0,
            train_val_split=0.2
        )
        
        # With 100 slices and 80/20 split, valid should have 20 samples
        assert len(dataset) == 20, f"Expected 20 validation samples (20% of 100), got {len(dataset)}"
        
        sample = dataset[0]
        assert 'image' in sample, "Sample should contain 'image' key"
        assert 'label' in sample, "Validation sample should contain 'label' key"
        assert 'slice_idx' in sample, "Sample should contain 'slice_idx' key"
        assert 'volume' in sample, "Sample should contain 'volume' key"
        
        assert sample['volume'] == 'AC4_valid', f"Valid volume should be AC4_valid, got {sample['volume']}"
        assert sample['slice_idx'] >= 80, f"Valid slices should start from index 80, got {sample['slice_idx']}"
    
    def test_test_split(self, data_dir):
        """Test test split loads correctly."""
        dataset = SNEMIDataset(
            root_dir=data_dir,
            split='test',
            cache_rate=0.0
        )
        
        assert len(dataset) == 100, f"Expected 100 test samples, got {len(dataset)}"
        
        sample = dataset[0]
        assert 'image' in sample, "Sample should contain 'image' key"
        assert 'label' not in sample, "Test sample should NOT contain 'label' key"
        assert 'slice_idx' in sample, "Sample should contain 'slice_idx' key"
        assert 'volume' in sample, "Sample should contain 'volume' key"
        
        assert sample['volume'] == 'AC3', f"Test volume should be AC3, got {sample['volume']}"
        assert sample['slice_idx'] < 100, f"Test slices should be < 100, got {sample['slice_idx']}"
    
    def test_invalid_split(self, data_dir):
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be"):
            SNEMIDataset(
                root_dir=data_dir,
                split='invalid_split'
            )
    
    def test_data_consistency(self, data_dir):
        """Test data consistency across splits."""
        train_dataset = SNEMIDataset(root_dir=data_dir, split='train', cache_rate=0.0)
        valid_dataset = SNEMIDataset(root_dir=data_dir, split='valid', cache_rate=0.0)
        test_dataset = SNEMIDataset(root_dir=data_dir, split='test', cache_rate=0.0)
        
        # Check that all images have the same spatial dimensions
        train_shape = train_dataset[0]['image'].shape
        valid_shape = valid_dataset[0]['image'].shape
        test_shape = test_dataset[0]['image'].shape
        
        assert train_shape == valid_shape == test_shape, \
            f"All images should have same shape, got train:{train_shape}, valid:{valid_shape}, test:{test_shape}"
    
    def test_cache_rate(self, data_dir):
        """Test different cache rates."""
        # Test no caching
        dataset_no_cache = SNEMIDataset(root_dir=data_dir, split='train', cache_rate=0.0, train_val_split=0.2)
        assert len(dataset_no_cache) == 80
        
        # Test partial caching
        dataset_partial_cache = SNEMIDataset(root_dir=data_dir, split='train', cache_rate=0.5, train_val_split=0.2)
        assert len(dataset_partial_cache) == 80
        
        # Test full caching
        dataset_full_cache = SNEMIDataset(root_dir=data_dir, split='train', cache_rate=1.0, train_val_split=0.2)
        assert len(dataset_full_cache) == 80


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])

