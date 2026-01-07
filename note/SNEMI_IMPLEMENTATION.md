# SNEMIDataset Implementation Summary

## Overview

Implemented `SNEMIDataset` class in `/home/nvidia/ariel/code/loader/datasets.py` for the SNEMI3D neuron segmentation challenge dataset.

## Implementation Details

### Class: `SNEMIDataset`

**Inherits from:** `monai.data.CacheDataset` and `monai.transforms.Randomizable`

**Key Features:**
- Supports three data splits: train, valid, and test
- Handles both HDF5 (.h5) and TIFF (.tiff) file formats
- Implements per-slice data loading for 2D training
- Built-in caching support via MONAI's CacheDataset

### Constructor Parameters

```python
SNEMIDataset(
    root_dir: str,           # Path to data directory
    split: str = "train",    # One of: 'train', 'valid', 'test'
    transform: Optional[Callable] = None,  # MONAI transforms
    cache_rate: float = 1.0,  # Fraction of data to cache (0.0-1.0)
    train_val_split: float = 0.2  # Fraction for validation (default: 0.2 = 80/20)
)
```

### Data Splits

| Split | Source Volume | Slices | # Samples | Has Labels | Description |
|-------|---------------|--------|-----------|------------|-------------|
| **train** | AC4 | 0-79 (default) | 80 | ✓ Yes | First 80% of AC4 for training |
| **valid** | AC4 | 80-99 (default) | 20 | ✓ Yes | Last 20% of AC4 for validation |
| **test** | AC3 | 0-99 | 100 | ✗ No | Full AC3 volume (no labels) |

**Note:** Both AC3 and AC4 volumes contain exactly 100 slices each. The original SNEMI3D description mentioned 256 slices for AC3, but the actual data has 100 slices. The train/valid split ratio is configurable via the `train_val_split` parameter (default: 0.2 = 80/20 split).

### Data Dictionary Structure

Each sample is a dictionary with the following keys:

**Training and Validation:**
```python
{
    'image': np.ndarray,      # Shape: (H, W), dtype: uint8
    'label': np.ndarray,      # Shape: (H, W), dtype: uint16
    'slice_idx': int,         # Original slice index in volume
    'volume': str             # Volume identifier ('AC4', 'AC3_valid', 'AC3_test')
}
```

**Test (no labels):**
```python
{
    'image': np.ndarray,      # Shape: (H, W), dtype: uint8
    'slice_idx': int,         # Original slice index in volume
    'volume': str             # Volume identifier ('AC3_test')
}
```

## File Structure

The dataset expects the following files in `root_dir`:

```
root_dir/
├── AC3_inputs.h5 or AC3_inputs.tiff    # 100 slices (test)
├── AC3_labels.h5 or AC3_labels.tiff    # 100 slices (not used in test split)
├── AC4_inputs.h5 or AC4_inputs.tiff    # 100 slices (train/valid)
└── AC4_labels.h5 or AC4_labels.tiff    # 100 slices (train/valid)
```

## Usage Examples

### Basic Usage

```python
from code.loader.datasets import SNEMIDataset

# Load training data (default 80% of AC4)
train_dataset = SNEMIDataset(root_dir='/path/to/data', split='train')
print(f"Training samples: {len(train_dataset)}")  # 80

# Load validation data (default 20% of AC4)
valid_dataset = SNEMIDataset(root_dir='/path/to/data', split='valid')
print(f"Validation samples: {len(valid_dataset)}")  # 20

# Load test data (all 100 slices from AC3)
test_dataset = SNEMIDataset(root_dir='/path/to/data', split='test')
print(f"Test samples: {len(test_dataset)}")  # 100

# Access samples
sample = train_dataset[0]
print(f"Image shape: {sample['image'].shape}")  # (1024, 1024)
print(f"Label shape: {sample['label'].shape}")  # (1024, 1024)
```

### With MONAI Transforms

```python
from monai.transforms import Compose, RandFlipd, RandRotate90d, ToTensord

transforms = Compose([
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=['image', 'label'], prob=0.5),
    ToTensord(keys=['image', 'label'])
])

train_dataset = SNEMIDataset(
    root_dir='/path/to/data',
    split='train',
    transform=transforms,
    cache_rate=0.5  # Cache 50% of data
)
```

### With DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

for batch in train_loader:
    images = batch['image']  # Shape: (4, 1024, 1024)
    labels = batch['label']  # Shape: (4, 1024, 1024)
    # Training code here...
```

## Implementation Highlights

### 1. Flexible File Format Support

The `_load_volume()` method automatically detects and loads both HDF5 and TIFF formats:

```python
def _load_volume(self, filename: str) -> np.ndarray:
    filepath = os.path.join(self.root_dir, filename)
    
    if filename.endswith('.h5'):
        with h5py.File(filepath, 'r') as f:
            return f['main'][:]
    elif filename.endswith('.tiff') or filename.endswith('.tif'):
        return tifffile.imread(filepath)
```

### 2. Split-Specific Data Preparation

The `_prepare_data()` method handles the different requirements for each split:

- **Train**: Loads first 80% of slices from AC4 (80 slices with default 0.2 split)
- **Valid**: Loads last 20% of slices from AC4 (20 slices with default 0.2 split)
- **Test**: Loads all 100 slices from AC3 (no labels)

### 3. Per-Slice Data Structure

Each 3D volume is decomposed into individual 2D slices, allowing for:
- Efficient 2D model training
- Easy augmentation per slice
- Memory-efficient data loading with caching

## Validation

Created test script: `test_snemi_dataset.py`

To run tests (requires dependencies):
```bash
pip install -r requirements.txt
python test_snemi_dataset.py
```

Expected output:
```
Train samples: 80 (expected: 80)
Valid samples: 20 (expected: 20)
Test samples: 100 (expected: 100)
```

## Dependencies

Updated `requirements.txt` with necessary packages:
- `torch>=2.0.0`
- `monai>=1.3.0`
- `numpy>=1.24.0`
- `h5py>=3.8.0`
- `tifffile>=2023.0.0`

## Dataset Specifications

- **Resolution**: 6×6×30 nm
- **Image size**: 1024×1024 pixels per slice
- **Image type**: Grayscale EM images (uint8)
- **Label type**: Instance segmentation masks (uint16)
- **Source**: Kasthuri et al. (2015) dataset
- **Challenge**: SNEMI3D (https://snemi3d.grand-challenge.org/)

## Notes

1. The implementation follows MONAI's dataset conventions
2. Compatible with MONAI's built-in transforms
3. Supports caching for faster training iterations
4. Test split intentionally has no labels (for challenge submission)
5. **Important**: Both AC3 and AC4 volumes contain 100 slices each, not 256 as originally documented
6. Valid split is created by splitting AC4 (default 80/20 train/val split)
7. Train uses first portion of AC4, valid uses last portion to ensure no overlap

## Files Modified/Created

1. ✅ `/home/nvidia/ariel/code/loader/datasets.py` - Implemented SNEMIDataset class
2. ✅ `/home/nvidia/ariel/requirements.txt` - Added dependencies
3. ✅ `/home/nvidia/ariel/README.md` - Updated documentation
4. ✅ `/home/nvidia/ariel/test_snemi_dataset.py` - Created test script
5. ✅ `/home/nvidia/ariel/SNEMI_IMPLEMENTATION.md` - This summary document

