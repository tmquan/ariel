# Ariel - Connectomics Foundation Model

Foundation model based on Vista2D and Vista3D for connectomics data.

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

### SNEMI3D Dataset

The SNEMI3D dataset is implemented in `code/loader/datasets.py` and provides neuron segmentation data from the Kasthuri dataset.

**Dataset Structure:**
- Resolution: 6x6x30 nm
- **Train split**: AC4 volume (80 slices by default with 80/20 split)
- **Valid split**: AC4 volume (20 slices by default with 80/20 split)
- **Test split**: AC3 volume (100 slices, no labels)

**Important Note:** Both AC3 and AC4 volumes contain 100 slices each. The validation split is created by splitting the AC4 training volume.

**Usage:**

```python
from code.loader.datasets import SNEMIDataset

# Load training data (80 slices from AC4 by default)
train_dataset = SNEMIDataset(
    root_dir='/path/to/data',
    split='train',
    transform=None,  # Add your transforms here
    cache_rate=1.0,  # Cache all data in memory
    train_val_split=0.2  # Use 20% for validation
)

# Load validation data (20 slices from AC4 by default)
valid_dataset = SNEMIDataset(
    root_dir='/path/to/data',
    split='valid',
    train_val_split=0.2  # Must match train split
)

# Load test data (100 slices from AC3, no labels)
test_dataset = SNEMIDataset(
    root_dir='/path/to/data',
    split='test'
)

# Access samples
sample = train_dataset[0]
print(f"Image shape: {sample['image'].shape}")  # (H, W)
print(f"Label shape: {sample['label'].shape}")  # (H, W)
print(f"Slice index: {sample['slice_idx']}")
print(f"Volume: {sample['volume']}")
```

**Expected Directory Structure:**

```
data/
├── AC3_inputs.h5 or AC3_inputs.tiff    # 100 slices (test)
├── AC3_labels.h5 or AC3_labels.tiff    # 100 slices (not used for test)
├── AC4_inputs.h5 or AC4_inputs.tiff    # 100 slices (train/valid)
└── AC4_labels.h5 or AC4_labels.tiff    # 100 slices (train/valid)
```

**Dataset Details:**

| Split | Volume | Slices | Has Labels | Size (default 80/20 split) |
|-------|--------|--------|------------|---------------------------|
| Train | AC4    | Configurable | Yes | 80 samples (80% of AC4) |
| Valid | AC4    | Configurable | Yes | 20 samples (20% of AC4) |
| Test  | AC3    | 100    | No  | 100 samples |

**Data Format:**
- Images: 2D grayscale EM images (uint8), shape (H, W)
- Labels: 2D segmentation masks (uint16), shape (H, W)
- Supported formats: HDF5 (.h5) or TIFF (.tiff)

## Testing

### Run Unit Tests

```bash
# Install dependencies first
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_snemi_dataset.py -v
```

### Quick Dataset Check

```bash
python examples/snemi_usage.py
```

This will verify that all three splits (train, valid, test) are loading correctly.

## Examples

See `examples/snemi_usage.py` for:
- Data loading with transforms
- Creating PyTorch DataLoaders
- Example training loop structure
- Quick dataset verification

```python
# Quick usage
python examples/snemi_usage.py
```

## References

- SNEMI3D: https://snemi3d.grand-challenge.org/
- Kasthuri et al. (2015) dataset
- Based on MONAI's CacheDataset
