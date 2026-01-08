# Ariel - Connectomics Foundation Model

Foundation model based on Vista2D and Vista3D for connectomics data.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with default configuration
python code/main.py

# Train with custom parameters
python code/main.py data.batch_size=8 training.max_epochs=200

# Multi-GPU training
python code/main.py training.devices=4 training.strategy=ddp

# Fast development run (test setup)
python code/main.py training.fast_dev_run=True
```

### Configuration

The project uses Hydra for configuration management. Main config file: `conf/config.yaml`

**Key configuration options:**

```yaml
# Data
data.batch_size: 4              # Batch size
data.train_val_split: 0.2       # Train/validation split ratio
data.cache_rate: 0.5            # Fraction of data to cache

# Model
model.net_config.init_filters: 32    # Initial filter count
model.net_config.feature_dim: 64     # Feature dimension
model.net_config.emb_dim: 16         # Embedding dimension

# Training
training.max_epochs: 100        # Maximum epochs
training.accelerator: auto      # cpu, gpu, or auto
training.devices: 1             # Number of devices
training.precision: 32-true     # 32-true, 16-mixed, bf16-mixed
```

**Override any parameter from command line:**

```bash
python code/main.py model.net_config.init_filters=64 training.max_epochs=150
```

### Hyperparameter Sweeps

```bash
# Run sweep over multiple embedding dimensions
python code/main.py --multirun model.net_config.emb_dim=8,16,32,64

# Sweep over learning rates and batch sizes
python code/main.py --multirun model.optimizer.lr=1e-3,1e-4,1e-5 data.batch_size=4,8,16
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
