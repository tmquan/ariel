# Dataset loader implementation

# Datasets loader implementation
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from monai.data import CacheDataset
from monai.transforms import LoadImaged, Randomizable

__all__ = [
    "SNEMIDataset", 
    "CREMIDataset", 
    "MiCRONSDataset",
]

class SNEMIDataset(CacheDataset, Randomizable):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        super().__init__(root_dir, transform=transform)

class CREMIDataset(CacheDataset, Randomizable):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        super().__init__(root_dir, transform=transform)