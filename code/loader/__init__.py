"""
Data loading utilities for Ariel.

Provides dataset and datamodule classes for various segmentation datasets:
- CVPPP14, CVPPP15, CVPPP17: Plant leaf segmentation
- CVPPPRL: Plant Phenotyping (PRL) dataset  
- SNEMI: Neuron segmentation (connectomics)
"""

from loader.dataset import (
    PairedDataset,
    CVPPP14Dataset,
    CVPPP15Dataset,
    CVPPP17Dataset,
    CVPPPRLDataset,
    SNEMIDataset,
)

from loader.datamodule import (
    PairedDataModule,
    CVPPP14DataModule,
    CVPPP15DataModule,
    CVPPP17DataModule,
    CVPPPRLDataModule,
    SNEMIDataModule,
)

__all__ = [
    # Datasets
    "PairedDataset",
    "CVPPP14Dataset",
    "CVPPP15Dataset",
    "CVPPP17Dataset",
    "CVPPPRLDataset",
    "SNEMIDataset",
    # DataModules
    "PairedDataModule",
    "CVPPP14DataModule",
    "CVPPP15DataModule",
    "CVPPP17DataModule",
    "CVPPPRLDataModule",
    "SNEMIDataModule",
]
