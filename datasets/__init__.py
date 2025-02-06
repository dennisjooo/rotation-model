"""Document rotation datasets package.

This package provides datasets and dataloaders for document rotation classification.
"""

from .base import BaseDocumentDataset
from .dataloader import create_dataloaders
from .rvl_cdip import RVLCDIPDataset
from .publaynet import PubLayNetDataset
from .midv500 import MIDV500Dataset
from .sroie import SROIEDataset

__all__ = [
    'BaseDocumentDataset',
    'create_dataloaders',
    'RVLCDIPDataset',
    'PubLayNetDataset',
    'MIDV500Dataset',
    'SROIEDataset',
] 