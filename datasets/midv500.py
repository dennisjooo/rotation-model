"""MIDV-500 dataset implementation.

The MIDV-500 dataset contains identity document images captured in various conditions.
For our rotation task, we use it without the document type labels, only for rotation prediction.

Reference:
    Bulatov, K., Matalov, D., Arlazarov, V.V.
    "MIDV-500: a dataset for identity document analysis and recognition on mobile devices in video stream."
    Компьютерная оптика 44(5), 818-824 (2020)
"""

import os
from pathlib import Path

from .base import BaseDocumentDataset


class MIDV500Dataset(BaseDocumentDataset):
    """MIDV-500 dataset for document rotation prediction.
    
    This implementation ignores the original document type labels and only
    uses the images for rotation prediction.
    
    The dataset structure after running download_datasets.sh:
    data/midv500/
        *.tif    # All TIFF images directly in root directory
    """
    
    def __init__(
        self,
        root_dir: str | Path = os.path.join(os.getcwd(), "data/midv500"),
        split: str = "train",
        img_size: int = 384,
        val_split: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """Initialize MIDV-500 dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', or 'test')
            img_size: Target image size
            val_split: Fraction of data to use for validation
            random_seed: Random seed for reproducible splits
        """
        super().__init__(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            val_split=val_split,
            random_seed=random_seed
        )
    
    def _load_dataset(self) -> None:
        """Load MIDV-500 dataset samples."""
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        # Find all TIFF images in root directory
        for img_path in self.root_dir.glob("*.tif"):
            self._all_samples.append((img_path, 0))  # All documents are upright
        
        if not self._all_samples:
            raise RuntimeError(
                f"No TIFF images found in {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        print(f"Loaded {len(self._all_samples)} images from MIDV-500 dataset") 