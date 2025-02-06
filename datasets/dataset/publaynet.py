"""PubLayNet dataset implementation.

The PubLayNet dataset contains document layout analysis data from scientific papers.
For our rotation task, we use it without the layout annotations, only for rotation prediction.

Reference:
    Zhong, X., Tang, J., Yepes, A.J.
    "PubLayNet: largest dataset ever for document layout analysis."
    International Conference on Document Analysis and Recognition (ICDAR), 2019
"""

import os
from pathlib import Path

from .base import BaseDocumentDataset


class PubLayNetDataset(BaseDocumentDataset):
    """PubLayNet dataset for document rotation prediction.
    
    This implementation ignores the original layout annotations and only
    uses the images for rotation prediction.
    
    The dataset structure after running download_datasets.sh:
    data/publaynet/
        *.png    # All PNG images directly in root directory
    """
    
    def __init__(
        self,
        root_dir: str | Path = os.path.join(os.getcwd(), "data/publaynet"),
        split: str = "train",
        img_size: int = 384,
        val_split: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """Initialize PubLayNet dataset.
        
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
        """Load PubLayNet dataset samples."""
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        # Find all PNG images in root directory
        for img_path in self.root_dir.glob("*.png"):
            self._all_samples.append((img_path, 0))  # All documents are upright
        
        if not self._all_samples:
            raise RuntimeError(
                f"No PNG images found in {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        print(f"Loaded {len(self._all_samples)} images from PubLayNet dataset") 