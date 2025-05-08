"""TableBench dataset implementation.

TableBench is a comprehensive dataset for table understanding and extraction tasks.
For our rotation task, we use the images without their table annotations, focusing only on rotation prediction.

Reference:
    TableBench: A Comprehensive Benchmark for Table Understanding
    https://huggingface.co/datasets/reducto/rd-tablebench
"""

import os
import json
from pathlib import Path
from ..base import BaseDocumentDataset


class TableBenchDataset(BaseDocumentDataset):
    """TableBench dataset for document rotation prediction.
    
    This implementation ignores the original table annotations and only
    uses the images for rotation prediction.
    
    The dataset structure after running download_datasets.sh:
    data/tablebench/
        *.png  # PNG images of tables
    """
    
    def __init__(
        self,
        root_dir: str | Path = os.path.join(os.getcwd(), "data/tablebench"),
        split: str = "train",
        img_size: int = 384,
        val_split: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """Initialize TableBench dataset.
        
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
        """Load TableBench dataset samples."""
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for dataset in: {self.root_dir}")
        
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        # Find all JPG images in root directory
        for img_path in self.root_dir.glob("*.jpg"):
            self._all_samples.append((img_path, 0))  # All documents are upright
        
        if not self._all_samples:
            raise RuntimeError(
                f"No JPG images found in {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        print(f"Loaded {len(self._all_samples)} images from TableBench dataset") 