"""SROIE dataset implementation.

The SROIE dataset contains receipt images with text annotations.
For our rotation task, we use it without the OCR annotations, only for rotation prediction.

Reference:
    Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., Jawahar, C.V.
    "ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction."
    International Conference on Document Analysis and Recognition (ICDAR), 2019
"""
import os
from pathlib import Path
from ..base import BaseDocumentDataset


class SROIEDataset(BaseDocumentDataset):
    """SROIE dataset for document rotation prediction.
    
    This implementation ignores the original OCR annotations and only
    uses the images for rotation prediction.
    
    The dataset structure after running download_datasets.sh:
    data/sroie/
        *.jpg    # All JPEG images directly in root directory
    """
    
    def __init__(
        self,
        root_dir: str | Path = os.path.join(os.getcwd(), "data/sroie"),
        split: str = "train",
        img_size: int = 384,
        val_split: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """Initialize SROIE dataset.
        
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
        """Load SROIE dataset samples."""
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        # Find all JPEG images in root directory
        for img_path in self.root_dir.glob("*.jpg"):
            self._all_samples.append((img_path, 0))  # All documents are upright
        
        if not self._all_samples:
            raise RuntimeError(
                f"No JPEG images found in {self.root_dir}\n"
                "Please run downloaders/download_datasets.sh first."
            )
        
        print(f"Loaded {len(self._all_samples)} images from SROIE dataset") 