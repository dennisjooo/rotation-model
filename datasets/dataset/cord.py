"""CORD dataset implementation.

The CORD (Consolidated Receipt Dataset) is designed for post-OCR parsing
of receipt images, containing a diverse collection of receipt documents.

Reference:
    Park, Seunghyun, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee.
    "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing."
    In Document Intelligence Workshop at Neural Information Processing Systems, 2019.
"""

import os
from pathlib import Path
from ..base import BaseDocumentDataset


class CORDDataset(BaseDocumentDataset):
    """CORD dataset containing credit card and receipt images.
    
    Dataset focused on credit card OCR and information extraction.
    Contains credit card images in various orientations and conditions.
    """
    def __init__(
        self,
        root_dir: str | Path = os.path.join(os.getcwd(), "data/cord"),
        split: str = "train",
        img_size: int = 384,
        val_split: float = 0.1,
        random_seed: int = 42,      
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            val_split=val_split,
            random_seed=random_seed,
        )
    
    def _load_dataset(self) -> None:
        """Load CORD dataset samples."""
        # Get all card images
        image_paths = list(self.root_dir.glob("**/*.jpg"))
        image_paths.extend(self.root_dir.glob("**/*.png"))
        
        if not image_paths:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                "Please run the dataset downloader first."
            )
        
        # Store (image_path, original_rotation) pairs
        # Original rotation is 0 since we'll handle rotations in augmentation
        self._all_samples = [(path, 0) for path in image_paths] 