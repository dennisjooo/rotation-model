"""DocBank dataset implementation.

The DocBank dataset is a benchmark dataset for document layout analysis,
containing academic documents with fine-grained token-level annotations.

Reference:
    Li, Minghao, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou.
    "DocBank: A Benchmark Dataset for Document Layout Analysis."
    arXiv preprint arXiv:2006.01038 (2020).
"""

import os
from pathlib import Path
from ..base import BaseDocumentDataset


class DocBankDataset(BaseDocumentDataset):
    """DocBank dataset for document layout analysis.
    
    Contains academic documents with fine-grained token-level annotations.
    Particularly rich in charts, tables, and scientific figures.
    """
    
    def __init__(
        self,
        root_dir: str | Path = os.path.join(os.getcwd(), "data/docbank"),
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
        """Load DocBank dataset samples."""
        # Get all PDF pages converted to images
        image_paths = list(self.root_dir.glob("**/*.jpg"))
        
        if not image_paths:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                "Please run the dataset downloader first."
            )
        
        # Store (image_path, original_rotation) pairs
        # Original rotation is 0 since these are standard academic documents
        self._all_samples = [(path, 0) for path in image_paths] 