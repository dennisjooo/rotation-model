"""DocBank dataset implementation.

The DocBank dataset is a benchmark dataset for document layout analysis,
containing academic documents with fine-grained token-level annotations.

Reference:
    Li, Minghao, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou.
    "DocBank: A Benchmark Dataset for Document Layout Analysis."
    arXiv preprint arXiv:2006.01038 (2020).
"""

from ..base import BaseDocumentDataset


class DocBankDataset(BaseDocumentDataset):
    """DocBank dataset for document layout analysis.
    
    Contains academic documents with fine-grained token-level annotations.
    Particularly rich in charts, tables, and scientific figures.
    """
    
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