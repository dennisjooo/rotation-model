"""ChartQA dataset implementation.

The ChartQA dataset is designed for question answering about charts,
requiring both visual and logical reasoning capabilities.

Reference:
    Masry, Ahmed, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque.
    "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning."
    In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263-2279.
    Dublin, Ireland: Association for Computational Linguistics, 2022.
"""

from ..base import BaseDocumentDataset


class ChartQADataset(BaseDocumentDataset):
    """ChartQA dataset containing various chart and visualization images.
    
    Dataset focused on charts, graphs, and other data visualizations.
    Includes bar charts, line graphs, pie charts, and scatter plots.
    
    The dataset is structured with images in JPG and PNG formats organized in subdirectories.
    All images are assumed to be in standard upright orientation (0 degrees rotation).
    
    Attributes:
        root_dir: Path to the root directory containing the dataset
        split: Dataset split ('train', 'val', or 'test')
        transform: Optional transforms to apply to images
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducible splits
    """
    
    def _load_dataset(self) -> None:
        """Load ChartQA dataset samples.
        
        Scans the dataset root directory recursively for JPG and PNG images.
        Stores tuples of (image_path, original_rotation) where original_rotation
        is always 0 since these are standard upright chart images.
        
        Raises:
            RuntimeError: If no images are found in the root directory
        """
        # Get all chart images
        image_paths = list(self.root_dir.glob("**/*.jpg"))
        image_paths.extend(self.root_dir.glob("**/*.png"))
        
        if not image_paths:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                "Please run the dataset downloader first."
            )
        
        # Store (image_path, original_rotation) pairs
        # Original rotation is 0 since these are standard chart images
        self._all_samples = [(path, 0) for path in image_paths]