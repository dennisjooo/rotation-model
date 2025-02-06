"""PlotQA dataset implementation.

The PlotQA dataset contains scientific plots and charts with question-answering annotations.
It is designed for training models to reason over scientific visualizations.

Reference:
    Methani, Nitesh, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar.
    "PlotQA: Reasoning over Scientific Plots."
    In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2020.
"""

from ..base import BaseDocumentDataset


class PlotQADataset(BaseDocumentDataset):
    """PlotQA dataset containing scientific plots and charts.
    
    Large-scale dataset focused on scientific plots with question-answering annotations.
    Includes line plots, bar charts, and scatter plots from scientific papers.
    The dataset was introduced in WACV 2020 for training models to reason over
    scientific visualizations through question answering.
    """
    
    def _load_dataset(self) -> None:
        """Load PlotQA dataset samples.
        
        Scans the dataset root directory recursively for JPG and PNG images.
        Stores tuples of (image_path, original_rotation) where original_rotation
        is always 0 since these are standard upright scientific plots.
        
        Raises:
            RuntimeError: If no images are found in the root directory
        """
        # Get all plot images
        image_paths = list(self.root_dir.glob("**/*.jpg"))
        image_paths.extend(self.root_dir.glob("**/*.png"))
        
        if not image_paths:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                "Please run the dataset downloader first."
            )
        
        # Store (image_path, original_rotation) pairs
        # Original rotation is 0 since these are standard plots
        self._all_samples = [(path, 0) for path in image_paths]