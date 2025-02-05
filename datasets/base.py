"""Base document dataset implementation."""

from __future__ import annotations
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from .transforms import get_train_transforms, get_val_transforms


class BaseDocumentDataset(Dataset):
    """Base class for document datasets.
    
    Implements common functionality for loading and rotating document images.
    Each subclass should implement the _load_dataset method.
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        img_size: int = 384,
        val_split: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train' or 'val')
            img_size: Target image size
            val_split: Fraction of data to use for validation (0.0 to 1.0)
            random_seed: Random seed for reproducible train/val splits
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.val_split = val_split
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        self.rng = random.Random(random_seed)
        
        # List of (image_path, original_rotation) tuples for all samples
        self._all_samples: List[Tuple[Path, int]] = []
        
        # List of (image_path, original_rotation) tuples for current split
        self.samples: List[Tuple[Path, int]] = []
        
        # Load and split dataset
        self._load_and_split_dataset()
        
        # Set up transforms based on split
        self.transform = get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size)
    
    @classmethod
    def create_splits(
        cls,
        root_dir: str | Path,
        img_size: int = 384,
        val_split: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[BaseDocumentDataset, BaseDocumentDataset]:
        """Create train and validation splits with consistent splitting.
        
        This is the recommended way to create train/val splits to ensure
        there are no overlaps between the splits.
        
        Args:
            root_dir: Root directory containing the dataset
            img_size: Target image size
            val_split: Fraction of data to use for validation
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create train split
        train_dataset = cls(
            root_dir=root_dir,
            split="train",
            img_size=img_size,
            val_split=val_split,
            random_seed=random_seed
        )
        
        # Create validation split with same parameters
        val_dataset = cls(
            root_dir=root_dir,
            split="val",
            img_size=img_size,
            val_split=val_split,
            random_seed=random_seed
        )
        
        # Verify splits are disjoint
        train_files = {str(path) for path, _ in train_dataset.samples}
        val_files = {str(path) for path, _ in val_dataset.samples}
        overlap = train_files & val_files
        
        if overlap:
            raise RuntimeError(
                f"Found {len(overlap)} overlapping samples between train and val splits. "
                "This should never happen! Please report this as a bug."
            )
        
        return train_dataset, val_dataset
    
    def _load_and_split_dataset(self) -> None:
        """Load dataset samples and split into train/val."""
        # Load all samples
        self._load_dataset()
        
        # For test split, use all samples
        if self.split == "test":
            self.samples = self._all_samples
            return
        
        # Shuffle samples
        shuffled_samples = list(self._all_samples)  # Create a copy to shuffle
        self.rng.shuffle(shuffled_samples)
        
        # Calculate split index
        split_idx = int(len(shuffled_samples) * (1 - self.val_split))
        
        # Assign samples based on split
        if self.split == "train":
            self.samples = shuffled_samples[:split_idx]
        else:  # val
            self.samples = shuffled_samples[split_idx:]
        
        print(f"Split '{self.split}': {len(self.samples)} samples")
    
    def _load_dataset(self) -> None:
        """Load dataset-specific samples.
        
        Should be implemented by each subclass to populate self.samples
        with (image_path, original_rotation) tuples.
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_background_color(self, img: np.ndarray, train_mode: bool) -> np.ndarray:
        """Calculate background color with optional randomization.
        
        Args:
            img: Input image
            train_mode: Whether to apply randomization
            
        Returns:
            Background color as uint8 RGB array
        """
        # Calculate mean color of non-black pixels
        non_black_mask = np.any(img > 30, axis=2)
        mean_color = np.mean(img[non_black_mask], axis=0).astype(np.uint8)
        white = np.array([255, 255, 255], dtype=np.uint8)
        
        if train_mode:
            # Randomized blend
            random_factor = self.rng.uniform(0.05, 0.15)
            return ((random_factor * mean_color + (1 - random_factor) * white) * 
                    self.rng.uniform(0.95, 1.05)).clip(0, 255).astype(np.uint8)
        else:
            # Fixed blend for validation
            return (0.1 * mean_color + 0.9 * white).astype(np.uint8)
    
    def _create_background(self, size: tuple[int, int], bg_color: np.ndarray, train_mode: bool) -> np.ndarray:
        """Create background canvas with optional noise.
        
        Args:
            size: (height, width) of background
            bg_color: Background color as RGB array
            train_mode: Whether to apply noise
            
        Returns:
            Background array of shape (height, width, 3)
        """
        background = np.full((*size, 3), bg_color, dtype=np.uint8)
        if train_mode:
            noise = np.random.normal(0, 2, background.shape).astype(np.uint8)
            background = cv2.add(background, noise)
        return background
    
    def _clean_document_edges(self, img: np.ndarray, bg_color: np.ndarray, train_mode: bool) -> np.ndarray:
        """Clean document edges by replacing black borders with background color.
        
        Args:
            img: Input image
            bg_color: Background color to use
            train_mode: Whether to use random threshold
            
        Returns:
            Cleaned image
        """
        threshold = self.rng.randint(25, 35) if train_mode else 30
        document_mask = np.any(img > threshold, axis=2)
        img_cleaned = img.copy()
        img_cleaned[~document_mask] = bg_color
        return img_cleaned
    
    def _get_placement_offsets(self, img_size: tuple[int, int], canvas_size: tuple[int, int], train_mode: bool) -> tuple[int, int]:
        """Calculate placement offsets with optional jitter.
        
        Args:
            img_size: (height, width) of image
            canvas_size: (height, width) of canvas
            train_mode: Whether to apply random jitter
            
        Returns:
            (y_offset, x_offset) tuple
        """
        h, w = img_size
        canvas_h, canvas_w = canvas_size
        
        y_offset = (canvas_h - h) // 2
        x_offset = (canvas_w - w) // 2
        
        if train_mode:
            y_offset += self.rng.randint(-5, 5)
            x_offset += self.rng.randint(-5, 5)
            
        return y_offset, x_offset
    
    def _rotate_with_background(self, img: np.ndarray, angle: float, train_mode: bool = False) -> np.ndarray:
        """Rotate image while preserving background and original canvas size.
        
        Args:
            img: Input image array of shape (H, W, C)
            angle: Rotation angle in degrees
            train_mode: Whether to apply additional randomization for training
            
        Returns:
            Rotated image with same dimensions as input
        """
        if angle == 0:
            return img

        h, w = img.shape[:2]
        original_size = (w, h)
        
        # Get background color and create canvas
        bg_color = self._get_background_color(img, train_mode)
        diagonal = int(np.ceil(np.sqrt(h*h + w*w)))
        background = self._create_background((diagonal, diagonal), bg_color, train_mode)
        
        # Clean document edges and place on background
        img_cleaned = self._clean_document_edges(img, bg_color, train_mode)
        y_offset, x_offset = self._get_placement_offsets((h, w), (diagonal, diagonal), train_mode)
        
        # Place image on background
        y_end = min(diagonal, y_offset + h)
        x_end = min(diagonal, x_offset + w)
        background[y_offset:y_end, x_offset:x_end] = img_cleaned[:y_end-y_offset, :x_end-x_offset]
        
        # Rotate with optional noise
        angle_noise = self.rng.uniform(-5, 5) if train_mode else 0
        M = cv2.getRotationMatrix2D((diagonal/2, diagonal/2), angle + angle_noise, 1.0)
        rotated = cv2.warpAffine(background, M, (diagonal, diagonal))
        
        # Apply random cropping in training mode
        if train_mode:
            crop_margin = int(diagonal * 0.1)  # 10% margin
            crop_x = self.rng.randint(0, crop_margin)
            crop_y = self.rng.randint(0, crop_margin)
            rotated = rotated[crop_y:diagonal-crop_y, crop_x:diagonal-crop_x]
        
        # Resize back to original size
        return cv2.resize(rotated, original_size, interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - rotation: Integer rotation class (0-7)
                - rotation_angle: Actual rotation angle in degrees
                - path: Path to the image file
        """
        img_path, _ = self.samples[idx]
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply random rotation for train, deterministic for val
        if self.split == "train":
            rotation = self.rng.randint(0, 7)  # 0=0°, 1=45°, 2=90°, ..., 7=315°
        else:
            # Use modulo to ensure even distribution of rotations in validation
            rotation = idx % 8
        
        angle = rotation * 45  # Each step is now 45 degrees
        img = self._rotate_with_background(img, angle, train_mode=self.split=="train")
        
        # Apply transforms (augmentation for train, just resize/normalize for val)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        return {
            "image": img,
            "rotation": rotation,
            "rotation_angle": angle,
            "path": str(img_path)
        } 