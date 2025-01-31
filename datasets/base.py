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
    
    def _rotate_with_background(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image while preserving background and original canvas size.
        
        Args:
            img: Input image array of shape (H, W, C)
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image with same dimensions as input
        """
        if angle == 0:
            return img
        
        # Store the dimensions for further calculations
        h, w = img.shape[:2]
        original_size = (w, h)
        
        # Calculate the size needed to contain the rotated image
        diagonal = int(np.ceil(np.sqrt(h*h + w*w)))
        
        # Create a square background with the mean color of the image
        bg_color = np.mean(img, axis=(0, 1)).astype(np.uint8)
        background = np.full((diagonal, diagonal, 3), bg_color, dtype=np.uint8)
        
        # Place the original image in the center of the background
        y_offset = (diagonal - h) // 2
        x_offset = (diagonal - w) // 2
        background[y_offset:y_offset+h, x_offset:x_offset+w] = img
        
        # Rotate around the center (with some noise as document rotation is not always perfect)
        M = cv2.getRotationMatrix2D((diagonal/2, diagonal/2), angle + self.rng.uniform(-5, 5), 1.0)
        rotated = cv2.warpAffine(background, M, (diagonal, diagonal))
        
        # Scale the rotated image back to original canvas size
        return cv2.resize(rotated, original_size, interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - rotation: Integer rotation class (0-15)
                - rotation_angle: Actual rotation angle in degrees
                - path: Path to the image file
        """
        img_path, _ = self.samples[idx]
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply random rotation for both train and val
        # For validation, we use a deterministic rotation based on the index
        if self.split == "train":
            rotation = self.rng.randint(0, 7)  # 0=0°, 1=45°, 2=90°, ..., 7=315°
        else:
            # Use modulo to ensure even distribution of rotations in validation
            rotation = idx % 8
        
        angle = rotation * 45  # Each step is now 45 degrees
        img = self._rotate_with_background(img, angle)
        
        # Apply transforms (augmentation for train, just resize/normalize for val)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        return {
            "image": img,
            "rotation": rotation,
            "rotation_angle": angle,
            "path": str(img_path)
        } 