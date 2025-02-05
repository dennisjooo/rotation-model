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
    
    def _calculate_background_color(self, img: np.ndarray) -> np.ndarray:
        """Calculate background color by blending mean document color with white.
        
        Args:
            img: Input image array of shape (H, W, C)
            
        Returns:
            Background color as RGB array
        """
        # Calculate mean color of non-black pixels
        non_black_mask = np.any(img > 30, axis=2)
        mean_color = np.mean(img[non_black_mask], axis=0).astype(np.uint8)
        
        # Blend with white (10% mean + 90% white)
        white = np.array([255, 255, 255], dtype=np.uint8)
        return (0.1 * mean_color + 0.9 * white).astype(np.uint8)
    
    def _clean_document_borders(self, img: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
        """Replace black borders with background color.
        
        Args:
            img: Input image array
            bg_color: Background color to use for borders
            
        Returns:
            Cleaned image with black borders replaced
        """
        document_mask = np.any(img > 30, axis=2)
        img_cleaned = img.copy()
        img_cleaned[~document_mask] = bg_color
        return img_cleaned
    
    def _create_padded_background(
        self, 
        img: np.ndarray, 
        bg_color: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Create padded background and calculate placement offsets.
        
        Args:
            img: Input image array
            bg_color: Background color to use
            
        Returns:
            Tuple of:
                - Padded background array
                - Original dimensions (h, w)
                - Offset coordinates (y_offset, x_offset)
        """
        h, w = img.shape[:2]
        diagonal = int(np.ceil(np.sqrt(h*h + w*w)))
        
        # Create square background
        background = np.full((diagonal, diagonal, 3), bg_color, dtype=np.uint8)
        
        # Calculate center offsets
        y_offset = (diagonal - h) // 2
        x_offset = (diagonal - w) // 2
        
        return background, (h, w), (y_offset, x_offset)
    
    def _apply_rotation(
        self,
        img: np.ndarray, 
        angle: float,
        is_train: bool,
        bg_color: np.ndarray
    ) -> np.ndarray:
        """Apply rotation to image with optional training noise.
        
        Args:
            img: Input image array
            angle: Base rotation angle in degrees
            is_train: Whether to add random noise in training
            bg_color: Background color for cleaning artifacts
            
        Returns:
            Rotated image array
        """
        h, w = img.shape[:2]
        rotation_angle = angle + (self.rng.uniform(-5, 5) if is_train else 0)
        
        M = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # Clean up rotation artifacts
        black_pixels = np.all(rotated < 30, axis=2)
        rotated[black_pixels] = bg_color
        
        return rotated
    
    def _rotate_with_background(self, img: np.ndarray, angle: float, is_train: bool = False) -> np.ndarray:
        """Rotate image while preserving background and original canvas size.
        
        Args:
            img: Input image array of shape (H, W, C)
            angle: Rotation angle in degrees
            is_train: Whether in training mode (adds random rotation noise if True)
            
        Returns:
            Rotated image with same dimensions as input
        """
        if angle == 0:
            return img
            
        # Get original dimensions for final resize
        original_size = (img.shape[1], img.shape[0])  # (w, h)
        
        # Calculate background color
        bg_color = self._calculate_background_color(img)
        
        # Clean document borders
        img_cleaned = self._clean_document_borders(img, bg_color)
        
        # Create padded background
        background, (h, w), (y_offset, x_offset) = self._create_padded_background(
            img_cleaned, bg_color
        )
        
        # Place cleaned image on background
        background[y_offset:y_offset+h, x_offset:x_offset+w] = img_cleaned
        
        # Apply rotation
        rotated = self._apply_rotation(background, angle, is_train, bg_color)
        
        # Return to original size
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
        
        # Apply random rotation for train, deterministic for val/test
        if self.split == "train":
            rotation = self.rng.randint(0, 7)  # 0=0°, 1=45°, 2=90°, ..., 7=315°
        else:
            # Use modulo to ensure even distribution of rotations in validation
            rotation = idx % 8
        
        angle = rotation * 45  # Each step is now 45 degrees
        img = self._rotate_with_background(img, angle, is_train=(self.split == "train"))
        
        # Apply transforms (augmentation for train, just resize/normalize for val)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        
        return {
            "image": img,
            "rotation": rotation,
            "rotation_angle": angle,
            "path": str(img_path)
        } 