"""Rotation transform with background preservation for document images."""

import random
import cv2
import numpy as np
import torchvision.transforms as transforms
from albumentations.core.transforms_interface import ImageOnlyTransform
from .auto_crop import AutoCropBars


class RotateWithBackground(ImageOnlyTransform):
    """Albumentations transform to rotate images while preserving background."""
    
    def __init__(
        self,
        angle: float = 0.0,
        is_train: bool = False,
        random_state: int = 42,
        always_apply: bool = False,
        p: float = 1.0
    ):
        """Initialize the transform.
        
        Args:
            angle: Base rotation angle in degrees
            is_train: Whether to add random noise in training
            random_state: Random seed for reproducibility
            always_apply: Whether to always apply the transform
            p: Probability of applying the transform
        """
        super().__init__(p=1.0 if always_apply else p)
        self.angle = angle
        self.is_train = is_train
        self.rng = random.Random(random_state)
    
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Apply the transform to the image.
        
        Args:
            image: Input image array
            
        Returns:
            Rotated image array
        """
        if self.angle == 0:
            return image
            
        # Get original dimensions for final resize
        original_size = (image.shape[1], image.shape[0])  # (w, h)
        
        # Calculate background color
        bg_color = self._calculate_background_color(image)
        
        # Clean document borders
        img_cleaned = self._clean_document_borders(image, bg_color)
        
        # Create padded background
        background, (h, w), (y_offset, x_offset) = self._create_padded_background(
            img_cleaned, bg_color
        )
        
        # Place cleaned image on background
        background[y_offset:y_offset+h, x_offset:x_offset+w] = img_cleaned
        
        # Apply rotation
        rotated = self._apply_rotation(background, self.angle, self.is_train, bg_color)
        
        # For right angles (90, 180, 270), apply autocropping
        if self.angle in [90, 180, 270]:
            rotated = AutoCropBars(p=1.0)(image=rotated)["image"]
        
        # Return to original size
        return cv2.resize(rotated, original_size, interpolation=cv2.INTER_LINEAR)
    
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
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
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
    
    def get_transform_init_args_names(self):
        """Get the transform initialization argument names."""
        return ("angle", "is_train", "random_state", "always_apply", "p") 
    