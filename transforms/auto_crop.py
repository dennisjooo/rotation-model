"""Auto-cropping transform for document images."""

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class AutoCropBars(ImageOnlyTransform):
    """Albumentations transform to automatically crop black and white bars from images."""
    
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        """Initialize the transform.
        
        Args:
            always_apply: Whether to always apply the transform
            p: Probability of applying the transform
        """
        super().__init__(p=1.0 if always_apply else p)
    
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Apply the transform to the image.
        
        Args:
            image: Input image array
            
        Returns:
            Cropped image array
        """
        # First crop black bars
        image = self.crop_black_bars(image)
        
        # Then crop white bars
        image = self.crop_white_bars(image)
        
        return image
    
    def crop_black_bars(self, image: np.ndarray) -> np.ndarray:
        """Crops black bars from an image.
        Taken from https://stackoverflow.com/a/30540322 modified to use Otsu's thresholding.

        Args:
            image: The input image array

        Returns:
            The cropped image array
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's thresholding
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the pixels that are not black (i.e., their value is above the threshold)
        rows = np.where(np.max(thresholded, 0) > 0)[0]
        cols = np.where(np.max(thresholded, 1) > 0)[0]

        # If there are non-black pixels
        if rows.size and cols.size:
            # Crop the image based on the non-black pixels
            image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
        else:
            image = image[:1, :1]

        return image

    def crop_white_bars(self, image: np.ndarray) -> np.ndarray:
        """Crops white bars from an image. Uses Otsu and Inverted Binary Thresholding.

        Args:
            image: The input image array

        Returns:
            The cropped image array
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding with inverted binary
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find the pixels that are white (i.e., their value is above the threshold)
        rows = np.where(np.max(thresholded, 0) > 0)[0]
        cols = np.where(np.max(thresholded, 1) > 0)[0]

        # If there are non-white pixels
        if rows.size and cols.size:
            # Crop the image based on the non-black or non-white pixels
            image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
        else:
            # If no non-white pixels are found, return a 1x1 pixel image
            image = image[:1, :1]

        return image

    def get_transform_init_args_names(self):
        """Get the transform initialization argument names."""
        return ("always_apply", "p") 