"""Common utilities for document rotation inference."""

import cv2
import torch
from typing import Dict, NamedTuple
from transforms import preprocess

class PredictionResult(NamedTuple):
    """Container for model prediction results.
    
    Attributes:
        angle: Predicted rotation angle in degrees (0°, 45°, 90°, ..., 315°)
        confidence: Confidence score for the prediction (0.0 to 1.0)
        voting_results: Optional dictionary containing detailed patch voting statistics.
            Only populated for patch-based inference. Keys are angles in degrees,
            values are dictionaries containing:
            - raw_count: Number of patches voting for this angle
            - weighted_count: Sum of confidence-weighted votes for this angle
            - confidence: Average confidence for patches predicting this angle
    """
    angle: int
    confidence: float
    voting_results: Dict[int, float] = None

def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image for model inference.
    
    Args:
        image_path: Path to the input image file. Must be readable by OpenCV.
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, size, size) where
            size is the model's expected input dimensions. The tensor is normalized
            and converted to the appropriate format for model inference.
            
    Raises:
        FileNotFoundError: If the image file cannot be found
        cv2.error: If the image cannot be loaded or processed
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess image
    return preprocess(img) 