import numpy as np
from omegaconf import OmegaConf
from .rotate_with_background import RotateWithBackground
from .document_transforms import DocumentTransforms

# Load the hydra config from configs/train.yaml
cfg = OmegaConf.load("configs/train.yaml")

# Initialize transforms used for inference
rotate_with_background = RotateWithBackground()
document_transforms = DocumentTransforms(image_size=cfg.data.image_size)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply preprocessing to prepare image for model input.
    
    Args:
        image: Input image array
        
    Returns:
        Preprocessed image tensor
    """
    # Calculate background color and clean borders using existing methods
    bg_color = rotate_with_background._calculate_background_color(image)
    img_cleaned = rotate_with_background._clean_document_borders(image, bg_color)
    
    # Create padded background
    background, (h, w), (y_offset, x_offset) = rotate_with_background._create_padded_background(
        img_cleaned, bg_color
    )
    
    # Place cleaned image on background
    background[y_offset:y_offset+h, x_offset:x_offset+w] = img_cleaned
    
    # Preprocess and add batch dimension
    return document_transforms.val_transform()(image=background)["image"].unsqueeze(0)