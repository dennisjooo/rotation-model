"""Document-specific transform pipelines for training and validation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DocumentTransforms:
    """Document-specific transform pipelines for training and validation."""
    
    def __init__(self, image_size: int = 384):
        """Initialize transform pipelines.
        
        Args:
            image_size: Target image size for resizing
        """
        self.image_size = image_size
    
    def train_transform(self) -> A.Compose:
        """Get training data augmentations.
        
        Returns:
            Composed augmentation pipeline
        """
        return A.Compose([
            # Basic resize - always applied
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            
            # Document-specific quality degradation - moderate probability
            A.OneOf([
                A.ImageCompression(quality_range=(50, 95), p=1.0),  # JPEG artifacts
                A.Downscale(scale_range=(0.8, 0.99), p=1.0),  # Resolution loss
                A.Blur(blur_limit=(3, 5), p=1.0),  # General blur
            ], p=0.4),
            
            # Realistic document noise - moderate probability
            A.OneOf([
                A.GaussNoise(std_range=(0.0, 0.25), p=1.0),  # Film grain
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),  # Scanner noise
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),  # Camera sensor noise
            ], p=0.3),
            
            # Document aging effects - lower probability
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(-0.1, 0.1),
                    p=1.0
                ),  # Fading
                A.ColorJitter(
                    brightness=(0.9, 1.1),
                    contrast=(0.9, 1.1),
                    saturation=(0.9, 1.1),
                    hue=(-0.1, 0.1),
                    p=1.0
                ),  # Color variation
                A.ToSepia(p=1.0),  # Age yellowing
            ], p=0.2),
            
            # Document lighting/scanning effects - moderate probability
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    p=1.0
                ),  # Page shadows
                A.RandomBrightnessContrast(
                    brightness_limit=(0, 0.1),
                    contrast_limit=(0, 0.1),
                    p=1.0
                ),  # Scanner light variation
                A.CLAHE(clip_limit=2.0, p=1.0),  # Local contrast enhancement
            ], p=0.3),
            
            # Document defects - lower probability
            A.OneOf([
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(8, 20),
                    hole_width_range=(8, 20),
                    fill=255,
                    p=1.0
                ),  # Small tears/holes
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=(-0.2, 0.2),
                    p=1.0
                ),  # Page warping
                A.ElasticTransform(
                    alpha=1.5,  # Deformation strength
                    sigma=12,   # Gaussian filter parameter
                    p=1.0
                ),  # Paper wrinkles
            ], p=0.15),
            
            # Edge effects - lower probability
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(0.1, 0.3),
                    p=1.0
                ),  # Edge lighting effects
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),  # Edge enhancement
            ], p=0.15),
            
            # Final normalization - always applied
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            ToTensorV2(),
        ])

    def val_transform(self) -> A.Compose:
        """Get validation/test data transforms.
        
        Returns:
            Composed transformation pipeline
        """
        return A.Compose([
            A.Resize(
                height=self.image_size,
                width=self.image_size,
                p=1.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]) 