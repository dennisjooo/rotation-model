"""Document-specific transform pipelines for training and validation.

This module provides a comprehensive set of image transformations specifically designed for document images.
It includes realistic augmentations that simulate various document conditions like aging, photocopying,
physical defects, and scanning artifacts.

The transforms are organized into logical groups (basic, quality, noise, aging, etc.) and can be combined
into training and validation pipelines. The training pipeline includes all augmentations while the 
validation pipeline only includes basic resizing and normalization.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DocumentTransforms:
    """Document-specific transform pipelines for training and validation.
    
    This class provides methods to create augmentation pipelines that simulate real-world document 
    conditions and variations. It includes transforms for:
    - Basic image processing (resizing)
    - Quality degradation (compression, blur)
    - Document noise (film grain, scanner noise)
    - Aging effects (fading, yellowing)
    - Lighting variations (shadows, contrast)
    - Physical defects (tears, wrinkles)
    - Photocopier effects (B&W and color variations)
    - Edge effects (enhancement, lighting)
    
    The transforms can be accessed individually or combined into training/validation pipelines.
    """
    
    def __init__(self, image_size: int = 384):
        """Initialize transform pipelines.
        
        Args:
            image_size: Target size (both height and width) that images will be resized to.
                       Default is 384 pixels.
        """
        self.image_size = image_size
    
    def _get_basic_transforms(self) -> list:
        """Get basic image transformations.
        
        Creates fundamental transforms that should be applied to all images,
        currently just resizing to the target dimensions.
        
        Returns:
            List containing the basic Albumentations transforms
        """
        return [
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
        ]
    
    def _get_quality_degradation(self) -> A.OneOf:
        """Get document quality degradation transforms.
        
        Creates transforms that simulate quality loss from digital processing:
        - JPEG compression artifacts
        - Resolution downscaling
        - General image blur
        
        Returns:
            OneOf composition of quality degradation transforms with 0.25 probability
        """
        return A.OneOf([
            A.ImageCompression(quality_range=(65, 95), p=1.0),  # JPEG artifacts - increased min quality
            A.Downscale(scale_range=(0.85, 0.99), p=1.0),  # Resolution loss - reduced scale range
            A.Blur(blur_limit=(2, 4), p=1.0),  # General blur - reduced blur range
        ], p=0.25)
    
    def _get_document_noise(self) -> A.OneOf:
        """Get realistic document noise transforms.
        
        Creates transforms that add realistic noise patterns found in scanned/photographed documents:
        - Film grain noise
        - Scanner sensor noise
        - Camera ISO noise
        
        Returns:
            OneOf composition of noise transforms with 0.2 probability
        """
        return A.OneOf([
            A.GaussNoise(std_range=(0.0, 0.15), p=1.0),  # Film grain - reduced intensity
            A.MultiplicativeNoise(multiplier=(0.97, 1.03), p=1.0),  # Scanner noise - reduced range
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),  # Camera sensor noise - reduced intensity
        ], p=0.2)
    
    def _get_aging_effects(self) -> A.OneOf:
        """Get document aging effect transforms.
        
        Creates transforms that simulate natural document aging:
        - Fading through brightness/contrast changes
        - Color variations over time
        - Sepia toning/yellowing
        
        Returns:
            OneOf composition of aging transforms with 0.15 probability
        """
        return A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.05, 0.05),
                contrast_limit=(-0.05, 0.05),
                p=1.0
            ),  # Fading - reduced range
            A.ColorJitter(
                brightness=(0.95, 1.05),
                contrast=(0.95, 1.05),
                saturation=(0.95, 1.05),
                hue=(-0.05, 0.05),
                p=1.0
            ),  # Color variation - reduced ranges
            A.ToSepia(p=1.0),  # Age yellowing
        ], p=0.15)
    
    def _get_lighting_effects(self) -> A.OneOf:
        """Get document lighting and scanning effect transforms.
        
        Creates transforms that simulate lighting variations during scanning/photography:
        - Page shadows from uneven lighting
        - Scanner light variations
        - Local contrast enhancements
        
        Returns:
            OneOf composition of lighting transforms with 0.2 probability
        """
        return A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=1,  # Reduced from (1,2) to just 1
                shadow_dimension=5,
                p=1.0
            ),  # Page shadows
            A.RandomBrightnessContrast(
                brightness_limit=(0, 0.05),
                contrast_limit=(0, 0.05),
                p=1.0
            ),  # Scanner light variation - reduced intensity
            A.CLAHE(clip_limit=1.5, p=1.0),  # Local contrast enhancement - reduced clip limit
        ], p=0.2)
    
    def _get_defect_effects(self) -> A.OneOf:
        """Get document physical defect transforms.
        
        Creates transforms that simulate physical document damage:
        - Small tears and holes
        - Page warping from moisture/heat
        - Paper wrinkles and creases
        
        Returns:
            OneOf composition of defect transforms with 0.1 probability
        """
        return A.OneOf([
            A.CoarseDropout(
                num_holes_range=(1, 4),  # Reduced holes
                hole_height_range=(4, 12),  # Smaller holes
                hole_width_range=(4, 12),
                fill=255,
                p=1.0
            ),  # Small tears/holes
            A.GridDistortion(
                num_steps=5,
                distort_limit=(-0.1, 0.1),  # Reduced distortion
                p=1.0
            ),  # Page warping
            A.ElasticTransform(
                alpha=1.0,  # Reduced deformation strength
                sigma=8,    # Reduced Gaussian filter parameter
                p=1.0
            ),  # Paper wrinkles
        ], p=0.1)
    
    def _get_photocopier_effects(self) -> A.OneOf:
        """Get photocopier effect transforms including color and B&W variations.
        
        Creates transforms that simulate different types of photocopies:
        - Color photocopier effects (contrast, noise, blur)
        - B&W photocopier effects (grayscale, high contrast)
        - Low quality B&W copies with artifacts
        
        Returns:
            OneOf composition of photocopier transforms with 0.3 probability
        """
        return A.OneOf([
            # Color photocopier effects
            A.Sequential([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0),
                    contrast_limit=(0.1, 0.2),
                    p=1.0
                ),  # High contrast typical of photocopies - reduced intensity
                A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),  # Copier noise - reduced variance
                A.Blur(blur_limit=(1, 2), p=1.0),  # Slight blur from scanning - reduced range
            ], p=1.0),
            # B&W photocopier effects
            A.Sequential([
                A.ToGray(p=1.0),  # Convert to grayscale
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0),
                    contrast_limit=(0.2, 0.4),  # Reduced contrast range
                    p=1.0
                ),
                A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),  # Reduced noise
                A.Blur(blur_limit=(1, 2), p=1.0),
            ], p=1.0),
            # Low quality B&W photocopy with artifacts
            A.Sequential([
                A.ToGray(p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, -0.1),
                    contrast_limit=(0.2, 0.4),
                    p=1.0
                ),
                A.GaussNoise(var_limit=(15.0, 35.0), p=1.0),
                A.CoarseDropout(
                    num_holes_range=(5, 10),  # Reduced number of holes
                    hole_height_range=(1, 2),
                    hole_width_range=(1, 2),
                    fill=255,
                    p=0.5  # Reduced probability
                ),  # Small white speckles
                A.Downscale(scale_range=(0.8, 0.9), p=1.0),  # Less aggressive downscaling
            ], p=1.0),
        ], p=0.3)
    
    def _get_edge_effects(self) -> A.OneOf:
        """Get edge effect transforms.
        
        Creates transforms that modify document edge appearance:
        - Edge lighting variations
        - Edge sharpness enhancement
        
        Returns:
            OneOf composition of edge effect transforms with 0.1 probability
        """
        return A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(0.05, 0.15),
                contrast_limit=(0.05, 0.15),
                p=1.0
            ),  # Edge lighting effects - reduced intensity
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 0.8), p=1.0),  # Edge enhancement - reduced intensity
        ], p=0.1)
    
    def _get_final_transforms(self) -> list:
        """Get final normalization transforms.
        
        Creates transforms for standardizing the image for neural network input:
        - Pixel value normalization to ImageNet statistics
        - Conversion to PyTorch tensor
        
        Returns:
            List of final normalization transforms
        """
        return [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            ),
            ToTensorV2(),
        ]
    
    def train_transform(self) -> A.Compose:
        """Get training data augmentations.
        
        Creates a comprehensive transform pipeline for training that includes
        all augmentations to simulate various real-world document conditions.
        
        Returns:
            Composed augmentation pipeline with all transforms
        """
        transforms = (
            self._get_basic_transforms() +
            [self._get_quality_degradation()] +
            [self._get_document_noise()] +
            [self._get_aging_effects()] +
            [self._get_lighting_effects()] +
            [self._get_defect_effects()] +
            [self._get_photocopier_effects()] +
            [self._get_edge_effects()] +
            self._get_final_transforms()
        )
        return A.Compose(transforms)

    def val_transform(self) -> A.Compose:
        """Get validation/test data transforms.
        
        Creates a minimal transform pipeline for validation/testing that only
        includes essential preprocessing (resizing and normalization).
        
        Returns:
            Composed transformation pipeline with basic transforms only
        """
        return A.Compose(
            self._get_basic_transforms() +
            self._get_final_transforms()
        ) 