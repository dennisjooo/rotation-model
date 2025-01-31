"""Transforms for document rotation datasets.

This module contains the transform pipelines for training and validation/testing.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 384) -> A.Compose:
    """Get training data augmentations.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed augmentation pipeline
    """
    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size,
            p=1.0
        ),
        A.OneOf([
            A.GaussNoise(var_limit=50.0),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(),
            A.CLAHE(),
        ], p=0.5),
        A.OneOf([
            A.ImageCompression(quality_lower=50),
            A.Downscale(scale_range=(0.7, 0.9)),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomShadow(p=0.2, shadow_roi=(0, 0.5, 1, 1)),
            A.RandomSunFlare(p=0.2),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 384) -> A.Compose:
    """Get validation/test data transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transformation pipeline
    """
    return A.Compose([
        A.Resize(
            height=image_size,
            width=image_size,
            p=1.0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]) 