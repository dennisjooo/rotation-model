"""Patch-based inference functionality."""

import numpy as np
import torch
import onnxruntime as ort
from typing import List, Tuple
from transforms import preprocess
from .utils import PredictionResult

def extract_patches(image: np.ndarray, patch_size: int) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
    """Extract and preprocess patches from an input image.
    
    Divides the input image into patches of the specified size, handling edge cases
    by padding if necessary. Each patch is preprocessed for model inference.
    
    Args:
        image: Input image as numpy array of shape (H, W, C)
        patch_size: Size of patches to extract (both width and height)
        
    Returns:
        Tuple containing:
            - List[torch.Tensor]: List of preprocessed patch tensors, each of shape
              (1, 3, patch_size, patch_size)
            - Tuple[int, int]: Grid dimensions (n_rows, n_cols) indicating the layout
              of the extracted patches
              
    Notes:
        - Patches at image edges that would be smaller than patch_size are padded
          with zeros to maintain consistent dimensions
        - The number of patches is determined by ceil(H/patch_size) * ceil(W/patch_size)
    """
    height, width = image.shape[:2]
    
    # Calculate number of patches in each dimension
    n_patches_h = height // patch_size + (1 if height % patch_size else 0)
    n_patches_w = width // patch_size + (1 if width % patch_size else 0)
    
    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Extract patch coordinates
            y_start = i * patch_size
            y_end = min((i + 1) * patch_size, height)
            x_start = j * patch_size
            x_end = min((j + 1) * patch_size, width)
            
            # Extract and preprocess patch
            patch = image[y_start:y_end, x_start:x_end]
            
            # Pad if necessary
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            patch_tensor = preprocess(patch)
            patches.append(patch_tensor)
    
    return patches, (n_patches_h, n_patches_w)

def predict_patches(session: ort.InferenceSession, patches: List[torch.Tensor]) -> PredictionResult:
    """Run inference on image patches and combine results using weighted voting.
    
    Performs patch-based inference by:
    1. Running prediction on each patch independently
    2. Collecting votes and confidence scores from all patches
    3. Using confidence-weighted voting to determine final prediction
    4. Providing detailed voting statistics
    
    Args:
        session: Initialized ONNX Runtime session
        patches: List of preprocessed patch tensors, each of shape (1, 3, H, W)
        
    Returns:
        PredictionResult containing:
            - angle: Final predicted rotation angle based on weighted voting
            - confidence: Average confidence across all patches
            - voting_results: Dictionary with detailed voting statistics for each angle
            
    Notes:
        - Prints detailed voting statistics to console
        - Uses both raw counts and confidence-weighted counts
        - Confidence scores are used to weight patch votes
        - Final angle is determined by highest weighted vote count
    """
    print(f"Processing {len(patches)} patches...")
    
    # Collect predictions for all patches
    patch_predictions = []
    patch_confidences = []
    input_name = session.get_inputs()[0].name
    
    for patch in patches:
        outputs = session.run(None, {input_name: patch.numpy()})
        logits, confidence = outputs
        pred = np.argmax(logits, axis=1)[0]
        conf = 1 / (1 + np.exp(-confidence[0][0]))  # sigmoid
        patch_predictions.append(pred)
        patch_confidences.append(conf)
    
    # Calculate both raw counts and confidence-weighted counts
    raw_counts = np.zeros(8)  # 8 possible rotation angles (0, 45, 90, ..., 315)
    weighted_counts = np.zeros(8)
    for pred, conf in zip(patch_predictions, patch_confidences):
        raw_counts[pred] += 1
        weighted_counts[pred] += conf
    
    pred_class = np.argmax(weighted_counts)
    confidence_score = np.mean(patch_confidences)
    
    # Create voting results dictionary with both raw and weighted counts
    voting_results = {
        angle * 45: {
            'raw_count': int(raw_counts[angle]),
            'weighted_count': weighted_counts[angle],
            'confidence': weighted_counts[angle] / raw_counts[angle] if raw_counts[angle] > 0 else 0
        }
        for angle in range(8)
    }
    
    print("\nPatch voting results:")
    print(f"{'Angle':>5} | {'Raw Count':>9} | {'Weighted Count':>13} | {'Avg Confidence':>13}")
    print("-" * 50)
    for angle, counts in voting_results.items():
        print(f"{angle:>5}° | {counts['raw_count']:>9} | {counts['weighted_count']:>13.2f} | {counts['confidence']:>12.2%}")
    
    print(f"\nTotal patches: {len(patches)}")
    print(f"Sum of raw counts: {int(raw_counts.sum())}")
    print(f"Sum of weighted counts: {weighted_counts.sum():.2f}")
    print(f"Average confidence across all patches: {confidence_score:.2%}")
    
    return PredictionResult(
        angle=pred_class * 45,
        confidence=confidence_score,
        voting_results=voting_results
    ) 