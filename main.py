"""Inference script for Document Rotation Model.

This script loads either a ONNX model and performs inference
on a single input image, predicting its rotation angle and confidence score.
"""

import argparse
import cv2
import numpy as np
import torch
import onnxruntime as ort
from transforms import preprocess

def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image for inference.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Preprocessed image tensor of shape (1, 3, size, size)
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess image
    return preprocess(img)      

def load_onnx_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model.
    
    Args:
        model_path: Path to ONNX model
        
    Returns:
        ONNX Runtime inference session
    """
    # Create ONNX Runtime session
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    return session

def main():
    parser = argparse.ArgumentParser(description='Document Rotation Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint or ONNX file')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save rotated image (optional)')
    args = parser.parse_args()
    
    # Load and preprocess image
    image_tensor = load_image(args.image_path)
    
    # Load ONNX model
    session = load_onnx_model(args.model_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: image_tensor.numpy()})
    logits, confidence = outputs
    
    # Print the logits and confidence
    print(f"Logits: {logits}")
    print(f"Confidence: {confidence}")
    
    # Get predictions
    pred_class = np.argmax(logits, axis=1)[0]
    pred_angle = pred_class * 45
    confidence = 1 / (1 + np.exp(-confidence[0][0]))  # sigmoid
    
    # Print results
    print(f"\nPredicted rotation: {pred_angle}°")
    print(f"Confidence: {confidence:.2%}")
    
    # Optionally save rotated image
    if args.output_path:
        # Load original image
        image = cv2.imread(args.image_path)
        
        # Get image center and dimensions
        height, width = image.shape[:2]
        
        # For 90 and 270 degree rotations, swap width and height
        if pred_angle in [90, 270]:
            new_width, new_height = height, width
        else:
            new_width, new_height = width, height
            
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, -pred_angle, 1.0)
        
        # Adjust translation part of the matrix for new dimensions
        if pred_angle in [90, 270]:
            rotation_matrix[0, 2] += (new_width - width) / 2
            rotation_matrix[1, 2] += (new_height - height) / 2
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                 flags=cv2.INTER_LINEAR)
        
        # Save rotated image
        cv2.imwrite(args.output_path, rotated)
        print(f"Rotated image saved to: {args.output_path}")

if __name__ == '__main__':
    main() 