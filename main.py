#!/usr/bin/env python3

"""Inference script for Document Rotation Model.

This script loads either a PyTorch or ONNX model and performs inference
on a single input image, predicting its rotation angle and confidence score.
"""

import argparse
import cv2
import numpy as np
import torch
import onnxruntime as ort
from torchvision import transforms

def load_image(image_path: str, size: int = 384) -> torch.Tensor:
    """Load and preprocess an image for inference.
    
    Args:
        image_path: Path to the input image
        size: Size to resize the image to (default: 384)
        
    Returns:
        Preprocessed image tensor of shape (1, 3, size, size)
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate background color
    non_black_mask = np.any(img > 30, axis=2)
    mean_color = np.mean(img[non_black_mask], axis=0).astype(np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    bg_color = (0.1 * mean_color + 0.9 * white).astype(np.uint8)
    
    # Clean document borders
    document_mask = np.any(img > 30, axis=2)
    img_cleaned = img.copy()
    img_cleaned[~document_mask] = bg_color
    
    # Create padded background
    h, w = img_cleaned.shape[:2]
    diagonal = int(np.ceil(np.sqrt(h*h + w*w)))
    background = np.full((diagonal, diagonal, 3), bg_color, dtype=np.uint8)
    
    # Calculate center offsets
    y_offset = (diagonal - h) // 2
    x_offset = (diagonal - w) // 2
    
    # Place cleaned image on background
    background[y_offset:y_offset+h, x_offset:x_offset+w] = img_cleaned
    
    # Resize to target size
    img_resized = cv2.resize(background, (size, size), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess and add batch dimension
    return preprocess(img_resized).unsqueeze(0)

def load_torch_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.nn.Module:
    """Load PyTorch model.
    
    Args:
        model_path: Path to PyTorch model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded PyTorch model
    """
    from model import DocRotationModel
    
    # Initialize model
    model = DocRotationModel(num_classes=8)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        # Lightning checkpoint
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    elif 'model_state_dict' in checkpoint:
        # Our saved state dict
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    else:
        # Regular PyTorch checkpoint
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

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
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    return session

def main():
    parser = argparse.ArgumentParser(description='Document Rotation Model Inference')
    parser.add_argument('--model_type', type=str, choices=['torch', 'onnx'], required=True,
                       help='Type of model to use (torch or onnx)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint or ONNX file')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save rotated image (optional)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run inference on (for PyTorch model)')
    args = parser.parse_args()
    
    # Load and preprocess image
    image_tensor = load_image(args.image_path)
    
    # Load model and run inference
    if args.model_type == 'torch':
        # Load PyTorch model
        model = load_torch_model(args.model_path, args.device)
        image_tensor = image_tensor.to(args.device)
        
        # Run inference
        with torch.no_grad():
            logits, confidence = model(image_tensor)
        
        # Print the logits and confidence
        print(f"Logits: {logits}")
        print(f"Confidence: {confidence}")
        
        # Get predictions
        pred_class = torch.argmax(logits, dim=1).item()
        pred_angle = pred_class * 45
        confidence = torch.sigmoid(confidence).item()
        
    else:  # ONNX model
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
        
        # Get image center
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, -pred_angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                flags=cv2.INTER_LINEAR)
        
        # Save rotated image
        cv2.imwrite(args.output_path, rotated)
        print(f"Rotated image saved to: {args.output_path}")

if __name__ == '__main__':
    main() 