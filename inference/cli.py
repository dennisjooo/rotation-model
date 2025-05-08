"""Command-line interface for document rotation inference."""

import argparse
import cv2
from . import (
    load_onnx_model,
    predict_single_image,
    predict_patches,
    extract_patches,
    rotate_and_save_image,
    load_image
)

def main():
    """Main entry point for the document rotation inference script.
    
    Parses command line arguments, loads model and image, performs inference,
    and optionally saves the rotated image.
    
    Command line arguments:
        --model_path: Path to the ONNX model file (required)
        --image_path: Path to input image file (required)
        --output_path: Path to save rotated image (optional)
        --patch_size: Size of patches for patch-based inference (optional)
            If not provided, uses full-image inference
            
    Example usage:
        python -m inference.cli --model_path model.onnx --image_path doc.jpg --patch_size 224
    """
    parser = argparse.ArgumentParser(description='Document Rotation Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint or ONNX file')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save rotated image (optional)')
    parser.add_argument('--patch_size', type=int, default=None,
                       help='Size of patches for patch-based inference (optional)')
    args = parser.parse_args()
    
    # Load model and image
    session = load_onnx_model(args.model_path)
    image = cv2.imread(args.image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    if args.patch_size is None:
        # Regular full-image inference
        image_tensor = load_image(args.image_path)
        result = predict_single_image(session, image_tensor)
    else:
        # Patch-based inference
        patches, _ = extract_patches(image_rgb, args.patch_size)
        result = predict_patches(session, patches)
    
    # Print results
    print(f"\nPredicted rotation: {result.angle}°")
    print(f"Confidence: {result.confidence:.2%}")
    
    # Optionally save rotated image
    if args.output_path:
        rotate_and_save_image(image, result.angle, args.output_path)

if __name__ == '__main__':
    main() 