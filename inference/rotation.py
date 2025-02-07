"""Image rotation functionality."""

import cv2
import numpy as np

def rotate_and_save_image(image: np.ndarray, angle: int, output_path: str):
    """Rotate an image by the specified angle and save it to disk.
    
    Handles rotation while preserving image content and adjusting output dimensions
    appropriately for different rotation angles.
    
    Args:
        image: Input image as numpy array of shape (H, W, C)
        angle: Rotation angle in degrees (positive = counterclockwise)
        output_path: Path where the rotated image should be saved
        
    Notes:
        - For 90° and 270° rotations, output dimensions are swapped
        - Uses bilinear interpolation for rotation
        - Adjusts transformation matrix to maintain image centering
        - Saves output in same format as input image
        
    Raises:
        cv2.error: If image cannot be rotated or saved
        IOError: If output path is not writable
    """
    height, width = image.shape[:2]
    
    # For 90 and 270 degree rotations, swap width and height
    if angle in [90, 270]:
        new_width, new_height = height, width
    else:
        new_width, new_height = width, height
        
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Adjust translation part of the matrix for new dimensions
    if angle in [90, 270]:
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                             flags=cv2.INTER_LINEAR)
    
    # Save rotated image
    cv2.imwrite(output_path, rotated)
    print(f"Rotated image saved to: {output_path}") 