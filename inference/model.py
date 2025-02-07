"""Model loading and inference functionality."""

import numpy as np
import torch
import onnxruntime as ort
from .utils import PredictionResult

def load_onnx_model(model_path: str) -> ort.InferenceSession:
    """Load and initialize an ONNX model for inference.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        ort.InferenceSession: Initialized ONNX Runtime session ready for inference
        
    Raises:
        FileNotFoundError: If the model file cannot be found
        ort.core.session.InvalidArgument: If the model is invalid or corrupted
        
    Notes:
        - Currently uses CPU execution provider only
        - Model is expected to have inputs and outputs compatible with the
          document rotation prediction task
    """
    # Create ONNX Runtime session
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    return session

def predict_single_image(session: ort.InferenceSession, image_tensor: torch.Tensor) -> PredictionResult:
    """Run inference on a single full-image tensor.
    
    Performs forward pass through the model to predict rotation angle and confidence
    for the entire image at once.
    
    Args:
        session: Initialized ONNX Runtime session
        image_tensor: Preprocessed image tensor of shape (1, 3, H, W)
        
    Returns:
        PredictionResult containing:
            - angle: Predicted rotation angle in degrees
            - confidence: Model's confidence in the prediction (0.0 to 1.0)
            - voting_results: None for single-image inference
            
    Notes:
        - Expects model to output logits and confidence scores
        - Converts logits to angle prediction using argmax
        - Applies sigmoid to raw confidence score
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_tensor.numpy()})
    logits, confidence = outputs
    pred_class = np.argmax(logits, axis=1)[0]
    confidence_score = 1 / (1 + np.exp(-confidence[0][0]))  # sigmoid
    
    return PredictionResult(
        angle=pred_class * 45,
        confidence=confidence_score
    ) 