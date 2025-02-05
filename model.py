"""Document Rotation Model for classifying document orientations.

This module implements a lightweight deep learning model for document rotation classification
using MobileNetV3-small as the backbone, enhanced with CoordConv and selective attention
for better spatial awareness and feature extraction.

The model architecture consists of:
- CoordConv input layer for spatial awareness
- MobileNetV3-small backbone with multi-scale feature extraction
- CBAM attention modules at different scales
- Multi-scale feature pooling and fusion
- Separate classification and confidence prediction heads

Key Features:
- Efficient lightweight architecture
- Enhanced spatial awareness through CoordConv
- Multi-scale feature extraction and attention
- Skip connections for better gradient flow
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch import Tensor
from typing import Tuple

class DocRotationModel(nn.Module):
    """Lightweight document rotation classification model.
    
    This model combines MobileNetV3-small backbone with CoordConv and selective attention
    to classify document rotations into 8 classes (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    and predict confidence scores.
    
    Args:
        num_classes (int): Number of rotation classes to predict. Defaults to 8.
    """
    
    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        
        # Enhanced spatial-aware input processing with CoordConv
        self.coord_conv = nn.Sequential(
            CoordConv(3, 3, kernel_size=3, stride=1, padding=1),  # Add coordinate channels
            nn.BatchNorm2d(3),                                    # Normalize features
            nn.Hardswish(inplace=True)                           # Non-linear activation
        )
        
        # Load pretrained MobileNetV3-small backbone
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Split backbone into three scales for hierarchical feature extraction
        self.early_features = nn.Sequential(*list(backbone.features[:2]))     # Low-level features (16 channels)
        self.mid_features = nn.Sequential(*list(backbone.features[2:8]))      # Mid-level features (24->48 channels)
        self.late_features = nn.Sequential(*list(backbone.features[8:]))      # High-level features (576 channels)
        
        # Add CBAM attention at each scale with appropriate reduction ratios
        self.early_attn = CBAM(16, reduction=4)    # Strong attention for early features
        self.mid_attn = CBAM(48, reduction=8)      # Balanced attention for mid features
        self.late_attn = CBAM(576, reduction=16)   # Light attention for semantic features
        
        # Multi-scale pooling for capturing both global and local patterns
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.ModuleDict({
            'early': nn.AdaptiveAvgPool2d((4, 4)),  # Fine local details
            'mid': nn.AdaptiveAvgPool2d((2, 2))     # Medium-scale patterns
        })
        
        # Calculate total feature dimensions after concatenation
        total_features = (
            576 +           # Global semantic features
            16 * 4 * 4 +   # Early local features (16 channels x 4x4)
            48 * 2 * 2     # Mid local features (48 channels x 2x2)
        )
        
        # Feature fusion network with skip connections for better gradient flow
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            SkipBlock(256, 128),
            nn.LayerNorm(128),
            nn.Hardswish(inplace=True)
        )
        
        # Classification head for rotation prediction
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Confidence prediction head with stronger regularization
        self.confidence = nn.Sequential(
            nn.Linear(128, 32),  # Reduced capacity
            nn.LayerNorm(32),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.3),  # Higher dropout for confidence
            nn.Linear(32, 1)
        )
        
        # Initialize model weights
        self.init_weights()
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with enhanced feature extraction and fusion.
        
        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple[Tensor, Tensor]: Tuple containing:
                - Rotation logits of shape (B, num_classes)
                - Confidence scores of shape (B, 1)
        """
        # Initial spatial-aware feature extraction
        x = self.coord_conv(x)
        
        # Multi-scale feature extraction with attention
        early_feat = self.early_attn(self.early_features(x))      # Extract and attend to low-level features
        mid_feat = self.mid_attn(self.mid_features(early_feat))   # Extract and attend to mid-level features
        late_feat = self.late_attn(self.late_features(mid_feat))  # Extract and attend to high-level features
        
        # Multi-scale feature pooling
        pooled_features = []
        
        # Global semantic context
        global_feat = self.global_pool(late_feat).flatten(1)
        pooled_features.append(global_feat)
        
        # Local context at different scales
        pooled_features.append(self.local_pool['early'](early_feat).flatten(1))  # Fine details
        pooled_features.append(self.local_pool['mid'](mid_feat).flatten(1))      # Medium patterns
        
        # Feature fusion and prediction
        fused = torch.cat(pooled_features, dim=1)
        fused = self.fusion(fused)
        
        return self.classifier(fused), self.confidence(fused)
    
    def init_weights(self) -> None:
        """Initialize model weights with improved schemes.
        
        Uses:
        - Kaiming initialization for convolutional layers
        - Xavier initialization for linear layers
        - Constant initialization for normalization layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SkipBlock(nn.Module):
    """Lightweight skip connection block for better gradient flow.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """
    
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_features, out_features)
        )
        # Optional downsampling if dimensions don't match
        self.downsample = None if in_features == out_features else nn.Linear(in_features, out_features)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        identity = self.downsample(x) if self.downsample else x
        return self.block(x) + identity


class CoordConv(nn.Module):
    """Enhanced CoordConv layer with better coordinate encoding.
    
    Adds coordinate channels to input features for better spatial awareness.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        **kwargs: Additional arguments for the convolution layer
    """
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with coordinate channel addition."""
        batch_size, _, h, w = x.shape
        
        # Create normalized coordinate grid
        xx = torch.linspace(-1, 1, w, device=x.device)
        yy = torch.linspace(-1, 1, h, device=x.device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        
        # Add coordinate channels to input
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return self.conv(torch.cat([x, coords], dim=1))


class CBAM(nn.Module):
    """Enhanced CBAM (Convolutional Block Attention Module).
    
    Applies both channel and spatial attention to input features.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction ratio for attention. Defaults to 16.
    """
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        
        # Channel attention with bottleneck
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1),  # Bottleneck, min 8 channels
            nn.Hardswish(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention with 5x5 kernel
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),  # Capture local spatial relationships
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass applying channel and spatial attention sequentially."""
        # Apply channel attention
        ca = self.channel_att(x)
        x = x * ca
        
        # Apply spatial attention
        max_pool = torch.amax(x, dim=1, keepdim=True)    # Max pooling across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # Average pooling across channels
        return x * self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))


if __name__ == "__main__":
    # Create model instance
    model = DocRotationModel(num_classes=8)
    print(f"Model: {model.__class__.__name__}\n")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
    
    # Test with different input sizes to verify size-agnostic behavior
    for size in [384, 512]:
        dummy_input = torch.randn(4, 3, size, size)
        logits, confidence = model(dummy_input)
        print(f"Input {size}x{size} -> Rotation logits {logits.shape}, Confidence {confidence.shape}")
