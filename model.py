"""Document Rotation Model for classifying document orientations.

This module implements a deep learning model for document rotation classification
using MobileNetV3-small as the backbone, enhanced with CoordConv and CBAM attention
mechanisms for better spatial awareness and feature extraction.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch import Tensor
from typing import Tuple

class DocRotationModel(nn.Module):
    """Document rotation classification model with attention mechanisms.
    
    This model combines MobileNetV3-small backbone with CoordConv and CBAM attention
    to classify document rotations into 8 classes (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    and predict confidence scores.
    
    Attributes:
        coord_conv: CoordConv layer for spatial awareness
        backbone: MobileNetV3-small feature extractor
        attn_low: CBAM attention for low-level features
        attn_high: CBAM attention for high-level features
        global_pool: Global average pooling
        local_pool: Local spatial pooling
        fusion: Feature fusion layer
        classifier: Rotation classification head
        confidence: Confidence score prediction head
    """
    
    def __init__(self, num_classes: int = 8) -> None:
        """Initialize the DocRotationModel.
        
        Args:
            num_classes: Number of rotation classes (default: 8)
        """
        super().__init__()
        
        # CoordConv for spatial awareness of document rotation
        self.coord_conv = CoordConv(3, 3, kernel_size=3, stride=2, padding=1)
        
        # MobileNetV3-small features (original structure preserved)
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        
        # Low-level attention (text strokes, small patterns)
        self.attn_low = CBAM(3, reduction=2)  # Changed reduction to avoid dimension issues
        
        # High-level attention (document layout, large structures)
        self.attn_high = CBAM(576, reduction=16)
        
        # Global context (entire document)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Local context (for small rotated objects)
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))  # Preserve spatial info
        
        # Feature fusion with reduced dimensionality
        self.fusion = nn.Sequential(
            nn.Linear(576 + 3*4*4, 512),  # 576 (global) + 48 (local 3ch x4x4)
            nn.BatchNorm1d(512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Rotation classification
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        
        # Prediction confidence (removed sigmoid since we'll use BCE with logits)
        self.confidence = nn.Sequential(
            nn.Linear(512, 1)
        )

        # Initialize weights
        self.init_weights()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Tuple containing:
                - Rotation logits of shape (batch_size, num_classes)
                - Confidence scores of shape (batch_size, 1)
        """
        
        x = self.coord_conv(x)  # [B, 16, H/2, W/2]
        x = self.attn_low(x)    # Focus on text strokes
        
        features = self.backbone(x)  # [B, 576, H/32, W/32]
        features = self.attn_high(features)  # Focus on document layout
        
        # Global context (batch_size, 576)
        global_feat = self.global_pool(features).flatten(1)
        
        # Local context from early features (batch_size, 16*16=256)
        local_feat = self.local_pool(x).flatten(1)
        
        # Concatenate and fuse features
        fused = torch.cat([global_feat, local_feat], dim=1)
        fused = self.fusion(fused)
        
        return self.classifier(fused), self.confidence(fused)
    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CoordConv(nn.Module):
    """CoordConv layer for enhanced spatial awareness.
    
    Adds coordinate channels to input features before convolution to help
    the network better understand spatial relationships.
    
    Attributes:
        conv: Convolutional layer that processes concatenated features
    """
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        """Initialize the CoordConv layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            **kwargs: Additional arguments for the conv layer
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of CoordConv.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
            
        Returns:
            Tensor of shape (batch_size, out_channels, H', W')
        """
        # Add coordinate channels
        batch_size, _, h, w = x.shape
        xx = torch.linspace(-1, 1, w, device=x.device).repeat(h, 1)
        yy = torch.linspace(-1, 1, h, device=x.device).repeat(w, 1).t()
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x = torch.cat([x, coords], dim=1)
        return self.conv(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Applies both channel and spatial attention to input features.
    Channel attention focuses on 'what' features to attend to,
    while spatial attention focuses on 'where' to attend.
    
    Attributes:
        channel_att: Channel attention module
        spatial_att: Spatial attention module
    """
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize the CBAM module.
        
        Args:
            channels: Number of input channels
            reduction: Channel reduction factor for attention (default: 16)
        """
        super().__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of CBAM.
        
        Args:
            x: Input tensor of shape (batch_size, channels, H, W)
            
        Returns:
            Attended tensor of shape (batch_size, channels, H, W)
        """
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca
        
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa


if __name__ == "__main__":
    model = DocRotationModel(num_classes=8)
    print(f"Model: {model.__class__.__name__}\n")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
    
    # Test input: batch of 4 384x384 RGB images
    dummy_input = torch.randn(4, 3, 384, 384)
    logits, confidence = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Rotation logits shape: {logits.shape}")  # Should be [4, 8]
    print(f"Confidence scores shape: {confidence.shape}\n")  # Should be [4, 1]

    # Test input two: batch of 4 512x512 RGB images
    dummy_input = torch.randn(4, 3, 512, 512)
    logits, confidence = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Rotation logits shape: {logits.shape}")  # Should be [4, 8]
    print(f"Confidence scores shape: {confidence.shape}")  # Should be [4, 1]
