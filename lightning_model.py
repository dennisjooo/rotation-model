"""Lightning module wrapper for DocRotationModel.

This module implements the training, validation, and testing logic for the
document rotation classification model using PyTorch Lightning.
"""

from typing import Any, Dict, Literal, Tuple

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from model import DocRotationModel
from torchmetrics import Accuracy

class DocRotationLightning(pl.LightningModule):
    """Lightning module for document rotation classification.
    
    Implements training, validation, and test loops for the DocRotationModel
    using PyTorch Lightning, including metrics tracking and optimizer configuration.
    Handles 8 rotation classes (0°, 45°, 90°, ..., 315°).
    
    Attributes:
        model: The underlying DocRotationModel
        train_accuracy: Accuracy metric for training
        val_accuracy: Accuracy metric for validation
        test_accuracy: Accuracy metric for testing
        learning_rate: Learning rate for optimization
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int = 8,
        scheduler_name: str = "cosine_warm_restarts",
        scheduler_T_0: int = 10,
        scheduler_T_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
        scheduler_interval: str = "step"
    ) -> None:
        """Initialize the lightning module.
        
        Args:
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for regularization
            num_classes: Number of rotation classes (default: 8)
            scheduler_name: Name of the learning rate scheduler
            scheduler_T_0: Number of epochs until first restart
            scheduler_T_mult: Factor to increase T_0 after a restart
            scheduler_eta_min: Minimum learning rate
            scheduler_interval: Scheduler step interval ('step' or 'epoch')
        """
        super().__init__()
        
        # Core model
        self.model = DocRotationModel(num_classes=num_classes)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.scheduler_T_0 = scheduler_T_0
        self.scheduler_T_mult = scheduler_T_mult
        self.scheduler_eta_min = scheduler_eta_min
        self.scheduler_interval = scheduler_interval
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (rotation_logits, confidence_scores)
        """
        return self.model(x)
    
    def _shared_step(
        self, 
        batch: Dict[str, Any],
        stage: Literal["train", "val", "test"]
    ) -> Dict[str, Any]:
        """Shared step logic for training, validation and testing.
        
        Args:
            batch: Data batch from dataloader containing:
                - image: Tensor of shape (batch_size, channels, height, width)
                - rotation: Tensor of rotation classes
                - rotation_angle: Tensor of actual rotation angles
                - path: List of image paths
            stage: Current stage ("train", "val", or "test")
            
        Returns:
            Dictionary containing the loss and predictions
        """
        # Extract image and rotation label from batch
        x = batch["image"]
        y = batch["rotation"]
        rotation_angles = batch["rotation_angle"]
        
        logits, confidence = self(x)
        
        # Get accuracy metric for current stage
        accuracy = getattr(self, f"{stage}_accuracy")
        accuracy(logits, y)
        
        # Classification loss with label smoothing
        cls_loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        
        # Confidence loss with focal-like weighting
        pred = torch.argmax(logits, dim=1)
        confidence_target = (pred == y).float()
        confidence_pred = torch.sigmoid(confidence.squeeze())
        
        # Weight confidence loss based on prediction certainty
        confidence_weights = torch.abs(confidence_target - confidence_pred).detach()
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence.squeeze(), 
            confidence_target,
            reduction='none'
        )
        confidence_loss = (confidence_weights * confidence_loss).mean()
        
        # Adaptive loss weighting
        confidence_weight = 0.1 if stage == "train" else 0.05
        loss = cls_loss + confidence_weight * confidence_loss
        
        # Log detailed metrics
        batch_size = x.size(0)
        self.log(f"{stage}/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/conf_loss", confidence_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return {
            "loss": loss, 
            "logits": logits, 
            "confidence": confidence,
            "pred": pred,
            "y": y,
            "x": x,
            "rotation_angles": rotation_angles
        }
    
    def _log_predictions(self, outputs: Dict[str, Any], stage: str, batch_idx: int) -> None:
        """Log sample predictions with images for visualization.
        
        Args:
            outputs: Dictionary containing model outputs
            stage: Current stage (val or test)
            batch_idx: Index of current batch
        """
        if batch_idx % 50 != 0:  # Log every 50 batches
            return
            
        # Get predictions and images
        images = outputs["x"]
        preds = outputs["pred"]
        angles = outputs["rotation_angles"]
        confidence = torch.sigmoid(outputs["confidence"].squeeze())  # Convert to probability
        
        # Convert rotation class to angle
        pred_angles = preds * 45
        
        # Take up to 4 samples
        num_samples = min(4, len(images))
        images = images[:num_samples]
        pred_angles = pred_angles[:num_samples]
        angles = angles[:num_samples]
        confidence = confidence[:num_samples]
        
        # Create a figure with subplots with more height for titles
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 5))
        if num_samples == 1:
            axes = [axes]
        
        # Plot each image with its predictions
        for i, (img, pred_angle, true_angle, conf) in enumerate(zip(images, pred_angles, angles, confidence)):
            # Convert image tensor to numpy and transpose to (H, W, C)
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            
            # Normalize image for display
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Plot image
            axes[i].imshow(img_np)
            axes[i].axis('off')
            
            # Add caption with increased padding and smaller font
            caption = f'Pred: {pred_angle.item():.0f}°\nTrue: {true_angle.item():.0f}°\nConf: {conf.item():.2f}'
            axes[i].set_title(caption, fontsize=8, pad=10)
        
        # Adjust layout with more space at the top
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Convert figure to numpy array
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        # Reshape it to a proper image array
        img_data = buf.reshape(h, w, 4)
        # Convert RGBA to RGB
        img_data = img_data[:, :, :3]
        plt.close()
        
        # Log the image
        self.logger.log_image(f"{stage}/predictions", images=[img_data])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Training step logic.
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing the loss and logs
        """
        outputs = self._shared_step(batch, "train")
        self._log_predictions(outputs, "train", batch_idx)
        return outputs
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step logic.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
        """
        outputs = self._shared_step(batch, "val")
        self._log_predictions(outputs, "val", batch_idx)
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step logic.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
        """
        outputs = self._shared_step(batch, "test")
        self._log_predictions(outputs, "test", batch_idx)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and learning rate scheduler for training.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler_name == "cosine_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_T_0,
                T_mult=self.scheduler_T_mult,
                eta_min=self.scheduler_eta_min
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                "frequency": 1,
                "monitor": "val/accuracy",
                "name": self.scheduler_name
            }
        }


if __name__ == "__main__":
    # Quick test of the lightning module
    model = DocRotationLightning()
    print(f"Lightning Model: {model.__class__.__name__}\n")
    
    # Test with dummy batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 384, 384)
    dummy_labels = torch.randint(0, 8, (batch_size,))  # Now 8 classes
    
    logits, confidence = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Labels shape: {dummy_labels.shape}")
    print(f"Logits shape: {logits.shape}")  # Should be [4, 8]
    print(f"Confidence shape: {confidence.shape}") 