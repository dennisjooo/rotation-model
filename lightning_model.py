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
from torchmetrics import Accuracy, MeanMetric

class DocRotationLightning(pl.LightningModule):
    """Lightning module for document rotation classification.
    
    Implements training, validation, and test loops for the DocRotationModel
    using PyTorch Lightning, including metrics tracking and optimizer configuration.
    Handles 8 rotation classes (0°, 45°, 90°, ..., 315°).
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
        
        # Set metrics
        self._set_metrics(num_classes)

        # Save hyperparameters for logging
        self.save_hyperparameters()
        
    def _set_metrics(self, num_classes: int):
        """Set up metrics for tracking model performance.
        
        Initializes metrics for tracking overall accuracy, per-class accuracy,
        angular error, and prediction confidence during training and validation.
        
        Args:
            num_classes: Number of rotation classes (e.g. 8 for 45-degree intervals)
        """
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Add per-class accuracy tracking - using multiclass instead of binary
        self.val_accuracy_per_class = torch.nn.ModuleList([
            Accuracy(task="multiclass", num_classes=num_classes, average='none') for _ in range(num_classes)
        ])
        
        # Add angular error metric
        self.val_angular_error = MeanMetric()
        
        # Track confidence metrics
        self.val_confidence_correct = MeanMetric()
        self.val_confidence_incorrect = MeanMetric()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (rotation_logits, confidence_scores)
        """
        return self.model(x)
    
    def _compute_losses(
        self,
        logits: torch.Tensor,
        confidence: torch.Tensor,
        pred: torch.Tensor,
        y: torch.Tensor,
        stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute classification, angular, and confidence losses.
        
        Args:
            logits: Model logits output
            confidence: Model confidence output
            pred: Predicted classes
            y: True labels
            stage: Current stage (train/val/test)
            
        Returns:
            Tuple containing:
                - total_loss: Combined weighted loss
                - cls_loss: Classification loss
                - angular_loss: Angular distance loss
                - confidence_loss: Confidence prediction loss
                - angular_diff: Angular difference between predictions and targets
        """
        # 1. Classification loss with label smoothing
        cls_loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        
        # 2. Angular loss - ensure float type
        angles_pred = pred.float() * 45
        angles_true = y.float() * 45
        angular_diff = torch.abs(angles_pred - angles_true)
        angular_diff = torch.min(angular_diff, 360 - angular_diff)
        angular_loss = angular_diff.mean()
        
        # 3. Enhanced confidence loss
        correct_mask = (pred == y)
        
        # Weight confidence loss based on angular difference
        angular_weight = angular_diff / 180.0  # Normalize to [0, 1]
        confidence_target = (1.0 - angular_weight) * correct_mask.float()
        
        confidence_loss = F.binary_cross_entropy_with_logits(
            confidence.squeeze(), 
            confidence_target,
            reduction='none'
        )
        confidence_loss = confidence_loss.mean()
        
        # Combine losses with adaptive weighting
        confidence_weight = 0.1 if stage == "train" else 0.05
        angular_weight = 0.1
        total_loss = cls_loss + confidence_weight * confidence_loss + angular_weight * angular_loss
        
        return total_loss, cls_loss, angular_loss, confidence_loss, angular_diff

    def _log_metrics(
        self,
        stage: str,
        batch_size: int,
        accuracy: torch.Tensor,
        cls_loss: torch.Tensor,
        confidence_loss: torch.Tensor,
        angular_loss: torch.Tensor,
        total_loss: torch.Tensor,
        angular_diff: torch.Tensor,
        confidence_pred: torch.Tensor,
        correct_mask: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor
    ) -> None:
        """Log all metrics for the current stage.
        
        Args:
            stage: Current stage (train/val/test)
            batch_size: Size of the current batch
            accuracy: Current accuracy metric
            cls_loss: Classification loss value
            confidence_loss: Confidence loss value
            angular_loss: Angular loss value
            total_loss: Combined loss value
            angular_diff: Angular difference between predictions and targets
            confidence_pred: Predicted confidence scores
            correct_mask: Boolean mask of correct predictions
            y: True labels
            pred: Predicted classes
        """
        # Log losses
        self.log(f"{stage}/cls_loss", cls_loss, batch_size=batch_size)
        self.log(f"{stage}/conf_loss", confidence_loss, batch_size=batch_size)
        self.log(f"{stage}/angular_loss", angular_loss, batch_size=batch_size)
        self.log(f"{stage}/total_loss", total_loss, batch_size=batch_size)
        self.log(f"{stage}/accuracy", accuracy, batch_size=batch_size)
        self.log(f"{stage}/angular_error", angular_diff.mean(), batch_size=batch_size)
        
        # Log validation-specific metrics
        if stage == "val":
            if correct_mask.any():
                self.val_confidence_correct(confidence_pred[correct_mask].mean())
            if (~correct_mask).any():
                self.val_confidence_incorrect(confidence_pred[~correct_mask].mean())
            
            # Per-class accuracy - now handling multiclass format
            for i in range(8):
                mask = y == i
                if mask.any():
                    class_acc = self.val_accuracy_per_class[i](pred[mask], y[mask])
                    # Take the accuracy for the current class from the per-class metrics
                    if isinstance(class_acc, torch.Tensor):
                        class_acc = class_acc[i]
                    self.log(f"val/accuracy_{i*45}deg", 
                            class_acc,
                            on_epoch=True)

    def _shared_step(
        self, 
        batch: Dict[str, Any],
        stage: Literal["train", "val", "test"]
    ) -> Dict[str, Any]:
        """Shared step logic for training, validation and testing.
        
        Args:
            batch: Data batch from dataloader
            stage: Current stage ("train", "val", or "test")
            
        Returns:
            Dictionary containing the loss and predictions
        """
        # Extract batch data
        x = batch["image"]
        y = batch["rotation"]
        rotation_angles = batch["rotation_angle"]
        batch_size = x.size(0)
        
        # Forward pass
        logits, confidence = self(x)
        pred = torch.argmax(logits, dim=1)
        
        # Get accuracy metric for current stage
        accuracy = getattr(self, f"{stage}_accuracy")
        accuracy(logits, y)
        
        # Compute all losses
        total_loss, cls_loss, angular_loss, confidence_loss, angular_diff = self._compute_losses(
            logits, confidence, pred, y, stage
        )
        
        # Get confidence predictions for logging
        confidence_pred = torch.sigmoid(confidence.squeeze())
        correct_mask = (pred == y)
        
        # Log all metrics
        self._log_metrics(
            stage=stage,
            batch_size=batch_size,
            accuracy=accuracy,
            cls_loss=cls_loss,
            confidence_loss=confidence_loss,
            angular_loss=angular_loss,
            total_loss=total_loss,
            angular_diff=angular_diff,
            confidence_pred=confidence_pred,
            correct_mask=correct_mask,
            y=y,
            pred=pred
        )
        
        return {
            "loss": total_loss,
            "logits": logits,
            "confidence": confidence,
            "pred": pred,
            "y": y,
            "x": x,
            "rotation_angles": rotation_angles,
            "angular_diff": angular_diff,
            "confidence_pred": confidence_pred,
            "correct_mask": correct_mask
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
        confidence = outputs["confidence_pred"]
        angular_diff = outputs["angular_diff"]
        
        # Take up to 4 samples
        num_samples = min(4, len(images))
        images = images[:num_samples]
        pred_angles = preds[:num_samples] * 45
        angles = angles[:num_samples]
        confidence = confidence[:num_samples]
        angular_diff = angular_diff[:num_samples]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 5))
        if num_samples == 1:
            axes = [axes]
        
        # Plot each image with enhanced information
        for i, (img, pred_angle, true_angle, conf, ang_err) in enumerate(
            zip(images, pred_angles, angles, confidence, angular_diff)):
            # Convert and normalize image
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Plot image
            axes[i].imshow(img_np)
            axes[i].axis('off')
            
            # Enhanced caption with angular error
            caption = (f'Pred: {pred_angle.item():.0f}°\n'
                      f'True: {true_angle.item():.0f}°\n'
                      f'Conf: {conf.item():.2f}\n'
                      f'Err: {ang_err.item():.1f}°')
            
            # Color-code title based on error
            color = 'green' if ang_err < 45 else 'red'
            axes[i].set_title(caption, fontsize=8, pad=10, color=color)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Convert to image and log
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(h, w, 4)[:, :, :3]
        plt.close()
        
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

    def on_validation_epoch_end(self):
        """Log additional validation metrics at epoch end."""
        # Log confidence metrics
        self.log("val/confidence_correct", self.val_confidence_correct.compute())
        self.log("val/confidence_incorrect", self.val_confidence_incorrect.compute())
        
        # Reset metrics
        self.val_confidence_correct.reset()
        self.val_confidence_incorrect.reset()
        self.val_angular_error.reset()


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