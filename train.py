"""Training script for document rotation classification model.

This script handles the training pipeline setup and execution, including:
- Data loading and preprocessing
- Model initialization
- Training configuration
- Callbacks and logging
- Model checkpointing
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

import hydra
import torch
import wandb
from datasets.dataloader import create_dataloaders
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lightning_model import DocRotationLightning
from omegaconf import DictConfig, OmegaConf


def setup_wandb_logger(cfg: DictConfig) -> WandbLogger:
    """Initialize and configure WandB logger.
    
    Args:
        cfg: Configuration object containing WandB settings
        
    Returns:
        Configured WandB logger instance
    """
    # Load environment variables from .env
    load_dotenv()
    
    # Login to wandb using API key from .env
    if "WANDB_API_KEY" not in os.environ:
        raise ValueError(
            "WANDB_API_KEY not found in environment variables. "
            "Please add it to your .env file."
        )
    
    wandb.login(key=os.environ["WANDB_API_KEY"])
    
    return WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        save_dir=cfg.paths.output_dir,
        log_model=True,
    )


def create_callbacks(cfg: DictConfig) -> List[Callback]:
    """Create training callbacks.
    
    Args:
        cfg: Configuration object containing callback settings
        
    Returns:
        List of PyTorch Lightning callbacks
    """
    return [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename=cfg.checkpoint.filename,
            monitor="val/accuracy",
            mode="max",
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            every_n_epochs=cfg.checkpoint.every_n_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/accuracy",
            patience=cfg.training.early_stop_patience,
            mode="max",
            min_delta=0.001,
        ),
        RichProgressBar(),
    ]


def setup_trainer(cfg: DictConfig, callbacks: List[Callback], logger: WandbLogger) -> Trainer:
    """Create and configure the PyTorch Lightning trainer.
    
    Args:
        cfg: Configuration object containing trainer settings
        callbacks: List of callbacks to use during training
        logger: Logger instance for tracking metrics
        
    Returns:
        Configured PyTorch Lightning trainer
    """
    return Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )


def save_state_dict(
    model: DocRotationLightning,
    cfg: DictConfig,
    export_dir: Path,
) -> None:
    """Save model state dictionary.
    
    Args:
        model: Trained model instance
        cfg: Configuration object
        export_dir: Directory to save the state dict
    """
    state_dict_path = export_dir / "model_state_dict.pt"
    
    state_dict = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'model_class': model.__class__.__name__
    }
    
    if cfg.export.state_dict.save_optimizer and hasattr(model, 'optimizer'):
        state_dict['optimizer_state_dict'] = model.optimizer.state_dict()
    
    torch.save(state_dict, state_dict_path)
    print(f"State dict saved to: {state_dict_path}")
    wandb.save(str(state_dict_path))


def export_to_onnx(
    model: DocRotationLightning,
    cfg: DictConfig,
    export_dir: Path,
) -> None:
    """Export model to ONNX format.
    
    Args:
        model: Trained model instance
        cfg: Configuration object
        export_dir: Directory to save the ONNX model
    """
    onnx_path = export_dir / "model.onnx"
    dummy_input = torch.randn(cfg.export.onnx.input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=cfg.export.onnx.opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['rotation_logits', 'confidence'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'rotation_logits': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    print(f"ONNX model saved to: {onnx_path}")
    wandb.save(str(onnx_path))


def export_model(model: DocRotationLightning, cfg: DictConfig, best_model_path: str) -> None:
    """Export the model to various formats.
    
    Args:
        model: Trained model instance
        cfg: Configuration object
        best_model_path: Path to the best checkpoint
    """
    export_dir = Path(cfg.paths.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.load_from_checkpoint(best_model_path)
    model.eval()
    
    if cfg.export.state_dict.enabled:
        print("\nSaving state dict...")
        save_state_dict(model, cfg, export_dir)
    
    if cfg.export.onnx.enabled:
        print("\nExporting to ONNX...")
        export_to_onnx(model, cfg, export_dir)


def sanitize_config(cfg: DictConfig) -> Dict[str, Any]:
    """Convert Hydra config to a WandB-friendly dictionary.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary with WandB-friendly values
    """
    # Convert to dictionary and resolve interpolations
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Ensure all values are serializable
    def sanitize_value(val):
        if isinstance(val, (bool, int, float, str)):
            return val
        elif isinstance(val, (list, tuple)):
            return [sanitize_value(v) for v in val]
        elif isinstance(val, dict):
            return {k: sanitize_value(v) for k, v in val.items()}
        elif val is None:
            return None
        else:
            return str(val)
    
    return {k: sanitize_value(v) for k, v in config_dict.items()}


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    seed_everything(cfg.seed)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    
    # Setup training components
    wandb_logger = setup_wandb_logger(cfg)
    callbacks = create_callbacks(cfg)
    trainer = setup_trainer(cfg, callbacks, wandb_logger)
    
    # Initialize model
    model = DocRotationLightning(
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        num_classes=cfg.model.num_classes,
        scheduler_name=cfg.model.scheduler.name,
        scheduler_T_0=cfg.model.scheduler.T_0,
        scheduler_T_mult=cfg.model.scheduler.T_mult,
        scheduler_eta_min=cfg.model.scheduler.eta_min,
        scheduler_interval=cfg.model.scheduler.interval,
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_datasets=cfg.data.train_datasets,
        val_datasets=cfg.data.val_datasets,
        test_datasets=cfg.data.test_datasets if cfg.data.test_datasets != [] else None,
        image_size=cfg.data.image_size,
    )
    
    # Log dataset sizes and configuration
    wandb_logger.log_hyperparams({
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset) if test_loader is not None else 0,
    })
    
    # Log full configuration
    wandb_logger.experiment.config.update(sanitize_config(cfg))
    wandb.watch(model, log="all")
    
    # Train and evaluate
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    if test_loader is not None:
        trainer.test(dataloaders=test_loader)
    
    # Handle model export
    if trainer.checkpoint_callback.best_model_path:
        print(f"\nBest model path: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation accuracy: {trainer.checkpoint_callback.best_model_score:.4f}")
        export_model(model, cfg, trainer.checkpoint_callback.best_model_path)
    
    wandb.finish()


if __name__ == "__main__":
    train() 