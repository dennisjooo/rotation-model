"""Setup functions for training environment and components."""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


def set_torch_matmul_precision():
    """Set the precision of the torch matmul to high if possible."""
    if torch.backends.mps.is_available():
        torch.set_float32_matmul_precision('high')
    else:
        print("MPS is not available, using default precision")


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