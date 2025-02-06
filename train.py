"""Training script for document rotation classification model.

This script handles the training pipeline setup and execution, including:
- Data loading and preprocessing
- Model initialization
- Training configuration
- Callbacks and logging
- Model checkpointing
"""

import os
import hydra
import wandb

from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from datasets.dataloader import create_dataloaders
from export import export_model
from lightning_model import DocRotationLightning

from setup import (
    set_torch_matmul_precision,
    setup_wandb_logger,
    create_callbacks,
    setup_trainer,
    sanitize_config,
)


def initialize_training(cfg: DictConfig) -> tuple:
    """Initialize all components needed for training.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (model, train_loader, val_loader, test_loader, trainer, wandb_logger)
    """
    set_torch_matmul_precision()
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
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor
    )
    
    return model, train_loader, val_loader, test_loader, trainer, wandb_logger


def log_training_info(wandb_logger: WandbLogger, cfg: DictConfig, train_loader, val_loader, test_loader, model):
    """Log training information to WandB.
    
    Args:
        wandb_logger: WandB logger instance
        cfg: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        model: Model instance
    """
    wandb_logger.log_hyperparams({
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset) if test_loader is not None else 0,
    })
    
    wandb_logger.experiment.config.update(sanitize_config(cfg))
    wandb.watch(model, log="all")


def train_and_evaluate(
    trainer,
    model,
    train_loader,
    val_loader,
    test_loader,
    cfg: DictConfig
) -> None:
    """Run training and evaluation.
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        cfg: Configuration object
    """
    # Train
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Test if test loader exists
    if test_loader is not None:
        trainer.test(dataloaders=test_loader)
    
    # Export if we have a best model
    if trainer.checkpoint_callback.best_model_path:
        print(f"\nBest model path: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation accuracy: {trainer.checkpoint_callback.best_model_score:.4f}")
        export_model(model, cfg, trainer.checkpoint_callback.best_model_path)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        # Initialize all training components
        model, train_loader, val_loader, test_loader, trainer, wandb_logger = initialize_training(cfg)
        
        # Log training information
        log_training_info(wandb_logger, cfg, train_loader, val_loader, test_loader, model)
        
        # Run training and evaluation
        train_and_evaluate(trainer, model, train_loader, val_loader, test_loader, cfg)
        
    finally:
        # Ensure wandb is properly closed
        wandb.finish()


if __name__ == "__main__":
    train() 