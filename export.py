"""Functions for exporting trained models to various formats."""

import argparse
from pathlib import Path

import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from lightning_model import DocRotationLightning


def save_artifact(artifact_path: Path) -> None:
    """Save artifact to wandb.
    """
    try:
        wandb.save(str(artifact_path))
    except Exception as e:
        print(f"Error saving artifact to wandb: {e}")

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
    save_artifact(state_dict_path)



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
    try:
        # Convert ListConfig to tuple for torch.randn
        input_shape = tuple(int(x) for x in cfg.export.onnx.input_shape)
        dummy_input = torch.randn(input_shape)
        
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
        save_artifact(onnx_path)
        
    except (TypeError, ValueError) as e:
        print(f"\nError during ONNX export - input shape conversion failed: {e}")
        print(f"Input shape from config: {cfg.export.onnx.input_shape}")
        print("Please ensure input_shape contains valid integers")
        raise
        
    except RuntimeError as e:
        print(f"\nError during ONNX export - model conversion failed: {e}")
        print("This might be due to unsupported operations or incorrect model configuration")
        raise
        
    except Exception as e:
        print(f"\nUnexpected error during ONNX export: {e}")
        raise


def export_model(model: DocRotationLightning, cfg: DictConfig, best_model_path: str) -> None:
    """Export the model to various formats.
    
    Args:
        model: Trained model instance
        cfg: Configuration object
        best_model_path: Path to the best checkpoint
    """
    export_dir = Path(cfg.paths.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the best checkpoint properly
    model = DocRotationLightning.load_from_checkpoint(
        best_model_path,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        num_classes=cfg.model.num_classes,
        scheduler_name=cfg.model.scheduler.name,
        scheduler_T_0=cfg.model.scheduler.T_0,
        scheduler_T_mult=cfg.model.scheduler.T_mult,
        scheduler_eta_min=cfg.model.scheduler.eta_min,
        scheduler_interval=cfg.model.scheduler.interval,
    )
    model.eval()
    
    if cfg.export.state_dict.enabled:
        print("\nSaving state dict...")
        save_state_dict(model, cfg, export_dir)
    
    if cfg.export.onnx.enabled:
        print("\nExporting to ONNX...")
        export_to_onnx(model, cfg, export_dir)


def main():
    parser = argparse.ArgumentParser(description='Export trained model to various formats')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the training configuration YAML file')
    parser.add_argument('--export-dir', type=str,
                      help='Directory to save exported models. If not provided, will use checkpoint directory.')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # Set export directory
    if args.export_dir:
        export_dir = Path(args.export_dir)
    else:
        # Use the checkpoint's directory as export directory
        checkpoint_path = Path(args.checkpoint)
        export_dir = checkpoint_path.parent / "exported"
    
    # Update config with concrete export directory
    cfg.paths.export_dir = str(export_dir)

    # Initialize dummy model for export
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

    # Export model
    export_model(model, cfg, args.checkpoint)


if __name__ == '__main__':
    main()
        
        
