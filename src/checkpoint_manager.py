"""Robust checkpoint and artifact management for wandb integration.

This module provides comprehensive checkpoint saving with wandb artifacts,
including model weights, code, configs, and metadata.
"""

import os
import shutil
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CheckpointManager:
    """Manages model checkpoints with wandb artifact integration.

    Features:
    - Save best model based on validation loss
    - Save periodic checkpoints
    - Save code and config as artifacts
    - Automatic cleanup of old checkpoints
    - Resume from checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        experiment_name: str,
        save_code: bool = True,
        save_config: bool = True,
        max_checkpoints: int = 5,
        use_wandb: bool = True
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name for wandb artifacts
            save_code: If True, save training script as artifact
            save_config: If True, save config as artifact
            max_checkpoints: Maximum periodic checkpoints to keep
            use_wandb: If True, save to wandb artifacts
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.save_code = save_code
        self.save_config = save_config
        self.max_checkpoints = max_checkpoints
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.best_loss = float('inf')
        self.best_checkpoint_path = None
        self.periodic_checkpoints = []

        # Save code and config on initialization
        if self.use_wandb and wandb.run is not None:
            if save_code:
                self._save_code_artifact()
            if save_config:
                self._save_config_artifact()

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        val_loss: float,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        is_periodic: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save checkpoint with comprehensive metadata.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            val_loss: Validation loss
            metrics: Additional metrics to save
            is_best: If True, save as best model
            is_periodic: If True, save as periodic checkpoint
            metadata: Additional metadata to include

        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'timestamp': datetime.now().isoformat(),
        }

        # Add metrics
        if metrics:
            checkpoint['metrics'] = metrics

        # Add custom metadata
        if metadata:
            checkpoint['metadata'] = metadata

        # Generate filename
        if is_best:
            filename = f"best_epoch{epoch:03d}_loss{val_loss:.6f}.pth"
            save_path = self.checkpoint_dir / filename

            # Update best tracking
            self.best_loss = val_loss
            self.best_checkpoint_path = save_path
        elif is_periodic:
            filename = f"periodic_epoch{epoch:03d}_loss{val_loss:.6f}.pth"
            save_path = self.checkpoint_dir / filename

            # Track periodic checkpoints
            self.periodic_checkpoints.append(save_path)
            self._cleanup_periodic_checkpoints()
        else:
            filename = f"latest_epoch{epoch:03d}_loss{val_loss:.6f}.pth"
            save_path = self.checkpoint_dir / filename

        # Save checkpoint locally
        torch.save(checkpoint, save_path)
        print(f"💾 Saved checkpoint: {save_path.name}")

        # Save to wandb if enabled
        if self.use_wandb and wandb.run is not None:
            self._save_wandb_checkpoint(save_path, epoch, val_loss, is_best, metrics)

        return save_path

    def _save_wandb_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        val_loss: float,
        is_best: bool,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Save checkpoint to wandb artifacts."""
        # Create artifact
        artifact_name = f"model-{self.experiment_name}"
        artifact = wandb.Artifact(
            artifact_name,
            type="model",
            metadata={
                'epoch': epoch,
                'val_loss': val_loss,
                'is_best': is_best,
                **(metrics or {})
            }
        )

        # Add checkpoint file
        artifact.add_file(str(checkpoint_path))

        # Set aliases
        aliases = ["latest"]
        if is_best:
            aliases.append("best")

        # Log artifact
        wandb.log_artifact(artifact, aliases=aliases)
        print(f"☁️  Uploaded to wandb: {artifact_name} ({', '.join(aliases)})")

    def _save_code_artifact(self):
        """Save training script as wandb artifact."""
        code_artifact = wandb.Artifact(
            f"code-{self.experiment_name}",
            type="code",
            metadata={'timestamp': datetime.now().isoformat()}
        )

        # Add training script
        script_path = Path(__file__).parent / "finetune_with_wandb.py"
        if script_path.exists():
            code_artifact.add_file(str(script_path))

        # Add model files
        model_files = [
            "vit_model.py",
            "vit_encoder.py",
            "ours_mamba.py",
            "losses.py",
            "utils.py"
        ]

        for model_file in model_files:
            file_path = Path(__file__).parent / model_file
            if file_path.exists():
                code_artifact.add_file(str(file_path))

        wandb.log_artifact(code_artifact)
        print("📝 Saved code artifact to wandb")

    def _save_config_artifact(self):
        """Save config as wandb artifact."""
        # Get config from wandb.config
        if wandb.run and wandb.run.config:
            config_artifact = wandb.Artifact(
                f"config-{self.experiment_name}",
                type="config",
                metadata={'timestamp': datetime.now().isoformat()}
            )

            # Save config as JSON
            config_path = self.checkpoint_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(dict(wandb.run.config), f, indent=2)

            config_artifact.add_file(str(config_path))
            wandb.log_artifact(config_artifact)
            print("⚙️  Saved config artifact to wandb")

    def _cleanup_periodic_checkpoints(self):
        """Remove old periodic checkpoints, keeping only max_checkpoints."""
        if len(self.periodic_checkpoints) > self.max_checkpoints:
            # Sort by modification time (oldest first)
            self.periodic_checkpoints.sort(key=lambda p: p.stat().st_mtime)

            # Remove oldest checkpoints
            while len(self.periodic_checkpoints) > self.max_checkpoints:
                old_checkpoint = self.periodic_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"🗑️  Removed old checkpoint: {old_checkpoint.name}")

    def save_best_if_improved(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        val_loss: float,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save checkpoint if validation loss improved.

        Returns:
            True if loss improved and checkpoint was saved, False otherwise
        """
        if val_loss < self.best_loss:
            self.save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                metrics=metrics,
                is_best=True,
                metadata=metadata
            )
            print(f"✨ New best model! Loss: {val_loss:.6f} (prev: {self.best_loss:.6f})")
            return True
        return False

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load checkpoint from path.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)

        Returns:
            Dictionary with epoch, val_loss, and any additional metadata
        """
        print(f"📂 Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Extract metadata
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        metrics = checkpoint.get('metrics', {})
        metadata = checkpoint.get('metadata', {})

        print(f"✓ Loaded checkpoint from epoch {epoch} (loss: {val_loss:.6f})")

        # Update best loss if this was a best checkpoint
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_checkpoint_path = checkpoint_path

        return {
            'epoch': epoch,
            'val_loss': val_loss,
            'metrics': metrics,
            'metadata': metadata
        }

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Return path to best checkpoint."""
        return self.best_checkpoint_path

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Return path to most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            return None

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
