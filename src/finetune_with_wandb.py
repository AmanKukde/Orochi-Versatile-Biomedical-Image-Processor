"""Finetuning script with Weights & Biases integration.

This script provides a unified training/finetuning pipeline for both
Mamba and Vision Transformer architectures with comprehensive experiment tracking.

Features:
- Supports both MambaULight and ViTULight models
- Comprehensive wandb logging (metrics, images, gradients)
- Multi-task training (registration, fusion, SR, IR)
- Checkpoint management and resuming
- Mixed precision training support
- Gradient accumulation

Usage:
    # Finetune Vision Transformer
    python src/finetune_with_wandb.py --config configs/vit_finetune.yaml --model vit

    # Finetune Mamba
    python src/finetune_with_wandb.py --config configs/mamba_finetune.yaml --model mamba

    # Resume from checkpoint
    python src/finetune_with_wandb.py --config configs/vit_finetune.yaml --resume checkpoints/best.pth
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from tqdm import tqdm
import tifffile

# wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orochi.configs.model_configs import ViT3DConfig, MambaULightConfig
from src.vit_model import ViTULight

# Try to import Mamba model (optional if mamba_ssm not installed)
try:
    from src.ours_mamba import MambaULight
    MAMBA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import MambaULight (mamba_ssm not available)")
    print(f"   Error: {e}")
    print(f"   Mamba model will not be available. Use --model vit instead.")
    MAMBA_AVAILABLE = False
    MambaULight = None

import src.utils as utils


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (slower)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class DummyDataset(Dataset):
    """Dummy dataset for testing and demonstration.

    Replace this with your actual dataset class.

    Args:
        num_samples: Number of samples in dataset
        img_size: Image size [D, H, W]
        in_chans: Number of input channels
    """

    def __init__(self, num_samples=100, img_size=(64, 128, 128), in_chans=1):
        self.num_samples = num_samples
        self.img_size = img_size
        self.in_chans = in_chans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Return a random sample.

        Returns:
            Dictionary containing:
                - 'image': Random 3D image tensor
                - 'idx': Sample index
        """
        # Generate random 3D volume
        image = torch.rand(self.in_chans, *self.img_size)

        return {
            'image': image,
            'idx': idx,
        }


class BiomedicalDataset(Dataset):
    """Simplified biomedical dataset matching original preprocessing.

    Based on Pretrain_Dataset from temp/experiments/3D/datasets.py.
    Uses min-max normalization and conditional upsampling.

    Args:
        data_root: Root directory with dataset folders
        datasets: List of dataset names (e.g., ['hipsc_3d', 'hipsc_2d'])
        img_size: Target size (D, H, W)
        split: 'train' or 'val'
        val_split: Validation split ratio (default: 0.1)
    """

    def __init__(
        self,
        data_root="/group/jug/aman/orochi/data",
        datasets=['hipsc_3d'],
        img_size=(64, 128, 128),
        split='train',
        val_split=0.1
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size

        # Collect image files (.tiff, .tif, .npy)
        self.image_files = []
        for dataset_name in datasets:
            dataset_path = self.data_root / dataset_name
            if not dataset_path.exists():
                print(f"⚠️  Dataset not found: {dataset_path}")
                continue

            if dataset_name == 'hipsc_3d':
                # Protein subdirectories
                for protein_dir in dataset_path.iterdir():
                    if protein_dir.is_dir():
                        self.image_files.extend(protein_dir.glob('*.tif*'))
                        self.image_files.extend(protein_dir.glob('*.npy'))
            else:
                # data/ subdirectory
                data_dir = dataset_path / 'data'
                if data_dir.exists():
                    self.image_files.extend(data_dir.glob('*.tif*'))
                    self.image_files.extend(data_dir.glob('*.npy'))

        # Sort and split
        self.image_files = sorted(self.image_files)
        n_total = len(self.image_files)
        n_val = int(n_total * val_split)

        if split == 'train':
            self.image_files = self.image_files[:n_total - n_val]
        else:
            self.image_files = self.image_files[n_total - n_val:]

        print(f"📁 {split.capitalize()}: {len(self.image_files)} images from {datasets}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load and preprocess image (original approach)."""
        img_path = self.image_files[idx]

        # Load image
        if img_path.suffix == '.npy':
            raw_image = np.load(str(img_path))
        else:  # .tiff, .tif
            raw_image = tifffile.imread(str(img_path))

        # To tensor with channel dimension
        image_tensor = torch.from_numpy(raw_image).float().unsqueeze(0)

        # Min-max normalization (original approach)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min() + 1e-8
        )

        # Resize to target size (upsample or downsample as needed)
        image_tensor = self._resize(image_tensor)

        return {
            'image': image_tensor,
            'idx': idx,
        }

    def _resize(self, image_tensor):
        """Resize to target size (upsample or downsample as needed)."""
        d, h, w = image_tensor.shape[1:]  # (C, D, H, W)
        td, th, tw = self.img_size

        # Resize if dimensions don't match target
        if (d, h, w) != (td, th, tw):
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(td, th, tw),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)

        return image_tensor


def create_model(config, model_type='vit', freeze_decoders=False):
    """Create model based on configuration.

    Args:
        config: Model configuration object
        model_type: 'vit' or 'mamba'
        freeze_decoders: If True, freeze all decoder parameters (train encoder only)

    Returns:
        Model instance

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == 'vit':
        print(f"Creating Vision Transformer model with {sum(config.depths)} blocks")
        model = ViTULight(config)
    elif model_type.lower() == 'mamba':
        if not MAMBA_AVAILABLE:
            raise ValueError(
                "Mamba model requested but mamba_ssm is not available.\n"
                "Please install mamba_ssm or use --model vit instead."
            )
        print(f"Creating Mamba model with {sum(config.depths)} blocks")
        model = MambaULight(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vit' or 'mamba'")

    # Freeze decoders if requested
    if freeze_decoders:
        print("Freezing decoder parameters (training encoder only)")

        # Freeze all decoder parameters
        for name, param in model.named_parameters():
            if any(decoder in name for decoder in ['reg_decoder', 'fus_decoder', 'SR_decoder', 'IR_decoder', 'spatial_trans']):
                param.requires_grad = False

        # Count trainable vs frozen parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = total_params - trainable_params

        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

    return model


def create_optimizer(model, config):
    """Create optimizer based on configuration.

    Args:
        model: Model to optimize
        config: Configuration object

    Returns:
        Optimizer instance
    """
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        config: Configuration object

    Returns:
        Scheduler instance or None
    """
    if config.scheduler is None:
        return None

    if config.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.lr_decay_epochs,
            gamma=config.lr_decay_rate,
        )
    elif config.scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr,
        )
    elif config.scheduler.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_decay_rate,
            patience=10,
            min_lr=config.min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")

    return scheduler


def compute_total_loss(aux_loss: Dict) -> torch.Tensor:
    """Compute total loss from auxiliary losses.

    Args:
        aux_loss: Dictionary of task-specific losses

    Returns:
        Total weighted loss
    """
    total_loss = 0.0

    # MSE losses for all tasks
    for task, loss_val in aux_loss['mse'].items():
        total_loss += loss_val

    # NCC loss for registration
    if 'ncc' in aux_loss:
        total_loss += aux_loss['ncc']['reg']

    # Gradient regularization for registration
    if 'grad' in aux_loss:
        total_loss += 0.01 * aux_loss['grad']['reg']  # Weight for smoothness

    return total_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    config,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to use
        epoch: Current epoch number
        config: Configuration object
        scaler: GradScaler for mixed precision (optional)

    Returns:
        Dictionary of average losses
    """
    model.train()

    # Track metrics
    loss_meter = utils.AverageMeter()
    reg_loss_meter = utils.AverageMeter()
    fus_loss_meter = utils.AverageMeter()
    sr_loss_meter = utils.AverageMeter()
    ir_loss_meter = utils.AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        # Forward pass
        if scaler is not None:
            with autocast():
                logits, aux_loss = model(images)
                loss = compute_total_loss(aux_loss)
        else:
            logits, aux_loss = model(images)
            loss = compute_total_loss(aux_loss)

        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Update meters
        loss_meter.update(loss.item())
        reg_loss_meter.update(aux_loss['mse']['reg'].item())
        fus_loss_meter.update(aux_loss['mse']['fus'].item())
        sr_loss_meter.update(aux_loss['mse']['SR'].item())
        ir_loss_meter.update(aux_loss['mse']['IR'].item())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'reg': f"{reg_loss_meter.avg:.4f}",
            'fus': f"{fus_loss_meter.avg:.4f}",
            'sr': f"{sr_loss_meter.avg:.4f}",
            'ir': f"{ir_loss_meter.avg:.4f}",
        })

        # Log to wandb
        if WANDB_AVAILABLE and config.wandb_project is not None:
            if batch_idx % config.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/reg_loss': aux_loss['mse']['reg'].item(),
                    'train/fus_loss': aux_loss['mse']['fus'].item(),
                    'train/sr_loss': aux_loss['mse']['SR'].item(),
                    'train/ir_loss': aux_loss['mse']['IR'].item(),
                    'train/ncc_loss': aux_loss['ncc']['reg'].item(),
                    'train/grad_loss': aux_loss['grad']['reg'].item(),
                    'epoch': epoch,
                    'batch': batch_idx,
                })

    return {
        'loss': loss_meter.avg,
        'reg_loss': reg_loss_meter.avg,
        'fus_loss': fus_loss_meter.avg,
        'sr_loss': sr_loss_meter.avg,
        'ir_loss': ir_loss_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    config,
) -> Dict[str, float]:
    """Validate model.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device to use
        epoch: Current epoch number
        config: Configuration object

    Returns:
        Dictionary of average validation losses
    """
    model.eval()

    # Track metrics
    loss_meter = utils.AverageMeter()
    reg_loss_meter = utils.AverageMeter()
    fus_loss_meter = utils.AverageMeter()
    sr_loss_meter = utils.AverageMeter()
    ir_loss_meter = utils.AverageMeter()

    pbar = tqdm(dataloader, desc=f"Validation")

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)

        # Forward pass
        logits, aux_loss = model(images)
        loss = compute_total_loss(aux_loss)

        # Update meters
        loss_meter.update(loss.item())
        reg_loss_meter.update(aux_loss['mse']['reg'].item())
        fus_loss_meter.update(aux_loss['mse']['fus'].item())
        sr_loss_meter.update(aux_loss['mse']['SR'].item())
        ir_loss_meter.update(aux_loss['mse']['IR'].item())

        # Log sample images to wandb
        if WANDB_AVAILABLE and config.wandb_project is not None and batch_idx == 0:
            # Log first sample
            log_images = {
                'val/raw': wandb.Image(logits['raw'][0, 0, logits['raw'].shape[2]//2]),
                'val/reg_deformed': wandb.Image(logits['reg']['deformed'][0, 0, logits['reg']['deformed'].shape[2]//2]),
                'val/reg_registered': wandb.Image(logits['reg']['registered'][0, 0, logits['reg']['registered'].shape[2]//2]),
                'val/fused': wandb.Image(logits['fus']['fused'][0, 0, logits['fus']['fused'].shape[2]//2]),
                'val/sr': wandb.Image(logits['SR']['super_resolution'][0, 0, logits['SR']['super_resolution'].shape[2]//2]),
                'val/ir': wandb.Image(logits['IR']['restored'][0, 0, logits['IR']['restored'].shape[2]//2]),
            }
            wandb.log(log_images)

    # Log validation metrics
    if WANDB_AVAILABLE and config.wandb_project is not None:
        wandb.log({
            'val/loss': loss_meter.avg,
            'val/reg_loss': reg_loss_meter.avg,
            'val/fus_loss': fus_loss_meter.avg,
            'val/sr_loss': sr_loss_meter.avg,
            'val/ir_loss': ir_loss_meter.avg,
            'epoch': epoch,
        })

    return {
        'loss': loss_meter.avg,
        'reg_loss': reg_loss_meter.avg,
        'fus_loss': fus_loss_meter.avg,
        'sr_loss': sr_loss_meter.avg,
        'ir_loss': ir_loss_meter.avg,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_loss: float,
    config,
    filename: str,
):
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state (optional)
        epoch: Current epoch
        best_loss: Best validation loss so far
        config: Configuration object
        filename: Filename for checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_loss': best_loss,
        'config': config.to_dict(),
    }

    save_path = config.checkpoint_dir / filename
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[int, float]:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)

    Returns:
        Tuple of (epoch, best_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))

    print(f"Loaded checkpoint from epoch {epoch} with best loss {best_loss:.4f}")

    return epoch, best_loss


def load_pretrained_vit_encoder(
    model: nn.Module,
    pretrained_model_name: str = "google/vit-base-patch16-224",
    freeze_early_layers: bool = True,
    num_frozen_layers: int = 8,
) -> None:
    """Load pretrained ViT encoder weights from HuggingFace.

    Loads 2D ViT weights and adapts them for 3D medical imaging by inflating
    the weights. This provides better initialization than random weights.

    Args:
        model: ViTULight model to load weights into
        pretrained_model_name: HuggingFace model name (e.g., 'google/vit-base-patch16-224')
        freeze_early_layers: Whether to freeze early transformer layers
        num_frozen_layers: Number of early layers to freeze (if freeze_early_layers=True)
    """
    try:
        from transformers import ViTModel
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        print("Skipping pretrained weight loading.")
        return

    print(f"\n{'='*60}")
    print(f"Loading pretrained ViT weights from: {pretrained_model_name}")
    print(f"{'='*60}")

    # Load pretrained 2D ViT
    try:
        pretrained_vit = ViTModel.from_pretrained(pretrained_model_name)
        print(f"✓ Successfully downloaded pretrained model")
    except Exception as e:
        print(f"❌ Failed to load pretrained model: {e}")
        return

    # Get encoder from model
    if hasattr(model, 'encoder'):
        encoder = model.encoder
    else:
        print("❌ Model doesn't have 'encoder' attribute")
        return

    # Load compatible weights
    loaded_count = 0

    # Load patch embedding weights (inflate 2D to 3D)
    if hasattr(pretrained_vit.embeddings, 'patch_embeddings'):
        pretrained_patch_weight = pretrained_vit.embeddings.patch_embeddings.projection.weight

        # Inflate 2D conv weights to 3D by repeating along depth dimension
        if hasattr(encoder.patch_embed, 'proj'):
            patch_size_depth = encoder.patch_embed.proj.kernel_size[0]
            inflated_weight = pretrained_patch_weight.unsqueeze(2).repeat(1, 1, patch_size_depth, 1, 1)
            inflated_weight = inflated_weight / patch_size_depth  # Average

            # Copy if dimensions match
            if inflated_weight.shape == encoder.patch_embed.proj.weight.shape:
                encoder.patch_embed.proj.weight.data.copy_(inflated_weight)
                loaded_count += 1
                print(f"✓ Loaded patch embedding weights (inflated 2D→3D)")
            else:
                print(f"⚠️  Patch embedding size mismatch: {inflated_weight.shape} vs {encoder.patch_embed.proj.weight.shape}")

    # Load transformer layer weights
    if hasattr(pretrained_vit.encoder, 'layer'):
        num_layers = min(len(pretrained_vit.encoder.layer), len(encoder.layers))

        for i in range(num_layers):
            pretrained_layer = pretrained_vit.encoder.layer[i]
            our_layer = encoder.layers[i]

            # Load attention weights
            if hasattr(our_layer, 'blocks') and len(our_layer.blocks) > 0:
                for block_idx, block in enumerate(our_layer.blocks):
                    if hasattr(block, 'attn') and hasattr(pretrained_layer, 'attention'):
                        # Load Q, K, V weights
                        if hasattr(pretrained_layer.attention.attention, 'query'):
                            qkv_weight = torch.cat([
                                pretrained_layer.attention.attention.query.weight,
                                pretrained_layer.attention.attention.key.weight,
                                pretrained_layer.attention.attention.value.weight
                            ], dim=0)

                            if qkv_weight.shape == block.attn.qkv.weight.shape:
                                block.attn.qkv.weight.data.copy_(qkv_weight)
                                loaded_count += 1

                        # Load output projection
                        if hasattr(pretrained_layer.attention.output, 'dense'):
                            if pretrained_layer.attention.output.dense.weight.shape == block.attn.proj.weight.shape:
                                block.attn.proj.weight.data.copy_(pretrained_layer.attention.output.dense.weight)
                                loaded_count += 1

                    # Load MLP weights
                    if hasattr(block, 'mlp') and hasattr(pretrained_layer, 'intermediate'):
                        if pretrained_layer.intermediate.dense.weight.shape == block.mlp.fc1.weight.shape:
                            block.mlp.fc1.weight.data.copy_(pretrained_layer.intermediate.dense.weight)
                            loaded_count += 1

                        if hasattr(pretrained_layer, 'output'):
                            if pretrained_layer.output.dense.weight.shape == block.mlp.fc2.weight.shape:
                                block.mlp.fc2.weight.data.copy_(pretrained_layer.output.dense.weight)
                                loaded_count += 1

    print(f"\n✓ Loaded {loaded_count} parameter groups from pretrained model")

    # Freeze early layers if requested
    if freeze_early_layers:
        print(f"\n{'='*60}")
        print(f"Freezing first {num_frozen_layers} transformer layers")
        print(f"{'='*60}")

        frozen_params = 0
        for i in range(min(num_frozen_layers, len(encoder.layers))):
            for param in encoder.layers[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()

        # Also freeze patch embedding
        for param in encoder.patch_embed.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in encoder.parameters())

        print(f"✓ Frozen {frozen_params:,} parameters in encoder")
        print(f"✓ Trainable encoder parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    print(f"{'='*60}\n")


def load_pretrained_decoders(
    checkpoint_path: str,
    model: nn.Module,
    strict: bool = False,
) -> None:
    """Load pretrained decoder weights from a Mamba checkpoint.

    This is useful for initializing ViT model with trained Mamba decoders,
    allowing the ViT encoder to learn the latent space expected by the decoders.

    Handles checkpoints saved with DataParallel (module.* prefix).

    Args:
        checkpoint_path: Path to pretrained Mamba checkpoint
        model: Model to load decoder weights into (typically ViTULight)
        strict: Whether to strictly enforce that keys match
    """
    print(f"\n{'='*60}")
    print(f"Loading pretrained decoders from checkpoint")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get state dict - try multiple common keys
    if 'model_state_dict' in checkpoint:
        pretrained_state = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        pretrained_state = checkpoint['state_dict']
    else:
        pretrained_state = checkpoint

    print(f"Found {len(pretrained_state)} parameters in checkpoint")

    # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
    cleaned_state = {}
    for key, value in pretrained_state.items():
        if key.startswith('module.'):
            # Remove 'module.' prefix
            new_key = key[7:]  # len('module.') = 7
            cleaned_state[new_key] = value
        else:
            cleaned_state[key] = value

    # Filter to only decoder parameters
    decoder_keywords = ['reg_decoder', 'fus_decoder', 'SR_decoder', 'IR_decoder']
    decoder_state = {}

    for key, value in cleaned_state.items():
        if any(decoder in key for decoder in decoder_keywords):
            decoder_state[key] = value

    if len(decoder_state) == 0:
        print("❌ WARNING: No decoder parameters found in checkpoint!")
        print(f"Available keys (first 10): {list(cleaned_state.keys())[:10]}")
        return

    print(f"\nFound {len(decoder_state)} decoder parameters to load")

    # Show breakdown by decoder
    for decoder_name in decoder_keywords:
        count = sum(1 for k in decoder_state.keys() if decoder_name in k)
        if count > 0:
            print(f"  - {decoder_name}: {count} parameters")

    # Inflate 2D weights to 3D if needed
    print(f"\n{'='*60}")
    print("Checking for 2D→3D weight inflation")
    print(f"{'='*60}")

    inflated_count = 0
    channel_adjusted_count = 0

    for key, value in list(decoder_state.items()):
        # Check if this is a Conv weight that needs inflation
        if 'weight' in key and value.ndim == 4:  # 2D conv: [out_ch, in_ch, H, W]
            # Get corresponding parameter in model to check expected shape
            try:
                model_param = model.state_dict()[key]
                if model_param.ndim == 5:  # Model expects 3D conv: [out_ch, in_ch, D, H, W]
                    # Inflate 2D → 3D by repeating along depth dimension and averaging
                    out_ch, in_ch, h, w = value.shape
                    target_d = model_param.shape[2]

                    # Repeat along depth dimension
                    inflated_weight = value.unsqueeze(2).repeat(1, 1, target_d, 1, 1)

                    # Average to preserve magnitude (divide by depth)
                    inflated_weight = inflated_weight / target_d

                    # Check if output channels also need adjustment (e.g., 2D→3D for displacement field)
                    if inflated_weight.shape[0] != model_param.shape[0]:
                        target_out_ch = model_param.shape[0]

                        if inflated_weight.shape[0] < target_out_ch:
                            # Need to add channels (e.g., reg_decoder.head: 2→3 for dx,dy,dz)
                            # Initialize new channels with zeros
                            missing_channels = target_out_ch - inflated_weight.shape[0]
                            zero_channels = torch.zeros(
                                missing_channels, inflated_weight.shape[1],
                                inflated_weight.shape[2], inflated_weight.shape[3],
                                inflated_weight.shape[4]
                            )
                            inflated_weight = torch.cat([inflated_weight, zero_channels], dim=0)
                            channel_adjusted_count += 1

                            if channel_adjusted_count <= 2:
                                print(f"✓ Adjusted output channels {key}: {out_ch}→{target_out_ch} (new channels initialized to zero)")

                    decoder_state[key] = inflated_weight
                    inflated_count += 1

                    if inflated_count <= 3:  # Show first few
                        print(f"✓ Inflated {key}: {value.shape} → {inflated_weight.shape}")
            except KeyError:
                # Parameter not in model, will be caught by load_state_dict
                pass

        # Handle bias terms with channel mismatch
        elif 'bias' in key:
            try:
                model_param = model.state_dict()[key]
                if value.shape[0] < model_param.shape[0]:
                    # Add zeros for missing bias terms
                    missing_biases = model_param.shape[0] - value.shape[0]
                    zero_biases = torch.zeros(missing_biases)
                    adjusted_bias = torch.cat([value, zero_biases], dim=0)
                    decoder_state[key] = adjusted_bias
                    channel_adjusted_count += 1

                    if channel_adjusted_count <= 2:
                        print(f"✓ Adjusted bias {key}: {value.shape[0]}→{model_param.shape[0]}")
            except KeyError:
                pass

    if inflated_count > 0:
        print(f"\n✓ Inflated {inflated_count} conv weights from 2D to 3D")
    if channel_adjusted_count > 0:
        print(f"✓ Adjusted {channel_adjusted_count} parameters for 2D→3D output channel mismatch")
    if inflated_count == 0 and channel_adjusted_count == 0:
        print("✓ No weight inflation needed (weights already match model dimensions)")
    print(f"{'='*60}\n")

    # Load decoder weights
    missing, unexpected = model.load_state_dict(decoder_state, strict=False)

    print(f"\n✓ Successfully loaded decoder weights!")

    if missing:
        # Filter out encoder parameters from missing (those are expected)
        missing_decoders = [k for k in missing if any(dec in k for dec in decoder_keywords)]
        if missing_decoders:
            print(f"⚠️  Warning: {len(missing_decoders)} decoder parameters not found in checkpoint")
            if len(missing_decoders) <= 10:
                print(f"Missing: {missing_decoders}")

    if unexpected:
        print(f"⚠️  Warning: {len(unexpected)} unexpected parameters in checkpoint")
        if len(unexpected) <= 10:
            print(f"Unexpected: {unexpected}")

    print(f"{'='*60}\n")


def main(args):
    """Main training function."""

    # Load configuration
    if args.model.lower() == 'vit':
        config = ViT3DConfig.from_yaml(args.config) if args.config else ViT3DConfig()
    else:
        config = MambaULightConfig.from_yaml(args.config) if args.config else MambaULightConfig()

    # Override config with command line arguments
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs

    # Set random seed
    set_seed(config.seed)

    # Create directories
    config.create_directories()

    # Initialize wandb
    if WANDB_AVAILABLE and config.wandb_project is not None:
        wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            config=config.to_dict(),
        )

    # Create model
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Freeze decoders if specified in config or args
    freeze_decoders = getattr(config, 'freeze_decoders', False) or args.freeze_decoders

    model = create_model(config, model_type=args.model, freeze_decoders=freeze_decoders)
    model = model.to(device)

    # Print model details
    if args.verbose:
        from src.vit_model import print_model_details
        print_model_details(model)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Mixed precision training
    scaler = GradScaler() if args.amp else None

    # Load pretrained ViT encoder from HuggingFace (only for ViT model)
    if args.model.lower() == 'vit' and args.pretrained_vit is not None:
        load_pretrained_vit_encoder(
            model,
            pretrained_model_name=args.pretrained_vit,
            freeze_early_layers=args.freeze_vit_early_layers,
            num_frozen_layers=args.num_frozen_vit_layers,
        )

    # Load pretrained decoders if specified (useful for ViT encoder training)
    pretrained_decoder_path = args.pretrained_decoders or getattr(config, 'pretrained_path', None)
    if pretrained_decoder_path is not None:
        print(f"\nLoading pretrained decoders from: {pretrained_decoder_path}")
        load_pretrained_decoders(pretrained_decoder_path, model)

        # Optionally unfreeze last N decoder layers for fine-tuning
        if hasattr(args, 'unfreeze_decoder_layers') and args.unfreeze_decoder_layers > 0:
            print(f"\n{'='*60}")
            print(f"Unfreezing last {args.unfreeze_decoder_layers} layers of each decoder")
            print(f"{'='*60}")

            for decoder_name in ['reg_decoder', 'fus_decoder', 'SR_decoder', 'IR_decoder']:
                if hasattr(model, decoder_name):
                    decoder = getattr(model, decoder_name)
                    if hasattr(decoder, 'decoder') and hasattr(decoder.decoder, 'children'):
                        decoder_layers = list(decoder.decoder.children())
                        # Unfreeze last N layers
                        for layer in decoder_layers[-args.unfreeze_decoder_layers:]:
                            for param in layer.parameters():
                                param.requires_grad = True

                    # Always unfreeze the head for fine-tuning
                    if hasattr(decoder, 'head'):
                        for param in decoder.head.parameters():
                            param.requires_grad = True

            trainable_decoder_params = sum(
                p.numel() for name, p in model.named_parameters()
                if p.requires_grad and any(d in name for d in ['reg_decoder', 'fus_decoder', 'SR_decoder', 'IR_decoder'])
            )
            print(f"✓ Trainable decoder parameters: {trainable_decoder_params:,}")
            print(f"{'='*60}\n")

    # Load checkpoint if resuming (do this after pretrained loading)
    start_epoch = 0
    best_loss = float('inf')
    if args.resume is not None:
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    # Create datasets and dataloaders
    # Load real biomedical datasets (simplified approach matching original)
    train_dataset = BiomedicalDataset(
        data_root="/group/jug/aman/orochi/data",
        datasets=['hipsc_3d', 'hipsc_2d', 'hipct_2d', 'idr_2d'],
        img_size=config.img_size,
        split='train',
        val_split=0.1
    )
    val_dataset = BiomedicalDataset(
        data_root="/group/jug/aman/orochi/data",
        datasets=['hipsc_3d', 'hipsc_2d', 'hipct_2d', 'idr_2d'],
        img_size=config.img_size,
        split='val',
        val_split=0.1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")

    # Training loop
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config, scaler
        )

        print(f"Epoch {epoch}/{config.num_epochs} - Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        if epoch % config.val_interval == 0:
            val_metrics = validate(
                model, val_loader, device, epoch, config
            )
            print(f"Epoch {epoch}/{config.num_epochs} - Val Loss: {val_metrics['loss']:.4f}")

            # Save best checkpoint
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_loss, config,
                    f"{config.experiment_name}_best.pth"
                )

        # Save periodic checkpoint
        if epoch % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_loss, config,
                f"{config.experiment_name}_epoch_{epoch}.pth"
            )

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Log learning rate
        if WANDB_AVAILABLE and config.wandb_project is not None:
            wandb.log({
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            })

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, config.num_epochs, best_loss, config,
        f"{config.experiment_name}_final.pth"
    )

    # Finish wandb
    if WANDB_AVAILABLE and config.wandb_project is not None:
        wandb.finish()

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Mamba/ViT with wandb')

    # Model arguments
    parser.add_argument('--model', type=str, required=True, choices=['vit', 'mamba'],
                        help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed model information')

    # Transfer learning arguments
    parser.add_argument('--freeze_decoders', action='store_true',
                        help='Freeze decoder parameters (train encoder only)')
    parser.add_argument('--pretrained_decoders', type=str, default=None,
                        help='Path to checkpoint with pretrained decoders (e.g., trained Mamba model)')
    parser.add_argument('--unfreeze_decoder_layers', type=int, default=0,
                        help='Number of last decoder layers to unfreeze for fine-tuning (0=all frozen)')

    # Pretrained ViT arguments
    parser.add_argument('--pretrained_vit', type=str, default=None,
                        help='HuggingFace ViT model name (e.g., google/vit-base-patch16-224)')
    parser.add_argument('--freeze_vit_early_layers', action='store_true',
                        help='Freeze early ViT transformer layers (recommended with pretrained weights)')
    parser.add_argument('--num_frozen_vit_layers', type=int, default=8,
                        help='Number of early ViT layers to freeze (default: 8 out of 12)')

    args = parser.parse_args()

    main(args)
