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
- (NEW) torch.distributed DistributedDataParallel support (multi-GPU, multi-node)

Usage:

# Single-node, multi-GPU with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/finetune_with_wandb.py --config configs/vit_finetune.yaml --model vit --distributed

# Multi-node (set MASTER_ADDR/MASTER_PORT outside, or via SLURM env)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
    src/finetune_with_wandb.py --config configs/vit_finetune.yaml --model vit --distributed

# Resume from checkpoint
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/finetune_with_wandb.py --config configs/vit_finetune.yaml \
    --model vit --distributed --resume checkpoints/best.pth
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from tqdm import tqdm
import tifffile

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    print("⚠️ Warning: Could not import MambaULight (mamba_ssm not available)")
    print(f"   Error: {e}")
    print("   Mamba model will not be available. Use --model vit instead.")
    MAMBA_AVAILABLE = False
    MambaULight = None

import src.utils as utils


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (slower), you can enable:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


class DummyDataset(Dataset):
    """Dummy dataset for testing and demonstration.

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
        """Return a random sample."""
        image = torch.rand(self.in_chans, *self.img_size)
        return {"image": image, "idx": idx}


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
        datasets=("hipsc_3d",),
        img_size=(64, 128, 128),
        split="train",
        val_split=0.1,
        subset_frac=None
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.subset_frac = subset_frac
        # Collect image files (.tiff, .tif, .npy)
        self.image_files = []
        for dataset_name in datasets:
            dataset_path = self.data_root / dataset_name
            if not dataset_path.exists():
                print(f"⚠️ Dataset not found: {dataset_path}")
                continue

            if dataset_name == "hipsc_3d":
                # Protein subdirectories
                for protein_dir in dataset_path.iterdir():
                    if protein_dir.is_dir():
                        self.image_files.extend(protein_dir.glob("*.tif*"))
                        self.image_files.extend(protein_dir.glob("*.npy"))
            else:
                # data/ subdirectory
                data_dir = dataset_path / "data"
                if data_dir.exists():
                    self.image_files.extend(data_dir.glob("*.tif*"))
                    self.image_files.extend(data_dir.glob("*.npy"))
        if self.subset_frac is not None and 0 < self.subset_frac <= 1.0:
            import random
            random.seed(42 + (1 if split == 'val' else 0))  # Reproducible train/val split
            n_total = len(self.image_files)
            n_subset = max(1, int(n_total * self.subset_frac))
            self.image_files = random.sample(self.image_files, n_subset)
            if split == 'train':
                print(f"🔹 TRAIN SUBSET: {len(self.image_files):,} / {n_total:,} "
                    f"({100*self.subset_frac:.1f}%)")
            else:
                print(f"🔹 VAL SUBSET: {len(self.image_files):,} / {n_total:,} "
                    f"({100*self.subset_frac:.1f}%)")
        # Sort and split
        self.image_files = sorted(self.image_files)
        n_total = len(self.image_files)
        n_val = int(n_total * val_split)
        if split == "train":
            self.image_files = self.image_files[: n_total - n_val]
        else:
            self.image_files = self.image_files[n_total - n_val :]

        print(f"📁 {split.capitalize()}: {len(self.image_files)} images from {datasets}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load and preprocess image."""
        img_path = self.image_files[idx]

        # Load image
        if img_path.suffix == ".npy":
            raw_image = np.load(str(img_path))
        else:  # .tiff, .tif
            raw_image = tifffile.imread(str(img_path))

        # To tensor with channel dimension
        image_tensor = torch.from_numpy(raw_image).float().unsqueeze(0)

        # Min-max normalization
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min() + 1e-8
        )

        # Resize to target size (upsample or downsample as needed)
        image_tensor = self._resize(image_tensor)

        return {"image": image_tensor, "idx": idx}

    def _resize(self, image_tensor):
        """Resize to target size (upsample or downsample as needed)."""
        d, h, w = image_tensor.shape[1:]  # (C, D, H, W)
        td, th, tw = self.img_size

        # Resize if dimensions don't match target
        if (d, h, w) != (td, th, tw):
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(td, th, tw),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return image_tensor


def create_model(config, model_type="vit", freeze_decoders=False):
    """Create model based on configuration."""
    if model_type.lower() == "vit":
        print(f"Creating Vision Transformer model with {sum(config.depths)} blocks")
        task = getattr(config, "task", "multi_task")
        print(f"Training task: {task.upper()}")
        model = ViTULight(config)
    elif model_type.lower() == "mamba":
        if not MAMBA_AVAILABLE:
            raise ValueError(
                "Mamba model requested but mamba_ssm is not available.\n"
                "Please install mamba_ssm or use --model vit instead."
            )
        print(f"Creating Mamba model with {sum(config.depths)} blocks")
        task = getattr(config, "task", "multi_task")
        print(f"Training task: {task.upper()}")
        model = MambaULight(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vit' or 'mamba'")

    # Freeze decoders if requested
    if freeze_decoders:
        print("Freezing decoder parameters (training encoder only)")
        for name, param in model.named_parameters():
            if any(
                decoder in name
                for decoder in ["reg_decoder", "fus_decoder", "SR_decoder", "IR_decoder", "spatial_trans"]
            ):
                param.requires_grad = False

    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    print(
        f"Trainable parameters: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    print(
        f"Frozen parameters:   {frozen_params:,} "
        f"({100 * frozen_params / total_params:.2f}%)"
    )
    return model


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.optimizer.lower() == "sgd":
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
    """Create learning rate scheduler."""
    if config.scheduler is None:
        return None

    if config.scheduler.lower() == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.lr_decay_epochs,
            gamma=config.lr_decay_rate,
        )
    elif config.scheduler.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr,
        )
    elif config.scheduler.lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_decay_rate,
            patience=10,
            min_lr=config.min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
    return scheduler


def compute_total_loss(aux_loss: Dict) -> torch.Tensor:
    """Compute total loss from auxiliary losses with NaN/Inf detection.

    Handles both single-task and multi-task modes by summing only present losses.
    Detects and handles NaN/Inf values to prevent training collapse.

    Args:
        aux_loss: Dictionary with structure {"mse": {...}, "ncc": {...}, "grad": {...}}

    Returns:
        Total loss tensor
    """
    total_loss = 0.0
    has_nan_or_inf = False

    # MSE losses for all tasks (with NaN/Inf checking)
    for task_name, loss_val in aux_loss["mse"].items():
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            print(f"WARNING: {task_name} MSE loss is {loss_val.item()} - skipping")
            has_nan_or_inf = True
            continue
        total_loss += loss_val

    # NCC loss for registration (if present)
    if "ncc" in aux_loss and "reg" in aux_loss["ncc"]:
        ncc_loss = aux_loss["ncc"]["reg"]
        if torch.isnan(ncc_loss) or torch.isinf(ncc_loss):
            print(f"WARNING: NCC loss is {ncc_loss.item()} - skipping")
            has_nan_or_inf = True
        else:
            total_loss += ncc_loss

    # Gradient regularization for registration (if present)
    if "grad" in aux_loss and "reg" in aux_loss["grad"]:
        grad_loss = aux_loss["grad"]["reg"]
        if torch.isnan(grad_loss) or torch.isinf(grad_loss):
            print(f"WARNING: Gradient loss is {grad_loss.item()} - skipping")
            has_nan_or_inf = True
        else:
            total_loss += 0.01 * grad_loss  # smoothness weight

    # Final check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"ERROR: Total loss is {total_loss.item()}")
        if has_nan_or_inf:
            print("Returning zero loss to continue training")
            return torch.tensor(0.0, device=total_loss.device, requires_grad=True)

    return total_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config,
    scaler: Optional[GradScaler] = None,
    is_master: bool = True,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch with gradient accumulation support.

    Args:
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating.
                                     Effective batch size = batch_size × accumulation_steps × num_gpus
    """
    model.train()

    loss_meter = utils.AverageMeter()
    reg_loss_meter = utils.AverageMeter()
    fus_loss_meter = utils.AverageMeter()
    sr_loss_meter = utils.AverageMeter()
    ir_loss_meter = utils.AverageMeter()

    pbar = dataloader
    if is_master:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)

        # Forward pass
        if scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits, aux_loss = model(images)
                loss = compute_total_loss(aux_loss)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
        else:
            logits, aux_loss = model(images)
            loss = compute_total_loss(aux_loss)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass (accumulate gradients)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every N steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                if config.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            # Zero gradients after optimizer step
            optimizer.zero_grad(set_to_none=True)

        # Update meters (use unscaled loss for logging)
        loss_meter.update(loss.item() * gradient_accumulation_steps)
        if "reg" in aux_loss["mse"]:
            reg_loss_meter.update(aux_loss["mse"]["reg"].item())
        if "fus" in aux_loss["mse"]:
            fus_loss_meter.update(aux_loss["mse"]["fus"].item())
        if "SR" in aux_loss["mse"]:
            sr_loss_meter.update(aux_loss["mse"]["SR"].item())
        if "IR" in aux_loss["mse"]:
            ir_loss_meter.update(aux_loss["mse"]["IR"].item())

        if is_master:
            postfix = {"loss": f"{loss_meter.avg:.4f}"}
            if "reg" in aux_loss["mse"]:
                postfix["reg"] = f"{reg_loss_meter.avg:.4f}"
            if "fus" in aux_loss["mse"]:
                postfix["fus"] = f"{fus_loss_meter.avg:.4f}"
            if "SR" in aux_loss["mse"]:
                postfix["sr"] = f"{sr_loss_meter.avg:.4f}"
            if "IR" in aux_loss["mse"]:
                postfix["ir"] = f"{ir_loss_meter.avg:.4f}"
            pbar.set_postfix(postfix)

            # Log to wandb (only master)
            if WANDB_AVAILABLE and config.wandb_project is not None:
                if batch_idx % config.log_interval == 0:
                    log_dict = {
                        "train/loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx,
                    }
                    if "reg" in aux_loss["mse"]:
                        log_dict["train/reg_loss"] = aux_loss["mse"]["reg"].item()
                    if "reg" in aux_loss.get("ncc", {}):
                        log_dict["train/ncc_loss"] = aux_loss["ncc"]["reg"].item()
                    if "reg" in aux_loss.get("grad", {}):
                        log_dict["train/grad_loss"] = aux_loss["grad"]["reg"].item()
                    if "fus" in aux_loss["mse"]:
                        log_dict["train/fus_loss"] = aux_loss["mse"]["fus"].item()
                    if "SR" in aux_loss["mse"]:
                        log_dict["train/sr_loss"] = aux_loss["mse"]["SR"].item()
                    if "IR" in aux_loss["mse"]:
                        log_dict["train/ir_loss"] = aux_loss["mse"]["IR"].item()
                    wandb.log(log_dict)

    return {
        "loss": loss_meter.avg,
        "reg_loss": reg_loss_meter.avg,
        "fus_loss": fus_loss_meter.avg,
        "sr_loss": sr_loss_meter.avg,
        "ir_loss": ir_loss_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    config,
    is_master: bool = True,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    loss_meter = utils.AverageMeter()
    reg_loss_meter = utils.AverageMeter()
    fus_loss_meter = utils.AverageMeter()
    sr_loss_meter = utils.AverageMeter()
    ir_loss_meter = utils.AverageMeter()

    pbar = dataloader
    if is_master:
        pbar = tqdm(dataloader, desc="Validation")

    first_batch_logged = False

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)

        logits, aux_loss = model(images)
        loss = compute_total_loss(aux_loss)

        loss_meter.update(loss.item())
        if "reg" in aux_loss["mse"]:
            reg_loss_meter.update(aux_loss["mse"]["reg"].item())
        if "fus" in aux_loss["mse"]:
            fus_loss_meter.update(aux_loss["mse"]["fus"].item())
        if "SR" in aux_loss["mse"]:
            sr_loss_meter.update(aux_loss["mse"]["SR"].item())
        if "IR" in aux_loss["mse"]:
            ir_loss_meter.update(aux_loss["mse"]["IR"].item())

        # Log sample images only once, on master
        if (
            is_master
            and WANDB_AVAILABLE
            and config.wandb_project is not None
            and not first_batch_logged
        ):
            try:
                log_images = {
                    "val/raw": wandb.Image(
                        logits["raw"][0, 0, logits["raw"].shape[2] // 2]
                    ),
                    "val/reg_deformed": wandb.Image(
                        logits["reg"]["deformed"][
                            0, 0, logits["reg"]["deformed"].shape[2] // 2
                        ]
                    ),
                    "val/reg_registered": wandb.Image(
                        logits["reg"]["registered"][
                            0, 0, logits["reg"]["registered"].shape[2] // 2
                        ]
                    ),
                    "val/fused": wandb.Image(
                        logits["fus"]["fused"][0, 0, logits["fus"]["fused"].shape[2] // 2]
                    ),
                    "val/sr": wandb.Image(
                        logits["SR"]["super_resolution"][
                            0,
                            0,
                            logits["SR"]["super_resolution"].shape[2] // 2,
                        ]
                    ),
                    "val/ir": wandb.Image(
                        logits["IR"]["restored"][
                            0, 0, logits["IR"]["restored"].shape[2] // 2
                        ]
                    ),
                }
                wandb.log(log_images)
                first_batch_logged = True
            except Exception:
                pass

    if is_master and WANDB_AVAILABLE and config.wandb_project is not None:
        wandb.log(
            {
                "val/loss": loss_meter.avg,
                "val/reg_loss": reg_loss_meter.avg,
                "val/fus_loss": fus_loss_meter.avg,
                "val/sr_loss": sr_loss_meter.avg,
                "val/ir_loss": ir_loss_meter.avg,
                "epoch": epoch,
            }
        )

    return {
        "loss": loss_meter.avg,
        "reg_loss": reg_loss_meter.avg,
        "fus_loss": fus_loss_meter.avg,
        "sr_loss": sr_loss_meter.avg,
        "ir_loss": ir_loss_meter.avg,
    }


def generate_checkpoint_name(
    config, epoch: int = None, checkpoint_type: str = "periodic"
) -> str:
    """Generate descriptive checkpoint filename."""
    model_name = "vit" if "ViT" in str(type(config).__name__) else "mamba"
    task = getattr(config, "task", "multitask")
    task_str = task.replace("_", "-")

    freeze_decoders = getattr(config, "freeze_decoders", False)
    freeze_encoder = getattr(config, "freeze_encoder", False)
    if freeze_decoders:
        mode = "frozen-decoders"
    elif freeze_encoder:
        mode = "frozen-encoder"
    else:
        mode = "full-finetune"

    parts = [model_name, task_str, mode, checkpoint_type]
    if epoch is not None:
        parts.append(f"epoch-{epoch:03d}")
    filename = "_".join(parts) + ".pth"
    return filename


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_loss: float,
    config,
    filename: str,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_loss": best_loss,
        "config": config.to_dict(),
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
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(f"Loaded checkpoint from epoch {epoch} with best loss {best_loss:.4f}")
    return epoch, best_loss


def load_pretrained_vit_encoder(
    model: nn.Module,
    pretrained_model_name: str = "google/vit-base-patch16-224",
    freeze_early_layers: bool = True,
    num_frozen_layers: int = 8,
) -> None:
    """Load pretrained ViT encoder weights from HuggingFace and inflate 2D→3D."""
    try:
        from transformers import ViTModel
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        print("Skipping pretrained weight loading.")
        return

    print("\n" + "=" * 60)
    print(f"Loading pretrained ViT weights from: {pretrained_model_name}")
    print("=" * 60)

    try:
        pretrained_vit = ViTModel.from_pretrained(pretrained_model_name)
        print("✓ Successfully downloaded pretrained model")
    except Exception as e:
        print(f"❌ Failed to load pretrained model: {e}")
        return

    if hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        print("❌ Model doesn't have 'encoder' attribute")
        return

    loaded_count = 0

    # Patch embedding weights
    if hasattr(pretrained_vit.embeddings, "patch_embeddings") and hasattr(
        encoder, "patch_embed"
    ):
        pretrained_patch_weight = (
            pretrained_vit.embeddings.patch_embeddings.projection.weight
        )
        patch_size_depth = encoder.patch_embed.proj.kernel_size[0]
        inflated_weight = pretrained_patch_weight.unsqueeze(2).repeat(
            1, 1, patch_size_depth, 1, 1
        )
        inflated_weight = inflated_weight / patch_size_depth
        if inflated_weight.shape == encoder.patch_embed.proj.weight.shape:
            encoder.patch_embed.proj.weight.data.copy_(inflated_weight)
            loaded_count += 1
            print("✓ Loaded patch embedding weights (inflated 2D→3D)")
        else:
            print(
                f"⚠️ Patch embedding size mismatch: "
                f"{inflated_weight.shape} vs {encoder.patch_embed.proj.weight.shape}"
            )

    # Transformer layers
    if hasattr(pretrained_vit.encoder, "layer"):
        num_layers = min(len(pretrained_vit.encoder.layer), len(encoder.layers))
        for i in range(num_layers):
            pretrained_layer = pretrained_vit.encoder.layer[i]
            our_layer = encoder.layers[i]

            if hasattr(our_layer, "blocks") and len(our_layer.blocks) > 0:
                for block_idx, block in enumerate(our_layer.blocks):
                    if hasattr(block, "attn") and hasattr(pretrained_layer, "attention"):
                        if hasattr(pretrained_layer.attention.attention, "query"):
                            qkv_weight = torch.cat(
                                [
                                    pretrained_layer.attention.attention.query.weight,
                                    pretrained_layer.attention.attention.key.weight,
                                    pretrained_layer.attention.attention.value.weight,
                                ],
                                dim=0,
                            )
                            if qkv_weight.shape == block.attn.qkv.weight.shape:
                                block.attn.qkv.weight.data.copy_(qkv_weight)
                                loaded_count += 1

                        if hasattr(pretrained_layer.attention.output, "dense"):
                            if (
                                pretrained_layer.attention.output.dense.weight.shape
                                == block.attn.proj.weight.shape
                            ):
                                block.attn.proj.weight.data.copy_(
                                    pretrained_layer.attention.output.dense.weight
                                )
                                loaded_count += 1

                    if hasattr(block, "mlp") and hasattr(pretrained_layer, "intermediate"):
                        if (
                            pretrained_layer.intermediate.dense.weight.shape
                            == block.mlp.fc1.weight.shape
                        ):
                            block.mlp.fc1.weight.data.copy_(
                                pretrained_layer.intermediate.dense.weight
                            )
                            loaded_count += 1

                        if hasattr(pretrained_layer, "output"):
                            if (
                                pretrained_layer.output.dense.weight.shape
                                == block.mlp.fc2.weight.shape
                            ):
                                block.mlp.fc2.weight.data.copy_(
                                    pretrained_layer.output.dense.weight
                                )
                                loaded_count += 1

    print(f"\n✓ Loaded {loaded_count} parameter groups from pretrained model")

    # Optionally freeze early layers
    if freeze_early_layers:
        print("\n" + "=" * 60)
        print(f"Freezing first {num_frozen_layers} transformer layers")
        print("=" * 60)
        frozen_params = 0
        for i in range(min(num_frozen_layers, len(encoder.layers))):
            for param in encoder.layers[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()

        for param in encoder.patch_embed.parameters():
            param.requires_grad = False
            frozen_params += param.numel()

        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"✓ Frozen {frozen_params:,} parameters in encoder")
        print(
            f"✓ Trainable encoder parameters: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.1f}%)"
        )
        print("=" * 60 + "\n")


def load_pretrained_decoders(
    checkpoint_path: str,
    model: nn.Module,
    strict: bool = False,
) -> None:
    """Load pretrained decoder weights from a Mamba checkpoint."""
    print("\n" + "=" * 60)
    print("Loading pretrained decoders from checkpoint")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        pretrained_state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        pretrained_state = checkpoint["state_dict"]
    else:
        pretrained_state = checkpoint
    print(f"Found {len(pretrained_state)} parameters in checkpoint")

    cleaned_state = {}
    for key, value in pretrained_state.items():
        if key.startswith("module."):
            cleaned_state[key[7:]] = value
        else:
            cleaned_state[key] = value

    decoder_keywords = ["reg_decoder", "fus_decoder", "SR_decoder", "IR_decoder"]
    decoder_state = {
        key: value
        for key, value in cleaned_state.items()
        if any(decoder in key for decoder in decoder_keywords)
    }

    if len(decoder_state) == 0:
        print("❌ WARNING: No decoder parameters found in checkpoint!")
        print(f"Available keys (first 10): {list(cleaned_state.keys())[:10]}")
        return

    print(f"\nFound {len(decoder_state)} decoder parameters to load")
    for decoder_name in decoder_keywords:
        count = sum(1 for k in decoder_state.keys() if decoder_name in k)
        if count > 0:
            print(f" - {decoder_name}: {count} parameters")

    print("\n" + "=" * 60)
    print("Checking for 2D→3D weight inflation")
    print("=" * 60)
    inflated_count = 0
    channel_adjusted_count = 0

    for key, value in list(decoder_state.items()):
        if "weight" in key and value.ndim == 4:
            try:
                model_param = model.state_dict()[key]
                if model_param.ndim == 5:
                    out_ch, in_ch, h, w = value.shape
                    target_d = model_param.shape[2]
                    inflated_weight = value.unsqueeze(2).repeat(
                        1, 1, target_d, 1, 1
                    )
                    inflated_weight = inflated_weight / target_d

                    if inflated_weight.shape[0] != model_param.shape[0]:
                        target_out_ch = model_param.shape[0]
                        if inflated_weight.shape[0] < target_out_ch:
                            missing_channels = (
                                target_out_ch - inflated_weight.shape[0]
                            )
                            zero_channels = torch.zeros(
                                missing_channels,
                                inflated_weight.shape[1],
                                inflated_weight.shape[2],
                                inflated_weight.shape[3],
                                inflated_weight.shape[4],
                            )
                            inflated_weight = torch.cat(
                                [inflated_weight, zero_channels], dim=0
                            )
                            channel_adjusted_count += 1
                            if channel_adjusted_count <= 2:
                                print(
                                    f"✓ Adjusted output channels {key}: "
                                    f"{out_ch}→{target_out_ch} "
                                    "(new channels initialized to zero)"
                                )

                    decoder_state[key] = inflated_weight
                    inflated_count += 1
                    if inflated_count <= 3:
                        print(
                            f"✓ Inflated {key}: {value.shape} "
                            f"→ {inflated_weight.shape}"
                        )
            except KeyError:
                pass

        elif "bias" in key:
            try:
                model_param = model.state_dict()[key]
                if value.shape[0] < model_param.shape[0]:
                    missing_biases = model_param.shape[0] - value.shape[0]
                    zero_biases = torch.zeros(missing_biases)
                    adjusted_bias = torch.cat([value, zero_biases], dim=0)
                    decoder_state[key] = adjusted_bias
                    channel_adjusted_count += 1
                    if channel_adjusted_count <= 2:
                        print(
                            f"✓ Adjusted bias {key}: "
                            f"{value.shape[0]}→{model_param.shape[0]}"
                        )
            except KeyError:
                pass

    if inflated_count > 0:
        print(f"\n✓ Inflated {inflated_count} conv weights from 2D to 3D")
    if channel_adjusted_count > 0:
        print(
            f"✓ Adjusted {channel_adjusted_count} parameters "
            "for 2D→3D output channel mismatch"
        )
    if inflated_count == 0 and channel_adjusted_count == 0:
        print("✓ No weight inflation needed (weights already match model dimensions)")
    print("=" * 60 + "\n")

    missing, unexpected = model.load_state_dict(decoder_state, strict=False)
    print("\n✓ Successfully loaded decoder weights!")
    if missing:
        missing_decoders = [
            k for k in missing if any(dec in k for dec in decoder_keywords)
        ]
        if missing_decoders:
            print(
                f"⚠️ Warning: {len(missing_decoders)} decoder parameters not found in checkpoint"
            )
            if len(missing_decoders) <= 10:
                print(f"Missing: {missing_decoders}")
    if unexpected:
        print(
            f"⚠️ Warning: {len(unexpected)} unexpected parameters in checkpoint"
        )
        if len(unexpected) <= 10:
            print(f"Unexpected: {unexpected}")
    print("=" * 60 + "\n")


def setup_distributed(args):
    """Initialize torch.distributed, using env vars set by torchrun/SLURM."""
    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.rank = 0
        args.world_size = 1

    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend=args.backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    torch.cuda.set_device(args.local_rank)
    # Optionally reseed per rank
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def cleanup_distributed(args):
    """Tear down distributed process group."""
    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()

def wandb_model(model, optimizer, scheduler, epoch, val_loss, config, step="latest", is_master=True):
    """✅ FIXED WandB checkpoint - correct aliases API."""
    if not (is_master and WANDB_AVAILABLE): 
        return
    
    ckpt = {
        'epoch': epoch,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config.to_dict(),
    }
    
    filename = f"e{epoch:03d}_l{val_loss:.4f}_{step}.pth"
    torch.save(ckpt, filename)
    
    # FIXED: Correct syntax - aliases in log_artifact()
    artifact = wandb.Artifact(f"model-{config.experiment_name}", type="model")
    artifact.add_file(filename)
    aliases = ["latest"]
    if step == "best":
        aliases.append("best")
    
    wandb.log_artifact(artifact, aliases=aliases)  # ✅ This works!
    
    os.remove(filename)
    print(f"💾 WandB {step.upper()}: e{epoch} l{val_loss:.4f}")

def main(args):
    """Ultra-clean training with smart WandB checkpointing."""
    
    # === 1. CONFIG ===
    if args.model.lower() == "vit":
        config = ViT3DConfig.from_yaml(args.config) if args.config else ViT3DConfig()
    else:
        config = MambaULightConfig.from_yaml(args.config) if args.config else MambaULightConfig()
    
    # CLI overrides
    overrides = ['wandb_project', 'experiment_name', 'batch_size', 'learning_rate', 'num_epochs']
    for attr in overrides:
        if getattr(args, attr) is not None:
            setattr(config, attr, getattr(args, attr))
    
    args.subset = args.subset or getattr(config, 'subset', None)
    set_seed(config.seed)
    config.create_directories()

    # === 2. DISTRIBUTED ===
    setup_distributed(args)
    is_master = not args.distributed or args.rank == 0

    # === 3. WANDB ===
    if is_master and WANDB_AVAILABLE and config.wandb_project:
        wandb.init(project=config.wandb_project, name=config.experiment_name, config=config.to_dict())

    # === 4. DEVICE ===
    device = torch.device("cuda", args.local_rank) if args.distributed else torch.device(config.device)
    if is_master: print(f"🚀 {device} | GPUs: {torch.cuda.device_count()}")

    # === 5. MODEL ===
    model = create_model(config, args.model, freeze_decoders=args.freeze_decoders)
    model.to(device)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, 
                   find_unused_parameters=False)

    model_for_optim = model.module if isinstance(model, DDP) else model

    # === 6. OPTIMIZER ===
    optimizer = create_optimizer(model_for_optim, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler(device='cuda', enabled=args.amp)

    # === 7. PRETRAINED ===
    if args.model.lower() == "vit" and args.pretrained_vit:
        load_pretrained_vit_encoder(model_for_optim, args.pretrained_vit, 
                                  args.freeze_vit_early_layers, args.num_frozen_vit_layers)
    
    pretrained_path = args.pretrained_decoders or getattr(config, "pretrained_path", None)
    if pretrained_path:
        load_pretrained_decoders(pretrained_path, model_for_optim)

    # === 8. RESUME ===
    start_epoch, best_loss = 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_for_optim.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch, best_loss = ckpt.get('epoch', 0), ckpt.get('val_loss', float('inf'))

    # === 9. DATA ===
    train_ds = BiomedicalDataset("/group/jug/aman/orochi/data", ["hipsc_3d", "hipsc_2d"], 
                                config.img_size, "train", subset_frac=args.subset)
    val_ds = BiomedicalDataset("/group/jug/aman/orochi/data", ["hipsc_3d", "hipsc_2d"], 
                              config.img_size, "val", subset_frac=args.subset)

    train_sampler = DistributedSampler(train_ds, args.world_size, args.rank) if args.distributed else None
    val_sampler = DistributedSampler(val_ds, args.world_size, args.rank) if args.distributed else None

    train_loader = DataLoader(train_ds, config.batch_size, sampler=train_sampler, 
                             num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, config.batch_size, sampler=val_sampler, 
                           num_workers=config.num_workers, pin_memory=True)

    if is_master: 
        print(f"📊 train={len(train_ds):,} val={len(val_ds):,}")
        if args.subset: print(f"🔹 {args.subset*100:.1f}% subset")

    # === 10. TRAIN ===
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        if args.distributed and train_sampler: train_sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config, scaler, is_master,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # Validate + Checkpoint
        if epoch % config.val_interval == 0:
            val_metrics = validate(model, val_loader, device, epoch, config, is_master)
            
            # BEST → 1 line!
            # BEST
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                wandb_model(model_for_optim, optimizer, scheduler, epoch, best_loss, config, "best", is_master)

            # LATEST/PERIODIC
            wandb_model(model_for_optim, optimizer, scheduler, epoch, val_metrics['loss'], config, "latest", is_master)

            
            # Log metrics
            if is_master and WANDB_AVAILABLE:
                metrics = {**{f"t_{k}": v for k, v in train_metrics.items()},
                          **{f"v_{k}": v for k, v in val_metrics.items()}}
                wandb.log(metrics)
        else:
            if is_master and WANDB_AVAILABLE:
                wandb.log({f"t_{k}": v for k, v in train_metrics.items()})

        if scheduler: scheduler.step(val_metrics['loss'] if epoch % config.val_interval == 0 else 0)

    cleanup_distributed(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Mamba/ViT with wandb")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vit", "mamba"],
        help="Model type to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )

    # Training overrides
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective batch size = batch_size × accumulation_steps × num_gpus)",
    )

    # Logging arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed model information",
    )

    # Transfer learning arguments
    parser.add_argument(
        "--freeze_decoders",
        action="store_true",
        help="Freeze decoder parameters (train encoder only)",
    )
    parser.add_argument(
        "--pretrained_decoders",
        type=str,
        default=None,
        help="Path to checkpoint with pretrained decoders (e.g., trained Mamba model)",
    )
    parser.add_argument(
        "--unfreeze_decoder_layers",
        type=int,
        default=0,
        help="Number of last decoder layers to unfreeze for fine-tuning (0=all frozen)",
    )

    # Pretrained ViT arguments
    parser.add_argument(
        "--pretrained_vit",
        type=str,
        default=None,
        help="HuggingFace ViT model name (e.g., google/vit-base-patch16-224)",
    )
    parser.add_argument(
        "--freeze_vit_early_layers",
        action="store_true",
        help="Freeze early ViT transformer layers (recommended with pretrained weights)",
    )
    parser.add_argument(
        "--num_frozen_vit_layers",
        type=int,
        default=8,
        help="Number of early ViT layers to freeze (default: 8 out of 12)",
    )

    # DDP / distributed arguments
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use DistributedDataParallel (torchrun launch).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="DDP backend (default: nccl).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by torchrun (do not set manually).",
    )
    parser.add_argument(
    "--subset", 
    type=float, 
    default=None,
    help="Fraction of dataset to use (e.g., 0.02 = 2%% for overfitting test)"
    )
    parser.add_argument(
    "--seed", 
    type=int, 
    default=42,
    help="Seed to set"
    )


    args = parser.parse_args()
    main(args)
