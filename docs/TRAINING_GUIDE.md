# Orochi-ViT Training Guide

Complete guide for training the ViT encoder with pretrained Mamba decoders for biomedical image processing.

## Table of Contents
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Training Configuration](#training-configuration)
- [Running Training](#running-training)
- [Monitoring Progress](#monitoring-progress)
- [Checkpoint Management](#checkpoint-management)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

**Single command to start training:**

```bash
python src/finetune_with_wandb.py --config configs/vit_finetune.yaml --model vit
```

This will:
- Load pretrained decoders from `mamba_fm_3d.pth.tar`
- Train ViT encoder on IR (isotropic restoration) task
- Save checkpoints to `./checkpoints/vit/`
- Log metrics to Weights & Biases

---

## Prerequisites

### 1. Environment Setup

**Required Packages**:
```bash
# Core dependencies
torch>=2.0
torchvision
numpy
scipy
tifffile  # For loading .tiff biomedical images

# Training utilities
wandb      # Experiment tracking
tqdm       # Progress bars
pyyaml     # Config file parsing

# Optional
jupyter    # For data visualization notebooks
matplotlib # For plotting
```

### 2. Data Setup

**Location**: `/group/jug/aman/orochi/data/`

**Structure**:
```
data/
├── hipsc_3d/
│   └── data/*.tiff
├── hipsc_2d/
│   └── data/*.tiff
├── hipct_2d/
│   └── data/*.tiff
└── idr_2d/
    └── data/*.tiff
```

**Total Files**: 19,730 .tiff images across 4 datasets

### 3. Pretrained Checkpoint

**Required**: `pretrained_checkpoints/mamba_fm_3d.pth.tar`

**Verification**:
```bash
ls -lh pretrained_checkpoints/mamba_fm_3d.pth.tar
```

---

## Training Configuration

### Main Config File: `configs/vit_finetune.yaml`

#### Essential Settings

```yaml
# Model Architecture
img_size: [32, 224, 224]      # Input dimensions (matches pretrained)
patch_size: 16                 # ViT patch size (memory efficient)
task: ir                       # Single task: 'ir', 'reg', 'fus', 'sr', or 'multi_task'

# Training
batch_size: 1                  # Fits in 16GB GPU
num_epochs: 100
learning_rate: 0.0001
weight_decay: 0.00001

# Freezing Strategy
freeze_decoders: true          # Train encoder only
freeze_encoder: false

# Pretrained Weights
pretrained_path: "/path/to/mamba_fm_3d.pth.tar"

# Logging
wandb_project: orochi-vit
experiment_name: vit_encoder_finetune
```

#### Task Configuration

**Single Task** (Recommended for faster training):
```yaml
task: ir    # Options: 'ir', 'reg', 'fus', 'sr'
```

| Task | Description | Speed | Best For |
|------|-------------|-------|----------|
| `ir` | Isotropic Restoration | Fast | Denoising |
| `reg` | Registration | Fast | Alignment |
| `fus` | Fusion | Fast | Mask filling |
| `sr` | Super-Resolution | Fast | Upsampling |
| `multi_task` | All 4 tasks | Slow (4x) | General purpose |

**Multi-Task** (For comprehensive training):
```yaml
task: multi_task  # Trains all 4 tasks simultaneously
```

#### Training Modes

**Mode 1: Frozen Decoders** (Current Default)
```yaml
freeze_encoder: false
freeze_decoders: true
```
- **Trainable**: 88% (encoder only)
- **Use Case**: ViT encoder distillation

**Mode 2: Frozen Encoder**
```yaml
freeze_encoder: true
freeze_decoders: false
```
- **Trainable**: 12% (decoders only)
- **Use Case**: Decoder adaptation

**Mode 3: Full Fine-tuning**
```yaml
freeze_encoder: false
freeze_decoders: false
```
- **Trainable**: 100%
- **Use Case**: Full model adaptation

---

## Running Training

### Basic Training

```bash
python src/finetune_with_wandb.py \
    --config configs/vit_finetune.yaml \
    --model vit
```

### With Command-Line Overrides

```bash
python src/finetune_with_wandb.py \
    --config configs/vit_finetune.yaml \
    --model vit \
    --batch_size 2 \
    --learning_rate 0.0002 \
    --num_epochs 50 \
    --experiment_name vit_ir_experiment1
```

### Available Arguments

```bash
# Model
--model vit                    # Model type ('vit' or 'mamba')
--config PATH                  # Config file path

# Training overrides
--batch_size INT
--learning_rate FLOAT
--num_epochs INT
--experiment_name STR
--wandb_project STR

# Pretrained weights
--pretrained_decoders PATH     # Override config pretrained_path

# Freezing
--freeze_decoders              # Freeze decoder parameters
--no_freeze_decoders           # Train decoders

# Logging
--verbose                      # Print model details
--amp                          # Use automatic mixed precision

# Resumption
--resume PATH                  # Resume from checkpoint
```

---

## Monitoring Progress

### Console Output

**Training Start**:
```
Creating Vision Transformer model with 10 blocks
Training task: IR
Loading pretrained decoders from: /path/to/mamba_fm_3d.pth.tar
Freezing decoder parameters (training encoder only)
Trainable parameters: 46,350,208 (88.29%)
Frozen parameters: 6,146,982 (11.71%)
📁 Train: 17757 images from ['hipsc_3d', 'hipsc_2d', 'hipct_2d', 'idr_2d']
📁 Val: 1973 images from ['hipsc_3d', 'hipsc_2d', 'hipct_2d', 'idr_2d']
```

**Training Progress**:
```
Epoch 1/100:  10%|██▎        | 1775/17757 [05:23<48:34, 0.28s/it, loss=0.0234, ir=0.0234]
```

**Key Metrics**:
- `loss`: Total combined loss
- `ir`: Isotropic restoration MSE loss
- `reg`: Registration MSE loss (if multi-task)
- `fus`: Fusion MSE loss (if multi-task)
- `sr`: Super-resolution MSE loss (if multi-task)

### Weights & Biases Dashboard

**Access**: https://wandb.ai/{username}/orochi-vit

**Logged Metrics**:
- `train/loss`: Training loss per batch
- `train/ir_loss`: IR task loss
- `val/loss`: Validation loss per epoch
- `learning_rate`: Current learning rate
- `epoch`: Current epoch number

**Visualizations**:
- Loss curves over time
- Learning rate schedule
- System metrics (GPU utilization, memory)

### Validation

**Frequency**: Every epoch (configurable via `val_interval`)

**Output**:
```
Epoch 1/100 - Train Loss: 0.0234
Epoch 1/100 - Val Loss: 0.0245
```

---

## Checkpoint Management

### Checkpoint Naming

**Format**: `{model}_{task}_{mode}_{type}_epoch-{XXX}.pth`

**Examples**:
```
vit_ir-task_frozen-decoders_best_epoch-042.pth
vit_ir-task_frozen-decoders_periodic_epoch-010.pth
vit_ir-task_frozen-decoders_final_epoch-100.pth
```

### Checkpoint Types

**Best Checkpoint**:
- Saved when validation loss improves
- Location: `checkpoints/vit/*_best_*.pth`
- Use for inference

**Periodic Checkpoints**:
- Saved every N epochs (`save_interval: 5`)
- Location: `checkpoints/vit/*_periodic_epoch-*.pth`
- Use for resuming training

**Final Checkpoint**:
- Saved at end of training
- Location: `checkpoints/vit/*_final_epoch-*.pth`
- Use for archival

### Resuming Training

```bash
python src/finetune_with_wandb.py \
    --config configs/vit_finetune.yaml \
    --model vit \
    --resume checkpoints/vit/vit_ir-task_frozen-decoders_periodic_epoch-050.pth
```

Resumes from:
- Epoch 50
- Saved optimizer state
- Saved learning rate scheduler state
- Best validation loss so far

---

## Troubleshooting

### Out of Memory (OOM)

**Error**:
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 15.80 GiB
```

**Solutions**:

1. **Reduce batch size**:
```yaml
batch_size: 1  # Already minimum
```

2. **Use smaller images** (already done):
```yaml
img_size: [32, 224, 224]  # Current setting
```

3. **Ensure single-task mode**:
```yaml
task: ir  # Not multi_task
```

4. **Enable gradient checkpointing**:
```yaml
use_checkpoint: true
```

### Checkpoint Not Loading

**Error**:
```
Loading pretrained decoders from: ...
KeyError: 'reg_decoder.decoder.up0.conv.weight'
```

**Cause**: Dimension mismatch between ViT and pretrained decoders

**Verify**:
```bash
python -c "
import torch
ckpt = torch.load('pretrained_checkpoints/mamba_fm_3d.pth.tar', map_location='cpu')
print('Expected img_size:', ckpt['config']['img_size'])
print('Expected patch_size:', ckpt['config']['patch_size'])
"
```

**Fix**: Ensure config matches checkpoint:
```yaml
img_size: [32, 224, 224]  # Must match checkpoint
grid_size: [32, 224, 224]  # Must match checkpoint
decoder_head_chan: 16      # Must match checkpoint
```

### Slow Training Speed

**Expected Speed** (single-task, batch_size=1):
- ~0.28s per iteration
- ~4950s (~82 minutes) per epoch for 17,757 samples

**Slow Speed Indicators**:
- >1s per iteration with task='ir'
- Progress bar not moving

**Checks**:

1. **Verify single-task mode**:
```bash
grep "task:" configs/vit_finetune.yaml
# Should show: task: ir
```

2. **Check GPU utilization**:
```bash
nvidia-smi
# GPU-Util should be >80%
```

3. **Verify data loading is not bottleneck**:
```yaml
num_workers: 4  # Parallel data loading
```

### Loss Not Decreasing

**Possible Causes**:

1. **Pretrained decoders not loaded**:
```
# Look for this in console output:
Loading pretrained decoders from: /path/to/mamba_fm_3d.pth.tar
```

2. **Learning rate too high/low**:
```yaml
learning_rate: 0.0001  # Try: 0.00005 or 0.0002
```

3. **Decoders accidentally trainable**:
```yaml
freeze_decoders: true  # Must be true for encoder-only training
```

4. **Wrong task for your data**:
```yaml
task: ir  # Try different tasks: 'reg', 'fus', 'sr'
```

---

## Advanced Usage

### Custom Task Selection

**Config**:
```yaml
task: fus  # Change to any task
```

**Code** (automatic, no changes needed):
```python
# src/vit_model.py automatically handles task selection
if self.task == 'fus':
    # Only runs fusion task
    fus_source_A = self.mask(raw)
    fus_source_B = self.mask(raw)
    x = torch.cat([fus_source_A, fus_source_B], dim=1)
    out_feats = self.encoder(x)
    fused = self.fus_decoder(out_feats)
```

### Multi-Task Training

**Config**:
```yaml
task: multi_task
batch_size: 1  # Keep at 1 due to 4x memory usage
```

**Expected Changes**:
- Training: ~0.28s/it → ~1.11s/it (4x slower)
- Memory: ~4GB → ~16GB (4x more)
- Progress bar shows all task losses:
```
loss=0.0456, reg=0.0123, fus=0.0145, sr=0.0089, ir=0.0099
```

### Gradual Decoder Unfreezing

**Step 1: Train encoder** (current setup)
```yaml
freeze_decoders: true
task: ir
num_epochs: 50
```

**Step 2: Unfreeze last decoder layer**
```python
# In src/finetune_with_wandb.py (future feature)
for param in model.IR_decoder.head.parameters():
    param.requires_grad = True
```

**Step 3: Fine-tune end-to-end**
```yaml
freeze_decoders: false
learning_rate: 0.00001  # Lower LR for stability
num_epochs: 20
```

### Custom Data Loading

**Modify**: `src/finetune_with_wandb.py::BiomedicalDataset`

**Example** - Add custom augmentation:
```python
def __getitem__(self, idx):
    # ... existing loading code ...

    # Custom augmentation
    if self.split == 'train' and random.random() < 0.5:
        image_tensor = torch.flip(image_tensor, dims=[2])  # Random flip

    return {'image': image_tensor, 'idx': idx}
```

---

## Performance Benchmarks

### Training Speed

| Configuration | Time/Epoch | Samples/Second |
|---------------|------------|----------------|
| Single-task (ir), batch=1 | 82 min | 3.6 |
| Multi-task, batch=1 | 328 min | 0.9 |
| Single-task (ir), batch=2 | OOM | - |

*Hardware: NVIDIA GPU with 16GB memory*

### Memory Usage

| Configuration | Peak Memory | Fits in 16GB? |
|---------------|-------------|---------------|
| task=ir, batch=1, patch=16 | 4 GB | ✓ Yes |
| task=multi_task, batch=1, patch=16 | 16 GB | ✓ Barely |
| task=ir, batch=1, patch=4 | OOM | ✗ No |
| task=ir, batch=2, patch=16 | 8 GB | ✓ Yes |

### Convergence

**Typical Loss Trajectory**:
```
Epoch 1:  loss=0.045
Epoch 10: loss=0.018
Epoch 25: loss=0.008
Epoch 50: loss=0.005
Epoch 100: loss=0.003
```

---

## Workflow Summary

```
1. Setup
   ├─ Install dependencies
   ├─ Verify data at /group/jug/aman/orochi/data/
   └─ Check pretrained checkpoint exists

2. Configure
   ├─ Edit configs/vit_finetune.yaml
   │  ├─ Set task (ir, reg, fus, sr, or multi_task)
   │  ├─ Set freeze_decoders=true
   │  └─ Set paths (data_root, pretrained_path)
   └─ Optional: Adjust hyperparameters

3. Train
   ├─ python src/finetune_with_wandb.py --config configs/vit_finetune.yaml --model vit
   ├─ Monitor console output
   └─ Check wandb dashboard

4. Evaluate
   ├─ Review validation loss
   ├─ Check checkpoint files
   └─ Load best checkpoint for inference

5. Iterate
   ├─ Try different tasks
   ├─ Adjust learning rate
   └─ Experiment with unfreezing strategies
```

---

## Next Steps

After successful training:

1. **Evaluate on held-out test set**
2. **Try different tasks** (reg, fus, sr)
3. **Experiment with multi-task** training
4. **Gradually unfreeze decoders** for full fine-tuning
5. **Deploy best model** for downstream applications

---

## Support

**Issues**: https://github.com/AmanKukde/Orochi-Versatile-Biomedical-Image-Processor/issues

**Documentation**:
- [GLOSSARY.md](GLOSSARY.md) - Terminology reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details

---

*Last Updated: 2026-01-01*
