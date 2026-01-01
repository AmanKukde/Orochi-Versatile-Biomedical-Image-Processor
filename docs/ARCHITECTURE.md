# Orochi-ViT Architecture Documentation

This document provides a comprehensive technical overview of the Orochi-ViT architecture, design decisions, and implementation details.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [ViT Encoder](#vit-encoder)
- [Pretrained Decoders](#pretrained-decoders)
- [Dimension Alignment](#dimension-alignment)
- [Training Pipeline](#training-pipeline)
- [Memory Optimization](#memory-optimization)
- [Design Decisions](#design-decisions)

---

## Overview

**Goal**: Replace the Mamba encoder with a Vision Transformer (ViT) encoder while reusing pretrained Mamba decoders for biomedical image processing tasks.

**Key Challenge**: ViT and Mamba use different patch sizes for memory/computational reasons:
- **ViT encoder**: `patch_size=16` (for memory efficiency)
- **Mamba decoders**: Pretrained with `patch_size=4`

**Solution**: Encoder outputs are interpolated to match decoder expectations, keeping decoders frozen and intact.

---

## System Architecture

```
Input Image (32x224x224)
         ↓
    [ViT Encoder]
    patch_size=16
         ↓
  Feature Maps (4 scales)
  [2, 14, 14]
  [4, 28, 28]
  [8, 56, 56]
  [16, 112, 112]
         ↓
   [Interpolation]
   ↓
  Aligned Features (4 scales)
  [1, 7, 7]    → matches patch_size=4 expectations
  [2, 14, 14]
  [4, 28, 28]
  [8, 56, 56]
         ↓
  [Task-Specific Decoders]
  (Pretrained, Frozen)
         ↓
  Reconstructed Images
  - IR: Denoised
  - REG: Registered
  - FUS: Fused
  - SR: Super-resolved
```

---

## ViT Encoder

### Architecture: Hierarchical Vision Transformer

**File**: `src/vit_encoder.py` (`ViTEncoderHiera` class)

**Key Components**:

1. **Patch Embedding** (`PatchEmbed3D`)
   - Converts 3D volume to patch tokens
   - Input: `(B, C, 32, 224, 224)`
   - Output: `(B, embed_dim, 2, 14, 14)` where 32/16=2, 224/16=14
   - Uses 3D convolution with stride=16

2. **Hierarchical Stages** (4 stages)
   - Stage 0: depth=2, dims=[2, 14, 14]
   - Stage 1: depth=2, dims=[4, 28, 28] (after upsampling in PatchMerging)
   - Stage 2: depth=4, dims=[8, 56, 56]
   - Stage 3: depth=2, dims=[16, 112, 112]

3. **PatchMerging**
   - Downsamples spatial dimensions by 2x
   - Handles odd dimensions with padding
   - Groups patches into 2x2x2 neighborhoods

4. **Multi-Head Self-Attention**
   - Heads: 4
   - Memory: O(n²) where n = number of patches
   - Window-based attention for efficiency

### Configuration

```yaml
# From configs/vit_finetune.yaml
img_size: [32, 224, 224]
patch_size: 16
embed_dim: 128
depths: [2, 2, 4, 2]
num_heads: 4
mlp_ratio: 4.0
window_size: [5, 6, 7]
```

### Memory Analysis

**Attention Memory Calculation**:
```
Number of patches = (32/16) × (224/16) × (224/16) = 2 × 14 × 14 = 392

Attention matrix per head = 392² × 4 bytes (float32)
                          = 614,656 bytes
                          = 0.59 MB

Total attention (4 heads) = 0.59 × 4 = 2.36 MB per layer
```

**Comparison with patch_size=4**:
```
Patches = (32/4) × (224/4) × (224/4) = 8 × 56 × 56 = 25,088

Attention = 25,088² × 4 bytes = 2.52 GB per layer
```

**Reduction**: 1,069x less memory with patch_size=16!

---

## Pretrained Decoders

### Source

**Checkpoint**: `pretrained_checkpoints/mamba_fm_3d.pth.tar`

**Original Training**:
- Model: Mamba encoder + 4 task decoders
- Image size: (32, 224, 224)
- Patch size: 4
- Tasks: Registration, Fusion, SR, IR
- Embed dim: 128
- Decoder head channels: 16

### Decoder Architecture

Each decoder follows the same structure:

```python
class TaskDecoder:
    def __init__(self):
        self.decoder = ConvDecoderBlock(
            # Hierarchical upsampling
            up0: 4x upsampling (H/8 → H/2)
            up1: 2x upsampling (H/2 → H)
            up2: 2x upsampling (H → 2H)
            # Note: Only up0-up2 used, up3 commented out
        )
        self.head = Conv3d(
            in_channels=decoder_head_chan,
            out_channels=out_chans
        )
```

**Critical Details**:
1. **First upsampling (up0)** expects features at `patch_size=4` scale
2. **Initial feature dims**: `[8, 56, 56]` (for img_size=(32,224,224))
3. **Skip connections**: Expects features at [8,56,56], [4,28,28], [2,14,14], [1,7,7]

### Loading Pretrained Weights

**File**: `src/finetune_with_wandb.py`
**Function**: `load_pretrained_decoders()`

**Process**:
1. Load checkpoint: `torch.load(pretrained_path)`
2. Extract decoder state_dicts
3. Handle `module.` prefix (from DataParallel training)
4. Load into model with `strict=False`
5. Report missing/unexpected keys

**Code**:
```python
pretrained_decoder_path = args.pretrained_decoders or config.pretrained_path
if pretrained_decoder_path is not None:
    print(f"Loading pretrained decoders from: {pretrained_decoder_path}")
    load_pretrained_decoders(pretrained_decoder_path, model)
```

---

## Dimension Alignment

### The Problem

**ViT Encoder** outputs features at scales based on `patch_size=16`:
```
After patch embedding: (32/16, 224/16, 224/16) = (2, 14, 14)
After stage 0: (2, 14, 14)
After stage 1: (4, 28, 28)  # PatchMerging 2x downsamples
After stage 2: (8, 56, 56)
After stage 3: (16, 112, 112)
```

**Pretrained Decoders** expect features at scales based on `patch_size=4`:
```
Expected stage 0: (8, 56, 56)   # 224/4 = 56
Expected stage 1: (4, 28, 28)   # 56/2 = 28
Expected stage 2: (2, 14, 14)   # 28/2 = 14
Expected stage 3: (1, 7, 7)     # 14/2 = 7
```

### The Solution: Interpolation

**File**: `src/vit_encoder.py`
**Location**: `ViTEncoderHiera.forward()` method, lines 571-620

**Implementation**:
```python
# Calculate target dimensions decoders expect
target_patch_size = 4
base_T = img_size[0] // target_patch_size  # 32/4 = 8
base_H = img_size[1] // target_patch_size  # 224/4 = 56
base_W = img_size[2] // target_patch_size  # 224/4 = 56

# Expected dimensions at each stage after 2x downsampling
target_dims = []
for i in range(num_layers):
    scale = 2 ** i
    target_dims.append((
        base_T // scale,   # [8, 4, 2, 1]
        base_H // scale,   # [56, 28, 14, 7]
        base_W // scale    # [56, 28, 14, 7]
    ))

# After processing each stage
if out.shape[2:] != (target_t, target_h, target_w):
    out = F.interpolate(
        out,
        size=(target_t, target_h, target_w),
        mode='trilinear',
        align_corners=False
    )
```

**Result**: ViT features are resized to exactly match decoder expectations!

---

## Training Pipeline

### Single-Task Mode (Current Default)

**File**: `src/vit_model.py`
**Class**: `ViTULight`

**Forward Pass (task='ir')**:
```python
def forward(self, raw):
    # Only run IR task
    IR_source = self.noise(raw)           # Add Gaussian noise
    x = torch.cat([IR_source, IR_source], dim=1)
    out_feats = self.encoder(x)           # ViT encoding (1 pass)
    IRed = self.IR_decoder(out_feats)     # Reconstruction

    # Compute loss
    loss = MSE(IRed, raw)
    return logits, aux_loss
```

**Advantages**:
- 4x faster (encoder runs once vs 4 times)
- Lower memory usage
- Clearer training signal per task

### Multi-Task Mode

**Forward Pass (task='multi_task')**:
```python
def forward(self, raw):
    tasks = ['reg', 'fus', 'sr', 'ir']

    for task in tasks:
        # Apply task-specific degradation
        degraded = apply_degradation(raw, task)

        # Encode (4 separate passes!)
        out_feats = self.encoder(torch.cat([degraded, raw], dim=1))

        # Decode with task-specific decoder
        restored = task_decoder(out_feats)

        # Compute task-specific loss
        losses[task] = MSE(restored, raw)

    total_loss = sum(losses.values())
    return logits, aux_loss
```

**Disadvantages**:
- 4x slower training
- 4x more GPU memory
- More complex loss landscape

---

## Memory Optimization

### Strategy 1: Increase Patch Size

**Change**: `patch_size: 4 → 16`

**Impact**:
- Patches: 25,088 → 392 (64x reduction)
- Attention memory: 2.52 GB → 2.36 MB (1,069x reduction)
- Total memory: ~32 GB → ~500 MB for single batch

### Strategy 2: Image Downsampling

**Implementation**: `src/finetune_with_wandb.py::BiomedicalDataset._resize()`

**Process**:
```python
def _resize(self, image_tensor):
    """Resize to target size (upsample or downsample)."""
    d, h, w = image_tensor.shape[1:]
    td, th, tw = self.img_size  # (32, 224, 224)

    if (d, h, w) != (td, th, tw):
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(td, th, tw),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

    return image_tensor
```

**Original Data**: Many images are 64x256x256
**Processed Data**: All resized to 32x224x224 (matches pretrained checkpoint)

**Memory Savings**:
- Volume reduction: (64×256×256)/(32×224×224) = 2.62x
- Combined with patch_size: ~2,800x total reduction!

### Strategy 3: Single-Task Training

**Change**: `task: multi_task → task: ir`

**Impact**:
- Encoder calls per batch: 4 → 1 (4x reduction)
- Training speed: 1.11s/it → ~0.28s/it
- GPU memory: ~16 GB → ~4 GB

### Strategy 4: Frozen Decoders

**Change**: `freeze_decoders: true`

**Impact**:
- No gradient computation for decoder backward pass
- Decoder parameters don't need gradient storage
- Faster backward pass through frozen components

---

## Design Decisions

### Why ViT Instead of Mamba?

| Aspect | ViT | Mamba |
|--------|-----|-------|
| **Availability** | Standard PyTorch | Requires mamba_ssm package |
| **Hardware** | Works on any GPU | May need specific CUDA versions |
| **Interpretability** | Attention maps visualizable | Black-box SSM |
| **Memory** | O(n²) but manageable with large patches | O(n) but requires special kernels |
| **Pretrained Weights** | Abundant (HuggingFace) | Limited availability |

**Decision**: Use ViT for better accessibility and ecosystem support.

### Why Freeze Decoders?

**Rationale**:
1. Decoders already trained on biomedical data (pretrained Mamba checkpoint)
2. Training encoder only = faster experimentation
3. Focuses learning on encoder-decoder alignment
4. Reduces risk of catastrophic forgetting

**Future Work**: Gradual unfreezing or decoder fine-tuning after encoder convergence.

### Why Single-Task by Default?

**Rationale**:
1. **Speed**: 4x faster iteration during development
2. **Memory**: Fits on smaller GPUs (16GB → 4GB)
3. **Debugging**: Easier to identify task-specific issues
4. **Flexibility**: Can switch tasks in config without code changes

**Multi-task benefits**: Preserved for future use when needed for cross-task learning.

### Why Interpolate in Encoder?

**Alternative**: Modify decoder to accept different input dimensions

**Decision**: Keep decoders frozen and intact
- Preserves pretrained knowledge exactly
- Aligns with "frozen-decoders" training paradigm
- Encoder-side interpolation is differentiable and lightweight

---

## File Structure

```
Orochi-Versatile-Biomedical-Image-Processor/
├── src/
│   ├── vit_encoder.py          # ViT encoder with interpolation
│   ├── vit_model.py             # ViTULight model (encoder + decoders)
│   ├── ours_mamba.py            # Decoder implementations
│   ├── finetune_with_wandb.py   # Training script
│   └── losses.py                # Loss functions
├── configs/
│   └── vit_finetune.yaml        # Training configuration
├── pretrained_checkpoints/
│   └── mamba_fm_3d.pth.tar      # Pretrained decoder weights
├── docs/
│   ├── GLOSSARY.md              # Nomenclature definitions
│   ├── ARCHITECTURE.md          # This file
│   └── TRAINING_GUIDE.md        # Training instructions
└── notebooks/
    └── test_dataloader_viz.ipynb  # Data loading visualization
```

---

## Key Parameters Reference

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `img_size` | [32, 224, 224] | Matches pretrained checkpoint |
| `patch_size` | 16 | ViT encoder (memory efficiency) |
| `decoder.patch_size` | 4 | Decoder expectations (pretrained) |
| `embed_dim` | 128 | Feature dimension |
| `decoder_head_chan` | 16 | Decoder output channels |
| `num_heads` | 4 | Attention heads |
| `depths` | [2, 2, 4, 2] | Blocks per stage |
| `batch_size` | 1 | Fits in 16GB GPU |
| `task` | 'ir' | Single task (fast training) |
| `freeze_decoders` | true | Train encoder only |

---

*Last Updated: 2026-01-01*
