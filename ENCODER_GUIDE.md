# Encoder Guide: Modular Encoder Swapping

This guide explains how to use different encoder backends with Orochi, including standard ViT, Mamba, 3DINO-ViT, and HuggingFace models.

## Table of Contents

1. [Overview](#overview)
2. [Available Encoders](#available-encoders)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Two-Phase 3DINO Training](#two-phase-3dino-training)
6. [Encoder Factory API](#encoder-factory-api)
7. [Adding Custom Encoders](#adding-custom-encoders)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Orochi supports multiple encoder backends that can be easily swapped via configuration. The encoder factory (`src/encoder_factory.py`) provides a unified interface for creating different encoders:

- **Mamba Encoder**: State-space models for efficient sequence modeling
- **ViT Encoder**: Standard Vision Transformer with multi-head self-attention
- **3DINO-ViT**: Pretrained biomedical Vision Transformer (384-dim)
- **HuggingFace**: Any ViT model from HuggingFace model hub

### Key Features

✅ **Easy swapping**: Change encoder via single config parameter
✅ **Pretrained weights**: Load pretrained encoders automatically
✅ **Dimension adaptation**: Bottleneck FFN handles dimension mismatches
✅ **Freezing support**: Selectively freeze encoder, decoders, or bottleneck
✅ **Two-phase training**: Train bottleneck first, then finetune encoder

---

## Available Encoders

### 1. Mamba Encoder (`encoder_type: "mamba"`)

**Description**: State-space model encoder with linear-time complexity

**Use case**: Fast training, efficient memory usage

**Configuration**:
```yaml
encoder_type: "mamba"
embed_dim: 128
depths: [4, 4, 4, 4]
```

**Example**:
```bash
./train_local.sh mamba
```

### 2. ViT Encoder (`encoder_type: "vit"`)

**Description**: Standard Vision Transformer with multi-head self-attention

**Use case**: Strong baseline, compatible with ViT pretrained weights

**Configuration**:
```yaml
encoder_type: "vit"
embed_dim: 128
depths: [2, 2, 4, 2]
num_heads: 8
mlp_ratio: 4.0
```

**Example**:
```bash
./train_local.sh vit
```

### 3. 3DINO-ViT Encoder (`encoder_type: "3dino"`)

**Description**: Pretrained biomedical Vision Transformer (384-dim output)

**Use case**: Transfer learning from biomedical pretraining

**Configuration**:
```yaml
encoder_type: "3dino"
pretrained_encoder_path: "/path/to/3dino/checkpoint.pth"
encoder_dim: 384  # 3DINO output dimension
embed_dim: 128    # Mamba decoder dimension
use_bottleneck: true  # Required for dimension adaptation
```

**Example**:
```bash
# Phase 1: Train bottleneck only
./train_local.sh 3dino-phase1

# Phase 2: Finetune encoder + bottleneck
./train_local.sh 3dino-phase2
```

### 4. HuggingFace Encoder (`encoder_type: "huggingface"`)

**Description**: Any ViT model from HuggingFace model hub

**Use case**: Experiment with different pretrained models

**Configuration**:
```yaml
encoder_type: "huggingface"
model_name: "google/vit-base-patch16-224"
```

**Note**: Currently a placeholder implementation - requires full 3D adaptation

---

## Quick Start

### Option 1: Using Training Script

```bash
# Train ViT encoder
./train_local.sh vit

# Train Mamba encoder
./train_local.sh mamba

# Train 3DINO bottleneck (phase 1)
./train_local.sh 3dino-phase1

# Finetune 3DINO encoder (phase 2)
./train_local.sh 3dino-phase2
```

### Option 2: Direct Python

```python
from orochi.configs.model_configs import ViT3DConfig
from vit_model import ViTULight

# Create config
config = ViT3DConfig(
    encoder_type='vit',
    img_size=[32, 224, 224],
    embed_dim=128,
    depths=[2, 2, 4, 2]
)

# Model automatically creates the right encoder
model = ViTULight(config)
```

---

## Configuration

### Basic Configuration Parameters

```yaml
# Encoder type ('mamba', 'vit', '3dino', 'huggingface')
encoder_type: "vit"

# Pretrained encoder weights (optional)
pretrained_encoder_path: "/path/to/encoder/checkpoint.pth"

# Freeze encoder during training
freeze_encoder: false

# Freeze decoders during training
freeze_decoders: true
```

### Bottleneck Configuration (for Dimension Mismatch)

When encoder output dimension differs from decoder input dimension (e.g., 3DINO-ViT 384-dim → Mamba decoders 128-dim):

```yaml
# Enable bottleneck FFN
use_bottleneck: true

# Encoder output dimension
encoder_dim: 384

# Decoder input dimension
embed_dim: 128

# Bottleneck architecture
bottleneck_hidden_ratio: 2.0  # Hidden dim = max(384, 128) * 2.0 = 768
bottleneck_dropout: 0.1
bottleneck_activation: "gelu"
```

### Complete Example: 3DINO-ViT

```yaml
# configs/3dino_bottleneck.yaml

# Encoder configuration
encoder_type: "3dino"
pretrained_encoder_path: "/path/to/3dino/checkpoint.pth"
freeze_encoder: true

# Decoder configuration
freeze_decoders: true
pretrained_path: "/path/to/mamba/decoders.pth"

# Bottleneck configuration
use_bottleneck: true
encoder_dim: 384
embed_dim: 128
bottleneck_hidden_ratio: 2.0
bottleneck_dropout: 0.1
```

---

## Two-Phase 3DINO Training

3DINO-ViT encoders require two-phase training due to dimension mismatch:

### Phase 1: Bottleneck Training

**Goal**: Train bottleneck FFN to adapt 3DINO features (384-dim) to Mamba decoders (128-dim)

**Training strategy**:
- ❄️ **Encoder**: FROZEN (preserve pretrained features)
- ❄️ **Decoders**: FROZEN (preserve task-specific knowledge)
- 🔥 **Bottleneck**: TRAINABLE (learn dimension adaptation)

**Configuration**: `configs/3dino_bottleneck.yaml`

```yaml
freeze_encoder: true
freeze_decoders: true
use_bottleneck: true
encoder_dim: 384
embed_dim: 128
```

**Training**:
```bash
./train_local.sh 3dino-phase1
```

**Duration**: ~50 epochs (faster than full training)

### Phase 2: Encoder Finetuning

**Goal**: Finetune 3DINO encoder for biomedical image processing tasks

**Training strategy**:
- 🔥 **Encoder**: TRAINABLE (adapt to tasks)
- ❄️ **Decoders**: FROZEN (preserve task-specific knowledge)
- 🔥 **Bottleneck**: TRAINABLE (continue adaptation)

**Configuration**: `configs/3dino_finetune.yaml`

```yaml
freeze_encoder: false  # Unfreeze encoder
freeze_decoders: true  # Keep decoders frozen
use_bottleneck: true
encoder_dim: 384
embed_dim: 128

# Resume from phase 1 checkpoint
resume_checkpoint: "./checkpoints/3dino_bottleneck/best_checkpoint.pth"
```

**Training**:
```bash
# Update resume_checkpoint in config first!
./train_local.sh 3dino-phase2
```

**Duration**: ~100 epochs (full finetuning)

**Learning rate**: Lower LR (1e-5) for finetuning pretrained encoder

### Why Two Phases?

1. **Dimension mismatch**: 3DINO (384-dim) ≠ Mamba decoders (128-dim)
2. **Preserve knowledge**: Don't destroy pretrained encoder immediately
3. **Stable training**: Train bottleneck first ensures stable gradients
4. **Better convergence**: Encoder finetuning builds on learned bottleneck

---

## Encoder Factory API

### Creating Encoders Programmatically

```python
from src.encoder_factory import create_encoder

# Create ViT encoder
encoder = create_encoder(config, encoder_type='vit')

# Create Mamba encoder
encoder = create_encoder(config, encoder_type='mamba')

# Create 3DINO with pretrained weights
encoder = create_encoder(
    config,
    encoder_type='3dino',
    pretrained_path='/path/to/3dino.pth',
    freeze=True
)

# Create HuggingFace ViT
encoder = create_encoder(
    config,
    encoder_type='huggingface',
    model_name='google/vit-base-patch16-224'
)
```

### Utility Functions

```python
from src.encoder_factory import get_encoder_output_dim, print_encoder_info

# Get encoder output dimension
output_dim = get_encoder_output_dim(encoder)

# Print encoder information
print_encoder_info(encoder)
# Output:
# ================================================================================
# ENCODER INFO: ViTEncoderHiera
# ================================================================================
# Total parameters: 45.23M
# Trainable parameters: 45.23M (100.0%)
# Embedding dimension: 128
# Feature dimensions: [128, 256, 512, 1024]
# ================================================================================
```

---

## Adding Custom Encoders

### Step 1: Create Encoder Class

```python
# src/my_custom_encoder.py

import torch.nn as nn

class MyCustomEncoder(nn.Module):
    """Custom encoder implementation."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_features = [self.embed_dim * (2 ** i) for i in range(4)]

        # Your encoder implementation here
        ...

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            List of feature maps at different scales
        """
        # Your forward implementation
        features = []
        ...
        return features
```

### Step 2: Add to Encoder Factory

```python
# src/encoder_factory.py

def _create_custom_encoder(config, **kwargs):
    """Create custom encoder."""
    from my_custom_encoder import MyCustomEncoder

    print(f"Creating Custom encoder (embed_dim={config.embed_dim})")
    encoder = MyCustomEncoder(config, **kwargs)

    return encoder

def create_encoder(config, encoder_type='mamba', **kwargs):
    """Factory function to create encoder based on type."""
    encoder_type = encoder_type.lower()

    if encoder_type == 'mamba':
        encoder = _create_mamba_encoder(config, **kwargs)
    elif encoder_type == 'vit':
        encoder = _create_vit_encoder(config, **kwargs)
    elif encoder_type == '3dino':
        encoder = _create_3dino_encoder(config, **kwargs)
    elif encoder_type == 'custom':
        encoder = _create_custom_encoder(config, **kwargs)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    return encoder
```

### Step 3: Update Config Validation

```python
# orochi/configs/model_configs.py

def __post_init__(self):
    # Validate encoder_type
    valid_encoder_types = ['mamba', 'vit', '3dino', 'huggingface', 'custom']
    if self.encoder_type not in valid_encoder_types:
        raise ValueError(f"encoder_type must be one of {valid_encoder_types}")
```

### Step 4: Use Custom Encoder

```yaml
# configs/custom_encoder.yaml
encoder_type: "custom"
```

```bash
python src/finetune_with_wandb.py --config configs/custom_encoder.yaml
```

---

## Troubleshooting

### Error: "Unknown encoder_type"

**Cause**: Invalid encoder_type in config

**Solution**: Use one of: `'mamba'`, `'vit'`, `'3dino'`, `'huggingface'`

```yaml
encoder_type: "vit"  # Must be lowercase
```

### Error: "Dimension mismatch"

**Cause**: Encoder output dimension ≠ decoder input dimension

**Solution**: Enable bottleneck

```yaml
use_bottleneck: true
encoder_dim: 384  # Set to encoder output
embed_dim: 128    # Set to decoder input
```

### Error: "Checkpoint not found"

**Cause**: Invalid pretrained_encoder_path

**Solution**: Check path exists

```bash
ls -la /path/to/3dino/checkpoint.pth
```

```yaml
pretrained_encoder_path: "/correct/path/to/checkpoint.pth"
```

### Warning: "Bottleneck unnecessary"

**Cause**: `use_bottleneck=true` but `encoder_dim == embed_dim`

**Solution**: Disable bottleneck

```yaml
use_bottleneck: false  # No dimension mismatch
```

### Error: "Missing keys in checkpoint"

**Cause**: Checkpoint structure doesn't match encoder

**Solution**: Check checkpoint format

```python
import torch
ckpt = torch.load('/path/to/checkpoint.pth')
print(ckpt.keys())  # Should contain 'model' or 'state_dict'
```

Update encoder factory to handle your checkpoint format.

### Out of Memory (OOM)

**Cause**: Large encoder + large batch size

**Solutions**:

1. Reduce batch size:
```bash
NUM_GPUS=4 BATCH_SIZE=2 ./train_local.sh vit
```

2. Use gradient accumulation:
```bash
GRAD_ACCUM=16 ./train_local.sh vit  # Effective batch size unchanged
```

3. Enable gradient checkpointing:
```yaml
use_checkpoint: true
```

4. Reduce patch size (fewer patches):
```yaml
patch_size: 16  # Instead of 4
```

---

## Best Practices

### 1. Start with Standard ViT

Before using pretrained encoders, verify training works with standard ViT:

```bash
./train_local.sh vit
```

### 2. Two-Phase Training for Pretrained Encoders

Always use two-phase training for pretrained encoders with dimension mismatch:

```bash
# Phase 1: Bottleneck only
./train_local.sh 3dino-phase1

# Phase 2: Encoder finetuning
./train_local.sh 3dino-phase2
```

### 3. Monitor Trainable Parameters

Use `print_trainable_status()` to verify freezing:

```python
model.print_trainable_status()
```

### 4. Use Lower Learning Rate for Finetuning

When finetuning pretrained encoders, use lower LR:

```yaml
# Phase 1 (bottleneck only)
learning_rate: 0.0001

# Phase 2 (encoder finetuning)
learning_rate: 0.00001  # 10x lower
```

### 5. Save Best Checkpoint from Phase 1

Always save and use best checkpoint from phase 1:

```yaml
# configs/3dino_finetune.yaml
resume_checkpoint: "./checkpoints/3dino_bottleneck/best_epoch050_loss0.001234.pth"
```

---

## Summary

**Encoder swapping** in Orochi is simple:

1. **Choose encoder**: Set `encoder_type` in config
2. **Load pretrained weights**: Set `pretrained_encoder_path` (optional)
3. **Handle dimension mismatch**: Enable `use_bottleneck` if needed
4. **Freeze components**: Set `freeze_encoder` and `freeze_decoders`
5. **Train**: Use `train_local.sh` or run directly

**For 3DINO-ViT**:

1. **Phase 1**: Train bottleneck only (encoder + decoders frozen)
2. **Phase 2**: Finetune encoder + bottleneck (decoders frozen)

The modular design ensures **easy experimentation** with different encoder architectures while maintaining **compatibility** with existing decoders and training infrastructure.
