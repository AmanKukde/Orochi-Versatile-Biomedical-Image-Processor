# HuggingFace Model Loading Guide

Complete guide for loading pretrained models from HuggingFace Hub and TIMM for biomedical image processing.

## Table of Contents

1. [Overview](#overview)
2. [Supported Models](#supported-models)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Available Models](#available-models)
6. [2D→3D Conversion](#2d3d-conversion)
7. [Dimension Adaptation](#dimension-adaptation)
8. [Training Workflow](#training-workflow)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Orochi now supports loading pretrained Vision Transformers from:
- **HuggingFace Hub**: Google ViT, Facebook DeiT, Microsoft Swin, and more
- **TIMM (PyTorch Image Models)**: 1000+ pretrained models

### Key Features

✅ **Automatic loading** from HuggingFace Hub
✅ **2D→3D weight inflation** for pretrained 2D models
✅ **Bottleneck adaptation** for dimension mismatch
✅ **Hierarchical features** compatible with decoders
✅ **Both HuggingFace and TIMM** supported

---

## Supported Models

### HuggingFace Models

| Model | Hidden Size | Parameters | Use Case |
|-------|-------------|------------|----------|
| `google/vit-base-patch16-224` | 768 | 86M | General purpose, recommended |
| `google/vit-large-patch16-224` | 1024 | 304M | Higher capacity |
| `facebook/deit-base-patch16-224` | 768 | 86M | Distilled ViT |
| `facebook/deit-tiny-patch16-224` | 192 | 5M | Lightweight |
| `microsoft/swin-tiny-patch4-window7-224` | 96-768 | 28M | Hierarchical |
| `microsoft/swin-base-patch4-window7-224` | 128-1024 | 88M | Hierarchical |

### TIMM Models

| Model | Hidden Size | Parameters | Use Case |
|-------|-------------|------------|----------|
| `vit_base_patch16_224` | 768 | 86M | Standard ViT |
| `vit_small_patch16_224` | 384 | 22M | Smaller ViT |
| `deit_base_patch16_224` | 768 | 86M | Distilled ViT |
| `swin_tiny_patch4_window7_224` | 96-768 | 28M | Swin Transformer |

---

## Quick Start

### Option 1: HuggingFace Hub (Recommended)

```bash
# Edit configs/huggingface_vit.yaml
hf_model_name: "google/vit-base-patch16-224"
encoder_type: "huggingface"
use_bottleneck: true
encoder_dim: 768  # ViT-Base hidden size

# Submit job
sbatch train_huggingface.sh
```

### Option 2: TIMM

```bash
# Edit config
hf_model_name: "vit_base_patch16_224"
encoder_type: "huggingface"
use_timm: true  # Use TIMM instead of HuggingFace
use_bottleneck: true
encoder_dim: 768

# Train
sbatch train_huggingface.sh
```

### Option 3: Interactive Python

```python
from orochi.configs.model_configs import ViT3DConfig
from vit_model import ViTULight

# Create config
config = ViT3DConfig(
    encoder_type='huggingface',
    hf_model_name='google/vit-base-patch16-224',
    use_bottleneck=True,
    encoder_dim=768,  # HF ViT hidden size
    embed_dim=128     # Decoder dimension
)

# Model automatically loads from HuggingFace
model = ViTULight(config)
```

---

## Configuration

### Basic Configuration

```yaml
# Encoder type
encoder_type: "huggingface"

# HuggingFace model name
hf_model_name: "google/vit-base-patch16-224"

# Use TIMM instead (optional)
use_timm: false

# Freeze settings
freeze_encoder: false
freeze_decoders: true

# Bottleneck (required for dimension mismatch)
use_bottleneck: true
encoder_dim: 768  # Model hidden size
embed_dim: 128    # Decoder dimension
```

### Complete Example

```yaml
# configs/huggingface_vit.yaml

# Model selection
encoder_type: "huggingface"
hf_model_name: "google/vit-base-patch16-224"
use_timm: false

# Dimension configuration
encoder_dim: 768  # ViT-Base output
embed_dim: 128    # Mamba decoder input

# Bottleneck adaptation
use_bottleneck: true
bottleneck_hidden_ratio: 2.0  # Hidden = max(768, 128) * 2 = 1536
bottleneck_dropout: 0.1
bottleneck_activation: "gelu"

# Training
freeze_encoder: false
freeze_decoders: true
learning_rate: 0.00001  # Lower for finetuning

# Task
task: multi_task
```

---

## Available Models

### Google ViT Models

**ViT-Base (Recommended)**
```yaml
hf_model_name: "google/vit-base-patch16-224"
encoder_dim: 768
```
- **Size**: 86M parameters
- **Use case**: General purpose, good balance
- **Pretrained**: ImageNet-21k

**ViT-Large**
```yaml
hf_model_name: "google/vit-large-patch16-224"
encoder_dim: 1024
```
- **Size**: 304M parameters
- **Use case**: Higher capacity, more data
- **Pretrained**: ImageNet-21k

### Facebook DeiT Models

**DeiT-Base**
```yaml
hf_model_name: "facebook/deit-base-patch16-224"
encoder_dim: 768
```
- **Size**: 86M parameters
- **Use case**: Distilled ViT, efficient
- **Pretrained**: ImageNet-1k with distillation

**DeiT-Tiny**
```yaml
hf_model_name: "facebook/deit-tiny-patch16-224"
encoder_dim: 192
```
- **Size**: 5M parameters
- **Use case**: Lightweight, fast
- **Pretrained**: ImageNet-1k

### Microsoft Swin Transformer

**Swin-Tiny**
```yaml
hf_model_name: "microsoft/swin-tiny-patch4-window7-224"
encoder_dim: 768  # Final hidden size
```
- **Size**: 28M parameters
- **Use case**: Hierarchical attention
- **Pretrained**: ImageNet-1k

**Swin-Base**
```yaml
hf_model_name: "microsoft/swin-base-patch4-window7-224"
encoder_dim: 1024  # Final hidden size
```
- **Size**: 88M parameters
- **Use case**: Strong baseline
- **Pretrained**: ImageNet-1k

---

## 2D→3D Conversion

HuggingFace models are typically pretrained on 2D images. Orochi automatically handles 2D→3D conversion.

### How It Works

1. **3D Patch Embedding**: Creates 3D convolutional patch embedding
2. **Weight Inflation**: Inflates 2D pretrained weights to 3D
   ```python
   weight_2d: (out, in, H, W)
   weight_3d: (out, in, D, H, W) = weight_2d.unsqueeze(2).repeat(1, 1, D, 1, 1) / D
   ```
3. **Slice Processing**: Processes each depth slice through 2D model
4. **3D Reconstruction**: Stacks slices back into 3D volume

### Example

```
Input: (B, 2, 32, 224, 224)  # 3D volume

Step 1: 3D Patch Embedding
→ (B, 768, 2, 14, 14)  # Patched 3D

Step 2: Slice-wise Processing
→ Process each of 2 depth slices through HF model

Step 3: 3D Reconstruction
→ (B, 768, 2, 14, 14)  # 3D features

Step 4: Bottleneck Projection
→ (B, 128, 2, 14, 14)  # Decoder-compatible
```

---

## Dimension Adaptation

### Why Bottleneck?

HuggingFace models have different output dimensions than Mamba decoders:

| Model | Hidden Size | Decoder Input | Bottleneck Needed? |
|-------|-------------|---------------|-------------------|
| ViT-Base | 768 | 128 | ✅ Yes |
| ViT-Large | 1024 | 128 | ✅ Yes |
| DeiT-Base | 768 | 128 | ✅ Yes |
| DeiT-Tiny | 192 | 128 | ✅ Yes |
| Swin-Tiny | 768 | 128 | ✅ Yes |

### Bottleneck Configuration

```yaml
use_bottleneck: true

# Encoder output dimension (from HuggingFace model)
encoder_dim: 768

# Decoder input dimension (Mamba decoders)
embed_dim: 128

# Bottleneck architecture
bottleneck_hidden_ratio: 2.0  # Hidden = max(768, 128) * 2 = 1536
bottleneck_dropout: 0.1
bottleneck_activation: "gelu"
```

### Bottleneck Architecture

```
Encoder Features (768-dim)
     ↓
LayerNorm
     ↓
Linear(768 → 1536) + GELU + Dropout
     ↓
Linear(1536 → 128) + Dropout
     ↓
LayerNorm
     ↓
Decoder Features (128-dim)
```

---

## Training Workflow

### Single-Phase Training

For most HuggingFace models, single-phase training works well:

```bash
# Configure model
vim configs/huggingface_vit.yaml
# Set hf_model_name, encoder_dim, use_bottleneck=true

# Train
sbatch train_huggingface.sh
```

**Settings**:
- Encoder: TRAINABLE
- Decoders: FROZEN
- Bottleneck: TRAINABLE
- Learning rate: Low (1e-5)

### Two-Phase Training (Optional)

For better stability, use two-phase training like 3DINO:

**Phase 1: Bottleneck Only**
```yaml
freeze_encoder: true
freeze_decoders: true
use_bottleneck: true
learning_rate: 0.0001
num_epochs: 50
```

**Phase 2: Encoder Finetuning**
```yaml
freeze_encoder: false  # Unfreeze
freeze_decoders: true
use_bottleneck: true
learning_rate: 0.00001  # Lower
num_epochs: 100
```

---

## Troubleshooting

### Error: "transformers library required"

**Solution**: Install transformers
```bash
pip install transformers
```

### Error: "timm library required"

**Solution**: Install timm
```bash
pip install timm
```

### Error: "Model not found"

**Cause**: Invalid HuggingFace model name

**Solution**: Check model exists on HuggingFace Hub
```bash
# Visit https://huggingface.co/models
# Search for "vision transformer" or specific model
```

### Error: "Dimension mismatch"

**Cause**: Bottleneck not enabled or wrong encoder_dim

**Solution**: Enable bottleneck and set correct encoder_dim
```yaml
use_bottleneck: true
encoder_dim: 768  # Must match model hidden size
```

### Warning: "Could not inflate 2D weights"

**Cause**: Model structure doesn't match expected ViT structure

**Effect**: 3D patch embedding uses random initialization

**Solution**: This is usually fine. Model will learn during training.

### Out of Memory (OOM)

**Cause**: Large model + large batch size

**Solutions**:

1. Reduce batch size:
```bash
--batch_size 2
```

2. Increase gradient accumulation:
```bash
--gradient_accumulation_steps 16
```

3. Use smaller model:
```yaml
hf_model_name: "facebook/deit-tiny-patch16-224"
encoder_dim: 192
```

4. Enable gradient checkpointing:
```yaml
use_checkpoint: true
```

### Slow Training

**Cause**: Depth-wise slice processing overhead

**Solutions**:

1. Reduce input depth:
```yaml
img_size: [16, 224, 224]  # Reduce from 32 to 16
```

2. Use TIMM with features_only:
```yaml
use_timm: true  # Usually faster than HuggingFace
```

---

## Best Practices

### 1. Start with ViT-Base

Google ViT-Base is the recommended starting point:

```yaml
hf_model_name: "google/vit-base-patch16-224"
encoder_dim: 768
use_bottleneck: true
```

### 2. Use Lower Learning Rate

Pretrained models need lower learning rates:

```yaml
learning_rate: 0.00001  # 10x lower than from-scratch
```

### 3. Enable Bottleneck

Always enable bottleneck for dimension adaptation:

```yaml
use_bottleneck: true
encoder_dim: 768  # Model hidden size
embed_dim: 128    # Decoder dimension
```

### 4. Monitor GPU Memory

Check memory usage:

```bash
watch -n 1 nvidia-smi
```

### 5. Verify Model Loading

Check logs for successful loading:

```
📦 Loading from HuggingFace Hub: google/vit-base-patch16-224
✓ Loaded HuggingFace model
  Model type: vit
  Hidden size: 768
  HuggingFace wrapper created:
    Input: 3D volumes (B, 2, D, H, W)
    Embed dim: 768
    Output features: [128, 256, 512, 1024]
  ✓ Inflated 2D→3D patch embedding weights
```

---

## Model Selection Guide

### For General Use

**Recommended**: `google/vit-base-patch16-224`
- Well-balanced size and performance
- Strong ImageNet-21k pretraining
- 768-dim output

### For Limited Memory

**Recommended**: `facebook/deit-tiny-patch16-224`
- Only 5M parameters
- Fast training
- 192-dim output

### For Hierarchical Features

**Recommended**: `microsoft/swin-tiny-patch4-window7-224`
- Hierarchical attention
- Efficient architecture
- 768-dim output

### For Maximum Performance

**Recommended**: `google/vit-large-patch16-224`
- 304M parameters
- Highest capacity
- 1024-dim output
- Requires more data and compute

---

## Examples

### Example 1: ViT-Base from HuggingFace

```yaml
# configs/huggingface_vit.yaml

encoder_type: "huggingface"
hf_model_name: "google/vit-base-patch16-224"
use_bottleneck: true
encoder_dim: 768
embed_dim: 128
freeze_encoder: false
freeze_decoders: true
learning_rate: 0.00001
```

```bash
sbatch train_huggingface.sh
```

### Example 2: DeiT-Tiny (Lightweight)

```yaml
encoder_type: "huggingface"
hf_model_name: "facebook/deit-tiny-patch16-224"
use_bottleneck: true
encoder_dim: 192
embed_dim: 128
batch_size: 4  # Can use larger batch
```

### Example 3: TIMM Model

```yaml
encoder_type: "huggingface"
hf_model_name: "vit_base_patch16_224"
use_timm: true  # Use TIMM
use_bottleneck: true
encoder_dim: 768
```

### Example 4: Two-Phase Training

**Phase 1**: Train bottleneck
```yaml
freeze_encoder: true
freeze_decoders: true
use_bottleneck: true
num_epochs: 50
```

**Phase 2**: Finetune encoder
```yaml
freeze_encoder: false
freeze_decoders: true
use_bottleneck: true
num_epochs: 100
resume_checkpoint: "./checkpoints/phase1/best.pth"
```

---

## Summary

**HuggingFace model loading** in Orochi:

1. **Choose model**: Select from HuggingFace Hub or TIMM
2. **Set dimensions**: Configure `encoder_dim` to match model
3. **Enable bottleneck**: Always use for dimension adaptation
4. **Configure freezing**: Freeze decoders, train encoder + bottleneck
5. **Set learning rate**: Use lower LR for pretrained models
6. **Train**: Submit job with `sbatch train_huggingface.sh`

**Key configuration**:
```yaml
encoder_type: "huggingface"
hf_model_name: "google/vit-base-patch16-224"
use_bottleneck: true
encoder_dim: 768  # Model hidden size
embed_dim: 128    # Decoder dimension
```

The modular design ensures **easy experimentation** with different pretrained models while maintaining **compatibility** with existing decoders and training infrastructure.
