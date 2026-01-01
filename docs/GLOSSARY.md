# Orochi-ViT Nomenclature Glossary

This document defines all terminology, abbreviations, and naming conventions used in the Orochi-ViT project.

## Table of Contents
- [Model Components](#model-components)
- [Training Tasks](#training-tasks)
- [Training Modes](#training-modes)
- [Checkpoint Types](#checkpoint-types)
- [Datasets](#datasets)
- [Loss Functions](#loss-functions)
- [Checkpoint Naming Convention](#checkpoint-naming-convention)

---

## Model Components

### **ViT (Vision Transformer)**
- **Full Name**: Vision Transformer
- **Description**: Transformer-based encoder using self-attention mechanism
- **Key Parameters**:
  - `patch_size`: Size of image patches (16 for memory efficiency)
  - `num_heads`: Number of attention heads (4)
  - `depths`: Number of blocks per stage [2, 2, 4, 2]
  - `embed_dim`: Embedding dimension (128)

### **Mamba**
- **Full Name**: Mamba State Space Model
- **Description**: Original encoder architecture using state space models
- **Note**: Currently not available in this environment (requires mamba_ssm)

### **Encoder**
- **Description**: Feature extraction component that processes input images
- **Current**: ViT-based hierarchical encoder
- **Output**: Multi-scale feature maps at 4 different resolutions

### **Decoder**
- **Description**: Task-specific reconstruction component
- **Source**: Pretrained from Mamba 3D foundation model
- **Purpose**: Converts encoded features back to image space
- **Types**: 4 task-specific decoders (see Training Tasks)

### **SpatialTransformer**
- **Description**: Differentiable spatial transformation for registration
- **Function**: Applies deformation fields to warp images
- **Used in**: Registration task

---

## Training Tasks

The model supports 4 self-supervised pretraining tasks:

### **IR (Isotropic Restoration)**
- **Full Name**: Isotropic Restoration
- **Degradation**: Gaussian noise
- **Objective**: Remove noise and restore clean image
- **Decoder**: `IR_decoder`
- **Current Default**: Yes (for faster single-task training)

### **REG (Registration)**
- **Full Name**: Image Registration
- **Degradation**: Spatial deformation (elastic warping)
- **Objective**: Align deformed image back to original
- **Decoder**: `reg_decoder`
- **Additional Losses**: NCC (normalized cross-correlation), gradient smoothness

### **FUS (Fusion)**
- **Full Name**: Image Fusion
- **Degradation**: Random masking
- **Objective**: Fuse two masked views into complete image
- **Decoder**: `fus_decoder`

### **SR (Super-Resolution)**
- **Full Name**: Super-Resolution
- **Degradation**: Downsampling + noise
- **Objective**: Upsample to original resolution
- **Decoder**: `SR_decoder`

### **MULTI_TASK**
- **Description**: All 4 tasks trained simultaneously
- **Note**: 4x slower than single-task (encoder runs 4 times per batch)

---

## Training Modes

### **frozen-decoders**
- **Description**: Decoders frozen, only encoder trainable
- **Use Case**: ViT encoder distillation from pretrained Mamba decoders
- **Trainable**: ~88% parameters (encoder only)
- **Frozen**: ~12% parameters (decoders)

### **frozen-encoder**
- **Description**: Encoder frozen, only decoders trainable
- **Use Case**: Decoder fine-tuning with fixed encoder
- **Trainable**: ~12% parameters (decoders only)
- **Frozen**: ~88% parameters (encoder)

### **full-finetune**
- **Description**: All parameters trainable
- **Use Case**: Full model adaptation to new data
- **Trainable**: 100% parameters

---

## Checkpoint Types

### **best**
- **Description**: Checkpoint with lowest validation loss
- **Saved When**: Validation loss improves
- **Naming**: `*_best_epoch-XXX.pth`

### **periodic**
- **Description**: Regular checkpoints at fixed intervals
- **Saved When**: Every N epochs (config: `save_interval`)
- **Naming**: `*_periodic_epoch-XXX.pth`

### **final**
- **Description**: Checkpoint at end of training
- **Saved When**: After last epoch
- **Naming**: `*_final_epoch-XXX.pth`

---

## Datasets

### **hipsc_3d**
- **Full Name**: Human Induced Pluripotent Stem Cells (3D)
- **Modality**: 3D microscopy
- **Source**: HuggingFace datasets

### **hipsc_2d**
- **Full Name**: Human Induced Pluripotent Stem Cells (2D)
- **Modality**: 2D microscopy slices

### **hipct_2d**
- **Full Name**: Human Organ Atlas (HiP-CT) Project (2D)
- **Modality**: 2D hierarchical phase-contrast tomography

### **idr_2d**
- **Full Name**: Image Data Resource (2D)
- **Modality**: 2D biomedical imaging

---

## Loss Functions

### **MSE (Mean Squared Error)**
- **Formula**: `(pred - target)²`
- **Used in**: All tasks
- **Purpose**: Pixel-wise reconstruction accuracy

### **NCC (Normalized Cross-Correlation)**
- **Full Name**: Normalized Cross-Correlation from VoxelMorph
- **Used in**: Registration task
- **Purpose**: Measure image similarity (better for deformed images)

### **Grad (Gradient Regularization)**
- **Full Name**: Spatial Gradient Smoothness
- **Used in**: Registration task
- **Purpose**: Penalize unrealistic deformation fields
- **Weight**: 0.01

### **SSIM (Structural Similarity)**
- **Full Name**: Structural Similarity Index (3D)
- **Used in**: Optional validation metric
- **Purpose**: Perceptual similarity measurement

---

## Checkpoint Naming Convention

**Format:**
```
{model}_{task}_{mode}_{type}_epoch-{XXX}.pth
```

**Components:**
- `{model}`: `vit` or `mamba`
- `{task}`: `ir-task`, `reg-task`, `fus-task`, `sr-task`, or `multitask`
- `{mode}`: `frozen-decoders`, `frozen-encoder`, or `full-finetune`
- `{type}`: `best`, `periodic`, or `final`
- `{XXX}`: Zero-padded epoch number (e.g., 042)

**Examples:**
```
vit_ir-task_frozen-decoders_best_epoch-042.pth
  ├─ Model: Vision Transformer
  ├─ Task: Isotropic Restoration
  ├─ Mode: Training encoder only (decoders frozen)
  ├─ Type: Best checkpoint
  └─ Epoch: 42

vit_multitask_full-finetune_periodic_epoch-010.pth
  ├─ Model: Vision Transformer
  ├─ Task: All 4 tasks
  ├─ Mode: Training all parameters
  ├─ Type: Periodic checkpoint
  └─ Epoch: 10

mamba_reg-task_frozen-encoder_final_epoch-100.pth
  ├─ Model: Mamba SSM
  ├─ Task: Registration
  ├─ Mode: Training decoders only
  ├─ Type: Final checkpoint
  └─ Epoch: 100
```

---

## Configuration Files

### **vit_finetune.yaml**
- **Purpose**: Main training configuration
- **Location**: `configs/vit_finetune.yaml`
- **Key Settings**:
  - `task`: Which task(s) to train
  - `freeze_decoders`: Whether to freeze decoder parameters
  - `pretrained_path`: Path to pretrained decoder weights
  - `img_size`: Input image dimensions
  - `patch_size`: ViT patch size

---

## Memory Management

### **OOM (Out of Memory)**
- **Description**: CUDA memory allocation failure
- **Common Causes**:
  - Image size too large
  - Patch size too small (more patches = more attention memory)
  - Batch size too large
  - Multi-task training (4x encoder calls)

### **Attention Memory**
- **Formula**: `O(n²)` where n = number of patches
- **Example**:
  - `patch_size=4`: 16,384 patches → 8.59GB attention matrix
  - `patch_size=16`: 1,024 patches → 33.55MB attention matrix
  - **Reduction**: 256x less memory

---

## Abbreviations Quick Reference

| Abbrev | Full Name | Category |
|--------|-----------|----------|
| ViT | Vision Transformer | Model |
| SSM | State Space Model | Model |
| IR | Isotropic Restoration | Task |
| REG | Registration | Task |
| FUS | Fusion | Task |
| SR | Super-Resolution | Task |
| MSE | Mean Squared Error | Loss |
| NCC | Normalized Cross-Correlation | Loss |
| SSIM | Structural Similarity | Loss |
| OOM | Out of Memory | Error |
| HiPSC | Human Induced Pluripotent Stem Cells | Dataset |
| HiP-CT | Hierarchical Phase-Contrast Tomography | Dataset |
| IDR | Image Data Resource | Dataset |

---

*Last Updated: 2026-01-01*
