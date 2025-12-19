# Orochi Model Dataflow Explanation

## Overview

**Orochi** is a foundation model for biomedical image processing built on the **Mamba architecture** (state-space models, not transformers). It supports both 2D and 3D biomedical images and uses a multi-task self-supervised pretraining approach.

### Key Capabilities
- **Pretraining**: Self-supervised learning on diverse biomedical images
- **Fine-tuning**: Task-specific adaptation for registration, super-resolution, fusion, denoising
- **Architecture**: Hierarchical Mamba encoder with task-specific decoders

---

## Architecture Components

### 1. Encoder: MambaEncoderHeria

**Location**: `src/ours_mamba.py:369-484`

The encoder is a hierarchical multi-scale feature extractor:

```
Input Image (B, C, D, H, W)
    ↓
[PatchEmbed] - 3D Conv projection
    ↓ (B, 128, D/4, H/4, W/4)
[Flatten & Transpose]
    ↓ (B, DHW/64, 128)
[Layer 0] - 4× Mamba2 blocks
    ↓
[PatchMerging] - Reduce spatial dims by 2×
    ↓ (B, DHW/512, 256)
[Layer 1] - 4× Mamba2 blocks
    ↓
[PatchMerging]
    ↓
[Layer 2] - 4× Mamba2 blocks
    ↓
[PatchMerging]
    ↓
[Layer 3] - 4× Mamba2 blocks
    ↓
Multi-scale features: [feat0, feat1, feat2, feat3, feat4]
```

**Configuration (Typical 3D)**:
- Image size: (32, 256, 256) or (160, 192, 224)
- Patch size: 4
- Embed dimension: 128
- Depths: [4, 4, 4, 4] (4 stages × 4 blocks)
- Parameters: ~15M trainable

### 2. Decoders (Task-Specific)

**Location**: `src/ours_mamba.py:485-850`

Four specialized decoders:

1. **reg_decoder**: Registration
   - Output: 3D flow field (B, 3, D, H, W)
   - Used with SpatialTransformer to warp images

2. **fus_decoder**: Fusion
   - Output: Fused image (B, 1, D, H, W)
   - Combines multiple modalities or masked views

3. **SR_decoder**: Super-Resolution
   - Output: High-resolution image (B, 1, D, H, W)
   - Upscales degraded inputs

4. **IR_decoder**: Image Restoration
   - Output: Denoised/restored image (B, 1, D, H, W)
   - Removes noise and artifacts

All decoders use **skip connections** from encoder outputs for multi-scale feature fusion.

---

## Pretraining Dataflow

### Model: `Orochi_Pretrain` or `MambaULight`

**Location**: `temp/experiments/3D/ours_mamba.py:857+` and `src/ours_mamba.py:860-1111`

### Self-Supervised Strategy

Pretraining uses **multi-task learning** with synthetically degraded data:

```
Raw Biomedical Image (B, 1, 32, 256, 256)
    ↓
Generate 4 Degradations in Parallel
    ↓
┌───────────────┬───────────────┬───────────────┬───────────────┐
│ Deformation   │ Masking       │ Downsampling  │ Noise         │
└───────────────┴───────────────┴───────────────┴───────────────┘
```

### Task 1: Registration (Deformation Recovery)

```python
# Step 1: Create deformation
deformed, flow_gt = deform(raw_image)
# - Generate Perlin noise-based deformation field
# - Apply spatial transformation

# Step 2: Encoder + Decoder
input_pair = torch.cat([deformed, raw], dim=1)  # (B, 2, D, H, W)
encoder_features = encoder(input_pair)
predicted_flow = reg_decoder(encoder_features)  # (B, 3, D, H, W)

# Step 3: Register deformed image
registered = SpatialTransformer(deformed, predicted_flow)

# Step 4: Loss
loss_reg = NCC(registered, raw) + λ_grad * Grad(predicted_flow) + MSE(registered, raw)
```

**Purpose**: Learn spatial correspondences and anatomical structure

### Task 2: Fusion (Masking Recovery)

```python
# Step 1: Create two independent masks
masked_A, mask_A = mask(raw, ratio=0.5)
masked_B, mask_B = mask(raw, ratio=0.5)
# - Random patch-based masking

# Step 2: Encoder + Decoder
input_pair = torch.cat([masked_A, masked_B], dim=1)  # (B, 2, D, H, W)
encoder_features = encoder(input_pair)
fused = fus_decoder(encoder_features)  # (B, 1, D, H, W)

# Step 3: Loss
loss_fus = MSE(fused, raw) + λ_ssim * SSIM(fused, raw)
```

**Purpose**: Learn to fill missing information and understand context

### Task 3: Super-Resolution (Upsampling Recovery)

```python
# Step 1: Create degraded low-resolution image
degraded = downsample(raw)
# - Random scale factor (0.25-0.75)
# - Blur (Gaussian filter)
# - Add noise (Gaussian + Poisson)
# - Upsample back to original size (blurry)

# Step 2: Encoder + Decoder
input_pair = torch.cat([degraded, degraded], dim=1)  # (B, 2, D, H, W)
encoder_features = encoder(input_pair)
super_resolved = SR_decoder(encoder_features)  # (B, 1, D, H, W)

# Step 3: Loss
loss_SR = MSE(super_resolved, raw) + λ_ssim * SSIM(super_resolved, raw)
```

**Purpose**: Learn fine-grained texture and detail recovery

### Task 4: Image Restoration (Denoising)

```python
# Step 1: Add multiple noise types
noisy = noise(raw)
# - Gaussian noise
# - Poisson noise
# - Salt-and-pepper noise
# - Clamp to [0, 1]

# Step 2: Encoder + Decoder
input_pair = torch.cat([noisy, noisy], dim=1)  # (B, 2, D, H, W)
encoder_features = encoder(input_pair)
restored = IR_decoder(encoder_features)  # (B, 1, D, H, W)

# Step 3: Loss
loss_IR = MSE(restored, raw) + λ_ssim * SSIM(restored, raw)
```

**Purpose**: Learn robust features invariant to noise

### Combined Pretraining Loss

```python
total_loss = w_reg * loss_reg + w_fus * loss_fus + w_SR * loss_SR + w_IR * loss_IR
```

### Pretraining Pipeline

**Script**: `temp/experiments/3D/train.py`

```python
# 1. Data Loading
dataset = HIPSCDataset(data_path="/root/daigaole/data/HIPSC_1T/")
# - hiPSC biomedical images
# - Random cropping to (32, 256, 256)
# - Distributed sampling for multi-GPU

loader = DataLoader(dataset, batch_size=4, num_workers=8)

# 2. Model Initialization
model = Orochi_Pretrain(config).cuda()
model = DDP(model)  # Distributed training

# 3. Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = WarmupCosineSchedule(
    warmup_steps=total_steps * 0.1,
    t_total=total_steps
)

# 4. Training Loop (50 epochs)
for epoch in range(50):
    for batch in loader:
        raw_image = batch.cuda()  # (B, 1, 32, 256, 256)

        # Forward pass
        logits, aux_losses = model(raw_image)
        # logits = {
        #     'reg': {'registered': ..., 'flow': ...},
        #     'fus': {'fused': ...},
        #     'SR': {'super_resolution': ...},
        #     'IR': {'restored': ...}
        # }

        # Compute losses
        loss = compute_total_loss(logits, raw_image, aux_losses)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    # Save checkpoint
    save_checkpoint(model, f"epoch_{epoch}.pth")
```

**Hardware**: 6× GPUs (DistributedDataParallel)
**Duration**: ~50 epochs on hiPSC dataset

---

## Fine-tuning Dataflow

### Model: `Orochi_Finetune`

**Location**: `temp/experiments/3D/ours_mamba.py:1134+`

Fine-tuning adapts the pretrained encoder to specific downstream tasks.

### Loading Pretrained Weights

```python
# 1. Load pretrained checkpoint
checkpoint = torch.load("pretrained_model.pth")

# 2. Select encoder modules only
selected_state_dict = {k: v for k, v in checkpoint['state_dict'].items()
                       if 'encoder' in k}

# 3. Load into fine-tuning model
model = Orochi_Finetune(config, finetune_mode='reg')
model.load_state_dict(selected_state_dict, strict=False)

# 4. Freeze specific layers (optional)
for name, param in model.named_parameters():
    if 'patch_embed' in name:  # Freeze patch embedding
        param.requires_grad = False
    if 'encoder.layers.0' in name:  # Freeze first layer
        param.requires_grad = False
```

### Example: Registration Fine-tuning

**Script**: `temp/experiments/3D/scripts/Registration/finetune_IXI_reg.py`

```python
# Dataset: IXI Brain MRI
dataset = IXIBrainDataset(
    atlas_dir="/data/IXI/atlas/",
    image_dir="/data/IXI/images/"
)
# Each sample: (moving_image, fixed_image, moving_seg, fixed_seg)

# Model Configuration
config.finetune_mode = 'reg'
model = Orochi_Finetune(config)

# Training Loop
for epoch in range(100):
    for batch in loader:
        moving, fixed, moving_seg, fixed_seg = batch

        # Forward pass
        source = moving.cuda()  # (B, 1, D, H, W)
        target = fixed.cuda()   # (B, 1, D, H, W)

        logits, aux_losses = model(source, target)
        # logits['reg']['flow']: (B, 3, D, H, W)
        # logits['reg']['registered']: (B, 1, D, H, W)

        # Compute registration loss
        registered = logits['reg']['registered']
        flow = logits['reg']['flow']

        loss_ncc = NCC(registered, target)
        loss_grad = Grad(flow)  # Smoothness penalty
        loss_mse = MSE(registered, target)

        loss = loss_ncc * 1.0 + loss_grad * 0.01 + loss_mse * 0.1

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Dataflow**:
```
Moving Image (B, 1, D, H, W)   Fixed Image (B, 1, D, H, W)
            ↓                            ↓
            └────────── Concat ──────────┘
                        ↓
                (B, 2, D, H, W)
                        ↓
        [Pretrained Encoder] ← Loaded weights
                        ↓
              Multi-scale features
                        ↓
            [reg_decoder] ← Fine-tuned
                        ↓
           Flow Field (B, 3, D, H, W)
                        ↓
        [SpatialTransformer]
                        ↓
        Registered Moving (B, 1, D, H, W)
                        ↓
        Compute Loss vs Fixed Image
```

### Example: Super-Resolution Fine-tuning

**Script**: `temp/experiments/2D/scripts/SR/finetune_UniFMIR_SR.py`

```python
# Dataset
dataset = UnifmirSRDataset2D(
    lr_dir="/data/UniFMIR/LR/",
    hr_dir="/data/UniFMIR/HR/"
)

config.finetune_mode = 'SR'
model = Orochi_Finetune(config)

# Training
for batch in loader:
    lr_image, hr_image = batch

    source = lr_image.cuda()  # (B, 1, H, W)
    target = hr_image.cuda()  # (B, 1, 2H, 2W)

    logits, _ = model(source, source)  # Paired input
    sr_output = logits['SR']['super_resolution']

    loss = MSE(sr_output, target) + 0.1 * SSIM(sr_output, target)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Metrics
    psnr = compute_psnr(sr_output, target)
```

### Fine-tuning Modes Summary

| Mode | Dataset Example | Input | Output | Key Loss |
|------|----------------|-------|--------|----------|
| `reg` | IXI Brain MRI | [Moving, Fixed] | Flow + Registered | NCC + Grad |
| `SR` | UniFMIR | [LR, LR] | SR Image | MSE + SSIM |
| `IR` | Noise/Artifact | [Noisy, Noisy] | Restored | MSE + SSIM |
| `fuse` | Multi-modal | [Mod1, Mod2] | Fused | MSE + SSIM |

---

## Inference Dataflow

### Pretrained Model Inference

**Script**: `src/inference_test.py`

```python
# 1. Load model
config = get_config()
model = MambaULight(config).cuda()
checkpoint = torch.load("pretrained_model.pth")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 2. Load test data
test_loader = get_dataloader(config, split='test')
source = next(iter(test_loader)).cuda()  # (1, 1, 32, 256, 256)

# 3. Inference
with torch.no_grad():
    logits, aux_loss = model(source)

# 4. Extract outputs
registered = logits['reg']['registered']  # (1, 1, 32, 256, 256)
fused = logits['fus']['fused']           # (1, 1, 32, 256, 256)
super_resolved = logits['SR']['super_resolution']  # (1, 1, 32, 256, 256)
restored = logits['IR']['restored']       # (1, 1, 32, 256, 256)

# 5. Visualize
visualize_slices(source, registered, fused, super_resolved, restored)
# Saves PNG slices to ./visualizations/
```

**Internal Flow**:
```
Raw Image (1, 1, D, H, W)
    ↓
Apply 4 degradations
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Deformation │ Masking     │ Downsampling│ Noise       │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Encoder →   │ Encoder →   │ Encoder →   │ Encoder →   │
│ reg_decoder │ fus_decoder │ SR_decoder  │ IR_decoder  │
└─────────────┴─────────────┴─────────────┴─────────────┘
    ↓
Collect all outputs in logits dict
```

### Fine-tuned Model Inference

**Example: Registration**

```python
# 1. Load fine-tuned model
model = Orochi_Finetune(config, finetune_mode='reg').cuda()
checkpoint = torch.load("finetuned_registration.pth")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 2. Prepare test images
moving = load_image("moving.nii.gz")  # (1, 1, D, H, W)
fixed = load_image("fixed.nii.gz")    # (1, 1, D, H, W)

# 3. Inference
with torch.no_grad():
    logits, _ = model(moving.cuda(), fixed.cuda())

# 4. Extract registration results
flow = logits['reg']['flow']           # (1, 3, D, H, W)
registered = logits['reg']['registered']  # (1, 1, D, H, W)

# 5. Evaluate
dice_score = compute_dice(registered_seg, fixed_seg)
ncc_score = compute_ncc(registered, fixed)
```

**Dataflow**:
```
Moving Image (1, 1, D, H, W)   Fixed Image (1, 1, D, H, W)
            ↓                            ↓
            └────────── Concat ──────────┘
                        ↓
                (1, 2, D, H, W)
                        ↓
        [Pretrained Encoder] (frozen)
                        ↓
              Multi-scale features
                        ↓
        [Fine-tuned reg_decoder]
                        ↓
           Flow Field (1, 3, D, H, W)
                        ↓
        [SpatialTransformer]
                        ↓
        Registered Moving (1, 1, D, H, W)
```

---

## Key Files and Locations

### Model Architecture
- `src/ours_mamba.py` - Core Mamba encoder/decoders (base implementation)
- `temp/experiments/3D/ours_mamba.py` - 3D pretraining and fine-tuning models
- `temp/experiments/2D/ours_mamba.py` - 2D pretraining and fine-tuning models

### Training Scripts
- `temp/experiments/3D/train.py` - 3D pretraining script
- `temp/experiments/3D/finetune.py` - Generic 3D fine-tuning
- `temp/experiments/3D/scripts/Registration/finetune_IXI_reg.py` - Registration
- `temp/experiments/3D/scripts/SR/` - Super-resolution fine-tuning
- `temp/experiments/3D/scripts/UniFMIR/` - Image restoration fine-tuning

### Data & Losses
- `src/data/datasets.py` - Dataset classes (HIPSC, IXI, etc.)
- `src/losses.py` - Loss functions (NCC, SSIM, Grad, MSE)
- `src/data/trans.py` - Data augmentation and transformations

### Utilities
- `src/utils.py` - Visualization, logging, metrics
- `src/inference_test.py` - Inference and testing scripts

---

## Training Configuration Example

```yaml
# Model Architecture
img_size: [32, 256, 256]        # [Depth, Height, Width]
patch_size: 4
embed_dim: 128                  # Hidden dimension
depths: [4, 4, 4, 4]           # Mamba blocks per layer
in_chans: 2                     # Paired input images

# Pretraining
batch_size: 4
lr: 0.0001
weight_decay: 0.01
max_epoch: 50
warmup_ratio: 0.1
gradient_clip: 1.0

# Loss Weights (Pretraining)
loss_weights:
  registration: 1.0
  fusion: 1.0
  super_resolution: 1.0
  restoration: 1.0

# Loss Components
mse_weight: 1.0
ssim_weight: 0.1
ncc_weight: 1.0
grad_weight: 0.01

# Fine-tuning Specific
checkpoint: "pretrained_model.pth"
finetune_mode: "reg"            # 'reg', 'SR', 'IR', 'fuse'
load_modules:
  load: ['encoder']             # Load encoder weights
  froze: ['patch_embed', 'encoder.layers.0']  # Freeze early layers
  unfroze_from_froze: ['norm']  # Unfreeze norms even if in frozen layers
```

---

## Summary

### Pretraining
- **Goal**: Learn general biomedical image representations
- **Method**: Multi-task self-supervised learning (registration + fusion + SR + restoration)
- **Data**: Synthetically degraded hiPSC images
- **Output**: Pretrained encoder with robust features

### Fine-tuning
- **Goal**: Adapt to specific downstream tasks
- **Method**: Load pretrained encoder, fine-tune with task-specific decoder
- **Data**: Task-specific paired datasets (IXI for registration, UniFMIR for SR, etc.)
- **Output**: Task-optimized model

### Inference
- **Pretrained**: Apply all 4 degradations and recover them (demonstration)
- **Fine-tuned**: Direct task-specific prediction (registration, SR, etc.)
- **Output**: Processed images with evaluation metrics

The key innovation is the **Mamba-based hierarchical encoder** that processes biomedical images efficiently, combined with **multi-task pretraining** that provides strong initialization for diverse downstream tasks.
