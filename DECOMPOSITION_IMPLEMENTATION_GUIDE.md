# Image Decomposition Task - Minimal Implementation Guide

This guide shows how to add a new decomposition/splitting task to the Orochi model. The task takes two different images, adds them together, and trains a decoder to split them back apart.

## Overview

**Task**: Image Decomposition
- **Input**: Two different biomedical images (A and B)
- **Degradation**: Composite = A + B (with normalization)
- **Decoder**: Splits composite → (A', B')
- **Loss**: MSE(A', A) + MSE(B', B)

---

## Step 1: Create the Decomposition Decoder

Add this new decoder class to `src/ours_mamba.py` (after line ~820, alongside other decoders):

```python
class decomp_decoder(nn.Module):
    """
    Decomposition decoder that outputs 2 separate images from a composite.
    """
    def __init__(self, config):
        super(decomp_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        # Output 2 channels (one for each decomposed image)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=2,  # 2 images to decompose
            kernel_size=3,
            padding=1,
        )

    def forward(self, out_feats):
        out = self.decoder(out_feats)
        out = self.head(out)  # (B, 2, D, H, W)
        # Split into two separate images
        image_A = out[:, 0:1, ...]  # (B, 1, D, H, W)
        image_B = out[:, 1:2, ...]  # (B, 1, D, H, W)
        return image_A, image_B
```

**Location**: Insert after `IR_decoder` class (~line 820)

---

## Step 2: Create the Composition Degradation Function

Add this method to the `MambaULight` class (after line ~1110, with other degradation functions):

```python
def compose(self, image_A, image_B):
    """
    Compose two images by adding them together with normalization.

    Args:
        image_A: First image (B, 1, D, H, W)
        image_B: Second image (B, 1, D, H, W)

    Returns:
        composite: Combined image (B, 1, D, H, W)
    """
    # Simple additive composition
    composite = image_A + image_B

    # Normalize to [0, 1] range to prevent overflow
    composite = torch.clamp(composite / 2.0, 0.0, 1.0)

    return composite
```

**Alternative** (weighted composition with optional blending):

```python
def compose(self, image_A, image_B, alpha=None):
    """
    Compose two images with optional random weighting.

    Args:
        image_A: First image (B, 1, D, H, W)
        image_B: Second image (B, 1, D, H, W)
        alpha: Optional mixing weight (default: random 0.3-0.7)

    Returns:
        composite: Combined image (B, 1, D, H, W)
    """
    if alpha is None:
        # Random weight between 0.3 and 0.7 for each sample in batch
        alpha = torch.rand(image_A.shape[0], 1, 1, 1, 1, device=image_A.device) * 0.4 + 0.3

    # Weighted composition: composite = alpha*A + (1-alpha)*B
    composite = alpha * image_A + (1 - alpha) * image_B

    # Ensure values stay in valid range
    composite = torch.clamp(composite, 0.0, 1.0)

    return composite, alpha
```

**Location**: Insert after `noise()` method (~line 1110)

---

## Step 3: Integrate into MambaULight Model

### 3a. Add decoder to `__init__` method

In `src/ours_mamba.py`, modify the `MambaULight.__init__()` method (around line 860-882):

```python
def __init__(self, config):
    super(MambaULight, self).__init__()
    # ... existing code ...

    self.encoder = MambaEncoderHeria(config)
    self.grid_size = config.grid_size
    self.spatial_trans = SpatialTransformer(config.grid_size)
    self.grid_img = self.create_grid_image()
    self.reg_decoder = reg_decoder(config)
    self.fus_decoder = fus_decoder(config)
    self.SR_decoder = SR_decoder(config)
    self.IR_decoder = IR_decoder(config)

    # ADD THIS LINE:
    self.decomp_decoder = decomp_decoder(config)  # Decomposition decoder

    # ... rest of init ...
```

### 3b. Modify `forward()` method

In `src/ours_mamba.py`, modify the `forward()` method (around line 883-959):

```python
def forward(self, raw, raw_B=None):
    """
    Args:
        raw: First input image (B, 1, D, H, W)
        raw_B: Optional second image for decomposition task (B, 1, D, H, W)
               If None, will sample from the batch
    """
    # ... existing degradation tasks (deform, mask, downsample, noise) ...

    # [Keep all existing code for reg, fus, SR, IR tasks]

    # ========== NEW: Decomposition task ==========
    # If raw_B not provided, create pairs from batch
    if raw_B is None:
        # Roll the batch to create pairs: pair each image with the next one
        raw_B = torch.roll(raw, shifts=1, dims=0)

    # Create composition
    composite = self.compose(raw, raw_B)

    # Encoder input: concatenate composite with itself (following existing pattern)
    x = torch.cat([composite, composite], dim=1)
    out_feats = self.encoder(x)

    # Decode to decompose
    decomp_A, decomp_B = self.decomp_decoder(out_feats)

    # Add to logits dict
    logits = {
        'raw': raw.detach().cpu().numpy(),
        # ... existing tasks ...
        'decomp': {
            'composite': composite.detach().cpu().numpy(),
            'decomposed_A': decomp_A.detach().cpu().numpy(),
            'decomposed_B': decomp_B.detach().cpu().numpy(),
            'original_A': raw.detach().cpu().numpy(),
            'original_B': raw_B.detach().cpu().numpy()
        }
    }

    # Add decomposition losses
    aux_loss = {
        'mse': {
            # ... existing losses ...
            'decomp_A': self.mse(decomp_A, raw),
            'decomp_B': self.mse(decomp_B, raw_B)
        },
        # Optional: Add SSIM loss
        # 'ssim': {
        #     'decomp_A': self.ssim(decomp_A, raw),
        #     'decomp_B': self.ssim(decomp_B, raw_B)
        # }
    }

    return logits, aux_loss
```

---

## Step 4: Update Training Script

In your training script (e.g., `temp/experiments/3D/train.py`), modify the loss computation:

```python
# In the training loop (around line 150-180)
for batch_idx, raw in enumerate(train_loader):
    raw = raw.cuda()

    # Forward pass
    logits, aux_losses = model(raw)

    # Compute total loss
    loss_reg = aux_losses['mse']['reg'] + \
               aux_losses['ncc']['reg'] + \
               0.01 * aux_losses['grad']['reg']

    loss_fus = aux_losses['mse']['fus']
    loss_SR = aux_losses['mse']['SR']
    loss_IR = aux_losses['mse']['IR']

    # ADD DECOMPOSITION LOSS:
    loss_decomp = aux_losses['mse']['decomp_A'] + \
                  aux_losses['mse']['decomp_B']

    # Total loss with equal weighting
    total_loss = loss_reg + loss_fus + loss_SR + loss_IR + loss_decomp

    # Or with custom weighting:
    # total_loss = 1.0 * loss_reg + \
    #              1.0 * loss_fus + \
    #              1.0 * loss_SR + \
    #              1.0 * loss_IR + \
    #              1.0 * loss_decomp

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Logging
    if batch_idx % 10 == 0:
        print(f'Batch {batch_idx}, Decomp Loss: {loss_decomp.item():.4f}')
```

---

## Step 5: Visualization (Optional)

Add visualization for decomposition results in your inference script:

```python
# In inference_test.py or similar
def visualize_decomposition(logits, save_dir='./visualizations'):
    """Visualize decomposition results."""
    import matplotlib.pyplot as plt

    composite = logits['decomp']['composite'][0, 0, :, :, :]  # (D, H, W)
    decomp_A = logits['decomp']['decomposed_A'][0, 0, :, :, :]
    decomp_B = logits['decomp']['decomposed_B'][0, 0, :, :, :]
    original_A = logits['decomp']['original_A'][0, 0, :, :, :]
    original_B = logits['decomp']['original_B'][0, 0, :, :, :]

    # Show middle slice
    mid_slice = composite.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: originals and composite
    axes[0, 0].imshow(original_A[mid_slice], cmap='gray')
    axes[0, 0].set_title('Original A')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(original_B[mid_slice], cmap='gray')
    axes[0, 1].set_title('Original B')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(composite[mid_slice], cmap='gray')
    axes[0, 2].set_title('Composite (A+B)')
    axes[0, 2].axis('off')

    # Bottom row: decomposed results and errors
    axes[1, 0].imshow(decomp_A[mid_slice], cmap='gray')
    axes[1, 0].set_title('Decomposed A')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(decomp_B[mid_slice], cmap='gray')
    axes[1, 1].set_title('Decomposed B')
    axes[1, 1].axis('off')

    # Error map
    error_A = np.abs(decomp_A[mid_slice] - original_A[mid_slice])
    error_B = np.abs(decomp_B[mid_slice] - original_B[mid_slice])
    axes[1, 2].imshow(error_A + error_B, cmap='hot')
    axes[1, 2].set_title('Error Map')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/decomposition_results.png', dpi=150, bbox_inches='tight')
    plt.close()
```

---

## Alternative: Using Different Images from Dataset

If you want to load actual different images instead of rolling the batch:

### Modify Dataset Class

In `src/data/datasets.py`, modify your dataset to return pairs:

```python
class HIPSCPairDataset(Dataset):
    """Dataset that returns pairs of different images for decomposition."""

    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.image_files = sorted(glob.glob(f"{data_path}/*.nii.gz"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load first image
        img_A = self.load_image(self.image_files[idx])

        # Load a different random image for image B
        idx_B = random.choice([i for i in range(len(self)) if i != idx])
        img_B = self.load_image(self.image_files[idx_B])

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

    def load_image(self, filepath):
        # Your existing image loading logic
        img = nib.load(filepath).get_fdata()
        img = torch.from_numpy(img).float().unsqueeze(0)  # Add channel dim
        return img
```

### Update Training Loop

```python
# In train.py
for batch_idx, (raw_A, raw_B) in enumerate(train_loader):
    raw_A = raw_A.cuda()
    raw_B = raw_B.cuda()

    # Forward pass with both images
    logits, aux_losses = model(raw_A, raw_B)

    # ... rest of training ...
```

---

## Minimal Testing Script

Quick test to verify the implementation:

```python
# test_decomposition.py
import torch
from src.ours_mamba import MambaULight, decomp_decoder
from config import get_config

# Config
config = get_config()
config.img_size = [32, 256, 256]
config.patch_size = 4
config.embed_dim = 128

# Create model
model = MambaULight(config).cuda()

# Create dummy data
batch_size = 2
raw_A = torch.rand(batch_size, 1, 32, 256, 256).cuda()
raw_B = torch.rand(batch_size, 1, 32, 256, 256).cuda()

# Forward pass
print("Testing decomposition task...")
logits, aux_losses = model(raw_A, raw_B)

# Check outputs
print(f"Composite shape: {logits['decomp']['composite'].shape}")
print(f"Decomposed A shape: {logits['decomp']['decomposed_A'].shape}")
print(f"Decomposed B shape: {logits['decomp']['decomposed_B'].shape}")
print(f"Loss decomp_A: {aux_losses['mse']['decomp_A'].item():.4f}")
print(f"Loss decomp_B: {aux_losses['mse']['decomp_B'].item():.4f}")
print("✓ Decomposition task working!")
```

---

## Summary of Changes

### Files to Modify:

1. **`src/ours_mamba.py`**:
   - Add `decomp_decoder` class (~line 820)
   - Add `compose()` method to `MambaULight` (~line 1110)
   - Add `self.decomp_decoder` to `__init__()` (~line 875)
   - Modify `forward()` to include decomposition task (~line 910)

2. **Training script** (e.g., `temp/experiments/3D/train.py`):
   - Update loss computation to include `loss_decomp`

3. **Optional - Dataset** (`src/data/datasets.py`):
   - Create `HIPSCPairDataset` if loading different images

### Estimated Lines of Code:
- Decoder: ~20 lines
- Degradation: ~15 lines
- Model integration: ~25 lines
- Training update: ~5 lines
- **Total: ~65 lines of code**

---

## Expected Behavior

After training, the model should learn to:
1. **Separate overlapping structures** in composed images
2. **Recover individual modalities** from mixed inputs
3. **Handle varying composition ratios** (if using weighted composition)

This can be useful for:
- Multi-modal image fusion tasks
- Separating overlapping tissue types
- Denoising where noise can be modeled as an additive component
- Source separation in biomedical imaging

---

## Next Steps

1. Implement the decoder and degradation functions
2. Integrate into `MambaULight`
3. Update training script
4. Run a quick test with dummy data
5. Train on your dataset
6. Visualize results to verify decomposition quality

Good luck with your implementation! Let me know if you need help with any specific part.
