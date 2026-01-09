# Loss Functions Quick Reference Guide

## Import Statement
```python
from orochi.losses import get_loss_function, SSIM2D, Grad3d, DiceLoss  # etc.
```

## Factory Function Quick Reference

### Syntax
```python
loss_fn = get_loss_function(name, **kwargs)
```

### All Available Loss Names

#### Structural Similarity
```python
loss = get_loss_function('ssim2d', window_size=11, size_average=True)
loss = get_loss_function('ssim3d', window_size=11, size_average=True)
loss = get_loss_function('s3im', B=1024, M=10, window_size=4)
loss = get_loss_function('s3im_stitched', patches_to_stitch=16, M=10)
```

#### Gradient Regularization
```python
loss = get_loss_function('grad2d', penalty='l1', loss_mult=0.1)
loss = get_loss_function('grad3d', penalty='l2', loss_mult=0.1)
loss = get_loss_function('grad3d_itv')  # Isotropic TV
loss = get_loss_function('displacement_reg', energy_type='bending')
# energy_type: 'gradient-l1', 'gradient-l2', 'bending'
```

#### Information Theory
```python
loss = get_loss_function('ncc', win=[9, 9, 9])
loss = get_loss_function('mi', sigma_ratio=1, num_bin=32)
loss = get_loss_function('local_mi', patch_size=5)
loss = get_loss_function('mind')
```

#### Segmentation
```python
loss = get_loss_function('dice', smooth=1e-6)  # Binary
loss = get_loss_function('dice', num_class=5)  # Multi-class
```

#### Pixel-wise & Perceptual
```python
loss = get_loss_function('mse')  # PyTorch MSELoss
loss = get_loss_function('l1')   # PyTorch L1Loss
loss = get_loss_function('log_mse', eps=1e-6)
loss = get_loss_function('lpips', net='vgg')  # Requires lpips package
loss = get_loss_function('ri_psnr', eps=1e-8)
```

#### Image Fusion
```python
loss = get_loss_function('max_grad_3d', loss_weight=1.0, isSignGrad=True)
loss = get_loss_function('max_pixel_3d', loss_weight=1.0)
loss = get_loss_function('pixel_3d', loss_weight=1.0)
```

## Common Use Cases

### 2D Image Reconstruction
```python
from orochi.losses import get_loss_function

ssim_loss = get_loss_function('ssim2d')
l1_loss = get_loss_function('l1')

def combined_loss(pred, target):
    return ssim_loss(pred, target) + 0.1 * l1_loss(pred, target)
```

### 3D Medical Image Registration
```python
ncc_loss = get_loss_function('ncc', win=[9, 9, 9])
smooth_loss = get_loss_function('grad3d', penalty='l2', loss_mult=0.01)

def registration_loss(warped_image, target, displacement_field):
    similarity = ncc_loss(target, warped_image)
    regularization = smooth_loss(displacement_field, None)
    return similarity + regularization
```

### Multi-class 3D Segmentation
```python
dice_loss = get_loss_function('dice', num_class=7)

# Input: (N, 7, D, H, W) - probabilities
# Target: (N, 1, D, H, W) - class labels [0-6]
loss = dice_loss(predictions, targets)
```

### Image Quality Enhancement
```python
ssim_loss = get_loss_function('ssim2d')
lpips_loss = get_loss_function('lpips', net='vgg')

def perceptual_loss(enhanced, ground_truth):
    # Combine structural similarity with perceptual distance
    return ssim_loss(enhanced, ground_truth) + 0.5 * lpips_loss(enhanced, ground_truth)
```

### Multi-modal Image Fusion (CT + MRI)
```python
grad_loss = get_loss_function('max_grad_3d', loss_weight=1.0)
pixel_loss = get_loss_function('pixel_3d', loss_weight=0.5)

def fusion_loss(fused, ct, mri):
    # Preserve gradients and average intensities
    return grad_loss(fused, ct, mri) + pixel_loss(fused, ct, mri)
```

## Shape Requirements

### 2D Losses
| Loss | Input Shape | Target Shape | Notes |
|------|-------------|--------------|-------|
| SSIM2D | (N, C, H, W) | (N, C, H, W) | Same as input |
| Grad2d | (N, C, H, W) | Not used | Regularization only |
| S3IM | (N, C, H, W) | (N, C, H, W) | H*W must be >= B |

### 3D Losses
| Loss | Input Shape | Target Shape | Notes |
|------|-------------|--------------|-------|
| SSIM3D | (N, C, D, H, W) | (N, C, D, H, W) | Same as input |
| Grad3d | (N, C, D, H, W) | Not used | Regularization only |
| NCC | (N, C, D, H, W) | (N, C, D, H, W) | Also works with 2D |

### Segmentation
| Loss | Input Shape | Target Shape | Notes |
|------|-------------|--------------|-------|
| DiceLoss (binary) | (N, C, *spatial) | (N, C, *spatial) | Continuous values |
| DiceLoss (multi-class) | (N, K, D, H, W) | (N, 1, D, H, W) | K=num_classes, target=class indices |

## Mathematical Formulas

### SSIM
```
SSIM(x,y) = (2μ_x·μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))
where C1 = (0.01)², C2 = (0.03)²
```

### Gradient Loss (L1)
```
L = (1/d) Σ_i mean(|∂y/∂x_i|)  where d is dimensionality (2 or 3)
```

### Dice Coefficient
```
Dice = 2|X ∩ Y| / (|X| + |Y|)
Loss = 1 - Dice
```

### Mutual Information
```
MI(X,Y) = Σ_x Σ_y p(x,y) log(p(x,y) / (p(x)p(y)))
Loss = -MI
```

### NCC
```
NCC = Σ[(I - μ_I)(J - μ_J)] / √(Σ(I - μ_I)² · Σ(J - μ_J)²)
Loss = -mean(NCC²)
```

## Parameter Tuning Guidelines

### Window Size (SSIM, NCC)
- **Small (3-5):** Fast, less smoothing, good for small features
- **Medium (7-11):** Balanced, good default
- **Large (13-21):** Slow, more smoothing, better for large-scale structure

### Gradient Penalty Type
- **L1:** Sparse gradients, sharper edges, more tolerance to noise
- **L2:** Smoother gradients, more diffusion

### Loss Weights
- **SSIM vs MSE:** Start with 1:1, adjust based on visual quality
- **Similarity vs Regularization:** Start with 1:0.01 to 1:0.1
- **Gradient vs Pixel (fusion):** Typically 1:0.5 to 1:1

### Mutual Information Bins
- **16-32:** Fast, less precise, good for real-time
- **32-64:** Balanced, good default
- **64-128:** Slow, more precise, for final optimization

## Common Pitfalls and Solutions

### Issue: NaN Loss with SSIM
**Cause:** Zero variance in patches
**Solution:** Ensure input is normalized, check for constant regions

### Issue: Gradient Loss Always Zero
**Cause:** No spatial variation in prediction
**Solution:** Check if displacement/deformation field is being optimized

### Issue: Dice Loss Stuck at 1.0
**Cause:** No overlap between prediction and target
**Solution:** Check class encoding, ensure predictions are in correct format

### Issue: MI Loss Very Negative
**Cause:** Poor intensity alignment
**Solution:** Normalize inputs to [0, 1] range, check modality compatibility

### Issue: LPIPS Not Available
**Cause:** Package not installed
**Solution:** `pip install lpips`

## Performance Tips

### GPU Memory
- **Reduce window size** for SSIM/NCC if memory-limited
- **Use smaller batch size** with perceptual losses (LPIPS)
- **Disable gradient computation** for target images: `target.detach()`

### Speed
- **Functional interfaces** (`ssim()`, `ssim3D()`) are slightly faster if called once
- **Class instances** (e.g., `SSIM2D()`) are faster if called repeatedly (caches windows)
- **Avoid mixed precision** with information theory losses (MI, MIND)

### Numerical Stability
- Always add small epsilon (1e-6) to denominators
- Clamp inputs to valid ranges before computing losses
- Use gradient clipping when combining multiple losses

## Debugging Checklist

- [ ] Check input/target shapes match loss requirements
- [ ] Verify input value ranges (e.g., [0, 1] for SSIM, MI)
- [ ] Ensure gradients are enabled on predictions: `requires_grad=True`
- [ ] Check for NaN/Inf in inputs before loss computation
- [ ] Verify device placement (CPU vs GPU) is consistent
- [ ] Test loss on simple synthetic examples first
- [ ] Monitor loss values - they should be in expected ranges

## Expected Loss Value Ranges

| Loss | Typical Range | Notes |
|------|---------------|-------|
| SSIM2D/3D | 0.0 - 2.0 | 0 = identical, higher = more different |
| MSE | 0.0 - ∞ | Depends on input scale |
| Dice | 0.0 - 1.0 | 0 = perfect overlap, 1 = no overlap |
| NCC | -∞ - 0.0 | ~0 = good, more negative = worse |
| MI | -∞ - 0.0 | ~0 = high mutual info, negative = loss |
| Grad* | 0.0 - ∞ | Depends on field smoothness |

## Quick Troubleshooting

```python
import torch
from orochi.losses import get_loss_function

# Basic sanity check
def test_loss(name, input_shape, target_shape=None):
    loss_fn = get_loss_function(name)
    x = torch.randn(*input_shape, requires_grad=True)
    y = torch.randn(*(target_shape or input_shape))

    try:
        loss = loss_fn(x, y)
        loss.backward()
        print(f"✓ {name}: loss={loss.item():.4f}, grad_norm={x.grad.norm().item():.4f}")
    except Exception as e:
        print(f"✗ {name}: {e}")

# Test suite
test_loss('ssim2d', (2, 3, 128, 128))
test_loss('grad3d', (1, 3, 32, 64, 64))
test_loss('dice', (2, 1, 64, 64), (2, 1, 64, 64))
```

## Additional Resources

- **Full Documentation:** See `LOSS_CONSOLIDATION_SUMMARY.md`
- **Source Code:** `/home/user/Orochi-Versatile-Biomedical-Image-Processor/orochi/losses/losses.py`
- **API Reference:** All docstrings include mathematical formulas and examples

## Contact & Support

For issues or questions:
1. Check docstrings: `help(loss_function_name)`
2. Review examples in consolidation summary
3. Verify input shapes and value ranges
4. Test with simple synthetic data first
