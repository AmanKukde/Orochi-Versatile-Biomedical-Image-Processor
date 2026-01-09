# Quick Start Guide: Consolidated Mamba Model

## Overview

The consolidated Mamba model is now available at:
```
/home/user/Orochi-Versatile-Biomedical-Image-Processor/orochi/models/mamba.py
```

This single file replaces all three previous implementations and supports both 2D and 3D via a `dimensions` parameter.

## Quick Examples

### Example 1: 2D Image Registration

```python
from types import SimpleNamespace
from orochi.models.mamba import MambaEncoderHeria, reg_decoder, SpatialTransformer
import torch

# Configuration
config = SimpleNamespace(
    dimensions=2,                    # ← KEY PARAMETER
    img_size=(256, 256),
    patch_size=4,
    in_chans=2,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    ssm_cfg={},
    residual_in_fp32=True,
    fused_add_norm=True,
    rms_norm=True,
    norm_epsilon=1e-5,
    initializer_cfg={},
    drop_rate=0.0,
    drop_path_rate=0.1,
    use_checkpoint=False,
    pat_merg_rf=2,
    patch_norm=True,
    out_indices=[0, 1, 2, 3],
    if_convskip=True,
    decoder_mode='2d',
    decoder_head_chan=32,
    decoder_bn=False,
    decoder_depthseparable=True,
    head_sparsity=0.0,
)

# Create model
encoder = MambaEncoderHeria(config)
decoder = reg_decoder(config)
spatial_trans = SpatialTransformer((256, 256))

# Forward pass
source = torch.randn(2, 1, 256, 256)
target = torch.randn(2, 1, 256, 256)
x = torch.cat([source, target], dim=1)

out_feats = encoder(x)
flow = decoder(out_feats)  # (2, 2, 256, 256) - 2D flow field
registered = spatial_trans(source, flow)
```

### Example 2: 3D Volume Super-Resolution

```python
from types import SimpleNamespace
from orochi.models.mamba import MambaEncoderHeria, SR_decoder
import torch

# Configuration
config = SimpleNamespace(
    dimensions=3,                    # ← KEY PARAMETER
    img_size=(32, 256, 256),
    patch_size=4,
    in_chans=2,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    ssm_cfg={},
    residual_in_fp32=True,
    fused_add_norm=True,
    rms_norm=True,
    norm_epsilon=1e-5,
    initializer_cfg={},
    drop_rate=0.0,
    drop_path_rate=0.1,
    use_checkpoint=False,
    pat_merg_rf=2,
    patch_norm=True,
    out_indices=[0, 1, 2, 3],
    if_convskip=True,
    decoder_mode='3d',
    decoder_head_chan=32,
    decoder_bn=False,
    decoder_depthseparable=True,
    head_sparsity=0.0,
)

# Create model
encoder = MambaEncoderHeria(config)
decoder = SR_decoder(config)

# Forward pass
lowres = torch.randn(2, 1, 32, 256, 256)
x = torch.cat([lowres, lowres], dim=1)

out_feats = encoder(x)
highres = decoder(out_feats)  # (2, 1, 32, 256, 256)
```

### Example 3: Loading Pre-trained Weights

```python
from orochi.models.mamba import MambaEncoderHeria
import torch

# For 2D weights
config_2d = create_config(dimensions=2)
model_2d = MambaEncoderHeria(config_2d)
model_2d.load_state_dict(torch.load('2d_weights.pth'))

# For 3D weights
config_3d = create_config(dimensions=3)
model_3d = MambaEncoderHeria(config_3d)
model_3d.load_state_dict(torch.load('3d_weights.pth'))
```

## Migration from Old Code

### Option 1: Update Config (Recommended)

Add `dimensions` parameter to your existing config:

```python
# Add this line to your config
config.dimensions = 2  # or 3 for 3D

# Update import
from orochi.models.mamba import MambaEncoderHeria  # New
# from temp.experiments.2D.ours_mamba import MambaEncoderHeria  # Old
```

### Option 2: Minimal Changes

If you don't want to modify config files, the model defaults to 3D:

```python
# Works for 3D without any config changes
from orochi.models.mamba import MambaEncoderHeria
encoder = MambaEncoderHeria(config)  # Defaults to 3D
```

## Available Decoders

All decoders automatically support both 2D and 3D:

```python
from orochi.models.mamba import (
    reg_decoder,        # Registration
    fus_decoder,        # Image fusion
    fusRGB_decoder,     # RGB fusion
    SR_decoder,         # Super-resolution
    IR_decoder,         # Image restoration
    den_decoder,        # Denoising
    Proj_decoder,       # Projection
    seg_decoder,        # Segmentation
    split_decoder,      # Image splitting (2 channels)
    split_decoder_single,  # Image splitting (1 channel)
)
```

## Key Differences by Dimension

| Feature | 2D | 3D |
|---------|-----|-----|
| Input shape | `(B, C, H, W)` | `(B, C, T, H, W)` |
| Patch embed | `Conv2d` | `Conv3d` |
| Patch merge | 4 patches | 8 patches |
| Upsampling | `bilinear` | `trilinear` |
| Reg decoder output | 2 channels | 3 channels |
| Head convolution | `Conv2d` | `Conv3d` |

## Common Configuration Template

```python
from types import SimpleNamespace

def create_config(dimensions=3):
    """Create standard configuration."""
    # Set image size based on dimensions
    if dimensions == 2:
        img_size = (256, 256)
    else:
        img_size = (32, 256, 256)

    return SimpleNamespace(
        # REQUIRED: Specify dimensions
        dimensions=dimensions,

        # Image/Volume parameters
        img_size=img_size,
        patch_size=4,
        in_chans=2,

        # Architecture
        embed_dim=96,
        depths=[2, 2, 2, 2],

        # Mamba-specific
        ssm_cfg={},
        residual_in_fp32=True,
        fused_add_norm=True,
        rms_norm=True,
        norm_epsilon=1e-5,
        initializer_cfg={},

        # Training
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,

        # Hierarchical structure
        pat_merg_rf=2,
        patch_norm=True,
        out_indices=[0, 1, 2, 3],

        # Decoder
        if_convskip=True,
        if_transskip=False,
        decoder_mode='2d' if dimensions == 2 else '3d',
        decoder_head_chan=32,
        decoder_bn=False,
        decoder_depthseparable=True,
        head_sparsity=0.0,
        class_embed_dim=0,
    )
```

## Testing Your Model

```python
# Test 2D
config_2d = create_config(dimensions=2)
encoder_2d = MambaEncoderHeria(config_2d)
x_2d = torch.randn(1, 2, 256, 256)
out_2d = encoder_2d(x_2d)
print(f"2D output levels: {len(out_2d)}")
for i, feat in enumerate(out_2d):
    print(f"  Level {i}: {feat.shape}")

# Test 3D
config_3d = create_config(dimensions=3)
encoder_3d = MambaEncoderHeria(config_3d)
x_3d = torch.randn(1, 2, 32, 256, 256)
out_3d = encoder_3d(x_3d)
print(f"3D output levels: {len(out_3d)}")
for i, feat in enumerate(out_3d):
    print(f"  Level {i}: {feat.shape}")
```

## Troubleshooting

### Issue: "dimensions attribute not found"

**Solution**: Add `dimensions` to your config:
```python
config.dimensions = 2  # or 3
```

### Issue: "decoder_mode mismatch"

**Solution**: Ensure `decoder_mode` matches `dimensions`:
```python
config.decoder_mode = '2d' if config.dimensions == 2 else '3d'
```

### Issue: "Weight loading fails"

**Check**: Make sure dimensions match between saved weights and model:
```python
# If weights are from 2D model, use dimensions=2
# If weights are from 3D model, use dimensions=3
```

### Issue: "Wrong output shape"

**Check**: Verify input shape matches dimensions:
```python
# 2D: (B, C, H, W)
# 3D: (B, C, T, H, W)
```

## Performance Tips

1. **Use gradient checkpointing** for large models:
   ```python
   config.use_checkpoint = True
   ```

2. **Enable mixed precision**:
   ```python
   config.residual_in_fp32 = True
   ```

3. **Adjust depth** based on GPU memory:
   ```python
   config.depths = [2, 2, 2, 2]  # Smaller
   config.depths = [3, 3, 6, 3]  # Larger
   ```

4. **Reduce embedding dimension** for faster inference:
   ```python
   config.embed_dim = 48  # Smaller, faster
   config.embed_dim = 96  # Default
   config.embed_dim = 192  # Larger, slower
   ```

## Next Steps

1. Review the full analysis: `MAMBA_CONSOLIDATION_ANALYSIS.md`
2. Check the demonstration script: `orochi/models/mamba_demo.py`
3. Run your own tests with the new consolidated model
4. Update your training scripts to use the new import path

## Support

For issues or questions:
1. Check the analysis document for detailed information
2. Verify your configuration includes `dimensions` parameter
3. Ensure input shapes match the specified dimensions
4. Test with the provided examples first

## Key Takeaway

**The only change needed**: Add `config.dimensions = 2` or `config.dimensions = 3` to your configuration, and update your import path!

Everything else remains the same, including weight loading compatibility.
