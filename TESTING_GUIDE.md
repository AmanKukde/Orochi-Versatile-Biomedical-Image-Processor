# Testing the Decomposition Implementation

## Quick Start

### Without Pretrained Weights (Random Initialization)

```bash
python test_decomposition.py
```

This will test the decomposition implementation with randomly initialized weights.

### With Pretrained Weights

If you have downloaded the pretrained checkpoint:

```bash
python test_decomposition.py --checkpoint pretrained_checkpoints/MambaULight2D_epoch_99_loss_-0.0624.pth.tar
```

**Note**: The decomposition decoder is new, so it won't have pretrained weights. The test will:
- ✅ Load encoder weights from checkpoint
- ✅ Load existing decoder weights (reg, fus, SR, IR)
- ✅ Initialize decomp_decoder with random weights (expected)

## Expected Output

```
================================================================================
Testing Decomposition Implementation
================================================================================

✓ Config created
  - Image size: [32, 256, 256]
  - Patch size: 4
  - Embed dim: 128

✓ Creating MambaULight model...
  - Model moved to CUDA
  - Total parameters: 54,107,736
  - Trainable parameters: 54,107,736

✓ Loading pretrained weights from: pretrained_checkpoints/...
  - Loaded checkpoint successfully!
  - New decomp_decoder parameters (expected): XX keys

✓ Creating dummy data
  - Batch size: 2
  - Shape: (2, 1, 32, 256, 256)

✓ Testing forward pass...
  - Forward pass successful!

✓ Checking outputs...
  ✓ 'raw' task present in logits
  ✓ 'reg' task present in logits
  ✓ 'fus' task present in logits
  ✓ 'SR' task present in logits
  ✓ 'IR' task present in logits
  ✓ 'decomp' task present in logits
  ✓ decomp['composite'] shape: (2, 1, 32, 256, 256)
  ✓ decomp['decomposed_A'] shape: (2, 1, 32, 256, 256)
  ✓ decomp['decomposed_B'] shape: (2, 1, 32, 256, 256)
  ✓ decomp['original_A'] shape: (2, 1, 32, 256, 256)
  ✓ decomp['original_B'] shape: (2, 1, 32, 256, 256)

✓ Checking losses...
  ✓ mse['reg'] = 0.123456
  ✓ mse['fus'] = 0.234567
  ✓ mse['SR'] = 0.345678
  ✓ mse['IR'] = 0.456789
  ✓ mse['decomp_A'] = 0.567890
  ✓ mse['decomp_B'] = 0.678901

✓ Testing total loss computation...
  - Flattened loss keys: ['mse_reg', 'mse_fus', 'mse_SR', 'mse_IR',
                           'mse_decomp_A', 'mse_decomp_B', 'ncc_reg', 'grad_reg']
  - Total loss: 1.234567
  ✓ Decomposition losses included in total!

✓ Testing with automatic pairing (raw_B=None)...
  ✓ Automatic pairing works!

================================================================================
✅ ALL TESTS PASSED!
================================================================================

The decomposition implementation is working correctly.
You can now train the model with the decomposition task included.
```

## Downloading Pretrained Weights

If you don't have the pretrained checkpoints yet:

### Option 1: HuggingFace (Original Pretrained Models)

```bash
# For 2D model
wget https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt/resolve/main/checkpoint.pth -O pretrained_checkpoints/mamba_fm_2d.pth

# For 3D model
wget https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt/resolve/main/checkpoint.pth -O pretrained_checkpoints/mamba_fm_3d.pth
```

### Option 2: Use Your Own Checkpoint

If you've already trained a model, use your checkpoint:

```bash
python test_decomposition.py --checkpoint path/to/your/checkpoint.pth
```

## Troubleshooting

### Issue: Dimension Mismatch Error

**Error**: `The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension Z`

**Solution**: Make sure your config's `img_size` matches your test data. The test script automatically uses `config.img_size` to create test data, but if you're using a custom config, ensure it matches your checkpoint's training configuration.

### Issue: Missing Keys Warning

**Warning**: `Missing keys (unexpected): N keys`

**If the missing keys are NOT related to `decomp_decoder`**: This might indicate a mismatch between your checkpoint and the current model architecture. Check:
1. Config parameters match checkpoint training config
2. Patch size, embed dim, depths match
3. Model version compatibility

**If missing keys are only `decomp_decoder.*`**: This is **expected** and normal! The decomposition decoder is new and won't be in old checkpoints.

### Issue: Checkpoint Format Not Recognized

The test handles multiple checkpoint formats:
- `{'state_dict': ...}`
- `{'model': ...}`
- Direct state dict

If your checkpoint has a different format, you may need to adjust the loading code.

## Next Steps After Testing

Once the test passes:

1. **Train with Decomposition**: The decomposition task is now integrated into training
2. **Monitor Losses**: Watch `mse_decomp_A` and `mse_decomp_B` during training
3. **Visualize Results**: Use inference scripts to see decomposition quality
4. **Fine-tune** (optional): Create a decomposition-specific fine-tuning mode

## Configuration Notes

The test uses these default settings:

```python
config.img_size = [32, 256, 256]  # [D, H, W]
config.patch_size = 4
config.embed_dim = 128
config.depths = [4, 4, 4, 4]
```

If your checkpoint was trained with different settings, modify `get_test_config()` in `test_decomposition.py` to match.

## Manual Testing (Python REPL)

```python
import torch
import sys
sys.path.insert(0, 'src')
import ours_mamba
import ml_collections

# Create config
config = ml_collections.ConfigDict()
config.img_size = [32, 256, 256]
config.patch_size = 4
config.in_chans = 2
config.embed_dim = 128
config.depths = [4, 4, 4, 4]
config.num_heads = [4, 8, 16, 32]
config.decoder_head_chan = 128
config.grid_size = config.img_size
config.if_convskip = True
config.if_transskip = True

# Create model
model = ours_mamba.MambaULight(config).cuda()

# Load checkpoint (optional)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Test
raw = torch.rand(2, 1, 32, 256, 256).cuda()
logits, losses = model(raw)

print("Decomposition outputs:", logits['decomp'].keys())
print("Decomposition losses:", losses['mse']['decomp_A'], losses['mse']['decomp_B'])
```
