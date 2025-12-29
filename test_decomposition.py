"""
Test script for the decomposition task implementation.
Tests that the new decomp_decoder works correctly with dummy data.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import ours_mamba
import ml_collections


def get_test_config():
    """Create a minimal config for testing."""
    config = ml_collections.ConfigDict()

    # Image and patch settings
    config.img_size = [32, 256, 256]  # [D, H, W]
    config.patch_size = 4
    config.in_chans = 2  # Paired images

    # Model architecture
    config.embed_dim = 128
    config.depths = [4, 4, 4, 4]
    config.num_heads = [4, 8, 16, 32]

    # Decoder settings
    config.decoder_head_chan = 128
    config.grid_size = config.img_size

    # Skip connection settings
    config.if_convskip = True
    config.if_transskip = True

    return config


def test_decomposition(checkpoint_path=None):
    """Test the decomposition task with dummy data.

    Args:
        checkpoint_path: Optional path to pretrained checkpoint.
                        If provided, will load encoder weights.
    """

    print("=" * 80)
    print("Testing Decomposition Implementation")
    print("=" * 80)

    # Create config
    config = get_test_config()
    print(f"\n✓ Config created")
    print(f"  - Image size: {config.img_size}")
    print(f"  - Patch size: {config.patch_size}")
    print(f"  - Embed dim: {config.embed_dim}")

    # Create model
    print("\n✓ Creating MambaULight model...")
    try:
        model = ours_mamba.MambaULight(config)
        if torch.cuda.is_available():
            model = model.cuda()
            device = 'cuda'
            print("  - Model moved to CUDA")
        else:
            device = 'cpu'
            print("  - Using CPU")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

    # Load pretrained weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n✓ Loading pretrained weights from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Extract state dict (handle different checkpoint formats)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present (from DDP training)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v

            # Load weights with strict=False to allow new decomp_decoder
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

            print(f"  - Loaded checkpoint successfully!")
            if missing_keys:
                decomp_keys = [k for k in missing_keys if 'decomp' in k]
                other_keys = [k for k in missing_keys if 'decomp' not in k]
                if decomp_keys:
                    print(f"  - New decomp_decoder parameters (expected): {len(decomp_keys)} keys")
                if other_keys:
                    print(f"  - Missing keys (unexpected): {len(other_keys)} keys")
                    print(f"    First few: {other_keys[:3]}")
            if unexpected_keys:
                print(f"  - Unexpected keys: {len(unexpected_keys)} (will be ignored)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load checkpoint: {e}")
            print(f"  - Continuing with random initialization...")
    elif checkpoint_path:
        print(f"\n⚠ Checkpoint path provided but file not found: {checkpoint_path}")
        print(f"  - Continuing with random initialization...")
    else:
        print(f"\n✓ Using random initialization (no checkpoint provided)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Create dummy data - use config dimensions
    batch_size = 2
    C = 1  # Single channel
    D, H, W = config.img_size  # Use config dimensions
    print(f"\n✓ Creating dummy data")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Shape: ({batch_size}, {C}, {D}, {H}, {W})")

    raw_A = torch.rand(batch_size, C, D, H, W)
    raw_B = torch.rand(batch_size, C, D, H, W)

    if device == 'cuda':
        raw_A = raw_A.cuda()
        raw_B = raw_B.cuda()

    # Test forward pass
    print("\n✓ Testing forward pass...")
    model.eval()

    try:
        with torch.no_grad():
            # Test with both images provided
            logits, aux_losses = model(raw_A, raw_B)
        print("  - Forward pass successful!")
    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check outputs
    print("\n✓ Checking outputs...")

    # Check logits structure
    expected_tasks = ['raw', 'reg', 'fus', 'SR', 'IR', 'decomp']
    for task in expected_tasks:
        if task in logits:
            print(f"  ✓ '{task}' task present in logits")
        else:
            print(f"  ✗ '{task}' task missing from logits!")
            return False

    # Check decomposition outputs specifically
    decomp = logits['decomp']
    expected_decomp_keys = ['composite', 'decomposed_A', 'decomposed_B', 'original_A', 'original_B']
    for key in expected_decomp_keys:
        if key in decomp:
            shape = decomp[key].shape
            print(f"  ✓ decomp['{key}'] shape: {shape}")
        else:
            print(f"  ✗ decomp['{key}'] missing!")
            return False

    # Check losses
    print("\n✓ Checking losses...")

    if 'mse' in aux_losses:
        mse_losses = aux_losses['mse']
        expected_mse_tasks = ['reg', 'fus', 'SR', 'IR', 'decomp_A', 'decomp_B']
        for task in expected_mse_tasks:
            if task in mse_losses:
                loss_value = mse_losses[task].item()
                print(f"  ✓ mse['{task}'] = {loss_value:.6f}")
            else:
                print(f"  ✗ mse['{task}'] missing!")
                return False
    else:
        print(f"  ✗ 'mse' losses missing!")
        return False

    # Test total loss computation (simulating training)
    print("\n✓ Testing total loss computation...")
    try:
        # Flatten loss dict (as done in training)
        def flatten_loss_dict(loss_dict, parent_key='', sep='_'):
            items = []
            for k, v in loss_dict.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_loss_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_losses = flatten_loss_dict(aux_losses)
        total_loss = sum(flat_losses.values())

        print(f"  - Flattened loss keys: {list(flat_losses.keys())}")
        print(f"  - Total loss: {total_loss.item():.6f}")

        if 'mse_decomp_A' in flat_losses and 'mse_decomp_B' in flat_losses:
            print(f"  ✓ Decomposition losses included in total!")
        else:
            print(f"  ✗ Decomposition losses not found in flattened dict!")
            return False

    except Exception as e:
        print(f"✗ Error computing total loss: {e}")
        return False

    # Test with default raw_B (roll)
    print("\n✓ Testing with automatic pairing (raw_B=None)...")
    try:
        with torch.no_grad():
            logits2, aux_losses2 = model(raw_A)  # No raw_B provided
        print("  ✓ Automatic pairing works!")
    except Exception as e:
        print(f"✗ Error with automatic pairing: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe decomposition implementation is working correctly.")
    print("You can now train the model with the decomposition task included.")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test decomposition implementation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint (e.g., pretrained_checkpoints/MambaULight2D_epoch_99_loss_-0.0624.pth.tar)')
    args = parser.parse_args()

    success = test_decomposition(checkpoint_path=args.checkpoint)
    sys.exit(0 if success else 1)
