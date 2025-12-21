#!/usr/bin/env python3
"""
Quick Test Script for Orochi

This script performs a quick test to verify that the environment is set up correctly
and the model can be loaded and run with dummy data.

Usage:
    python quick_test.py
"""

import os
import sys
import torch
import numpy as np


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_imports():
    """Test that all required packages can be imported."""
    print_section("Testing Package Imports")

    packages = [
        "torch",
        "torchvision",
        "numpy",
        "einops",
        "mamba_ssm",
        "monai",
        "timm",
    ]

    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed.append(package)

    if failed:
        print(f"\n⚠ Warning: {len(failed)} package(s) failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True


def test_cuda():
    """Test CUDA availability and GPU information."""
    print_section("Testing CUDA")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")

        # Test GPU tensor operations
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("\n✓ GPU tensor operations working!")
            return True
        except Exception as e:
            print(f"\n✗ GPU tensor operations failed: {e}")
            return False
    else:
        print("\n⚠ CUDA not available. Model will run on CPU (slow).")
        return False


def test_mamba_ssm():
    """Test mamba_ssm package and the fix."""
    print_section("Testing mamba_ssm")

    try:
        import mamba_ssm
        print(f"✓ mamba_ssm imported successfully")
        print(f"  Version: {mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'Unknown'}")

        # Try to import specific modules
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        print("✓ mamba_chunk_scan_combined imported (fix likely applied)")

        return True
    except ImportError as e:
        print(f"✗ Failed to import mamba_ssm: {e}")
        return False
    except Exception as e:
        print(f"⚠ mamba_ssm imported but with warning: {e}")
        return True


def test_model_creation():
    """Test creating an Orochi model instance."""
    print_section("Testing Orochi Model Creation")

    # Add the experiments directory to path
    experiments_2d_path = os.path.join(os.path.dirname(__file__), "temp", "experiments", "2D")
    if os.path.exists(experiments_2d_path):
        sys.path.insert(0, experiments_2d_path)
    else:
        print(f"✗ Experiments directory not found: {experiments_2d_path}")
        return False

    try:
        from ours_mamba import Orochi

        print("✓ Imported Orochi model class")

        # Try to create a model instance with minimal configuration
        # Note: You may need to adjust parameters based on the actual model requirements
        print("\nAttempting to create model instance...")

        # Create a simple config for testing
        model = Orochi(
            in_channels=1,
            out_channels=1,
            img_size=64,  # Small size for testing
        )

        print("✓ Model instance created successfully!")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModel Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        return True

    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test a forward pass with dummy data."""
    print_section("Testing Forward Pass")

    experiments_2d_path = os.path.join(os.path.dirname(__file__), "temp", "experiments", "2D")
    sys.path.insert(0, experiments_2d_path)

    try:
        from ours_mamba import Orochi

        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        model = Orochi(
            in_channels=1,
            out_channels=1,
            img_size=64,
        )
        model = model.to(device)
        model.eval()

        print("✓ Model moved to device")

        # Create dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 1, 64, 64).to(device)

        print(f"✓ Created dummy input: {dummy_input.shape}")

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Forward pass successful!")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")

        # Check output validity
        if torch.isnan(output).any():
            print("⚠ Warning: Output contains NaN values")
            return False
        elif torch.isinf(output).any():
            print("⚠ Warning: Output contains Inf values")
            return False
        else:
            print("✓ Output values are valid (no NaN or Inf)")
            return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print a summary of all tests."""
    print_section("Test Summary")

    total = len(results)
    passed = sum(results.values())

    print(f"\nTests passed: {passed}/{total}\n")

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    if passed == total:
        print("\n🎉 All tests passed! Your environment is ready.")
        print("   See SETUP_GUIDE.md for next steps.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please review the output above.")
        print("   See SETUP_GUIDE.md for troubleshooting.")


def main():
    """Run all tests."""
    print_section("Orochi Quick Test Suite")
    print("\nThis script will test your environment setup.")
    print("It may take a minute to complete...\n")

    results = {}

    # Run tests
    results["Package Imports"] = test_imports()
    results["CUDA"] = test_cuda()
    results["mamba_ssm"] = test_mamba_ssm()
    results["Model Creation"] = test_model_creation()
    results["Forward Pass"] = test_forward_pass()

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
