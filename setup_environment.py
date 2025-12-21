#!/usr/bin/env python3
"""
Orochi Environment Setup Script

This script automates the setup process for the Orochi repository:
1. Finds the mamba_ssm package location
2. Applies the required fix to ssd_combined.py
3. Verifies the installation
4. Tests key dependencies

Usage:
    python setup_environment.py
"""

import os
import sys
import site
import subprocess


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def find_mamba_ssm_path():
    """Find the path to the mamba_ssm package."""
    print_header("Step 1: Finding mamba_ssm package")

    site_packages = site.getsitepackages()
    print(f"Site packages locations: {site_packages}")

    for path in site_packages:
        mamba_path = os.path.join(path, "mamba_ssm")
        if os.path.exists(mamba_path):
            print(f"✓ Found mamba_ssm at: {mamba_path}")
            return mamba_path

    print("✗ Could not find mamba_ssm package")
    print("  Please ensure mamba_ssm is installed: pip install mamba-ssm==2.2.2")
    return None


def apply_mamba_fix(mamba_path):
    """Apply the required fix to mamba_ssm package."""
    print_header("Step 2: Applying mamba_ssm fix")

    target_file = os.path.join(mamba_path, "ops", "triton", "ssd_combined.py")

    if not os.path.exists(target_file):
        print(f"✗ Target file not found: {target_file}")
        return False

    print(f"Target file: {target_file}")

    # Read the file
    try:
        with open(target_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

    # Check if fix is already applied
    fix_line = "        xBC = xBC.contiguous()"

    for i, line in enumerate(lines, 1):
        if fix_line.strip() in line:
            print(f"✓ Fix already applied at line {i}")
            return True

    # Find the line to insert after (around line 777)
    # Looking for: zx0, z, xBC, dt = torch.split(zxbcdt, ...)
    insert_after_idx = None
    for i, line in enumerate(lines):
        if "zx0, z, xBC, dt = torch.split(zxbcdt" in line:
            insert_after_idx = i
            print(f"Found split line at index {i} (line {i+1})")
            break

    if insert_after_idx is None:
        print("✗ Could not find the line to insert the fix")
        print("  Please apply the fix manually as described in the README")
        return False

    # Insert the fix
    try:
        # Create backup
        backup_file = target_file + ".backup"
        with open(backup_file, 'w') as f:
            f.writelines(lines)
        print(f"✓ Created backup: {backup_file}")

        # Insert the fix
        lines.insert(insert_after_idx + 1, fix_line + "\n")

        # Write the modified file
        with open(target_file, 'w') as f:
            f.writelines(lines)

        print(f"✓ Applied fix at line {insert_after_idx + 2}")
        return True

    except Exception as e:
        print(f"✗ Error applying fix: {e}")
        return False


def verify_installation():
    """Verify that all required packages are installed."""
    print_header("Step 3: Verifying installation")

    packages_to_test = [
        ("torch", "PyTorch"),
        ("mamba_ssm", "Mamba SSM"),
        ("monai", "MONAI"),
        ("einops", "Einops"),
        ("timm", "TIMM"),
        ("transformers", "Transformers"),
        ("wandb", "Weights & Biases"),
    ]

    all_passed = True

    for package, name in packages_to_test:
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - MISSING")
            all_passed = False

    return all_passed


def test_cuda():
    """Test CUDA availability."""
    print_header("Step 4: Testing CUDA")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ CUDA not available. GPU training will not work.")
            return False

    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        return False


def test_model_import():
    """Test if we can import the Orochi model."""
    print_header("Step 5: Testing model import")

    # Try to import from the experiments directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "temp", "experiments", "2D"))

    try:
        from ours_mamba import Orochi
        print("✓ Successfully imported Orochi model")
        return True
    except Exception as e:
        print(f"✗ Error importing Orochi model: {e}")
        print("  This is expected if you haven't set up the full codebase yet")
        return False


def check_directory_structure():
    """Check if the expected directory structure exists."""
    print_header("Directory Structure Check")

    base_dir = os.path.dirname(__file__)

    expected_dirs = [
        "temp/environment.yaml",
        "temp/experiments/2D",
        "temp/experiments/3D",
        "temp/data_preprocess",
    ]

    all_exist = True
    for dir_path in expected_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - NOT FOUND")
            all_exist = False

    # Check for optional directories
    optional_dirs = ["data", "checkpoints"]
    print("\nOptional directories (you'll create these):")
    for dir_path in optional_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            print(f"✓ {dir_path}")
        else:
            print(f"○ {dir_path} - Not created yet (normal)")

    return all_exist


def print_next_steps():
    """Print the next steps for the user."""
    print_header("Setup Complete!")

    print("""
Next Steps:

1. Download pretrained checkpoints:
   - 2D: https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt
   - 3D: https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt

   mkdir -p checkpoints/2D checkpoints/3D
   huggingface-cli download eternalaudrey/mamba-fm-2d-ckpt --local-dir checkpoints/2D
   huggingface-cli download eternalaudrey/mamba-fm-3d-ckpt --local-dir checkpoints/3D

2. Download datasets (see SETUP_GUIDE.md for details):
   - For 2D tasks: BioSR, Isotropic Restoration, Fusion datasets
   - For 3D tasks: InverseSR, Registration datasets

3. Run experiments:
   - Navigate to temp/experiments/2D/scripts/ or temp/experiments/3D/scripts/
   - Run the appropriate training scripts
   - Use Jupyter notebooks for inference and evaluation

4. For detailed instructions, see:
   - SETUP_GUIDE.md - Complete setup and usage guide
   - README.md - Original repository documentation

Happy experimenting! 🚀
    """)


def main():
    """Main setup function."""
    print_header("Orochi Environment Setup")

    print("This script will:")
    print("  1. Find and fix the mamba_ssm package")
    print("  2. Verify your environment")
    print("  3. Test CUDA availability")
    print("  4. Provide next steps")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    # Check directory structure
    check_directory_structure()

    # Find and fix mamba_ssm
    mamba_path = find_mamba_ssm_path()
    if mamba_path:
        apply_mamba_fix(mamba_path)

    # Verify installation
    verify_installation()

    # Test CUDA
    test_cuda()

    # Test model import
    test_model_import()

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
