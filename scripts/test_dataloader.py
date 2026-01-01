#!/usr/bin/env python3
"""Test script for BiomedicalDataset data loading.

This script verifies that:
- Dataset finds files correctly
- Images load without errors
- Preprocessing produces correct shapes
- Train/val split works properly

Usage:
    python scripts/test_dataloader.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from orochi.configs.model_configs import ViT3DConfig

# Import the dataset (need to import from the training script)
from src.finetune_with_wandb import BiomedicalDataset


def test_dataset_creation():
    """Test dataset initialization."""
    print("=" * 60)
    print("TEST 1: Dataset Creation")
    print("=" * 60)

    config = ViT3DConfig()

    # Create datasets
    train_dataset = BiomedicalDataset(
        data_root="/group/jug/aman/orochi/data",
        datasets=['hipsc_3d', 'hipsc_2d', 'hipct_2d', 'idr_2d'],
        img_size=config.img_size,
        split='train',
        val_split=0.1
    )

    val_dataset = BiomedicalDataset(
        data_root="/group/jug/aman/orochi/data",
        datasets=['hipsc_3d', 'hipsc_2d', 'hipct_2d', 'idr_2d'],
        img_size=config.img_size,
        split='val',
        val_split=0.1
    )

    print(f"\n✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")
    print(f"✓ Total: {len(train_dataset) + len(val_dataset)} samples")

    return train_dataset, val_dataset


def test_data_loading(dataset, num_samples=5):
    """Test loading individual samples."""
    print("\n" + "=" * 60)
    print("TEST 2: Data Loading")
    print("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            image = sample['image']
            idx = sample['idx']

            print(f"\nSample {i}:")
            print(f"  ✓ Index: {idx}")
            print(f"  ✓ Image shape: {image.shape}")
            print(f"  ✓ Image dtype: {image.dtype}")
            print(f"  ✓ Image range: [{image.min():.4f}, {image.max():.4f}]")
            print(f"  ✓ Image mean: {image.mean():.4f}")

            # Verify shape
            assert len(image.shape) == 4, f"Expected 4D tensor, got {len(image.shape)}D"
            assert image.shape[1:] == tuple(dataset.img_size), \
                f"Expected shape (C, {dataset.img_size}), got {image.shape}"

            # Verify normalization
            assert image.min() >= 0 and image.max() <= 1, \
                f"Expected normalized range [0, 1], got [{image.min()}, {image.max()}]"

        except Exception as e:
            print(f"\n✗ Error loading sample {i}: {e}")
            raise

    print(f"\n✓ All {num_samples} samples loaded successfully!")


def test_dataloader(dataset, batch_size=2):
    """Test DataLoader with batching."""
    print("\n" + "=" * 60)
    print("TEST 3: DataLoader Batching")
    print("=" * 60)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    print(f"\nDataLoader config:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: 2")
    print(f"  Total batches: {len(loader)}")

    # Load first batch
    batch = next(iter(loader))
    images = batch['image']
    indices = batch['idx']

    print(f"\nFirst batch:")
    print(f"  ✓ Batch shape: {images.shape}")
    print(f"  ✓ Indices: {indices}")
    print(f"  ✓ Memory usage: {images.element_size() * images.nelement() / 1024**2:.2f} MB")

    # Verify batch shape
    expected_shape = (batch_size, 1) + tuple(dataset.img_size)
    assert images.shape == expected_shape, \
        f"Expected batch shape {expected_shape}, got {images.shape}"

    print(f"\n✓ DataLoader working correctly!")


def test_file_distribution(dataset):
    """Test file distribution across datasets."""
    print("\n" + "=" * 60)
    print("TEST 4: File Distribution")
    print("=" * 60)

    # Count files per dataset
    file_counts = {}
    for img_file in dataset.image_files[:100]:  # Sample first 100
        dataset_name = img_file.parts[-3] if 'data' in img_file.parts else img_file.parts[-2]
        file_counts[dataset_name] = file_counts.get(dataset_name, 0) + 1

    print("\nFile distribution (first 100 files):")
    for dataset_name, count in sorted(file_counts.items()):
        print(f"  {dataset_name}: {count} files")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BiomedicalDataset Data Loading Test")
    print("=" * 60)

    try:
        # Test 1: Dataset creation
        train_dataset, val_dataset = test_dataset_creation()

        # Test 2: Data loading
        test_data_loading(train_dataset, num_samples=5)

        # Test 3: DataLoader batching
        test_dataloader(train_dataset, batch_size=2)

        # Test 4: File distribution
        test_file_distribution(train_dataset)

        # Summary
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print(f"\nDataset ready for training:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Image size: {train_dataset.img_size}")
        print(f"\nNext step: Run training with:")
        print(f"  python src/finetune_with_wandb.py --config configs/vit_finetune.yaml --model vit")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
