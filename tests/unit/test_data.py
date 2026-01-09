"""
Unit tests for Orochi data modules.

Tests cover:
- Dataset instantiation
- Data loading
- Transforms
- Collate functions
- Data utilities
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import pickle
from pathlib import Path

from orochi.data import get_dataset
from orochi.data.datasets import (
    PretrainDataset,
    CollateFn,
    random_crop,
)
from orochi.data.data_utils import (
    init_fn,
    pkload,
    add_mask,
    sample,
    get_all_coords,
    gen_feats,
    normalize_intensity,
    pad_to_size,
    crop_center,
)


class TestPretrainDataset:
    """Test PretrainDataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = PretrainDataset(root_dir=tmpdir, img_size=(256, 256))
            assert dataset is not None
            assert len(dataset) == 0  # Empty directory

    def test_with_npy_files(self):
        """Test dataset with .npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test .npy files
            for i in range(3):
                img = np.random.rand(100, 100).astype(np.float32)
                np.save(os.path.join(tmpdir, f"test_{i}.npy"), img)

            dataset = PretrainDataset(root_dir=tmpdir, img_size=(64, 64))
            assert len(dataset) == 3

            # Load a sample
            sample = dataset[0]
            assert isinstance(sample, torch.Tensor)
            assert sample.shape[0] == 1  # Channel dimension
            assert sample.min() >= 0 and sample.max() <= 1  # Normalized

    def test_normalization(self):
        """Test images are normalized to [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image with specific range
            img = np.random.rand(100, 100).astype(np.float32) * 255  # [0, 255]
            np.save(os.path.join(tmpdir, "test.npy"), img)

            dataset = PretrainDataset(root_dir=tmpdir, img_size=(64, 64))
            sample = dataset[0]

            # Should be normalized to [0, 1]
            assert 0 <= sample.min() <= sample.max() <= 1


class TestCollateFn:
    """Test CollateFn for batch collation."""

    def test_random_crop_collation(self):
        """Test collate function with random cropping."""
        collate_fn = CollateFn(target_size=(64, 64))

        # Create batch of different-sized images
        batch = [
            torch.randn(1, 100, 100),
            torch.randn(1, 120, 120),
            torch.randn(1, 80, 80),
        ]

        result = collate_fn(batch)

        # Should return a batched tensor
        assert result.shape == (3, 1, 64, 64)

    def test_random_crop_function(self):
        """Test random_crop function."""
        img = torch.randn(1, 128, 128)
        cropped = random_crop(img, target_size=(64, 64))

        assert cropped.shape == (1, 64, 64)


class TestDataUtils:
    """Test data utility functions."""

    def test_init_fn(self):
        """Test worker initialization function."""
        # Should not raise error
        init_fn(worker=0)
        init_fn(worker=5)

    def test_pkload(self):
        """Test pickle loading."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            data = {"key": "value", "array": np.array([1, 2, 3])}
            pickle.dump(data, f)
            fname = f.name

        try:
            loaded = pkload(fname)
            assert loaded["key"] == "value"
            assert np.array_equal(loaded["array"], np.array([1, 2, 3]))
        finally:
            os.unlink(fname)

    def test_add_mask(self):
        """Test add_mask function."""
        x = torch.randn(2, 3, 64, 64)
        mask = torch.randint(0, 21, (2, 64, 64))

        result = add_mask(x, mask, dim=1)

        # Should have 21 + 3 = 24 channels
        assert result.shape == (2, 24, 64, 64)

    def test_sample(self):
        """Test sampling function."""
        x = np.arange(100).reshape(100, 1)
        sampled = sample(x, size=10)

        assert sampled.shape == (10, 1)
        assert sampled.dtype == torch.int16

    def test_get_all_coords(self):
        """Test coordinate grid generation."""
        coords = get_all_coords(stride=16, shape=(64, 64, 64))

        # Check shape
        assert coords.ndim == 2
        assert coords.shape[1] == 3  # 3D coordinates

        # Check coordinates are within bounds
        assert coords.min() >= 0
        assert coords[:, 0].max() < 64  # D dimension
        assert coords[:, 1].max() < 64  # H dimension
        assert coords[:, 2].max() < 64  # W dimension

    def test_gen_feats(self):
        """Test position feature generation."""
        feats = gen_feats(shape=(64, 64, 64))

        # Check shape
        assert feats.shape == (64, 64, 64, 3)

        # Check normalization (center should be ~0)
        center = feats[32, 32, 32]
        assert np.allclose(center, [0, 0, 0], atol=0.1)

        # Check range
        assert feats.min() >= -0.5
        assert feats.max() <= 0.5

    def test_normalize_intensity(self):
        """Test intensity normalization."""
        img = np.random.randn(100, 100) * 50 + 100  # mean=100, std=50

        normalized = normalize_intensity(img, percentile=(1, 99))

        # Should be approximately in [0, 1]
        assert normalized.min() >= -0.1  # Allow small negative due to percentile
        assert normalized.max() <= 1.1

    def test_pad_to_size(self):
        """Test padding function."""
        img = np.random.rand(50, 50, 50)
        padded = pad_to_size(img, (64, 64, 64))

        assert padded.shape == (64, 64, 64)

    def test_crop_center(self):
        """Test center cropping."""
        img = np.random.rand(100, 100, 100)
        cropped = crop_center(img, (64, 64, 64))

        assert cropped.shape == (64, 64, 64)

    def test_pad_and_crop_roundtrip(self):
        """Test padding then cropping returns to original size."""
        original_shape = (50, 50, 50)
        img = np.random.rand(*original_shape)

        # Pad to larger size
        padded = pad_to_size(img, (100, 100, 100))

        # Crop back to original size
        cropped = crop_center(padded, original_shape)

        assert cropped.shape == original_shape


class TestDatasetFactory:
    """Test dataset factory function."""

    def test_factory_pretrain(self):
        """Test factory creates PretrainDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = get_dataset("pretrain", root_dir=tmpdir, img_size=(256, 256))
            assert isinstance(dataset, PretrainDataset)

    def test_factory_invalid_name(self):
        """Test factory raises error for invalid dataset name."""
        with pytest.raises(ValueError):
            get_dataset("invalid_dataset_name")


class TestDataNumericalStability:
    """Test numerical stability of data operations."""

    def test_normalize_zero_variance(self):
        """Test normalize handles zero variance."""
        img = np.ones((100, 100))  # Constant image

        normalized = normalize_intensity(img)

        # Should handle zero variance gracefully
        assert not np.isnan(normalized).any()

    def test_random_crop_edge_cases(self):
        """Test random crop with edge case sizes."""
        # Image smaller than target
        img = torch.randn(1, 32, 32)
        cropped = random_crop(img, target_size=(64, 64))
        assert cropped.shape == (1, 64, 64)

        # Image same size as target
        img = torch.randn(1, 64, 64)
        cropped = random_crop(img, target_size=(64, 64))
        assert cropped.shape == (1, 64, 64)

        # Image larger than target
        img = torch.randn(1, 128, 128)
        cropped = random_crop(img, target_size=(64, 64))
        assert cropped.shape == (1, 64, 64)


class TestDataReproducibility:
    """Test reproducibility of data operations."""

    def test_init_fn_deterministic(self):
        """Test init_fn produces deterministic results."""
        # Set seeds
        torch.manual_seed(42)
        init_fn(0)
        val1 = np.random.random()

        # Reset and repeat
        torch.manual_seed(42)
        init_fn(0)
        val2 = np.random.random()

        # Should be the same
        assert val1 == val2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDataCUDA:
    """Test data operations with CUDA."""

    def test_data_to_cuda(self):
        """Test data can be moved to CUDA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            img = np.random.rand(100, 100).astype(np.float32)
            np.save(os.path.join(tmpdir, "test.npy"), img)

            dataset = PretrainDataset(root_dir=tmpdir, img_size=(64, 64))
            sample = dataset[0]

            # Move to CUDA
            sample_cuda = sample.cuda()

            assert sample_cuda.is_cuda
