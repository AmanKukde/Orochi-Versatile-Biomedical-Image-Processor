"""
Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import torch
import numpy as np
from types import SimpleNamespace


@pytest.fixture
def device():
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_2d_image():
    """Generate a sample 2D image batch for testing."""
    # Shape: [B, C, H, W]
    return torch.randn(2, 1, 64, 64)


@pytest.fixture
def sample_3d_volume():
    """Generate a sample 3D volume batch for testing."""
    # Shape: [B, C, D, H, W]
    return torch.randn(2, 1, 32, 64, 64)


@pytest.fixture
def sample_config_2d():
    """Generate a sample 2D configuration."""
    return SimpleNamespace(
        dimensions=2,
        img_size=(64, 64),
        in_chans=1,
        embed_dim=96,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=4,
        patch_size=4,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
    )


@pytest.fixture
def sample_config_3d():
    """Generate a sample 3D configuration."""
    return SimpleNamespace(
        dimensions=3,
        img_size=(32, 64, 64),
        in_chans=1,
        embed_dim=96,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=(4, 4, 4),
        patch_size=4,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
    )


@pytest.fixture
def sample_config_train():
    """Generate a sample training configuration."""
    return SimpleNamespace(
        data_dir="/tmp/test_data",
        img_size=(256, 256),
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        learning_rate=1e-4,
        epochs=10,
        save_frequency=5,
    )
