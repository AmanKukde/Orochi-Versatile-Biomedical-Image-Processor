"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Get device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_2d_batch():
    """Create a sample 2D batch for testing."""
    torch.manual_seed(42)
    return torch.randn(4, 1, 256, 256)


@pytest.fixture
def sample_3d_batch():
    """Create a sample 3D batch for testing."""
    torch.manual_seed(42)
    return torch.randn(2, 1, 32, 256, 256)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from orochi.configs import Mamba2DConfig

    return Mamba2DConfig(
        img_size=[256, 256],
        batch_size=2,
        num_epochs=2,  # Short for testing
        log_interval=1,
    )
