"""Tests for loss functions.

Run with:
    pytest tests/test_losses/
    pytest tests/test_losses/test_reconstruction.py -v
    pytest tests/test_losses/ --cov=orochi.losses --cov-report=html
"""

import pytest
import torch

from orochi.losses import SSIM, PSNRLoss, L1Loss, L2Loss


class TestSSIM:
    """Test SSIM loss function."""

    @pytest.fixture
    def sample_2d_images(self):
        """Create sample 2D images for testing."""
        torch.manual_seed(42)
        return torch.randn(2, 1, 256, 256)

    @pytest.fixture
    def sample_3d_volumes(self):
        """Create sample 3D volumes for testing."""
        torch.manual_seed(42)
        return torch.randn(2, 1, 32, 256, 256)

    def test_ssim_2d_shape(self, sample_2d_images):
        """Test SSIM with 2D inputs returns correct shape."""
        ssim = SSIM(window_size=11)
        img1, img2 = sample_2d_images[:1], sample_2d_images[1:2]
        loss = ssim(img1, img2)

        assert loss.ndim == 0, "SSIM should return scalar"
        assert torch.is_tensor(loss), "Output should be a tensor"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_ssim_3d_shape(self, sample_3d_volumes):
        """Test SSIM with 3D inputs returns correct shape."""
        ssim = SSIM(window_size=11)
        vol1, vol2 = sample_3d_volumes[:1], sample_3d_volumes[1:2]
        loss = ssim(vol1, vol2)

        assert loss.ndim == 0, "SSIM should return scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_ssim_identical_images(self, sample_2d_images):
        """SSIM of identical images should be close to 0."""
        ssim = SSIM(window_size=11)
        img = sample_2d_images[:1]
        loss = ssim(img, img)

        assert loss < 0.01, f"SSIM of identical images should be ~0, got {loss.item()}"

    def test_ssim_range(self, sample_2d_images):
        """SSIM loss should be in valid range [0, 2]."""
        ssim = SSIM(window_size=11)
        img1, img2 = sample_2d_images[:1], sample_2d_images[1:2]
        loss = ssim(img1, img2)

        assert 0 <= loss <= 2, f"SSIM loss should be in [0, 2], got {loss.item()}"

    def test_ssim_invalid_input(self):
        """Test error handling for invalid inputs."""
        ssim = SSIM()

        with pytest.raises(ValueError, match="Expected 4D or 5D tensor"):
            # 3D tensor (missing batch dimension)
            ssim(torch.randn(1, 256, 256), torch.randn(1, 256, 256))

    def test_ssim_shape_mismatch(self, sample_2d_images):
        """Test error handling for shape mismatch."""
        ssim = SSIM()
        img1 = sample_2d_images[:1]
        img2 = torch.randn(1, 1, 128, 128)  # Different size

        with pytest.raises(ValueError, match="Shape mismatch"):
            ssim(img1, img2)

    def test_ssim_batch_processing(self):
        """Test SSIM with batch of images."""
        ssim = SSIM(window_size=11)
        batch_size = 4
        imgs1 = torch.randn(batch_size, 1, 256, 256)
        imgs2 = torch.randn(batch_size, 1, 256, 256)

        loss = ssim(imgs1, imgs2)
        assert loss.ndim == 0, "Should return scalar with reduction='mean'"

    def test_ssim_window_size(self):
        """Test SSIM with different window sizes."""
        for window_size in [7, 9, 11, 13]:
            ssim = SSIM(window_size=window_size)
            img1 = torch.randn(1, 1, 256, 256)
            img2 = torch.randn(1, 1, 256, 256)
            loss = ssim(img1, img2)
            assert not torch.isnan(loss), f"Failed for window_size={window_size}"

    def test_ssim_multichannel(self):
        """Test SSIM with multi-channel images."""
        ssim = SSIM(channel=3)
        img1 = torch.randn(2, 3, 256, 256)
        img2 = torch.randn(2, 3, 256, 256)
        loss = ssim(img1, img2)
        assert not torch.isnan(loss)


class TestPSNRLoss:
    """Test PSNR loss function."""

    def test_psnr_2d(self):
        """Test PSNR with 2D images."""
        psnr_loss = PSNRLoss(max_val=1.0)
        img1 = torch.randn(2, 1, 256, 256)
        img2 = torch.randn(2, 1, 256, 256)
        loss = psnr_loss(img1, img2)

        assert torch.is_tensor(loss)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_psnr_identical(self):
        """PSNR of identical images should be very negative (high PSNR)."""
        psnr_loss = PSNRLoss(max_val=1.0)
        img = torch.randn(1, 1, 256, 256)
        loss = psnr_loss(img, img + 1e-10)  # Nearly identical

        # PSNR should be very high (loss very negative)
        assert loss < -30, "PSNR of identical images should be high"


class TestL1L2Losses:
    """Test L1 and L2 losses."""

    def test_l1_loss(self):
        """Test L1 loss computation."""
        l1_loss = L1Loss()
        pred = torch.randn(2, 1, 256, 256)
        target = torch.randn(2, 1, 256, 256)
        loss = l1_loss(pred, target)

        assert torch.is_tensor(loss)
        assert loss >= 0, "L1 loss should be non-negative"

    def test_l2_loss(self):
        """Test L2 loss computation."""
        l2_loss = L2Loss()
        pred = torch.randn(2, 1, 256, 256)
        target = torch.randn(2, 1, 256, 256)
        loss = l2_loss(pred, target)

        assert torch.is_tensor(loss)
        assert loss >= 0, "L2 loss should be non-negative"

    def test_l1_identical(self):
        """L1 loss of identical tensors should be 0."""
        l1_loss = L1Loss()
        x = torch.randn(2, 1, 256, 256)
        loss = l1_loss(x, x)

        assert loss < 1e-6, "L1 loss of identical tensors should be ~0"

    def test_l2_identical(self):
        """L2 loss of identical tensors should be 0."""
        l2_loss = L2Loss()
        x = torch.randn(2, 1, 256, 256)
        loss = l2_loss(x, x)

        assert loss < 1e-6, "L2 loss of identical tensors should be ~0"
