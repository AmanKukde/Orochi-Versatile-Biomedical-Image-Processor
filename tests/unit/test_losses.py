"""
Unit tests for Orochi loss functions.

Tests cover:
- Loss instantiation
- Forward pass functionality
- Shape compatibility
- Numerical stability
- Gradient flow
- Factory function
"""

import pytest
import torch
import numpy as np
from orochi.losses import (
    get_loss_function,
    SSIM2D,
    SSIM3D,
    Grad,
    Grad3d,
    NCC_vxm,
    DiceLoss,
    DisplacementRegularizer,
    MutualInformation,
)


class TestSSIMLosses:
    """Test SSIM-based loss functions."""

    def test_ssim2d_instantiation(self):
        """Test SSIM2D can be instantiated."""
        loss_fn = SSIM2D(window_size=11)
        assert loss_fn is not None

    def test_ssim2d_forward(self, sample_2d_image):
        """Test SSIM2D forward pass."""
        loss_fn = SSIM2D(window_size=11)
        img1 = sample_2d_image
        img2 = sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)

        loss = loss_fn(img1, img2)

        # Check loss is scalar
        assert loss.ndim == 0
        # SSIM loss should be between 0 and 1
        assert 0 <= loss <= 1, f"SSIM loss {loss} out of range [0, 1]"

    def test_ssim3d_forward(self, sample_3d_volume):
        """Test SSIM3D forward pass."""
        loss_fn = SSIM3D(window_size=11)
        vol1 = sample_3d_volume
        vol2 = sample_3d_volume + 0.1 * torch.randn_like(sample_3d_volume)

        loss = loss_fn(vol1, vol2)

        # Check loss is scalar
        assert loss.ndim == 0
        # SSIM loss should be between 0 and 1
        assert 0 <= loss <= 1

    def test_ssim_perfect_match(self, sample_2d_image):
        """Test SSIM returns ~0 for identical images."""
        loss_fn = SSIM2D(window_size=11)
        loss = loss_fn(sample_2d_image, sample_2d_image)

        # Perfect match should give very small loss
        assert loss < 0.01, f"Expected loss ~0 for identical images, got {loss}"

    def test_ssim_gradient_flow(self, sample_2d_image):
        """Test gradients flow through SSIM."""
        loss_fn = SSIM2D(window_size=11)
        img1 = sample_2d_image.clone().requires_grad_(True)
        img2 = sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)

        loss = loss_fn(img1, img2)
        loss.backward()

        # Check gradients exist
        assert img1.grad is not None
        assert not torch.isnan(img1.grad).any()


class TestGradientLosses:
    """Test gradient-based regularization losses."""

    def test_grad_2d_instantiation(self):
        """Test Grad (2D) can be instantiated."""
        loss_fn = Grad(penalty="l2")
        assert loss_fn is not None

    def test_grad_3d_instantiation(self):
        """Test Grad3d can be instantiated."""
        loss_fn = Grad3d(penalty="l2")
        assert loss_fn is not None

    def test_grad_2d_forward(self):
        """Test Grad (2D) forward pass."""
        loss_fn = Grad(penalty="l2")

        # Create 2D displacement field [B, 2, H, W]
        flow = torch.randn(2, 2, 64, 64)

        loss = loss_fn(None, flow)  # First arg can be None

        # Check loss is scalar
        assert loss.ndim == 0
        # Gradient penalty should be non-negative
        assert loss >= 0

    def test_grad_3d_forward(self):
        """Test Grad3d forward pass."""
        loss_fn = Grad3d(penalty="l2")

        # Create 3D displacement field [B, 3, D, H, W]
        flow = torch.randn(2, 3, 32, 64, 64)

        loss = loss_fn(None, flow)  # First arg can be None

        # Check loss is scalar
        assert loss.ndim == 0
        # Gradient penalty should be non-negative
        assert loss >= 0

    def test_grad_l1_vs_l2(self):
        """Test L1 vs L2 gradient penalties give different results."""
        flow = torch.randn(2, 2, 64, 64)

        loss_l1 = Grad(penalty="l1")(None, flow)
        loss_l2 = Grad(penalty="l2")(None, flow)

        # L1 and L2 should give different values
        assert not torch.allclose(loss_l1, loss_l2)

    def test_displacement_regularizer(self):
        """Test DisplacementRegularizer."""
        loss_fn = DisplacementRegularizer(energy_type="gradient-l2")

        # Create 3D displacement field
        flow = torch.randn(2, 3, 32, 64, 64)

        loss = loss_fn(flow)

        # Check loss is scalar and non-negative
        assert loss.ndim == 0
        assert loss >= 0


class TestInformationTheoryLosses:
    """Test information theory-based losses."""

    def test_ncc_instantiation(self):
        """Test NCC can be instantiated."""
        loss_fn = NCC_vxm(win=[9, 9])
        assert loss_fn is not None

    def test_ncc_2d_forward(self, sample_2d_image):
        """Test NCC forward pass on 2D images."""
        loss_fn = NCC_vxm(win=[9, 9])
        img1 = sample_2d_image
        img2 = sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)

        loss = loss_fn(img1, img2)

        # Check loss is scalar
        assert loss.ndim == 0
        # NCC loss should be finite
        assert torch.isfinite(loss)

    def test_ncc_3d_forward(self, sample_3d_volume):
        """Test NCC forward pass on 3D volumes."""
        loss_fn = NCC_vxm(win=[9, 9, 9])
        vol1 = sample_3d_volume
        vol2 = sample_3d_volume + 0.1 * torch.randn_like(sample_3d_volume)

        loss = loss_fn(vol1, vol2)

        # Check loss is scalar
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_mutual_information(self, sample_2d_image):
        """Test MutualInformation loss."""
        loss_fn = MutualInformation(num_bins=32)
        img1 = sample_2d_image
        img2 = sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)

        loss = loss_fn(img1, img2)

        # Check loss is scalar
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestSegmentationLosses:
    """Test segmentation losses."""

    def test_dice_binary_instantiation(self):
        """Test binary Dice loss."""
        loss_fn = DiceLoss()
        assert loss_fn is not None

    def test_dice_multiclass_instantiation(self):
        """Test multi-class Dice loss."""
        loss_fn = DiceLoss(num_class=5)
        assert loss_fn is not None

    def test_dice_binary_forward(self):
        """Test binary Dice loss forward pass."""
        loss_fn = DiceLoss()

        # Create binary prediction and target
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()

        loss = loss_fn(pred, target)

        # Check loss is scalar
        assert loss.ndim == 0
        # Dice loss should be between 0 and 1
        assert 0 <= loss <= 1

    def test_dice_multiclass_forward(self):
        """Test multi-class Dice loss forward pass."""
        num_classes = 5
        loss_fn = DiceLoss(num_class=num_classes)

        # Create multi-class prediction and target
        pred = torch.softmax(torch.randn(2, num_classes, 64, 64), dim=1)
        target = torch.randint(0, num_classes, (2, 1, 64, 64))

        loss = loss_fn(pred, target)

        # Check loss is scalar
        assert loss.ndim == 0
        assert 0 <= loss <= 1


class TestLossFactory:
    """Test loss function factory."""

    def test_factory_ssim2d(self):
        """Test factory creates SSIM2D."""
        loss_fn = get_loss_function("ssim2d", window_size=11)
        assert isinstance(loss_fn, SSIM2D)

    def test_factory_grad(self):
        """Test factory creates Grad."""
        loss_fn = get_loss_function("grad2d", penalty="l2")
        assert isinstance(loss_fn, Grad)

    def test_factory_grad3d(self):
        """Test factory creates Grad3d."""
        loss_fn = get_loss_function("grad3d", penalty="l2")
        assert isinstance(loss_fn, Grad3d)

    def test_factory_ncc(self):
        """Test factory creates NCC."""
        loss_fn = get_loss_function("ncc", win=[9, 9])
        assert isinstance(loss_fn, NCC_vxm)

    def test_factory_dice(self):
        """Test factory creates DiceLoss."""
        loss_fn = get_loss_function("dice", num_class=5)
        assert isinstance(loss_fn, DiceLoss)

    def test_factory_invalid_name(self):
        """Test factory raises error for invalid loss name."""
        with pytest.raises((ValueError, KeyError)):
            get_loss_function("invalid_loss_name")


class TestLossNumericalStability:
    """Test numerical stability of loss functions."""

    def test_ssim_no_nan(self, sample_2d_image):
        """Test SSIM doesn't produce NaN."""
        loss_fn = SSIM2D(window_size=11)
        img1 = sample_2d_image
        img2 = sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)

        loss = loss_fn(img1, img2)

        assert not torch.isnan(loss), "NaN detected in SSIM loss"

    def test_grad_no_nan(self):
        """Test Grad doesn't produce NaN."""
        loss_fn = Grad(penalty="l2")
        flow = torch.randn(2, 2, 64, 64)

        loss = loss_fn(None, flow)

        assert not torch.isnan(loss), "NaN detected in Grad loss"

    def test_ncc_no_nan(self, sample_2d_image):
        """Test NCC doesn't produce NaN."""
        loss_fn = NCC_vxm(win=[9, 9])
        img1 = sample_2d_image
        img2 = sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)

        loss = loss_fn(img1, img2)

        assert not torch.isnan(loss), "NaN detected in NCC loss"


class TestLossCombinations:
    """Test combinations of losses (common in training)."""

    def test_registration_loss_combination(self, sample_2d_image):
        """Test typical registration loss combination."""
        ncc = NCC_vxm(win=[9, 9])
        grad = Grad(penalty="l2", loss_mult=0.1)

        img1 = sample_2d_image
        img2 = sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)
        flow = torch.randn(2, 2, 64, 64)

        similarity_loss = ncc(img1, img2)
        smoothness_loss = grad(None, flow)

        total_loss = similarity_loss + smoothness_loss

        # Check total loss is finite
        assert torch.isfinite(total_loss)
        # Check gradients can flow
        total_loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLossesCUDA:
    """Test loss functions on CUDA."""

    def test_ssim2d_cuda(self, sample_2d_image):
        """Test SSIM2D on CUDA."""
        loss_fn = SSIM2D(window_size=11).cuda()
        img1 = sample_2d_image.cuda()
        img2 = (sample_2d_image + 0.1 * torch.randn_like(sample_2d_image)).cuda()

        loss = loss_fn(img1, img2)

        assert loss.is_cuda
        assert torch.isfinite(loss)

    def test_grad_cuda(self):
        """Test Grad on CUDA."""
        loss_fn = Grad(penalty="l2")
        flow = torch.randn(2, 2, 64, 64).cuda()

        loss = loss_fn(None, flow)

        assert loss.is_cuda
        assert torch.isfinite(loss)
