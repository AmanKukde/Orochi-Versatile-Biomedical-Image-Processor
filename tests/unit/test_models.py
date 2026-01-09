"""
Unit tests for Orochi model architectures.

Tests cover:
- Model instantiation for 2D and 3D
- Forward pass functionality
- Shape consistency
- Weight loading compatibility
- Decoder variants
"""

import pytest
import torch
import torch.nn as nn
from orochi.models import (
    MambaEncoderHeria,
    PatchEmbed,
    PatchMerging,
    reg_decoder,
    fus_decoder,
    SR_decoder,
    IR_decoder,
)


class TestPatchEmbed:
    """Test PatchEmbed module for both 2D and 3D."""

    def test_patch_embed_2d(self, sample_2d_image, sample_config_2d):
        """Test 2D patch embedding."""
        patch_embed = PatchEmbed(
            img_size=sample_config_2d.img_size,
            patch_size=sample_config_2d.patch_size,
            in_chans=sample_config_2d.in_chans,
            embed_dim=sample_config_2d.embed_dim,
            dimensions=2,
        )

        output = patch_embed(sample_2d_image)

        # Check output shape
        expected_h = sample_config_2d.img_size[0] // sample_config_2d.patch_size
        expected_w = sample_config_2d.img_size[1] // sample_config_2d.patch_size
        assert output.shape == (
            2,
            sample_config_2d.embed_dim,
            expected_h,
            expected_w,
        ), f"Expected shape (2, {sample_config_2d.embed_dim}, {expected_h}, {expected_w}), got {output.shape}"

    def test_patch_embed_3d(self, sample_3d_volume, sample_config_3d):
        """Test 3D patch embedding."""
        patch_embed = PatchEmbed(
            img_size=sample_config_3d.img_size,
            patch_size=sample_config_3d.patch_size,
            in_chans=sample_config_3d.in_chans,
            embed_dim=sample_config_3d.embed_dim,
            dimensions=3,
        )

        output = patch_embed(sample_3d_volume)

        # Check output shape
        expected_d = sample_config_3d.img_size[0] // sample_config_3d.patch_size
        expected_h = sample_config_3d.img_size[1] // sample_config_3d.patch_size
        expected_w = sample_config_3d.img_size[2] // sample_config_3d.patch_size
        assert output.shape == (
            2,
            sample_config_3d.embed_dim,
            expected_d,
            expected_h,
            expected_w,
        )


class TestMambaEncoder:
    """Test MambaEncoderHeria for 2D and 3D inputs."""

    def test_encoder_2d_instantiation(self, sample_config_2d):
        """Test that 2D encoder can be instantiated."""
        encoder = MambaEncoderHeria(sample_config_2d)
        assert encoder is not None
        assert encoder.dimensions == 2

    def test_encoder_3d_instantiation(self, sample_config_3d):
        """Test that 3D encoder can be instantiated."""
        encoder = MambaEncoderHeria(sample_config_3d)
        assert encoder is not None
        assert encoder.dimensions == 3

    def test_encoder_2d_forward(self, sample_2d_image, sample_config_2d):
        """Test 2D encoder forward pass."""
        encoder = MambaEncoderHeria(sample_config_2d)
        encoder.eval()

        with torch.no_grad():
            features = encoder(sample_2d_image)

        # Should return list of features at different scales
        assert isinstance(features, list)
        assert len(features) > 0

        # Check that features are tensors
        for feat in features:
            assert isinstance(feat, torch.Tensor)
            assert feat.shape[0] == sample_2d_image.shape[0]  # Batch size preserved

    def test_encoder_3d_forward(self, sample_3d_volume, sample_config_3d):
        """Test 3D encoder forward pass."""
        encoder = MambaEncoderHeria(sample_config_3d)
        encoder.eval()

        with torch.no_grad():
            features = encoder(sample_3d_volume)

        # Should return list of features at different scales
        assert isinstance(features, list)
        assert len(features) > 0

        # Check that features are tensors
        for feat in features:
            assert isinstance(feat, torch.Tensor)
            assert feat.shape[0] == sample_3d_volume.shape[0]  # Batch size preserved

    def test_encoder_gradient_flow(self, sample_2d_image, sample_config_2d):
        """Test that gradients flow through the encoder."""
        encoder = MambaEncoderHeria(sample_config_2d)
        encoder.train()

        # Enable gradients for input
        sample_2d_image.requires_grad = True

        features = encoder(sample_2d_image)

        # Create a dummy loss from features
        loss = sum(f.sum() for f in features)
        loss.backward()

        # Check that gradients exist
        assert sample_2d_image.grad is not None
        assert not torch.isnan(sample_2d_image.grad).any()


class TestDecoders:
    """Test various decoder architectures."""

    def test_reg_decoder_2d(self, sample_config_2d):
        """Test 2D registration decoder."""
        decoder = reg_decoder(sample_config_2d)
        assert decoder is not None

        # Create dummy features
        feat1 = torch.randn(2, 96, 16, 16)
        feat2 = torch.randn(2, 192, 8, 8)
        features = [feat1, feat2]

        with torch.no_grad():
            flow = decoder(features)

        # Check flow shape (should be 2 channels for 2D flow)
        assert flow.shape[0] == 2  # Batch size
        assert flow.shape[1] == 2  # 2D flow (x, y)

    def test_reg_decoder_3d(self, sample_config_3d):
        """Test 3D registration decoder."""
        decoder = reg_decoder(sample_config_3d)
        assert decoder is not None

        # Create dummy features
        feat1 = torch.randn(2, 96, 8, 16, 16)
        feat2 = torch.randn(2, 192, 4, 8, 8)
        features = [feat1, feat2]

        with torch.no_grad():
            flow = decoder(features)

        # Check flow shape (should be 3 channels for 3D flow)
        assert flow.shape[0] == 2  # Batch size
        assert flow.shape[1] == 3  # 3D flow (x, y, z)

    def test_sr_decoder_2d(self, sample_config_2d):
        """Test 2D super-resolution decoder."""
        decoder = SR_decoder(sample_config_2d)
        assert decoder is not None

        # Create dummy features
        feat1 = torch.randn(2, 96, 16, 16)
        features = [feat1]

        with torch.no_grad():
            output = decoder(features)

        # Check output shape
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 1  # Single channel output

    def test_fus_decoder_2d(self, sample_config_2d):
        """Test 2D fusion decoder."""
        decoder = fus_decoder(sample_config_2d)
        assert decoder is not None

        # Create dummy features
        feat1 = torch.randn(2, 96, 16, 16)
        features = [feat1]

        with torch.no_grad():
            output = decoder(features)

        # Check output shape
        assert output.shape[0] == 2  # Batch size
        assert output.ndim == 4  # [B, C, H, W]


class TestModelStateDict:
    """Test that model state dicts maintain compatibility."""

    def test_2d_state_dict_keys(self, sample_config_2d):
        """Test 2D model state dict has expected keys."""
        encoder = MambaEncoderHeria(sample_config_2d)
        state_dict = encoder.state_dict()

        # Check for essential keys
        assert any("patch_embed" in key for key in state_dict.keys())
        assert any("layers" in key for key in state_dict.keys())

    def test_3d_state_dict_keys(self, sample_config_3d):
        """Test 3D model state dict has expected keys."""
        encoder = MambaEncoderHeria(sample_config_3d)
        state_dict = encoder.state_dict()

        # Check for essential keys
        assert any("patch_embed" in key for key in state_dict.keys())
        assert any("layers" in key for key in state_dict.keys())

    def test_state_dict_loading(self, sample_config_2d):
        """Test that state dict can be loaded."""
        encoder1 = MambaEncoderHeria(sample_config_2d)
        state_dict = encoder1.state_dict()

        # Create new model and load weights
        encoder2 = MambaEncoderHeria(sample_config_2d)
        encoder2.load_state_dict(state_dict)

        # Verify weights are the same
        for (name1, param1), (name2, param2) in zip(
            encoder1.named_parameters(), encoder2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


class TestModelNumericalStability:
    """Test numerical stability of models."""

    def test_no_nan_in_forward_2d(self, sample_2d_image, sample_config_2d):
        """Test that 2D forward pass doesn't produce NaNs."""
        encoder = MambaEncoderHeria(sample_config_2d)
        encoder.eval()

        with torch.no_grad():
            features = encoder(sample_2d_image)

        # Check no NaNs in output
        for feat in features:
            assert not torch.isnan(feat).any(), "NaN detected in feature output"

    def test_no_inf_in_forward_2d(self, sample_2d_image, sample_config_2d):
        """Test that 2D forward pass doesn't produce Infs."""
        encoder = MambaEncoderHeria(sample_config_2d)
        encoder.eval()

        with torch.no_grad():
            features = encoder(sample_2d_image)

        # Check no Infs in output
        for feat in features:
            assert not torch.isinf(feat).any(), "Inf detected in feature output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestModelCUDA:
    """Test model behavior on CUDA devices."""

    def test_model_to_cuda(self, sample_config_2d):
        """Test that model can be moved to CUDA."""
        encoder = MambaEncoderHeria(sample_config_2d)
        encoder = encoder.cuda()

        # Check that parameters are on CUDA
        for param in encoder.parameters():
            assert param.is_cuda

    def test_forward_on_cuda(self, sample_2d_image, sample_config_2d):
        """Test forward pass on CUDA."""
        encoder = MambaEncoderHeria(sample_config_2d).cuda()
        sample_2d_image = sample_2d_image.cuda()

        with torch.no_grad():
            features = encoder(sample_2d_image)

        # Check outputs are on CUDA
        for feat in features:
            assert feat.is_cuda
