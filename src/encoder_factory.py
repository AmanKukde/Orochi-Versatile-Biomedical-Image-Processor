"""Encoder Factory for modular encoder swapping.

This module provides a unified interface for creating different encoder backends
including Mamba, ViT, 3DINO-ViT, and HuggingFace pretrained models.

Usage:
    from encoder_factory import create_encoder

    # Create Mamba encoder (default)
    encoder = create_encoder(config, encoder_type='mamba')

    # Create ViT encoder
    encoder = create_encoder(config, encoder_type='vit')

    # Create 3DINO-ViT with pretrained weights
    encoder = create_encoder(config, encoder_type='3dino',
                            pretrained_path='path/to/checkpoint.pth')

    # Create HuggingFace ViT
    encoder = create_encoder(config, encoder_type='huggingface',
                            model_name='google/vit-base-patch16-224')
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path


def create_encoder(
    config,
    encoder_type: str = 'mamba',
    pretrained_path: Optional[str] = None,
    freeze: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function to create encoder based on type.

    Args:
        config: Model configuration object
        encoder_type: Type of encoder ('mamba', 'vit', '3dino', 'huggingface')
        pretrained_path: Path to pretrained weights (optional)
        freeze: Whether to freeze encoder weights
        **kwargs: Additional encoder-specific arguments

    Returns:
        Encoder module

    Raises:
        ValueError: If encoder_type is not supported
    """
    encoder_type = encoder_type.lower()

    if encoder_type == 'mamba':
        encoder = _create_mamba_encoder(config, **kwargs)
    elif encoder_type == 'vit':
        encoder = _create_vit_encoder(config, **kwargs)
    elif encoder_type == '3dino':
        encoder = _create_3dino_encoder(config, pretrained_path, **kwargs)
    elif encoder_type == 'huggingface':
        encoder = _create_huggingface_encoder(config, **kwargs)
    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Supported types: 'mamba', 'vit', '3dino', 'huggingface'"
        )

    # Load pretrained weights if provided
    if pretrained_path and encoder_type != '3dino':  # 3dino loads in _create
        encoder = _load_pretrained_weights(encoder, pretrained_path)

    # Freeze if requested
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        print(f"❄️  {encoder_type.upper()} encoder frozen")

    return encoder


def _create_mamba_encoder(config, **kwargs) -> nn.Module:
    """Create Mamba encoder."""
    from ours_mamba import MambaEncoderHeria

    print(f"Creating Mamba encoder (embed_dim={config.embed_dim})")
    encoder = MambaEncoderHeria(config, **kwargs)

    return encoder


def _create_vit_encoder(config, **kwargs) -> nn.Module:
    """Create standard ViT encoder."""
    from vit_encoder import ViTEncoderHiera

    print(f"Creating ViT encoder (embed_dim={config.embed_dim})")
    encoder = ViTEncoderHiera(config, **kwargs)

    return encoder


def _create_3dino_encoder(
    config,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """Create 3DINO-ViT encoder with pretrained weights.

    Args:
        config: Model configuration
        pretrained_path: Path to 3DINO checkpoint
        **kwargs: Additional arguments

    Returns:
        3DINO-ViT encoder
    """
    from vit_encoder import ViTEncoderHiera

    # Create encoder with 3DINO dimensions
    encoder_dim = getattr(config, 'encoder_dim', 384)  # 3DINO default
    print(f"Creating 3DINO-ViT encoder (embed_dim={encoder_dim})")

    # Temporarily override embed_dim for encoder creation
    original_embed_dim = config.embed_dim
    config.embed_dim = encoder_dim

    encoder = ViTEncoderHiera(config, **kwargs)

    # Restore original embed_dim
    config.embed_dim = original_embed_dim

    # Load pretrained weights
    if pretrained_path:
        encoder = _load_3dino_weights(encoder, pretrained_path)
    else:
        print("⚠️  Warning: No pretrained_path provided for 3DINO encoder")
        print("   Encoder will use random initialization")

    return encoder


def _create_huggingface_encoder(
    config,
    model_name: str = 'google/vit-base-patch16-224',
    **kwargs
) -> nn.Module:
    """Create encoder from HuggingFace model hub.

    Args:
        config: Model configuration
        model_name: HuggingFace model identifier
        **kwargs: Additional arguments

    Returns:
        HuggingFace ViT encoder wrapped for compatibility
    """
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError:
        raise ImportError(
            "transformers library required for HuggingFace models. "
            "Install with: pip install transformers"
        )

    print(f"Loading HuggingFace model: {model_name}")

    # Load model
    hf_model = AutoModel.from_pretrained(model_name)
    hf_config = AutoConfig.from_pretrained(model_name)

    # Wrap in compatibility layer
    encoder = _wrap_huggingface_encoder(hf_model, hf_config, config)

    print(f"✓ Loaded HuggingFace encoder (hidden_size={hf_config.hidden_size})")

    return encoder


def _wrap_huggingface_encoder(hf_model, hf_config, config) -> nn.Module:
    """Wrap HuggingFace model to match our encoder interface.

    This adapter ensures HuggingFace models produce hierarchical features
    compatible with our decoders.
    """
    class HuggingFaceEncoderWrapper(nn.Module):
        """Wrapper to adapt HuggingFace models to our interface."""

        def __init__(self, hf_model, hf_config, config):
            super().__init__()
            self.hf_model = hf_model
            self.hf_config = hf_config
            self.config = config

            # Store output dimension
            self.embed_dim = hf_config.hidden_size

            # Create feature projection layers for hierarchical output
            # (if needed for compatibility)
            self.num_features = [self.embed_dim] * len(config.out_indices)

        def forward(self, x):
            """Forward pass producing hierarchical features.

            Args:
                x: Input tensor (B, C, D, H, W)

            Returns:
                List of feature maps at different scales
            """
            # Note: This is a simplified wrapper
            # Full implementation needs proper 3D handling and hierarchical features
            # TODO: Implement proper 3D patch embedding and hierarchical extraction

            raise NotImplementedError(
                "HuggingFace encoder wrapper needs full implementation. "
                "Current version is a placeholder for the architecture."
            )

    return HuggingFaceEncoderWrapper(hf_model, hf_config, config)


def _load_3dino_weights(encoder: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load 3DINO pretrained weights into encoder.

    Args:
        encoder: Encoder module
        checkpoint_path: Path to 3DINO checkpoint

    Returns:
        Encoder with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"3DINO checkpoint not found: {checkpoint_path}")

    print(f"Loading 3DINO weights from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict (handle different checkpoint formats)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'encoder.' prefix if present
    state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}

    # Load weights (strict=False to allow for architecture differences)
    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"   - {key}")

    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"   - {key}")

    print(f"✓ Loaded 3DINO weights successfully")

    return encoder


def _load_pretrained_weights(encoder: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load pretrained weights from checkpoint.

    Args:
        encoder: Encoder module
        checkpoint_path: Path to checkpoint file

    Returns:
        Encoder with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading pretrained weights from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract encoder state dict
    if 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    elif 'model_state_dict' in checkpoint:
        # Try to extract encoder weights from full model
        state_dict = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('encoder.')
        }
    else:
        state_dict = checkpoint

    # Load weights
    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"⚠️  Warning: {len(missing_keys)} missing keys")
    if unexpected_keys:
        print(f"⚠️  Warning: {len(unexpected_keys)} unexpected keys")

    print(f"✓ Loaded pretrained weights")

    return encoder


def get_encoder_output_dim(encoder: nn.Module) -> int:
    """Get the output dimension of an encoder.

    Args:
        encoder: Encoder module

    Returns:
        Output feature dimension
    """
    if hasattr(encoder, 'embed_dim'):
        return encoder.embed_dim
    elif hasattr(encoder, 'num_features'):
        # Return final stage dimension
        return encoder.num_features[-1]
    else:
        raise AttributeError(
            f"Cannot determine output dimension for encoder type: {type(encoder)}"
        )


def print_encoder_info(encoder: nn.Module):
    """Print encoder information.

    Args:
        encoder: Encoder module
    """
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"\n{'='*80}")
    print(f"ENCODER INFO: {type(encoder).__name__}")
    print(f"{'='*80}")
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")

    if hasattr(encoder, 'embed_dim'):
        print(f"Embedding dimension: {encoder.embed_dim}")

    if hasattr(encoder, 'num_features'):
        print(f"Feature dimensions: {encoder.num_features}")

    print(f"{'='*80}\n")
