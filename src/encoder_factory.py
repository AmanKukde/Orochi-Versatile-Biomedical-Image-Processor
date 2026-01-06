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
    model_name: Optional[str] = None,
    use_timm: bool = False,
    **kwargs
) -> nn.Module:
    """Create encoder from HuggingFace model hub or TIMM.

    Supports both HuggingFace transformers and TIMM (PyTorch Image Models).
    For biomedical 3D ViT models, use HuggingFace.
    For 2D pretrained models, use TIMM with 2D→3D inflation.

    Args:
        config: Model configuration
        model_name: Model identifier (HuggingFace or TIMM)
                   If None, uses config.hf_model_name or config.pretrained_encoder_path
        use_timm: Use TIMM instead of HuggingFace (default: False)
        **kwargs: Additional arguments

    Returns:
        HuggingFace/TIMM encoder wrapped for compatibility

    Examples:
        # HuggingFace biomedical models
        model_name = "microsoft/swin-tiny-patch4-window7-224"
        model_name = "facebook/deit-base-patch16-224"

        # TIMM models
        model_name = "vit_base_patch16_224"  # with use_timm=True
    """
    # Get model name from config if not provided
    if model_name is None:
        model_name = getattr(config, 'hf_model_name', None)
        if model_name is None:
            # Try pretrained_encoder_path as HF model name
            model_name = getattr(config, 'pretrained_encoder_path', 'google/vit-base-patch16-224')
            if isinstance(model_name, Path):
                model_name = str(model_name)

    print(f"Loading pretrained model: {model_name}")
    print(f"Source: {'TIMM' if use_timm else 'HuggingFace'}")

    if use_timm:
        encoder = _load_timm_encoder(model_name, config, **kwargs)
    else:
        encoder = _load_huggingface_encoder(model_name, config, **kwargs)

    return encoder


def _load_huggingface_encoder(model_name: str, config, **kwargs) -> nn.Module:
    """Load encoder from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier
        config: Model configuration
        **kwargs: Additional arguments

    Returns:
        Wrapped HuggingFace encoder
    """
    try:
        from transformers import AutoModel, AutoConfig, AutoImageProcessor
    except ImportError:
        raise ImportError(
            "transformers library required for HuggingFace models. "
            "Install with: pip install transformers"
        )

    print(f"📦 Loading from HuggingFace Hub: {model_name}")

    # Load model and config
    hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Get hidden size
    hidden_size = getattr(hf_config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = getattr(hf_config, 'embed_dim', None)
    if hidden_size is None:
        hidden_size = getattr(hf_config, 'd_model', 768)

    print(f"✓ Loaded HuggingFace model")
    print(f"  Model type: {hf_config.model_type}")
    print(f"  Hidden size: {hidden_size}")

    # Wrap in compatibility layer
    encoder = _wrap_huggingface_encoder(hf_model, hf_config, config)

    return encoder


def _load_timm_encoder(model_name: str, config, pretrained: bool = True, **kwargs) -> nn.Module:
    """Load encoder from TIMM (PyTorch Image Models).

    Args:
        model_name: TIMM model name
        config: Model configuration
        pretrained: Load pretrained weights (default: True)
        **kwargs: Additional arguments

    Returns:
        Wrapped TIMM encoder
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm library required for TIMM models. "
            "Install with: pip install timm"
        )

    print(f"📦 Loading from TIMM: {model_name}")

    # Load model
    timm_model = timm.create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,  # Return hierarchical features
        out_indices=config.out_indices,
        **kwargs
    )

    # Get feature dimensions
    feature_info = timm_model.feature_info
    feature_dims = [info['num_chs'] for info in feature_info]

    print(f"✓ Loaded TIMM model")
    print(f"  Model: {model_name}")
    print(f"  Feature dimensions: {feature_dims}")
    print(f"  Pretrained: {pretrained}")

    # Wrap in compatibility layer
    encoder = _wrap_timm_encoder(timm_model, config, feature_dims)

    return encoder


def _wrap_huggingface_encoder(hf_model, hf_config, config) -> nn.Module:
    """Wrap HuggingFace model to match our encoder interface.

    This adapter ensures HuggingFace models produce hierarchical features
    compatible with our decoders. Handles 2D→3D conversion and hierarchical
    feature extraction.
    """
    class HuggingFaceEncoderWrapper(nn.Module):
        """Wrapper to adapt HuggingFace models to our 3D hierarchical interface."""

        def __init__(self, hf_model, hf_config, config):
            super().__init__()
            self.hf_model = hf_model
            self.hf_config = hf_config
            self.config = config

            # Get embedding dimension
            self.embed_dim = getattr(hf_config, 'hidden_size', None)
            if self.embed_dim is None:
                self.embed_dim = getattr(hf_config, 'embed_dim', 768)

            # Create hierarchical feature dimensions
            # Match expected decoder dimensions
            num_stages = len(config.out_indices)
            base_dim = config.embed_dim
            self.num_features = [base_dim * (2 ** i) for i in range(num_stages)]

            # Create 3D patch embedding (2D→3D inflation)
            self.patch_embed_3d = nn.Conv3d(
                config.in_chans,
                self.embed_dim,
                kernel_size=(config.patch_size, config.patch_size, config.patch_size),
                stride=(config.patch_size, config.patch_size, config.patch_size)
            )

            # Initialize 3D conv from 2D weights if possible
            self._inflate_2d_to_3d()

            # Create projection layers for hierarchical outputs
            self.projections = nn.ModuleList()
            for i, feat_dim in enumerate(self.num_features):
                if self.embed_dim != feat_dim:
                    # Need projection to match decoder expectations
                    self.projections.append(nn.Conv3d(self.embed_dim, feat_dim, 1))
                else:
                    self.projections.append(nn.Identity())

            print(f"  HuggingFace wrapper created:")
            print(f"    Input: 3D volumes (B, {config.in_chans}, D, H, W)")
            print(f"    Embed dim: {self.embed_dim}")
            print(f"    Output features: {self.num_features}")

        def _inflate_2d_to_3d(self):
            """Inflate 2D pretrained weights to 3D."""
            # Try to get 2D patch embedding weights from HF model
            try:
                if hasattr(self.hf_model, 'embeddings'):
                    if hasattr(self.hf_model.embeddings, 'patch_embeddings'):
                        patch_embed_2d = self.hf_model.embeddings.patch_embeddings.projection
                        if isinstance(patch_embed_2d, nn.Conv2d):
                            # Inflate 2D conv to 3D
                            with torch.no_grad():
                                weight_2d = patch_embed_2d.weight  # (out, in, h, w)
                                # Repeat along depth dimension and average
                                weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, self.patch_embed_3d.kernel_size[0], 1, 1)
                                weight_3d = weight_3d / self.patch_embed_3d.kernel_size[0]
                                self.patch_embed_3d.weight.copy_(weight_3d)

                                if patch_embed_2d.bias is not None:
                                    self.patch_embed_3d.bias.copy_(patch_embed_2d.bias)

                            print(f"  ✓ Inflated 2D→3D patch embedding weights")
            except Exception as e:
                print(f"  ⚠️  Could not inflate 2D weights: {e}")
                print(f"     Using random initialization for 3D patch embedding")

        def forward(self, x):
            """Forward pass producing hierarchical features.

            Args:
                x: Input tensor (B, C, D, H, W)

            Returns:
                List of feature maps at different scales for each out_indices
            """
            B, C, D, H, W = x.shape

            # 3D patch embedding
            x_3d = self.patch_embed_3d(x)  # (B, embed_dim, D', H', W')
            _, E, D_p, H_p, W_p = x_3d.shape

            # Process each depth slice through HuggingFace model
            # This is a simplified approach - process 2D slices independently
            features_list = []

            for d in range(D_p):
                # Extract 2D slice
                x_slice = x_3d[:, :, d, :, :]  # (B, E, H', W')

                # Reshape for HuggingFace model (B, H', W', E) or (B, HW, E)
                B_s, E_s, H_s, W_s = x_slice.shape
                x_flat = x_slice.flatten(2).transpose(1, 2)  # (B, HW, E)

                # Pass through HuggingFace encoder
                try:
                    # Try different HF model interfaces
                    if hasattr(self.hf_model, 'encoder'):
                        outputs = self.hf_model.encoder(x_flat, return_dict=True)
                    else:
                        outputs = self.hf_model(x_flat, return_dict=True)

                    # Extract last hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden = outputs.last_hidden_state  # (B, HW, E)
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hidden = outputs.hidden_states[-1]
                    else:
                        # Fallback: use outputs directly if it's a tensor
                        hidden = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

                    # Reshape back to spatial
                    hidden = hidden.transpose(1, 2).reshape(B_s, E_s, H_s, W_s)  # (B, E, H', W')
                    features_list.append(hidden)

                except Exception as e:
                    print(f"Warning: HuggingFace model forward failed: {e}")
                    # Fallback: use input as output
                    features_list.append(x_slice)

            # Stack depth slices back
            features_3d = torch.stack(features_list, dim=2)  # (B, E, D', H', W')

            # Create hierarchical outputs by projecting to different dimensions
            hierarchical_features = []
            for proj in self.projections:
                feat = proj(features_3d)
                hierarchical_features.append(feat)

            return hierarchical_features

    return HuggingFaceEncoderWrapper(hf_model, hf_config, config)


def _wrap_timm_encoder(timm_model, config, feature_dims) -> nn.Module:
    """Wrap TIMM model to match our 3D hierarchical interface.

    Args:
        timm_model: TIMM model with features_only=True
        config: Model configuration
        feature_dims: List of feature dimensions from TIMM model

    Returns:
        Wrapped TIMM encoder for 3D processing
    """
    class TIMMEncoderWrapper(nn.Module):
        """Wrapper to adapt TIMM models to our 3D interface."""

        def __init__(self, timm_model, config, feature_dims):
            super().__init__()
            self.timm_model = timm_model
            self.config = config
            self.num_features = feature_dims
            self.embed_dim = feature_dims[-1]  # Use last feature dim

            print(f"  TIMM wrapper created:")
            print(f"    Input: 3D volumes (B, {config.in_chans}, D, H, W)")
            print(f"    Output features: {self.num_features}")
            print(f"    Processing: Depth-wise 2D slices")

        def forward(self, x):
            """Forward pass processing each depth slice.

            Args:
                x: Input tensor (B, C, D, H, W)

            Returns:
                List of hierarchical feature maps
            """
            B, C, D, H, W = x.shape

            # Process each depth slice
            all_features = [[] for _ in range(len(self.num_features))]

            for d in range(D):
                x_slice = x[:, :, d, :, :]  # (B, C, H, W)

                # Forward through TIMM model
                slice_features = self.timm_model(x_slice)  # List of features

                # Accumulate features for each level
                for level, feat in enumerate(slice_features):
                    all_features[level].append(feat)

            # Stack depth dimension
            hierarchical_features = []
            for level_features in all_features:
                # Stack (B, C, H, W) tensors along depth
                feat_3d = torch.stack(level_features, dim=2)  # (B, C, D, H, W)
                hierarchical_features.append(feat_3d)

            return hierarchical_features

    return TIMMEncoderWrapper(timm_model, config, feature_dims)


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
