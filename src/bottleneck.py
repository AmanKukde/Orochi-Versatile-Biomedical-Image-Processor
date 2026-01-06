"""Bottleneck projection module for encoder-decoder dimension adaptation.

This module provides a trainable projection layer to adapt between
different embedding dimensions (e.g., 3DINO-ViT encoder → Mamba decoders).

Training strategy:
1. Phase 1: Freeze encoder + decoders, train only bottleneck
2. Phase 2: Unfreeze encoder, keep decoders frozen, train encoder + bottleneck
"""

import torch
import torch.nn as nn


class BottleneckFFN(nn.Module):
    """Feed-forward bottleneck for dimension adaptation.

    Projects encoder outputs to decoder input dimensions using a
    two-layer MLP with layer normalization and activation.

    Args:
        encoder_dim: Input dimension from encoder
        decoder_dim: Output dimension for decoder
        hidden_ratio: Hidden dimension = max(encoder_dim, decoder_dim) * hidden_ratio
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: GELU)
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        hidden_ratio: float = 2.0,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # Hidden dimension is larger for better representation
        hidden_dim = int(max(encoder_dim, decoder_dim) * hidden_ratio)

        # Layer normalization for input
        self.norm1 = nn.LayerNorm(encoder_dim)

        # Two-layer MLP
        self.fc1 = nn.Linear(encoder_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, decoder_dim)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer normalization for output
        self.norm2 = nn.LayerNorm(decoder_dim)

        # Residual connection if dimensions match (for fine-tuning stability)
        self.use_residual = (encoder_dim == decoder_dim)
        if self.use_residual:
            self.residual_scale = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bottleneck.

        Args:
            x: Input tensor of shape (B, C, D, H, W) or (B, N, C)

        Returns:
            Projected tensor with decoder_dim
        """
        # Store input for potential residual
        identity = x

        # Normalize input
        x = self.norm1(x)

        # First projection
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Second projection
        x = self.fc2(x)
        x = self.dropout(x)

        # Residual connection (if dimensions match)
        if self.use_residual:
            x = x + self.residual_scale * identity

        # Output normalization
        x = self.norm2(x)

        return x

    def extra_repr(self) -> str:
        """String representation for print."""
        return f"encoder_dim={self.encoder_dim}, decoder_dim={self.decoder_dim}"


class HierarchicalBottleneck(nn.Module):
    """Hierarchical bottleneck for multi-scale feature adaptation.

    Applies different projection layers to each hierarchical level
    of the encoder output. Useful when encoder and decoder have
    different feature dimensions at each level.

    Args:
        encoder_dims: List of encoder output dimensions per level
        decoder_dims: List of decoder input dimensions per level
        hidden_ratio: Hidden dimension ratio for each bottleneck
        dropout: Dropout probability
        activation: Activation function
    """

    def __init__(
        self,
        encoder_dims: list,
        decoder_dims: list,
        hidden_ratio: float = 2.0,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        assert len(encoder_dims) == len(decoder_dims), \
            "encoder_dims and decoder_dims must have same length"

        self.num_levels = len(encoder_dims)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Create bottleneck for each level
        self.bottlenecks = nn.ModuleList([
            BottleneckFFN(
                enc_dim, dec_dim,
                hidden_ratio=hidden_ratio,
                dropout=dropout,
                activation=activation
            )
            for enc_dim, dec_dim in zip(encoder_dims, decoder_dims)
        ])

    def forward(self, features: list) -> list:
        """Forward pass through all bottlenecks.

        Args:
            features: List of encoder features, one per level

        Returns:
            List of projected features, one per level
        """
        assert len(features) == self.num_levels, \
            f"Expected {self.num_levels} features, got {len(features)}"

        projected = [
            bottleneck(feat)
            for bottleneck, feat in zip(self.bottlenecks, features)
        ]

        return projected

    def freeze(self):
        """Freeze all bottleneck parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all bottleneck parameters."""
        for param in self.parameters():
            param.requires_grad = True


def count_bottleneck_params(encoder_dim: int, decoder_dim: int, hidden_ratio: float = 2.0) -> int:
    """Calculate number of parameters in a bottleneck."""
    hidden_dim = int(max(encoder_dim, decoder_dim) * hidden_ratio)

    # Layer norms
    ln_params = 2 * encoder_dim + 2 * decoder_dim

    # Linear layers (weights + biases)
    fc1_params = encoder_dim * hidden_dim + hidden_dim
    fc2_params = hidden_dim * decoder_dim + decoder_dim

    # Residual scale (if dims match)
    residual_params = 1 if encoder_dim == decoder_dim else 0

    total = ln_params + fc1_params + fc2_params + residual_params

    return total


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Bottleneck FFN Module Test")
    print("=" * 70)

    # Single bottleneck
    encoder_dim = 384  # 3DINO-ViT
    decoder_dim = 128  # Mamba decoder
    batch_size = 2
    seq_len = 1024  # Flattened spatial dimensions

    bottleneck = BottleneckFFN(encoder_dim, decoder_dim)
    print(f"\nBottleneck: {encoder_dim} → {decoder_dim}")
    print(f"Parameters: {sum(p.numel() for p in bottleneck.parameters()):,}")
    print(f"Calculated: {count_bottleneck_params(encoder_dim, decoder_dim):,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, encoder_dim)
    y = bottleneck(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (batch_size, seq_len, decoder_dim), "Shape mismatch!"

    # Hierarchical bottleneck
    print("\n" + "=" * 70)
    print("Hierarchical Bottleneck Test")
    print("=" * 70)

    encoder_dims = [96, 192, 384, 768]  # ViT hierarchical dims
    decoder_dims = [96, 192, 384, 768]  # Mamba hierarchical dims

    hierarchical = HierarchicalBottleneck(encoder_dims, decoder_dims)
    print(f"\nHierarchical bottleneck:")
    print(f"Levels: {len(encoder_dims)}")
    total_params = sum(p.numel() for p in hierarchical.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    features = [
        torch.randn(batch_size, 64**3 // (4**i)**3, dim)
        for i, dim in enumerate(encoder_dims)
    ]
    projected = hierarchical(features)

    print("\nForward pass:")
    for i, (feat, proj) in enumerate(zip(features, projected)):
        print(f"  Level {i}: {feat.shape} → {proj.shape}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
