"""Vision Transformer encoder for biomedical image processing.

This module implements a hierarchical Vision Transformer (ViT) encoder that serves
as an alternative to the Mamba encoder. It maintains the same interface and output
format for compatibility with existing decoders.

The architecture uses standard multi-head self-attention instead of Mamba's state-space
models while preserving the hierarchical multi-scale feature extraction.

Classes:
    TransformerBlock: Standard transformer block with multi-head self-attention
    ViTBasicLayer: Stack of transformer blocks forming one encoder stage
    ViTEncoderHiera: Hierarchical Vision Transformer encoder
"""

# Standard library imports
import math
from functools import partial
from typing import Optional

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# Local imports - reuse components from ours_mamba
from src.ours_mamba import PatchEmbed, PatchMerging, segm_init_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Implements scaled dot-product attention with multiple attention heads for
    capturing different aspects of relationships in the input sequence.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads. Default: 8
        qkv_bias: Whether to include bias in qkv projection. Default: True
        attn_drop: Attention dropout rate. Default: 0.0
        proj_drop: Output projection dropout rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (B, N, C) where N = H*W*T

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """Feed-forward network (MLP) for transformer blocks.

    Two-layer MLP with GELU activation and dropout.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features. Default: None (4x input)
        out_features: Number of output features. Default: None (same as input)
        drop: Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass through MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor with same shape as input
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Vision Transformer block with self-attention and MLP.

    Implements the standard pre-norm transformer architecture:
    x -> LayerNorm -> Attention -> Add -> LayerNorm -> MLP -> Add

    Args:
        dim: Model dimension
        num_heads: Number of attention heads. Default: 8
        mlp_ratio: Ratio of MLP hidden dim to embedding dim. Default: 4.0
        qkv_bias: Whether to add bias to QKV projection. Default: True
        drop: Dropout rate. Default: 0.0
        attn_drop: Attention dropout rate. Default: 0.0
        drop_path: Stochastic depth rate. Default: 0.0
        norm_layer: Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

    def forward(self, x):
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (B, N, C)

        Returns:
            Output tensor of shape (B, N, C)
        """
        # Self-attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ViTBasicLayer(nn.Module):
    """A basic hierarchical layer consisting of multiple transformer blocks.

    Represents one stage in the hierarchical encoder, containing a stack of
    transformer blocks and an optional downsampling layer.

    Args:
        dim: Number of feature channels
        depth: Number of blocks in this stage
        num_heads: Number of attention heads. Default: 8
        mlp_ratio: Ratio of MLP hidden dim to embedding dim. Default: 4.0
        qkv_bias: Whether to add bias to QKV projection. Default: True
        drop: Dropout rate. Default: 0.0
        attn_drop: Attention dropout rate. Default: 0.0
        drop_path: Stochastic depth rate (list or float). Default: 0.0
        norm_layer: Normalization layer. Default: nn.LayerNorm
        downsample: Downsample layer at the end. Default: None
        pat_merg_rf: Patch merging reduction factor. Default: 2
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        pat_merg_rf=2,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.pat_merg_rf = pat_merg_rf

        # Build transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # Patch merging layer for downsampling
        if downsample:
            self.downsample = downsample(
                dim=dim,
                norm_layer=norm_layer,
                reduce_factor=self.pat_merg_rf,
            )
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """Forward pass through all blocks in this layer.

        Args:
            x: Input tensor of shape (B, H*W*T, C)
            H: Height of feature map
            W: Width of feature map
            T: Depth of feature map

        Returns:
            Tuple of (x, H, W, T, x_down, Wh, Ww, Wt) where:
                - x: Output features before downsampling
                - x_down: Downsampled features (or x if no downsampling)
                - Wh, Ww, Wt: Downsampled dimensions
        """
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply downsampling if needed
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


class ViTEncoderHiera(nn.Module):
    """Hierarchical Vision Transformer encoder with multiple stages.

    Multi-stage encoder that progressively downsamples the input while increasing
    the number of feature channels. Each stage consists of multiple transformer blocks.

    This encoder is designed to be a drop-in replacement for MambaEncoderHeria,
    maintaining the same interface and output format for compatibility with existing
    decoders.

    Args:
        config: Configuration object containing:
            - img_size: Input image size [D, H, W]
            - patch_size: Size of image patches
            - in_chans: Number of input channels
            - embed_dim: Base embedding dimension
            - depths: List of depths for each stage
            - num_heads: Number of attention heads (default: 8)
            - mlp_ratio: MLP expansion ratio (default: 4.0)
            - qkv_bias: Whether to use bias in QKV projection (default: True)
            - drop_rate: Dropout rate
            - attn_drop_rate: Attention dropout rate (default: 0.0)
            - drop_path_rate: Stochastic depth rate
            - patch_norm: Whether to use norm after patch embedding
            - out_indices: Indices of output stages
            - pat_merg_rf: Patch merging reduction factor
    """

    def __init__(self, config, **kwargs):
        super().__init__()

        # Configuration
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.pat_merg_rf = config.pat_merg_rf
        self.embed_dim = config.embed_dim
        self.depths = config.depths
        self.num_layers = len(self.depths)
        self.in_chans = config.in_chans
        self.norm_layer = nn.LayerNorm if config.patch_norm else None
        self.drop_rate = config.drop_rate
        self.drop_path_rate = config.drop_path_rate
        self.out_indices = config.out_indices

        # ViT-specific parameters
        self.num_heads = getattr(config, 'num_heads', 8)
        self.mlp_ratio = getattr(config, 'mlp_ratio', 4.0)
        self.qkv_bias = getattr(config, 'qkv_bias', True)
        self.attn_drop_rate = getattr(config, 'attn_drop_rate', 0.0)

        # Patch embedding (reuse from ours_mamba)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
        )
        self.num_patches = self.patch_embed.num_patches

        # Stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]

        # Build hierarchical layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ViTBasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                pat_merg_rf=self.pat_merg_rf,
            )
            self.layers.append(layer)

        # Feature dimensions for each stage
        num_features = [int(self.embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # Add a norm layer for each output
        for i_layer in self.out_indices:
            layer = nn.LayerNorm(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        # Apply initialization
        self.apply(segm_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return set of parameters that should not have weight decay."""
        return {"pos_embed", "cls_token"}

    def get_num_layers(self):
        """Return number of encoder layers."""
        return len(self.layers)

    def forward(self, x, inference_params=None):
        """Forward pass through hierarchical encoder.

        Args:
            x: Input tensor of shape (B, C, D, H, W)
            inference_params: Parameters for inference mode (unused, for compatibility)

        Returns:
            List of output features from each stage, including input
        """
        # Store original input
        outs = [x.clone()]

        # Patch embedding
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, C, T*H*W) -> (B, T*H*W, C)

        # Process through hierarchical layers
        in_T, in_H, in_W = T, H, W
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, T, H, W, x, in_T, in_H, in_W = layer(x, in_T, in_H, in_W)

            # Add output features if this stage is in out_indices
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.contiguous().view(B, in_T, in_H, in_W, self.num_features[i])
                out = out.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
                outs.append(out)

        return outs


def _init_vit_weights(module):
    """Initialize weights for Vision Transformer.

    Uses truncated normal for linear layers and constant initialization for
    normalization layers.

    Args:
        module: Module to initialize
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv3d):
        # Fan-out initialization for conv layers
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
