"""Mamba-based U-Net architecture for biomedical image processing.

This module implements a hierarchical encoder-decoder architecture based on the Mamba
state-space model for various biomedical image processing tasks including:
- Image registration
- Image fusion
- Super-resolution (SR)
- Isotropic restoration (IR)

The architecture uses patch embedding, hierarchical encoding with Mamba blocks,
and task-specific decoders for different image processing objectives.

Classes:
    PatchEmbed: Convert 3D images to patch embeddings
    PatchMerging: Downsample feature maps by merging patches
    Block: Mamba transformer block with residual connections
    BasicLayer: Stack of Mamba blocks forming one encoder stage
    MambaEncoderHeria: Hierarchical encoder with multiple stages
    ConvReLU: Standard convolutional block with ReLU activation
    ConvReLULight: Depthwise separable convolutional block
    ConvDecoderBlock: Decoder block with upsampling and skip connections
    ConvDecoder: Multi-stage decoder for feature reconstruction
    Head: Final convolutional layer for output prediction
    reg_decoder: Decoder for registration tasks
    fus_decoder: Decoder for fusion tasks
    SR_decoder: Decoder for super-resolution tasks
    IR_decoder: Decoder for isotropic restoration tasks
    SpatialTransformer: Apply spatial transformations using flow fields
    MambaULight: Main model combining encoder and multiple task-specific decoders
"""

# Standard library imports
import math
import random
from functools import partial
from typing import Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, to_3tuple, trunc_normal_
from timm.models.vision_transformer import _load_weights
from torch import Tensor
from torch.distributions.normal import Normal

# Local imports
import losses

# Try to import Mamba SSM (optional - only needed for MambaULight model)
try:
    from mamba_ssm.modules.mamba2 import Mamba2
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    Mamba2 = None
    print("⚠️  Warning: mamba_ssm not installed. MambaULight model will not be available.")
    print("   Decoders and utility classes can still be used with ViT.")

# Try to import optional Triton-optimized operations
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class PatchEmbed(nn.Module):
    """Convert 3D images to patch embeddings.

    Divides the input volume into non-overlapping 3D patches and projects them
    to an embedding space using 3D convolution.

    Args:
        img_size: Size of input image [D, H, W]
        patch_size: Size of each patch. Default: 4
        in_chans: Number of input image channels. Default: 3
        embed_dim: Number of linear projection output channels. Default: 96
        norm_layer: Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
    ):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.num_patches = (
            (img_size[2] // patch_size[2])
            * (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
        )
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward pass with automatic padding.

        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Patch embeddings of shape (B, embed_dim, D', H', W')
        """
        _, _, T, H, W = x.size()
        # Pad input if dimensions are not divisible by patch size
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wt, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wt, Wh, Ww)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer for downsampling feature maps.

    Reduces spatial dimensions by a factor of 2 in each dimension by merging
    neighboring patches and projecting to higher dimensional space.

    Args:
        dim: Number of input channels
        norm_layer: Normalization layer. Default: nn.LayerNorm
        reduce_factor: Channel dimension multiplication factor. Default: 2
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, reduce_factor * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """Forward pass merging 2x2x2 neighborhoods.

        Args:
            x: Input tensor of shape (B, H*W*T, C)
            H: Height of feature map
            W: Width of feature map
            T: Depth of feature map

        Returns:
            Merged tensor of shape (B, H/2*W/2*T/2, reduce_factor*C)
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert (
            H % 2 == 0 and W % 2 == 0 and T % 2 == 0
        ), f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # Pad input if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        # Extract 8 sub-volumes from 2x2x2 neighborhood
        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Block(nn.Module):
    """Mamba transformer block with residual connections.

    This block wraps a Mamba mixer with layer normalization and residual connections.
    The structure differs from standard prenorm Transformer blocks:
    - Standard: LN -> MHA/MLP -> Add
    - This block: Add -> LN -> Mixer (for performance, allows fused operations)

    Args:
        dim: Model dimension
        mixer_cls: Mixer class (typically Mamba2)
        norm_cls: Normalization class. Default: nn.LayerNorm
        fused_add_norm: Whether to use fused add+norm operations. Default: False
        residual_in_fp32: Whether to keep residuals in FP32. Default: False
        drop_path: Drop path rate. Default: 0.0
    """

    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        use_checkpoint=False,
    ):
        """Pass input through the Mamba block.

        Args:
            hidden_states: Input sequence (required)
            residual: Residual connection. If None, uses hidden_states
            inference_params: Parameters for inference mode
            use_checkpoint: Whether to use gradient checkpointing

        Returns:
            Tuple of (hidden_states, residual) for next block
        """
        if not self.fused_add_norm:
            residual = (
                (residual + self.drop_path(hidden_states))
                if residual is not None
                else hidden_states
            )
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            hidden_states, residual = fused_add_norm_fn(
                (
                    hidden_states
                    if residual is None
                    else self.drop_path(hidden_states)
                ),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(
                self.mixer, hidden_states, inference_params
            )
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate KV cache for inference."""
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


class BasicLayer(nn.Module):
    """A basic hierarchical layer consisting of multiple Mamba blocks.

    Represents one stage in the hierarchical encoder, containing a stack of
    Mamba blocks and an optional downsampling layer.

    Args:
        dim: Number of feature channels
        depth: Number of blocks in this stage
        window_size: Local window size. Default: (7, 7, 7)
        drop: Dropout rate. Default: 0.0
        drop_path: Stochastic depth rate. Default: 0.0
        norm_layer: Normalization layer. Default: nn.LayerNorm
        downsample: Downsample layer at the end of the layer. Default: None
        use_checkpoint: Whether to use gradient checkpointing. Default: False
        pat_merg_rf: Patch merging reduction factor. Default: 2
        fused_add_norm: Whether to use fused add+norm. Default: True
        residual_in_fp32: Keep residuals in FP32. Default: True
        ssm_cfg: SSM configuration dict. Default: None
        norm_epsilon: Epsilon for normalization. Default: 1e-5
        rms_norm: Whether to use RMSNorm. Default: True
    """

    def __init__(
        self,
        dim,
        depth,
        window_size=(7, 7, 7),
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        pat_merg_rf=2,
        fused_add_norm=True,
        residual_in_fp32=True,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=True,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (
            window_size[0] // 2,
            window_size[1] // 2,
            window_size[2] // 2,
        )
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf

        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.norm_layer = norm_layer
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.ssm_cfg = ssm_cfg

        # Build blocks
        self.blocks = nn.ModuleList(
            [
                create_block(
                    dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # Patch merging layer
        if downsample:
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf
            )
        else:
            self.downsample = None

    def forward(self, x, H, W, T, inference_params=None):
        """Forward pass through all blocks in this layer.

        Args:
            x: Input tensor of shape (B, H*W*T, C)
            H: Height of feature map
            W: Width of feature map
            T: Depth of feature map
            inference_params: Parameters for inference mode

        Returns:
            Tuple of (x, H, W, T, x_down, Wh, Ww, Wt) where:
                - x: Output features before downsampling
                - x_down: Downsampled features (or x if no downsampling)
                - Wh, Ww, Wt: Downsampled dimensions
        """
        residual = None
        hidden_states = x
        for idx, block in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = block(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                    use_checkpoint=True,
                )
            else:
                hidden_states, residual = block(
                    hidden_states, residual, inference_params=inference_params
                )

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """Factory function to create a Mamba block.

    Args:
        d_model: Model dimension
        ssm_cfg: SSM configuration dict
        norm_epsilon: Epsilon for normalization
        drop_path: Drop path rate
        rms_norm: Whether to use RMSNorm
        residual_in_fp32: Keep residuals in FP32
        fused_add_norm: Use fused add+norm operations
        layer_idx: Layer index for the mixer
        device: Device to create block on
        dtype: Data type for block parameters

    Returns:
        Block instance with configured mixer and normalization
    """
    if not MAMBA_SSM_AVAILABLE:
        raise ImportError(
            "Mamba2 is required to create Mamba blocks but mamba_ssm is not installed.\n"
            "Please install mamba_ssm or use ViT model instead."
        )

    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba2, layer_idx=layer_idx)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    """Initialize weights following GPT-2 initialization scheme.

    Reference: https://openai.com/blog/better-language-models/
    From: https://github.com/huggingface/transformers GPT-2 implementation

    Args:
        module: Module to initialize
        n_layer: List of layer depths
        initializer_range: Range for embedding initialization
        rescale_prenorm_residual: Whether to rescale prenorm residual weights
        n_residuals_per_layer: Number of residual connections per layer
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Scale weights of residual layers at initialization by 1/√N
        # where N is the number of residual layers
        # Reference: https://github.com/NVIDIA/Megatron-LM
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * sum(n_layer))


def segm_init_weights(m):
    """Initialize weights for segmentation layers.

    Args:
        m: Module to initialize
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MambaEncoderHeria(nn.Module):
    """Hierarchical Mamba encoder with multiple stages.

    Multi-stage encoder that progressively downsamples the input while increasing
    the number of feature channels. Each stage consists of multiple Mamba blocks.

    Args:
        config: Configuration object containing:
            - img_size: Input image size [D, H, W]
            - patch_size: Size of image patches
            - in_chans: Number of input channels
            - embed_dim: Base embedding dimension
            - depths: List of depths for each stage
            - drop_rate: Dropout rate
            - drop_path_rate: Stochastic depth rate
            - And other encoder configuration parameters
    """

    def __init__(self, config, **kwargs):
        super().__init__()

        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "MambaEncoderHeria requires mamba_ssm but it is not installed.\n"
                "Please install mamba_ssm or use ViTEncoderHiera instead."
            )

        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.pat_merg_rf = config.pat_merg_rf
        self.embed_dim = config.embed_dim
        self.depths = config.depths
        self.num_layers = len(self.depths)
        self.in_chans = config.in_chans
        self.norm_layer = nn.LayerNorm if config.patch_norm else None
        self.norm_epsilon = config.norm_epsilon
        self.drop_rate = config.drop_rate
        self.drop_path_rate = config.drop_path_rate
        self.ssm_cfg = config.ssm_cfg
        self.rms_norm = config.rms_norm
        self.norm_epsilon = config.norm_epsilon
        self.initializer_cfg = config.initializer_cfg
        self.out_indices = config.out_indices
        self.window_size = config.window_size
        self.use_checkpoint = config.use_checkpoint

        # Split image into non-overlapping patches
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
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                depth=self.depths[i_layer],
                window_size=self.window_size,
                drop=self.drop_rate,
                drop_path=dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=self.use_checkpoint,
                pat_merg_rf=self.pat_merg_rf,
                norm_layer=self.norm_layer,
                ssm_cfg=self.ssm_cfg,
                fused_add_norm=self.fused_add_norm,
                residual_in_fp32=self.residual_in_fp32,
                norm_epsilon=self.norm_epsilon,
                rms_norm=self.rms_norm,
            )
            self.layers.append(layer)

        num_features = [int(self.embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # Add a norm layer for each output
        for i_layer in self.out_indices:
            layer = self.norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        # Apply initialization
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=self.depths))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate KV cache for all layers during inference.

        Args:
            batch_size: Batch size for inference
            max_seqlen: Maximum sequence length
            dtype: Data type for cache
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping layer index to cache
        """
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return set of parameters that should not have weight decay."""
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self):
        """Return number of encoder layers."""
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        """Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            prefix: Prefix for parameter names
        """
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, x, inference_params=None):
        """Forward pass through hierarchical encoder.

        Args:
            x: Input tensor of shape (B, C, D, H, W)
            inference_params: Parameters for inference mode

        Returns:
            List of output features from each stage, including input
        """
        outs = [x.clone()]
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        in_T, in_H, in_W = T, H, W
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, T, H, W, x, in_T, in_H, in_W = layer(x, in_T, in_H, in_W)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.contiguous().view(B, self.num_features[i], T, H, W)
                outs.append(out)
        return outs


def inflate_weight(weight_2d, time_dim, center=True):
    """Inflate 2D convolutional weights to 3D.

    Args:
        weight_2d: 2D convolutional weights
        time_dim: Temporal dimension size
        center: If True, place 2D weights in center of temporal dimension.
                If False, replicate and average across temporal dimension.

    Returns:
        3D convolutional weights
    """
    print(f"Init center: {center}")
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


class ConvReLU(nn.Sequential):
    """Standard convolutional block with batch/instance norm and LeakyReLU.

    Args:
        mode: '2d' or '3d' for 2D or 3D convolutions
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        padding: Padding size. Default: 0
        stride: Stride size. Default: 1
        use_batchnorm: If True use BatchNorm, else use InstanceNorm. Default: True
    """

    def __init__(
        self,
        mode,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        if mode == "2d":
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            if not use_batchnorm:
                nm = nn.InstanceNorm2d(out_channels)
            else:
                nm = nn.BatchNorm2d(out_channels)
        elif mode == "3d":
            conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            if not use_batchnorm:
                nm = nn.InstanceNorm3d(out_channels)
            else:
                nm = nn.BatchNorm3d(out_channels)
        else:
            raise ValueError(f"Unknown mode: {mode} (2d or 3d expected)")
        relu = nn.LeakyReLU(inplace=True)
        super(ConvReLU, self).__init__(conv, nm, relu)


class ConvReLULight(nn.Sequential):
    """Depthwise separable convolutional block with norm and LeakyReLU.

    Uses depthwise convolution followed by pointwise convolution for efficiency.

    Args:
        mode: '2d' or '3d' for 2D or 3D convolutions
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        padding: Padding size. Default: 0
        stride: Stride size. Default: 1
        use_batchnorm: If True use BatchNorm, else use InstanceNorm. Default: True
    """

    def __init__(
        self,
        mode,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        if mode == "2d":
            depthwise = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            )
            pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            if not use_batchnorm:
                nm = nn.InstanceNorm2d(out_channels)
            else:
                nm = nn.BatchNorm2d(out_channels)
        elif mode == "3d":
            depthwise = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            )
            pointwise = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            if not use_batchnorm:
                nm = nn.InstanceNorm3d(out_channels)
            else:
                nm = nn.BatchNorm3d(out_channels)
        else:
            raise ValueError(f"Unknown mode: {mode} (2d or 3d expected)")
        relu = nn.LeakyReLU(inplace=True)
        super(ConvReLULight, self).__init__(depthwise, pointwise, nm, relu)


class ConvDecoderBlock(nn.Module):
    """Decoder block with upsampling and optional skip connections.

    Args:
        mode: '2d' or '3d' for 2D or 3D operations
        in_channels: Number of input channels
        out_channels: Number of output channels
        skip_channels: Number of skip connection channels. Default: 0
        scale_factor: Upsampling factor. Default: 2
        use_batchnorm: Whether to use batch normalization. Default: True
        use_depthseparatble: Whether to use depthwise separable convolutions. Default: False
    """

    def __init__(
        self,
        mode,
        in_channels,
        out_channels,
        skip_channels=0,
        scale_factor=2,
        use_batchnorm=True,
        use_depthseparatble=False,
    ):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="trilinear", align_corners=False
        )
        if not use_depthseparatble:
            self.conv1 = ConvReLU(
                mode,
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
            self.conv2 = ConvReLU(
                mode,
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.conv1 = ConvReLULight(
                mode,
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
            self.conv2 = ConvReLULight(
                mode,
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )

    def forward(self, x, skip=None):
        """Forward pass with optional skip connection.

        Args:
            x: Input tensor
            skip: Skip connection tensor. Default: None

        Returns:
            Upsampled and processed tensor
        """
        x = self.up(x)
        if skip is not None:
            # Handle spatial dimension mismatch by interpolating x to match skip
            if x.shape[2:] != skip.shape[2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:], mode='trilinear', align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvDecoder(nn.Module):
    """Multi-stage convolutional decoder with optional skip connections.

    Progressively upsamples features from the encoder to reconstruct
    the original resolution.

    Args:
        config: Configuration object containing:
            - if_convskip: Whether to use convolutional skip connections
            - if_transskip: Whether to use transformer skip connections
            - embed_dim: Base embedding dimension
            - img_size: Target image size
            - in_chans: Number of input channels
            - patch_size: Patch size for final upsampling
            - depths: List of encoder depths
            - pat_merg_rf: Patch merging reduction factor
            - decoder_head_chan: Number of channels in decoder head
    """

    def __init__(self, config):
        super(ConvDecoder, self).__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        self.embed_dim = config.embed_dim
        self.img_size = config.img_size
        self.in_chans = config.in_chans
        self.patch_size = config.patch_size
        self.depths = config.depths
        self.reduce_factor = config.pat_merg_rf
        self.decoder_head_chan = config.decoder_head_chan

        for i in range(len(self.depths) - 1, 0, -1):
            setattr(
                self,
                f"up{i}",
                ConvDecoderBlock(
                    "3d",
                    self.embed_dim * self.reduce_factor**i,
                    self.embed_dim * self.reduce_factor ** (i - 1),
                    skip_channels=(
                        self.embed_dim * self.reduce_factor ** (i - 1)
                        if self.if_convskip
                        else 0
                    ),
                    scale_factor=self.reduce_factor,
                    use_batchnorm=False,
                    use_depthseparatble=True,
                ),
            )
        self.up0 = ConvDecoderBlock(
            "3d",
            self.embed_dim,
            self.decoder_head_chan,
            skip_channels=self.in_chans if self.if_convskip else 0,
            scale_factor=self.patch_size,
            use_batchnorm=False,
            use_depthseparatble=False,
        )

    def forward(self, out_feats):
        """Forward pass through decoder.

        Args:
            out_feats: List of encoder features or single feature tensor

        Returns:
            Reconstructed feature map at original resolution
        """
        if len(out_feats) == 1:
            out_feats = out_feats[0]
            for i in range(len(self.depths) - 1, 0, -1):
                out_feats = getattr(self, f"up{i}")(out_feats, None)
            out_feats = self.up0(out_feats, None)
        else:
            assert len(out_feats) == len(
                self.depths
            ) + 1, f"Expected {len(self.depths)+1} features, got {len(out_feats)}"
            x = out_feats[-1]
            for i in range(len(self.depths) - 1, 0, -1):
                x = getattr(self, f"up{i}")(x, out_feats[i])
            x = self.up0(x, out_feats[0])
        return x


class Head(nn.Sequential):
    """Final convolutional head for output prediction.

    Initialized with small weights for stable training.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size. Default: 3
        padding: Padding size. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class reg_decoder(nn.Module):
    """Decoder for image registration task.

    Predicts a 3-channel displacement field for spatial transformation.

    Args:
        config: Configuration object for ConvDecoder
    """

    def __init__(self, config):
        super(reg_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=3,
            kernel_size=3,
            padding=1,
        )

    def forward(self, out_feats):
        """Forward pass to predict displacement field.

        Args:
            out_feats: Encoder features

        Returns:
            3-channel displacement field
        """
        out = self.decoder(out_feats)
        flow = self.head(out)
        return flow


class fus_decoder(nn.Module):
    """Decoder for image fusion task.

    Predicts a single-channel fused image.

    Args:
        config: Configuration object for ConvDecoder
    """

    def __init__(self, config):
        super(fus_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, out_feats):
        """Forward pass to predict fused image.

        Args:
            out_feats: Encoder features

        Returns:
            Fused image
        """
        out = self.decoder(out_feats)
        out = self.head(out)
        return out


class SR_decoder(nn.Module):
    """Decoder for super-resolution task.

    Predicts a single-channel super-resolved image.

    Args:
        config: Configuration object for ConvDecoder
    """

    def __init__(self, config):
        super(SR_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, out_feats):
        """Forward pass to predict super-resolved image.

        Args:
            out_feats: Encoder features

        Returns:
            Super-resolved image
        """
        out = self.decoder(out_feats)
        out = self.head(out)
        return out


class IR_decoder(nn.Module):
    """Decoder for isotropic restoration task.

    Predicts a single-channel isotropically restored image.

    Args:
        config: Configuration object for ConvDecoder
    """

    def __init__(self, config):
        super(IR_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(self, out_feats):
        """Forward pass to predict restored image.

        Args:
            out_feats: Encoder features

        Returns:
            Isotropically restored image
        """
        out = self.decoder(out_feats)
        out = self.head(out)
        return out


class SpatialTransformer(nn.Module):
    """Apply spatial transformations using displacement fields.

    Implements differentiable spatial transformation for image registration.

    Args:
        size: Size of the image [D, H, W]
        mode: Interpolation mode. Default: 'bilinear'
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.size = size

    def create_grid(self, flow):
        """Create coordinate grid on the same device as flow.

        Args:
            flow: Displacement field

        Returns:
            Coordinate grid
        """
        vectors = [torch.arange(0, s, device=flow.device) for s in self.size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = grid.unsqueeze(0).to(flow.device)
        return grid

    def apply_flow(self, grid, flow):
        """Apply displacement field to coordinate grid.

        Args:
            grid: Coordinate grid
            flow: Displacement field

        Returns:
            Transformed coordinates normalized to [-1, 1]
        """
        new_locs = grid + flow
        shape = flow.shape[2:]

        # Normalize grid values to [-1, 1] range
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Move channel dimension to last position and reverse channel order
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return new_locs

    def forward(self, src, flow):
        """Apply spatial transformation to source image.

        Args:
            src: Source image to transform
            flow: Displacement field

        Returns:
            Transformed image
        """
        grid = self.create_grid(flow)
        new_locs = self.apply_flow(grid, flow)
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class MambaULight(nn.Module):
    """Main Mamba U-Net model for multiple image processing tasks.

    Combines a hierarchical Mamba encoder with multiple task-specific decoders
    for registration, fusion, super-resolution, and isotropic restoration.

    Args:
        config: Configuration object containing all model parameters
    """

    def __init__(self, config):
        super(MambaULight, self).__init__()

        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "MambaULight requires mamba_ssm but it is not installed.\n"
                "Please install mamba_ssm or use ViTULight instead."
            )

        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        self.embed_dim = config.embed_dim
        self.img_size = config.img_size

        self.encoder = MambaEncoderHeria(config)
        self.grid_size = config.grid_size
        self.spatial_trans = SpatialTransformer(config.grid_size)
        self.grid_img = self.create_grid_image()
        self.reg_decoder = reg_decoder(config)
        self.fus_decoder = fus_decoder(config)
        self.SR_decoder = SR_decoder(config)
        self.IR_decoder = IR_decoder(config)

        # Loss functions
        self.mse = nn.MSELoss()
        self.ncc = losses.NCC_vxm()
        self.grad = losses.Grad3d(penalty="l2")
        self.ssim = losses.SSIM3D()

    def forward(self, raw):
        """Forward pass through all tasks.

        Applies synthetic degradations and restores them using task-specific decoders.

        Args:
            raw: Input image of shape (B, C, D, H, W)

        Returns:
            Tuple of (logits, aux_loss) where:
                - logits: Dictionary containing degraded and restored images for each task
                - aux_loss: Dictionary containing loss values for each task
        """
        # Registration: deformation degradation
        reg_source, reg_flow = self.deform(raw)
        x = torch.cat([reg_source, raw], dim=1)
        out_feats = self.encoder(x)
        reg_inv_flow = self.reg_decoder(out_feats)
        reged = self.spatial_trans(reg_source, reg_inv_flow)

        # Fusion: mask degradation
        fus_source_A = self.mask(raw)
        fus_source_B = self.mask(raw)
        x = torch.cat([fus_source_A, fus_source_B], dim=1)
        out_feats = self.encoder(x)
        fused = self.fus_decoder(out_feats)

        # Super-resolution: downsampling degradation
        SR_source = self.downsample(raw)
        x = torch.cat([SR_source, SR_source], dim=1)
        out_feats = self.encoder(x)
        SRed = self.SR_decoder(out_feats)

        # Isotropic restoration: noise degradation
        IR_source = self.noise(raw)
        x = torch.cat([IR_source, IR_source], dim=1)
        out_feats = self.encoder(x)
        IRed = self.IR_decoder(out_feats)

        # Output results
        logits = {
            "raw": raw.detach().cpu().numpy(),
            "reg": {
                "deformed": reg_source.detach().cpu().numpy(),
                "registered": reged.detach().cpu().numpy(),
            },
            "fus": {
                "masked_A": fus_source_A.detach().cpu().numpy(),
                "masked_B": fus_source_B.detach().cpu().numpy(),
                "fused": fused.detach().cpu().numpy(),
            },
            "SR": {
                "downsampled": SR_source.detach().cpu().numpy(),
                "super_resolution": SRed.detach().cpu().numpy(),
            },
            "IR": {
                "noisy": IR_source.detach().cpu().numpy(),
                "restored": IRed.detach().cpu().numpy(),
            },
        }

        # Compute losses
        aux_loss = {
            "mse": {
                "reg": self.mse(reged, raw),
                "fus": self.mse(fused, raw),
                "SR": self.mse(SRed, raw),
                "IR": self.mse(IRed, raw),
            },
            "ncc": {"reg": self.ncc(reged, raw)},
            "grad": {"reg": self.grad(reg_inv_flow, raw)},
        }

        return logits, aux_loss

    def create_grid_image(self, grid_spacing=4, line_width=1):
        """Create a 3D grid image for visualizing deformations.

        Args:
            grid_spacing: Spacing between grid lines. Default: 4
            line_width: Width of grid lines. Default: 1

        Returns:
            3D grid image tensor
        """
        depth, height, width = self.grid_size
        grid = torch.zeros((1, 1, depth, height, width), dtype=torch.float32)

        # Create horizontal lines
        for y in range(0, height, grid_spacing):
            grid[:, :, :, y : y + line_width, :] = 1

        # Create vertical lines
        for x in range(0, width, grid_spacing):
            grid[:, :, :, :, x : x + line_width] = 1

        # Create depth lines
        for z in range(0, depth, grid_spacing):
            grid[:, :, z : z + line_width, :, :] = 1

        return grid

    def deform(self, image):
        """Apply synthetic deformation to image.

        Generates natural-looking deformation fields using multi-scale Perlin noise
        and applies spatial transformation.

        Args:
            image: Input image of shape (B, C, D, H, W)

        Returns:
            Tuple of (deformed_image, flow_field)
        """
        b, c, d, h, w = image.shape

        # Generate low-resolution deformation field
        lowres_d, lowres_h, lowres_w = d // 2, h // 2, w // 2

        flow = self.generate_natural_deformation_field(
            b, lowres_d, lowres_h, lowres_w, device=image.device
        )

        # Apply nonlinear transformation for natural deformation
        flow = torch.tanh(flow) * 0.6

        # Apply spatially-varying Gaussian filtering
        sigma_range = [1.5, 3.5]
        flow = self.spatially_varying_gaussian_filter(flow, sigma_range)

        # Upsample to original resolution
        flow = F.interpolate(flow, size=(d, h, w), mode="trilinear", align_corners=True)

        return self.spatial_trans(image, flow), flow

    def generate_natural_deformation_field(self, b, d, h, w, device):
        """Generate natural deformation field using multi-scale Perlin noise.

        Args:
            b: Batch size
            d, h, w: Dimensions of deformation field
            device: Device to create tensors on

        Returns:
            3-channel deformation field
        """

        def perlin_noise(coords, octaves=4, persistence=0.5):
            """Generate Perlin noise with multiple octaves."""
            noise = torch.zeros(b, d, h, w, device=device)
            frequency = 1
            amplitude = 1
            for _ in range(octaves):
                noise += amplitude * self.simplex_noise(frequency * coords)
                frequency *= 2
                amplitude *= persistence
            return noise

        coords = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, d),
                    torch.linspace(-1, 1, h),
                    torch.linspace(-1, 1, w),
                    indexing="ij",
                ),
                dim=-1,
            )
            .to(device)
        )
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1, -1)

        flow = torch.stack(
            [perlin_noise(coords), perlin_noise(coords), perlin_noise(coords)], dim=1
        )

        return flow - flow.mean(dim=(2, 3, 4), keepdim=True)

    def simplex_noise(self, x):
        """Generate simplex noise for natural deformation patterns.

        Args:
            x: Input coordinates of shape (B, D, H, W, 3)

        Returns:
            Simplex noise values
        """
        b, d, h, w, _ = x.shape
        x = x.view(-1, 3)

        dot = lambda a, b: torch.sum(a * b, dim=-1)

        corners = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            device=x.device,
        )

        noise = torch.zeros(x.shape[0], device=x.device)
        for corner in corners:
            grid = x.floor() + corner
            P = x - grid
            N = torch.exp(-dot(P, P) / 0.5)
            grad = torch.randn_like(grid)
            grad = grad / grad.norm(dim=-1, keepdim=True)
            noise += N * dot(grad, P)

        return noise.view(b, d, h, w)

    def spatially_varying_gaussian_filter(self, input, sigma_range):
        """Apply spatially-varying Gaussian filtering.

        Args:
            input: Input tensor
            sigma_range: [min_sigma, max_sigma] for Gaussian kernel

        Returns:
            Filtered tensor
        """

        def gaussian_kernel_1d(sigma, kernel_size):
            """Create 1D Gaussian kernel."""
            x = torch.arange(kernel_size) - (kernel_size - 1) / 2
            return torch.exp(-(x**2) / (2 * sigma**2))

        b, c, d, h, w = input.shape

        # Generate different sigma values for each spatial location
        sigma_map = (
            torch.rand(b, 1, d, h, w, device=input.device)
            * (sigma_range[1] - sigma_range[0])
            + sigma_range[0]
        )

        # Create 3D kernel with maximum kernel size
        max_kernel_size = int(4 * sigma_range[1] + 1)
        kernel_x = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_y = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_z = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_3d = (
            kernel_x.view(-1, 1, 1) * kernel_y.view(1, -1, 1) * kernel_z.view(1, 1, -1)
        )
        kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape)

        # Apply different Gaussian filtering to each location
        output = torch.zeros_like(input)
        for i in range(c):
            channel_input = input[:, i : i + 1]
            channel_output = F.conv3d(
                F.pad(channel_input, (max_kernel_size // 2,) * 6, mode="reflect"),
                kernel_3d.expand(1, -1, -1, -1, -1),
                groups=1,
            )
            # Adjust output based on sigma_map
            output[:, i : i + 1] = channel_input + (
                channel_output - channel_input
            ) * (sigma_map - sigma_range[0]) / (sigma_range[1] - sigma_range[0])

        return output

    def mask(self, image):
        """Apply random masking degradation.

        Args:
            image: Input image

        Returns:
            Masked image with 50% random pixels set to zero
        """
        mask = torch.rand_like(image) < 0.5
        return image * mask

    def downsample(self, image):
        """Apply downsampling degradation.

        Args:
            image: Input image

        Returns:
            Downsampled and upsampled image (blurred)
        """
        sigma_down = [1, 2, 2]
        self.spatially_varying_gaussian_filter(image, sigma_down)
        # Downsample
        down = F.interpolate(
            image, scale_factor=0.5, mode="trilinear", align_corners=True
        )
        # Upsample back to original size
        up = F.interpolate(
            down, size=image.shape[2:], mode="trilinear", align_corners=True
        )
        return up

    def noise(self, image):
        """Apply noise degradation simulating low-light conditions.

        Combines Gaussian noise and Poisson noise to simulate photon noise.

        Args:
            image: Input image with values in [0, 1]

        Returns:
            Noisy image clipped to [0, 1]
        """
        # Add Gaussian noise (simulate photon noise in low light)
        noise_level = random.uniform(0.05, 0.15)
        noise = torch.randn_like(image) * noise_level
        noisy = image + noise
        # Ensure noisy >= 0 to avoid negative lambda_poisson
        noisy = torch.clamp(noisy, min=0.0)
        # Simulate Poisson distribution noise for photon counting
        lambda_poisson = noisy * 255  # Convert from [0,1] to [0,255]
        noisy = torch.poisson(lambda_poisson) / 255.0
        return torch.clamp(noisy, 0, 1)


def print_model_details(model):
    """Print detailed model statistics.

    Args:
        model: PyTorch model to analyze
    """
    print(model)
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param.numel()
    total_params = sum(params_dict.values())
    trainable_params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_dict[name] = param.numel()
    trainable_params = sum(trainable_params_dict.values())
    print(f"Total parameters: {total_params}")
    print(
        f"Trainable parameters: {trainable_params}, Ratio: {trainable_params/total_params*100:.2f}%"
    )
    print(
        f"Top 5 largest layers: {sorted(params_dict.items(), key=lambda x: x[1], reverse=True)[:5]}, "
        f"Ratio: {[x[1]/total_params*100 for x in sorted(params_dict.items(), key=lambda x: x[1], reverse=True)[:5]]}"
    )
    print(
        f"Top 5 largest trainable layers: {sorted(trainable_params_dict.items(), key=lambda x: x[1], reverse=True)[:5]}, "
        f"Ratio: {[x[1]/trainable_params*100 for x in sorted(trainable_params_dict.items(), key=lambda x: x[1], reverse=True)[:5]]}"
    )
