"""
Consolidated Mamba-based model architecture supporting both 2D and 3D medical image processing.

This module implements a unified Mamba architecture that can operate on both 2D images
and 3D volumes based on a dimensions parameter. The architecture maintains backward
compatibility with existing trained weights.
"""

from typing import Optional, Tuple, Union, List, Callable
from functools import partial
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.distributions.normal import Normal
import numpy as np

from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import _load_weights
from mamba_ssm.modules.mamba2 import Mamba2

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def to_ntuple(x: Union[int, Tuple], n: int) -> Tuple:
    """Convert input to n-tuple.

    Args:
        x: Integer or tuple to convert
        n: Number of dimensions (2 or 3)

    Returns:
        Tuple with n elements
    """
    if isinstance(x, tuple):
        return x
    return tuple([x] * n)


def to_2tuple(x: Union[int, Tuple]) -> Tuple[int, int]:
    """Convert input to 2-tuple."""
    return to_ntuple(x, 2)


def to_3tuple(x: Union[int, Tuple]) -> Tuple[int, int, int]:
    """Convert input to 3-tuple."""
    return to_ntuple(x, 3)


class PatchEmbed(nn.Module):
    """Image/Volume to Patch Embedding.

    Supports both 2D images and 3D volumes based on dimensions parameter.

    Args:
        img_size: Input image/volume size
        patch_size: Patch token size. Default: 4
        in_chans: Number of input channels. Default: 3
        embed_dim: Number of linear projection output channels. Default: 96
        norm_layer: Normalization layer. Default: None
        dimensions: Number of spatial dimensions (2 or 3). Default: 3
    """

    def __init__(
        self,
        img_size: Union[Tuple, List],
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None,
        dimensions: int = 3,
    ):
        super().__init__()
        self.dimensions = dimensions

        if dimensions == 2:
            patch_size = to_2tuple(patch_size)
            self.patch_size = patch_size
            self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif dimensions == 3:
            patch_size = to_3tuple(patch_size)
            self.patch_size = patch_size
            self.num_patches = (
                (img_size[2] // patch_size[2]) *
                (img_size[1] // patch_size[1]) *
                (img_size[0] // patch_size[0])
            )
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise ValueError(f"dimensions must be 2 or 3, got {dimensions}")

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with automatic padding."""
        if self.dimensions == 2:
            _, _, H, W = x.size()
            # Padding
            if W % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
            if H % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

            x = self.proj(x)  # B C Wh Ww

            if self.norm is not None:
                Wh, Ww = x.size(2), x.size(3)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        else:  # 3D
            _, _, T, H, W = x.size()
            # Padding
            if T % self.patch_size[2] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[2] - T % self.patch_size[2]))
            if W % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
            if H % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

            x = self.proj(x)  # B C Wt Wh Ww

            if self.norm is not None:
                Wt, Wh, Ww = x.size(2), x.size(3), x.size(4)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2).view(-1, self.embed_dim, Wt, Wh, Ww)

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer for hierarchical feature extraction.

    Supports both 2D and 3D based on the number of spatial dimensions in input.

    Args:
        dim: Number of input channels
        norm_layer: Normalization layer. Default: nn.LayerNorm
        reduce_factor: Channel reduction factor. Default: 2
        dimensions: Number of spatial dimensions (2 or 3). Default: 3
    """

    def __init__(
        self,
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
        reduce_factor: int = 2,
        dimensions: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.dimensions = dimensions

        if dimensions == 2:
            self.reduction = nn.Linear(4 * dim, reduce_factor * dim, bias=False)
            self.norm = norm_layer(4 * dim)
        else:  # 3D
            self.reduction = nn.Linear(8 * dim, reduce_factor * dim, bias=False)
            self.norm = norm_layer(8 * dim)

    def forward(self, x: Tensor, H: int, W: int, T: Optional[int] = None) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, H*W*T, C) or (B, H*W, C)
            H: Height dimension
            W: Width dimension
            T: Time/Depth dimension (only for 3D)
        """
        B, L, C = x.shape

        if self.dimensions == 2:
            assert L == H * W, f"Input feature has wrong size: {L} != {H} * {W}"
            assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even"

            x = x.view(B, H, W, C)

            # Padding
            pad_input = (H % 2 == 1) or (W % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

            # Split into 4 patches
            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        else:  # 3D
            assert T is not None, "T dimension required for 3D mode"
            assert L == H * W * T, f"Input feature has wrong size: {L} != {H} * {W} * {T}"
            assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}*{T}) are not even"

            x = x.view(B, H, W, T, C)

            # Padding
            pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

            # Split into 8 patches
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
    """Mamba block with residual connection and normalization.

    This block wraps a Mamba2 mixer with LayerNorm/RMSNorm and residual connections.
    The structure is: Add -> LN -> Mixer, which allows for fused add and LayerNorm.

    Args:
        dim: Model dimension
        mixer_cls: Mixer class (Mamba2)
        norm_cls: Normalization class. Default: nn.LayerNorm
        fused_add_norm: Whether to use fused add normalization. Default: False
        residual_in_fp32: Whether to keep residual in fp32. Default: False
        drop_path: Stochastic depth rate. Default: 0.
    """

    def __init__(
        self,
        dim: int,
        mixer_cls: Callable,
        norm_cls: nn.Module = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import failed"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params: Optional[dict] = None,
        use_checkpoint: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Pass the input through the block.

        Args:
            hidden_states: Input tensor
            residual: Residual from previous block
            inference_params: Parameters for inference
            use_checkpoint: Whether to use gradient checkpointing

        Returns:
            Tuple of (hidden_states, residual)
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: Optional[torch.dtype] = None, **kwargs):
        """Allocate inference cache for the mixer."""
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class BasicLayer(nn.Module):
    """A basic layer for one stage of the hierarchical architecture.

    Args:
        dim: Number of feature channels
        depth: Number of blocks in this layer
        drop: Dropout rate. Default: 0.
        drop_path: Stochastic depth rate. Default: 0.
        norm_layer: Normalization layer. Default: nn.LayerNorm
        downsample: Downsample layer at the end. Default: None
        use_checkpoint: Whether to use gradient checkpointing. Default: False
        pat_merg_rf: Patch merging reduction factor. Default: 2
        fused_add_norm: Whether to use fused add norm. Default: True
        residual_in_fp32: Whether to keep residual in fp32. Default: True
        ssm_cfg: SSM configuration dict. Default: None
        norm_epsilon: Epsilon for normalization. Default: 1e-5
        rms_norm: Whether to use RMSNorm. Default: True
        dimensions: Number of spatial dimensions (2 or 3). Default: 3
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        drop: float = 0.,
        drop_path: Union[float, List[float]] = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        pat_merg_rf: int = 2,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        ssm_cfg: Optional[dict] = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        dimensions: int = 3,
    ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        self.dimensions = dimensions

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
                dim=dim,
                norm_layer=norm_layer,
                reduce_factor=self.pat_merg_rf,
                dimensions=dimensions
            )
        else:
            self.downsample = None

    def forward(
        self,
        x: Tensor,
        H: int,
        W: int,
        T: Optional[int] = None,
        inference_params: Optional[dict] = None
    ) -> Union[Tuple[Tensor, int, int, Tensor, int, int],
               Tuple[Tensor, int, int, int, Tensor, int, int, int]]:
        """Forward pass through the layer.

        Args:
            x: Input tensor
            H: Height
            W: Width
            T: Depth (for 3D only)
            inference_params: Inference parameters

        Returns:
            For 2D: (x, H, W, x_down, Wh, Ww)
            For 3D: (x, H, W, T, x_down, Wh, Ww, Wt)
        """
        residual = None
        hidden_states = x

        for idx, block in enumerate(self.blocks):
            if self.use_checkpoint and hasattr(self, 'checkpoint_num') and idx < self.checkpoint_num:
                hidden_states, residual = block(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = block(
                    hidden_states, residual, inference_params=inference_params
                )

        if self.downsample is not None:
            if self.dimensions == 2:
                x_down = self.downsample(x, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
                return x, H, W, x_down, Wh, Ww
            else:  # 3D
                x_down = self.downsample(x, H, W, T)
                Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
                return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            if self.dimensions == 2:
                return x, H, W, x, H, W
            else:  # 3D
                return x, H, W, T, x, H, W, T


def create_block(
    d_model: int,
    ssm_cfg: Optional[dict] = None,
    norm_epsilon: float = 1e-5,
    drop_path: float = 0.,
    rms_norm: bool = True,
    residual_in_fp32: bool = True,
    fused_add_norm: bool = True,
    layer_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Block:
    """Create a Mamba block.

    Args:
        d_model: Model dimension
        ssm_cfg: SSM configuration
        norm_epsilon: Normalization epsilon
        drop_path: Stochastic depth rate
        rms_norm: Whether to use RMSNorm
        residual_in_fp32: Whether to keep residual in fp32
        fused_add_norm: Whether to use fused add norm
        layer_idx: Layer index
        device: Device
        dtype: Data type

    Returns:
        Block instance
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg)
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
    module: nn.Module,
    n_layer: List[int],
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    """Initialize weights following GPT-2 initialization scheme.

    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * sum(n_layer))


def segm_init_weights(m: nn.Module):
    """Initialize weights for segmentation models."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MambaEncoderHeria(nn.Module):
    """Hierarchical Mamba Encoder supporting both 2D and 3D inputs.

    This encoder uses a hierarchical architecture with patch embedding and merging
    to process images or volumes at multiple scales.

    Args:
        config: Configuration object with the following attributes:
            - residual_in_fp32: Whether to keep residual in fp32
            - fused_add_norm: Whether to use fused add norm
            - img_size: Input image/volume size
            - patch_size: Patch size
            - pat_merg_rf: Patch merging reduction factor
            - embed_dim: Embedding dimension
            - depths: Number of blocks at each stage
            - in_chans: Number of input channels
            - patch_norm: Whether to use patch normalization
            - norm_epsilon: Normalization epsilon
            - drop_rate: Dropout rate
            - drop_path_rate: Stochastic depth rate
            - ssm_cfg: SSM configuration
            - rms_norm: Whether to use RMSNorm
            - initializer_cfg: Initializer configuration
            - out_indices: Output indices
            - use_checkpoint: Whether to use gradient checkpointing
            - dimensions: Number of spatial dimensions (2 or 3)
    """

    def __init__(self, config, **kwargs):
        super().__init__()
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
        self.initializer_cfg = config.initializer_cfg
        self.out_indices = config.out_indices
        self.use_checkpoint = config.use_checkpoint
        self.dimensions = getattr(config, 'dimensions', 3)  # Default to 3D for backward compatibility

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
            dimensions=self.dimensions
        )
        self.num_patches = self.patch_embed.num_patches

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                depth=self.depths[i_layer],
                drop=self.drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=self.use_checkpoint,
                pat_merg_rf=self.pat_merg_rf,
                norm_layer=self.norm_layer,
                ssm_cfg=self.ssm_cfg,
                fused_add_norm=self.fused_add_norm,
                residual_in_fp32=self.residual_in_fp32,
                norm_epsilon=self.norm_epsilon,
                rms_norm=self.rms_norm,
                dimensions=self.dimensions,
            )
            self.layers.append(layer)

        num_features = [int(self.embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # Add a norm layer for each output
        for i_layer in self.out_indices:
            layer = self.norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # Initialize weights
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=self.depths))

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: Optional[torch.dtype] = None, **kwargs):
        """Allocate inference cache for all layers."""
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        """Parameters that should not use weight decay."""
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    def get_num_layers(self) -> int:
        """Get number of layers."""
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = ""):
        """Load pretrained weights."""
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, x: Tensor, inference_params: Optional[dict] = None, class_embed: Optional[Tensor] = None) -> List[Tensor]:
        """Forward pass through the encoder.

        Args:
            x: Input tensor of shape (B, C, H, W) for 2D or (B, C, T, H, W) for 3D
            inference_params: Inference parameters
            class_embed: Class embedding (optional)

        Returns:
            List of output features at different scales
        """
        outs = [x.clone()]
        x = self.patch_embed(x)

        if self.dimensions == 2:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

            in_H, in_W = H, W
            for i in range(self.num_layers):
                layer = self.layers[i]
                x_out, H, W, x, in_H, in_W = layer(x, in_H, in_W, inference_params=inference_params)

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x_out)
                    out = x_out.contiguous().view(B, self.num_features[i], H, W)
                    outs.append(out)
        else:  # 3D
            B, C, T, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)

            in_T, in_H, in_W = T, H, W
            for i in range(self.num_layers):
                layer = self.layers[i]
                x_out, T, H, W, x, in_T, in_H, in_W = layer(x, in_T, in_H, in_W, inference_params=inference_params)

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x_out)
                    out = x_out.contiguous().view(B, self.num_features[i], T, H, W)
                    outs.append(out)

        return outs


def inflate_weight(weight_2d: Tensor, time_dim: int, center: bool = True) -> Tensor:
    """Inflate 2D weights to 3D by replicating along temporal dimension.

    Args:
        weight_2d: 2D weight tensor
        time_dim: Temporal dimension size
        center: If True, place 2D weights at center. If False, average across temporal dim

    Returns:
        3D weight tensor
    """
    print(f'Init center: {center}')
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
    """Convolution + Normalization + ReLU block.

    Supports both 2D and 3D convolutions based on mode parameter.

    Args:
        mode: '2d' or '3d'
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
        padding: Padding. Default: 0
        stride: Stride. Default: 1
        use_batchnorm: Whether to use BatchNorm (True) or InstanceNorm (False). Default: True
    """

    def __init__(
        self,
        mode: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        if mode == '2d':
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
            nm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.InstanceNorm2d(out_channels)
        elif mode == '3d':
            conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
            nm = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.InstanceNorm3d(out_channels)
        else:
            raise ValueError(f'Unknown mode: {mode} (2d or 3d expected)')

        relu = nn.LeakyReLU(inplace=True)
        super(ConvReLU, self).__init__(conv, nm, relu)


class ConvReLULight(nn.Sequential):
    """Depthwise separable convolution + Normalization + ReLU block.

    Uses depthwise separable convolutions for efficiency.
    Supports both 2D and 3D based on mode parameter.

    Args:
        mode: '2d' or '3d'
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size
        padding: Padding. Default: 0
        stride: Stride. Default: 1
        use_batchnorm: Whether to use BatchNorm (True) or InstanceNorm (False). Default: True
    """

    def __init__(
        self,
        mode: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        if mode == '2d':
            depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                 groups=in_channels, bias=False)
            pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            nm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.InstanceNorm2d(out_channels)
        elif mode == '3d':
            depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                 groups=in_channels, bias=False)
            pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            nm = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.InstanceNorm3d(out_channels)
        else:
            raise ValueError(f'Unknown mode: {mode} (2d or 3d expected)')

        relu = nn.LeakyReLU(inplace=True)
        super(ConvReLULight, self).__init__(depthwise, pointwise, nm, relu)


class ConvDecoderBlock(nn.Module):
    """Decoder block with upsampling and convolutions.

    Args:
        mode: '2d' or '3d'
        in_channels: Number of input channels
        out_channels: Number of output channels
        skip_channels: Number of skip connection channels. Default: 0
        scale_factor: Upsampling scale factor. Default: 2
        use_batchnorm: Whether to use batch normalization. Default: True
        use_depthseparable: Whether to use depthwise separable convolutions. Default: False
    """

    def __init__(
        self,
        mode: str,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        scale_factor: int = 2,
        use_batchnorm: bool = True,
        use_depthseparable: bool = False,
    ):
        super().__init__()

        if mode == '2d':
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        elif mode == '3d':
            self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
        else:
            raise ValueError(f'Unknown mode: {mode} (2d or 3d expected)')

        conv_class = ConvReLULight if use_depthseparable else ConvReLU

        self.conv1 = conv_class(mode, in_channels + skip_channels, out_channels,
                               kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = conv_class(mode, out_channels, out_channels,
                               kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        """Forward pass with optional skip connection."""
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvDecoder(nn.Module):
    """Convolutional decoder with skip connections.

    Args:
        config: Configuration object with attributes:
            - depths: Number of layers
            - if_convskip: Whether to use skip connections
            - decoder_mode: '2d' or '3d'
            - embed_dim: Embedding dimension
            - pat_merg_rf: Patch merging reduction factor
            - in_chans: Number of input channels
            - patch_size: Patch size
            - decoder_head_chan: Number of output channels before head
            - decoder_bn: Whether to use batch normalization
            - decoder_depthseparable: Whether to use depthwise separable convolutions
            - class_embed_dim: Class embedding dimension (optional)
    """

    def __init__(self, config):
        super(ConvDecoder, self).__init__()
        self.depths = config.depths
        self.if_convskip = config.if_convskip
        self.decoder_mode = getattr(config, 'decoder_mode', '3d')

        self.class_token_dim = getattr(config, 'class_embed_dim', 0)

        # Build upsampling blocks
        for i in range(len(self.depths) - 1, 0, -1):
            original_skip_ch = config.embed_dim * config.pat_merg_rf**(i - 1)

            skip_channels_for_block = 0
            if self.if_convskip:
                skip_channels_for_block = original_skip_ch
                if self.class_token_dim > 0:
                    skip_channels_for_block += self.class_token_dim

            setattr(self, f'up{i}', ConvDecoderBlock(
                mode=self.decoder_mode,
                in_channels=config.embed_dim * config.pat_merg_rf**i,
                out_channels=config.embed_dim * config.pat_merg_rf**(i - 1),
                skip_channels=skip_channels_for_block,
                scale_factor=config.pat_merg_rf,
                use_batchnorm=config.decoder_bn,
                use_depthseparable=config.decoder_depthseparable,
            ))

        # Final upsampling block
        original_skip_ch_up0 = config.in_chans
        skip_channels_for_up0 = 0
        if self.if_convskip:
            skip_channels_for_up0 = original_skip_ch_up0
            if self.class_token_dim > 0:
                skip_channels_for_up0 += self.class_token_dim

        self.up0 = ConvDecoderBlock(
            mode=self.decoder_mode,
            in_channels=config.embed_dim,
            out_channels=config.decoder_head_chan,
            skip_channels=skip_channels_for_up0,
            scale_factor=config.patch_size,
            use_batchnorm=config.decoder_bn,
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        """Forward pass through decoder.

        Args:
            out_feats: List of feature tensors from encoder
            class_embed: Optional class embedding tensor

        Returns:
            Decoded output tensor
        """
        if len(out_feats) == 1:
            x = out_feats[0]
            for i in range(len(self.depths) - 1, 0, -1):
                x = getattr(self, f'up{i}')(x, None)
            x = self.up0(x, None)
        else:
            if not (len(out_feats) == len(self.depths) + 1):
                raise ValueError(f'Expected {len(self.depths) + 1} features, got {len(out_feats)}')

            x = out_feats[-1]

            for i in range(len(self.depths) - 1, 0, -1):
                skip_feature = out_feats[i]
                final_skip_to_pass = None

                if self.if_convskip and skip_feature is not None:
                    final_skip_to_pass = skip_feature
                    if self.class_token_dim > 0 and class_embed is not None:
                        class_embed = class_embed.squeeze(1) if class_embed.dim() > 2 else class_embed
                        B, _, *spatial_dims = skip_feature.shape

                        if class_embed.shape[0] != B or class_embed.shape[1] != self.class_token_dim:
                            raise ValueError(f"Shape mismatch for class_embed at up{i}. "
                                           f"Expected B={B}, C={self.class_token_dim}. Got {class_embed.shape}")

                        token_reshaped = class_embed.view(B, self.class_token_dim, *([1] * len(spatial_dims)))
                        token_expanded = token_reshaped.expand(-1, -1, *spatial_dims)
                        final_skip_to_pass = torch.cat([skip_feature, token_expanded], dim=1)

                x = getattr(self, f'up{i}')(x, final_skip_to_pass)

            # Final upsampling
            skip_feature_0 = out_feats[0]
            final_skip_to_pass_0 = None
            if self.if_convskip and skip_feature_0 is not None:
                final_skip_to_pass_0 = skip_feature_0
                if self.class_token_dim > 0 and class_embed is not None:
                    B, _, *spatial_dims_0 = skip_feature_0.shape

                    if class_embed.shape[0] != B or class_embed.shape[1] != self.class_token_dim:
                        raise ValueError(f"Shape mismatch for class_embed at up0. "
                                       f"Expected B={B}, C={self.class_token_dim}. Got {class_embed.shape}")

                    token_reshaped_0 = class_embed.view(B, self.class_token_dim, *([1] * len(spatial_dims_0)))
                    token_expanded_0 = token_reshaped_0.expand(-1, -1, *spatial_dims_0)
                    final_skip_to_pass_0 = torch.cat([skip_feature_0, token_expanded_0], dim=1)

            x = self.up0(x, final_skip_to_pass_0)

        return x


class Head(nn.Sequential):
    """Output head with optional sparsity.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size. Default: 3
        padding: Padding. Default: 1
        sparsity: Sparsity ratio for weight pruning. Default: 0.0
        dimensions: Number of spatial dimensions (2 or 3). Default: 3
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        sparsity: float = 0.0,
        dimensions: int = 3
    ):
        super().__init__()

        if dimensions == 2:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        else:
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

        conv.weight = nn.Parameter(Normal(0, 1e-5).sample(conv.weight.shape))
        conv.bias = nn.Parameter(torch.zeros(conv.bias.shape))

        if sparsity > 0:
            self.apply_sparse_mask(conv, sparsity)

        self.add_module('conv', conv)

    def apply_sparse_mask(self, conv: nn.Module, sparsity: float):
        """Apply sparse mask to convolutional weights."""
        mask = torch.rand(conv.weight.shape).to(conv.weight.device) > sparsity
        conv.weight.data *= mask.float()


class SpatialTransformer(nn.Module):
    """Spatial transformer for warping images/volumes with displacement fields.

    Args:
        size: Spatial size of the input
        mode: Interpolation mode ('bilinear' for 2D, 'trilinear' for 3D). Default: 'bilinear'
    """

    def __init__(self, size: Union[Tuple, List], mode: str = 'bilinear'):
        super().__init__()
        self.mode = mode
        self.size = size

    def create_grid(self, flow: Tensor) -> Tensor:
        """Create sampling grid."""
        vectors = [torch.arange(0, s, device=flow.device) for s in self.size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = grid.unsqueeze(0).to(flow.device)
        return grid

    def apply_flow(self, grid: Tensor, flow: Tensor) -> Tensor:
        """Apply flow field to grid."""
        new_locs = grid + flow
        shape = flow.shape[2:]

        # Normalize grid values to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Adjust channel dimension to last position and reverse order
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return new_locs

    def forward(self, src: Tensor, flow: Tensor) -> Tensor:
        """Apply spatial transformation.

        Args:
            src: Source image/volume
            flow: Displacement field

        Returns:
            Warped image/volume
        """
        grid = self.create_grid(flow)
        new_locs = self.apply_flow(grid, flow)
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


# Decoder classes
class reg_decoder(nn.Module):
    """Registration decoder for predicting displacement fields."""

    def __init__(self, config):
        super(reg_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        dimensions = getattr(config, 'dimensions', 3)
        out_channels = 2 if dimensions == 2 else 3
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=dimensions
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        flow = self.head(out)
        return flow


class fus_decoder(nn.Module):
    """Fusion decoder for image fusion tasks."""

    def __init__(self, config):
        super(fus_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class fusRGB_decoder(nn.Module):
    """RGB fusion decoder for color image fusion."""

    def __init__(self, config):
        super(fusRGB_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=3,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class SR_decoder(nn.Module):
    """Super-resolution decoder."""

    def __init__(self, config):
        super(SR_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class IR_decoder(nn.Module):
    """Image restoration decoder."""

    def __init__(self, config):
        super(IR_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class den_decoder(nn.Module):
    """Denoising decoder."""

    def __init__(self, config):
        super(den_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class Proj_decoder(nn.Module):
    """Projection decoder."""

    def __init__(self, config):
        super(Proj_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class seg_decoder(nn.Module):
    """Segmentation decoder."""

    def __init__(self, config):
        super(seg_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class split_decoder(nn.Module):
    """Split decoder for separating images into two categories."""

    def __init__(self, config):
        super(split_decoder, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=2,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class split_decoder_single(nn.Module):
    """Split decoder with single output channel."""

    def __init__(self, config):
        super(split_decoder_single, self).__init__()
        self.decoder = ConvDecoder(config)
        self.head = Head(
            in_channels=config.decoder_head_chan,
            out_channels=1,
            kernel_size=3,
            padding=1,
            sparsity=getattr(config, 'head_sparsity', 0.0),
            dimensions=getattr(config, 'dimensions', 3)
        )

    def forward(self, out_feats: List[Tensor], class_embed: Optional[Tensor] = None) -> Tensor:
        out = self.decoder(out_feats, class_embed)
        out = self.head(out)
        return out


class Seg_prompt_encoder(nn.Module):
    """Prompt encoder for segmentation tasks using class embeddings."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, class_ids: Tensor) -> Tensor:
        """Encode class IDs to embeddings."""
        prompt_embed = self.class_embedding(class_ids)
        return prompt_embed


class Seg_point_encoder(nn.Module):
    """Prompt encoder for segmentation using point coordinates."""

    def __init__(self, embed_dim: int, img_size: Tuple, num_point_types: int = 2):
        super().__init__()
        self.img_size = img_size
        self.positional_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.type_encoder = nn.Embedding(num_point_types, embed_dim)

    def _normalize_coords(self, coords: Tensor, image_size: Tuple) -> Tensor:
        """Normalize coordinates to [0, 1]."""
        h, w = image_size
        scale = torch.tensor([w, h], device=coords.device).view(1, 1, 2)
        return coords / scale

    def forward(self, points: Tensor, labels: Tensor) -> Tensor:
        """Encode point coordinates and labels."""
        points_normalized = self._normalize_coords(points, self.img_size)
        position_embedding = self.positional_encoder(points_normalized)
        type_embedding = self.type_encoder(labels)
        return position_embedding + type_embedding


def print_model_details(model: nn.Module):
    """Print detailed model statistics."""
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

    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}, Ratio: {trainable_params/total_params*100:.2f}%')

    top_5 = sorted(params_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f'Top 5 largest layers: {top_5}')
    print(f'Ratios: {[x[1]/total_params*100 for x in top_5]}')

    top_5_train = sorted(trainable_params_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f'Top 5 largest trainable layers: {top_5_train}')
    print(f'Ratios: {[x[1]/trainable_params*100 for x in top_5_train]}')
