from ast import List
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal

import numpy as np
import losses, random, math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tabnanny import check
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from noise import pnoise3

import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_3tuple
from timm.models.vision_transformer import _load_weights
from mamba_ssm.modules.mamba2 import Mamba2
from einops import rearrange
from typing import Optional
from functools import partial
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.num_patches = (img_size[2] // patch_size[2]) * (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, T, H, W= x.size()
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
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 norm_layer=nn.LayerNorm,
                 reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, reduce_factor * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

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
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
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

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                dim,
                depth,
                window_size=(7, 7, 7),
                drop=0.,
                drop_path=0.,
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
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.norm_layer = norm_layer
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.ssm_cfg = ssm_cfg
    
        # build blocks
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

        # patch merging layer
        if downsample:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None
            
    def forward(self, x, H, W, T, inference_params=None):
        residual = None
        hidden_states = x
        for idx, block in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = block(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
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
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
):
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


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * sum(n_layer))


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    

class MambaEncoderHeria(nn.Module):
    def __init__(
            self, 
            config,
            **kwargs,
        ):
        super().__init__()
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.pat_merg_rf = config.pat_merg_rf
        self.embed_dim = config.embed_dim  # num_features for consistency with other models
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
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size = self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer)
        self.num_patches = self.patch_embed.num_patches

        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        # self.pos_drop = nn.Dropout(p=self.drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                                depth=self.depths[i_layer],
                                window_size=self.window_size,
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
                               )
            self.layers.append(layer)

        num_features = [int(self.embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        # add a norm layer for each output
        for i_layer in self.out_indices:
            layer = self.norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # original init
        self.apply(segm_init_weights)
        # trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=self.depths,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, x, inference_params=None):
        outs = [x.clone()]
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # x = x + self.pos_embed
        # x = self.pos_drop(x)

        in_T, in_H, in_W = T, H, W
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, T, H, W, x, in_T, in_H, in_W = layer(x, in_T, in_H, in_W)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.contiguous().view(B, self.num_features[i], T, H, W)
                outs.append(out)
        return outs


def inflate_weight(weight_2d, time_dim, center=True):
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

# class SpatialTransformer(nn.Module):
#     """
#     N-D Spatial Transformer
#     Obtained from https://github.com/voxelmorph/voxelmorph
#     """

#     def __init__(self,
#                  size,
#                  mode='bilinear'):
#         super().__init__()

#         self.mode = mode

#         # create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors)
#         grid = torch.stack(grids)
#         grid = torch.unsqueeze(grid, 0)
#         grid = grid.type(torch.FloatTensor)

#         # registering the grid as a buffer cleanly moves it to the GPU, but it also
#         # adds it to the state dict. this is annoying since everything in the state dict
#         # is included when saving weights to disk, so the model files are way bigger
#         # than they need to be. so far, there does not appear to be an elegant solution.
#         # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
#         self.register_buffer('grid', grid)

#     def forward(self, src, flow):
#         # new locations
#         new_locs = self.grid + flow
#         shape = flow.shape[2:]

#         # need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

#         # move channels dim to last position
#         # also not sure why, but the channels need to be reversed
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2, 1, 0]]

#         return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ConvReLU(nn.Sequential):
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
        if mode == '2d':
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
        elif mode == '3d':
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
            raise ValueError(f'Unknown mode: {mode} (2d or 3d expected)')
        relu = nn.LeakyReLU(inplace=True)
        super(ConvReLU, self).__init__(conv, nm, relu)

class ConvReLULight(nn.Sequential):
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
        if mode == '2d':
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
        elif mode == '3d':
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
            raise ValueError(f'Unknown mode: {mode} (2d or 3d expected)')
        relu = nn.LeakyReLU(inplace=True)
        super(ConvReLULight, self).__init__(depthwise, pointwise, nm, relu)


class ConvDecoderBlock(nn.Module):
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
        self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
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
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ConvDecoder(nn.Module):
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
        
        for i in range(len(self.depths)-1, 0, -1):
            setattr(self, f'up{i}', ConvDecoderBlock('3d',
                                                     self.embed_dim*self.reduce_factor**i,
                                                     self.embed_dim*self.reduce_factor**(i-1),
                                                     skip_channels= self.embed_dim*self.reduce_factor**(i-1) if self.if_convskip else 0,
                                                     scale_factor=self.reduce_factor,
                                                     use_batchnorm=False,
                                                     use_depthseparatble=True,
                                                     ))
        self.up0 = ConvDecoderBlock('3d',
                                    self.embed_dim,
                                    self.decoder_head_chan,
                                    skip_channels=self.in_chans if self.if_convskip else 0,
                                    scale_factor=self.patch_size,
                                    use_batchnorm=False,
                                    use_depthseparatble=False,
                                    )
    def forward(self, out_feats):
        if len(out_feats) == 1:
            out_feats = out_feats[0]
            for i in range(len(self.depths)-1, 0, -1):
                out_feats = getattr(self, f'up{i}')(out_feats, None)
            out_feats = self.up0(out_feats, None)
        else:
            assert len(out_feats) == len(self.depths)+1 , f'Expected {len(self.depths)+1} features, got {len(out_feats)}'
            x = out_feats[-1]
            for i in range(len(self.depths)-1, 0, -1):
                x = getattr(self, f'up{i}')(x, out_feats[i])
            x = self.up0(x, out_feats[0])
        return x

class Head(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        conv3d = nn.Conv3d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=padding)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class reg_decoder(nn.Module):
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
        out = self.decoder(out_feats)
        flow = self.head(out)
        return flow

class fus_decoder(nn.Module):
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
        out = self.decoder(out_feats)
        out = self.head(out)
        return out

class SR_decoder(nn.Module):
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
        out = self.decoder(out_feats)
        out = self.head(out)
        return out

class IR_decoder(nn.Module):
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
        out = self.decoder(out_feats)
        out = self.head(out)
        return out


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        self.size = size

    def create_grid(self, flow):
        vectors = [torch.arange(0, s, device=flow.device) for s in self.size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = grid.unsqueeze(0).to(flow.device)
        return grid

    def apply_flow(self, grid, flow):
        new_locs = grid + flow
        shape = flow.shape[2:]

        # 将网格值归一化到 [-1, 1] 范围内
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # 调整通道维度到最后一个位置
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return new_locs

    def forward(self, src, flow):
        grid = self.create_grid(flow)
        new_locs = self.apply_flow(grid, flow)
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class MambaULight(nn.Module):
    def __init__(self, config):
        super(MambaULight, self).__init__()
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
        # 像素损失
        self.mse = nn.MSELoss()
        # 配准损失
        self.ncc = losses.NCC_vxm()
        self.grad = losses.Grad3d(penalty='l2')
        self.ssim = losses.SSIM3D()
    
    def forward(self, raw):
        # 变形退化
        reg_source, reg_flow = self.deform(raw)
        x = torch.cat([reg_source, raw], dim=1)
        out_feats = self.encoder(x)
        reg_inv_flow = self.reg_decoder(out_feats)
        reged = self.spatial_trans(reg_source, reg_inv_flow)
        # grid_img = self.grid_img.expand(raw.shape[0], -1, -1, -1, -1)
        # # 应用变形场到网格图像
        # deformed_grid = self.spatial_trans(grid_img, reg_flow)
        # restored_grid = self.spatial_trans(grid_img, reg_inv_flow)
        
        # 掩码退化
        fus_source_A = self.mask(raw)
        fus_source_B = self.mask(raw)
        x = torch.cat([fus_source_A, fus_source_B], dim=1)
        out_feats = self.encoder(x)
        fused = self.fus_decoder(out_feats)
        
        # 下采样退化
        SR_source = self.downsample(raw)
        x = torch.cat([SR_source, SR_source], dim=1)
        out_feats = self.encoder(x)        
        SRed = self.SR_decoder(out_feats)
        
        # 噪声退化
        IR_source = self.noise(raw)
        x = torch.cat([IR_source, IR_source], dim=1)
        out_feats = self.encoder(x)
        IRed = self.IR_decoder(out_feats)
        
        # 输出结果
        logits = {
            'raw': raw.detach().cpu().numpy(),
            'reg': {
                'deformed': reg_source.detach().cpu().numpy(),
                # 'flow': reg_flow.detach().cpu().numpy(),
                # 'deformed_grid': deformed_grid.detach().cpu().numpy(),
                # 'inverted_flow': reg_inv_flow.detach().cpu().numpy(),
                # 'restored_grid': restored_grid.detach().cpu().numpy(),
                'registered': reged.detach().cpu().numpy()
            },
            'fus': {
                'masked_A': fus_source_A.detach().cpu().numpy(),
                'masked_B': fus_source_B.detach().cpu().numpy(),
                'fused': fused.detach().cpu().numpy()
            },
            'SR': {
                'downsampled': SR_source.detach().cpu().numpy(),
                'super_resolution': SRed.detach().cpu().numpy()
            },
            'IR': {
                'noisy': IR_source.detach().cpu().numpy(),
                'restored': IRed.detach().cpu().numpy()
            }
        }
        
        # 计算损失
        aux_loss = {
            'mse': {
                'reg': self.mse(reged, raw),
                'fus': self.mse(fused, raw),
                'SR': self.mse(SRed, raw),
                'IR': self.mse(IRed, raw)
            },
            # 'ssim': {
            #     'fus': self.ssim(fused, raw),
            #     'SR': self.ssim(SRed, raw),
            #     'IR': self.ssim(IRed, raw),
            # },
            'ncc': {
                'reg': self.ncc(reged, raw) # from TransMorph
            },
            'grad': {
                'reg': self.grad(reg_inv_flow, raw) # from TransMorph
            }
        }
        
        return logits, aux_loss

    def create_grid_image(self, grid_spacing=4, line_width=1):
        """
        创建一个网格图像
        shape: 元组，表示图像的形状 (depth, height, width)
        grid_spacing: 网格线之间的间距
        line_width: 网格线的宽度
        """
        depth, height, width = self.grid_size
        grid = torch.zeros((1, 1, depth, height, width), dtype=torch.float32)
        
        # 创建水平线
        for y in range(0, height, grid_spacing):
            grid[:, :, :, y:y+line_width, :] = 1
        
        # 创建垂直线
        for x in range(0, width, grid_spacing):
            grid[:, :, :, :, x:x+line_width] = 1
        
        # 创建深度方向的线
        for z in range(0, depth, grid_spacing):
            grid[:, :, z:z+line_width, :, :] = 1
        
        return grid
        
    def deform(self, image):
        b, c, d, h, w = image.shape
        
        # 生成低分辨率的形变场
        lowres_d, lowres_h, lowres_w = d//2, h//2, w//2
        
        # 生成低分辨率的形变场
        flow = self.generate_natural_deformation_field(b, lowres_d, lowres_h, lowres_w, device=image.device)
        
        # 应用非线性变换增强形变的自然性
        flow = torch.tanh(flow)*0.6  # 使用tanh函数限制形变幅度，并缩放到合适范围
        
        # 应用空间变化的高斯滤波
        sigma_range = [1.5, 3.5]  # 高斯滤波的sigma范围
        flow = self.spatially_varying_gaussian_filter(flow, sigma_range)
        
        # 上采样到原始分辨率
        flow = F.interpolate(flow, size=(d, h, w), mode='trilinear', align_corners=True)
        
        return self.spatial_trans(image, flow), flow

    def generate_natural_deformation_field(self, b, d, h, w, device):
        # 使用多尺度Perlin噪声生成更自然的形变场
        def perlin_noise(coords, octaves=4, persistence=0.5):
            noise = torch.zeros(b, d, h, w, device=device)
            frequency = 1
            amplitude = 1
            for _ in range(octaves):
                noise += amplitude * self.simplex_noise(frequency * coords)
                frequency *= 2
                amplitude *= persistence
            return noise

        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, d),
                                            torch.linspace(-1, 1, h),
                                            torch.linspace(-1, 1, w),
                                            indexing='ij'), dim=-1).to(device)
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1, -1)

        flow = torch.stack([
            perlin_noise(coords),
            perlin_noise(coords),
            perlin_noise(coords)
        ], dim=1)

        return flow - flow.mean(dim=(2, 3, 4), keepdim=True)  # 中心化流场

    def simplex_noise(self, x):
        b, d, h, w, _ = x.shape
        x = x.view(-1, 3)
        
        dot = lambda a, b: torch.sum(a*b, dim=-1)
        
        corners = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=x.device)
        
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
        def gaussian_kernel_1d(sigma, kernel_size):
            x = torch.arange(kernel_size) - (kernel_size - 1) / 2
            return torch.exp(-x**2 / (2*sigma**2))
        b, c, d, h, w = input.shape
        
        # 为每个空间位置生成不同的sigma值
        sigma_map = torch.rand(b, 1, d, h, w, device=input.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        
        # 创建最大kernel size的3D kernel
        max_kernel_size = int(4*sigma_range[1]+1)
        kernel_x = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_y = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_z = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_3d = (kernel_x.view(-1, 1, 1) * kernel_y.view(1, -1, 1) * kernel_z.view(1, 1, -1))
        kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape)
        
        # 对每个位置应用不同的高斯滤波
        output = torch.zeros_like(input)
        for i in range(c):
            channel_input = input[:, i:i+1]
            channel_output = F.conv3d(
                F.pad(channel_input, (max_kernel_size//2,)*6, mode='reflect'),
                kernel_3d.expand(1, -1, -1, -1, -1),
                groups=1
            )
            # 根据sigma_map调整输出
            output[:, i:i+1] = channel_input + (channel_output - channel_input) * (sigma_map - sigma_range[0]) / (sigma_range[1] - sigma_range[0])
        
        return output

    def mask(self, image):
        mask = torch.rand_like(image) < 0.5
        return image * mask

    def downsample(self, image):
        sigma_down = [1, 2, 2]  # 可以调整这些值
        self.spatially_varying_gaussian_filter(image, sigma_down)
        # 下采样
        down = F.interpolate(image, scale_factor=0.5, mode='trilinear', align_corners=True)
        # 上采样回原始大小
        up = F.interpolate(down, size=image.shape[2:], mode='trilinear', align_corners=True)
        return up

    def noise(self, image):
        # 添加高斯噪声（模拟低光照条件下的光子噪声）
        noise_level = random.uniform(0.05, 0.15)  # 增加噪声水平
        noise = torch.randn_like(image) * noise_level
        noisy = image + noise
        # 确保 noisy >= 0，以避免 lambda_poisson 为负
        noisy = torch.clamp(noisy, min=0.0)
        # 模拟光子计数的泊松分布噪声
        lambda_poisson = noisy * 255  # 假设像素值范围为0-1，转换为0-255
        noisy = torch.poisson(lambda_poisson) / 255.0        
        return torch.clamp(noisy, 0, 1)  # 确保像素值在 [0, 1] 范围内
    
    


def print_model_details(model):
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
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}, Ratio: {trainable_params/total_params*100:.2f}%')
    print(f'Top 5 largest layers: {sorted(params_dict.items(), key=lambda x: x[1], reverse=True)[:5]}, Ratio: {[x[1]/total_params*100 for x in sorted(params_dict.items(), key=lambda x: x[1], reverse=True)[:5]]}')
    print(f'Top 5 largest trainable layers: {sorted(trainable_params_dict.items(), key=lambda x: x[1], reverse=True)[:5]}, Ratio: {[x[1]/trainable_params*100 for x in sorted(trainable_params_dict.items(), key=lambda x: x[1], reverse=True)[:5]]}')  
