"""Model-specific configurations for Orochi.

Provides configurations for different model architectures:
- Mamba2DConfig: 2D Mamba model configuration
- Mamba3DConfig: 3D Mamba model configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from orochi.configs.base_config import BaseConfig


@dataclass
class Mamba2DConfig(BaseConfig):
    """Configuration for 2D Mamba model.

    Extends BaseConfig with 2D-specific model parameters.

    Attributes:
        # Model architecture
        img_size: Input image size [H, W]
        patch_size: Patch size for patch embedding
        in_chans: Number of input channels
        out_chans: Number of output channels
        embed_dim: Embedding dimension
        depths: Number of blocks at each level
        num_heads: Number of attention heads (if using)

        # Mamba-specific
        d_state: State dimension for Mamba SSM
        d_conv: Convolution dimension
        expand: Expansion factor for hidden dimension

        # Task-specific
        task: Task type ('super_resolution', 'restoration', 'fusion', 'segmentation')
        num_classes: Number of classes (for segmentation)

        # Pretrained
        pretrained_path: Path to pretrained checkpoint (None to train from scratch)
        freeze_encoder: Freeze encoder weights during finetuning

    Example:
        >>> # Super-resolution config
        >>> config = Mamba2DConfig(
        ...     task='super_resolution',
        ...     img_size=[256, 256],
        ...     patch_size=16,
        ...     embed_dim=128
        ... )
        >>>
        >>> # Segmentation config
        >>> seg_config = Mamba2DConfig(
        ...     task='segmentation',
        ...     num_classes=5,
        ...     img_size=[512, 512]
        ... )
    """

    # Model architecture
    img_size: List[int] = field(default_factory=lambda: [256, 256])
    patch_size: int = 16
    in_chans: int = 1
    out_chans: int = 1
    embed_dim: int = 128
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: int = 4

    # Mamba-specific
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # Task-specific
    task: str = "super_resolution"
    num_classes: Optional[int] = None

    # Pretrained
    pretrained_path: Optional[Path] = None
    freeze_encoder: bool = False

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()

        # Validate image size
        if len(self.img_size) != 2:
            raise ValueError(f"img_size must be [H, W], got {self.img_size}")

        # Validate task
        valid_tasks = ['super_resolution', 'restoration', 'fusion', 'segmentation']
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{self.task}'")

        # Validate num_classes for segmentation
        if self.task == 'segmentation' and self.num_classes is None:
            raise ValueError("num_classes must be specified for segmentation task")

        # Convert pretrained_path if specified
        if self.pretrained_path is not None:
            self.pretrained_path = Path(self.pretrained_path)


@dataclass
class Mamba3DConfig(BaseConfig):
    """Configuration for 3D Mamba model.

    Extends BaseConfig with 3D-specific model parameters.

    Attributes:
        # Model architecture
        img_size: Input volume size [D, H, W]
        patch_size: Patch size for patch embedding (can be int or [D, H, W])
        in_chans: Number of input channels
        out_chans: Number of output channels
        embed_dim: Embedding dimension
        depths: Number of blocks at each level
        num_heads: Number of attention heads (if using)

        # Mamba-specific
        d_state: State dimension for Mamba SSM
        d_conv: Convolution dimension
        expand: Expansion factor for hidden dimension

        # Task-specific
        task: Task type ('super_resolution', 'registration', 'segmentation')
        num_classes: Number of classes (for segmentation)

        # Registration-specific
        use_deformation: Enable deformation module for registration
        int_steps: Number of integration steps for deformation
        int_downsize: Downsampling factor for integration

        # Pretrained
        pretrained_path: Path to pretrained checkpoint
        freeze_encoder: Freeze encoder weights

    Example:
        >>> # 3D super-resolution
        >>> config = Mamba3DConfig(
        ...     task='super_resolution',
        ...     img_size=[32, 256, 256],
        ...     patch_size=16,
        ...     embed_dim=192
        ... )
        >>>
        >>> # 3D registration
        >>> reg_config = Mamba3DConfig(
        ...     task='registration',
        ...     img_size=[64, 256, 256],
        ...     use_deformation=True,
        ...     int_steps=7
        ... )
    """

    # Model architecture
    img_size: List[int] = field(default_factory=lambda: [32, 256, 256])
    patch_size: int = 16
    in_chans: int = 1
    out_chans: int = 1
    embed_dim: int = 192
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: int = 6

    # Mamba-specific
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # Task-specific
    task: str = "super_resolution"
    num_classes: Optional[int] = None

    # Registration-specific
    use_deformation: bool = False
    int_steps: int = 7
    int_downsize: int = 2

    # Pretrained
    pretrained_path: Optional[Path] = None
    freeze_encoder: bool = False

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()

        # Validate image size
        if len(self.img_size) != 3:
            raise ValueError(f"img_size must be [D, H, W], got {self.img_size}")

        # Validate task
        valid_tasks = ['super_resolution', 'registration', 'segmentation']
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{self.task}'")

        # Validate num_classes for segmentation
        if self.task == 'segmentation' and self.num_classes is None:
            raise ValueError("num_classes must be specified for segmentation task")

        # Convert pretrained_path if specified
        if self.pretrained_path is not None:
            self.pretrained_path = Path(self.pretrained_path)


@dataclass
class SuperResolutionConfig(Mamba2DConfig):
    """Convenience config for 2D super-resolution.

    Example:
        >>> config = SuperResolutionConfig(
        ...     scale_factor=2,
        ...     img_size=[256, 256]
        ... )
    """
    task: str = "super_resolution"
    scale_factor: int = 2

    def __post_init__(self):
        super().__post_init__()
        # Output size will be scale_factor times input size
        self.out_chans = self.in_chans


@dataclass
class IsotropicRestorationConfig(Mamba2DConfig):
    """Convenience config for isotropic restoration.

    Example:
        >>> config = IsotropicRestorationConfig(
        ...     img_size=[256, 256]
        ... )
    """
    task: str = "restoration"


@dataclass
class ImageFusionConfig(Mamba2DConfig):
    """Convenience config for image fusion.

    Example:
        >>> config = ImageFusionConfig(
        ...     in_chans=2,  # Two input modalities
        ...     img_size=[256, 256]
        ... )
    """
    task: str = "fusion"
    in_chans: int = 2
    out_chans: int = 1


@dataclass
class Segmentation2DConfig(Mamba2DConfig):
    """Convenience config for 2D segmentation.

    Example:
        >>> config = Segmentation2DConfig(
        ...     num_classes=5,
        ...     img_size=[512, 512]
        ... )
    """
    task: str = "segmentation"
    num_classes: int = 2  # Default binary segmentation


@dataclass
class Registration3DConfig(Mamba3DConfig):
    """Convenience config for 3D registration.

    Example:
        >>> config = Registration3DConfig(
        ...     img_size=[64, 256, 256],
        ...     int_steps=7
        ... )
    """
    task: str = "registration"
    in_chans: int = 2  # Moving and fixed images
    out_chans: int = 3  # Deformation field (x, y, z)
    use_deformation: bool = True


@dataclass
class SuperResolution3DConfig(Mamba3DConfig):
    """Convenience config for 3D super-resolution.

    Example:
        >>> config = SuperResolution3DConfig(
        ...     scale_factor=2,
        ...     img_size=[32, 256, 256]
        ... )
    """
    task: str = "super_resolution"
    scale_factor: int = 2


# ============================================================================
# Vision Transformer (ViT) Configurations
# ============================================================================


@dataclass
class ViT3DConfig(BaseConfig):
    """Configuration for 3D Vision Transformer model.

    Extends BaseConfig with ViT-specific parameters. This config is designed
    to be comparable to Mamba3DConfig but uses standard transformer attention
    instead of Mamba state-space models.

    Attributes:
        # Model architecture
        img_size: Input volume size [D, H, W]
        patch_size: Patch size for patch embedding
        in_chans: Number of input channels
        out_chans: Number of output channels
        embed_dim: Embedding dimension
        depths: Number of transformer blocks at each level
        num_heads: Number of attention heads per block

        # ViT-specific
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: Whether to add bias to QKV projection
        attn_drop_rate: Attention dropout rate
        proj_drop_rate: Projection dropout rate

        # Architecture details
        patch_norm: Whether to use normalization after patch embedding
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate
        out_indices: Indices of stages to output features from
        if_convskip: Whether to use convolutional skip connections
        if_transskip: Whether to use transformer skip connections
        pat_merg_rf: Patch merging reduction factor
        decoder_head_chan: Number of channels in decoder head
        grid_size: Size for spatial transformation grid
        window_size: Window size (kept for compatibility, unused in ViT)
        use_checkpoint: Whether to use gradient checkpointing

        # SSM parameters (kept for interface compatibility, unused in ViT)
        ssm_cfg: SSM configuration (None for ViT)
        fused_add_norm: Whether to use fused operations (False for ViT)
        residual_in_fp32: Keep residuals in FP32 (False for ViT)
        rms_norm: Whether to use RMS norm (False for ViT)
        norm_epsilon: Epsilon for normalization

        # Encoder/decoder configuration
        initializer_cfg: Initializer configuration

        # Task-specific
        task: Task type ('multi_task' for all tasks together)

        # Pretrained
        pretrained_path: Path to pretrained checkpoint
        freeze_encoder: Freeze encoder weights during finetuning

    Example:
        >>> # ViT for multi-task learning
        >>> config = ViT3DConfig(
        ...     task='multi_task',
        ...     img_size=[64, 128, 128],
        ...     patch_size=4,
        ...     embed_dim=96,
        ...     depths=[2, 2, 4, 2],
        ...     num_heads=8
        ... )
    """

    # Model architecture
    img_size: List[int] = field(default_factory=lambda: [64, 128, 128])
    patch_size: int = 4
    in_chans: int = 2  # Two input images
    out_chans: int = 3
    embed_dim: int = 96
    depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])
    num_heads: int = 8

    # ViT-specific parameters
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    attn_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0

    # Architecture details
    patch_norm: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    out_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    if_convskip: bool = True
    if_transskip: bool = True
    pat_merg_rf: int = 2
    decoder_head_chan: int = 32
    grid_size: List[int] = field(default_factory=lambda: [64, 128, 128])
    window_size: List[int] = field(default_factory=lambda: [7, 7, 7])
    use_checkpoint: bool = False

    # SSM parameters (kept for compatibility, unused in ViT)
    ssm_cfg: Optional[dict] = None
    fused_add_norm: bool = False
    residual_in_fp32: bool = False
    rms_norm: bool = False
    norm_epsilon: float = 1e-5
    initializer_cfg: Optional[dict] = None

    # Task-specific
    task: str = "multi_task"

    # Pretrained
    pretrained_path: Optional[Path] = None
    freeze_encoder: bool = False

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()

        # Validate image size
        if len(self.img_size) != 3:
            raise ValueError(f"img_size must be [D, H, W], got {self.img_size}")

        # Validate grid_size
        if len(self.grid_size) != 3:
            raise ValueError(f"grid_size must be [D, H, W], got {self.grid_size}")

        # Validate task
        valid_tasks = ['multi_task', 'registration', 'fusion', 'super_resolution', 'isotropic_restoration']
        if self.task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{self.task}'")

        # Convert pretrained_path if specified
        if self.pretrained_path is not None:
            self.pretrained_path = Path(self.pretrained_path)

        # Ensure num_heads divides embed_dim evenly
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")


@dataclass
class MambaULightConfig(BaseConfig):
    """Configuration for MambaULight multi-task model.

    This is a wrapper around the original config structure used by MambaULight.

    Example:
        >>> config = MambaULightConfig(
        ...     img_size=[64, 128, 128],
        ...     patch_size=4,
        ...     embed_dim=96
        ... )
    """

    # Model architecture
    img_size: List[int] = field(default_factory=lambda: [64, 128, 128])
    patch_size: int = 4
    in_chans: int = 2
    embed_dim: int = 96
    depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])

    # Mamba-specific
    ssm_cfg: Optional[dict] = None
    fused_add_norm: bool = True
    residual_in_fp32: bool = True
    rms_norm: bool = True
    norm_epsilon: float = 1e-5

    # Architecture details
    patch_norm: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    out_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    if_convskip: bool = True
    if_transskip: bool = True
    pat_merg_rf: int = 2
    decoder_head_chan: int = 32
    grid_size: List[int] = field(default_factory=lambda: [64, 128, 128])
    window_size: List[int] = field(default_factory=lambda: [7, 7, 7])
    use_checkpoint: bool = False
    initializer_cfg: Optional[dict] = None

    # Pretrained
    pretrained_path: Optional[Path] = None
    freeze_encoder: bool = False

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()

        # Validate sizes
        if len(self.img_size) != 3:
            raise ValueError(f"img_size must be [D, H, W], got {self.img_size}")
        if len(self.grid_size) != 3:
            raise ValueError(f"grid_size must be [D, H, W], got {self.grid_size}")

        # Convert pretrained_path if specified
        if self.pretrained_path is not None:
            self.pretrained_path = Path(self.pretrained_path)
