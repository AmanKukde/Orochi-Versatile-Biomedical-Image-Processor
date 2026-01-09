"""Loss functions for training."""

from orochi.losses.losses import (
    # Factory function
    get_loss_function,

    # SSIM losses
    SSIM,
    SSIM2D,
    SSIM3D,
    ssim,
    ssim3D,
    S3IMLoss,
    S3IMLossStitched,

    # Gradient losses
    Grad,
    Grad2d,
    Grad3d,
    Grad3DiTV,
    DisplacementRegularizer,

    # Information theory losses
    NCC_vxm,
    MIND_loss,
    MutualInformation,
    localMutualInformation,

    # Segmentation losses
    DiceLoss,

    # Pixel-wise losses
    LogMSELoss,
    LpipsLoss,
    RangeInvariantPSNRLoss,

    # Fusion losses
    MaxGradLoss3D,
    MaxPixelLoss3D,
    PixelLoss3D,
    MaxGradTokenSelect3D,
)

__all__ = [
    # Factory function
    "get_loss_function",

    # SSIM losses
    "SSIM",
    "SSIM2D",
    "SSIM3D",
    "ssim",
    "ssim3D",
    "S3IMLoss",
    "S3IMLossStitched",

    # Gradient losses
    "Grad",
    "Grad2d",
    "Grad3d",
    "Grad3DiTV",
    "DisplacementRegularizer",

    # Information theory losses
    "NCC_vxm",
    "MIND_loss",
    "MutualInformation",
    "localMutualInformation",

    # Segmentation losses
    "DiceLoss",

    # Pixel-wise losses
    "LogMSELoss",
    "LpipsLoss",
    "RangeInvariantPSNRLoss",

    # Fusion losses
    "MaxGradLoss3D",
    "MaxPixelLoss3D",
    "PixelLoss3D",
    "MaxGradTokenSelect3D",
]
