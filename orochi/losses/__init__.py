"""Loss functions for biomedical image processing.

Provides unified, dimension-agnostic loss functions for:

Reconstruction:
    - SSIM: Structural Similarity Index Measure
    - PSNRLoss: Peak Signal-to-Noise Ratio
    - L1Loss, L2Loss: Basic reconstruction losses

Registration:
    - NCC: Normalized Cross-Correlation
    - MutualInformation: Mutual Information loss
    - GradientLoss: First-order smoothness regularization
    - BendingEnergyLoss: Second-order smoothness regularization

Segmentation:
    - DiceLoss: Dice coefficient loss
    - GeneralizedDiceLoss: Weighted Dice for imbalanced data
    - TverskyLoss: Precision-recall trade-off control
    - FocalLoss: Focus on hard examples

All losses automatically handle 2D and 3D inputs.

Example:
    >>> from orochi.losses import SSIM, DiceLoss, NCC
    >>>
    >>> # Reconstruction
    >>> ssim_loss = SSIM(window_size=11)
    >>> loss = ssim_loss(pred, target)  # Works for 2D or 3D
    >>>
    >>> # Segmentation
    >>> dice_loss = DiceLoss(num_classes=5)
    >>> loss = dice_loss(logits, labels)
    >>>
    >>> # Registration
    >>> ncc_loss = NCC(window_size=9)
    >>> loss = ncc_loss(moving, fixed)
"""

# Reconstruction losses
from orochi.losses.reconstruction import SSIM, PSNRLoss, L1Loss, L2Loss

# Segmentation losses
from orochi.losses.segmentation import (
    DiceLoss,
    GeneralizedDiceLoss,
    TverskyLoss,
    FocalLoss,
)

# Registration losses
from orochi.losses.registration import (
    NCC,
    MutualInformation,
    GradientLoss,
    BendingEnergyLoss,
)

# Base classes (for extending)
from orochi.losses.base import BaseLoss, DimensionAgnosticLoss

__all__ = [
    # Reconstruction
    "SSIM",
    "PSNRLoss",
    "L1Loss",
    "L2Loss",
    # Segmentation
    "DiceLoss",
    "GeneralizedDiceLoss",
    "TverskyLoss",
    "FocalLoss",
    # Registration
    "NCC",
    "MutualInformation",
    "GradientLoss",
    "BendingEnergyLoss",
    # Base classes
    "BaseLoss",
    "DimensionAgnosticLoss",
]
