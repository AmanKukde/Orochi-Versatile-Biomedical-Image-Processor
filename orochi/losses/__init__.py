"""Loss functions for biomedical image processing.

Provides unified, dimension-agnostic loss functions for:

Reconstruction:
    - SSIM: Structural Similarity Index Measure
    - MSELoss: Mean Squared Error
    - L1Loss: Mean Absolute Error

Registration:
    - NCC: Normalized Cross-Correlation
    - MutualInformation: Mutual Information loss
    - GradientLoss: Gradient-based regularization

Segmentation:
    - DiceLoss: Dice coefficient loss
    - SurfaceLoss: Surface distance-based loss

Fusion:
    - GradientFusion: Gradient-based fusion loss
    - IntensityFusion: Intensity-based fusion loss

All losses automatically handle 2D and 3D inputs.

Example:
    >>> from orochi.losses import SSIM, DiceLoss
    >>>
    >>> # Reconstruction
    >>> ssim_loss = SSIM(window_size=11)
    >>> loss = ssim_loss(pred, target)  # Works for 2D or 3D
    >>>
    >>> # Segmentation
    >>> dice_loss = DiceLoss(num_classes=5)
    >>> loss = dice_loss(logits, labels)
"""

# Will be populated as we migrate code
__all__ = []
