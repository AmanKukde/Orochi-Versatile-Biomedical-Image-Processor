"""Reconstruction losses for image quality assessment.

Provides losses for measuring similarity between reconstructed and target images:
- SSIM: Structural Similarity Index Measure
- MS_SSIM: Multi-Scale SSIM
- PSNR: Peak Signal-to-Noise Ratio
- L1Loss, L2Loss: Basic reconstruction losses

All losses automatically handle 2D and 3D inputs.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from orochi.losses.base import BaseLoss, get_gaussian_kernel
from orochi.losses.constants import SSIM_WINDOW_SIZE, SSIM_C1, SSIM_C2, SSIM_SIGMA


class SSIM(BaseLoss):
    """Structural Similarity Index Measure (SSIM) Loss.

    Unified implementation that automatically handles both 2D and 3D inputs.
    Returns 1 - SSIM for use as a loss function (lower is better).

    SSIM measures the structural similarity between two images by comparing
    luminance, contrast, and structure. Originally designed for 2D images,
    this implementation extends naturally to 3D volumes.

    Reference:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to
        structural similarity." IEEE TIP 13.4 (2004): 600-612.

    Args:
        window_size: Size of the Gaussian window. Default: 11
        size_average: If True, returns average SSIM. If False, returns SSIM per batch.
            Default: True
        val_range: Value range of input (max - min). If None, inferred from data.
            Default: None
        channel: Number of channels. Default: 1
        sigma: Standard deviation for Gaussian window. Default: 1.5
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Shape:
        - Input: (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
        - Output: Scalar if reduction != 'none', otherwise same as input

    Example:
        >>> # 2D images
        >>> ssim_loss = SSIM(window_size=11)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> target = torch.rand(4, 1, 256, 256)
        >>> loss = ssim_loss(pred, target)
        >>> loss.item()  # Returns value in [0, 2], where 0 is identical
        0.123
        >>>
        >>> # 3D volumes - same API!
        >>> pred_3d = torch.rand(2, 1, 32, 256, 256)
        >>> target_3d = torch.rand(2, 1, 32, 256, 256)
        >>> loss_3d = ssim_loss(pred_3d, target_3d)
        >>>
        >>> # Identical images should have loss ~0
        >>> loss_identical = ssim_loss(pred, pred)
        >>> assert loss_identical < 0.01

    Note:
        - Window size should be odd
        - For small images/volumes, reduce window size
        - SSIM is not a proper distance metric (doesn't satisfy triangle inequality)
    """

    def __init__(
        self,
        window_size: int = SSIM_WINDOW_SIZE,
        size_average: bool = True,
        val_range: Optional[float] = None,
        channel: int = 1,
        sigma: float = SSIM_SIGMA,
        reduction: str = "mean",
    ):
        """Initialize SSIM loss."""
        super().__init__(reduction=reduction)

        if window_size % 2 == 0:
            raise ValueError(f"Window size must be odd, got {window_size}")

        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = channel
        self.sigma = sigma

        # Windows will be created dynamically based on input dimensions
        self.window_2d = None
        self.window_3d = None

    def _create_window(self, channel: int, ndim: int) -> torch.Tensor:
        """Create Gaussian window for given dimensions.

        Args:
            channel: Number of channels
            ndim: Number of spatial dimensions (2 or 3)

        Returns:
            Gaussian window tensor
        """
        return get_gaussian_kernel(self.window_size, self.sigma, channel, ndim)

    def _get_window(self, channel: int, ndim: int, device: torch.device) -> torch.Tensor:
        """Get or create Gaussian window.

        Args:
            channel: Number of channels
            ndim: Number of spatial dimensions
            device: Device to create window on

        Returns:
            Gaussian window on the correct device
        """
        if ndim == 2:
            if self.window_2d is None or self.window_2d.size(0) != channel:
                self.window_2d = self._create_window(channel, 2)
            window = self.window_2d
        else:  # ndim == 3
            if self.window_3d is None or self.window_3d.size(0) != channel:
                self.window_3d = self._create_window(channel, 3)
            window = self.window_3d

        return window.to(device)

    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        ndim: int,
    ) -> torch.Tensor:
        """Compute SSIM between two images/volumes.

        Args:
            img1: First image/volume
            img2: Second image/volume
            window: Gaussian window
            window_size: Size of window
            channel: Number of channels
            ndim: Number of spatial dimensions

        Returns:
            SSIM map
        """
        # Determine convolution function and padding
        if ndim == 2:
            conv_fn = F.conv2d
        else:  # ndim == 3
            conv_fn = F.conv3d

        padding = window_size // 2

        # Compute local means
        mu1 = conv_fn(img1, window, padding=padding, groups=channel)
        mu2 = conv_fn(img2, window, padding=padding, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = conv_fn(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = conv_fn(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = conv_fn(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

        # Compute SSIM
        C1 = SSIM_C1
        C2 = SSIM_C2

        if self.val_range is not None:
            C1 = (self.val_range * 0.01) ** 2
            C2 = (self.val_range * 0.03) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=tuple(range(1, ssim_map.ndim)))

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            pred: Predicted image/volume
            target: Target image/volume
            spatial_dims: Number of spatial dimensions (2 or 3)

        Returns:
            SSIM loss (1 - SSIM)
        """
        channel = pred.size(1)
        window = self._get_window(channel, spatial_dims, pred.device)

        ssim_value = self._ssim(
            pred, target, window, self.window_size, channel, spatial_dims
        )

        # Return 1 - SSIM for use as loss (lower is better)
        return 1 - ssim_value


class PSNRLoss(BaseLoss):
    """Peak Signal-to-Noise Ratio (PSNR) Loss.

    Computes PSNR between predicted and target images. Returns negative PSNR
    so that lower values indicate better similarity (for use as a loss).

    PSNR is defined as: 10 * log10(MAX^2 / MSE)

    Args:
        max_val: Maximum possible value in the images. Default: 1.0
        reduction: Reduction method. Default: 'mean'

    Example:
        >>> psnr_loss = PSNRLoss(max_val=1.0)
        >>> pred = torch.rand(4, 1, 256, 256)
        >>> target = torch.rand(4, 1, 256, 256)
        >>> loss = psnr_loss(pred, target)  # Returns negative PSNR
    """

    def __init__(self, max_val: float = 1.0, reduction: str = "mean"):
        """Initialize PSNR loss."""
        super().__init__(reduction=reduction)
        self.max_val = max_val

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute PSNR loss.

        Args:
            pred: Predicted image/volume
            target: Target image/volume
            spatial_dims: Number of spatial dimensions (ignored, works for any)

        Returns:
            Negative PSNR value (lower is better)
        """
        mse = F.mse_loss(pred, target, reduction="mean")
        psnr = 10 * torch.log10((self.max_val ** 2) / mse)
        return -psnr  # Negative so lower is better


# Aliases for convenience
class L1Loss(BaseLoss):
    """L1 (Mean Absolute Error) Loss.

    Simple wrapper around F.l1_loss with dimension validation.

    Example:
        >>> l1_loss = L1Loss()
        >>> loss = l1_loss(pred, target)
    """

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute L1 loss."""
        return F.l1_loss(pred, target, reduction="none")


class L2Loss(BaseLoss):
    """L2 (Mean Squared Error) Loss.

    Simple wrapper around F.mse_loss with dimension validation.

    Example:
        >>> l2_loss = L2Loss()
        >>> loss = l2_loss(pred, target)
    """

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute L2 loss."""
        return F.mse_loss(pred, target, reduction="none")
