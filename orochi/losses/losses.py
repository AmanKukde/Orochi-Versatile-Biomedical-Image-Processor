"""
Consolidated Loss Functions for Orochi Biomedical Image Processor

This module contains all loss functions for 2D and 3D biomedical image processing tasks,
including:
- Structural Similarity (SSIM) losses
- Gradient-based regularization losses
- Information theory losses (NCC, MI, MIND)
- Segmentation losses (Dice)
- Pixel-wise losses (MSE, L1, PSNR)
- Image fusion losses

All loss functions support both 2D and 3D inputs where applicable.

Mathematical notation:
- x, y: input tensors (predicted and target)
- N: batch size
- C: number of channels
- H, W: height and width (2D)
- D, H, W: depth, height, width (3D)
"""

from typing import Optional, Union, Tuple, List, Literal
from math import exp
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


# ============================================================================
# Utility Functions for SSIM Computation
# ============================================================================

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Generate a 1D Gaussian kernel.

    The Gaussian function is defined as:
    G(x) = exp(-(x - μ)² / (2σ²))

    where μ is the center of the window (window_size // 2).

    Args:
        window_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        Normalized 1D Gaussian kernel as a tensor

    Example:
        >>> kernel = gaussian(11, 1.5)
        >>> assert kernel.sum().item() == 1.0  # Normalized
    """
    # Compute Gaussian values for each position in the window
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    # Normalize so sum equals 1
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Create a 2D Gaussian window for SSIM computation.

    The 2D Gaussian is created by outer product of 1D Gaussian:
    G_2D = G_1D ⊗ G_1D^T

    Args:
        window_size: Size of the window (e.g., 11 for 11x11 window)
        channel: Number of channels to expand the window for

    Returns:
        2D Gaussian window of shape (channel, 1, window_size, window_size)

    Example:
        >>> window = create_window(11, 3)  # For RGB image
        >>> assert window.shape == (3, 1, 11, 11)
    """
    # Create 1D Gaussian kernel
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    # Create 2D Gaussian by matrix multiplication (outer product)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    # Expand for all channels (each channel gets same window)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def create_window_3D(window_size: int, channel: int) -> torch.Tensor:
    """
    Create a 3D Gaussian window for 3D SSIM computation.

    The 3D Gaussian is created by extending the 2D Gaussian:
    G_3D = G_1D ⊗ G_2D

    Args:
        window_size: Size of the cubic window (e.g., 11 for 11x11x11)
        channel: Number of channels to expand the window for

    Returns:
        3D Gaussian window of shape (channel, 1, window_size, window_size, window_size)

    Example:
        >>> window = create_window_3D(11, 1)  # For single channel 3D volume
        >>> assert window.shape == (1, 1, 11, 11, 11)
    """
    # Create 1D Gaussian kernel
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    # Create 2D Gaussian first
    _2D_window = _1D_window.mm(_1D_window.t())

    # Extend to 3D by multiplying with 1D window again
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(
        window_size, window_size, window_size
    ).float().unsqueeze(0).unsqueeze(0)

    # Expand for all channels
    window = Variable(
        _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    )
    return window


# ============================================================================
# SSIM Computation Functions
# ============================================================================

def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) for 2D images.

    SSIM is defined as:
    SSIM(x,y) = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    where:
    - μ_x, μ_y: mean of x and y
    - σ_x², σ_y²: variance of x and y
    - σ_xy: covariance of x and y
    - C1, C2: constants to stabilize division (C1 = (0.01)², C2 = (0.03)²)

    Args:
        img1: First image tensor of shape (N, C, H, W)
        img2: Second image tensor of shape (N, C, H, W)
        window: Gaussian window for local statistics
        window_size: Size of the window
        channel: Number of channels
        size_average: If True, return mean SSIM; else return per-pixel SSIM

    Returns:
        SSIM value(s) as tensor

    Reference:
        Wang et al. "Image quality assessment: from error visibility to structural similarity"
        IEEE TIP 2004
    """
    # Compute local means using Gaussian-weighted convolution
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # Compute squares and products of means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants for numerical stability
    C1 = 0.01 ** 2  # (K1 * L)² where K1=0.01, L=1 (for normalized images)
    C2 = 0.03 ** 2  # (K2 * L)² where K2=0.03

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Return average or per-channel average
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) for 3D volumes.

    Same formulation as 2D SSIM but operates on 3D volumes:
    SSIM(x,y) = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    Args:
        img1: First volume tensor of shape (N, C, D, H, W)
        img2: Second volume tensor of shape (N, C, D, H, W)
        window: 3D Gaussian window for local statistics
        window_size: Size of the cubic window
        channel: Number of channels
        size_average: If True, return mean SSIM; else return per-voxel SSIM

    Returns:
        SSIM value(s) as tensor
    """
    # Compute local means using 3D Gaussian-weighted convolution
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    # Compute squares and products of means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants for numerical stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Return average or per-channel average
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ============================================================================
# SSIM Loss Classes (for use in training)
# ============================================================================

class SSIM(torch.nn.Module):
    """
    Structural Similarity Index as a loss function for 2D images.

    Returns the raw SSIM value (not negated, not 1-SSIM).
    Use 1 - SSIM or -SSIM depending on your optimization setup.

    Args:
        window_size: Size of the Gaussian window (default: 11)
        size_average: If True, return mean SSIM over batch

    Shape:
        - Input: (N, C, H, W) where N is batch size, C is channels
        - Target: (N, C, H, W)
        - Output: Scalar if size_average=True, else (N,)

    Example:
        >>> ssim_loss = SSIM(window_size=11)
        >>> pred = torch.randn(2, 3, 256, 256)
        >>> target = torch.randn(2, 3, 256, 256)
        >>> loss = 1 - ssim_loss(pred, target)  # Convert to loss
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between two images.

        Args:
            img1: Predicted image (N, C, H, W)
            img2: Target image (N, C, H, W)

        Returns:
            SSIM value (higher is better, range [-1, 1], typically [0, 1])
        """
        (_, channel, _, _) = img1.size()

        # Update window if channel count changed or device changed
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            # Move window to same device as input
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            # Cache the window
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM2D(torch.nn.Module):
    """
    Structural Similarity Index as a loss for 2D images.

    Returns 1 - SSIM as loss (so lower is better).

    Args:
        window_size: Size of the Gaussian window (default: 11)
        size_average: If True, return mean loss over batch

    Shape:
        - Input: (N, C, H, W)
        - Target: (N, C, H, W)
        - Output: Scalar loss value

    Example:
        >>> loss_fn = SSIM2D(window_size=11)
        >>> pred = torch.randn(2, 1, 256, 256)
        >>> target = torch.randn(2, 1, 256, 256)
        >>> loss = loss_fn(pred, target)  # Lower is better
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIM2D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            img1: Predicted image (N, C, H, W)
            img2: Target image (N, C, H, W)

        Returns:
            SSIM loss (lower is better)
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # Return as loss (1 - SSIM, so lower is better)
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    """
    Structural Similarity Index as a loss for 3D volumes.

    Returns 1 - SSIM as loss (so lower is better).

    Args:
        window_size: Size of the cubic Gaussian window (default: 11)
        size_average: If True, return mean loss over batch

    Shape:
        - Input: (N, C, D, H, W) where D is depth
        - Target: (N, C, D, H, W)
        - Output: Scalar loss value

    Example:
        >>> loss_fn = SSIM3D(window_size=11)
        >>> pred = torch.randn(1, 1, 64, 128, 128)
        >>> target = torch.randn(1, 1, 64, 128, 128)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D SSIM loss (1 - SSIM).

        Args:
            img1: Predicted volume (N, C, D, H, W)
            img2: Target volume (N, C, D, H, W)

        Returns:
            SSIM loss (lower is better)
        """
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # Return as loss (1 - SSIM)
        return 1 - _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


# ============================================================================
# SSIM Functional Interfaces
# ============================================================================

def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """
    Functional interface for 2D SSIM computation.

    Computes SSIM without requiring instantiation of a loss module.

    Args:
        img1: First image (N, C, H, W)
        img2: Second image (N, C, H, W)
        window_size: Size of Gaussian window
        size_average: Whether to average over batch

    Returns:
        SSIM value (higher is better)

    Example:
        >>> img1 = torch.randn(1, 3, 256, 256)
        >>> img2 = torch.randn(1, 3, 256, 256)
        >>> ssim_val = ssim(img1, img2)
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    # Move to same device as input
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """
    Functional interface for 3D SSIM computation.

    Args:
        img1: First volume (N, C, D, H, W)
        img2: Second volume (N, C, D, H, W)
        window_size: Size of cubic Gaussian window
        size_average: Whether to average over batch

    Returns:
        SSIM value (higher is better)
    """
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


# ============================================================================
# Stochastic SSIM Losses (S3IM)
# ============================================================================

class S3IMLoss(nn.Module):
    """
    Stochastic Structural Similarity (S3IM) Loss.

    S3IM randomly samples B pixels from the image, arranges them into a patch,
    and computes SSIM on these patches. This is repeated M times and averaged.

    Formula:
    S3IM = (1/M) Σ_i SSIM(patch_i^pred, patch_i^target)
    Loss = 1 - S3IM

    Args:
        B: Number of pixels to sample (must be a perfect square, default: 1024 = 32x32)
        M: Number of random samples to average (default: 10)
        window_size: SSIM window size (default: 4)

    Shape:
        - Input: (N, C, H, W)
        - Target: (N, C, H, W)
        - Output: Scalar loss

    Reference:
        Inspired by stochastic structural similarity approaches for efficient
        similarity computation on large images.

    Example:
        >>> loss_fn = S3IMLoss(B=1024, M=10)
        >>> pred = torch.randn(2, 3, 512, 512)
        >>> target = torch.randn(2, 3, 512, 512)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, B: int = 1024, M: int = 10, window_size: int = 4):
        super().__init__()
        self.B = B
        self.M = M
        self.window_size = window_size

        # B must be a perfect square so we can arrange pixels into a 2D patch
        self.patch_dim = int(B ** 0.5)
        if self.patch_dim * self.patch_dim != B:
            raise ValueError(
                f"B (number of sampled pixels) must be a perfect square, got {B}"
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute S3IM loss.

        Args:
            input: Predicted image (N, C, H, W)
            target: Target image (N, C, H, W)

        Returns:
            S3IM loss value
        """
        n_batch, n_channels, height, width = input.shape

        # Flatten spatial dimensions
        input_flat = input.view(n_batch * n_channels, height * width)
        target_flat = target.view(n_batch * n_channels, height * width)

        num_pixels_per_image = height * width

        # Create SSIM window once
        window = create_window(self.window_size, 1).to(input.device)

        ssim_values = []

        # Repeat M times with different random samples
        for _ in range(self.M):
            # Randomly sample B pixel indices
            rand_indices = torch.randperm(num_pixels_per_image, device=input.device)[:self.B]

            # Extract sampled pixels
            sampled_input = torch.index_select(input_flat, 1, rand_indices)
            sampled_target = torch.index_select(target_flat, 1, rand_indices)

            # Reshape into 2D patches
            input_patch = sampled_input.view(n_batch * n_channels, 1, self.patch_dim, self.patch_dim)
            target_patch = sampled_target.view(n_batch * n_channels, 1, self.patch_dim, self.patch_dim)

            # Compute SSIM on these patches
            ssim_val = _ssim(input_patch, target_patch, window, self.window_size,
                           channel=1, size_average=True)
            ssim_values.append(ssim_val)

        # Average over M samples
        avg_s3im = torch.mean(torch.stack(ssim_values))

        # Return as loss (1 - SSIM)
        return 1.0 - avg_s3im


class S3IMLossStitched(nn.Module):
    """
    S3IM Loss with Pre-Divided Patches (Stitched variant).

    This variant assumes the input is already divided into patches. It randomly
    selects a subset of these patches and stitches them into a pseudo-image,
    then computes SSIM.

    Args:
        patches_to_stitch: Number of patches to stitch (must be perfect square)
        M: Number of random stitchings to average
        window_size: SSIM window size

    Shape:
        - Input: (num_patches, C, patch_h, patch_w)
        - Target: (num_patches, C, patch_h, patch_w)
        - Output: Scalar loss

    Example:
        >>> loss_fn = S3IMLossStitched(patches_to_stitch=16, M=10)
        >>> # Assume we have 64 patches of size 3x16x16
        >>> pred_patches = torch.randn(64, 3, 16, 16)
        >>> target_patches = torch.randn(64, 3, 16, 16)
        >>> loss = loss_fn(pred_patches, target_patches)
    """

    def __init__(self, patches_to_stitch: int = 16, M: int = 10, window_size: int = 4):
        super().__init__()
        self.patches_to_stitch = patches_to_stitch
        self.M = M
        self.window_size = window_size

        # Must be perfect square to arrange into grid
        self.grid_dim = int(self.patches_to_stitch ** 0.5)
        if self.grid_dim * self.grid_dim != self.patches_to_stitch:
            raise ValueError(
                f"patches_to_stitch must be a perfect square, got {self.patches_to_stitch}"
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute stitched S3IM loss.

        Args:
            input: Predicted patches (num_patches, C, patch_h, patch_w)
            target: Target patches (num_patches, C, patch_h, patch_w)

        Returns:
            S3IM loss value
        """
        num_available_patches, n_channels, patch_h, patch_w = input.shape

        # Validate we have enough patches
        if num_available_patches < self.patches_to_stitch:
            raise ValueError(
                f"num_available_patches ({num_available_patches}) is less than "
                f"patches_to_stitch ({self.patches_to_stitch})"
            )

        # Create SSIM window
        window = create_window(self.window_size, 1).to(input.device)

        ssim_values = []

        # Repeat M times with different random selections
        for _ in range(self.M):
            # Randomly select patches
            rand_indices = torch.randperm(num_available_patches, device=input.device)[
                :self.patches_to_stitch
            ]

            selected_input_patches = input[rand_indices]
            selected_target_patches = target[rand_indices]

            # Split into individual patches
            list_of_input_patches = torch.chunk(selected_input_patches,
                                               chunks=self.patches_to_stitch, dim=0)
            list_of_target_patches = torch.chunk(selected_target_patches,
                                                chunks=self.patches_to_stitch, dim=0)

            # Stitch patches into rows
            input_rows = [
                torch.cat(list_of_input_patches[i*self.grid_dim:(i+1)*self.grid_dim], dim=3)
                for i in range(self.grid_dim)
            ]
            target_rows = [
                torch.cat(list_of_target_patches[i*self.grid_dim:(i+1)*self.grid_dim], dim=3)
                for i in range(self.grid_dim)
            ]

            # Stitch rows into full image
            stitched_input = torch.cat(input_rows, dim=2)
            stitched_target = torch.cat(target_rows, dim=2)

            # Reshape for SSIM computation
            stitched_input_reshaped = stitched_input.view(
                n_channels, 1, self.grid_dim * patch_h, self.grid_dim * patch_w
            )
            stitched_target_reshaped = stitched_target.view(
                n_channels, 1, self.grid_dim * patch_h, self.grid_dim * patch_w
            )

            # Compute SSIM
            ssim_val = _ssim(stitched_input_reshaped, stitched_target_reshaped,
                           window, self.window_size, channel=1, size_average=True)
            ssim_values.append(ssim_val)

        # Average over M samples
        avg_s3im = torch.mean(torch.stack(ssim_values))

        return 1.0 - avg_s3im


# ============================================================================
# Gradient-Based Regularization Losses
# ============================================================================

class Grad2d(torch.nn.Module):
    """
    2D Gradient Regularization Loss for smoothness.

    Penalizes spatial gradients to encourage smooth predictions. Useful for
    image registration, optical flow, and other tasks requiring spatial smoothness.

    The loss computes:
    L = (1/2) * (mean(|∂y/∂x|^p) + mean(|∂y/∂y|^p))

    where p=1 for L1 penalty and p=2 for L2 penalty.

    Args:
        penalty: Type of penalty ('l1' or 'l2')
        loss_mult: Optional multiplier for the loss

    Shape:
        - Input: (N, C, H, W)
        - Target: Not used (can be None)
        - Output: Scalar loss

    Example:
        >>> grad_loss = Grad2d(penalty='l1')
        >>> displacement = torch.randn(1, 2, 128, 128)  # 2D displacement field
        >>> loss = grad_loss(displacement, None)
    """

    def __init__(self, penalty: Literal['l1', 'l2'] = 'l1', loss_mult: Optional[float] = None):
        super(Grad2d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute 2D gradient loss.

        Args:
            y_pred: Predicted field (N, C, H, W)
            y_true: Not used (kept for API compatibility)

        Returns:
            Gradient loss value
        """
        # Compute spatial gradients using finite differences
        # dy: gradient in y-direction (vertical)
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        # dx: gradient in x-direction (horizontal)
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        # Apply penalty
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        # Average gradients
        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        # Apply multiplier if specified
        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


# Alias for backward compatibility
Grad = Grad2d


class Grad3d(torch.nn.Module):
    """
    3D Gradient Regularization Loss for volumetric smoothness.

    Extends 2D gradient loss to 3D volumes, penalizing gradients in all three
    spatial dimensions.

    The loss computes:
    L = (1/3) * (mean(|∂y/∂x|^p) + mean(|∂y/∂y|^p) + mean(|∂y/∂z|^p))

    where p=1 for L1 penalty and p=2 for L2 penalty.

    Args:
        penalty: Type of penalty ('l1' or 'l2')
        loss_mult: Optional multiplier for the loss

    Shape:
        - Input: (N, C, D, H, W)
        - Target: Not used
        - Output: Scalar loss

    Example:
        >>> grad_loss = Grad3d(penalty='l2')
        >>> displacement = torch.randn(1, 3, 64, 128, 128)  # 3D displacement
        >>> loss = grad_loss(displacement, None)
    """

    def __init__(self, penalty: Literal['l1', 'l2'] = 'l1', loss_mult: Optional[float] = None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute 3D gradient loss.

        Args:
            y_pred: Predicted field (N, C, D, H, W)
            y_true: Not used

        Returns:
            Gradient loss value
        """
        # Compute spatial gradients in all three dimensions
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        # Apply penalty
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        # Average gradients
        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        # Apply multiplier if specified
        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


class Grad3DiTV(torch.nn.Module):
    """
    3D Isotropic Total Variation (iTV) Loss.

    Computes the isotropic total variation by taking the L2 norm of gradients
    at each voxel, then averaging. This encourages piecewise-smooth solutions.

    Formula:
    L = (1/3) * mean(√(|∂y/∂x|² + |∂y/∂y|² + |∂y/∂z|² + ε))

    where ε=1e-6 for numerical stability.

    Shape:
        - Input: (N, C, D, H, W)
        - Target: Not used
        - Output: Scalar loss

    Note:
        This uses aligned gradients (all computed at same spatial location)
        to properly compute isotropic gradient magnitude.

    Example:
        >>> itv_loss = Grad3DiTV()
        >>> field = torch.randn(1, 3, 64, 128, 128)
        >>> loss = itv_loss(field, None)
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute 3D isotropic TV loss.

        Args:
            y_pred: Predicted field (N, C, D, H, W)
            y_true: Not used

        Returns:
            Isotropic TV loss value
        """
        # Compute gradients at aligned spatial locations
        # We crop to [1:, 1:, 1:] to ensure all gradients computed at same voxels
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])

        # Square the gradients
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

        # Compute L2 norm of gradient vector at each voxel
        d = torch.mean(torch.sqrt(dx + dy + dz + 1e-6))
        grad = d / 3.0

        return grad


class DisplacementRegularizer(torch.nn.Module):
    """
    Advanced Displacement Field Regularizer for Image Registration.

    Supports three types of regularization:
    1. 'gradient-l1': L1 norm of spatial gradients
    2. 'gradient-l2': L2 norm (squared) of spatial gradients
    3. 'bending': Bending energy (penalizes second derivatives)

    The bending energy is computed as:
    E_bend = mean(∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z² + 2(∂²T/∂x∂y + ∂²T/∂x∂z + ∂²T/∂y∂z))

    where T is the displacement field.

    Args:
        energy_type: Type of regularization ('gradient-l1', 'gradient-l2', or 'bending')

    Shape:
        - Input: (N, 3, D, H, W) - displacement field with 3 components
        - Target: Not used
        - Output: Scalar loss

    Reference:
        Modersitzki, J. "Numerical Methods for Image Registration" (2004)

    Example:
        >>> reg = DisplacementRegularizer(energy_type='bending')
        >>> displacement = torch.randn(1, 3, 64, 128, 128)
        >>> loss = reg(displacement, None)
    """

    def __init__(self, energy_type: Literal['gradient-l1', 'gradient-l2', 'bending']):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv: torch.Tensor) -> torch.Tensor:
        """Compute gradient in x-direction using central differences."""
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv: torch.Tensor) -> torch.Tensor:
        """Compute gradient in y-direction using central differences."""
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv: torch.Tensor) -> torch.Tensor:
        """Compute gradient in z-direction using central differences."""
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz: torch.Tensor, fn) -> torch.Tensor:
        """
        Apply gradient function to each component of displacement field.

        Args:
            Txyz: Displacement field (N, 3, D, H, W)
            fn: Gradient function to apply

        Returns:
            Gradients for each component stacked
        """
        return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement: torch.Tensor, flag_l1: bool = False) -> torch.Tensor:
        """
        Compute gradient norm regularization.

        Args:
            displacement: Displacement field (N, 3, D, H, W)
            flag_l1: If True, use L1 norm; else L2 norm

        Returns:
            Gradient norm loss
        """
        # Compute first-order spatial derivatives
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)

        if flag_l1:
            # L1 norm: sum of absolute values
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            # L2 norm: sum of squares
            norms = dTdx**2 + dTdy**2 + dTdz**2

        return torch.mean(norms) / 3.0

    def compute_bending_energy(self, displacement: torch.Tensor) -> torch.Tensor:
        """
        Compute bending energy (second derivative penalty).

        Bending energy penalizes non-affine deformations by penalizing
        second derivatives of the displacement field.

        Args:
            displacement: Displacement field (N, 3, D, H, W)

        Returns:
            Bending energy
        """
        # Compute first derivatives
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)

        # Compute second derivatives (diagonal terms)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)

        # Compute mixed second derivatives
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)

        # Bending energy: sum of squared second derivatives
        # Factor of 2 for mixed derivatives (symmetry)
        return torch.mean(
            dTdxx**2 + dTdyy**2 + dTdzz**2 +
            2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2
        )

    def forward(self, disp: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute regularization loss.

        Args:
            disp: Displacement field (N, 3, D, H, W)
            _: Placeholder for API compatibility

        Returns:
            Regularization loss
        """
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp, flag_l1=False)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise ValueError(
                f"Unrecognized regularizer type: {self.energy_type}. "
                f"Must be 'gradient-l1', 'gradient-l2', or 'bending'."
            )
        return energy


# ============================================================================
# Information Theory Losses (NCC, MI, MIND)
# ============================================================================

class NCC_vxm(torch.nn.Module):
    """
    Normalized Cross-Correlation (NCC) Loss.

    Computes local NCC over sliding windows. Works for 1D, 2D, and 3D inputs.

    Formula for local NCC:
    NCC(I,J) = Σ_w [(I - μ_I)(J - μ_J)] / √(Σ_w (I - μ_I)² · Σ_w (J - μ_J)²)

    where μ_I, μ_J are local means computed over window w.

    Args:
        win: Window size as list [w1, w2, ...] or None (default [9]*ndims)

    Shape:
        - Input: (N, C, *spatial_dims) where spatial_dims can be (H,W) or (D,H,W)
        - Target: Same as input
        - Output: Scalar loss (negative NCC, so minimizing increases similarity)

    Reference:
        Avants et al. "Symmetric diffeomorphic image registration with
        cross-correlation" Medical Image Analysis 2008

    Example:
        >>> ncc_loss = NCC_vxm(win=[9, 9, 9])
        >>> pred = torch.randn(1, 1, 64, 128, 128)
        >>> target = torch.randn(1, 1, 64, 128, 128)
        >>> loss = ncc_loss(target, pred)
    """

    def __init__(self, win: Optional[List[int]] = None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute NCC loss.

        Args:
            y_true: Target image/volume
            y_pred: Predicted image/volume

        Returns:
            Negative NCC (to minimize)
        """
        Ii = y_true
        Ji = y_pred

        # Determine dimensionality (1D, 2D, or 3D)
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], f"Supports 1D/2D/3D, found: {ndims}D"

        # Set window size (default 9 in each dimension)
        win = [9] * ndims if self.win is None else self.win

        # Create box filter (all ones) for computing local sums
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        # Set stride and padding based on dimensionality
        if ndims == 1:
            stride = (1,)
            padding = (pad_no,)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Get appropriate convolution function
        conv_fn = getattr(F, f'conv{ndims}d')

        # Compute local sums and cross products
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        # Compute local means
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # Compute local cross-correlation and variances
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Compute NCC (squared for stability)
        cc = cross * cross / (I_var * J_var + 1e-5)

        # Return negative mean (to minimize)
        return -torch.mean(cc)


class MIND_loss(torch.nn.Module):
    """
    Modality Independent Neighborhood Descriptor (MIND) Loss.

    MIND creates self-similarity descriptors that are robust to different
    imaging modalities (e.g., MRI T1 vs T2, CT vs MRI).

    The descriptor compares a voxel to its 6-neighborhood (adjacent voxels)
    using patch-based SSDs.

    Formula:
    MIND(x, δ) = exp(-SSD(x, x+δ) / V(x))

    where δ is an offset, SSD is sum of squared differences, and V is
    local variance estimate.

    Args:
        win: Not used (kept for API compatibility)

    Shape:
        - Input: (N, C, D, H, W) - 3D volumes only
        - Target: Same as input
        - Output: Scalar loss

    Reference:
        Heinrich et al. "MIND: Modality independent neighbourhood descriptor
        for multi-modal deformable registration" Medical Image Analysis 2012

    Note:
        This implementation is specifically for 3D volumes and uses
        a 6-neighborhood connectivity pattern.

    Example:
        >>> mind_loss = MIND_loss()
        >>> pred = torch.randn(1, 1, 64, 128, 128).cuda()
        >>> target = torch.randn(1, 1, 64, 128, 128).cuda()
        >>> loss = mind_loss(pred, target)
    """

    def __init__(self, win: Optional[int] = None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared distances.

        Args:
            x: Input tensor (N, D, K) where K is number of points

        Returns:
            Squared distance matrix (N, K, K)
        """
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0  # Replace NaN with 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img: torch.Tensor, radius: int = 2, dilation: int = 2) -> torch.Tensor:
        """
        Compute MIND Self-Similarity Context descriptor.

        Args:
            img: Input 3D volume (N, C, D, H, W)
            radius: Radius for patch averaging
            dilation: Dilation for neighborhood sampling

        Returns:
            MIND descriptor (N, 12, D, H, W) with 12 channels for 6-neighborhood pairs
        """
        # Define 6-neighborhood in 3D (face-adjacent voxels)
        six_neighbourhood = torch.Tensor([
            [0, 1, 1],  # Center
            [1, 1, 0],  # Left
            [1, 0, 1],  # Down
            [1, 1, 2],  # Right
            [2, 1, 1],  # Up
            [1, 2, 1]   # Forward
        ]).long()

        # Compute squared distances between neighborhood points
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # Create mask for pairs at distance 2 (adjacent faces)
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # Build kernels for comparing voxel pairs
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]

        # Create 3D convolution kernels for each pair
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(img.device)
        mshift1.view(-1)[
            torch.arange(12) * 27 +
            idx_shift1[:, 0] * 9 +
            idx_shift1[:, 1] * 3 +
            idx_shift1[:, 2]
        ] = 1

        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(img.device)
        mshift2.view(-1)[
            torch.arange(12) * 27 +
            idx_shift2[:, 0] * 9 +
            idx_shift2[:, 1] * 3 +
            idx_shift2[:, 2]
        ] = 1

        # Padding layers
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # Compute patch-based SSD between pairs
        kernel_size = radius * 2 + 1
        ssd = F.avg_pool3d(
            rpad2(
                (F.conv3d(rpad1(img), mshift1, dilation=dilation) -
                 F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2
            ),
            kernel_size, stride=1
        )

        # Apply MIND transformation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        # Clamp variance for stability
        mind_var = torch.clamp(
            mind_var,
            (mind_var.mean() * 0.001).item(),
            (mind_var.mean() * 1000).item()
        )
        mind /= mind_var
        mind = torch.exp(-mind)

        # Reorder channels to match C++ implementation
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute MIND loss (MSE between MIND descriptors).

        Args:
            y_pred: Predicted volume
            y_true: Target volume

        Returns:
            MIND loss value
        """
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class MutualInformation(torch.nn.Module):
    """
    Mutual Information (MI) Loss.

    Mutual Information measures statistical dependence between two images.
    Higher MI means more similar intensity distributions.

    Formula:
    MI(X,Y) = Σ_x Σ_y p(x,y) log(p(x,y) / (p(x)p(y)))

    This implementation approximates histograms using Gaussian kernels.

    Args:
        sigma_ratio: Ratio for Gaussian kernel width (default: 1)
        minval: Minimum intensity value (default: 0)
        maxval: Maximum intensity value (default: 1)
        num_bin: Number of histogram bins (default: 32)

    Shape:
        - Input: (N, C, *spatial) - any dimensionality
        - Target: Same as input
        - Output: Scalar loss (negative MI, to minimize)

    Reference:
        Mattes et al. "PET-CT image registration in the chest using free-form
        deformations" IEEE TMI 2003

    Example:
        >>> mi_loss = MutualInformation(num_bin=64)
        >>> pred = torch.randn(1, 1, 128, 128, 128).cuda()
        >>> target = torch.randn(1, 1, 128, 128, 128).cuda()
        >>> loss = mi_loss(target, pred)
    """

    def __init__(
        self,
        sigma_ratio: float = 1,
        minval: float = 0.,
        maxval: float = 1.,
        num_bin: int = 32
    ):
        super(MutualInformation, self).__init__()

        # Create bin centers for histogram
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(
            torch.linspace(minval, maxval, num_bin),
            requires_grad=False
        )
        num_bins = len(bin_centers)

        # Compute Gaussian kernel width
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information.

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            MI value (higher is better)
        """
        # Clamp values to valid range
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0., self.max_clip)

        # Flatten spatial dimensions
        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]

        # Reshape bin centers for broadcasting
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).to(y_pred.device)

        # Compute soft histogram using Gaussian kernel
        # I_a[n, v, b] represents probability that voxel v belongs to bin b
        I_a = torch.exp(-self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(-self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # Compute joint and marginal probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)  # Joint histogram
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)  # Marginal for image A
        pb = torch.mean(I_b, dim=1, keepdim=True)  # Marginal for image B

        # Compute MI
        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)

        return mi.mean()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute MI loss (negative MI).

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Negative MI (to minimize)
        """
        return -self.mi(y_true, y_pred)


class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for Non-Overlapping Patches.

    Computes MI on non-overlapping patches instead of the whole image.
    This can be more robust to local intensity variations.

    Args:
        sigma_ratio: Ratio for Gaussian kernel width
        minval: Minimum intensity value
        maxval: Maximum intensity value
        num_bin: Number of histogram bins
        patch_size: Size of non-overlapping patches

    Shape:
        - Input: (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        - Target: Same as input
        - Output: Scalar loss

    Example:
        >>> lmi_loss = localMutualInformation(patch_size=7)
        >>> pred = torch.randn(1, 1, 64, 128, 128).cuda()
        >>> target = torch.randn(1, 1, 64, 128, 128).cuda()
        >>> loss = lmi_loss(target, pred)
    """

    def __init__(
        self,
        sigma_ratio: float = 1,
        minval: float = 0.,
        maxval: float = 1.,
        num_bin: int = 32,
        patch_size: int = 5
    ):
        super(localMutualInformation, self).__init__()

        # Create bin centers
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(
            torch.linspace(minval, maxval, num_bin),
            requires_grad=False
        )
        num_bins = len(bin_centers)

        # Compute Gaussian width
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute local MI on patches.

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Average MI across patches
        """
        # Clamp intensities
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0., self.max_clip)

        # Reshape bin centers
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).to(y_pred.device)

        # Determine dimensionality and compute padding
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2,
                      y_r // 2, y_r - y_r // 2,
                      x_r // 2, x_r - x_r // 2,
                      0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2,
                      x_r // 2, x_r - x_r // 2,
                      0, 0, 0, 0)
        else:
            raise ValueError(f'Supports 2D and 3D, got {len(list(y_pred.size())[2:])}D')

        # Apply padding
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        # Reshape into non-overlapping patches
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (
                y_true.shape[0], y_true.shape[1],
                (x + x_r) // self.patch_size, self.patch_size,
                (y + y_r) // self.patch_size, self.patch_size,
                (z + z_r) // self.patch_size, self.patch_size
            ))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (
                y_pred.shape[0], y_pred.shape[1],
                (x + x_r) // self.patch_size, self.patch_size,
                (y + y_r) // self.patch_size, self.patch_size,
                (z + z_r) // self.patch_size, self.patch_size
            ))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (
                y_true.shape[0], y_true.shape[1],
                (x + x_r) // self.patch_size, self.patch_size,
                (y + y_r) // self.patch_size, self.patch_size
            ))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (
                y_pred.shape[0], y_pred.shape[1],
                (x + x_r) // self.patch_size, self.patch_size,
                (y + y_r) // self.patch_size, self.patch_size
            ))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        # Compute MI for each patch
        I_a_patch = torch.exp(-self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(-self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        # Compute joint and marginal distributions
        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        # Compute MI
        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)

        return mi.mean()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute local MI loss.

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Negative local MI
        """
        return -self.local_mi(y_true, y_pred)


# ============================================================================
# Dice and Segmentation Losses
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for Image Segmentation.

    Dice coefficient measures overlap between predicted and ground truth:
    Dice = 2|X ∩ Y| / (|X| + |Y|)

    This implementation supports two modes:
    1. Binary/continuous: Direct overlap computation (smooth=1e-6 for stability)
    2. Multi-class: One-hot encoding with num_class parameter

    Args:
        smooth: Smoothing factor for numerical stability (default: 1e-6)
        num_class: If provided, treats as multi-class segmentation with one-hot encoding

    Shape:
        - Binary mode:
            - Input: (N, C, *spatial) - probabilities or logits
            - Target: (N, C, *spatial) - binary or continuous labels
        - Multi-class mode:
            - Input: (N, num_class, D, H, W) - class probabilities
            - Target: (N, 1, D, H, W) - integer class labels
        - Output: Scalar loss

    Example:
        >>> # Binary segmentation
        >>> dice_loss = DiceLoss(smooth=1e-6)
        >>> pred = torch.sigmoid(torch.randn(2, 1, 128, 128))
        >>> target = torch.randint(0, 2, (2, 1, 128, 128)).float()
        >>> loss = dice_loss(pred, target)
        >>>
        >>> # Multi-class segmentation
        >>> dice_loss = DiceLoss(num_class=5)
        >>> pred = torch.softmax(torch.randn(2, 5, 64, 128, 128), dim=1)
        >>> target = torch.randint(0, 5, (2, 1, 64, 128, 128))
        >>> loss = dice_loss(pred, target)
    """

    def __init__(self, smooth: float = 1e-6, num_class: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.num_class = num_class

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            input: Predicted segmentation
            target: Ground truth segmentation

        Returns:
            Dice loss (1 - Dice coefficient)
        """
        input = input.float()

        # Multi-class mode with one-hot encoding
        if self.num_class is not None:
            # Convert target to one-hot
            target = nn.functional.one_hot(target.long(), num_classes=self.num_class)
            target = torch.squeeze(target, 1)
            target = target.permute(0, 4, 1, 2, 3).contiguous()
            target = target.float()

            # Compute Dice for each class and average
            intersection = input * target
            intersection = intersection.sum(dim=[2, 3, 4])
            union = torch.pow(input, 2).sum(dim=[2, 3, 4]) + \
                    torch.pow(target, 2).sum(dim=[2, 3, 4])
            dice = (2. * intersection) / (union + 1e-5)
            dice = 1 - torch.mean(dice)
        else:
            # Binary/continuous mode
            target = target.float()

            # Compute intersection and union
            intersection = (input * target).sum()
            total_area = input.sum() + target.sum()

            # Compute Dice coefficient
            dice = (2. * intersection + self.smooth) / (total_area + self.smooth)
            dice = 1 - dice

        return dice


# ============================================================================
# Pixel-wise and Perceptual Losses
# ============================================================================

class LogMSELoss(nn.Module):
    """
    Logarithm of Mean Squared Error Loss.

    Formula:
    L = log(MSE(x, y) + ε) = log((1/N)Σ(x - y)² + ε)

    Taking the log can help with optimization dynamics by reducing the
    scale of the loss and gradients.

    Args:
        eps: Small constant for numerical stability (default: 1e-6)

    Shape:
        - Input: Any shape (N, C, *spatial)
        - Target: Same as input
        - Output: Scalar loss

    Example:
        >>> log_mse = LogMSELoss(eps=1e-6)
        >>> pred = torch.randn(2, 3, 256, 256)
        >>> target = torch.randn(2, 3, 256, 256)
        >>> loss = log_mse(pred, target)
    """

    def __init__(self, eps: float = 1e-6):
        super(LogMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute log MSE loss.

        Args:
            input: Predicted tensor
            target: Target tensor

        Returns:
            Log of MSE
        """
        mse_loss = self.mse(input, target)
        log_mse_loss = torch.log(mse_loss + self.eps)
        return log_mse_loss


class LpipsLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Loss.

    Uses a pretrained deep network (VGG, AlexNet, or SqueezeNet) to compute
    perceptual similarity. More aligned with human perception than pixel-wise metrics.

    Args:
        net: Network architecture ('vgg', 'alex', or 'squeeze')
        use_gpu: Whether to use GPU

    Shape:
        - Input: (N, 3, H, W) - RGB images in range [-1, 1]
        - Target: (N, 3, H, W)
        - Output: Scalar loss

    Reference:
        Zhang et al. "The Unreasonable Effectiveness of Deep Features as a
        Perceptual Metric" CVPR 2018

    Note:
        Requires 'lpips' package: pip install lpips

    Example:
        >>> lpips_loss = LpipsLoss(net='vgg')
        >>> pred = torch.randn(2, 3, 256, 256) * 2 - 1  # Range [-1, 1]
        >>> target = torch.randn(2, 3, 256, 256) * 2 - 1
        >>> loss = lpips_loss(pred, target)
    """

    def __init__(self, net: str = 'vgg', use_gpu: bool = True):
        super(LpipsLoss, self).__init__()

        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips package not found. Install with: pip install lpips"
            )

        self.lpips_loss = lpips.LPIPS(net=net)
        if use_gpu:
            self.lpips_loss = self.lpips_loss.cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS loss.

        Args:
            input: Predicted image (N, 3, H, W) in [-1, 1]
            target: Target image (N, 3, H, W) in [-1, 1]

        Returns:
            Average LPIPS distance
        """
        lpips_loss_value = self.lpips_loss(input, target)
        return lpips_loss_value.mean()


class RangeInvariantPSNRLoss(nn.Module):
    """
    Range-Invariant Peak Signal-to-Noise Ratio Loss.

    This variant of PSNR normalizes both prediction and target to have zero mean
    and scales the prediction to match the target's range before computing PSNR.
    This makes it invariant to linear intensity transformations.

    Formula:
    1. Zero-mean: x' = x - mean(x)
    2. Scale match: x'' = x' * (std(target) / std(x'))
    3. PSNR = 20 * log10(range / RMSE)

    Args:
        eps: Small constant for numerical stability (default: 1e-8)

    Shape:
        - Input: (N, *any_shape)
        - Target: Same as input
        - Output: Scalar loss (negative PSNR, to minimize)

    Example:
        >>> ri_psnr = RangeInvariantPSNRLoss()
        >>> pred = torch.randn(2, 3, 256, 256)
        >>> target = torch.randn(2, 3, 256, 256)
        >>> loss = ri_psnr(pred, target)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _zero_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Remove mean from each sample."""
        return x - torch.mean(x, dim=1, keepdim=True)

    def _fix_range(self, gt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scale x to match range of gt using least squares."""
        # Compute optimal scale factor: a = (gt · x) / (x · x)
        a = torch.sum(gt * x, dim=1, keepdim=True) / \
            (torch.sum(x * x, dim=1, keepdim=True) + self.eps)
        return x * a

    def _fix(self, gt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Zero-mean and scale match."""
        gt_ = self._zero_mean(gt)
        return self._fix_range(gt_, self._zero_mean(x))

    def _psnr_internal(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        range_: torch.Tensor
    ) -> torch.Tensor:
        """Compute PSNR given normalized inputs and range."""
        mse = torch.mean((gt - pred) ** 2, dim=1)
        return 20 * torch.log10(range_ / torch.sqrt(mse + self.eps))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute range-invariant PSNR loss.

        Args:
            input: Predicted tensor
            target: Target tensor

        Returns:
            Negative PSNR (lower is better)
        """
        assert len(input.shape) > 1, "Input must have at least 2 dimensions"

        # Flatten all dimensions except batch
        original_shape = input.shape
        input_flat = input.view(original_shape[0], -1)
        target_flat = target.view(original_shape[0], -1)

        # Normalize target to zero mean and unit std
        target_std = torch.std(target_flat, dim=1, keepdim=True)
        gt_ = self._zero_mean(target_flat) / (target_std + self.eps)

        # Fix prediction to match target
        pred_fixed = self._fix(gt_, input_flat)

        # Compute range as ratio of range to std
        ra = (torch.max(target_flat, dim=1).values -
              torch.min(target_flat, dim=1).values) / \
             (target_std.squeeze() + self.eps)

        # Compute PSNR
        psnr = self._psnr_internal(gt_, pred_fixed, ra)

        # Return negative (to minimize)
        return -psnr.mean()


# ============================================================================
# Image Fusion Losses (for multi-modal fusion)
# ============================================================================

def to_gray3d(img: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to grayscale for 3D volumes.

    Uses standard luminance conversion:
    Gray = 0.2989*R + 0.587*G + 0.114*B

    Args:
        img: RGB volume (N, 3, D, H, W)

    Returns:
        Grayscale volume (N, 1, D, H, W)

    Example:
        >>> rgb = torch.randn(1, 3, 64, 128, 128)
        >>> gray = to_gray3d(rgb)
        >>> assert gray.shape == (1, 1, 64, 128, 128)
    """
    r, g, b = img.unbind(dim=-4)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-4)
    return l_img


class SobelxyRGB3D(nn.Module):
    """
    3D Sobel Edge Detector for RGB Volumes.

    Applies 3D Sobel filters to detect edges in x, y, and z directions.
    Useful for computing gradient-based fusion losses.

    Args:
        isSignGrad: If True, return signed gradients; else absolute values

    Shape:
        - Input: (N, 3, D, H, W)
        - Output: (N, 3, D, H, W) - gradient magnitudes

    Example:
        >>> sobel = SobelxyRGB3D(isSignGrad=False)
        >>> img = torch.randn(1, 3, 64, 128, 128)
        >>> edges = sobel(img)
    """

    def __init__(self, isSignGrad: bool = True):
        super(SobelxyRGB3D, self).__init__()
        self.isSignGrad = isSignGrad

        # 3D Sobel kernels for x, y, z directions
        kernelx = [
            [[0.2, 0, -0.2], [1, 0, -1], [0.2, 0, -0.2]],
            [[1, 0, -1], [4, 0, -4], [1, 0, -1]],
            [[0.2, 0, -0.2], [1, 0, -1], [0.2, 0, -0.2]]
        ]
        kernely = [
            [[0.2, 1, 0.2], [0, 0, 0], [-0.2, -1, -0.2]],
            [[1, 4, 1], [0, 0, 0], [-1, -4, -1]],
            [[0.2, 1, 0.2], [0, 0, 0], [-0.2, -1, -0.2]]
        ]
        kernelz = [
            [[0.2, 1, 0.2], [1, 4, 1], [0.2, 1, 0.2]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-0.2, -1, -0.2], [-1, -4, -1], [-0.2, -1, -0.2]]
        ]

        # Convert to parameters (non-trainable)
        self.weightx = nn.Parameter(
            torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1),
            requires_grad=False
        )
        self.weighty = nn.Parameter(
            torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1),
            requires_grad=False
        )
        self.weightz = nn.Parameter(
            torch.FloatTensor(kernelz).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1),
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D Sobel filter.

        Args:
            x: Input volume (N, 3, D, H, W)

        Returns:
            Gradient map (N, 3, D, H, W)
        """
        sobelx = F.conv3d(x, self.weightx, padding=1)
        sobely = F.conv3d(x, self.weighty, padding=1)
        sobelz = F.conv3d(x, self.weightz, padding=1)

        if self.isSignGrad:
            return sobelx + sobely + sobelz
        else:
            return torch.abs(sobelx) + torch.abs(sobely) + torch.abs(sobelz)


class MaxGradLoss3D(nn.Module):
    """
    Maximum Gradient Loss for Image Fusion.

    Encourages the fused image to preserve the maximum gradient from source images.
    This helps maintain edge sharpness from both modalities.

    Formula (when two sources):
    L = ||∇F - max(|∇A|, |∇B|)||₁

    where F is fused image, A and B are source images.

    Args:
        loss_weight: Multiplier for the loss (default: 1.0)
        isSignGrad: Whether to use signed gradients

    Shape:
        - Fusion: (N, C, D, H, W) - fused image
        - Source 1: (N, C, D, H, W) - first source
        - Source 2: (N, C, D, H, W) or None - second source (optional)
        - Output: Scalar loss

    Example:
        >>> loss_fn = MaxGradLoss3D(loss_weight=1.0)
        >>> fused = torch.randn(1, 1, 64, 128, 128)
        >>> source1 = torch.randn(1, 1, 64, 128, 128)
        >>> source2 = torch.randn(1, 1, 64, 128, 128)
        >>> loss = loss_fn(fused, source1, source2)
    """

    def __init__(self, loss_weight: float = 1.0, isSignGrad: bool = True):
        super(MaxGradLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = SobelxyRGB3D(isSignGrad)
        self.L1_loss = nn.L1Loss()

    def forward(
        self,
        im_fusion: torch.Tensor,
        im_rgb: torch.Tensor,
        im_tir: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute max gradient loss.

        Args:
            im_fusion: Fused image
            im_rgb: First source image
            im_tir: Second source image (optional)

        Returns:
            Max gradient loss
        """
        # Ensure all inputs have 3 channels for Sobel
        if im_fusion.shape[1] == 1:
            im_fusion = im_fusion.repeat(1, 3, 1, 1, 1)
        if im_rgb.shape[1] == 1:
            im_rgb = im_rgb.repeat(1, 3, 1, 1, 1)
        if im_tir is not None and im_tir.shape[1] == 1:
            im_tir = im_tir.repeat(1, 3, 1, 1, 1)

        if im_tir is not None:
            # Two-source fusion: use maximum gradient
            rgb_grad = self.sobelconv(im_rgb)
            tir_grad = self.sobelconv(im_tir)

            # Create mask for maximum gradients
            mask = torch.ge(torch.abs(rgb_grad), torch.abs(tir_grad))
            max_grad_joint = tir_grad.masked_fill_(mask, 0) + \
                           rgb_grad.masked_fill_(~mask, 0)

            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
        else:
            # Single-source: match source gradient
            rgb_grad = self.sobelconv(im_rgb)
            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad, rgb_grad)

        loss_grad = self.loss_weight * sobel_loss
        return loss_grad


class MaxPixelLoss3D(nn.Module):
    """
    Maximum Pixel Intensity Loss for Image Fusion.

    Encourages fused image to match the maximum pixel intensity from sources.

    Formula:
    L = ||F - max(A, B)||₁

    Args:
        loss_weight: Multiplier for the loss

    Shape:
        - Input: (N, C, D, H, W)
        - Output: Scalar loss

    Example:
        >>> loss_fn = MaxPixelLoss3D(loss_weight=1.0)
        >>> fused = torch.randn(1, 1, 64, 128, 128)
        >>> source1 = torch.randn(1, 1, 64, 128, 128)
        >>> source2 = torch.randn(1, 1, 64, 128, 128)
        >>> loss = loss_fn(fused, source1, source2)
    """

    def __init__(self, loss_weight: float = 1.0):
        super(MaxPixelLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(
        self,
        im_fusion: torch.Tensor,
        im_rgb: torch.Tensor,
        im_tir: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute max pixel loss.

        Args:
            im_fusion: Fused image
            im_rgb: First source
            im_tir: Second source (optional)

        Returns:
            Max pixel loss
        """
        if im_tir is not None:
            pixel_max = torch.max(im_rgb, im_tir).detach()
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, pixel_max)
        else:
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, im_rgb)

        return pixel_loss


class PixelLoss3D(nn.Module):
    """
    Average Pixel Intensity Loss for Image Fusion.

    Encourages fused image to match the average of source intensities.

    Formula:
    L = ||F - (A + B)/2||₁

    Args:
        loss_weight: Multiplier for the loss

    Shape:
        - Input: (N, C, D, H, W)
        - Output: Scalar loss

    Example:
        >>> loss_fn = PixelLoss3D(loss_weight=1.0)
        >>> fused = torch.randn(1, 1, 64, 128, 128)
        >>> source1 = torch.randn(1, 1, 64, 128, 128)
        >>> source2 = torch.randn(1, 1, 64, 128, 128)
        >>> loss = loss_fn(fused, source1, source2)
    """

    def __init__(self, loss_weight: float = 1.0):
        super(PixelLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(
        self,
        im_fusion: torch.Tensor,
        im_rgb: torch.Tensor,
        im_tir: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute average pixel loss.

        Args:
            im_fusion: Fused image
            im_rgb: First source
            im_tir: Second source (optional)

        Returns:
            Average pixel loss
        """
        if im_tir is not None:
            pixel_mean = (im_rgb + im_tir) / 2.0
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, pixel_mean)
        else:
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, im_rgb)

        return pixel_loss


class MaxGradTokenSelect3D(nn.Module):
    """
    Gradient-Based Token Selection for Vision Transformer Fusion.

    Selects patches from source images based on gradient magnitude.
    For each patch, chooses the source with larger gradient.

    This is not a loss function but a selection mechanism used during fusion.

    Args:
        loss_weight: Not used (kept for API consistency)

    Shape:
        - Input: (N, 3, D, H, W) where D, H, W are divisible by 16
        - Output: (N, 3, D, H, W) - fused image

    Note:
        Requires input dimensions to be multiples of 16 (patch size).

    Example:
        >>> selector = MaxGradTokenSelect3D()
        >>> source1 = torch.randn(1, 3, 64, 128, 128)
        >>> source2 = torch.randn(1, 3, 64, 128, 128)
        >>> fused = selector(source1, source2)
    """

    def __init__(self, loss_weight: float = 1.0):
        super(MaxGradTokenSelect3D, self).__init__()
        self.sobelconv = SobelxyRGB3D()

    def patchify3d(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Divide image into non-overlapping 3D patches.

        Args:
            imgs: Input volume (N, C, D, H, W)

        Returns:
            - Patches: (N, num_patches, patch_volume*C)
            - Info: (D, H, W, C) for reconstruction
        """
        p = 16  # Patch size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0

        C, D, H, W = imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]

        # Reshape into patches
        x = imgs.reshape(shape=(imgs.shape[0], C, D//p, p, H//p, p, W//p, p))
        x = torch.einsum('ncdphqwr->ndhwpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], D//p * H//p * W//p, p*p*p*C))

        return x, (D, H, W, C)

    def unpatchify3d(self, x: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Reconstruct image from patches.

        Args:
            x: Patches (N, num_patches, patch_volume*C)
            shape: Original shape info (D, H, W, C)

        Returns:
            Reconstructed volume (N, C, D, H, W)
        """
        p = 16
        D, H, W, C = shape
        assert D * H * W // (p**3) == x.shape[1]

        x = x.reshape(shape=(x.shape[0], D//p, H//p, W//p, p, p, p, C))
        x = torch.einsum('ndhwpqrc->ncdphqwr', x)
        imgs = x.reshape(shape=(x.shape[0], C, D, H, W))

        return imgs

    def forward(self, im_rgb: torch.Tensor, im_tir: torch.Tensor) -> torch.Tensor:
        """
        Select patches based on gradient magnitude.

        Args:
            im_rgb: First source (N, 3, D, H, W)
            im_tir: Second source (N, 3, D, H, W)

        Returns:
            Fused image with patches from higher-gradient source
        """
        # Convert to grayscale and compute gradients
        im_rgb_gray = to_gray3d(im_rgb)
        im_tir_gray = to_gray3d(im_tir)
        rgb_grad = self.sobelconv(im_rgb_gray)
        tir_grad = self.sobelconv(im_tir_gray)

        # Patchify images and gradients
        im_rgb, info = self.patchify3d(im_rgb)
        im_tir, _ = self.patchify3d(im_tir)
        rgb_grad_patch, info_grad = self.patchify3d(rgb_grad)
        tir_grad_patch, _ = self.patchify3d(tir_grad)

        # Compute max gradient per patch
        rgb_grad, _ = torch.max(rgb_grad_patch, -1)
        tir_grad, _ = torch.max(tir_grad_patch, -1)

        # Select patches with higher gradient
        AB_mask = (rgb_grad >= tir_grad).unsqueeze(dim=-1)
        out = torch.where(AB_mask, im_rgb, im_tir)

        # Reconstruct image
        out = self.unpatchify3d(out, info)

        return out


# ============================================================================
# Loss Factory Function
# ============================================================================

def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.

    This provides a convenient interface for instantiating loss functions
    from configuration files or command-line arguments.

    Args:
        name: Name of the loss function (case-insensitive)
        **kwargs: Parameters to pass to the loss function constructor

    Returns:
        Instantiated loss function module

    Raises:
        ValueError: If loss function name is not recognized

    Available losses:
        SSIM-based:
            - 'ssim', 'ssim2d': SSIM loss for 2D images
            - 'ssim3d': SSIM loss for 3D volumes
            - 's3im': Stochastic SSIM loss
            - 's3im_stitched': Stitched variant of S3IM

        Gradient-based:
            - 'grad', 'grad2d': 2D gradient regularization
            - 'grad3d': 3D gradient regularization
            - 'grad3d_itv': 3D isotropic total variation
            - 'displacement_reg': Displacement field regularizer

        Information theory:
            - 'ncc', 'ncc_vxm': Normalized cross-correlation
            - 'mi', 'mutual_information': Mutual information
            - 'local_mi': Local mutual information
            - 'mind': MIND descriptor loss

        Segmentation:
            - 'dice': Dice loss

        Pixel-wise:
            - 'mse': Mean squared error (from torch.nn)
            - 'l1': L1 loss (from torch.nn)
            - 'log_mse': Logarithmic MSE
            - 'lpips': Perceptual LPIPS loss
            - 'ri_psnr': Range-invariant PSNR

        Fusion:
            - 'max_grad_3d': Maximum gradient loss
            - 'max_pixel_3d': Maximum pixel loss
            - 'pixel_3d': Average pixel loss

    Examples:
        >>> # Basic usage
        >>> loss = get_loss_function('ssim2d', window_size=11)
        >>>
        >>> # With parameters
        >>> loss = get_loss_function('grad3d', penalty='l2', loss_mult=0.1)
        >>>
        >>> # For segmentation
        >>> loss = get_loss_function('dice', num_class=5)
        >>>
        >>> # Combining with PyTorch losses
        >>> mse_loss = get_loss_function('mse')
    """
    # Normalize name
    name = name.lower().strip()

    # SSIM-based losses
    if name in ['ssim', 'ssim2d']:
        return SSIM2D(**kwargs)
    elif name == 'ssim3d':
        return SSIM3D(**kwargs)
    elif name == 's3im':
        return S3IMLoss(**kwargs)
    elif name in ['s3im_stitched', 's3im_stitch']:
        return S3IMLossStitched(**kwargs)

    # Gradient-based losses
    elif name in ['grad', 'grad2d']:
        return Grad2d(**kwargs)
    elif name == 'grad3d':
        return Grad3d(**kwargs)
    elif name in ['grad3d_itv', 'itv']:
        return Grad3DiTV(**kwargs)
    elif name in ['displacement_reg', 'disp_reg']:
        return DisplacementRegularizer(**kwargs)

    # Information theory losses
    elif name in ['ncc', 'ncc_vxm']:
        return NCC_vxm(**kwargs)
    elif name in ['mi', 'mutual_information']:
        return MutualInformation(**kwargs)
    elif name in ['local_mi', 'lmi']:
        return localMutualInformation(**kwargs)
    elif name == 'mind':
        return MIND_loss(**kwargs)

    # Segmentation losses
    elif name == 'dice':
        return DiceLoss(**kwargs)

    # Pixel-wise losses
    elif name == 'log_mse':
        return LogMSELoss(**kwargs)
    elif name == 'lpips':
        return LpipsLoss(**kwargs)
    elif name in ['ri_psnr', 'range_invariant_psnr']:
        return RangeInvariantPSNRLoss(**kwargs)
    elif name == 'mse':
        return nn.MSELoss(**kwargs)
    elif name == 'l1':
        return nn.L1Loss(**kwargs)

    # Fusion losses
    elif name in ['max_grad_3d', 'max_grad']:
        return MaxGradLoss3D(**kwargs)
    elif name in ['max_pixel_3d', 'max_pixel']:
        return MaxPixelLoss3D(**kwargs)
    elif name in ['pixel_3d', 'pixel']:
        return PixelLoss3D(**kwargs)

    else:
        raise ValueError(
            f"Unknown loss function: {name}\n"
            f"Available losses: ssim, ssim2d, ssim3d, s3im, s3im_stitched, "
            f"grad, grad2d, grad3d, grad3d_itv, displacement_reg, "
            f"ncc, mi, local_mi, mind, dice, log_mse, lpips, ri_psnr, mse, l1, "
            f"max_grad_3d, max_pixel_3d, pixel_3d"
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Utility functions
    'gaussian',
    'create_window',
    'create_window_3D',
    'to_gray3d',

    # SSIM functions and classes
    'ssim',
    'ssim3D',
    '_ssim',
    '_ssim_3D',
    'SSIM',
    'SSIM2D',
    'SSIM3D',
    'S3IMLoss',
    'S3IMLossStitched',

    # Gradient losses
    'Grad',
    'Grad2d',
    'Grad3d',
    'Grad3DiTV',
    'DisplacementRegularizer',

    # Information theory losses
    'NCC_vxm',
    'MIND_loss',
    'MutualInformation',
    'localMutualInformation',

    # Segmentation losses
    'DiceLoss',

    # Pixel-wise losses
    'LogMSELoss',
    'LpipsLoss',
    'RangeInvariantPSNRLoss',

    # Fusion losses
    'SobelxyRGB3D',
    'MaxGradLoss3D',
    'MaxPixelLoss3D',
    'PixelLoss3D',
    'MaxGradTokenSelect3D',

    # Factory function
    'get_loss_function',
]
