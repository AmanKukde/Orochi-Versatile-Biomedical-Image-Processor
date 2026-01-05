"""Loss functions for biomedical image processing tasks.

This module implements various loss functions for different biomedical imaging tasks:
- Reconstruction losses: SSIM for image quality assessment
- Registration losses: NCC, Mutual Information, MIND for image alignment
- Regularization losses: Gradient-based smoothness, displacement regularizers
- Fusion losses: Gradient and pixel-based fusion objectives

The losses support both 2D and 3D images and are designed to work with GPU acceleration.

Classes:
    SSIM: Structural Similarity Index Measure for 2D images
    SSIM3D: Structural Similarity Index Measure for 3D volumes
    Grad: 2D gradient smoothness regularization
    Grad3d: 3D gradient smoothness regularization
    Grad3DiTV: 3D isotropic total variation
    DisplacementRegularizer: Various regularization energies for displacement fields
    NCC_vxm: Normalized Cross-Correlation from VoxelMorph
    MIND_loss: Modality Independent Neighborhood Descriptor
    MutualInformation: Global mutual information for multi-modal registration
    localMutualInformation: Local mutual information for patch-based registration
    SobelxyRGB3D: 3D Sobel gradient operator
    MaxGradLoss3D: Maximum gradient loss for image fusion
    MaxPixelLoss3D: Maximum pixel intensity loss for fusion
    PixelLoss3D: Average pixel intensity loss for fusion
    MaxGradTokenSelect3D: Gradient-based token selection for fusion
"""

# Standard library imports
import math
from math import exp

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Constants for SSIM
SSIM_C1 = 0.01 ** 2  # Stability constant for luminance comparison
SSIM_C2 = 0.03 ** 2  # Stability constant for contrast comparison
GAUSSIAN_SIGMA = 1.5  # Sigma for Gaussian window


def gaussian(window_size, sigma):
    """Create 1D Gaussian kernel.

    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian

    Returns:
        Normalized 1D Gaussian kernel
    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM.

    Args:
        window_size: Size of the window (square)
        channel: Number of channels to expand the window for

    Returns:
        2D Gaussian window tensor of shape (channel, 1, window_size, window_size)
    """
    _1D_window = gaussian(window_size, GAUSSIAN_SIGMA).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def create_window_3D(window_size, channel):
    """Create 3D Gaussian window for SSIM.

    Args:
        window_size: Size of the window (cubic)
        channel: Number of channels to expand the window for

    Returns:
        3D Gaussian window tensor of shape (channel, 1, window_size, window_size, window_size)
    """
    _1D_window = gaussian(window_size, GAUSSIAN_SIGMA).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = Variable(
        _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Compute SSIM for 2D images.

    Args:
        img1: First image
        img2: Second image
        window: Gaussian window for local statistics
        window_size: Size of the window
        channel: Number of channels
        size_average: If True, return mean SSIM, else return per-image SSIM

    Returns:
        SSIM value(s)
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + SSIM_C1) * (2 * sigma12 + SSIM_C2)) / (
        (mu1_sq + mu2_sq + SSIM_C1) * (sigma1_sq + sigma2_sq + SSIM_C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    """Compute SSIM for 3D volumes.

    Args:
        img1: First volume
        img2: Second volume
        window: Gaussian window for local statistics
        window_size: Size of the window
        channel: Number of channels
        size_average: If True, return mean SSIM, else return per-volume SSIM

    Returns:
        SSIM value(s)
    """
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + SSIM_C1) * (2 * sigma12 + SSIM_C2)) / (
        (mu1_sq + mu2_sq + SSIM_C1) * (sigma1_sq + sigma2_sq + SSIM_C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    """Structural Similarity Index Measure for 2D images.

    Computes the SSIM between two images, which measures perceived quality
    by comparing local patterns of pixel intensities.

    Args:
        window_size: Size of Gaussian window. Default: 11
        size_average: If True, return mean SSIM across batch. Default: True
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """Compute SSIM between two images.

        Args:
            img1: First image of shape (B, C, H, W)
            img2: Second image of shape (B, C, H, W)

        Returns:
            SSIM value (scalar or per-image)
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

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    """Structural Similarity Index Measure for 3D volumes.

    Computes the SSIM between two 3D volumes, which measures perceived quality
    by comparing local patterns of voxel intensities.

    Args:
        window_size: Size of Gaussian window. Default: 11
        size_average: If True, return mean SSIM across batch. Default: True
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        """Compute SSIM loss (1 - SSIM) between two volumes.

        Args:
            img1: First volume of shape (B, C, D, H, W)
            img2: Second volume of shape (B, C, D, H, W)

        Returns:
            SSIM loss value (1 - SSIM)
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

        return 1 - _ssim_3D(
            img1, img2, window, self.window_size, channel, self.size_average
        )


def ssim(img1, img2, window_size=11, size_average=True):
    """Functional interface for 2D SSIM.

    Args:
        img1: First image of shape (B, C, H, W)
        img2: Second image of shape (B, C, H, W)
        window_size: Size of Gaussian window. Default: 11
        size_average: If True, return mean SSIM. Default: True

    Returns:
        SSIM value
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    """Functional interface for 3D SSIM.

    Args:
        img1: First volume of shape (B, C, D, H, W)
        img2: Second volume of shape (B, C, D, H, W)
        window_size: Size of Gaussian window. Default: 11
        size_average: If True, return mean SSIM. Default: True

    Returns:
        SSIM value
    """
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


class Grad(torch.nn.Module):
    """2D gradient-based smoothness regularization.

    Penalizes spatial gradients in the predicted field to encourage smoothness.

    Args:
        penalty: Type of penalty ('l1' or 'l2'). Default: 'l1'
        loss_mult: Multiplicative factor for loss. Default: None
    """

    def __init__(self, penalty="l1", loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        """Compute gradient loss.

        Args:
            y_pred: Predicted field of shape (B, C, H, W)
            y_true: Target (unused, kept for interface compatibility)

        Returns:
            Gradient regularization loss
        """
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad3d(torch.nn.Module):
    """3D gradient-based smoothness regularization.

    Penalizes spatial gradients in the predicted 3D field to encourage smoothness.

    Args:
        penalty: Type of penalty ('l1' or 'l2'). Default: 'l1'
        loss_mult: Multiplicative factor for loss. Default: None
    """

    def __init__(self, penalty="l1", loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        """Compute 3D gradient loss.

        Args:
            y_pred: Predicted field of shape (B, C, D, H, W)
            y_true: Target (unused, kept for interface compatibility)

        Returns:
            Gradient regularization loss
        """
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Grad3DiTV(torch.nn.Module):
    """3D isotropic total variation regularization.

    Computes isotropic TV by taking the L2 norm of gradients at each voxel.
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()

    def forward(self, y_pred, y_true):
        """Compute isotropic total variation.

        Args:
            y_pred: Predicted field of shape (B, C, D, H, W)
            y_true: Target (unused, kept for interface compatibility)

        Returns:
            Isotropic TV loss
        """
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(torch.sqrt(dx + dy + dz + 1e-6))
        grad = d / 3.0
        return grad


class DisplacementRegularizer(torch.nn.Module):
    """Regularization for displacement fields with multiple energy types.

    Supports various regularization energies including bending energy and
    gradient-based penalties.

    Args:
        energy_type: Type of energy ('bending', 'gradient-l2', or 'gradient-l1')
    """

    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv):
        """Compute central difference gradient in x direction."""
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv):
        """Compute central difference gradient in y direction."""
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv):
        """Compute central difference gradient in z direction."""
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        """Apply gradient function to each channel of displacement field."""
        return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        """Compute L1 or L2 norm of displacement gradients.

        Args:
            displacement: Displacement field
            flag_l1: If True use L1 norm, else use L2 norm

        Returns:
            Gradient norm
        """
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms) / 3.0

    def compute_bending_energy(self, displacement):
        """Compute bending energy of displacement field.

        Bending energy penalizes second-order derivatives.

        Args:
            displacement: Displacement field

        Returns:
            Bending energy
        """
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(
            dTdxx**2
            + dTdyy**2
            + dTdzz**2
            + 2 * dTdxy**2
            + 2 * dTdxz**2
            + 2 * dTdyz**2
        )

    def forward(self, disp, _):
        """Compute regularization energy.

        Args:
            disp: Displacement field
            _: Placeholder for compatibility (unused)

        Returns:
            Regularization energy

        Raises:
            Exception: If energy_type is not recognized
        """
        if self.energy_type == "bending":
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == "gradient-l2":
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == "gradient-l1":
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception("Not recognised local regulariser!")
        return energy


class NCC_vxm(torch.nn.Module):
    """Normalized Cross-Correlation loss from VoxelMorph.

    Computes local NCC over a sliding window. Works for 1D, 2D, and 3D inputs.

    Args:
        win: Window size (list or None). If None, uses [9] * ndims
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):
        """Compute negative NCC (loss to minimize).

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Negative NCC value (to minimize)
        """
        Ii = y_true
        Ji = y_pred

        # Get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # Set window size
        win = [9] * ndims if self.win is None else self.win

        # Compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # Get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)

        # Compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Clamp variances to prevent division by zero/very small numbers
        I_var = torch.clamp(I_var, min=1e-5)
        J_var = torch.clamp(J_var, min=1e-5)

        cc = cross * cross / (I_var * J_var + 1e-5)

        # Clamp cc to prevent inf/-inf values
        cc = torch.clamp(cc, min=-1e6, max=1e6)

        return -torch.mean(cc)


class MIND_loss(torch.nn.Module):
    """Modality Independent Neighborhood Descriptor (MIND) loss.

    MIND-SSC descriptor for multi-modal image registration.
    Reference: http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf

    Args:
        win: Window parameter (unused, kept for interface compatibility)
    """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        """Compute pairwise squared distances.

        Args:
            x: Input tensor

        Returns:
            Pairwise squared distance matrix
        """
        xx = (x**2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0  # Replace NaN with 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        """Compute MIND-SSC descriptor.

        Args:
            img: Input image
            radius: Radius for patch comparison. Default: 2
            dilation: Dilation for neighborhood. Default: 2

        Returns:
            MIND descriptor
        """
        # Kernel size
        kernel_size = radius * 2 + 1

        # Define 6-neighborhood pattern
        six_neighbourhood = torch.Tensor(
            [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]
        ).long()

        # Squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # Define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = (x > y).view(-1) & (dist == 2).view(-1)

        # Build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[
            torch.arange(12) * 27
            + idx_shift1[:, 0] * 9
            + idx_shift1[:, 1] * 3
            + idx_shift1[:, 2]
        ] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[
            torch.arange(12) * 27
            + idx_shift2[:, 0] * 9
            + idx_shift2[:, 1] * 3
            + idx_shift2[:, 2]
        ] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # Compute patch-ssd
        ssd = F.avg_pool3d(
            rpad2(
                (
                    F.conv3d(rpad1(img), mshift1, dilation=dilation)
                    - F.conv3d(rpad1(img), mshift2, dilation=dilation)
                )
                ** 2
            ),
            kernel_size,
            stride=1,
        )

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(
            mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item()
        )
        mind /= mind_var
        mind = torch.exp(-mind)

        # Permute to have same ordering as C++ code
        mind = mind[
            :, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :
        ]

        return mind

    def forward(self, y_pred, y_true):
        """Compute MIND loss.

        Args:
            y_pred: Predicted image
            y_true: Target image

        Returns:
            MIND descriptor dissimilarity
        """
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class MutualInformation(torch.nn.Module):
    """Global Mutual Information loss for multi-modal registration.

    Computes MI using Gaussian approximation of intensity histograms.

    Args:
        sigma_ratio: Ratio for Gaussian sigma. Default: 1
        minval: Minimum intensity value. Default: 0.0
        maxval: Maximum intensity value. Default: 1.0
        num_bin: Number of histogram bins. Default: 32
    """

    def __init__(self, sigma_ratio=1, minval=0.0, maxval=1.0, num_bin=32):
        super(MutualInformation, self).__init__()

        # Create bin centers
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(
            torch.linspace(minval, maxval, num_bin), requires_grad=False
        ).cuda()
        num_bins = len(bin_centers)

        # Sigma for Gaussian approximation
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        """Compute mutual information.

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Mutual information value
        """
        y_pred = torch.clamp(y_pred, 0.0, self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # Total num of voxels

        # Reshape bin centers
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        # Compute image terms by approx. Gaussian dist.
        I_a = torch.exp(-self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(-self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # Compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(
            torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1
        )
        return mi.mean()  # Average across batch

    def forward(self, y_true, y_pred):
        """Compute negative MI (loss to minimize).

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Negative MI value
        """
        return -self.mi(y_true, y_pred)


class localMutualInformation(torch.nn.Module):
    """Local Mutual Information for non-overlapping patches.

    Computes MI locally over non-overlapping patches for better sensitivity
    to local alignment.

    Args:
        sigma_ratio: Ratio for Gaussian sigma. Default: 1
        minval: Minimum intensity value. Default: 0.0
        maxval: Maximum intensity value. Default: 1.0
        num_bin: Number of histogram bins. Default: 32
        patch_size: Size of patches. Default: 5
    """

    def __init__(self, sigma_ratio=1, minval=0.0, maxval=1.0, num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        # Create bin centers
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(
            torch.linspace(minval, maxval, num_bin), requires_grad=False
        ).cuda()
        num_bins = len(bin_centers)

        # Sigma for Gaussian approximation
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        """Compute local mutual information over patches.

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Local MI value
        """
        y_pred = torch.clamp(y_pred, 0.0, self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        # Reshape bin centers
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        # Making image paddings
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # Compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (
                z_r // 2,
                z_r - z_r // 2,
                y_r // 2,
                y_r - y_r // 2,
                x_r // 2,
                x_r - x_r // 2,
                0,
                0,
                0,
                0,
            )
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # Compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception("Supports 2D and 3D but not {}".format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        # Reshaping images into non-overlapping patches
        if ndim == 3:
            y_true_patch = torch.reshape(
                y_true,
                (
                    y_true.shape[0],
                    y_true.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                    (z + z_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size**3, 1))

            y_pred_patch = torch.reshape(
                y_pred,
                (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                    (z + z_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size**3, 1))
        else:
            y_true_patch = torch.reshape(
                y_true,
                (
                    y_true.shape[0],
                    y_true.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size**2, 1))

            y_pred_patch = torch.reshape(
                y_pred,
                (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    (x + x_r) // self.patch_size,
                    self.patch_size,
                    (y + y_r) // self.patch_size,
                    self.patch_size,
                ),
            )
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size**2, 1))

        # Compute MI
        I_a_patch = torch.exp(-self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(-self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size**ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(
            torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1
        )
        return mi.mean()

    def forward(self, y_true, y_pred):
        """Compute negative local MI (loss to minimize).

        Args:
            y_true: Target image
            y_pred: Predicted image

        Returns:
            Negative local MI value
        """
        return -self.local_mi(y_true, y_pred)


# Fusion losses


class SobelxyRGB3D(nn.Module):
    """3D Sobel gradient operator for extracting edges.

    Computes gradients in x, y, and z directions using 3D Sobel kernels.

    Args:
        isSignGrad: If True, return signed gradients, else absolute. Default: True
    """

    def __init__(self, isSignGrad=True):
        super(SobelxyRGB3D, self).__init__()
        self.isSignGrad = isSignGrad
        kernelx = [
            [[0.2, 0, -0.2], [1, 0, -1], [0.2, 0, -0.2]],
            [[1, 0, -1], [4, 0, -4], [1, 0, -1]],
            [[0.2, 0, -0.2], [1, 0, -1], [0.2, 0, -0.2]],
        ]
        kernely = [
            [[0.2, 1, 0.2], [0, 0, 0], [-0.2, -1, -0.2]],
            [[1, 4, 1], [0, 0, 0], [-1, -4, -1]],
            [[0.2, 1, 0.2], [0, 0, 0], [-0.2, -1, -0.2]],
        ]
        kernelz = [
            [[0.2, 1, 0.2], [1, 4, 1], [0.2, 1, 0.2]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-0.2, -1, -0.2], [-1, -4, -1], [-0.2, -1, -0.2]],
        ]
        self.weightx = nn.Parameter(
            torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1),
            requires_grad=False,
        )
        self.weighty = nn.Parameter(
            torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1),
            requires_grad=False,
        )
        self.weightz = nn.Parameter(
            torch.FloatTensor(kernelz).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1),
            requires_grad=False,
        )

    def forward(self, x):
        """Compute 3D gradients.

        Args:
            x: Input volume

        Returns:
            Sum of gradients in x, y, z directions
        """
        sobelx = F.conv3d(x, self.weightx, padding=1)
        sobely = F.conv3d(x, self.weighty, padding=1)
        sobelz = F.conv3d(x, self.weightz, padding=1)
        if self.isSignGrad:
            return sobelx + sobely + sobelz
        else:
            return torch.abs(sobelx) + torch.abs(sobely) + torch.abs(sobelz)


class MaxGradLoss3D(nn.Module):
    """Maximum gradient loss for image fusion.

    Encourages the fused image to preserve the strongest gradients from inputs.

    Args:
        loss_weight: Weight for the loss. Default: 1.0
        isSignGrad: If True, use signed gradients. Default: True
    """

    def __init__(self, loss_weight=1.0, isSignGrad=True):
        super(MaxGradLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = SobelxyRGB3D(isSignGrad)
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir=None):
        """Compute maximum gradient loss.

        Args:
            im_fusion: Fused image
            im_rgb: First input image
            im_tir: Second input image (optional)

        Returns:
            Gradient preservation loss
        """
        if im_fusion.shape[1] == 1:
            im_fusion = im_fusion.repeat(1, 3, 1, 1, 1)
        if im_rgb.shape[1] == 1:
            im_rgb = im_rgb.repeat(1, 3, 1, 1, 1)
        if im_tir is not None and im_tir.shape[1] == 1:
            im_tir = im_tir.repeat(1, 3, 1, 1, 1)
        if im_tir is not None:
            rgb_grad = self.sobelconv(im_rgb)
            tir_grad = self.sobelconv(im_tir)

            mask = torch.ge(torch.abs(rgb_grad), torch.abs(tir_grad))
            max_grad_joint = tir_grad.masked_fill_(mask, 0) + rgb_grad.masked_fill_(
                ~mask, 0
            )

            generate_img_grad = self.sobelconv(im_fusion)

            sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
            loss_grad = self.loss_weight * sobel_loss
        else:
            rgb_grad = self.sobelconv(im_rgb)
            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad, rgb_grad)
            loss_grad = self.loss_weight * sobel_loss

        return loss_grad


def to_gray3d(img):
    """Convert RGB to grayscale for 3D images.

    Args:
        img: RGB image tensor

    Returns:
        Grayscale image
    """
    r, g, b = img.unbind(dim=-4)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-4)
    return l_img


class MaxPixelLoss3D(nn.Module):
    """Maximum pixel intensity loss for image fusion.

    Encourages the fused image to preserve the brightest pixels from inputs.

    Args:
        loss_weight: Weight for the loss. Default: 1.0
    """

    def __init__(self, loss_weight=1.0):
        super(MaxPixelLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir=None):
        """Compute maximum pixel loss.

        Args:
            im_fusion: Fused image
            im_rgb: First input image
            im_tir: Second input image (optional)

        Returns:
            Pixel intensity preservation loss
        """
        if im_tir is not None:
            pixel_max = torch.max(im_rgb, im_tir).detach()
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, pixel_max)
        else:
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, im_rgb)

        return pixel_loss


class PixelLoss3D(nn.Module):
    """Average pixel intensity loss for image fusion.

    Encourages the fused image to match the average of input intensities.

    Args:
        loss_weight: Weight for the loss. Default: 1.0
    """

    def __init__(self, loss_weight=1.0):
        super(PixelLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir=None):
        """Compute average pixel loss.

        Args:
            im_fusion: Fused image
            im_rgb: First input image
            im_tir: Second input image (optional)

        Returns:
            Average intensity matching loss
        """
        if im_tir is not None:
            pixel_mean = (im_rgb + im_tir) / 2.0
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, pixel_mean)
        else:
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, im_rgb)

        return pixel_loss


class MaxGradTokenSelect3D(nn.Module):
    """Gradient-based token selection for image fusion.

    Selects tokens (patches) from inputs based on maximum gradient magnitude.

    Args:
        loss_weight: Weight for the loss. Default: 1.0
    """

    def __init__(self, loss_weight=1.0):
        super(MaxGradTokenSelect3D, self).__init__()
        self.sobelconv = SobelxyRGB3D()

    def forward(self, im_rgb, im_tir):
        """Select patches based on maximum gradients.

        Args:
            im_rgb: First input image
            im_tir: Second input image

        Returns:
            Fused image with gradient-based patch selection
        """
        im_rgb_gray = to_gray3d(im_rgb)
        im_tir_gray = to_gray3d(im_tir)
        rgb_grad = self.sobelconv(im_rgb_gray)
        tir_grad = self.sobelconv(im_tir_gray)

        im_rgb, info = self.patchify3d(im_rgb)
        im_tir, _ = self.patchify3d(im_tir)
        rgb_grad_patch, info_grad = self.patchify3d(rgb_grad)
        tir_grad_patch, _ = self.patchify3d(tir_grad)

        rgb_grad, _ = torch.max(rgb_grad_patch, -1)
        tir_grad, _ = torch.max(tir_grad_patch, -1)

        AB_mask = (rgb_grad >= tir_grad).unsqueeze(dim=-1)
        out = torch.where(AB_mask, im_rgb, im_tir)

        out = self.unpatchify3d(out, info)

        return out

    def patchify3d(self, imgs):
        """Convert images to patches.

        Args:
            imgs: Input images

        Returns:
            Tuple of (patches, shape_info)
        """
        p = 16
        assert (
            imgs.shape[2] % p == 0
            and imgs.shape[3] % p == 0
            and imgs.shape[4] % p == 0
        )
        C, D, H, W = imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]
        x = imgs.reshape(
            shape=(imgs.shape[0], C, D // p, p, H // p, p, W // p, p)
        )
        x = torch.einsum("ncdphqwr->ndhwpqrc", x)
        x = x.reshape(shape=(imgs.shape[0], D // p * H // p * W // p, p * p * p * C))
        return x, (D, H, W, C)

    def unpatchify3d(self, x, shape):
        """Reconstruct images from patches.

        Args:
            x: Patches
            shape: Original image shape info

        Returns:
            Reconstructed image
        """
        p = 16
        D, H, W, C = shape
        assert D * H * W // (p**3) == x.shape[1]

        x = x.reshape(shape=(x.shape[0], D // p, H // p, W // p, p, p, p, C))
        x = torch.einsum("ndhwpqrc->ncdphqwr", x)
        imgs = x.reshape(shape=(x.shape[0], C, D, H, W))
        return imgs
