"""Registration losses for deformable image alignment.

Provides losses for image registration tasks:
- NCC: Normalized Cross-Correlation
- MutualInformation: Mutual Information loss
- GradientLoss: Regularization for smooth deformations
- BendingEnergyLoss: Second-order smoothness regularization

All losses support both 2D and 3D inputs automatically.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from orochi.losses.base import BaseLoss
from orochi.losses.constants import MI_NUM_BINS, MI_SIGMA_RATIO, NCC_WINDOW_SIZE


class NCC(BaseLoss):
    """Normalized Cross-Correlation (NCC) Loss.

    Measures local correlation between images. Effective for mono-modal
    registration where intensities are linearly related. Computed locally
    using a sliding window.

    NCC = Σ[(I1 - μ1)(I2 - μ2)] / √[Σ(I1 - μ1)² Σ(I2 - μ2)²]

    Args:
        window_size: Size of local window. Default: 9
        reduction: Reduction method. Default: 'mean'

    Shape:
        - Input: (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
        - Output: Scalar if reduction != 'none'

    Example:
        >>> ncc_loss = NCC(window_size=9)
        >>> # 2D registration
        >>> fixed = torch.randn(2, 1, 256, 256)
        >>> moving = torch.randn(2, 1, 256, 256)
        >>> loss = ncc_loss(fixed, moving)
        >>>
        >>> # 3D registration - same API
        >>> fixed_3d = torch.randn(2, 1, 64, 256, 256)
        >>> moving_3d = torch.randn(2, 1, 64, 256, 256)
        >>> loss_3d = ncc_loss(fixed_3d, moving_3d)

    Reference:
        Avants, Brian B., et al. "Symmetric diffeomorphic image registration
        with cross-correlation: evaluating automated labeling of elderly and
        neurodegenerative brain." MIA 12.1 (2008): 26-41.
    """

    def __init__(self, window_size: int = NCC_WINDOW_SIZE, reduction: str = "mean"):
        """Initialize NCC loss."""
        super().__init__(reduction=reduction)
        if window_size % 2 == 0:
            raise ValueError(f"Window size must be odd, got {window_size}")
        self.window_size = window_size

    def _compute_local_sums(
        self, img: torch.Tensor, spatial_dims: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute local sums using average pooling.

        Args:
            img: Input image
            spatial_dims: Number of spatial dimensions

        Returns:
            Tuple of (sum, sum_of_squares, num_elements)
        """
        kernel_size = self.window_size
        padding = kernel_size // 2

        if spatial_dims == 2:
            pool_fn = F.avg_pool2d
        else:  # spatial_dims == 3
            pool_fn = F.avg_pool3d

        # Compute local sums using average pooling
        sum_img = pool_fn(
            img,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        sum_img_sq = pool_fn(
            img ** 2,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        # Number of elements in window
        num_elements = kernel_size ** spatial_dims

        return sum_img * num_elements, sum_img_sq * num_elements, num_elements

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute NCC loss.

        Args:
            pred: Predicted/moving image
            target: Target/fixed image
            spatial_dims: Number of spatial dimensions

        Returns:
            NCC loss (1 - NCC for minimization)
        """
        # Compute local sums
        sum_pred, sum_pred_sq, n_elem = self._compute_local_sums(pred, spatial_dims)
        sum_target, sum_target_sq, _ = self._compute_local_sums(target, spatial_dims)

        # Compute cross-correlation
        if spatial_dims == 2:
            pool_fn = F.avg_pool2d
        else:
            pool_fn = F.avg_pool3d

        cross = pool_fn(
            pred * target,
            kernel_size=self.window_size,
            stride=1,
            padding=self.window_size // 2
        ) * n_elem

        # Compute means
        mean_pred = sum_pred / n_elem
        mean_target = sum_target / n_elem

        # Compute variances
        var_pred = sum_pred_sq / n_elem - mean_pred ** 2
        var_target = sum_target_sq / n_elem - mean_target ** 2

        # Compute covariance
        covar = cross / n_elem - mean_pred * mean_target

        # Compute NCC
        ncc = covar / (torch.sqrt(var_pred * var_target) + 1e-5)

        # Return 1 - NCC for minimization (NCC is in [-1, 1])
        return 1.0 - ncc


class MutualInformation(BaseLoss):
    """Mutual Information (MI) Loss for multi-modal registration.

    Measures statistical dependence between images. Effective for multi-modal
    registration where intensities are not linearly related (e.g., CT to MRI).

    MI(X,Y) = H(X) + H(Y) - H(X,Y)

    where H is entropy.

    Args:
        num_bins: Number of bins for histogram. Default: 32
        sigma_ratio: Ratio of sigma to number of bins. Default: 0.5
        reduction: Reduction method. Default: 'mean'

    Example:
        >>> mi_loss = MutualInformation(num_bins=32)
        >>> # Multi-modal registration (e.g., CT to MRI)
        >>> ct = torch.randn(2, 1, 64, 256, 256)
        >>> mri = torch.randn(2, 1, 64, 256, 256)
        >>> loss = mi_loss(ct, mri)

    Reference:
        Mattes, David, et al. "PET-CT image registration in the chest using
        free-form deformations." IEEE TMI 22.1 (2003): 120-128.
    """

    def __init__(
        self,
        num_bins: int = MI_NUM_BINS,
        sigma_ratio: float = MI_SIGMA_RATIO,
        reduction: str = "mean",
    ):
        """Initialize MI loss."""
        super().__init__(reduction=reduction)
        self.num_bins = num_bins
        self.sigma = sigma_ratio * num_bins

    def _compute_joint_histogram(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> torch.Tensor:
        """Compute joint histogram using Parzen windowing.

        Args:
            img1: First image
            img2: Second image

        Returns:
            Joint histogram
        """
        batch_size = img1.shape[0]

        # Normalize images to [0, num_bins-1]
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-10)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-10)

        img1_bins = img1_norm * (self.num_bins - 1)
        img2_bins = img2_norm * (self.num_bins - 1)

        # Flatten
        img1_flat = img1_bins.reshape(batch_size, -1)
        img2_flat = img2_bins.reshape(batch_size, -1)

        # Create bin centers
        bin_centers = torch.arange(
            0, self.num_bins, dtype=torch.float32, device=img1.device
        )

        # Compute distances to bin centers using broadcasting
        # img_flat: (B, N), bin_centers: (num_bins,)
        # dist: (B, N, num_bins)
        dist1 = (img1_flat.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0)) ** 2
        dist2 = (img2_flat.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0)) ** 2

        # Apply Gaussian Parzen window
        w1 = torch.exp(-dist1 / (2 * self.sigma ** 2))
        w2 = torch.exp(-dist2 / (2 * self.sigma ** 2))

        # Normalize weights
        w1 = w1 / (w1.sum(dim=-1, keepdim=True) + 1e-10)
        w2 = w2 / (w2.sum(dim=-1, keepdim=True) + 1e-10)

        # Compute joint histogram: (B, num_bins, num_bins)
        joint_hist = torch.einsum('bni,bnj->bij', w1, w2)

        return joint_hist

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute MI loss.

        Args:
            pred: Predicted/moving image
            target: Target/fixed image
            spatial_dims: Number of spatial dimensions

        Returns:
            Negative MI (for minimization)
        """
        # Compute joint histogram
        joint_hist = self._compute_joint_histogram(pred, target)

        # Compute marginal histograms
        p_x = joint_hist.sum(dim=2)  # (B, num_bins)
        p_y = joint_hist.sum(dim=1)  # (B, num_bins)

        # Add small epsilon for numerical stability
        eps = 1e-10
        joint_hist = joint_hist + eps
        p_x = p_x + eps
        p_y = p_y + eps

        # Compute entropies
        h_x = -(p_x * torch.log(p_x)).sum(dim=1)
        h_y = -(p_y * torch.log(p_y)).sum(dim=1)
        h_xy = -(joint_hist * torch.log(joint_hist)).sum(dim=(1, 2))

        # Compute MI
        mi = h_x + h_y - h_xy

        # Return negative MI for minimization
        return -mi


class GradientLoss(BaseLoss):
    """Gradient-based smoothness regularization for deformation fields.

    Penalizes large gradients in deformation fields to encourage smooth,
    realistic deformations. Essential for deformable registration.

    Args:
        penalty: Type of penalty ('l1' or 'l2'). Default: 'l2'
        reduction: Reduction method. Default: 'mean'

    Example:
        >>> grad_loss = GradientLoss(penalty='l2')
        >>> # Flow field: (B, 2, H, W) for 2D or (B, 3, D, H, W) for 3D
        >>> flow_2d = torch.randn(2, 2, 256, 256)
        >>> loss_2d = grad_loss(flow_2d, flow_2d)  # Dummy target
        >>>
        >>> flow_3d = torch.randn(2, 3, 64, 256, 256)
        >>> loss_3d = grad_loss(flow_3d, flow_3d)

    Note:
        This loss typically doesn't use a target - the second argument
        can be a dummy tensor with the same shape.
    """

    def __init__(self, penalty: str = "l2", reduction: str = "mean"):
        """Initialize Gradient loss."""
        super().__init__(reduction=reduction)
        if penalty not in ["l1", "l2"]:
            raise ValueError(f"Penalty must be 'l1' or 'l2', got '{penalty}'")
        self.penalty = penalty

    def _gradient_2d(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute gradients for 2D flow.

        Args:
            flow: Flow field of shape (B, 2, H, W)

        Returns:
            Gradient magnitude
        """
        # Compute gradients
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])

        # Handle boundary (pad to original size)
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))

        if self.penalty == "l2":
            return dy ** 2 + dx ** 2
        else:  # l1
            return dy + dx

    def _gradient_3d(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute gradients for 3D flow.

        Args:
            flow: Flow field of shape (B, 3, D, H, W)

        Returns:
            Gradient magnitude
        """
        # Compute gradients
        dz = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dx = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

        # Pad to original size
        dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dx = F.pad(dx, (0, 1, 0, 0, 0, 0))

        if self.penalty == "l2":
            return dz ** 2 + dy ** 2 + dx ** 2
        else:  # l1
            return dz + dy + dx

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute gradient loss.

        Args:
            pred: Flow field
            target: Unused (dummy tensor)
            spatial_dims: Number of spatial dimensions

        Returns:
            Gradient penalty
        """
        if spatial_dims == 2:
            grad = self._gradient_2d(pred)
        else:  # spatial_dims == 3
            grad = self._gradient_3d(pred)

        return grad


class BendingEnergyLoss(BaseLoss):
    """Bending energy regularization for deformation fields.

    Second-order smoothness penalty that penalizes high curvature in
    deformation fields. More restrictive than first-order gradient loss.

    Args:
        reduction: Reduction method. Default: 'mean'

    Example:
        >>> bending_loss = BendingEnergyLoss()
        >>> flow = torch.randn(2, 2, 256, 256)
        >>> loss = bending_loss(flow, flow)

    Reference:
        Rueckert, Daniel, et al. "Nonrigid registration using free-form
        deformations: application to breast MR images." IEEE TMI 18.8 (1999).
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize bending energy loss."""
        super().__init__(reduction=reduction)

    def _compute_bending_energy_2d(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute bending energy for 2D flow.

        Args:
            flow: Flow field of shape (B, 2, H, W)

        Returns:
            Bending energy
        """
        # Second derivatives
        dxx = flow[:, :, 2:, 1:-1] - 2 * flow[:, :, 1:-1, 1:-1] + flow[:, :, :-2, 1:-1]
        dyy = flow[:, :, 1:-1, 2:] - 2 * flow[:, :, 1:-1, 1:-1] + flow[:, :, 1:-1, :-2]
        dxy = (
            flow[:, :, 2:, 2:] - flow[:, :, 2:, :-2] -
            flow[:, :, :-2, 2:] + flow[:, :, :-2, :-2]
        ) / 4.0

        return dxx ** 2 + dyy ** 2 + 2 * dxy ** 2

    def _compute_bending_energy_3d(self, flow: torch.Tensor) -> torch.Tensor:
        """Compute bending energy for 3D flow.

        Args:
            flow: Flow field of shape (B, 3, D, H, W)

        Returns:
            Bending energy
        """
        # Second derivatives (simplified computation)
        dzz = (
            flow[:, :, 2:, 1:-1, 1:-1] - 2 * flow[:, :, 1:-1, 1:-1, 1:-1] +
            flow[:, :, :-2, 1:-1, 1:-1]
        )
        dyy = (
            flow[:, :, 1:-1, 2:, 1:-1] - 2 * flow[:, :, 1:-1, 1:-1, 1:-1] +
            flow[:, :, 1:-1, :-2, 1:-1]
        )
        dxx = (
            flow[:, :, 1:-1, 1:-1, 2:] - 2 * flow[:, :, 1:-1, 1:-1, 1:-1] +
            flow[:, :, 1:-1, 1:-1, :-2]
        )

        return dzz ** 2 + dyy ** 2 + dxx ** 2

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute bending energy loss.

        Args:
            pred: Flow field
            target: Unused (dummy tensor)
            spatial_dims: Number of spatial dimensions

        Returns:
            Bending energy
        """
        if spatial_dims == 2:
            energy = self._compute_bending_energy_2d(pred)
        else:  # spatial_dims == 3
            energy = self._compute_bending_energy_3d(pred)

        return energy
