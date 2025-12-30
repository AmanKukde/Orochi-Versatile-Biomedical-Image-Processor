"""Base classes for loss functions.

Provides dimension-agnostic base classes that automatically handle
2D and 3D inputs.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionAgnosticLoss(nn.Module):
    """Base class for losses that work with both 2D and 3D inputs.

    Automatically detects input dimensions and dispatches to appropriate
    implementation.

    Attributes:
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize dimension-agnostic loss.

        Args:
            reduction: Reduction method ('none', 'mean', or 'sum')

        Raises:
            ValueError: If reduction is not one of the valid options
        """
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'")
        self.reduction = reduction

    def _get_spatial_dims(self, x: torch.Tensor) -> int:
        """Get number of spatial dimensions from tensor.

        Args:
            x: Input tensor of shape (B, C, *spatial)

        Returns:
            Number of spatial dimensions (2 or 3)

        Raises:
            ValueError: If tensor has invalid number of dimensions
        """
        ndim = x.ndim - 2  # Remove batch and channel dimensions
        if ndim not in [2, 3]:
            raise ValueError(
                f"Expected 4D (2D images) or 5D (3D volumes) tensor, "
                f"got {x.ndim}D tensor with shape {x.shape}"
            )
        return ndim

    def _validate_inputs(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[int, torch.Size]:
        """Validate input tensors.

        Args:
            pred: Prediction tensor
            target: Target tensor

        Returns:
            Tuple of (spatial_dims, shape)

        Raises:
            ValueError: If inputs have mismatched shapes or invalid dimensions
            TypeError: If inputs are not tensors
        """
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor inputs, got {type(pred)} and {type(target)}"
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} != target {target.shape}"
            )

        spatial_dims = self._get_spatial_dims(pred)
        return spatial_dims, pred.shape

    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor.

        Args:
            loss: Loss tensor

        Returns:
            Reduced loss according to self.reduction
        """
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BaseLoss(DimensionAgnosticLoss):
    """Base class for all Orochi loss functions.

    Provides common functionality:
    - Input validation
    - Dimension detection
    - Reduction handling
    - Device management

    Example:
        >>> class MyLoss(BaseLoss):
        ...     def compute_loss(self, pred, target, spatial_dims):
        ...         # Your loss computation here
        ...         return loss
        >>>
        >>> loss_fn = MyLoss(reduction='mean')
        >>> loss = loss_fn(pred, target)
    """

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute the loss value.

        This method should be overridden by subclasses.

        Args:
            pred: Prediction tensor
            target: Target tensor
            spatial_dims: Number of spatial dimensions (2 or 3)

        Returns:
            Loss tensor

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pred: Prediction tensor of shape (B, C, *spatial)
            target: Target tensor of shape (B, C, *spatial)

        Returns:
            Loss value (scalar if reduction != 'none')

        Raises:
            ValueError: If inputs are invalid
            TypeError: If inputs are not tensors
        """
        # Validate inputs
        spatial_dims, _ = self._validate_inputs(pred, target)

        # Compute loss
        loss = self.compute_loss(pred, target, spatial_dims)

        # Apply reduction
        return self._apply_reduction(loss)


def get_gaussian_kernel(kernel_size: int, sigma: float, channels: int, ndim: int) -> torch.Tensor:
    """Create a Gaussian kernel for SSIM and other losses.

    Args:
        kernel_size: Size of the kernel
        sigma: Standard deviation
        channels: Number of channels
        ndim: Number of dimensions (2 or 3)

    Returns:
        Gaussian kernel tensor

    Example:
        >>> kernel_2d = get_gaussian_kernel(11, 1.5, 1, 2)
        >>> kernel_2d.shape
        torch.Size([1, 1, 11, 11])
        >>>
        >>> kernel_3d = get_gaussian_kernel(11, 1.5, 1, 3)
        >>> kernel_3d.shape
        torch.Size([1, 1, 11, 11, 11])
    """
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    # Expand to required dimensions
    if ndim == 2:
        kernel = g.view(1, -1) * g.view(-1, 1)
    elif ndim == 3:
        kernel = (g.view(1, 1, -1) * g.view(1, -1, 1) * g.view(-1, 1, 1))
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    # Expand for channels
    kernel = kernel.view(1, 1, *kernel.shape)
    kernel = kernel.repeat(channels, 1, *([1] * ndim))

    return kernel
