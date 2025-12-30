"""Segmentation losses for medical image analysis.

Provides losses for semantic segmentation and boundary detection:
- DiceLoss: Dice coefficient-based loss
- SurfaceLoss: Surface distance-based loss
- FocalLoss: Focal loss for class imbalance
- TverskyLoss: Generalization of Dice loss

All losses support both 2D and 3D inputs automatically.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from orochi.losses.base import BaseLoss
from orochi.losses.constants import DICE_SMOOTH


class DiceLoss(BaseLoss):
    """Dice Loss for segmentation tasks.

    Unified implementation that handles both 2D and 3D inputs, with support
    for multi-class segmentation. The Dice coefficient measures overlap between
    predicted and target segmentation masks.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    DiceLoss = 1 - Dice

    Args:
        num_classes: Number of segmentation classes. Default: 2 (binary)
        smooth: Smoothing constant to avoid division by zero. Default: 1e-5
        squared_pred: If True, square the predictions in denominator. Default: False
        include_background: If True, include background class in loss. Default: True
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Shape:
        - Input: (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
          where C is num_classes
        - Target: (B, H, W) for 2D or (B, D, H, W) for 3D
          with integer class labels, or same shape as input for one-hot
        - Output: Scalar if reduction != 'none'

    Example:
        >>> # Binary segmentation (2D)
        >>> dice_loss = DiceLoss(num_classes=2)
        >>> logits = torch.randn(4, 2, 256, 256)
        >>> target = torch.randint(0, 2, (4, 256, 256))
        >>> loss = dice_loss(logits, target)
        >>>
        >>> # Multi-class segmentation (3D)
        >>> dice_loss = DiceLoss(num_classes=5)
        >>> logits = torch.randn(2, 5, 32, 256, 256)
        >>> target = torch.randint(0, 5, (2, 32, 256, 256))
        >>> loss = dice_loss(logits, target)
        >>>
        >>> # Exclude background from loss
        >>> dice_loss = DiceLoss(num_classes=5, include_background=False)

    Reference:
        Milletari, Fausto, et al. "V-net: Fully convolutional neural networks
        for volumetric medical image segmentation." 3DV 2016.
    """

    def __init__(
        self,
        num_classes: int = 2,
        smooth: float = DICE_SMOOTH,
        squared_pred: bool = False,
        include_background: bool = True,
        reduction: str = "mean",
    ):
        """Initialize Dice loss."""
        super().__init__(reduction=reduction)
        self.num_classes = num_classes
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.include_background = include_background

    def _to_one_hot(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert integer labels to one-hot encoding.

        Args:
            target: Integer labels of shape (B, *spatial)
            num_classes: Number of classes

        Returns:
            One-hot encoded tensor of shape (B, C, *spatial)
        """
        # Create one-hot encoding
        shape = target.shape
        one_hot = torch.zeros(
            (shape[0], num_classes, *shape[1:]),
            dtype=torch.float32,
            device=target.device
        )
        target_expanded = target.unsqueeze(1)
        one_hot.scatter_(1, target_expanded.long(), 1)
        return one_hot

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: Predicted logits of shape (B, C, *spatial)
            target: Target labels of shape (B, *spatial) or (B, C, *spatial)
            spatial_dims: Number of spatial dimensions (2 or 3)

        Returns:
            Dice loss
        """
        # Apply softmax to predictions
        pred_softmax = F.softmax(pred, dim=1)

        # Convert target to one-hot if needed
        if target.ndim == pred.ndim - 1:
            target_one_hot = self._to_one_hot(target, self.num_classes)
        else:
            target_one_hot = target

        # Determine which classes to include
        if not self.include_background:
            pred_softmax = pred_softmax[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        # Flatten spatial dimensions
        pred_flat = pred_softmax.flatten(2)  # (B, C, N)
        target_flat = target_one_hot.flatten(2)  # (B, C, N)

        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=2)  # (B, C)

        if self.squared_pred:
            denominator = (pred_flat ** 2).sum(dim=2) + target_flat.sum(dim=2)
        else:
            denominator = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # Return 1 - Dice as loss
        return 1.0 - dice.mean(dim=1)  # Average over classes, return per-batch


class GeneralizedDiceLoss(BaseLoss):
    """Generalized Dice Loss for imbalanced segmentation.

    Addresses class imbalance by weighting each class inversely proportional
    to its frequency. Particularly useful for medical images where some
    structures are much smaller than others.

    Args:
        num_classes: Number of segmentation classes
        smooth: Smoothing constant. Default: 1e-5
        include_background: Include background class. Default: True
        reduction: Reduction method. Default: 'mean'

    Example:
        >>> gdl = GeneralizedDiceLoss(num_classes=5)
        >>> logits = torch.randn(2, 5, 32, 256, 256)
        >>> target = torch.randint(0, 5, (2, 32, 256, 256))
        >>> loss = gdl(logits, target)

    Reference:
        Sudre, Carole H., et al. "Generalised dice overlap as a deep learning
        loss function for highly unbalanced segmentations." DLMIA 2017.
    """

    def __init__(
        self,
        num_classes: int = 2,
        smooth: float = DICE_SMOOTH,
        include_background: bool = True,
        reduction: str = "mean",
    ):
        """Initialize Generalized Dice loss."""
        super().__init__(reduction=reduction)
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute Generalized Dice loss.

        Args:
            pred: Predicted logits
            target: Target labels
            spatial_dims: Number of spatial dimensions

        Returns:
            Generalized Dice loss
        """
        # Apply softmax
        pred_softmax = F.softmax(pred, dim=1)

        # Convert target to one-hot if needed
        if target.ndim == pred.ndim - 1:
            target_one_hot = F.one_hot(
                target.long(),
                num_classes=self.num_classes
            ).permute(0, -1, *range(1, target.ndim)).float()
        else:
            target_one_hot = target

        # Exclude background if needed
        if not self.include_background:
            pred_softmax = pred_softmax[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        # Flatten
        pred_flat = pred_softmax.flatten(2)
        target_flat = target_one_hot.flatten(2)

        # Compute weights (inverse of class frequency)
        target_sum = target_flat.sum(dim=2)
        weights = 1.0 / (target_sum ** 2 + self.smooth)

        # Weighted intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        denominator = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        # Weighted Dice
        weighted_dice = (
            (2.0 * weights * intersection + self.smooth) /
            (weights * denominator + self.smooth)
        )

        return 1.0 - weighted_dice.mean(dim=1)


class TverskyLoss(BaseLoss):
    """Tversky Loss for segmentation.

    Generalization of Dice loss that allows for different weightings of
    false positives and false negatives. Useful when you want to control
    the trade-off between precision and recall.

    Args:
        alpha: Weight of false positives. Default: 0.5
        beta: Weight of false negatives. Default: 0.5
        num_classes: Number of classes. Default: 2
        smooth: Smoothing constant. Default: 1e-5
        reduction: Reduction method. Default: 'mean'

    Note:
        - alpha = beta = 0.5 reduces to Dice loss
        - alpha < beta emphasizes recall (reduces false negatives)
        - alpha > beta emphasizes precision (reduces false positives)

    Example:
        >>> # Emphasize recall (reduce false negatives)
        >>> tversky = TverskyLoss(alpha=0.3, beta=0.7, num_classes=2)
        >>> loss = tversky(pred, target)

    Reference:
        Salehi, Seyed Sadegh Mohseni, et al. "Tversky loss function for
        image segmentation using 3D fully convolutional deep networks."
        MLMI 2017.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        num_classes: int = 2,
        smooth: float = DICE_SMOOTH,
        reduction: str = "mean",
    ):
        """Initialize Tversky loss."""
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smooth = smooth

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute Tversky loss.

        Args:
            pred: Predicted logits
            target: Target labels
            spatial_dims: Number of spatial dimensions

        Returns:
            Tversky loss
        """
        # Apply softmax
        pred_softmax = F.softmax(pred, dim=1)

        # Convert target to one-hot
        if target.ndim == pred.ndim - 1:
            target_one_hot = F.one_hot(
                target.long(),
                num_classes=self.num_classes
            ).permute(0, -1, *range(1, target.ndim)).float()
        else:
            target_one_hot = target

        # Flatten
        pred_flat = pred_softmax.flatten(2)
        target_flat = target_one_hot.flatten(2)

        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)

        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1.0 - tversky.mean(dim=1)


class FocalLoss(BaseLoss):
    """Focal Loss for addressing class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard
    examples. Particularly effective for datasets with extreme class imbalance.

    Args:
        alpha: Weighting factor in [0, 1] to balance classes. Default: 0.25
        gamma: Focusing parameter >= 0. Default: 2.0
        reduction: Reduction method. Default: 'mean'

    Note:
        - gamma = 0 reduces to standard cross-entropy
        - Higher gamma increases focus on hard examples
        - alpha can be a scalar or list of per-class weights

    Example:
        >>> focal = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(4, 2, 256, 256)
        >>> target = torch.randint(0, 2, (4, 256, 256))
        >>> loss = focal(logits, target)

    Reference:
        Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
        ICCV 2017.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize Focal loss."""
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, spatial_dims: int
    ) -> torch.Tensor:
        """Compute Focal loss.

        Args:
            pred: Predicted logits
            target: Target labels
            spatial_dims: Number of spatial dimensions

        Returns:
            Focal loss
        """
        # Compute standard cross-entropy
        ce_loss = F.cross_entropy(pred, target.long(), reduction="none")

        # Get class probabilities
        p = F.softmax(pred, dim=1)
        # Get probability of true class
        p_t = p.gather(1, target.long().unsqueeze(1)).squeeze(1)

        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss
