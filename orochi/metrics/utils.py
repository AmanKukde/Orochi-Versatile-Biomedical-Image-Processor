"""
Metrics utilities for Orochi.

Provides factory functions and utilities for evaluation metrics.
"""

from typing import Optional, Callable, Dict, Any


def get_metric(name: str, **kwargs) -> Callable:
    """
    Factory function to get evaluation metrics by name.

    Args:
        name (str): Metric name ('psnr', 'ssim', 'dice', etc.)
        **kwargs: Additional arguments for the metric

    Returns:
        Callable: Metric function

    Raises:
        ValueError: If metric name is not recognized

    Example:
        >>> from orochi.metrics import get_metric
        >>> psnr_fn = get_metric('psnr')
        >>> ssim_fn = get_metric('ssim', window_size=11)
    """
    metrics: Dict[str, Callable] = {
        'psnr': _psnr,
        'ssim': _ssim,
        'dice': _dice,
        'iou': _iou,
        'mse': _mse,
        'mae': _mae,
    }

    name_lower = name.lower()
    if name_lower not in metrics:
        raise ValueError(
            f"Unknown metric: {name}. Available: {list(metrics.keys())}"
        )

    return metrics[name_lower]


def _psnr(pred, target, max_val: Optional[float] = None) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image
        target: Target image
        max_val: Maximum possible value (default: auto-detect)

    Returns:
        float: PSNR value in dB
    """
    import torch
    import numpy as np

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')

    if max_val is None:
        max_val = max(target.max(), pred.max())

    return 20 * np.log10(max_val / np.sqrt(mse))


def _ssim(pred, target, **kwargs) -> float:
    """
    Calculate Structural Similarity Index.

    Args:
        pred: Predicted image
        target: Target image
        **kwargs: Additional arguments (window_size, etc.)

    Returns:
        float: SSIM value
    """
    from skimage.metrics import structural_similarity as ssim
    import torch
    import numpy as np

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Handle multi-dimensional arrays
    if pred.ndim > 2:
        # For batches or multi-channel images, average across channels
        return ssim(pred, target, channel_axis=0 if pred.ndim == 3 else None, **kwargs)

    return ssim(pred, target, **kwargs)


def _dice(pred, target, smooth: float = 1e-5) -> float:
    """
    Calculate Dice coefficient.

    Args:
        pred: Predicted segmentation
        target: Target segmentation
        smooth: Smoothing factor

    Returns:
        float: Dice score
    """
    import torch
    import numpy as np

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Binarize if needed
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def _iou(pred, target, smooth: float = 1e-5) -> float:
    """
    Calculate Intersection over Union (IoU).

    Args:
        pred: Predicted segmentation
        target: Target segmentation
        smooth: Smoothing factor

    Returns:
        float: IoU score
    """
    import torch
    import numpy as np

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Binarize if needed
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)

    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection

    return (intersection + smooth) / (union + smooth)


def _mse(pred, target) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        float: MSE value
    """
    import torch
    import numpy as np

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return np.mean((pred - target) ** 2)


def _mae(pred, target) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        pred: Predicted values
        target: Target values

    Returns:
        float: MAE value
    """
    import torch
    import numpy as np

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    return np.mean(np.abs(pred - target))
