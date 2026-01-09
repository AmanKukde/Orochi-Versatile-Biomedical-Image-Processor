"""
Consolidated utility functions for Orochi biomedical image processing.

This module provides essential utilities for:
- Visualization (logits, flows, grids)
- Metrics (PSNR, SSIM, Dice, Jacobian)
- Training utilities (logging, checkpointing, learning rate scheduling)
- Image transformations (spatial transforms, padding)
- Uncertainty estimation (Monte Carlo predictions)
- Distributed training (setup, cleanup)
"""

import math
import os
import sys
import glob
import shutil
import socket
import random
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# ============================================================================
# Metrics
# ============================================================================

def psnr(img1: Union[np.ndarray, torch.Tensor],
         img2: Union[np.ndarray, torch.Tensor],
         data_range: Optional[float] = None) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1: First image (numpy array or tensor)
        img2: Second image (numpy array or tensor)
        data_range: Data range of the images. If None, computed from images.

    Returns:
        PSNR value in dB. Returns inf if MSE is 0.

    Example:
        >>> psnr_val = psnr(pred_image, gt_image)
        >>> print(f"PSNR: {psnr_val:.2f} dB")
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    if data_range is None:
        data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def carepsnr(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Calculate CARE-PSNR (Content-Aware Peak Signal-to-Noise Ratio).

    This metric normalizes for brightness and contrast differences before
    computing PSNR, making it more robust for microscopy images.

    Args:
        img1: Predicted image tensor
        img2: Ground truth image tensor
        eps: Small epsilon for numerical stability

    Returns:
        CARE-PSNR value

    Example:
        >>> care_psnr_val = carepsnr(pred, gt)
        >>> print(f"CARE-PSNR: {care_psnr_val:.2f}")
    """
    def _zero_mean(x: torch.Tensor) -> torch.Tensor:
        return x - torch.mean(x, dim=1, keepdim=True)

    def _fix_range(gt: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = torch.sum(gt * x, dim=1, keepdim=True) / (torch.sum(x * x, dim=1, keepdim=True) + eps)
        return x * a

    def _fix(gt: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        gt_ = _zero_mean(gt)
        return _fix_range(gt_, _zero_mean(x), eps)

    def _psnr_internal(gt: torch.Tensor, pred: torch.Tensor, range_: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mse = torch.mean((gt - pred) ** 2, dim=1)
        return 20 * torch.log10(range_ / torch.sqrt(mse + eps))

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    img1 = img1.float()
    img2 = img2.float()

    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 2:
        img2 = img2.unsqueeze(0)

    assert len(img1.shape) == 3, f"Input should be (B, H, W) or (H, W), got: {img1.shape}"
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    pred_flat = img1.view(img1.shape[0], -1)
    gt_flat = img2.view(img2.shape[0], -1)

    gt_std_squeezed = torch.std(gt_flat, dim=1)
    ra = (torch.max(gt_flat, dim=1).values - torch.min(gt_flat, dim=1).values) / (gt_std_squeezed + eps)

    gt_std_unsqueezed = gt_std_squeezed.unsqueeze(1)
    gt_ = _zero_mean(gt_flat) / (gt_std_unsqueezed + eps)

    pred_fixed = _fix(gt_, pred_flat, eps)

    psnr_values = _psnr_internal(gt_, pred_fixed, ra, eps)

    return psnr_values.mean().item()


def microssim(pred_img: Union[np.ndarray, torch.Tensor],
              gt_img: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate Micro-SSIM metric for microscopy images.

    Requires microssim package to be installed.

    Args:
        pred_img: Predicted image
        gt_img: Ground truth image

    Returns:
        Micro-SSIM score

    Example:
        >>> ssim_score = microssim(prediction, ground_truth)
    """
    try:
        from microssim import MicroSSIM
    except ImportError:
        raise ImportError("microssim package is required. Install with: pip install microssim")

    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    assert pred_img.shape == gt_img.shape, f"Shape mismatch: {pred_img.shape} vs {gt_img.shape}"

    shape = pred_img.shape
    if len(shape) == 4:
        B, C, H, W = shape
        pred_stack = pred_img.reshape(B * C, H, W)
        gt_stack = gt_img.reshape(B * C, H, W)
    elif len(shape) == 3:
        pred_stack = pred_img
        gt_stack = gt_img
    elif len(shape) == 2:
        pred_stack = pred_img[np.newaxis, ...]
        gt_stack = gt_img[np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    microssim_evaluator = MicroSSIM()
    microssim_evaluator.fit(gt_stack, pred_stack)

    scores = []
    for i in range(len(gt_stack)):
        score = microssim_evaluator.score(gt_stack[i], pred_stack[i])
        scores.append(score)

    return np.mean(scores)


def microms3im(pred_img: Union[np.ndarray, torch.Tensor],
               gt_img: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate Micro-MS3IM metric for microscopy images.

    Multi-scale variant of Micro-SSIM.

    Args:
        pred_img: Predicted image
        gt_img: Ground truth image

    Returns:
        Micro-MS3IM score
    """
    try:
        from microssim import MicroMS3IM
    except ImportError:
        raise ImportError("microssim package is required. Install with: pip install microssim")

    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    assert pred_img.shape == gt_img.shape, f"Shape mismatch: {pred_img.shape} vs {gt_img.shape}"

    shape = pred_img.shape
    if len(shape) == 4:
        B, C, H, W = shape
        pred_stack = pred_img.reshape(B * C, H, W)
        gt_stack = gt_img.reshape(B * C, H, W)
    elif len(shape) == 3:
        pred_stack = pred_img
        gt_stack = gt_img
    elif len(shape) == 2:
        pred_stack = pred_img[np.newaxis, ...]
        gt_stack = gt_img[np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    microms3im_evaluator = MicroMS3IM()
    microms3im_evaluator.fit(gt_stack, pred_stack)

    scores = []
    for i in range(len(gt_stack)):
        score = microms3im_evaluator.score(gt_stack[i], pred_stack[i]).item()
        scores.append(score)

    return np.mean(scores)


def dice(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate Dice coefficient for binary segmentation.

    Args:
        y_pred: Predicted binary mask
        y_true: Ground truth binary mask

    Returns:
        Dice score in [0, 1]

    Example:
        >>> dice_score = dice(pred_mask, gt_mask)
    """
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2. * intersection) / (union + 1e-5)
    return dsc


def dice_val(y_pred: torch.Tensor, y_true: torch.Tensor, num_clus: int) -> torch.Tensor:
    """
    Calculate Dice score for multi-class 3D segmentation.

    Args:
        y_pred: Predicted segmentation (B, 1, D, H, W)
        y_true: Ground truth segmentation (B, 1, D, H, W)
        num_clus: Number of classes

    Returns:
        Mean Dice score across all classes

    Example:
        >>> dice_score = dice_val(predictions, ground_truth, num_clus=46)
    """
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2. * intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def dice_val_VOI(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Dice score for volumes of interest (brain regions).

    Args:
        y_pred: Predicted segmentation
        y_true: Ground truth segmentation

    Returns:
        Mean Dice score across specified VOI labels
    """
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23,
                25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2. * intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        idx += 1
    return np.mean(DSCs)


def dice_val_substruct(y_pred: torch.Tensor, y_true: torch.Tensor, std_idx: int) -> str:
    """
    Calculate Dice score for each substructure and return as CSV line.

    Args:
        y_pred: Predicted segmentation
        y_true: Ground truth segmentation
        std_idx: Subject index

    Returns:
        CSV formatted string with Dice scores for each class
    """
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = f'p_{std_idx}'
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2. * intersection) / (union + 1e-5)
        line = line + ',' + str(dsc)
    return line


def smooth_seg(binary_img: np.ndarray, sigma: float = 1.5, thresh: float = 0.4) -> np.ndarray:
    """
    Smooth binary segmentation mask using Gaussian filter.

    Args:
        binary_img: Binary segmentation mask
        sigma: Gaussian kernel sigma
        thresh: Threshold for binarization after smoothing

    Returns:
        Smoothed binary mask
    """
    binary_img = gaussian_filter(binary_img.astype(np.float32), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img


def jacobian_determinant_vxm(disp: np.ndarray) -> np.ndarray:
    """
    Calculate Jacobian determinant of displacement field.

    Used to measure local volume changes in deformation fields.

    Args:
        disp: Displacement field of shape (3, D, H, W) for 3D or (2, H, W) for 2D

    Returns:
        Jacobian determinant at each spatial location

    Example:
        >>> jac_det = jacobian_determinant_vxm(displacement_field)
        >>> folding_voxels = np.sum(jac_det < 0)  # Count folding locations
    """
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'Flow must be 2D or 3D'

    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    J = np.gradient(disp + grid)

    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # 2D
        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


# ============================================================================
# Debugging and Monitoring
# ============================================================================

def check_nan(tensor: torch.Tensor, name: str) -> bool:
    """
    Check if tensor contains NaN values.

    Args:
        tensor: Input tensor
        name: Name for logging

    Returns:
        True if NaN detected, False otherwise
    """
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False


def check_grad_nan(model: nn.Module) -> bool:
    """
    Check if model gradients contain NaN values.

    Args:
        model: PyTorch model

    Returns:
        True if NaN detected in any gradient, False otherwise
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                return True
    return False


def flatten_loss_dict(loss_dict: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, float]:
    """
    Flatten nested loss dictionary for logging.

    Args:
        loss_dict: Nested dictionary of losses
        parent_key: Parent key for recursion
        sep: Separator for flattened keys

    Returns:
        Flattened dictionary

    Example:
        >>> losses = {'reg': {'ncc': 0.5, 'smooth': 0.1}, 'total': 0.6}
        >>> flat = flatten_loss_dict(losses)
        >>> print(flat)  # {'reg_ncc': 0.5, 'reg_smooth': 0.1, 'total': 0.6}
    """
    items = []
    for k, v in loss_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_loss_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AverageMeter(object):
    """
    Computes and stores the average and current value.

    Example:
        >>> loss_meter = AverageMeter()
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     loss_meter.update(loss.item())
        >>> print(f"Average loss: {loss_meter.avg:.4f} ± {loss_meter.std:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


# ============================================================================
# Logging
# ============================================================================

class Logger(object):
    """
    Logger that writes to both terminal and log file.

    Example:
        >>> logger = Logger(save_dir='./logs')
        >>> sys.stdout = logger
        >>> print("This goes to both terminal and log file")
        >>> logger.close()
    """

    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory to save log file
        """
        self.terminal = sys.stdout
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log = open(os.path.join(save_dir, "logfile.log"), "a")

    def write(self, message: str):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close log file."""
        self.log.close()


def write2csv(line: str, name: str):
    """
    Append line to CSV file.

    Args:
        line: Line to write
        name: CSV filename (without extension)
    """
    with open(name + '.csv', 'a') as file:
        file.write(line)
        file.write('\n')


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(state: Dict[str, Any],
                    config,
                    is_best: bool,
                    save_dir: str = 'models',
                    filename: str = 'checkpoint.pth.tar',
                    max_model_num: int = 4):
    """
    Save model checkpoint and manage checkpoint files.

    Args:
        state: Dictionary containing model state
        config: Configuration object
        is_best: Whether this is the best model so far
        save_dir: Directory to save checkpoints
        filename: Checkpoint filename
        max_model_num: Maximum number of checkpoints to keep

    Example:
        >>> state = {
        ...     'epoch': epoch,
        ...     'state_dict': model.state_dict(),
        ...     'optimizer': optimizer.state_dict(),
        ...     'best_loss': best_loss,
        ... }
        >>> save_checkpoint(state, config, is_best=True)
    """
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'model_best.pth.tar'))

    all_checkpoints = glob.glob(os.path.join(save_dir, '*.pth.tar'))
    all_checkpoints = [ckpt for ckpt in all_checkpoints if not ckpt.endswith('model_best.pth.tar')]

    if len(all_checkpoints) > max_model_num:
        oldest_file = min(all_checkpoints, key=os.path.getctime)
        os.remove(oldest_file)


# ============================================================================
# Learning Rate Scheduling
# ============================================================================

class WarmupCosineSchedule(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine annealing.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        t_total: Total number of training steps
        cycles: Number of cosine cycles (default: 0.5 for half cycle)
        last_epoch: The index of last epoch
        warmup_start_factor: Starting factor for warmup (default: 0.01)

    Example:
        >>> scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1000, t_total=10000)
        >>> for epoch in range(num_epochs):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(self, optimizer, warmup_steps: int, t_total: int,
                 cycles: float = 0.5, last_epoch: int = -1,
                 warmup_start_factor: float = 0.01):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.warmup_start_factor = warmup_start_factor
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * ((self.warmup_start_factor - 1) * (self.warmup_steps - self.last_epoch) / self.warmup_steps + 1)
                    for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.t_total - self.warmup_steps)
            return [base_lr * (0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
                    for base_lr in self.base_lrs]


# ============================================================================
# Distributed Training
# ============================================================================

def get_free_port() -> int:
    """
    Get free network port for distributed training.

    Returns:
        Free port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup(rank: int, world_size: int, port: int):
    """
    Setup distributed training process group.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Master port

    Example:
        >>> setup(rank=0, world_size=4, port=12355)
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Cleanup distributed training process group."""
    dist.destroy_process_group()


# ============================================================================
# Spatial Transformations
# ============================================================================

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer for applying deformation fields.

    Supports both 2D and 3D spatial transformations.

    Args:
        size: Spatial size (H, W) for 2D or (D, H, W) for 3D
        mode: Interpolation mode ('bilinear' or 'nearest')

    Example:
        >>> transformer = SpatialTransformer(size=(160, 192, 224), mode='bilinear')
        >>> warped = transformer(source_image, flow_field)
    """

    def __init__(self, size: Tuple[int, ...], mode: str = 'bilinear'):
        super().__init__()
        self.mode = mode

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial transformation.

        Args:
            src: Source image (B, C, *spatial_dims)
            flow: Flow field (B, len(spatial_dims), *spatial_dims)

        Returns:
            Warped image (B, C, *spatial_dims)
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Move channels to last position and reverse order
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class RegisterModel(nn.Module):
    """
    Registration model wrapper using spatial transformer.

    Args:
        img_size: Image size tuple
        mode: Interpolation mode
    """

    def __init__(self, img_size: Tuple[int, ...] = (64, 256, 256), mode: str = 'bilinear'):
        super(RegisterModel, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply registration.

        Args:
            x: List of [image, flow]

        Returns:
            Warped image
        """
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out


def pad_image(img: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Pad 3D image to target size.

    Args:
        img: Input image (B, C, D, H, W)
        target_size: Target size (D, H, W)

    Returns:
        Padded image
    """
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def mk_grid_img(grid_step: int, line_thickness: int = 1,
                grid_sz: Tuple[int, int, int] = (160, 192, 224)) -> torch.Tensor:
    """
    Create 3D grid image for visualization.

    Args:
        grid_step: Grid spacing
        line_thickness: Thickness of grid lines
        grid_sz: Grid size (D, H, W)

    Returns:
        Grid image tensor on GPU
    """
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


# ============================================================================
# Uncertainty Estimation
# ============================================================================

def get_mc_preds(net: nn.Module, inputs: torch.Tensor, mc_iter: int = 25) -> Tuple[List, List]:
    """
    Get Monte Carlo predictions for uncertainty estimation.

    Args:
        net: Neural network with dropout
        inputs: Input tensor
        mc_iter: Number of MC samples

    Returns:
        Tuple of (image_list, flow_list)

    Example:
        >>> img_list, flow_list = get_mc_preds(model, inputs, mc_iter=50)
        >>> uncertainty = calc_uncert(target, img_list)
    """
    img_list = []
    flow_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
    return img_list, flow_list


def calc_uncert(tar: torch.Tensor, img_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Calculate uncertainty from MC predictions.

    Args:
        tar: Target image
        img_list: List of MC predictions

    Returns:
        Uncertainty map
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar) ** 2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def calc_error(tar: torch.Tensor, img_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Calculate error from MC predictions.

    Args:
        tar: Target image
        img_list: List of MC predictions

    Returns:
        Error map
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar) ** 2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def get_mc_preds_w_errors(net: nn.Module, inputs: torch.Tensor,
                          target: torch.Tensor, mc_iter: int = 25) -> Tuple[List, List, List]:
    """
    Get MC predictions with error tracking.

    Args:
        net: Neural network
        inputs: Input tensor
        target: Target tensor
        mc_iter: Number of MC samples

    Returns:
        Tuple of (image_list, flow_list, error_list)
    """
    img_list = []
    flow_list = []
    MSE = nn.MSELoss()
    err = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            err.append(MSE(img, target).item())
    return img_list, flow_list, err


def get_diff_mc_preds(net: nn.Module, inputs: torch.Tensor,
                      mc_iter: int = 25) -> Tuple[List, List, List]:
    """
    Get MC predictions for diffeomorphic registration.

    Args:
        net: Neural network
        inputs: Input tensor
        mc_iter: Number of MC samples

    Returns:
        Tuple of (image_list, flow_list, displacement_list)
    """
    img_list = []
    flow_list = []
    disp_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, _, flow, disp = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            disp_list.append(disp)
    return img_list, flow_list, disp_list


def uncert_regression_gal(img_list: List[torch.Tensor],
                          reduction: str = 'mean') -> Union[Tuple[float, float, float],
                                                            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Calculate aleatoric and epistemic uncertainty (Gal et al.).

    Args:
        img_list: List of MC predictions with uncertainty
        reduction: Reduction method ('mean', 'sum', or 'none')

    Returns:
        Tuple of (aleatoric, epistemic, total_uncertainty)
    """
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:, :-1].mean(dim=0, keepdim=True)
    ale = img_list[:, -1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:, :-1], dim=0, keepdim=True)
    epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi

    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'sum':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()


def uceloss(errors: torch.Tensor, uncert: torch.Tensor,
            n_bins: int = 15, outlier: float = 0.0,
            range: Optional[Tuple[float, float]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate Uncertainty Calibration Error (UCE) loss.

    Args:
        errors: Error tensor
        uncert: Uncertainty tensor
        n_bins: Number of bins
        outlier: Minimum bin proportion to include
        range: Uncertainty range for binning

    Returns:
        Tuple of (uce, errors_in_bin, avg_uncert_in_bin, prop_in_bin)
    """
    device = errors.device
    if range is None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []

    uce = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float().mean()
            avg_uncert_in_bin = uncert[in_bin].mean()
            uce += torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
    prop_in_bin = torch.tensor(prop_in_bin_list, device=device)

    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin


# ============================================================================
# Visualization
# ============================================================================

def visualize_logits(logits: Dict[str, Any],
                     output_folder: Optional[str] = None,
                     slice_idx: Optional[int] = None,
                     figsize: Tuple[int, int] = (20, 20),
                     verbose: bool = True):
    """
    Visualize model outputs including images, flows, and overlays.

    Supports both 2D and 3D data visualization.

    Args:
        logits: Dictionary containing model outputs
        output_folder: Directory to save visualizations (if None, displays plots)
        slice_idx: Slice index for 3D data (if None, uses middle slice)
        figsize: Figure size
        verbose: Whether to print structure information

    Example:
        >>> logits = {
        ...     'raw': raw_image,
        ...     'registration': {
        ...         'registered': registered_img,
        ...         'flow': flow_field,
        ...         'deformed_grid': grid
        ...     }
        ... }
        >>> visualize_logits(logits, output_folder='./vis')
    """
    def print_dict_structure(d, indent=0):
        for key, value in d.items():
            print('  ' * indent + str(key))
            if isinstance(value, dict):
                print_dict_structure(value, indent+1)
            else:
                print('  ' * (indent+1) + f"{type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")

    if verbose:
        print("Logits structure:")
        print_dict_structure(logits)
        print("\n")

    def get_slice(img, idx):
        """Extract slice from image (handles both 2D and 3D)."""
        if img.ndim == 5:  # 3D: (B, C, D, H, W)
            if img.shape[1] == 3:  # Flow data
                return img[0, :, idx].transpose(1, 2, 0)
            else:
                return img[0, 0, idx]
        elif img.ndim == 4:  # Could be 2D (B, C, H, W) or 3D (B, D, H, W)
            if img.shape[1] == 2:  # 2D flow
                return img[0].transpose(1, 2, 0)
            else:
                return img[0, 0]  # 2D image
        elif img.ndim == 3:  # (D, H, W) or (C, H, W)
            if idx is not None and img.shape[0] > idx:
                return img[idx]
            else:
                return img[0] if img.shape[0] <= 3 else img
        return img

    def visualize_flow(flow, ax):
        """Visualize flow field."""
        if flow.ndim == 2:
            magnitude = np.linalg.norm(flow, axis=-1)
            ax.imshow(magnitude, cmap='viridis')
            ax.set_title('Flow Magnitude')
        elif flow.ndim == 3:
            if flow.shape[-1] == 3:  # 3D flow
                flow_norm = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-8)
                flow_rgb = (flow_norm + 1) / 2
                ax.imshow(flow_rgb)
                ax.set_title('Flow (RGB)')
            elif flow.shape[-1] == 2:  # 2D flow
                flow_norm = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-8)
                flow_rgb = np.zeros((flow.shape[0], flow.shape[1], 3))
                flow_rgb[..., :2] = (flow_norm + 1) / 2
                ax.imshow(flow_rgb)
                ax.set_title('Flow (RGB)')
        else:
            print(f"Unexpected flow shape: {flow.shape}")

    def create_misalignment_overlay(img1, img2, enhance_diff=True):
        """Create overlay showing misalignment between two images."""
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())

        diff = np.abs(img1_norm - img2_norm)

        if enhance_diff:
            diff = np.power(diff, 0.5)
            threshold = 0.1
            diff[diff < threshold] = 0

        overlay = np.zeros((img1.shape[0], img1.shape[1], 3))
        overlay[:, :, 0] = img1_norm
        overlay[:, :, 2] = img2_norm
        overlay[:, :, 1] = diff  # Green shows difference

        return overlay

    def create_complementary_image(raw_img, mask_a, mask_b):
        """Create complementary image from two masks."""
        complementary = np.where(mask_a > 0, raw_img, np.where(mask_b > 0, raw_img, 0))
        return complementary

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Determine slice index
    raw_key = 'raw' if 'raw' in logits else list(logits.keys())[0]
    if slice_idx is None and hasattr(logits[raw_key], 'shape'):
        if logits[raw_key].ndim >= 5:  # 3D data
            slice_idx = logits[raw_key].shape[2] // 2
        else:
            slice_idx = None

    raw_slice = get_slice(logits[raw_key], slice_idx)

    for key, value in logits.items():
        if isinstance(value, dict):
            n_images = 1  # Raw image
            for sub_key in value.keys():
                n_images += 1

            # Add overlay images
            if 'registered' in value:
                n_images += 1
            if 'deformed' in value:
                n_images += 1
            if 'masked_A' in value and 'masked_B' in value:
                n_images += 1

            # Calculate grid layout
            n_cols = int(np.ceil(np.sqrt(n_images)))
            n_rows = int(np.ceil(n_images / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            fig.suptitle(f"Visualization of {key}", fontsize=16)

            # Plot raw image first
            axes[0].imshow(raw_slice, cmap='gray')
            axes[0].set_title('Raw')
            axes[0].axis('off')

            ax_index = 1
            for sub_key, img in value.items():
                if ax_index < len(axes):
                    ax = axes[ax_index]
                    try:
                        if sub_key in ['deformed_grid', 'restored_grid']:
                            grid_slice = get_slice(img, slice_idx)
                            ax.imshow(grid_slice, cmap='gray', vmin=0, vmax=1)
                            ax.set_title(f'{sub_key} (Grid Only)')
                        elif sub_key in ['flow', 'inverted_flow']:
                            visualize_flow(get_slice(img, slice_idx), ax)
                        else:
                            ax.imshow(get_slice(img, slice_idx), cmap='gray')

                        ax.set_title(sub_key)
                        ax.axis('off')
                        ax_index += 1
                    except Exception as e:
                        print(f"Error visualizing {sub_key}: {e}")

            # Add overlays
            if 'registered' in value and ax_index < len(axes):
                ax = axes[ax_index]
                raw_img = get_slice(logits[raw_key], slice_idx)
                reg_img = get_slice(value['registered'], slice_idx)
                overlay = create_misalignment_overlay(raw_img, reg_img, enhance_diff=True)
                ax.imshow(overlay)
                ax.set_title('Misalignment (Raw: Red, Registered: Blue)\nGreen: Difference')
                ax.axis('off')
                ax_index += 1

            if 'deformed' in value and ax_index < len(axes):
                ax = axes[ax_index]
                raw_img = get_slice(logits[raw_key], slice_idx)
                deformed_img = get_slice(value['deformed'], slice_idx)
                overlay = create_misalignment_overlay(raw_img, deformed_img, enhance_diff=True)
                ax.imshow(overlay)
                ax.set_title('Misalignment (Raw: Red, Deformed: Blue)\nGreen: Difference')
                ax.axis('off')
                ax_index += 1

            if 'masked_A' in value and 'masked_B' in value and ax_index < len(axes):
                ax = axes[ax_index]
                raw_slice = get_slice(logits[raw_key], slice_idx)
                mask_a = get_slice(value['masked_A'], slice_idx)
                mask_b = get_slice(value['masked_B'], slice_idx)
                complementary_img = create_complementary_image(raw_slice, mask_a, mask_b)
                ax.imshow(complementary_img, cmap='gray')
                ax.set_title('Complementary Image\n(A and B combined)')
                ax.axis('off')
                ax_index += 1

            # Remove extra subplots
            for i in range(ax_index, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()

            if output_folder:
                output_path = os.path.join(output_folder, f"{key}.png")
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()


# ============================================================================
# Label Processing (FreeSurfer specific)
# ============================================================================

def process_label() -> Dict[int, str]:
    """
    Process FreeSurfer labeling information.

    Returns:
        Dictionary mapping label indices to names

    Note:
        Requires label_info.txt file at hardcoded path.
    """
    import re

    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                 63, 72, 77, 80, 85, 251, 252, 253, 254, 255]

    try:
        file1 = open('./data/IXI_data/label_info.txt', 'r')
    except FileNotFoundError:
        print("Warning: label_info.txt not found, returning empty dict")
        return {}

    Lines = file1.readlines()
    label_dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ', line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                label_dict[seg_i] = line[1]
        seg_i += 1
    return label_dict


# Export all functions and classes
__all__ = [
    # Metrics
    'psnr',
    'carepsnr',
    'microssim',
    'microms3im',
    'dice',
    'dice_val',
    'dice_val_VOI',
    'dice_val_substruct',
    'smooth_seg',
    'jacobian_determinant_vxm',

    # Debugging
    'check_nan',
    'check_grad_nan',
    'flatten_loss_dict',
    'AverageMeter',

    # Logging
    'Logger',
    'write2csv',

    # Checkpointing
    'save_checkpoint',

    # Scheduling
    'WarmupCosineSchedule',

    # Distributed
    'get_free_port',
    'setup',
    'cleanup',

    # Spatial Transforms
    'SpatialTransformer',
    'RegisterModel',
    'pad_image',
    'mk_grid_img',

    # Uncertainty
    'get_mc_preds',
    'calc_uncert',
    'calc_error',
    'get_mc_preds_w_errors',
    'get_diff_mc_preds',
    'uncert_regression_gal',
    'uceloss',

    # Visualization
    'visualize_logits',

    # Label Processing
    'process_label',
]
