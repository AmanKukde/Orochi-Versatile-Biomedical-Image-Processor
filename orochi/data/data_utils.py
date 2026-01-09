"""
Data loading and preprocessing utilities for Orochi.

This module provides utility functions for:
- Loading pickled data files
- Worker initialization for reproducible data loading
- Coordinate grid generation for medical imaging
- Mask operations for segmentation
- Sampling utilities

Author: Orochi Team
"""

import random
import pickle
from typing import Tuple, List, Optional, Any

import numpy as np
import torch


# Maximum seed value for random number generators (2^32 - 1)
M = 2**32 - 1


def init_fn(worker: int) -> None:
    """
    Initialize worker with unique random seed.

    This function ensures each data loader worker has a unique, reproducible
    random seed for deterministic data augmentation.

    Args:
        worker (int): Worker ID assigned by PyTorch DataLoader

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=init_fn)

    Note:
        - Seed is derived from a random value plus worker ID
        - Seeds both numpy and Python random generators
        - Ensures reproducibility across training runs when combined with manual seed setting
    """
    # Generate a random seed
    seed = torch.LongTensor(1).random_().item()

    # Make seed unique for each worker
    seed = (seed + worker) % M

    # Seed all random number generators
    np.random.seed(seed)
    random.seed(seed)


def pkload(fname: str) -> Any:
    """
    Load data from a pickle file.

    Args:
        fname (str): Path to pickle file

    Returns:
        Any: Unpickled Python object

    Example:
        >>> data, labels = pkload('brain_scan.pkl')
        >>> print(data.shape)  # (160, 192, 224)

    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.UnpicklingError: If file is corrupted or not a valid pickle
    """
    with open(fname, "rb") as f:
        return pickle.load(f)


def add_mask(
    x: torch.Tensor, mask: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """
    Add one-hot encoded mask to tensor along specified dimension.

    This function expands the tensor along the specified dimension and prepends
    a one-hot encoded version of the mask.

    Args:
        x (torch.Tensor): Input tensor
        mask (torch.Tensor): Mask tensor with integer labels
        dim (int): Dimension along which to add mask (default: 1)

    Returns:
        torch.Tensor: Tensor with mask prepended along specified dimension

    Example:
        >>> x = torch.randn(2, 3, 64, 64)  # [B, C, H, W]
        >>> mask = torch.randint(0, 21, (2, 64, 64))  # [B, H, W]
        >>> result = add_mask(x, mask, dim=1)
        >>> print(result.shape)  # torch.Size([2, 24, 64, 64])  # 21 + 3 channels

    Note:
        - Mask values should be in range [0, 20] (will create 21 one-hot channels)
        - Original tensor is placed after the one-hot mask channels
    """
    # Add dimension to mask to match tensor
    mask = mask.unsqueeze(dim)

    # Calculate new shape with 21 additional channels for one-hot encoding
    shape = list(x.shape)
    shape[dim] += 21

    # Create new tensor initialized with zeros
    new_x = x.new(*shape).zero_()

    # Scatter 1.0 values at mask positions (one-hot encoding)
    new_x = new_x.scatter_(dim, mask, 1.0)

    # Copy original tensor after the one-hot mask
    s = [slice(None)] * len(shape)
    s[dim] = slice(21, None)  # Channels 21 onwards
    new_x[s] = x

    return new_x


def sample(x: np.ndarray, size: int) -> torch.Tensor:
    """
    Randomly sample elements from array without replacement.

    Args:
        x (np.ndarray): Input array to sample from
        size (int): Number of elements to sample

    Returns:
        torch.Tensor: Sampled elements as int16 tensor

    Example:
        >>> coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
        >>> sampled = sample(coords, 2)
        >>> print(sampled.shape)  # torch.Size([2, 3])

    Note:
        - Sampling is without replacement (each element appears at most once)
        - Returns tensor with dtype int16 to save memory
    """
    # Randomly sample indices
    i = random.sample(range(x.shape[0]), size)

    # Return sampled elements as tensor
    return torch.tensor(x[i], dtype=torch.int16)


# Default brain MRI shape (standard T1 MRI dimensions)
_DEFAULT_BRAIN_SHAPE = (240, 240, 155)


def get_all_coords(stride: int, shape: Tuple[int, int, int] = _DEFAULT_BRAIN_SHAPE) -> torch.Tensor:
    """
    Generate 3D coordinate grid with specified stride.

    Creates a regular grid of 3D coordinates for sampling or patch extraction
    from volumetric medical images.

    Args:
        stride (int): Spacing between grid points
        shape (Tuple[int, int, int]): Volume dimensions (D, H, W)
            Default: (240, 240, 155) for standard brain MRI

    Returns:
        torch.Tensor: Coordinate array of shape [N, 3] where N is number of grid points
            Each row contains (z, y, x) coordinates

    Example:
        >>> coords = get_all_coords(stride=16)
        >>> print(coords.shape)  # torch.Size([3465, 3]) for default brain shape
        >>> print(coords[0])  # tensor([8, 8, 8])  # First coordinate at stride//2 offset

    Note:
        - Grid starts at stride//2 offset to avoid boundary issues
        - Coordinates are center-aligned within stride blocks
        - Returned as int16 tensor to save memory
    """
    # Create meshgrid with stride offset
    # Start at stride//2 to center-align grid points
    grids = np.meshgrid(
        *[stride // 2 + np.arange(0, s, stride) for s in shape],
        indexing="ij"  # Matrix indexing (z, y, x)
    )

    # Stack and reshape to [N, 3] format
    coords = np.stack([v.reshape(-1) for v in grids], axis=-1)

    return torch.tensor(coords, dtype=torch.int16)


_zero = torch.tensor([0])


def gen_feats(shape: Tuple[int, int, int] = _DEFAULT_BRAIN_SHAPE) -> np.ndarray:
    """
    Generate normalized 3D position features.

    Creates a feature grid where each voxel contains its normalized 3D position
    relative to the volume center. Useful for position encoding in neural networks.

    Args:
        shape (Tuple[int, int, int]): Volume dimensions (D, H, W)
            Default: (240, 240, 155) for standard brain MRI

    Returns:
        np.ndarray: Position features of shape [D, H, W, 3]
            - Values in range [-0.5, 0.5]
            - Center of volume is at (0, 0, 0)
            - Each voxel contains (z_norm, y_norm, x_norm)

    Example:
        >>> feats = gen_feats(shape=(64, 64, 64))
        >>> print(feats.shape)  # (64, 64, 64, 3)
        >>> print(feats[32, 32, 32])  # [0., 0., 0.]  # Center voxel
        >>> print(feats.min(), feats.max())  # -0.5 0.5

    Note:
        - Features are normalized by volume dimensions
        - Center of volume maps to (0, 0, 0)
        - Useful for adding position information to neural networks
        - Compatible with coordinate attention mechanisms
    """
    x, y, z = shape

    # Create coordinate meshgrid
    feats = np.stack(
        np.meshgrid(
            np.arange(x), np.arange(y), np.arange(z), indexing="ij"
        ),
        axis=-1,
    ).astype("float32")

    # Normalize: shift to [-shape/2, shape/2] then divide by shape
    shape_array = np.array([x, y, z])
    feats -= shape_array / 2.0
    feats /= shape_array

    return feats


def normalize_intensity(
    img: np.ndarray,
    percentile: Tuple[float, float] = (1, 99),
    clip: bool = True
) -> np.ndarray:
    """
    Normalize image intensity using percentile-based scaling.

    Args:
        img (np.ndarray): Input image
        percentile (Tuple[float, float]): Lower and upper percentiles for normalization
        clip (bool): Whether to clip values to [0, 1] after normalization

    Returns:
        np.ndarray: Normalized image with values approximately in [0, 1]

    Example:
        >>> img = np.random.randn(100, 100, 100) * 50 + 100
        >>> img_norm = normalize_intensity(img, percentile=(1, 99))
        >>> print(img_norm.min(), img_norm.max())  # ~0.0 ~1.0
    """
    # Compute percentile values
    pmin, pmax = np.percentile(img, percentile)

    # Normalize to [0, 1]
    if pmax > pmin:
        img = (img - pmin) / (pmax - pmin)

    # Clip outliers if requested
    if clip:
        img = np.clip(img, 0, 1)

    return img


def pad_to_size(
    img: np.ndarray,
    target_shape: Tuple[int, ...],
    mode: str = "constant",
    constant_value: float = 0
) -> np.ndarray:
    """
    Pad image to target shape.

    Args:
        img (np.ndarray): Input image of any dimension
        target_shape (Tuple[int, ...]): Target shape (must be >= img.shape)
        mode (str): Padding mode ('constant', 'reflect', 'edge')
        constant_value (float): Value for constant padding

    Returns:
        np.ndarray: Padded image

    Example:
        >>> img = np.random.rand(50, 50, 50)
        >>> padded = pad_to_size(img, (64, 64, 64))
        >>> print(padded.shape)  # (64, 64, 64)
    """
    assert len(img.shape) == len(target_shape), "Shape dimension mismatch"

    # Calculate padding amounts
    pad_width = []
    for current, target in zip(img.shape, target_shape):
        assert target >= current, f"Target size {target} < current size {current}"
        pad_before = (target - current) // 2
        pad_after = target - current - pad_before
        pad_width.append((pad_before, pad_after))

    # Apply padding
    if mode == "constant":
        return np.pad(img, pad_width, mode=mode, constant_values=constant_value)
    else:
        return np.pad(img, pad_width, mode=mode)


def crop_center(img: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Extract center crop from image.

    Args:
        img (np.ndarray): Input image of any dimension
        target_shape (Tuple[int, ...]): Target crop shape (must be <= img.shape)

    Returns:
        np.ndarray: Center-cropped image

    Example:
        >>> img = np.random.rand(100, 100, 100)
        >>> cropped = crop_center(img, (64, 64, 64))
        >>> print(cropped.shape)  # (64, 64, 64)
    """
    assert len(img.shape) == len(target_shape), "Shape dimension mismatch"

    # Calculate crop slices
    slices = []
    for current, target in zip(img.shape, target_shape):
        assert target <= current, f"Target size {target} > current size {current}"
        start = (current - target) // 2
        slices.append(slice(start, start + target))

    return img[tuple(slices)]
