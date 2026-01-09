"""
Consolidated dataset classes for Orochi biomedical image processing.

This module provides a unified interface for various biomedical imaging datasets,
supporting both 2D and 3D data formats across multiple imaging modalities including:
- Brain imaging (IXI, OASIS)
- Liver imaging (ISO)
- Microscopy (projection, denoising, super-resolution)
- Image fusion (FAT, BSA)
- Medical segmentation (MedSAM, SAM)

Example:
    >>> from orochi.data import get_dataset
    >>> # Get a 2D dataset
    >>> dataset = get_dataset('medsam_seg_2d', config=config, is_train=True)
    >>> # Get a 3D dataset
    >>> dataset = get_dataset('ixi_brain_3d', data_path=paths, atlas_path=atlas, transforms=transforms)
"""

import os
import glob
import random
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any

import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
from collections.abc import Sequence


# ============================================================================
# Utility Functions
# ============================================================================

def pkload(fname: str) -> Any:
    """
    Load pickle file.

    Args:
        fname: Path to pickle file

    Returns:
        Unpickled data
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)


def random_crop(image_tensor: torch.Tensor, target_size: Tuple[int, ...]) -> torch.Tensor:
    """
    Randomly crop image tensor to target size.

    Args:
        image_tensor: Input tensor of shape (C, H, W) for 2D or (C, D, H, W) for 3D
        target_size: Target dimensions (H, W) for 2D or (D, H, W) for 3D

    Returns:
        Cropped and resized tensor
    """
    if len(image_tensor.shape) == 3:  # 2D
        _, h, w = image_tensor.shape
        th, tw = target_size

        if h > th:
            start_h = torch.randint(0, h - th + 1, (1,)).item()
            image_tensor = image_tensor[:, start_h:start_h+th]
        if w > tw:
            start_w = torch.randint(0, w - tw + 1, (1,)).item()
            image_tensor = image_tensor[:, :, start_w:start_w+tw]

        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=target_size,
                                     mode='bilinear', align_corners=False).squeeze(0)
    else:  # 3D
        _, d, h, w = image_tensor.shape
        td, th, tw = target_size

        if d > td:
            start_d = torch.randint(0, d - td + 1, (1,)).item()
            image_tensor = image_tensor[:, start_d:start_d+td]
        if h > th:
            start_h = torch.randint(0, h - th + 1, (1,)).item()
            image_tensor = image_tensor[:, :, start_h:start_h+th]
        if w > tw:
            start_w = torch.randint(0, w - tw + 1, (1,)).item()
            image_tensor = image_tensor[:, :, :, start_w:start_w+tw]

        image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=target_size,
                                     mode='trilinear', align_corners=False).squeeze(0)

    return image_tensor


class CollateFn:
    """Collate function for batching with random cropping."""

    def __init__(self, target_size: Tuple[int, ...]):
        """
        Args:
            target_size: Target size for cropping (H, W) for 2D or (D, H, W) for 3D
        """
        self.target_size = target_size

    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Process batch with random cropping.

        Args:
            batch: List of tensors

        Returns:
            Stacked batch tensor
        """
        processed_batch = [random_crop(item, self.target_size) for item in batch]
        return torch.stack(processed_batch, dim=0)


# ============================================================================
# Transform Classes (for compatibility with 3D datasets)
# ============================================================================

class Base(object):
    """Base transformation class."""

    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)

        if isinstance(img, Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]

        return self.tf(img)

    def __str__(self):
        return 'Identity()'


Identity = Base


class RandomFlip(Base):
    """Random flip transformation for 3D images."""

    def __init__(self, axis=0):
        self.axis = (1, 2, 3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        self.z_buffer = np.random.choice([True, False])
        return list(shape)

    def tf(self, img, k=0):
        if self.x_buffer:
            img = np.flip(img, axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img, axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img, axis=self.axis[2])
        return img


class Seg_norm(Base):
    """Segmentation normalization transformation."""

    def __init__(self):
        self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])

    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out


class NumpyType(Base):
    """Convert numpy array types."""

    def __init__(self, types, num=-1):
        self.types = types
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return f'NumpyType(({s}))'


# ============================================================================
# Pretraining Datasets
# ============================================================================

class PretrainDataset2D(Dataset):
    """
    2D pretraining dataset for biomedical images.

    Supports TIFF and NPY formats with automatic upsampling and normalization.

    Args:
        root_dir: Root directory containing image files
        img_size: Target image size (H, W)

    Example:
        >>> dataset = PretrainDataset2D('/path/to/data', img_size=(256, 256))
        >>> image = dataset[0]  # Returns normalized tensor of shape (1, H, W)
    """

    def __init__(self, root_dir: str, img_size: Tuple[int, int]):
        self.root_dir = root_dir
        self.img_size = img_size
        self.samples = self._create_sample_list()

    def _create_sample_list(self) -> List[str]:
        """Get list of valid image files."""
        return [f for f in os.listdir(self.root_dir) if (f.endswith('.tiff') or f.endswith('.npy'))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get preprocessed image.

        Returns:
            Tensor of shape (1, H, W) with values in [0, 1]
        """
        sample_path = os.path.join(self.root_dir, self.samples[idx])

        if sample_path.endswith('.tiff'):
            raw_image = tifffile.imread(sample_path)
        elif sample_path.endswith('.npy'):
            raw_image = np.load(sample_path)
        else:
            raise ValueError(f"Unsupported file format: {sample_path}")

        # Handle 3D images by taking middle slice
        if raw_image.ndim == 3:
            raw_image = raw_image[raw_image.shape[0]//2]
        elif raw_image.ndim > 3:
            raise ValueError(f"Unexpected image dimensions: {raw_image.shape}")

        image_tensor = torch.from_numpy(raw_image).float().unsqueeze(0)

        # Normalize to [0, 1]
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

        # Upsample if needed
        image_tensor = self.upsample(image_tensor)

        return image_tensor

    def upsample(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Upsample image to target size if needed."""
        h, w = image_tensor.shape[1:]
        th, tw = self.img_size

        h_factor = max(1, th / h)
        w_factor = max(1, tw / w)

        if h_factor > 1 or w_factor > 1:
            image_tensor = F.interpolate(image_tensor.unsqueeze(0),
                                         size=(int(h*h_factor), int(w*w_factor)),
                                         mode='bilinear',
                                         align_corners=False).squeeze(0)

        return image_tensor


class PretrainDataset3D(Dataset):
    """
    3D pretraining dataset for volumetric biomedical images.

    Supports TIFF and NPY formats with automatic upsampling and normalization.

    Args:
        root_dir: Root directory containing volume files
        img_size: Target volume size (D, H, W)

    Example:
        >>> dataset = PretrainDataset3D('/path/to/data', img_size=(64, 256, 256))
        >>> volume = dataset[0]  # Returns normalized tensor of shape (1, D, H, W)
    """

    def __init__(self, root_dir: str, img_size: Tuple[int, int, int]):
        self.root_dir = root_dir
        self.img_size = img_size
        self.samples = self._create_sample_list()

    def _create_sample_list(self) -> List[str]:
        """Get list of valid volume files."""
        return [f for f in os.listdir(self.root_dir) if (f.endswith('.tiff') or f.endswith('.npy'))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get preprocessed volume.

        Returns:
            Tensor of shape (1, D, H, W) with values in [0, 1]
        """
        sample_path = os.path.join(self.root_dir, self.samples[idx])

        if sample_path.endswith('.tiff'):
            raw_image = tifffile.imread(sample_path)
        elif sample_path.endswith('.npy'):
            raw_image = np.load(sample_path)
        else:
            raise ValueError(f"Unknown file format: {sample_path}")

        image_tensor = torch.from_numpy(raw_image).float().unsqueeze(0)

        # Normalize to [0, 1]
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

        # Upsample if needed
        image_tensor = self.upsample(image_tensor)

        return image_tensor

    def upsample(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Upsample volume to target size if needed."""
        d, h, w = image_tensor.shape[1:]
        td, th, tw = self.img_size

        d_factor = max(1, td / d)
        h_factor = max(1, th / h)
        w_factor = max(1, tw / w)

        if d_factor > 1 or h_factor > 1 or w_factor > 1:
            image_tensor = F.interpolate(image_tensor.unsqueeze(0),
                                         size=(int(d*d_factor), int(h*h_factor), int(w*w_factor)),
                                         mode='trilinear',
                                         align_corners=False).squeeze(0)

        return image_tensor


# ============================================================================
# Brain Imaging Datasets (3D)
# ============================================================================

class IXIBrainDataset3D(Dataset):
    """
    IXI brain MRI dataset for 3D image registration.

    Loads brain MRI volumes and segmentation masks for atlas-based registration tasks.

    Args:
        data_path: List of paths to subject data files
        atlas_path: Path to atlas reference file
        transforms: Transformation pipeline to apply

    Example:
        >>> dataset = IXIBrainDataset3D(data_paths, atlas_path, transforms)
        >>> x, y, x_seg, y_seg = dataset[0]
    """

    def __init__(self, data_path: List[str], atlas_path: str, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img: np.ndarray, C: int) -> np.ndarray:
        """Convert segmentation to one-hot encoding."""
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get image pair with segmentations.

        Returns:
            Tuple of (atlas_img, subject_img, atlas_seg, subject_seg)
        """
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self) -> int:
        return len(self.paths)


class IXIBrainInferDataset3D(Dataset):
    """
    IXI brain MRI inference dataset.

    Same as IXIBrainDataset3D but for inference/testing.
    """

    def __init__(self, data_path: List[str], atlas_path: str, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img: np.ndarray, C: int) -> np.ndarray:
        """Convert segmentation to one-hot encoding."""
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self) -> int:
        return len(self.paths)


class OASISBrainDataset3D(Dataset):
    """
    OASIS brain MRI dataset for 3D image registration.

    Performs pairwise registration between subjects in the dataset.
    """

    def __init__(self, data_path: List[str], transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img: np.ndarray, C: int) -> np.ndarray:
        """Convert segmentation to one-hot encoding."""
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get random pair of subjects for registration.

        Returns:
            Tuple of (source_img, target_img, source_seg, target_seg)
        """
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self) -> int:
        return len(self.paths)


class OASISBrainInferDataset3D(Dataset):
    """OASIS brain MRI inference dataset."""

    def __init__(self, data_path: List[str], transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img: np.ndarray, C: int) -> np.ndarray:
        """Convert segmentation to one-hot encoding."""
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self) -> int:
        return len(self.paths)


# ============================================================================
# Liver Imaging Datasets
# ============================================================================

class ISOLiverDataset2D(Dataset):
    """
    2D isotropic liver dataset.

    Args:
        config: Configuration object with data_dir, img_size attributes
        is_train: Whether this is training or test set

    Example:
        >>> dataset = ISOLiverDataset2D(config, is_train=True)
        >>> source, target = dataset[0]
    """

    def __init__(self, config, is_train: bool = True):
        self.data_dir = config.data_dir
        self.is_train = is_train
        self.img_size = config.img_size

        info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        if is_train:
            self.original_shape = info['original_train_shape']
        else:
            self.original_shape = info['original_test_shape']

        if is_train:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'train', 'slice_*.npz')))
        else:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'test', 'xz_slice_*.npz')))

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")

        sample_data = np.load(self.file_list[0])
        source_shape = sample_data['source'].shape
        print(f"Original data shape: {source_shape}")
        print(f"Target resize shape: {self.img_size}")

    def resize_2d(self, img: torch.Tensor) -> torch.Tensor:
        """Resize 2D image to target size."""
        c, h, w = img.shape
        img_resized = F.interpolate(
            img.unsqueeze(0),
            size=self.img_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        return img_resized

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                              Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
        """
        Get data sample.

        Returns:
            Training: (source, target)
            Testing: (source, target, position)
        """
        data = np.load(self.file_list[idx])
        source = torch.from_numpy(data['source']).float()
        target = torch.from_numpy(data['target']).float()

        if self.is_train:
            source = self.resize_2d(source)
            target = self.resize_2d(target)
            return source, target
        else:
            position = data['position']
            return source, target, position

    def __len__(self) -> int:
        return len(self.file_list)


class ISOLiverDataset3D(Dataset):
    """
    3D isotropic liver dataset.

    Args:
        config: Configuration object with data_dir attribute
        is_train: Whether this is training or test set
    """

    def __init__(self, config, is_train: bool = True):
        self.data_dir = config.data_dir
        self.is_train = is_train

        info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        self.original_shape = info['original_test_shape'] if not is_train else None

        if is_train:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'train', 'cluster_*.npz')))
        else:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'test', 'patch_*.npz')))

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")

        sample_data = np.load(self.file_list[0])
        source_shape = sample_data['source'].shape
        print(f"Original data shape: {source_shape}")

        if not is_train:
            self.patch_coords = []
            for file_path in self.file_list:
                data = np.load(file_path)
                self.patch_coords.append(data['position'])

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                              Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
        """Get data sample with optional position for test set."""
        data = np.load(self.file_list[idx])
        source = data['source']
        target = data['target']

        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()

        if self.is_train:
            return source, target
        else:
            position = data['position']
            source = source.squeeze(0)
            target = target.squeeze(0)
            return source, target, position

    def __len__(self) -> int:
        return len(self.file_list)


# ============================================================================
# Fusion Datasets
# ============================================================================

class BSAFusionDataset2D(Dataset):
    """
    2D medical image fusion dataset (CT-MRI, PET-MRI, SPECT-MRI).

    Args:
        config: Configuration object with data_dir and modularities attributes
        is_train: Whether this is training or test set

    Example:
        >>> config.modularities = "CT-MRI"
        >>> dataset = BSAFusionDataset2D(config, is_train=True)
        >>> source1, source2, target = dataset[0]
    """

    def __init__(self, config, is_train: bool = True):
        data_path = Path(config.data_dir)
        self.modularities = config.modularities
        modularity1 = self.modularities.split('-')[0]
        modularity2 = self.modularities.split('-')[1]

        if is_train:
            image_dir = data_path / self.modularities / "train"
            mod1_images = glob.glob(str(image_dir / modularity1 / "*.png"))
            mod2_images = glob.glob(str(image_dir / modularity2 / "*.png"))
        else:
            image_dir = data_path / self.modularities / "test"
            mod1_images = glob.glob(str(image_dir / modularity1 / "*.png"))
            mod2_images = glob.glob(str(image_dir / modularity2 / "*.png"))

        mod1_images = sorted(mod1_images)
        mod2_images = sorted(mod2_images)
        assert len(mod1_images) == len(mod2_images), "Number of images must match"
        for i in range(len(mod1_images)):
            assert Path(mod1_images[i]).name == Path(mod2_images[i]).name, "Image names must match"

        self.mod1_images = mod1_images
        self.mod2_images = mod2_images
        self.length = len(mod1_images)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get fusion pair.

        Returns:
            Tuple of (source1, source2, target)
        """
        source1_pil = Image.open(self.mod1_images[idx]).convert('RGB')
        source1 = transforms.ToTensor()(source1_pil)
        source2_pil = Image.open(self.mod2_images[idx]).convert('L')
        source2 = transforms.ToTensor()(source2_pil)
        source2 = source2.expand(3, -1, -1)

        target = source2
        return source1, source2, target


class FATFusionDataset3D(Dataset):
    """
    3D medical image fusion dataset.

    Args:
        config: Configuration object with data_dir attribute
        is_train: Whether this is training or test set
    """

    def __init__(self, config, is_train: bool = True):
        if is_train:
            self.data = np.load(os.path.join(config.data_dir, 'train_data.npy'))
        else:
            self.data = np.load(os.path.join(config.data_dir, 'test_data.npy'))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get fusion triplet.

        Returns:
            Tuple of (source1, source2, target) with shape (1, D, H, W)
        """
        item = self.data[idx]
        source1 = item[0, :, :, :]
        source2 = item[1, :, :, :]
        target = item[2, :, :, :]

        # Normalize to [0, 1]
        source1 = (source1 - source1.min()) / (source1.max() - source1.min()) if source1.max() != source1.min() else np.zeros_like(source1)
        source2 = (source2 - source2.min()) / (source2.max() - source2.min()) if source2.max() != source2.min() else np.zeros_like(source2)
        target = (target - target.min()) / (target.max() - target.min()) if target.max() != target.min() else np.zeros_like(target)

        # Transpose from (H, W, D) to (D, H, W)
        source1 = np.transpose(source1, (2, 0, 1))
        source2 = np.transpose(source2, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))

        source1 = torch.tensor(source1, dtype=torch.float32).unsqueeze(0)
        source2 = torch.tensor(source2, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        return source1, source2, target


# ============================================================================
# Super-Resolution Datasets
# ============================================================================

class UnifmirSRDataset2D(Dataset):
    """
    2D microscopy super-resolution dataset (CCPs, F-actin, ER, Microtubules).

    Args:
        config: Configuration object with data_dir, data_types, and img_size attributes
        is_train: Whether this is training or test set

    Example:
        >>> config.data_types = ['CCPs', 'F-actin']
        >>> dataset = UnifmirSRDataset2D(config, is_train=True)
        >>> source, target = dataset[0]
    """

    def __init__(self, config, is_train: bool = True):
        self.data_dir = Path(config.data_dir)
        self.data_types = config.data_types
        self.is_train = is_train

        if self.is_train:
            self.data_length = {'CCPs': 19440, 'F-actin': 19584, 'ER': 19584, 'Microtubules': 19800}
        else:
            self.data_length = {'CCPs': 100, 'F-actin': 100, 'ER': 100, 'Microtubules': 100}

        self.img_size = config.img_size

    def resize_2d(self, img: torch.Tensor) -> torch.Tensor:
        """Resize 2D image to target size."""
        h, w = img.shape
        img_resized = F.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            size=self.img_size,
            mode='bilinear',
            align_corners=False
        )
        return img_resized.squeeze()

    def __len__(self) -> int:
        length = 0
        for data_type in self.data_types:
            length += self.data_length[data_type]
        self.length = length
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get LR-HR pair.

        Returns:
            Tuple of (low_res, high_res) with shape (1, H, W)
        """
        data_length = self.data_length
        data_type = None
        for key in self.data_types:
            if idx < data_length[key]:
                data_type = key
                break
            else:
                idx -= data_length[key]

        if self.is_train:
            data_path = self.data_dir / 'train' / data_type
            source = np.load(data_path / f"preprocessed/X_{idx}.npy").squeeze()
            target = np.load(data_path / f"preprocessed/Y_{idx}.npy").squeeze()
        else:
            data_path = self.data_dir / 'test' / data_type
            source = tifffile.imread(data_path / 'LR' / f"im{idx+1}_LR.tif").squeeze()
            target = tifffile.imread(data_path / 'GT' / f"im{idx+1}_GT.tif").squeeze()

        # Normalize
        source = (source - source.min()) / (source.max() - source.min()) if source.max() != source.min() else np.zeros_like(source)
        target = (target - target.min()) / (target.max() - target.min()) if target.max() != target.min() else np.zeros_like(target)
        source = torch.tensor(source, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        source = self.resize_2d(source).unsqueeze(0)
        target = self.resize_2d(target).unsqueeze(0)
        return source, target


class InverseSRDataset3D(Dataset):
    """
    3D inverse super-resolution dataset.

    Args:
        config: Configuration object with data_dir and down_factor attributes
        is_train: Whether this is training or test set
    """

    def __init__(self, config, is_train: bool = True):
        if config.down_factor is None:
            config.down_factor = 4
        self.down_factor = config.down_factor
        self.data_dir = config.data_dir
        self.is_train = is_train

        if self.is_train:
            self.length = 464 if config.train_length is None else config.train_length
        else:
            self.length = 117 if config.val_length is None else config.val_length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get LR-HR volume pair.

        Returns:
            Tuple of (low_res, high_res) with shape (1, D, H, W)
        """
        if self.is_train:
            item = np.load(os.path.join(self.data_dir, f'train_data_{self.down_factor}x_{idx}.npy'))
        else:
            item = np.load(os.path.join(self.data_dir, f'val_data_{self.down_factor}x_{idx}.npy'))

        source = item[0].transpose(1, 0, 2)[np.newaxis, :, :, :]
        target = item[1].transpose(1, 0, 2)[np.newaxis, :, :, :]

        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()

        return source, target


# ============================================================================
# Projection and Denoising Datasets
# ============================================================================

class ProjFlywingDataset2D(Dataset):
    """
    2D projection fly wing dataset.

    Args:
        config: Configuration object with data_dir attribute
        is_train: Whether this is training or test set
        condition: Condition index for test data
    """

    def __init__(self, config, is_train: bool = True, condition: int = 0):
        self.config = config
        self.is_train = is_train
        self.condition = condition
        self.iso = ['Projection_Flywing']

        if self.is_train:
            self._load_train_data()
        else:
            self._load_test_data()

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {self.lenth} samples")

    def _load_train_data(self):
        datapath = f"{self.config.data_dir}/train_data/my_training_data.npz"
        X1, Y1 = self._load_npz_data(datapath)
        self.nm_lr = X1
        self.nm_hr = Y1
        self.lenth = len(self.nm_lr)

    def _load_test_data(self):
        dir_lr = f"{self.config.data_dir}/test_data/"
        self.nm_lr = sorted(glob.glob(f"{dir_lr}Input/C{self.condition}/*.tif"))
        self.nm_hr = sorted(glob.glob(f"{dir_lr}GT/C{self.condition}/*.tif"))
        self.lenth = len(self.nm_lr)

    def _load_npz_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load .npz data (784, 1, 50, 128, 128), SCZYX format."""
        data = np.load(path)
        return data['X'], data['Y']

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LR-HR pair."""
        idx = idx % self.lenth
        if self.is_train:
            lr, hr = self.nm_lr[idx], self.nm_hr[idx]
        else:
            lr = np.float32(tifffile.imread(self.nm_lr[idx]))
            hr = np.expand_dims(np.float32(tifffile.imread(self.nm_hr[idx])), 0)

        lr = torch.from_numpy(np.ascontiguousarray(lr * self.config.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.config.rgb_range)).float()

        return lr, hr

    def __len__(self) -> int:
        return self.lenth


class ProjFlywingDataset3D(Dataset):
    """
    3D projection fly wing dataset.

    Args:
        config: Configuration object with data_dir and condition attributes
        is_train: Whether this is training or test set
    """

    def __init__(self, config, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        self.condition = config.condition
        self.iso = ['Projection_Flywing']

        if self.is_train:
            self._load_train_data()
        else:
            self._load_test_data()

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {self.lenth} samples")

    def _load_train_data(self):
        datapath = f"{self.config.data_dir}/train_data/my_training_data.npz"
        X1, Y1 = self._load_npz_data(datapath)
        self.nm_lr = X1
        self.nm_hr = Y1
        self.lenth = len(self.nm_lr)

    def _load_test_data(self):
        dir_lr = f"{self.config.data_dir}/test_data/"
        self.nm_lr = sorted(glob.glob(f"{dir_lr}Input/C{self.condition}/*.tif"))
        self.nm_hr = sorted(glob.glob(f"{dir_lr}GT/C{self.condition}/*.tif"))
        self.lenth = len(self.nm_lr)

    def _load_npz_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load npz (784, 1, 50, 128, 128), SCZYX format."""
        data = np.load(path)
        return data['X'], data['Y']

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get LR-HR volume pair.

        Returns:
            Tuple of (low_res, high_res) with shape (1, 64, H, W)
        """
        idx = idx % self.lenth
        if self.is_train:
            lr, hr = self.nm_lr[idx], self.nm_hr[idx]
        else:
            lr = np.float32(tifffile.imread(self.nm_lr[idx]))
            hr = np.expand_dims(np.float32(tifffile.imread(self.nm_hr[idx])), 0)

        lr = torch.from_numpy(np.ascontiguousarray(lr)).float()

        # Add zero padding to first and last depth slices
        H = lr.shape[2]
        W = lr.shape[3]
        lr = torch.cat([torch.zeros(1, 7, H, W), lr, torch.zeros(1, 7, H, W)], dim=1)
        hr = torch.from_numpy(np.ascontiguousarray(hr)).float()
        hr = hr.repeat(1, 64, 1, 1)

        return lr, hr

    def __len__(self) -> int:
        return self.lenth


class DenoisePlanriaDataset3D(Dataset):
    """
    3D denoising planaria dataset.

    Args:
        config: Configuration object with data_dir and condition attributes
        is_train: Whether this is training or test set
    """

    def __init__(self, config, is_train: bool = True):
        self.data_dir = config.data_dir
        self.is_train = is_train
        self.condition = config.condition

        info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        self.original_shape = info['original_test_shape'] if not is_train else None
        self.target_num = info['data_number'] if not is_train else None

        if is_train:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'train', 'cluster_*.npz')))
        else:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'test', 'patch_*.npz')))

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")

        sample_data = np.load(self.file_list[0])
        if is_train:
            source_shape = sample_data['source'].shape
        else:
            source_shape = sample_data[self.condition].shape
        print(f"Original data shape: {source_shape}")

        if not is_train:
            self.patch_coords = []
            for file_path in self.file_list:
                data = np.load(file_path)
                self.patch_coords.append(data['start_point'])

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor],
                                              Tuple[torch.Tensor, torch.Tensor, np.ndarray, int]]:
        """
        Get denoising pair.

        Returns:
            Training: (noisy, clean)
            Testing: (noisy, clean, position, data_index)
        """
        data = np.load(self.file_list[idx])
        if self.is_train:
            source = data['source']
            target = data['target']
        else:
            source = data[self.condition]
            target = data['GT']

        source = source / source.max()

        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()

        if self.is_train:
            source = torch.vstack([source, source])
            target = torch.vstack([target, target])
            return source.unsqueeze(0), target.unsqueeze(0)
        else:
            position = data['start_point']
            data_index = data['data_index'].item()
            source = torch.vstack([source, source])
            target = torch.vstack([target, target])
            source = source.unsqueeze(0)
            target = target.unsqueeze(0)
            return source, target, position, data_index

    def __len__(self) -> int:
        return len(self.file_list)


# ============================================================================
# Segmentation Datasets
# ============================================================================

class MedSAMSegDataset2D(Dataset):
    """
    2D medical segmentation dataset with SAM-style prompts.

    Multi-organ segmentation with 13 organ classes. Supports bounding box prompts
    for Segment Anything Model (SAM) style training.

    Args:
        config: Configuration object with data_dir, bbox_shift, include_labels attributes
        is_train: Whether this is training or test set

    Example:
        >>> config.include_labels = ['Liver', 'Spleen', 'Pancreas']
        >>> dataset = MedSAMSegDataset2D(config, is_train=True)
        >>> img, mask, bbox, label_id, organ_name, filename = dataset[0]
    """

    LABEL_MAPPING = {
        1: "Liver",
        2: "Right Kidney",
        3: "Spleen",
        4: "Pancreas",
        5: "Aorta",
        6: "Inferior Vena Cava (IVC)",
        7: "Right Adrenal Gland (RAG)",
        8: "Left Adrenal Gland (LAG)",
        9: "Gallbladder",
        10: "Esophagus",
        11: "Stomach",
        12: "Duodenum",
        13: "Left Kidney"
    }

    ORGAN_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

    def __init__(self, config, is_train: bool = True):
        if is_train:
            self.data_dir = os.path.join(config.data_dir, "train")
        else:
            self.data_dir = os.path.join(config.data_dir, "val")
        self.gt_path = os.path.join(self.data_dir, "gts")
        self.img_path = os.path.join(self.data_dir, "imgs")
        self.bbox_shift = config.bbox_shift if config.bbox_shift is not None else 20

        self.include_labels = self._process_include_labels(config.include_labels)
        self.samples = self._preprocess_samples()

        self.index_map = []
        for sample_idx, labels in enumerate(self.samples):
            self.index_map.extend([(sample_idx, label) for label in labels["valid_labels"]])

        print(f"Total instances: {len(self)}")
        print(f"Class distribution: {self.get_class_distribution()}")

    def _process_include_labels(self, include_labels) -> set:
        """Process include_labels to set of label IDs."""
        if include_labels is None:
            valid_labels = set(self.LABEL_MAPPING.keys())
        else:
            valid_labels = set()
            for item in include_labels:
                if isinstance(item, str):
                    if item in self.ORGAN_TO_LABEL:
                        valid_labels.add(self.ORGAN_TO_LABEL[item])
                    else:
                        raise ValueError(f"Invalid organ name: {item}")
                elif isinstance(item, int):
                    if item in self.LABEL_MAPPING:
                        valid_labels.add(item)
                    else:
                        raise ValueError(f"Invalid label ID: {item}")
                else:
                    raise TypeError("Labels must be str or int")

        valid_labels.discard(0)
        return valid_labels

    def _preprocess_samples(self) -> List[Dict]:
        """Preprocess and filter samples."""
        samples = []
        gt_files = sorted(glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True))

        for gt_file in gt_files:
            img_name = os.path.basename(gt_file)
            img_file = os.path.join(self.img_path, img_name)

            if not os.path.exists(img_file):
                continue

            gt = np.load(gt_file, 'r', allow_pickle=True)
            present_labels = set(np.unique(gt))

            valid_labels = []
            for label in present_labels:
                if label == 0:
                    continue
                if self.include_labels is None or label in self.include_labels:
                    valid_labels.append(label)

            if valid_labels:
                samples.append({
                    "img_path": img_file,
                    "gt_path": gt_file,
                    "valid_labels": valid_labels
                })

        return samples

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor, str, str]:
        """
        Get segmentation sample with SAM-style bbox prompt.

        Returns:
            Tuple of (image, mask, bbox, label_id, organ_name, filename)
            - image: (1, H, W) tensor
            - mask: (1, H, W) binary tensor
            - bbox: (4,) tensor [x_min, y_min, x_max, y_max]
            - label_id: scalar tensor
            - organ_name: string
            - filename: string
        """
        sample_idx, label_id = self.index_map[idx]
        sample = self.samples[sample_idx]

        img_1024 = np.load(sample["img_path"], 'r', allow_pickle=True)
        img_1024 = img_1024[:, :, 0]

        assert 0.0 <= img_1024.min() and img_1024.max() <= 1.0, "Image values out of [0,1] range"

        gt = np.load(sample["gt_path"], 'r', allow_pickle=True)
        gt2D = np.uint8(gt == label_id)

        if np.sum(gt2D) == 0:
            print(f"Warning: Empty mask for label {label_id} in {sample['gt_path']}")

        # Compute bounding box
        y_indices, x_indices = np.where(gt2D > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            x_min, x_max = 0, gt2D.shape[1]
            y_min, y_max = 0, gt2D.shape[0]
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))

        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).unsqueeze(0).float(),
            torch.tensor(gt2D).unsqueeze(0).float(),
            torch.tensor(bboxes).float(),
            torch.tensor(label_id).long(),
            self.LABEL_MAPPING[label_id],
            os.path.basename(sample["img_path"])
        )

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of organ classes in dataset."""
        label_counts = {}
        for sample in self.samples:
            for label in sample["valid_labels"]:
                organ_name = self.LABEL_MAPPING[label]
                label_counts[organ_name] = label_counts.get(organ_name, 0) + 1
        return label_counts

    @classmethod
    def get_available_labels(cls) -> List[Tuple[int, str]]:
        """Get list of available organ labels."""
        return list(cls.LABEL_MAPPING.items())


class MedicalSAMSegDataset2D(Dataset):
    """
    2D medical segmentation with SAM and point prompts.

    Uses multiple rater annotations with majority voting for ground truth.

    Args:
        config: Configuration object with data_dir and img_size attributes
        is_train: Whether this is training or test set
    """

    def random_click(self, mask: np.ndarray, point_label: int = 1) -> Tuple[int, np.ndarray]:
        """Generate random point prompt from mask."""
        max_label = max(set(mask.flatten()))
        if round(max_label) == 0:
            point_label = round(max_label)
        indices = np.argwhere(mask == max_label)
        return point_label, indices[np.random.randint(len(indices))]

    def __init__(self, config, is_train: bool):
        self.data_path = config.data_dir
        self.img_size = config.img_size
        self.prompt = 'click'
        self.mask_size = self.img_size[0]
        if is_train:
            self.mode = 'Training'
        else:
            self.mode = 'Test'

        self.subfolders = [f.path for f in os.scandir(os.path.join(self.data_path, self.mode + '-400')) if f.is_dir()]
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.subfolders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get segmentation sample with point prompt.

        Returns:
            Tuple of (image, mask, point, point_label)
        """
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        img_path = os.path.join(subfolder, name + '_cropped.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '_cropped.jpg') for i in range(1, 8)]

        img = Image.open(img_path).convert('L')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >= 0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)
            torch.set_rng_state(state)

        point_label_cup, pt_cup = self.random_click(np.array((multi_rater_cup.mean(axis=0)).squeeze(0)), point_label=1)

        selected_rater_mask_cup_ori = multi_rater_cup.mean(axis=0)
        selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float()

        selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)
        selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()

        return img, selected_rater_mask_cup, torch.tensor(pt_cup), torch.tensor(point_label_cup)


class MicroSplitDataset2D(Dataset):
    """
    2D microscopy image splitting dataset.

    Splits single microscopy image into multiple outputs.

    Args:
        config: Configuration object with data_dir and img_size attributes
        is_train: Whether this is training or test set
    """

    def __init__(self, config, is_train: bool = True):
        self.data_dir = Path(config.data_dir)
        self.is_train = is_train
        self.img_size = config.img_size

        if is_train:
            self.file_list = sorted(glob.glob(str(self.data_dir / "**" / 'train' / '*.npz'), recursive=True))
        else:
            self.file_list = sorted(glob.glob(str(self.data_dir / "**" / 'test' / '*.npz'), recursive=True))

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image splitting sample.

        Returns:
            Tuple of (source, target) where target has 2 channels
        """
        data = np.load(self.file_list[idx])
        source = torch.from_numpy(data['source']).float().unsqueeze(0)
        target1 = torch.from_numpy(data['target1']).float().unsqueeze(0)
        target2 = torch.from_numpy(data['target2']).float().unsqueeze(0)

        # Resize to target size
        source = F.interpolate(source.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        target1 = F.interpolate(target1.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        target2 = F.interpolate(target2.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        target = torch.cat((target1, target2), dim=0)
        return source, target


# ============================================================================
# DataLoader Factory Functions
# ============================================================================

def get_dataloader(config, is_train: bool = True) -> DataLoader:
    """
    Get dataloader for pretraining datasets.

    Args:
        config: Configuration object with necessary attributes
        is_train: Whether to create training or validation dataloader

    Returns:
        DataLoader instance

    Example:
        >>> dataloader = get_dataloader(config, is_train=True)
    """
    # Determine if 2D or 3D based on img_size length
    if len(config.img_size) == 2:
        dataset = PretrainDataset2D(config.data_dir, img_size=config.img_size)
    else:
        dataset = PretrainDataset3D(config.data_dir, img_size=config.img_size)

    sampler = DistributedSampler(dataset) if is_train else None
    collate = CollateFn(config.img_size)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=True,
    )


# ============================================================================
# Dataset Factory Function
# ============================================================================

def get_dataset(name: str, **kwargs) -> Dataset:
    """
    Factory function to get dataset by name.

    Args:
        name: Dataset name (case-insensitive)
        **kwargs: Dataset-specific arguments

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset name is not recognized

    Available datasets:
        2D Datasets:
        - 'pretrain_2d': PretrainDataset2D
        - 'isoliver_2d': ISOLiverDataset2D
        - 'bsa_fusion_2d': BSAFusionDataset2D
        - 'unifmir_sr_2d': UnifmirSRDataset2D
        - 'medsam_seg_2d': MedSAMSegDataset2D
        - 'medical_sam_seg_2d': MedicalSAMSegDataset2D
        - 'micro_split_2d': MicroSplitDataset2D
        - 'proj_flywing_2d': ProjFlywingDataset2D

        3D Datasets:
        - 'pretrain_3d': PretrainDataset3D
        - 'ixi_brain_3d': IXIBrainDataset3D
        - 'ixi_brain_infer_3d': IXIBrainInferDataset3D
        - 'oasis_brain_3d': OASISBrainDataset3D
        - 'oasis_brain_infer_3d': OASISBrainInferDataset3D
        - 'isoliver_3d': ISOLiverDataset3D
        - 'fat_fusion_3d': FATFusionDataset3D
        - 'inverse_sr_3d': InverseSRDataset3D
        - 'proj_flywing_3d': ProjFlywingDataset3D
        - 'denoise_planria_3d': DenoisePlanriaDataset3D

    Example:
        >>> # Get 2D dataset
        >>> dataset = get_dataset('medsam_seg_2d', config=config, is_train=True)
        >>>
        >>> # Get 3D dataset
        >>> dataset = get_dataset('ixi_brain_3d', data_path=paths,
        ...                       atlas_path=atlas, transforms=transforms)
        >>>
        >>> # Get super-resolution dataset
        >>> dataset = get_dataset('unifmir_sr_2d', config=config, is_train=True)
    """
    name = name.lower()

    # 2D Datasets
    if name == 'pretrain_2d':
        return PretrainDataset2D(**kwargs)
    elif name == 'isoliver_2d':
        return ISOLiverDataset2D(**kwargs)
    elif name == 'bsa_fusion_2d':
        return BSAFusionDataset2D(**kwargs)
    elif name == 'unifmir_sr_2d':
        return UnifmirSRDataset2D(**kwargs)
    elif name == 'medsam_seg_2d':
        return MedSAMSegDataset2D(**kwargs)
    elif name == 'medical_sam_seg_2d':
        return MedicalSAMSegDataset2D(**kwargs)
    elif name == 'micro_split_2d':
        return MicroSplitDataset2D(**kwargs)
    elif name == 'proj_flywing_2d':
        return ProjFlywingDataset2D(**kwargs)

    # 3D Datasets
    elif name == 'pretrain_3d':
        return PretrainDataset3D(**kwargs)
    elif name == 'ixi_brain_3d':
        return IXIBrainDataset3D(**kwargs)
    elif name == 'ixi_brain_infer_3d':
        return IXIBrainInferDataset3D(**kwargs)
    elif name == 'oasis_brain_3d':
        return OASISBrainDataset3D(**kwargs)
    elif name == 'oasis_brain_infer_3d':
        return OASISBrainInferDataset3D(**kwargs)
    elif name == 'isoliver_3d':
        return ISOLiverDataset3D(**kwargs)
    elif name == 'fat_fusion_3d':
        return FATFusionDataset3D(**kwargs)
    elif name == 'inverse_sr_3d':
        return InverseSRDataset3D(**kwargs)
    elif name == 'proj_flywing_3d':
        return ProjFlywingDataset3D(**kwargs)
    elif name == 'denoise_planria_3d':
        return DenoisePlanriaDataset3D(**kwargs)

    else:
        available = [
            '2D: pretrain_2d, isoliver_2d, bsa_fusion_2d, unifmir_sr_2d, '
            'medsam_seg_2d, medical_sam_seg_2d, micro_split_2d, proj_flywing_2d',
            '3D: pretrain_3d, ixi_brain_3d, ixi_brain_infer_3d, oasis_brain_3d, '
            'oasis_brain_infer_3d, isoliver_3d, fat_fusion_3d, inverse_sr_3d, '
            'proj_flywing_3d, denoise_planria_3d'
        ]
        raise ValueError(f"Unknown dataset: {name}\nAvailable datasets:\n" + '\n'.join(available))


# Export all dataset classes and factory function
__all__ = [
    # Factory functions
    'get_dataset',
    'get_dataloader',

    # 2D Datasets
    'PretrainDataset2D',
    'ISOLiverDataset2D',
    'BSAFusionDataset2D',
    'UnifmirSRDataset2D',
    'MedSAMSegDataset2D',
    'MedicalSAMSegDataset2D',
    'MicroSplitDataset2D',
    'ProjFlywingDataset2D',

    # 3D Datasets
    'PretrainDataset3D',
    'IXIBrainDataset3D',
    'IXIBrainInferDataset3D',
    'OASISBrainDataset3D',
    'OASISBrainInferDataset3D',
    'ISOLiverDataset3D',
    'FATFusionDataset3D',
    'InverseSRDataset3D',
    'ProjFlywingDataset3D',
    'DenoisePlanriaDataset3D',

    # Utilities
    'pkload',
    'random_crop',
    'CollateFn',
]
