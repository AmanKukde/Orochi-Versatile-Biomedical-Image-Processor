"""Orochi: Versatile Biomedical Image Processor.

A PyTorch-based framework for biomedical image processing tasks including:
- Super-resolution (2D and 3D)
- Isotropic restoration
- Image fusion
- Image registration
- Segmentation

Based on the Mamba state-space model architecture.

Example:
    >>> from orochi.models import Mamba2D
    >>> from orochi.losses import SSIM, DiceLoss
    >>> from orochi.configs import Mamba2DConfig
    >>>
    >>> config = Mamba2DConfig(img_size=[256, 256])
    >>> model = Mamba2D(config)
    >>> loss_fn = SSIM()

Citation:
    @article{dai2025orochi,
      title={Orochi: Versatile Biomedical Image Processor},
      author={Dai, Gaole and Zhou, Chenghao and Zhou, Yu and Zhang, Rongyu and
              Zhang, Yuan and Hou, Chengkai and Huang, Tiejun and Chen, Jianxu and
              Zhang, Shanghang},
      journal={arXiv preprint arXiv:2509.22583},
      year={2025}
    }
"""

__version__ = "1.0.0"
__author__ = "Orochi Team"
__license__ = "MIT"

# Import key components for easy access
from orochi import configs, data, losses, models, training, utils

__all__ = [
    "models",
    "losses",
    "data",
    "training",
    "configs",
    "utils",
    "__version__",
]
