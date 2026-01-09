"""
Orochi: Versatile Biomedical Image Processor

A foundation model for biomedical image processing using Mamba architecture.
Supports 2D and 3D tasks including:
- Image fusion
- Super-resolution
- Image restoration
- Medical image registration
"""

__version__ = "0.1.0"
__author__ = "Gaole Dai, Chenghao Zhou, Yu Zhou, et al."

from orochi.models import MambaEncoderHeria
from orochi.data import get_dataset
from orochi.losses import get_loss_function
from orochi.metrics import get_metric

# Alias for convenience
MambaModel = MambaEncoderHeria

__all__ = [
    "__version__",
    "MambaModel",
    "MambaEncoderHeria",
    "get_dataset",
    "get_loss_function",
    "get_metric",
]
