"""Model implementations for Orochi.

This module contains:
- Mamba2D: 2D Mamba-based model for image processing
- Mamba3D: 3D Mamba-based model for volumetric processing
- Model components (PatchEmbed, blocks, spatial transformers)
- Model registry for easy model selection

Example:
    >>> from orochi.models import Mamba2D
    >>> from orochi.configs import Mamba2DConfig
    >>>
    >>> config = Mamba2DConfig()
    >>> model = Mamba2D(config)
"""

# Will be populated as we migrate code
__all__ = []
