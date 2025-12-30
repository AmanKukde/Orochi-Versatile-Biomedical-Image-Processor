"""Configuration management for Orochi.

Provides dataclass-based configurations for:
- Base configuration (paths, devices, training params)
- Model configurations (Mamba2D, Mamba3D)
- Task-specific configurations
- YAML loading/saving
- Environment variable support

Example:
    >>> from orochi.configs import Mamba2DConfig, SuperResolutionConfig
    >>>
    >>> # Create config
    >>> config = Mamba2DConfig(
    ...     img_size=[256, 256],
    ...     batch_size=4,
    ...     learning_rate=1e-4
    ... )
    >>>
    >>> # Save to YAML
    >>> config.to_yaml("config.yaml")
    >>>
    >>> # Load from YAML
    >>> config = Mamba2DConfig.from_yaml("config.yaml")
    >>>
    >>> # Use convenience configs
    >>> sr_config = SuperResolutionConfig(scale_factor=2)
"""

from orochi.configs.base_config import BaseConfig, load_config_from_env
from orochi.configs.model_configs import (
    Mamba2DConfig,
    Mamba3DConfig,
    SuperResolutionConfig,
    IsotropicRestorationConfig,
    ImageFusionConfig,
    Segmentation2DConfig,
    Registration3DConfig,
    SuperResolution3DConfig,
)

__all__ = [
    # Base
    "BaseConfig",
    "load_config_from_env",
    # Model configs
    "Mamba2DConfig",
    "Mamba3DConfig",
    # Task-specific configs
    "SuperResolutionConfig",
    "IsotropicRestorationConfig",
    "ImageFusionConfig",
    "Segmentation2DConfig",
    "Registration3DConfig",
    "SuperResolution3DConfig",
]
