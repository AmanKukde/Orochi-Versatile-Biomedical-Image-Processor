"""Configuration management for Orochi.

Provides dataclass-based configurations for:
- Base configuration (paths, devices, training params)
- Model configurations (Mamba2D, Mamba3D)
- Task-specific configurations
- YAML loading/saving

Example:
    >>> from orochi.configs import Mamba2DConfig
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
"""

__all__ = []
