"""Base configuration classes for Orochi.

Provides dataclass-based configuration management with YAML support.
All configurations inherit from BaseConfig for consistent interface.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Any, Dict
import yaml
import os


@dataclass
class BaseConfig:
    """Base configuration class for all Orochi experiments.

    Provides common settings for paths, devices, training, and logging.
    All task-specific configs should inherit from this class.

    Attributes:
        # Paths
        data_root: Root directory for datasets
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
        output_dir: Directory for outputs (predictions, visualizations)

        # Device
        device: Device to use ('cuda' or 'cpu')
        gpu_ids: List of GPU IDs to use for multi-GPU training
        num_workers: Number of data loading workers

        # Training
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        grad_clip: Gradient clipping value (None to disable)

        # Optimizer
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        momentum: Momentum for SGD (ignored for Adam variants)
        betas: Beta parameters for Adam variants

        # Scheduler
        scheduler: Learning rate scheduler ('step', 'cosine', 'plateau', None)
        lr_decay_epochs: Epochs to decay LR (for step scheduler)
        lr_decay_rate: LR decay rate
        min_lr: Minimum learning rate

        # Logging
        log_interval: Log every N iterations
        save_interval: Save checkpoint every N epochs
        val_interval: Validate every N epochs
        wandb_project: Weights & Biases project name (None to disable)
        experiment_name: Name for this experiment

        # Reproducibility
        seed: Random seed for reproducibility
        deterministic: Use deterministic algorithms (slower but reproducible)

    Example:
        >>> config = BaseConfig(
        ...     data_root=Path("./data"),
        ...     batch_size=4,
        ...     learning_rate=1e-4
        ... )
        >>> config.to_yaml("config.yaml")
        >>> loaded = BaseConfig.from_yaml("config.yaml")
    """

    # Paths
    data_root: Path = field(default_factory=lambda: Path("./data"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))

    # Device
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4

    # Training
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 1.0

    # Optimizer
    optimizer: str = "adam"
    momentum: float = 0.9
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))

    # Scheduler
    scheduler: Optional[str] = "step"
    lr_decay_epochs: List[int] = field(default_factory=lambda: [50, 75])
    lr_decay_rate: float = 0.1
    min_lr: float = 1e-7

    # Logging
    log_interval: int = 10
    save_interval: int = 5
    val_interval: int = 1
    wandb_project: Optional[str] = None
    experiment_name: str = "orochi_experiment"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    subset: Optional[float] = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.data_root = Path(self.data_root)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config

        Example:
            >>> config = BaseConfig()
            >>> config_dict = config.to_dict()
        """
        config_dict = asdict(self)
        # Convert Path objects to strings for serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        return config_dict

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file

        Example:
            >>> config = BaseConfig()
            >>> config.to_yaml("experiment_config.yaml")
        """
        config_dict = self.to_dict()
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            Config instance

        Example:
            >>> config_dict = {'batch_size': 8, 'learning_rate': 1e-3}
            >>> config = BaseConfig.from_dict(config_dict)
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist

        Example:
            >>> config = BaseConfig.from_yaml("experiment_config.yaml")
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def create_directories(self) -> None:
        """Create all required directories.

        Example:
            >>> config = BaseConfig()
            >>> config.create_directories()
        """
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, epoch: int) -> Path:
        """Get path for saving checkpoint.

        Args:
            epoch: Epoch number

        Returns:
            Path to checkpoint file

        Example:
            >>> config = BaseConfig(experiment_name="my_exp")
            >>> path = config.get_checkpoint_path(50)
            >>> print(path)
            checkpoints/my_exp_epoch_50.pth
        """
        return self.checkpoint_dir / f"{self.experiment_name}_epoch_{epoch}.pth"

    def get_best_checkpoint_path(self) -> Path:
        """Get path for best checkpoint.

        Returns:
            Path to best checkpoint file

        Example:
            >>> config = BaseConfig(experiment_name="my_exp")
            >>> path = config.get_best_checkpoint_path()
            >>> print(path)
            checkpoints/my_exp_best.pth
        """
        return self.checkpoint_dir / f"{self.experiment_name}_best.pth"

    def __repr__(self) -> str:
        """Pretty string representation."""
        lines = [f"{self.__class__.__name__}:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def load_config_from_env(config_class=BaseConfig) -> BaseConfig:
    """Load configuration from environment variables.

    Environment variables should be prefixed with OROCHI_ and uppercase.
    Example: OROCHI_BATCH_SIZE=8, OROCHI_LEARNING_RATE=0.001

    Args:
        config_class: Config class to instantiate

    Returns:
        Config instance with values from environment

    Example:
        >>> # After setting: export OROCHI_BATCH_SIZE=8
        >>> config = load_config_from_env()
        >>> print(config.batch_size)
        8
    """
    config_dict = {}

    # Get all environment variables with OROCHI_ prefix
    for key, value in os.environ.items():
        if key.startswith("OROCHI_"):
            # Remove prefix and convert to lowercase
            config_key = key[7:].lower()

            # Try to convert to appropriate type
            try:
                # Try int
                config_dict[config_key] = int(value)
            except ValueError:
                try:
                    # Try float
                    config_dict[config_key] = float(value)
                except ValueError:
                    # Keep as string
                    if value.lower() == 'true':
                        config_dict[config_key] = True
                    elif value.lower() == 'false':
                        config_dict[config_key] = False
                    else:
                        config_dict[config_key] = value

    return config_class.from_dict(config_dict)
