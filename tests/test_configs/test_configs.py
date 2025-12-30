"""Test configuration classes."""

import pytest
from pathlib import Path
import tempfile
import os

from orochi.configs import (
    BaseConfig,
    Mamba2DConfig,
    Mamba3DConfig,
    SuperResolutionConfig,
    load_config_from_env,
)


class TestBaseConfig:
    """Test BaseConfig functionality."""

    def test_create_config(self):
        """Test creating a basic config."""
        config = BaseConfig()
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4
        assert isinstance(config.data_root, Path)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = BaseConfig(batch_size=8)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["batch_size"] == 8
        assert isinstance(config_dict["data_root"], str)  # Path converted to string

    def test_config_yaml_save_load(self):
        """Test saving and loading config from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaseConfig(
                batch_size=8,
                learning_rate=1e-3,
                num_epochs=50,
            )

            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(str(yaml_path))

            assert yaml_path.exists()

            loaded = BaseConfig.from_yaml(str(yaml_path))
            assert loaded.batch_size == 8
            assert loaded.learning_rate == 1e-3
            assert loaded.num_epochs == 50

    def test_create_directories(self):
        """Test automatic directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaseConfig(
                data_root=Path(tmpdir) / "data",
                checkpoint_dir=Path(tmpdir) / "checkpoints",
            )
            config.create_directories()

            assert config.data_root.exists()
            assert config.checkpoint_dir.exists()

    def test_checkpoint_paths(self):
        """Test checkpoint path generation."""
        config = BaseConfig(experiment_name="test_exp")

        epoch_path = config.get_checkpoint_path(50)
        assert "test_exp" in str(epoch_path)
        assert "epoch_50" in str(epoch_path)

        best_path = config.get_best_checkpoint_path()
        assert "best" in str(best_path)


class TestMamba2DConfig:
    """Test Mamba2D configuration."""

    def test_create_mamba2d_config(self):
        """Test creating Mamba2D config."""
        config = Mamba2DConfig(
            img_size=[512, 512],
            patch_size=16,
            embed_dim=192,
        )

        assert config.img_size == [512, 512]
        assert config.patch_size == 16
        assert config.embed_dim == 192

    def test_invalid_img_size(self):
        """Test validation of image size."""
        with pytest.raises(ValueError, match="img_size must be"):
            Mamba2DConfig(img_size=[256])  # Should be [H, W]

        with pytest.raises(ValueError, match="img_size must be"):
            Mamba2DConfig(img_size=[256, 256, 256])  # Should be 2D

    def test_task_validation(self):
        """Test task validation."""
        valid_config = Mamba2DConfig(task="super_resolution")
        assert valid_config.task == "super_resolution"

        with pytest.raises(ValueError, match="task must be one of"):
            Mamba2DConfig(task="invalid_task")

    def test_segmentation_requires_num_classes(self):
        """Test that segmentation task requires num_classes."""
        with pytest.raises(ValueError, match="num_classes must be specified"):
            Mamba2DConfig(task="segmentation")

        # This should work
        valid_config = Mamba2DConfig(task="segmentation", num_classes=5)
        assert valid_config.num_classes == 5


class TestMamba3DConfig:
    """Test Mamba3D configuration."""

    def test_create_mamba3d_config(self):
        """Test creating Mamba3D config."""
        config = Mamba3DConfig(
            img_size=[64, 256, 256],
            patch_size=16,
            embed_dim=192,
        )

        assert config.img_size == [64, 256, 256]
        assert len(config.img_size) == 3

    def test_invalid_img_size(self):
        """Test validation of 3D image size."""
        with pytest.raises(ValueError, match="img_size must be"):
            Mamba3DConfig(img_size=[256, 256])  # Should be 3D

    def test_registration_config(self):
        """Test registration-specific config."""
        config = Mamba3DConfig(
            task="registration",
            use_deformation=True,
            int_steps=7,
        )

        assert config.use_deformation is True
        assert config.int_steps == 7


class TestConvenienceConfigs:
    """Test convenience configuration classes."""

    def test_super_resolution_config(self):
        """Test SuperResolutionConfig."""
        config = SuperResolutionConfig(scale_factor=2)

        assert config.task == "super_resolution"
        assert config.scale_factor == 2
        assert config.out_chans == config.in_chans

    def test_segmentation_config(self):
        """Test Segmentation2DConfig."""
        config = pytest.importorskip("orochi.configs").Segmentation2DConfig(
            num_classes=5
        )

        assert config.task == "segmentation"
        assert config.num_classes == 5


class TestEnvironmentConfig:
    """Test loading config from environment variables."""

    def test_load_from_env(self):
        """Test loading config from environment variables."""
        # Set environment variables
        os.environ["OROCHI_BATCH_SIZE"] = "16"
        os.environ["OROCHI_LEARNING_RATE"] = "0.001"
        os.environ["OROCHI_NUM_EPOCHS"] = "200"

        try:
            config = load_config_from_env(BaseConfig)

            assert config.batch_size == 16
            assert config.learning_rate == 0.001
            assert config.num_epochs == 200
        finally:
            # Cleanup
            del os.environ["OROCHI_BATCH_SIZE"]
            del os.environ["OROCHI_LEARNING_RATE"]
            del os.environ["OROCHI_NUM_EPOCHS"]

    def test_env_boolean(self):
        """Test boolean environment variables."""
        os.environ["OROCHI_DETERMINISTIC"] = "true"

        try:
            config = load_config_from_env(BaseConfig)
            assert config.deterministic is True
        finally:
            del os.environ["OROCHI_DETERMINISTIC"]
