# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-09

### Added

#### Package Structure
- Created proper Python package structure with `pyproject.toml`
- Organized code into logical modules: models, losses, data, metrics, training, utils, cli
- Added proper `__init__.py` files with explicit exports
- Configured setuptools for pip installation support

#### Models (`orochi/models/`)
- **Consolidated** 3 duplicate `ours_mamba.py` files (~3500 lines) into single implementation (1346 lines)
- Created unified `MambaEncoderHeria` supporting both 2D and 3D via `dimensions` parameter
- Added comprehensive docstrings with mathematical formulations
- Implemented full type hints throughout
- Included all decoder variants:
  - `reg_decoder`: Registration
  - `fus_decoder`: Image fusion
  - `SR_decoder`: Super-resolution
  - `IR_decoder`: Image restoration
  - `seg_decoder`: Segmentation
  - And more specialized decoders
- Maintained backward compatibility for weight loading
- Created `mamba_demo.py` with usage examples

#### Losses (`orochi/losses/`)
- **Consolidated** 3 duplicate loss files into comprehensive module (2500 lines)
- Added 23 loss function classes:
  - 5 SSIM-based losses
  - 4 gradient regularization losses
  - 4 information theory losses
  - 1 segmentation loss (Dice)
  - 5 pixel-wise and perceptual losses
  - 4 fusion-specific losses
- Implemented factory function `get_loss_function(name, **kwargs)`
- Added mathematical formulas in docstrings
- Included usage examples and references to papers
- Full type hints and input validation

#### Data (`orochi/data/`)
- Consolidated dataset classes with comprehensive documentation
- Created `PretrainDataset` for large-scale 2D biomedical images
- Implemented `IXIBrainDataset` and `IXIBrainInferDataset` for 3D registration
- Added factory function `get_dataset(name, **kwargs)`
- Included data utilities:
  - `pkload()`: Pickle file loading
  - `init_fn()`: Worker initialization for reproducibility
  - `get_all_coords()`: 3D coordinate grid generation
  - `gen_feats()`: Normalized position features
  - `normalize_intensity()`: Percentile-based normalization
  - `pad_to_size()`, `crop_center()`: Image manipulation
- Added `CollateFn` for batch collation with random cropping
- Full type hints and extensive inline documentation

#### Utilities (`orochi/utils/`)
- Consolidated helper functions from multiple sources
- Visualization utilities
- Metrics computation (PSNR, SSIM, Dice, Jacobian)
- Training utilities (logging, checkpointing, LR scheduling)
- Image transformations
- Distributed training support

#### Testing (`tests/`)
- Created comprehensive test suite with pytest
- Unit tests for models (`test_models.py`)
- Unit tests for losses (`test_losses.py`)
- Unit tests for data (`test_data.py`)
- Configured pytest with coverage reporting
- Created shared fixtures in `conftest.py`
- Tests cover:
  - Model instantiation and forward passes
  - Loss function correctness
  - Data loading and preprocessing
  - Numerical stability
  - Gradient flow
  - CUDA compatibility
  - Edge cases

#### Documentation
- Created `MAMBA_CONSOLIDATION_ANALYSIS.md` (16KB): Technical analysis of model consolidation
- Created `QUICK_START_GUIDE.md` (8.5KB): Practical examples for 2D and 3D usage
- Created `LOSS_CONSOLIDATION_SUMMARY.md`: Overview of all loss functions
- Created `LOSS_QUICK_REFERENCE.md`: Quick lookup guide for losses
- Created `CONTRIBUTING.md`: Comprehensive contribution guidelines
- Created `CHANGELOG.md`: This file

#### Code Quality
- Configured Black (line length: 100)
- Configured isort (Black-compatible profile)
- Configured flake8 with custom rules
- Configured mypy for type checking
- Created `.pre-commit-config.yaml` with automated checks
- Created `.flake8` configuration
- Created `.gitattributes` for consistent line endings

#### CI/CD
- Created `.github/workflows/test.yml`: Automated testing on Python 3.10, 3.11, 3.12
- Created `.github/workflows/lint.yml`: Automated linting and formatting checks
- Configured Codecov integration for coverage reporting
- Added caching for faster CI runs

#### Configuration
- Created `pyproject.toml` with:
  - Project metadata
  - Dependencies
  - Optional dev dependencies
  - Tool configurations (black, isort, mypy, pytest)
  - Entry points for CLI (future)

### Changed
- **Reduced code duplication by ~60%** (from ~3500 to ~1350 lines for models alone)
- Improved code organization and maintainability
- Enhanced documentation quality throughout
- Standardized naming conventions
- Unified 2D and 3D implementations with parameterization

### Fixed
- Cleaned up duplicate imports across files
- Fixed inconsistent module structures
- Resolved missing type hints
- Corrected docstring formatting

### Maintained
- **100% backward compatibility** for trained model weights
- Exact state_dict key names preserved
- Original API signatures maintained where applicable
- Support for all original tasks (fusion, SR, IR, registration, segmentation)

## Migration Guide

### Import Path Changes

**Models:**
```python
# Old
from src.ours_mamba import MambaEncoderHeria
from temp.experiments.2D.ours_mamba import MambaEncoderHeria

# New
from orochi.models import MambaEncoderHeria
config.dimensions = 2  # or 3 for 3D
```

**Losses:**
```python
# Old
from src.losses import SSIM, Grad3d
from temp.experiments.2D.losses import SSIM2D

# New
from orochi.losses import SSIM, SSIM2D, Grad3d
# Or use factory
loss_fn = get_loss_function('ssim2d', window_size=11)
```

**Datasets:**
```python
# Old
from src.data.datasets import IXIBrainDataset
from temp.experiments.2D.datasets import Pretrain_Dataset

# New
from orochi.data import IXIBrainDataset, PretrainDataset
# Or use factory
dataset = get_dataset('pretrain', root_dir=path, img_size=(256, 256))
```

### Configuration Changes

Add `dimensions` parameter to model configs:
```python
config.dimensions = 2  # for 2D tasks
config.dimensions = 3  # for 3D tasks
```

### Weight Loading

No changes needed! Existing weights load directly:
```python
model = MambaEncoderHeria(config)
model.load_state_dict(torch.load('checkpoint.pth'))
```

## [Unreleased]

### Planned
- Complete transforms module consolidation
- Add configuration file support (YAML/JSON)
- Implement unified training scripts
- Add inference CLI
- Create Jupyter notebook examples
- Add model zoo with pretrained weights
- Improve documentation with architecture diagrams
- Add benchmarking suite

---

## Version History

- **0.1.0** (2026-01-09): Major restructuring and consolidation
- **0.0.1** (2025): Initial research code release

[0.1.0]: https://github.com/jnjnnjzch/foundation-mamba-biomed/releases/tag/v0.1.0
