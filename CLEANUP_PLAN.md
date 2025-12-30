# Orochi Repository Cleanup Plan

## Executive Summary

This plan addresses the technical debt identified in the Orochi repository analysis:
- **40-50% code duplication** across 2D/3D implementations
- **0% test coverage** - no automated tests
- **5% documentation** - most functions lack docstrings
- **Hard-coded configurations** - paths, hyperparameters embedded in code
- **Import and style issues** - inconsistent organization

**Goal**: Bring repository to research AI industry standards while maintaining all functionality.

---

## Phase 1: Code Organization & Project Structure

### Objectives
- Create clear, logical directory structure
- Separate concerns (models, training, data, configs)
- Maintain backward compatibility

### Actions

#### 1.1 Create New Directory Structure
```
orochi/                              # Main package
├── __init__.py
├── models/                          # Model implementations
│   ├── __init__.py
│   ├── base.py                      # Base model class
│   ├── mamba_2d.py                  # 2D Mamba model
│   ├── mamba_3d.py                  # 3D Mamba model
│   ├── components/                  # Shared components
│   │   ├── __init__.py
│   │   ├── patch_embed.py          # PatchEmbed2D/3D
│   │   ├── blocks.py               # SS2D, VSS blocks
│   │   └── spatial_transform.py    # SpatialTransformer
│   └── registry.py                  # Model registry
├── data/                            # Data loading
│   ├── __init__.py
│   ├── datasets.py                  # Unified dataset classes
│   ├── transforms.py                # Data augmentation
│   └── loaders.py                   # DataLoader utilities
├── losses/                          # Loss functions
│   ├── __init__.py
│   ├── base.py                      # Base loss class
│   ├── reconstruction.py            # SSIM, PSNR, MSE
│   ├── registration.py              # NCC, MI, gradient
│   ├── segmentation.py              # Dice, surface distance
│   └── fusion.py                    # Fusion-specific losses
├── training/                        # Training logic
│   ├── __init__.py
│   ├── trainer.py                   # Base trainer class
│   ├── callbacks.py                 # Training callbacks
│   └── metrics.py                   # Evaluation metrics
├── utils/                           # Utilities
│   ├── __init__.py
│   ├── visualization.py             # Visualization utilities
│   ├── io.py                        # File I/O
│   └── helpers.py                   # Helper functions
└── configs/                         # Configuration
    ├── __init__.py
    ├── base_config.py               # Base configuration
    ├── model_configs.py             # Model configurations
    └── task_configs.py              # Task-specific configs

experiments/                         # Experiment scripts
├── 2d/
│   ├── super_resolution/
│   ├── isotropic_restoration/
│   └── fusion/
└── 3d/
    ├── super_resolution/
    └── registration/

tests/                               # Unit tests
├── test_models/
├── test_losses/
├── test_data/
└── test_training/

docs/                                # Documentation
└── api/                            # API documentation

legacy/                              # Original code (reference)
└── [moved from src/ and temp/]
```

#### 1.2 Migration Strategy
- Keep original files in `legacy/` directory
- Create new structure incrementally
- Maintain import compatibility with aliases
- Add deprecation warnings for old imports

---

## Phase 2: Eliminate Code Duplication

### 2.1 Unify Loss Functions

**Problem**: 3 nearly identical loss.py files (884, 730, 884 lines)

**Solution**: Create unified loss module with dimension-agnostic implementations

```python
# orochi/losses/reconstruction.py
class SSIM(nn.Module):
    """Structural Similarity Index Measure.

    Supports both 2D and 3D inputs automatically.

    Args:
        window_size: Size of the Gaussian window (default: 11)
        size_average: If True, returns averaged SSIM (default: True)
        val_range: Value range of input (default: None, inferred from data)

    Returns:
        SSIM loss (1 - SSIM for minimization)
    """
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

    def forward(self, img1, img2):
        # Auto-detect 2D vs 3D and use appropriate implementation
        ndim = img1.ndim - 2  # Subtract batch and channel dims
        if ndim == 2:
            return self._ssim_2d(img1, img2)
        elif ndim == 3:
            return self._ssim_3d(img1, img2)
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {img1.ndim}D")
```

**Unified Losses to Create**:
- `SSIM` (replaces SSIM2D, SSIM3D)
- `GradientLoss` (replaces Grad, Grad2d, Grad3d)
- `DiceLoss` (single implementation with shape inference)
- `NCC` (unify NCC_vxm variants)
- `MutualInformation` (consolidate MI implementations)

### 2.2 Unify Model Components

**Problem**: Duplicate PatchEmbed, SS2D blocks across files

**Solution**: Abstract common patterns

```python
# orochi/models/components/patch_embed.py
class PatchEmbed(nn.Module):
    """Unified patch embedding for 2D/3D.

    Automatically handles 2D (H, W) or 3D (D, H, W) inputs.
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        # Dimension-agnostic implementation
        pass
```

### 2.3 Unify Utilities

**Problem**: 2 utils.py files with 80%+ duplication

**Solution**: Single utils module with clear organization

```python
# orochi/utils/visualization.py
def visualize_logits(logits: torch.Tensor, ...):
    """Visualize model predictions (works for 2D/3D)."""
    pass

# orochi/utils/helpers.py
class AverageMeter:
    """Track running averages."""
    pass

def check_nan(tensor: torch.Tensor, name: str = "") -> bool:
    """Check for NaN values in tensor."""
    pass
```

---

## Phase 3: Configuration Management

### 3.1 Create Configuration System

**Problem**: Hard-coded paths, hyperparameters scattered everywhere

**Solution**: Centralized, hierarchical configuration

```python
# orochi/configs/base_config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class BaseConfig:
    """Base configuration for all experiments."""

    # Paths
    data_root: Path = Path("./data")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    output_dir: Path = Path("./outputs")

    # Device
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    # Training
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Logging
    log_interval: int = 10
    save_interval: int = 5
    wandb_project: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file."""
        pass

    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        pass

# orochi/configs/model_configs.py
@dataclass
class Mamba2DConfig(BaseConfig):
    """Configuration for 2D Mamba model."""
    img_size: List[int] = field(default_factory=lambda: [256, 256])
    patch_size: int = 16
    in_chans: int = 1
    out_chans: int = 1
    embed_dim: int = 128
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])

@dataclass
class Mamba3DConfig(BaseConfig):
    """Configuration for 3D Mamba model."""
    img_size: List[int] = field(default_factory=lambda: [32, 256, 256])
    patch_size: int = 16
    # ...
```

### 3.2 Environment Configuration

Create `.env` file support and remove hard-coded paths:

```python
# orochi/utils/io.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

def get_data_path(dataset_name: str) -> Path:
    """Get dataset path from environment or config.

    Falls back to default if not configured.
    """
    env_var = f"OROCHI_{dataset_name.upper()}_PATH"
    path = os.getenv(env_var)
    if path:
        return Path(path)
    return Path("./data") / dataset_name
```

### 3.3 Remove Magic Numbers

Extract all magic numbers to named constants:

```python
# Before:
C1 = 0.01 ** 2
C2 = 0.03 ** 2
window_size = 11

# After (in config or constants.py):
SSIM_C1 = 0.01 ** 2  # Stability constant for luminance
SSIM_C2 = 0.03 ** 2  # Stability constant for contrast
SSIM_WINDOW_SIZE = 11  # Gaussian window size (standard)
```

---

## Phase 4: Documentation

### 4.1 Add Comprehensive Docstrings

**Target**: 100% public API documentation

**Style**: Google-style docstrings with type hints

```python
def patch_embed_forward(
    self,
    x: torch.Tensor,
    return_patches: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Forward pass for patch embedding.

    Args:
        x: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
        return_patches: If True, also return patch positions

    Returns:
        If return_patches is False:
            Embedded patches of shape (B, N, embed_dim)
        If return_patches is True:
            Tuple of (embedded_patches, patch_positions)

    Raises:
        ValueError: If input dimensions are invalid

    Example:
        >>> embedder = PatchEmbed(img_size=256, patch_size=16, embed_dim=128)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> embedded = embedder(x)
        >>> embedded.shape
        torch.Size([2, 256, 128])
    """
    pass
```

### 4.2 Module Documentation

Add module-level docstrings:

```python
"""Loss functions for biomedical image processing.

This module provides various loss functions optimized for:
- Image reconstruction (SSIM, PSNR, MSE)
- Image registration (NCC, Mutual Information, Gradient)
- Image segmentation (Dice, Surface Distance)
- Image fusion (gradient-based, intensity-based)

All losses support both 2D and 3D inputs automatically.

Examples:
    >>> from orochi.losses import SSIM, DiceLoss
    >>>
    >>> # Reconstruction loss
    >>> ssim_loss = SSIM(window_size=11)
    >>> loss = ssim_loss(pred, target)
    >>>
    >>> # Segmentation loss
    >>> dice_loss = DiceLoss(num_classes=5)
    >>> loss = dice_loss(logits, labels)
"""
```

### 4.3 Type Hints

Add type hints to all functions:

```python
from typing import Optional, Tuple, Union, List
import torch

def compute_jacobian_determinant(
    flow: torch.Tensor,
    return_negative_det: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute Jacobian determinant of deformation field."""
    pass
```

### 4.4 API Documentation

Generate API docs with Sphinx:
- Install Sphinx
- Configure autodoc
- Generate HTML documentation

---

## Phase 5: Code Quality & Style

### 5.1 Fix Import Issues

**Remove wildcard imports**:
```python
# Before:
from metrics import *

# After:
from orochi.training.metrics import (
    compute_dice,
    compute_surface_distance,
    compute_jacobian
)
```

**Organize imports** (PEP 8):
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Third-party
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local
from orochi.models import Mamba2D
from orochi.losses import SSIM, DiceLoss
from orochi.utils import AverageMeter
```

**Remove duplicate imports**: Automated scan and removal

### 5.2 Fix Naming Conventions

**Functions**: `snake_case`
```python
# Before:
def visualize_logits()
def check_nan()
def compute_surface_distances()  # Inconsistent

# After (all consistent):
def visualize_logits()
def check_nan()
def compute_surface_distance()  # Singular, matches others
```

**Classes**: `PascalCase`
```python
class SpatialTransformer  # Good
class SSIM  # Good (acronym)
class Mamba2D  # Good
```

**Constants**: `UPPER_SNAKE_CASE`
```python
SSIM_WINDOW_SIZE = 11
DEFAULT_LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
```

### 5.3 Remove Deprecated Patterns

**Remove torch.autograd.Variable**:
```python
# Before:
from torch.autograd import Variable
window = Variable(_2D_window.expand(...))

# After:
window = _2D_window.expand(...)  # No Variable needed
```

**Update CUDA usage**:
```python
# Before:
model.cuda()
tensor.cuda()

# After:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
tensor.to(device)
```

### 5.4 Improve Error Handling

**Add input validation**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.ndim not in [4, 5]:
        raise ValueError(f"Expected 4D or 5D tensor, got {x.ndim}D")

    # Process...
```

**Use context managers**:
```python
# Before:
file1 = open('./data/IXI_data/label_info.txt', 'r')
Lines = file1.readlines()
# No close!

# After:
with open(data_path / 'label_info.txt', 'r') as f:
    lines = f.readlines()
```

### 5.5 Code Formatting

**Run Black formatter**:
```bash
black orochi/ --line-length 100
```

**Run isort for imports**:
```bash
isort orochi/ --profile black
```

**Run flake8 for linting**:
```bash
flake8 orochi/ --max-line-length 100 --ignore E203,W503
```

---

## Phase 6: Testing

### 6.1 Test Infrastructure

**Setup pytest**:
```bash
pip install pytest pytest-cov pytest-xdist
```

**Create test structure**:
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_models/
│   ├── test_mamba_2d.py
│   ├── test_mamba_3d.py
│   └── test_components.py
├── test_losses/
│   ├── test_reconstruction.py
│   ├── test_registration.py
│   └── test_segmentation.py
├── test_data/
│   ├── test_datasets.py
│   └── test_transforms.py
└── test_utils/
    ├── test_visualization.py
    └── test_helpers.py
```

### 6.2 Write Unit Tests

**Loss function tests**:
```python
# tests/test_losses/test_reconstruction.py
import pytest
import torch
from orochi.losses import SSIM

class TestSSIM:
    """Test SSIM loss function."""

    @pytest.fixture
    def sample_2d_images(self):
        """Create sample 2D images for testing."""
        return torch.randn(2, 1, 256, 256)

    @pytest.fixture
    def sample_3d_images(self):
        """Create sample 3D images for testing."""
        return torch.randn(2, 1, 32, 256, 256)

    def test_ssim_2d_shape(self, sample_2d_images):
        """Test SSIM with 2D inputs."""
        ssim = SSIM()
        img1, img2 = sample_2d_images[:1], sample_2d_images[1:2]
        loss = ssim(img1, img2)
        assert loss.ndim == 0, "SSIM should return scalar"
        assert 0 <= loss <= 2, "SSIM loss should be in [0, 2]"

    def test_ssim_3d_shape(self, sample_3d_images):
        """Test SSIM with 3D inputs."""
        ssim = SSIM()
        img1, img2 = sample_3d_images[:1], sample_3d_images[1:2]
        loss = ssim(img1, img2)
        assert loss.ndim == 0, "SSIM should return scalar"

    def test_ssim_identical_images(self, sample_2d_images):
        """SSIM of identical images should be 0."""
        ssim = SSIM()
        img = sample_2d_images[:1]
        loss = ssim(img, img)
        assert loss < 0.01, "SSIM of identical images should be ~0"

    def test_ssim_invalid_input(self):
        """Test error handling for invalid inputs."""
        ssim = SSIM()
        with pytest.raises(ValueError):
            ssim(torch.randn(2, 3), torch.randn(2, 3))  # Wrong dims
```

**Model tests**:
```python
# tests/test_models/test_mamba_2d.py
import pytest
import torch
from orochi.models import Mamba2D
from orochi.configs import Mamba2DConfig

class TestMamba2D:
    """Test 2D Mamba model."""

    @pytest.fixture
    def model_config(self):
        return Mamba2DConfig(
            img_size=[256, 256],
            patch_size=16,
            embed_dim=128
        )

    @pytest.fixture
    def model(self, model_config):
        return Mamba2D(model_config)

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self, model):
        """Test forward pass with dummy data."""
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1, 256, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, model):
        """Test backward pass works."""
        x = torch.randn(2, 1, 256, 256)
        target = torch.randn(2, 1, 256, 256)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
```

### 6.3 Integration Tests

**End-to-end training test**:
```python
# tests/test_training/test_trainer.py
def test_training_loop():
    """Test complete training loop."""
    config = Mamba2DConfig(batch_size=2, num_epochs=2)
    model = Mamba2D(config)
    trainer = Trainer(model, config)

    # Use tiny dummy dataset
    train_loader = create_dummy_dataloader(num_samples=10)
    val_loader = create_dummy_dataloader(num_samples=5)

    # Train for 2 epochs
    history = trainer.fit(train_loader, val_loader)

    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2
```

### 6.4 Test Coverage

**Target: 80%+ coverage**

```bash
# Run tests with coverage
pytest tests/ --cov=orochi --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

---

## Phase 7: Final Cleanup

### 7.1 Remove Commented Code

Systematically remove all commented-out code blocks:
- Keep only explanatory comments
- Remove old implementations
- Document why code was removed (in git commit)

### 7.2 Clean Up Files

**Remove SurfaceDice lookup table from code**:
- Move to JSON/YAML data file
- Load at runtime
- Clean separation of code and data

**Fix file organization**:
- Delete duplicate files
- Consolidate similar scripts
- Clean up temp directories

### 7.3 Dependency Cleanup

**Create proper setup.py**:
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="orochi",
    version="1.0.0",
    description="Versatile Biomedical Image Processor",
    author="Orochi Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "mamba-ssm==2.2.2",
        "einops>=0.8.0",
        "monai>=1.4.0",
        # ... other deps
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ]
    },
    python_requires=">=3.8",
)
```

**Move requirements.txt to root**

### 7.4 Add Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### 7.5 Verification

**Run complete test suite**:
```bash
pytest tests/ -v --cov=orochi
```

**Verify all experiments still work**:
- Test 2D super-resolution
- Test 2D isotropic restoration
- Test 2D fusion
- Test 3D super-resolution
- Test 3D registration

**Benchmark performance**:
- Ensure no performance regression
- Compare training speed before/after
- Verify memory usage unchanged

---

## Implementation Timeline

### Week 1-2: Foundation
- Phase 1: Code organization
- Phase 2: Eliminate duplication (losses, utils)
- Create new directory structure
- Migrate critical components

### Week 3-4: Infrastructure
- Phase 3: Configuration management
- Phase 4: Documentation (50% complete)
- Set up testing infrastructure

### Week 5-6: Quality & Testing
- Phase 5: Code quality fixes
- Phase 6: Write unit tests
- Achieve 60%+ test coverage

### Week 7-8: Finalization
- Phase 4: Complete documentation
- Phase 6: Increase test coverage to 80%+
- Phase 7: Final cleanup
- Verification and validation

---

## Success Criteria

✅ **Code Quality**
- [ ] Code duplication reduced from 40-50% to < 5%
- [ ] All imports properly organized
- [ ] No hard-coded paths
- [ ] No magic numbers
- [ ] Black/isort/flake8 passing

✅ **Documentation**
- [ ] 100% public API documented
- [ ] All classes have docstrings
- [ ] All public functions have docstrings
- [ ] Type hints on all functions
- [ ] API documentation generated

✅ **Testing**
- [ ] 80%+ test coverage
- [ ] All critical paths tested
- [ ] Integration tests passing
- [ ] CI/CD pipeline configured

✅ **Configuration**
- [ ] Central config system
- [ ] No hard-coded values
- [ ] Environment variable support
- [ ] YAML config files

✅ **Functionality**
- [ ] All original features work
- [ ] No performance regression
- [ ] All experiments reproducible
- [ ] Backward compatibility maintained

---

## Risk Mitigation

1. **Breaking Changes**
   - Keep legacy code in separate directory
   - Maintain compatibility layer
   - Extensive testing before removal

2. **Performance Regression**
   - Benchmark before/after
   - Profile critical paths
   - Optimize if needed

3. **Incomplete Migration**
   - Incremental approach
   - Each phase independently testable
   - Can pause and resume

4. **Loss of Functionality**
   - Comprehensive test suite
   - Manual verification of all tasks
   - User acceptance testing

---

## Deliverables

1. **Clean Codebase**
   - Organized directory structure
   - Deduplicated code
   - Well-documented
   - Fully tested

2. **Documentation**
   - API documentation
   - User guide
   - Developer guide
   - Migration guide

3. **Testing Suite**
   - Unit tests
   - Integration tests
   - Coverage reports
   - CI/CD configuration

4. **Configuration System**
   - Config files
   - Environment templates
   - Setup scripts

5. **Migration Guide**
   - How to update existing code
   - Deprecated feature list
   - Compatibility layer docs

---

## Next Steps

**Ready to proceed?** I recommend:

1. **Start with Phase 1** (Code Organization) - Low risk, high impact
2. **Then Phase 2** (Eliminate Duplication) - Biggest win for code quality
3. **Phase 3** (Configuration) - Makes everything else easier
4. **Phases 4-7** - Polish and professionalism

**Estimated total time**: 6-8 weeks for complete cleanup

**Would you like me to:**
- Start with Phase 1 immediately?
- Focus on a specific phase first?
- Create a minimal viable cleanup (Phases 1-3 only)?
- Proceed with full cleanup plan?

Please approve the plan or suggest modifications before I begin execution.
