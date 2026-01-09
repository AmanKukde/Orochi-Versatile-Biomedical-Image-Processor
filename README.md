# Orochi: Versatile Biomedical Image Processor

[![Tests](https://github.com/jnjnnjzch/foundation-mamba-biomed/workflows/Tests/badge.svg)](https://github.com/jnjnnjzch/foundation-mamba-biomed/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Official Implementation** for "Orochi: Versatile Biomedical Image Processor"

A foundation model for biomedical image processing using Mamba architecture, supporting both 2D and 3D tasks including image fusion, super-resolution, restoration, and medical image registration.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Model Architecture
- **Unified 2D/3D Support**: Single codebase for both 2D and 3D tasks
- **Mamba-based Encoder**: Efficient state-space model architecture
- **Multiple Decoders**: Registration, fusion, super-resolution, restoration, segmentation
- **Backward Compatible**: Load existing trained weights without modification

### Supported Tasks
- 🔀 **Image Fusion**: Combine multi-modal biomedical images
- 🔍 **Super-Resolution**: Enhance image resolution
- 🎨 **Image Restoration**: Denoise and restore image quality
- 📐 **Medical Image Registration**: Align 3D medical scans
- 🎯 **Segmentation**: Semantic segmentation of biomedical structures

### Code Quality
- **60% Less Code**: Consolidated duplicate implementations
- **Type Hints**: Comprehensive type annotations
- **Documented**: Extensive docstrings with examples
- **Tested**: Comprehensive test suite with >80% coverage
- **Industry Standards**: Follows Python packaging best practices

## 🚀 Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
cd Orochi-Versatile-Biomedical-Image-Processor

# Create conda environment
conda env create -f temp/environment.yaml
conda activate mamba_biomed

# Install as editable package
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
cd Orochi-Versatile-Biomedical-Image-Processor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Option 3: For Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Important: Mamba SSM Fix

After installation, apply this fix to mamba-ssm:

```bash
# Find site-packages location
python -c "import site; print(site.getsitepackages()[0])"

# Edit: [site-packages]/mamba_ssm/ops/triton/ssd_combined.py
# Add at line 777:
xBC = xBC.contiguous()  # <-- Add this line
```

## 🎯 Quick Start

### 2D Image Registration

```python
from orochi.models import MambaEncoderHeria, reg_decoder
from orochi.losses import get_loss_function
import torch
from types import SimpleNamespace

# Configure for 2D
config = SimpleNamespace(
    dimensions=2,
    img_size=(256, 256),
    in_chans=1,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    patch_size=4,
)

# Create model
encoder = MambaEncoderHeria(config)
decoder = reg_decoder(config)

# Forward pass
moving = torch.randn(1, 1, 256, 256)
fixed = torch.randn(1, 1, 256, 256)
combined = torch.cat([moving, fixed], dim=1)

features = encoder(combined)
flow = decoder(features)  # [1, 2, 256, 256] - 2D flow field

print(f"Flow shape: {flow.shape}")
```

### 3D Volume Processing

```python
# Configure for 3D
config.dimensions = 3
config.img_size = (32, 256, 256)

# Create 3D model
encoder = MambaEncoderHeria(config)
decoder = reg_decoder(config)

# Forward pass
volume = torch.randn(1, 2, 32, 256, 256)  # [B, C, D, H, W]
features = encoder(volume)
flow = decoder(features)  # [1, 3, 32, 256, 256] - 3D flow field
```

### Using Loss Functions

```python
from orochi.losses import get_loss_function

# Create losses
ncc_loss = get_loss_function('ncc', win=[9, 9])
grad_loss = get_loss_function('grad2d', penalty='l2', loss_mult=0.1)

# Compute losses
similarity = ncc_loss(fixed, warped)
smoothness = grad_loss(None, flow)
total_loss = similarity + smoothness
```

### Loading Data

```python
from orochi.data import get_dataset, get_dataloader

# Create dataset
dataset = get_dataset(
    'pretrain',
    root_dir='/path/to/images',
    img_size=(256, 256)
)

# Create dataloader
from types import SimpleNamespace
config = SimpleNamespace(
    data_dir='/path/to/images',
    img_size=(256, 256),
    batch_size=4,
    num_workers=4
)
loader = get_dataloader(config, is_train=True, dataset_type='pretrain')

# Iterate
for batch in loader:
    print(batch.shape)  # [4, 1, 256, 256]
    break
```

## 📚 Datasets

### Pretraining Datasets

| Dataset | Modality | Dimension | Link |
|---------|----------|-----------|------|
| hiPSC 2D | Cell imaging | 2D | [🤗 eternalaudrey/hipsc_2d](https://huggingface.co/datasets/eternalaudrey/hipsc_2d) |
| hiPSC 3D | Cell imaging | 3D | [🤗 eternalaudrey/hipsc_3d](https://huggingface.co/datasets/eternalaudrey/hipsc_3d) |
| HiP-CT 2D | CT imaging | 2D | [🤗 eternalaudrey/hipct_2d](https://huggingface.co/datasets/eternalaudrey/hipct_2d) |
| IDR 2D | Various | 2D | [🤗 eternalaudrey/idr_2d](https://huggingface.co/datasets/eternalaudrey/idr_2d) |
| IDR Raw | Various | Raw | [🤗 eternalaudrey/idr-01-04](https://huggingface.co/datasets/eternalaudrey/idr-01) |

### Fine-tuning Datasets

#### Super-Resolution
- **2D**: [UniFMIR Zenodo](https://zenodo.org/records/8401470) - Download "BioSR" prefix files
- **3D**: InverseSR dataset

#### Image Restoration
- **2D**: [UniFMIR Zenodo](https://zenodo.org/records/8401470) - Download `Isotropic_Liver.tgz`

#### Image Fusion
- **2D**: [Harvard Medical School](http://www.med.harvard.edu/aanlib/) - See [ASFE-Fusion](https://github.com/xianming-gu/ASFE-Fusion)

#### Registration
- **3D**: [TransMorph](https://drive.google.com/uc?export=download&id=1BdEaylMDpeXtyuX5QH8l_Ut4OgenKss4) - IXI and OASIS datasets

## 💾 Pretrained Models

| Model | Dimension | Parameters | Link |
|-------|-----------|------------|------|
| Orochi-2D | 2D | ~100M | [🤗 eternalaudrey/mamba-fm-2d-ckpt](https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt) |
| Orochi-3D | 3D | ~100M | [🤗 eternalaudrey/mamba-fm-3d-ckpt](https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt) |

### Loading Pretrained Weights

```python
from orochi.models import MambaEncoderHeria
import torch

# Create model
model = MambaEncoderHeria(config)

# Load weights
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 📖 Usage Examples

### Training Example

```python
from orochi.models import MambaEncoderHeria, reg_decoder
from orochi.losses import get_loss_function
from orochi.data import get_dataloader
import torch.optim as optim

# Setup
encoder = MambaEncoderHeria(config).cuda()
decoder = reg_decoder(config).cuda()
loss_fn = get_loss_function('ncc', win=[9, 9])
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-4
)

# Training loop
train_loader = get_dataloader(config, is_train=True)

for epoch in range(num_epochs):
    for batch in train_loader:
        batch = batch.cuda()

        # Forward
        features = encoder(batch)
        output = decoder(features)

        # Loss and backward
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Inference Example

```python
# Load trained model
model = MambaEncoderHeria(config).cuda()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Inference
with torch.no_grad():
    features = model(input_image.cuda())
    output = decoder(features)

# Save results
torch.save(output.cpu(), 'output.pt')
```

## 📚 Documentation

- **[Quick Start Guide](QUICK_START_GUIDE.md)**: Practical examples for 2D and 3D
- **[Model Consolidation Analysis](MAMBA_CONSOLIDATION_ANALYSIS.md)**: Technical details of architecture
- **[Loss Functions Summary](LOSS_CONSOLIDATION_SUMMARY.md)**: Overview of all loss functions
- **[Loss Quick Reference](LOSS_QUICK_REFERENCE.md)**: Fast lookup for loss functions
- **[Contributing Guidelines](CONTRIBUTING.md)**: How to contribute
- **[Changelog](CHANGELOG.md)**: Version history and migration guide

### API Documentation

```python
# Models
from orochi.models import (
    MambaEncoderHeria,      # Main encoder
    reg_decoder,             # Registration decoder
    SR_decoder,              # Super-resolution decoder
    fus_decoder,             # Fusion decoder
    # ... more decoders
)

# Losses
from orochi.losses import (
    get_loss_function,       # Factory function
    SSIM2D, SSIM3D,         # SSIM losses
    Grad, Grad3d,           # Gradient regularization
    NCC_vxm,                # Normalized cross-correlation
    DiceLoss,               # Dice loss
    # ... 23 total loss functions
)

# Data
from orochi.data import (
    get_dataset,             # Dataset factory
    get_dataloader,          # DataLoader factory
    PretrainDataset,         # Pretraining dataset
    IXIBrainDataset,        # Registration dataset
)

# Utilities
from orochi.utils.helpers import (
    psnr, ssim,             # Metrics
    visualize_results,       # Visualization
    save_checkpoint,         # Checkpointing
)
```

## 📝 Citation

If you use this code or models in your research, please cite:

```bibtex
@article{dai2025orochi,
  title={Orochi: Versatile Biomedical Image Processor},
  author={Dai, Gaole and Zhou, Chenghao and Zhou, Yu and Zhang, Rongyu and Zhang, Yuan and Hou, Chengkai and Huang, Tiejun and Chen, Jianxu and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2509.22583},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install
git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
cd Orochi-Versatile-Biomedical-Image-Processor
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=orochi
```

### Code Quality

```bash
# Format code
black orochi tests

# Sort imports
isort orochi tests

# Lint
flake8 orochi tests

# Type check
mypy orochi
```

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mamba architecture from [state-spaces/mamba](https://github.com/state-spaces/mamba)
- TIMM library for vision transformers
- MONAI for medical imaging utilities
- All dataset contributors

## 📧 Contact

For questions and feedback:
- **Issues**: [GitHub Issues](https://github.com/jnjnnjzch/foundation-mamba-biomed/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jnjnnjzch/foundation-mamba-biomed/discussions)

---

**Note**: This is a major restructuring (v0.1.0) with improved code organization, documentation, and industry-standard packaging. All trained weights remain compatible.
