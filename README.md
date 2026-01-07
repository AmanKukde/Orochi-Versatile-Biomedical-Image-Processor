# Orochi: Versatile Biomedical Image Processor

Official Implementation for "Orochi: Versatile Biomedical Image Processor"

**Enhanced with modular encoder support, including Vision Transformers, HuggingFace models, and 3DINO-ViT integration.**

## Citation

```bibtex
@article{dai2025orochi,
  title={Orochi: Versatile Biomedical Image Processor},
  author={Dai, Gaole and Zhou, Chenghao and Zhou, Yu and Zhang, Rongyu and Zhang, Yuan and Hou, Chengkai and Huang, Tiejun and Chen, Jianxu and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2509.22583},
  year={2025}
}
```

## Features

✨ **Multi-Task Learning**: Registration, Fusion, Super-Resolution, Isotropic Restoration
🔧 **Modular Encoder Architecture**: Easy swapping between Mamba, ViT, 3DINO-ViT, HuggingFace models
📦 **HuggingFace Integration**: Load any pretrained ViT directly from HuggingFace Hub
🎯 **Bottleneck FFN**: Automatic dimension adaptation for pretrained models
🔮 **Native 3D Support**: Direct 3D processing for medical imaging models
📊 **Robust Checkpointing**: WandB integration with code/config artifacts
🚀 **Production Ready**: SLURM batch scripts and comprehensive documentation

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/Orochi-Versatile-Biomedical-Image-Processor.git
cd Orochi-Versatile-Biomedical-Image-Processor

# Create environment
conda env create -f environment.yaml
conda activate orochi

# Install required packages
pip install transformers timm wandb
```

### 2. Mamba SSM Fix

Add this line at line 777 in `[site-packages]/mamba_ssm/ops/triton/ssd_combined.py`:

```python
xBC = xBC.contiguous()  # <-- Add this line
```

### 3. Download Pretrained Checkpoints

**Mamba Pretrained Models:**
- [2D Model](https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt)
- [3D Model](https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt)

```bash
# Download and place in pretrained_checkpoints/
wget https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt/resolve/main/mamba_fm_3d.pth.tar
```

---

## Training

### Option 1: Standard ViT Encoder

```bash
# Edit config
vim configs/vit_finetune.yaml

# Submit SLURM job
sbatch train.sh
```

### Option 2: HuggingFace Pretrained ViT

```bash
# Edit config
vim configs/huggingface_vit.yaml
# Set: hf_model_name: "google/vit-base-patch16-224"

# Submit job
sbatch train_huggingface.sh
```

### Option 3: 3DINO-ViT (Two-Phase)

**Phase 1: Train bottleneck**
```bash
# Edit config with 3DINO checkpoint path
vim configs/3dino_bottleneck.yaml

# Train bottleneck only (encoder + decoders frozen)
sbatch train_3dino_phase1.sh
```

**Phase 2: Finetune encoder**
```bash
# Update phase 1 checkpoint path in script
vim train_3dino_phase2.sh

# Finetune encoder (decoders frozen)
sbatch train_3dino_phase2.sh
```

### Option 4: Mamba Encoder

```bash
sbatch train_mamba.sh
```

---

## Documentation

### Core Guides

📖 **[ENCODER_GUIDE.md](ENCODER_GUIDE.md)** - Complete guide to encoder swapping
- Available encoders (Mamba, ViT, 3DINO, HuggingFace)
- Configuration examples
- Two-phase training strategy
- Troubleshooting

📦 **[HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md)** - HuggingFace model loading
- Supported models (Google ViT, DeiT, Swin)
- 2D→3D weight inflation
- Native 3D model support
- Model selection guide

🖥️ **[SLURM_TRAINING_GUIDE.md](SLURM_TRAINING_GUIDE.md)** - SLURM cluster training
- Batch script usage
- Job management
- Resource configuration
- Two-phase training workflow

### Additional Resources

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Model architecture details
- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Training best practices
- **[docs/GLOSSARY.md](docs/GLOSSARY.md)** - Technical terminology

---

## Supported Encoders

| Encoder | Type | Pretrained | Bottleneck | Use Case |
|---------|------|------------|------------|----------|
| **Mamba** | SSM | ✓ | Optional | Fast, efficient |
| **ViT** | Transformer | ✓ | Optional | Standard baseline |
| **3DINO-ViT** | Biomedical ViT | ✓ | Required | Transfer learning |
| **HuggingFace ViT** | Any HF model | ✓ | Required | Experimentation |
| **TIMM** | 1000+ models | ✓ | Required | Wide selection |
| **Native 3D** | Medical/Video | ✓ | Required | Direct 3D |

---

## Configuration Examples

### Standard ViT

```yaml
encoder_type: "vit"
embed_dim: 128
freeze_decoders: true
```

### HuggingFace ViT-Base

```yaml
encoder_type: "huggingface"
hf_model_name: "google/vit-base-patch16-224"
use_bottleneck: true
encoder_dim: 768
embed_dim: 128
```

### 3DINO-ViT

```yaml
encoder_type: "3dino"
pretrained_encoder_path: "/path/to/3dino.pth"
use_bottleneck: true
encoder_dim: 384
embed_dim: 128
freeze_encoder: true  # Phase 1
freeze_decoders: true
```

### Native 3D Model

```yaml
encoder_type: "huggingface"
hf_model_name: "your-org/your-3d-model"
is_3d_native: true  # No weight inflation
use_bottleneck: true
encoder_dim: 768
```

---

## Training Scripts

| Script | Purpose | Encoder | Duration |
|--------|---------|---------|----------|
| `train.sh` | ViT training | ViT | 36h |
| `train_mamba.sh` | Mamba training | Mamba | 36h |
| `train_huggingface.sh` | HF models | HF ViT | 36h |
| `train_3dino_phase1.sh` | Bottleneck only | 3DINO | 36h |
| `train_3dino_phase2.sh` | Encoder finetune | 3DINO | 72h |
| `train_local.sh` | Local/interactive | Any | Variable |

---

## Datasets

### Pretraining Datasets

- **hiPSC 2D**: [eternalaudrey/hipsc_2d](https://huggingface.co/datasets/eternalaudrey/hipsc_2d)
- **hiPSC 3D**: [eternalaudrey/hipsc_3d](https://huggingface.co/datasets/eternalaudrey/hipsc_3d)
- **HiP-CT 2D**: [eternalaudrey/hipct_2d](https://huggingface.co/datasets/eternalaudrey/hipct_2d)
- **IDR 2D**: [eternalaudrey/idr_2d](https://huggingface.co/datasets/eternalaudrey/idr_2d)
- **IDR Raw**: [eternalaudrey/idr-01-04](https://huggingface.co/collections/eternalaudrey/idr)

### Finetuning Datasets

- **Super-Resolution**: [UniFMIR](https://zenodo.org/records/8401470) (BioSR datasets)
- **Isotropic Restoration**: [UniFMIR](https://zenodo.org/records/8401470) (Isotropic_Liver)
- **Fusion**: [Harvard Medical](http://www.med.harvard.edu/aanlib/)
- **Registration**: [TransMorph](https://drive.google.com/uc?export=download&id=1BdEaylMDpeXtyuX5QH8l_Ut4OgenKss4)

---

## Architecture Overview

```
Input 3D Volume (B, 2, 32, 224, 224)
    ↓
┌─────────────────────────────────────┐
│  Encoder (Mamba/ViT/HF/3DINO)      │
│  - Hierarchical features            │
│  - Multi-scale processing           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Bottleneck FFN (optional)          │
│  - Dimension adaptation             │
│  - e.g., 768→128 or 384→128        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Task-Specific Decoders             │
│  - Registration (flow prediction)   │
│  - Fusion (multi-modal merge)       │
│  - Super-Resolution (upsampling)    │
│  - Isotropic Restoration (denoise)  │
└─────────────────────────────────────┘
    ↓
Multi-Task Outputs
```

---

## Key Improvements

### Modular Encoder Factory

Easily swap encoders via configuration:

```python
from src.encoder_factory import create_encoder

# Create ViT encoder
encoder = create_encoder(config, encoder_type='vit')

# Load HuggingFace ViT
encoder = create_encoder(config, encoder_type='huggingface',
                        model_name='google/vit-base-patch16-224')
```

### Bottleneck FFN

Automatically adapt dimensions between encoder and decoders:

```python
# 3DINO (384-dim) → Mamba decoders (128-dim)
config.use_bottleneck = True
config.encoder_dim = 384
config.embed_dim = 128
```

### Robust Checkpointing

Automatic saving of best models, code, and config:

```python
from src.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='./checkpoints',
    save_code=True,
    save_config=True,
    use_wandb=True
)
```

---

## Testing

### Visualization Notebooks

- **`visualize_degradations.ipynb`**: See synthetic degradations
- **`test_metrics.ipynb`**: Batched inference and metrics
- **`test_trained_model.ipynb`**: Load and test trained models

### Run Tests

```bash
# Test dataloader
python scripts/test_dataloader.py

# Test inference
python src/inference_test.py
```

---

## Project Structure

```
Orochi/
├── configs/                    # Training configurations
│   ├── vit_finetune.yaml
│   ├── huggingface_vit.yaml
│   ├── 3dino_bottleneck.yaml
│   └── 3dino_finetune.yaml
├── src/                        # Source code
│   ├── encoder_factory.py      # Modular encoder creation
│   ├── bottleneck.py           # Dimension adaptation
│   ├── checkpoint_manager.py   # Checkpoint management
│   ├── vit_model.py            # Main model
│   ├── vit_encoder.py          # ViT encoder
│   ├── ours_mamba.py           # Mamba encoder + decoders
│   └── finetune_with_wandb.py  # Training script
├── docs/                       # Additional documentation
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Utility scripts
├── train*.sh                   # SLURM batch scripts
└── README.md                   # This file
```

---

## Troubleshooting

### Common Issues

1. **SLURM DOS line breaks**: Scripts have Unix line endings (LF)
2. **HuggingFace not found**: Install with `pip install transformers`
3. **TIMM not found**: Install with `pip install timm`
4. **OOM errors**: Reduce batch size or use gradient accumulation
5. **Dimension mismatch**: Enable bottleneck with correct `encoder_dim`

See **[ENCODER_GUIDE.md](ENCODER_GUIDE.md)** and **[HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md)** for detailed troubleshooting.

---

## Contributing

Contributions are welcome! Please see our contribution guidelines.

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Acknowledgments

- Original Orochi paper and codebase
- HuggingFace Transformers library
- TIMM (PyTorch Image Models)
- Mamba SSM architecture
- 3DINO-ViT for biomedical pretraining

---

## Contact

For questions and issues, please open a GitHub issue or contact the authors.
