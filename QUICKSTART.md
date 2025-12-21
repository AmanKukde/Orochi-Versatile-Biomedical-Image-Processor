# Orochi Quick Start Guide

Get up and running with Orochi in minutes!

## 🚀 Quick Setup (5 steps)

### 1. Clone the Repository (if not already done)
```bash
git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
cd foundation-mamba-biomed
```

### 2. Create the Environment
```bash
conda env create -f temp/environment.yaml
conda activate mamba_biomed
```

### 3. Run Automated Setup
```bash
python setup_environment.py
```
This script will:
- Fix the mamba_ssm package issue
- Verify your installation
- Test CUDA availability

### 4. Test Your Installation
```bash
python quick_test.py
```
This runs a comprehensive test suite to ensure everything works.

### 5. Download Checkpoints & Data

#### Option A: Quick Demo (No downloads needed)
```bash
# Just test the model architecture
python quick_test.py  # Already done in step 4
```

#### Option B: Run Full Experiments
```bash
# Download pretrained checkpoints
mkdir -p checkpoints/2D checkpoints/3D
huggingface-cli download eternalaudrey/mamba-fm-2d-ckpt --local-dir checkpoints/2D
huggingface-cli download eternalaudrey/mamba-fm-3d-ckpt --local-dir checkpoints/3D

# Download datasets (see SETUP_GUIDE.md for details)
# For example, for 2D Super-Resolution:
# 1. Visit https://zenodo.org/records/8401470
# 2. Download BioSR dataset
# 3. Extract to data/2D/BioSR/
```

## 📁 What You Get

After setup, you'll have:
```
Orochi-Versatile-Biomedical-Image-Processor/
├── QUICKSTART.md          ← You are here!
├── SETUP_GUIDE.md         ← Detailed setup instructions
├── REPLICATION_GUIDE.md   ← How to replicate paper results
├── setup_environment.py   ← Automated setup script
├── quick_test.py          ← Test your installation
├── temp/
│   ├── environment.yaml   ← Conda environment
│   └── experiments/       ← Training & inference scripts
│       ├── 2D/            ← 2D experiments (SR, IR, Fusion)
│       └── 3D/            ← 3D experiments (SR, Registration)
└── README.md              ← Original repository README
```

## 🎯 Next Steps

Choose your path:

### Path 1: Just Exploring
✓ You're done! You've verified the installation works.
- Read SETUP_GUIDE.md to learn more
- Browse the code in temp/experiments/

### Path 2: Run Experiments
1. Read REPLICATION_GUIDE.md
2. Download datasets for your task of interest
3. Download pretrained checkpoints
4. Run training scripts in temp/experiments/2D/ or temp/experiments/3D/

### Path 3: Research & Development
1. Complete Path 2
2. Read the paper: arXiv:2509.22583
3. Modify model architecture in temp/experiments/*/ours_mamba.py
4. Experiment with different datasets or tasks

## 📊 Available Experiments

### 2D Tasks
| Task | Dataset | Script Location |
|------|---------|----------------|
| Super-Resolution | UniFMIR (BioSR) | `temp/experiments/2D/scripts/SR/` |
| Isotropic Restoration | UniFMIR | `temp/experiments/2D/scripts/IR/` |
| Image Fusion | BSAFusion | `temp/experiments/2D/scripts/Fusion/` |

### 3D Tasks
| Task | Dataset | Script Location |
|------|---------|----------------|
| Super-Resolution | InverseSR | `temp/experiments/3D/scripts/SR/` |
| Registration | OASIS/IXI | `temp/experiments/3D/scripts/Registration/` |

## 🐛 Troubleshooting

### "CUDA not available"
- Check GPU drivers: `nvidia-smi`
- Reinstall PyTorch with CUDA support

### "mamba_ssm import error"
- Run `python setup_environment.py` again
- The script will fix the mamba_ssm package

### "Module not found"
- Activate environment: `conda activate mamba_biomed`
- Reinstall: `conda env create -f temp/environment.yaml --force`

### "Out of memory" during testing
- This is normal for large models
- For quick_test.py, it uses small input sizes
- For real experiments, see REPLICATION_GUIDE.md for memory tips

## 📚 Documentation

- **QUICKSTART.md** (this file): Get started fast
- **SETUP_GUIDE.md**: Complete setup with all details
- **REPLICATION_GUIDE.md**: Step-by-step result replication
- **README.md**: Original repository documentation

## 🔗 Useful Links

- **Paper**: [arXiv:2509.22583](https://arxiv.org/abs/2509.22583)
- **2D Checkpoint**: [HuggingFace](https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt)
- **3D Checkpoint**: [HuggingFace](https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt)
- **Datasets**: See SETUP_GUIDE.md for all dataset links

## ⏱️ Time Estimates

| Task | Time Required |
|------|---------------|
| Environment Setup | 10-30 minutes |
| Download Checkpoints | 5-15 minutes |
| Download Single Dataset | 30 minutes - 2 hours |
| Run quick_test.py | 1-2 minutes |
| Train Single Task | 4-24 hours (varies by task) |

## 💡 Tips

1. **Start small**: Run quick_test.py before downloading large datasets
2. **Use wandb**: Sign up for [Weights & Biases](https://wandb.ai/) to track experiments
3. **Check paper**: Read the paper for expected results and implementation details
4. **GPU memory**: Reduce batch size if you hit memory limits
5. **Ask for help**: Check REPLICATION_GUIDE.md for common issues

## 🎉 Ready?

You're all set! Choose your next step above and dive in.

**Happy experimenting!** 🚀
