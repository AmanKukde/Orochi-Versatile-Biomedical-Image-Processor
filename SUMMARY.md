# Setup Summary - Orochi Repository

This document summarizes the setup and replication materials created for the Orochi repository.

## 📋 What Was Created

I've created a comprehensive set of guides and scripts to help you clone, set up, and replicate results from the Orochi paper. Here's what's available:

### 📖 Documentation Files

1. **QUICKSTART.md** - Start here!
   - 5-step quick setup process
   - Choose your path (exploring, experiments, or research)
   - Common troubleshooting tips
   - Time estimates for each task

2. **SETUP_GUIDE.md** - Complete setup guide
   - Detailed environment setup instructions
   - How to download all datasets
   - How to download pretrained checkpoints
   - Step-by-step verification process
   - Directory structure explanation
   - Comprehensive troubleshooting section

3. **REPLICATION_GUIDE.md** - Replicate paper results
   - Specific instructions for each experiment
   - Expected training times and GPU requirements
   - Expected performance metrics
   - Common issues and solutions
   - Validation and testing procedures

### 🔧 Automated Scripts

1. **setup_environment.py** - Automated setup
   - Finds and fixes the mamba_ssm package issue
   - Verifies all dependencies
   - Tests CUDA availability
   - Checks directory structure
   - Provides next steps

2. **quick_test.py** - Installation verification
   - Tests all package imports
   - Verifies CUDA and GPU functionality
   - Tests mamba_ssm fix
   - Creates and tests Orochi model
   - Runs a forward pass with dummy data
   - Provides comprehensive test summary

## 🚀 Getting Started

### Recommended Workflow

```bash
# 1. Start with the quick start guide
cat QUICKSTART.md

# 2. Run automated setup
python setup_environment.py

# 3. Verify installation
python quick_test.py

# 4. For experiments, read the detailed guides
cat SETUP_GUIDE.md        # For environment details
cat REPLICATION_GUIDE.md  # For running experiments
```

## 📊 Repository Status

The repository is already cloned at:
```
/home/user/Orochi-Versatile-Biomedical-Image-Processor/
```

Current branch: `claude/clone-and-replicate-S7u2c`

### What's Ready
✓ Repository cloned
✓ Documentation created
✓ Setup scripts created
✓ Test scripts created

### What's Needed (by you)
- [ ] Create conda environment (`conda env create -f temp/environment.yaml`)
- [ ] Run setup script (`python setup_environment.py`)
- [ ] Download pretrained checkpoints
- [ ] Download datasets for tasks you want to run
- [ ] Run experiments

## 🎯 Experiments Available

### 2D Tasks
| Task | Dataset | Location |
|------|---------|----------|
| Super-Resolution | BioSR | temp/experiments/2D/scripts/SR/ |
| Isotropic Restoration | UniFMIR | temp/experiments/2D/scripts/IR/ |
| Image Fusion | BSAFusion | temp/experiments/2D/scripts/Fusion/ |

### 3D Tasks
| Task | Dataset | Location |
|------|---------|----------|
| Super-Resolution | InverseSR | temp/experiments/3D/scripts/SR/ |
| Registration | OASIS/IXI | temp/experiments/3D/scripts/Registration/ |

## 📦 Key Resources

### Pretrained Checkpoints
- 2D: https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt
- 3D: https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt

### Datasets
All dataset links are provided in SETUP_GUIDE.md, including:
- BioSR (2D SR)
- UniFMIR (2D IR)
- BSAFusion (2D Fusion)
- InverseSR (3D SR)
- OASIS/IXI (3D Registration)
- Pretraining datasets (hiPSC, HiP-CT, IDR)

## 🔍 Quick Reference

### Environment Setup
```bash
conda env create -f temp/environment.yaml
conda activate mamba_biomed
python setup_environment.py
```

### Test Installation
```bash
python quick_test.py
```

### Download Checkpoints
```bash
huggingface-cli download eternalaudrey/mamba-fm-2d-ckpt --local-dir checkpoints/2D
huggingface-cli download eternalaudrey/mamba-fm-3d-ckpt --local-dir checkpoints/3D
```

### Run Example Experiment (2D SR)
```bash
cd temp/experiments/2D/scripts/SR
# Edit finetune_UniFMIR_SR.py to set paths
python finetune_UniFMIR_SR.py
```

## 💡 Tips for Success

1. **Follow the order**: QUICKSTART → setup_environment.py → quick_test.py → experiments
2. **Read the guides**: Each guide has specific, detailed information
3. **Start small**: Test with quick_test.py before downloading large datasets
4. **Check paper**: Reference arXiv:2509.22583 for implementation details
5. **Monitor training**: Use wandb or tensorboard to track experiments
6. **Save checkpoints**: Models are large, so ensure you have storage space

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA not available | Check GPU drivers with `nvidia-smi` |
| mamba_ssm error | Run `python setup_environment.py` |
| Module not found | Activate environment: `conda activate mamba_biomed` |
| Out of memory | Reduce batch size in training scripts |
| Dataset not found | Check paths in SETUP_GUIDE.md |

## 📚 Documentation Map

```
QUICKSTART.md          → Quick 5-step setup, choose your path
    ↓
setup_environment.py   → Automated setup and verification
    ↓
quick_test.py          → Test your installation
    ↓
SETUP_GUIDE.md         → Detailed setup for everything
    ↓
REPLICATION_GUIDE.md   → Run experiments and replicate results
```

## 🎓 Learning Path

### Beginner
1. Read QUICKSTART.md
2. Run setup_environment.py
3. Run quick_test.py
4. Explore the code

### Intermediate
1. Complete beginner steps
2. Read SETUP_GUIDE.md
3. Download one dataset
4. Run one experiment

### Advanced
1. Complete intermediate steps
2. Read REPLICATION_GUIDE.md
3. Read the paper thoroughly
4. Replicate all experiments
5. Modify and extend the model

## 🔗 Additional Resources

- **Paper**: arXiv:2509.22583
- **Original README**: README.md in this repository
- **Model Code**: temp/experiments/2D/ours_mamba.py (2D) and temp/experiments/3D/ours_mamba.py (3D)
- **Training Scripts**: temp/experiments/{2D,3D}/scripts/*/

## ✅ Next Steps

1. **Read QUICKSTART.md** - Get oriented
2. **Run setup_environment.py** - Set up your environment
3. **Run quick_test.py** - Verify everything works
4. **Choose an experiment** - Pick from REPLICATION_GUIDE.md
5. **Download data** - Get datasets for your chosen experiment
6. **Start training** - Run the experiment scripts

## 📝 Notes

- All scripts are Python 3.12 compatible
- CUDA 12.1+ required for GPU support
- Minimum 16GB GPU memory recommended (24GB+ for 3D tasks)
- Training times vary from 4-24 hours per task
- Datasets can be large (10-50GB per dataset)

## 🎉 You're Ready!

You now have everything you need to:
- ✓ Set up the Orochi environment
- ✓ Verify your installation
- ✓ Download necessary data
- ✓ Run experiments
- ✓ Replicate paper results

Start with QUICKSTART.md and follow the guides. Good luck with your experiments! 🚀
