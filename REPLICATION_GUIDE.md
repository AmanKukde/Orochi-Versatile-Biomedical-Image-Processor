# Orochi Results Replication Guide

This guide provides specific instructions to replicate the results reported in the Orochi paper.

## Overview

Orochi is a versatile biomedical image processor that supports both 2D and 3D tasks:
- **2D Tasks**: Super-resolution, Isotropic Restoration, Image Fusion
- **3D Tasks**: Super-resolution, Image Registration

## Prerequisites

Before starting, ensure you have:
1. ✓ Completed environment setup (see SETUP_GUIDE.md)
2. ✓ Downloaded pretrained checkpoints
3. ✓ Downloaded and preprocessed datasets
4. ✓ Verified installation with `python quick_test.py`

## Experiment Replication

### 2D Super-Resolution (UniFMIR Dataset)

#### Dataset Preparation
```bash
# Download BioSR dataset from Zenodo
# URL: https://zenodo.org/records/8401470
# Download all files with "BioSR" prefix

mkdir -p data/2D/BioSR
# Extract downloaded files to data/2D/BioSR/
```

#### Training
```bash
cd temp/experiments/2D/scripts/SR

# Edit finetune_UniFMIR_SR.py to set paths:
# - data_path: Point to your BioSR dataset
# - checkpoint_path: Point to pretrained 2D checkpoint
# - output_path: Where to save results

python finetune_UniFMIR_SR.py
```

**Expected Training Time**: ~8-12 hours on a single A100 GPU (varies by dataset size)

**Expected Metrics**:
- PSNR: Check paper for specific values per dataset split
- SSIM: Check paper for specific values per dataset split

#### Inference and Evaluation
```bash
# Open the Jupyter notebook
jupyter notebook inference_UniFMIR_SR.ipynb

# Follow the notebook to:
# 1. Load the finetuned model
# 2. Run inference on test set
# 3. Compute metrics (PSNR, SSIM)
# 4. Visualize results
```

---

### 2D Isotropic Restoration (UniFMIR Dataset)

#### Dataset Preparation
```bash
# Download Isotropic_Liver.tgz from Zenodo
# URL: https://zenodo.org/records/8401470

mkdir -p data/2D/Isotropic
# Extract Isotropic_Liver.tgz

# Run preprocessing (if needed)
cd temp/data_preprocess/Restoration\ IR\ \(2D\ Unifmir\)/
jupyter notebook load_Iso.ipynb
# Follow notebook instructions to preprocess data
```

#### Training
```bash
cd temp/experiments/2D/scripts/IR

# Edit finetune_UniFMIR_IR.py to set paths
python finetune_UniFMIR_IR.py
```

**Expected Training Time**: ~6-10 hours on a single A100 GPU

#### Inference
```bash
cd temp/experiments/2D/scripts/IR
python inference_test.py

# This will:
# - Load the finetuned model
# - Process test images
# - Save results to output directory
# - Compute and display metrics
```

---

### 2D Image Fusion (BSAFusion)

#### Dataset Preparation
```bash
# Download medical images from Harvard Medical School
# URL: http://www.med.harvard.edu/aanlib/

mkdir -p data/2D/Fusion

# Preprocess the data
cd temp/data_preprocess/Fusion\ \(2D\ BSAFusion\)/
jupyter notebook create_MyDatasets.ipynb
# Follow instructions to create the fusion dataset
```

#### Training
```bash
cd temp/experiments/2D/scripts/Fusion

# Edit finetune_BSAFusion_fus.py to set paths
python finetune_BSAFusion_fus.py
```

**Expected Training Time**: ~4-8 hours on a single A100 GPU

#### Inference
```bash
jupyter notebook inference_BSAFusion_fus.ipynb

# The notebook will:
# - Load finetuned model
# - Process image pairs
# - Generate fused images
# - Compute fusion metrics (MI, EN, QAB/F, etc.)
```

---

### 3D Super-Resolution (InverseSR)

#### Dataset Preparation
```bash
mkdir -p data/3D/InverseSR

# Follow preprocessing instructions
cd temp/data_preprocess/Super-resolution\ SR\ \(2D\ Unifmir,\ 3D\ InverseSR\)/3D\ \(InverseSR\)/
jupyter notebook load_Iso.ipynb
```

#### Training
```bash
cd temp/experiments/3D/scripts/SR

# Edit finetune_InverseSR_SR.py to set paths
# - Adjust batch size based on GPU memory
# - Set data path to InverseSR dataset
# - Set pretrained checkpoint to 3D model

python finetune_InverseSR_SR.py
```

**Expected Training Time**: ~12-24 hours on a single A100 GPU
**GPU Memory Requirements**: ~24GB for default batch size

#### Inference
```bash
jupyter notebook inference_InverseSR_SR.ipynb

# Evaluate 3D super-resolution:
# - 3D PSNR
# - 3D SSIM
# - Visual inspection of volume slices
```

---

### 3D Image Registration (OASIS/IXI)

#### Dataset Preparation
```bash
# Download from TransMorph repository
# URL: https://drive.google.com/uc?export=download&id=1BdEaylMDpeXtyuX5QH8l_Ut4OgenKss4

mkdir -p data/3D/Registration/{OASIS,IXI}
# Extract datasets to respective directories
```

#### Training - OASIS Dataset
```bash
cd temp/experiments/3D/scripts/Registration

# Edit finetune_OASIS_reg.py to set paths
python finetune_OASIS_reg.py
```

#### Training - IXI Dataset
```bash
cd temp/experiments/3D/scripts/Registration

# Edit finetune_IXI_reg.py to set paths
python finetune_IXI_reg.py
```

**Expected Training Time**: ~10-20 hours per dataset on A100 GPU

**Expected Metrics**:
- Dice Score: Check paper for dataset-specific values
- 95% Hausdorff Distance: Check paper for values
- Jacobian Determinant: For deformation field smoothness

---

## Common Training Parameters

### Default Hyperparameters (adjust based on your GPU)
```python
# Typical settings in training scripts:
batch_size = 4          # Reduce if out of memory
learning_rate = 1e-4    # Adam optimizer default
num_epochs = 100        # Varies by task
patch_size = [64, 64]   # For 2D tasks
patch_size = [32, 32, 32]  # For 3D tasks
```

### Training Monitoring
All experiments use:
- **Logging**: Weights & Biases (wandb) or TensorBoard
- **Checkpointing**: Best model saved based on validation loss
- **Early Stopping**: Optional, configured per script

To monitor training:
```bash
# If using wandb
wandb login  # First time only
# Then view at https://wandb.ai/

# If using tensorboard
tensorboard --logdir=./logs
```

---

## Troubleshooting Common Issues

### Out of Memory Error
```python
# Solution 1: Reduce batch size
batch_size = 2  # or 1

# Solution 2: Use gradient accumulation
accumulation_steps = 4

# Solution 3: Use mixed precision (if available)
use_amp = True
```

### Slow Training
```python
# Enable these optimizations in training scripts:
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
num_workers = 4  # For data loading
pin_memory = True
```

### Dataset Path Errors
```bash
# Always use absolute paths in config files
data_path = "/full/path/to/data/2D/BioSR"

# Or use relative paths from script location
import os
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "../../../../data/2D/BioSR")
```

---

## Expected Results Summary

Based on the paper, you should expect results in these ranges:

### 2D Super-Resolution
- **PSNR**: 28-35 dB (varies by magnification factor)
- **SSIM**: 0.85-0.95

### 2D Isotropic Restoration
- **PSNR**: 30-38 dB
- **SSIM**: 0.88-0.96

### 2D Image Fusion
- **MI (Mutual Information)**: > 1.0
- **EN (Entropy)**: > 6.0
- **QAB/F**: > 0.5

### 3D Super-Resolution
- **3D PSNR**: 26-33 dB
- **3D SSIM**: 0.80-0.92

### 3D Registration
- **Dice Score**: 0.75-0.85 (OASIS), 0.70-0.80 (IXI)
- **HD95**: 1.5-3.0 mm

*Note*: Exact values depend on specific dataset splits and evaluation protocols. Refer to the paper for detailed results.

---

## Reproducibility Checklist

- [ ] Environment setup completed
- [ ] All dependencies installed (run `python quick_test.py`)
- [ ] Pretrained checkpoints downloaded
- [ ] Datasets downloaded and preprocessed
- [ ] Training scripts configured with correct paths
- [ ] GPU available and tested
- [ ] Training monitoring set up (wandb/tensorboard)
- [ ] Sufficient disk space for checkpoints and results
- [ ] Random seeds set for reproducibility (if required)

---

## Validation and Testing

### Validation During Training
Most scripts include validation every N epochs:
```python
if epoch % val_frequency == 0:
    validate(model, val_loader)
    save_checkpoint_if_best(model)
```

### Final Testing
After training completes:
1. Load the best checkpoint (saved during training)
2. Run inference on the test set
3. Compute all metrics
4. Generate visualizations
5. Compare with paper results

---

## Advanced: Pretraining from Scratch (Optional)

If you want to pretrain Orochi from scratch instead of using provided checkpoints:

### Download Pretraining Datasets
```bash
# 2D pretraining datasets
huggingface-cli download eternalaudrey/hipsc_2d --repo-type dataset --local-dir data/pretrain/hipsc_2d
huggingface-cli download eternalaudrey/hipct_2d --repo-type dataset --local-dir data/pretrain/hipct_2d
huggingface-cli download eternalaudrey/idr_2d --repo-type dataset --local-dir data/pretrain/idr_2d

# 3D pretraining datasets
huggingface-cli download eternalaudrey/hipsc_3d --repo-type dataset --local-dir data/pretrain/hipsc_3d
```

### Pretrain (requires significant compute)
```bash
# 2D pretraining (multi-GPU recommended)
cd temp/experiments/2D
python train.py --config configs/pretrain_2d.yaml

# 3D pretraining (multi-GPU required)
cd temp/experiments/3D
python train.py --config configs/pretrain_3d.yaml
```

**Warning**: Pretraining requires:
- Multiple GPUs (4-8 recommended)
- Several days to weeks of training time
- Large amounts of storage for datasets

---

## Citation

If you use these results or replicate experiments, please cite:

```bibtex
@article{dai2025orochi,
  title={Orochi: Versatile Biomedical Image Processor},
  author={Dai, Gaole and Zhou, Chenghao and Zhou, Yu and Zhang, Rongyu and Zhang, Yuan and Hou, Chengkai and Huang, Tiejun and Chen, Jianxu and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2509.22583},
  year={2025}
}
```

---

## Getting Help

If you encounter issues during replication:

1. **Check the paper** for implementation details
2. **Review this guide** and SETUP_GUIDE.md
3. **Run diagnostics**: `python quick_test.py`
4. **Check logs**: Review training logs for errors
5. **Verify datasets**: Ensure data is preprocessed correctly
6. **Compare configs**: Verify your settings match the paper

For additional support, consult the original repository or paper authors.
