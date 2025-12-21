# Orochi Setup and Replication Guide

This guide provides step-by-step instructions to clone, set up, and replicate results for the Orochi: Versatile Biomedical Image Processor.

## Prerequisites

- Linux system with CUDA-capable GPU
- Conda/Miniconda installed
- Git installed
- At least 50GB of free disk space (for datasets and checkpoints)

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
cd foundation-mamba-biomed

# Or if you already have it cloned, navigate to the directory
cd Orochi-Versatile-Biomedical-Image-Processor
```

## Step 2: Create the Conda Environment

```bash
# Create the environment from the YAML file
conda env create -f temp/environment.yaml

# Activate the environment
conda activate mamba_biomed
```

## Step 3: Modify mamba_ssm Package (CRITICAL)

This step is required to fix a compatibility issue with the mamba_ssm package.

```bash
# Find your site-packages path
python -c "import site; print(site.getsitepackages())"

# The output will be something like:
# ['/path/to/miniconda3/envs/mamba_biomed/lib/python3.12/site-packages']

# Edit the file at line 777 in:
# [site-package path]/mamba_ssm/ops/triton/ssd_combined.py
#
# Add this line after line 777:
#     xBC = xBC.contiguous()  # <--- Add this line
```

Use the automated script in Step 4 to do this automatically.

## Step 4: Automated Setup Script

We provide a setup script to automate the mamba_ssm fix:

```bash
# Run the setup script
python setup_environment.py
```

## Step 5: Download Pretrained Checkpoints

Download the pretrained models from HuggingFace:

### 2D Pretrained Model
```bash
# Create checkpoint directory
mkdir -p checkpoints/2D

# Download 2D checkpoint (use git lfs or download directly from HuggingFace)
# Visit: https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt
# Or use huggingface-cli:
huggingface-cli download eternalaudrey/mamba-fm-2d-ckpt --local-dir checkpoints/2D
```

### 3D Pretrained Model
```bash
# Create checkpoint directory
mkdir -p checkpoints/3D

# Download 3D checkpoint
# Visit: https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt
# Or use huggingface-cli:
huggingface-cli download eternalaudrey/mamba-fm-3d-ckpt --local-dir checkpoints/3D
```

## Step 6: Download Datasets

### For 2D Tasks

#### Super Resolution (2D)
```bash
# Download BioSR dataset from Zenodo
# Visit: https://zenodo.org/records/8401470
# Download files with "BioSR" prefix and extract to:
mkdir -p data/2D/BioSR
# Extract downloaded files to the above directory
```

#### Isotropic Restoration (2D)
```bash
# Download Isotropic_Liver.tgz from Zenodo
# Visit: https://zenodo.org/records/8401470
mkdir -p data/2D/Isotropic
# Extract Isotropic_Liver.tgz to the above directory
# Run preprocessing scripts in temp/data_preprocess/Restoration IR (2D Unifmir)/
```

#### Fusion (2D)
```bash
# Download datasets from Harvard Medical School
# Visit: http://www.med.harvard.edu/aanlib/
mkdir -p data/2D/Fusion
# Copy downloaded files and run preprocessing
# See: temp/data_preprocess/Fusion (2D BSAFusion)/create_MyDatasets.ipynb
```

### For 3D Tasks

#### Super Resolution (3D)
```bash
# InverseSR dataset
# Follow instructions in: temp/data_preprocess/Super-resolution SR (2D Unifmir, 3D InverseSR)/3D (InverseSR)/
mkdir -p data/3D/InverseSR
```

#### Registration (3D)
```bash
# Download from TransMorph repository
mkdir -p data/3D/Registration
# Download from: https://drive.google.com/uc?export=download&id=1BdEaylMDpeXtyuX5QH8l_Ut4OgenKss4
# Extract to the above directory
```

### Pretraining Datasets (Optional)

If you want to pretrain from scratch:

```bash
# Download from HuggingFace datasets
# 2D datasets:
huggingface-cli download eternalaudrey/hipsc_2d --repo-type dataset --local-dir data/pretrain/hipsc_2d
huggingface-cli download eternalaudrey/hipct_2d --repo-type dataset --local-dir data/pretrain/hipct_2d
huggingface-cli download eternalaudrey/idr_2d --repo-type dataset --local-dir data/pretrain/idr_2d

# 3D datasets:
huggingface-cli download eternalaudrey/hipsc_3d --repo-type dataset --local-dir data/pretrain/hipsc_3d
```

## Step 7: Verify Installation

```bash
# Activate the environment
conda activate mamba_biomed

# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Test mamba_ssm
python -c "import mamba_ssm; print('mamba_ssm imported successfully')"

# Test other key packages
python -c "import monai, einops, timm; print('All key packages imported successfully')"
```

## Step 8: Run Experiments

### 2D Experiments

#### Super Resolution (2D)
```bash
# Finetune
cd temp/experiments/2D/scripts/SR
python finetune_UniFMIR_SR.py

# Inference/Evaluation
jupyter notebook inference_UniFMIR_SR.ipynb
```

#### Isotropic Restoration (2D)
```bash
cd temp/experiments/2D/scripts/IR
python finetune_UniFMIR_IR.py
python inference_test.py
```

#### Fusion (2D)
```bash
cd temp/experiments/2D/scripts/Fusion
python finetune_BSAFusion_fus.py
jupyter notebook inference_BSAFusion_fus.ipynb
```

### 3D Experiments

#### Super Resolution (3D)
```bash
cd temp/experiments/3D/scripts/SR
python finetune_InverseSR_SR.py
jupyter notebook inference_InverseSR_SR.ipynb
```

#### Registration (3D)
```bash
cd temp/experiments/3D/scripts/Registration

# For IXI dataset
python finetune_IXI_reg.py

# For OASIS dataset
python finetune_OASIS_reg.py
```

## Step 9: Expected Results

After running the experiments, you should see:

- **Checkpoints**: Saved in experiment-specific directories
- **Logs**: Training logs with loss curves and metrics
- **Evaluation Metrics**: PSNR, SSIM, and task-specific metrics
- **Output Images**: Reconstructed/processed images for visual comparison

## Troubleshooting

### Issue: CUDA Out of Memory
- Reduce batch size in the training scripts
- Use gradient accumulation if needed

### Issue: mamba_ssm import error
- Ensure you've applied the fix in Step 3
- Reinstall mamba_ssm: `pip install --upgrade mamba-ssm==2.2.2`

### Issue: Dataset not found
- Check dataset paths in the scripts
- Ensure datasets are extracted to the correct directories

### Issue: Missing dependencies
- Reinstall the environment: `conda env create -f temp/environment.yaml --force`

## Quick Start Example (2D Super Resolution)

For a quick test, here's a minimal example:

```bash
# 1. Activate environment
conda activate mamba_biomed

# 2. Download a small test dataset and checkpoint
mkdir -p checkpoints/2D data/2D/test

# 3. Run a test script (create a minimal test)
cd temp/experiments/2D
python -c "import torch; from ours_mamba import Orochi; model = Orochi(); print('Model created successfully!')"
```

## Directory Structure

```
Orochi-Versatile-Biomedical-Image-Processor/
├── temp/
│   ├── environment.yaml          # Conda environment specification
│   ├── requirements.txt          # Pip requirements
│   ├── experiments/
│   │   ├── 2D/                   # 2D experiments
│   │   │   ├── scripts/
│   │   │   │   ├── SR/           # Super resolution
│   │   │   │   ├── IR/           # Isotropic restoration
│   │   │   │   └── Fusion/       # Image fusion
│   │   │   └── ours_mamba.py     # Model implementation
│   │   └── 3D/                   # 3D experiments
│   │       ├── scripts/
│   │       │   ├── SR/           # Super resolution
│   │       │   └── Registration/ # Image registration
│   │       └── ours_mamba.py     # Model implementation
│   └── data_preprocess/          # Data preprocessing scripts
├── data/                         # Datasets (you create this)
├── checkpoints/                  # Model checkpoints (you create this)
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@article{dai2025orochi,
  title={Orochi: Versatile Biomedical Image Processor},
  author={Dai, Gaole and Zhou, Chenghao and Zhou, Yu and Zhang, Rongyu and Zhang, Yuan and Hou, Chengkai and Huang, Tiejun and Chen, Jianxu and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2509.22583},
  year={2025}
}
```

## Additional Resources

- **Paper**: arXiv:2509.22583
- **2D Checkpoint**: https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt
- **3D Checkpoint**: https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt
- **Datasets**: See links in README.md and this guide

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the original repository issues
3. Verify your environment matches the requirements
