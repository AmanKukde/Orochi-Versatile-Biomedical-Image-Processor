# Troubleshooting Guide - Orochi

Common issues and their solutions when setting up and running Orochi.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Import Errors](#import-errors)
- [CUDA Issues](#cuda-issues)
- [Training Issues](#training-issues)
- [Dataset Issues](#dataset-issues)

---

## Installation Issues

### Issue: "No module named 'lpips'" or other missing packages

**Symptoms:**
```python
ModuleNotFoundError: No module named 'lpips'
```

**Cause:** The pip dependencies from `temp/requirements.txt` weren't installed when creating the conda environment.

**Solution 1 - Quick Fix (Recommended):**
```bash
# Make sure you're in the mamba_biomed environment
conda activate mamba_biomed

# Run the fix script
bash fix_environment.sh
```

**Solution 2 - Manual Fix:**
```bash
# Make sure you're in the mamba_biomed environment
conda activate mamba_biomed

# Install the missing package
pip install lpips==0.1.4

# Install all pip requirements
pip install -r temp/requirements.txt
```

**Solution 3 - Reinstall Everything:**
```bash
# Remove the old environment
conda deactivate
conda env remove -n mamba_biomed

# Recreate from scratch
conda env create -f temp/environment.yaml
conda activate mamba_biomed

# Install pip dependencies
pip install -r temp/requirements.txt
```

### Issue: "No module named 'mamba_ssm'"

**Cause:** mamba_ssm package not installed properly.

**Solution:**
```bash
conda activate mamba_biomed
pip install mamba-ssm==2.2.2
pip install causal-conv1d==1.4.0

# Then run the setup script to apply the fix
python setup_environment.py
```

### Issue: Environment creation fails

**Symptoms:**
```
Solving environment: failed
PackagesNotFoundError: The following packages are not available...
```

**Solution:**
```bash
# Try with different channel priority
conda config --set channel_priority flexible

# Create environment
conda env create -f temp/environment.yaml

# Reset channel priority
conda config --set channel_priority strict
```

**Alternative - Use pip instead:**
```bash
# Create a basic environment
conda create -n mamba_biomed python=3.12
conda activate mamba_biomed

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install everything else via pip
pip install -r temp/requirements.txt
```

---

## Import Errors

### Issue: "Cannot import name 'Orochi'" or model import fails

**Cause:** Python can't find the model files.

**Solution:**
```python
import sys
import os

# Add the experiments directory to Python path
sys.path.insert(0, os.path.join(os.getcwd(), "temp/experiments/2D"))

# Now you can import
from ours_mamba import Orochi
```

### Issue: "No module named 'monai'" or other medical imaging packages

**Solution:**
```bash
conda activate mamba_biomed
pip install monai==1.4.0
pip install nibabel simpleitk
```

### Issue: Import fails with "AttributeError" in mamba_ssm

**Cause:** The mamba_ssm fix hasn't been applied.

**Solution:**
```bash
python setup_environment.py
```

This will automatically find and fix the mamba_ssm package.

---

## CUDA Issues

### Issue: "CUDA not available" but you have a GPU

**Check your setup:**
```bash
# Check if GPU is visible
nvidia-smi

# Check CUDA version
nvidia-smi | grep "CUDA Version"
```

**Solution 1 - Reinstall PyTorch with correct CUDA version:**
```bash
conda activate mamba_biomed

# For CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Solution 2 - Check driver compatibility:**
```bash
# Your driver must support the CUDA version
# CUDA 12.1 requires driver >= 530.30.02
# If your driver is older, either update it or use an older CUDA version
```

### Issue: "CUDA out of memory"

**During quick_test.py:**
- This shouldn't happen with the small test sizes
- If it does, you may have other processes using GPU memory

**During training:**
```python
# In training scripts, reduce batch size:
batch_size = 2  # or even 1

# Enable gradient accumulation:
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps

# Use mixed precision:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Issue: "RuntimeError: CUDA error: no kernel image available"

**Cause:** PyTorch was compiled for different GPU architecture.

**Solution:**
```bash
# Reinstall PyTorch
conda activate mamba_biomed
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Training Issues

### Issue: Training is very slow

**Checklist:**
```python
# Enable these optimizations in your training script:
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Use multiple workers for data loading:
DataLoader(..., num_workers=4, pin_memory=True)

# Ensure you're using GPU:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Issue: Loss is NaN

**Causes and solutions:**
```python
# 1. Learning rate too high
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Try smaller LR

# 2. Gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check for NaN in data
assert not torch.isnan(input).any(), "NaN in input data"
```

### Issue: Model not improving

**Checklist:**
- Is the learning rate appropriate? (try 1e-4 to 1e-5)
- Is the data loading correctly? (visualize some batches)
- Is the model on GPU? (check with `next(model.parameters()).device`)
- Is the pretrained checkpoint loaded? (check loading logs)
- Are gradients flowing? (check with gradient norm logging)

---

## Dataset Issues

### Issue: "FileNotFoundError: Dataset not found"

**Solution:**
```bash
# Check the path in your training script
# Paths should be absolute or relative to the script location

# Example fix:
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../../../data/2D/BioSR")
```

### Issue: Dataset preprocessing fails

**For 2D datasets:**
```bash
cd temp/data_preprocess/[task_name]/
jupyter notebook  # Open the preprocessing notebook
# Follow the instructions in the notebook
```

**For 3D datasets:**
- Ensure you have enough disk space (3D data is large)
- Check that nibabel/SimpleITK can read your files
- Verify the data format matches expected format

### Issue: "RuntimeError: DataLoader worker died"

**Causes:**
- Out of memory (reduce num_workers)
- Corrupted data files (check dataset integrity)
- Path issues (use absolute paths)

**Solution:**
```python
# In your DataLoader:
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=2,  # Reduce from 4
    pin_memory=False,  # Disable if causing issues
    persistent_workers=False
)
```

---

## Package-Specific Issues

### mamba_ssm Issues

**Issue: "xBC.contiguous() not found" or similar errors**

Run the automated fix:
```bash
python setup_environment.py
```

**Manual fix:**
```bash
# Find site-packages
python -c "import site; print(site.getsitepackages())"

# Edit [site-packages]/mamba_ssm/ops/triton/ssd_combined.py
# Add at line 777 (after the torch.split line):
#     xBC = xBC.contiguous()
```

### wandb Issues

**Issue: "wandb login required"**

```bash
# Login to wandb
wandb login

# Or disable wandb in training scripts:
import os
os.environ["WANDB_MODE"] = "disabled"
```

### Jupyter Issues

**Issue: Kernel dies when running notebooks**

**Solution:**
```bash
# Make sure jupyter is installed in the conda environment
conda activate mamba_biomed
conda install jupyter ipykernel
python -m ipykernel install --user --name mamba_biomed --display-name "Python (mamba_biomed)"

# Then select "Python (mamba_biomed)" kernel in Jupyter
```

---

## Quick Diagnostic Commands

Run these to diagnose your setup:

```bash
# 1. Check environment
conda activate mamba_biomed
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. Check packages
python -c "import lpips; print('lpips OK')"
python -c "import mamba_ssm; print('mamba_ssm OK')"
python -c "import monai; print('monai OK')"
python -c "import einops; print('einops OK')"

# 3. Check GPU
nvidia-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 4. Run comprehensive test
python quick_test.py
```

---

## Getting More Help

If you're still stuck:

1. **Check the logs:** Look at error messages carefully
2. **Read the paper:** arXiv:2509.22583 has implementation details
3. **Check paths:** Most issues are path-related
4. **Verify environment:** Make sure conda environment is activated
5. **Check GPU memory:** `nvidia-smi` to see usage
6. **Try minimal example:** Start with quick_test.py before full training

---

## Common Error Messages Reference

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| `ModuleNotFoundError` | Package not installed | `pip install <package>` |
| `CUDA out of memory` | Batch size too large | Reduce batch_size |
| `No module named 'lpips'` | Pip deps not installed | `bash fix_environment.sh` |
| `FileNotFoundError` | Wrong path | Use absolute paths |
| `RuntimeError: no kernel` | Wrong PyTorch build | Reinstall PyTorch |
| `DataLoader worker died` | Too many workers | Reduce num_workers |
| `Loss is NaN` | Learning rate too high | Reduce learning rate |

---

## Prevention Tips

To avoid issues:

1. **Always activate environment:** `conda activate mamba_biomed`
2. **Use absolute paths:** Don't rely on relative paths
3. **Start small:** Test with quick_test.py first
4. **Monitor GPU:** Keep an eye on `nvidia-smi`
5. **Save checkpoints:** Don't lose your training progress
6. **Use version control:** Commit changes before modifying code
7. **Read the docs:** Check SETUP_GUIDE.md and REPLICATION_GUIDE.md

---

## Still Having Issues?

Create a diagnostic report:

```bash
# Run this and share the output
echo "=== Environment ==="
conda info
echo "=== Packages ==="
pip list
echo "=== CUDA ==="
nvidia-smi
echo "=== PyTorch ==="
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
echo "=== Test Results ==="
python quick_test.py
```

This helps identify the problem quickly!
