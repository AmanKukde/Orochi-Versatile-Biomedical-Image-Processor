# Running Orochi Without a Local GPU

This guide helps you run Orochi when you don't have a compatible NVIDIA GPU on your local machine.

## ⚠️ Important: Why You Need a GPU

**mamba_ssm REQUIRES an NVIDIA GPU with CUDA support.** It cannot run on:
- CPU-only systems
- AMD GPUs
- Apple Silicon (M1/M2) GPUs
- Intel GPUs

If you don't have a compatible NVIDIA GPU, you MUST use cloud computing.

## 🔍 Check Your System First

Run this to see what options you have:

```bash
python check_system_compatibility.py
```

This will tell you:
- If you have a compatible GPU
- What's missing from your system
- What alternatives you should use

---

## ☁️ Cloud Computing Options

### Option 1: Google Colab (FREE - RECOMMENDED for beginners)

**Pros:**
- ✓ Completely free tier available
- ✓ No setup required
- ✓ Tesla T4 GPU included
- ✓ Pre-installed Python/PyTorch
- ✓ Great for testing and small experiments

**Cons:**
- ✗ Limited session time (12 hours max)
- ✗ Files don't persist (need to save to Google Drive)
- ✗ Can be slow during peak hours
- ✗ Limited disk space

**Setup Steps:**

1. **Go to Google Colab**
   - Visit: https://colab.research.google.com/
   - Sign in with Google account

2. **Create a new notebook**
   ```python
   # Enable GPU
   # Go to: Runtime > Change runtime type > Hardware accelerator > GPU > Save
   ```

3. **Clone the repository**
   ```python
   !git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
   %cd foundation-mamba-biomed
   ```

4. **Install dependencies**
   ```python
   # Install conda (optional, use pip instead)
   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   !pip install -r temp/requirements.txt
   ```

5. **Mount Google Drive (to save results)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # Save checkpoints to Drive
   checkpoint_dir = '/content/drive/MyDrive/Orochi/checkpoints'
   ```

6. **Run experiments**
   ```python
   %cd temp/experiments/2D/scripts/SR
   !python finetune_UniFMIR_SR.py
   ```

**Colab Pro:** $9.99/month
- Longer sessions (24 hours)
- Better GPUs (Tesla V100, A100)
- More RAM
- Priority access

---

### Option 2: Kaggle Notebooks (FREE - Good alternative)

**Pros:**
- ✓ Completely free
- ✓ 30 hours/week of GPU time
- ✓ Tesla P100 GPUs
- ✓ Persistent storage (datasets)
- ✓ Can run for up to 12 hours

**Cons:**
- ✗ Weekly quota (30 hours)
- ✗ Internet access is limited
- ✗ 20GB disk limit

**Setup Steps:**

1. **Create account**
   - Visit: https://www.kaggle.com/
   - Sign up for free

2. **Create new notebook**
   - Click "Create" > "Notebook"
   - Settings > Accelerator > GPU T4 x2

3. **Clone and setup**
   ```python
   !git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
   %cd foundation-mamba-biomed
   !pip install -r temp/requirements.txt
   ```

4. **Upload datasets as Kaggle datasets**
   - Create dataset from your data
   - Attach to notebook
   - Access at `/kaggle/input/your-dataset/`

---

### Option 3: AWS EC2 (PAID - For serious work)

**Best for:** Long training runs, production work

**Costs:**
- g4dn.xlarge: ~$0.50/hour (Tesla T4)
- p3.2xlarge: ~$3.00/hour (Tesla V100)
- p4d.24xlarge: ~$32/hour (A100, for large-scale)

**Setup Steps:**

1. **Create AWS account**
   - Visit: https://aws.amazon.com/
   - Sign up (requires credit card)

2. **Launch EC2 instance**
   ```bash
   # Choose Deep Learning AMI (Ubuntu)
   # Instance type: g4dn.xlarge (cheapest GPU option)
   # Storage: At least 100GB
   # Security group: Allow SSH (port 22)
   ```

3. **Connect via SSH**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

4. **Setup environment**
   ```bash
   # Clone repository
   git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
   cd foundation-mamba-biomed

   # Create environment
   conda env create -f temp/environment.yaml
   conda activate mamba_biomed
   pip install -r temp/requirements.txt
   ```

5. **Run experiments**
   ```bash
   # Use tmux or screen to keep running after disconnect
   tmux new -s training
   python temp/experiments/2D/scripts/SR/finetune_UniFMIR_SR.py

   # Detach: Ctrl+B, then D
   # Re-attach later: tmux attach -t training
   ```

6. **Important: Stop instance when done!**
   - Go to EC2 console
   - Stop (not terminate) instance to save costs
   - You're charged only when instance is running

**Cost Saving Tips:**
- Use Spot Instances (up to 70% cheaper)
- Stop instance when not in use
- Use S3 for dataset storage (cheaper than EBS)
- Set up billing alerts

---

### Option 4: Google Cloud Platform (PAID)

**Similar to AWS, competitive pricing**

**Setup:**

1. **Create GCP account**
   - Visit: https://cloud.google.com/
   - $300 free credit for new users

2. **Create VM with GPU**
   ```bash
   gcloud compute instances create orochi-gpu \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --maintenance-policy=TERMINATE \
     --boot-disk-size=100GB
   ```

3. **SSH and setup**
   ```bash
   gcloud compute ssh orochi-gpu
   # Then follow same setup as AWS
   ```

---

### Option 5: Lambda Labs (PAID - ML-focused)

**Best for:** Machine learning workloads, competitive pricing

**Costs:**
- 1x A100: ~$1.10/hour
- 1x RTX 6000: ~$0.50/hour

**Setup:**

1. Visit: https://lambdalabs.com/
2. Create account
3. Launch instance with PyTorch
4. SSH and setup Orochi

**Advantages:**
- Designed for ML workloads
- Often cheaper than AWS/GCP
- Simple interface
- Good for academic users

---

### Option 6: Paperspace Gradient (PAID)

**Good for:** Jupyter-based workflows

**Free tier:** Limited free GPU hours
**Paid:** Starting at $0.45/hour

1. Visit: https://www.paperspace.com/
2. Create notebook with GPU
3. Clone and run Orochi

---

## 🎓 University/Institutional Resources

### Check if you have access to:

1. **University GPU Clusters**
   - Most universities have GPU clusters for research
   - Often free for students
   - Contact: Research Computing Center, HPC team, IT department

2. **Research Lab Resources**
   - Ask your advisor/lab if they have GPU servers
   - Many labs have shared GPU workstations

3. **Cloud Credits for Education**
   - AWS Educate
   - Google Cloud for Education
   - Azure for Students
   - Often provide $100-$300 in free credits

---

## 📊 Comparison Table

| Platform | Cost | GPU Options | Session Time | Best For |
|----------|------|-------------|--------------|----------|
| **Google Colab Free** | Free | T4 | 12 hours | Testing, learning |
| **Colab Pro** | $10/month | V100, A100 | 24 hours | Regular use |
| **Kaggle** | Free | P100 | 12 hours | Competitions, small projects |
| **AWS EC2** | $0.50-$3/hr | T4, V100, A100 | Unlimited | Production, long training |
| **GCP** | $0.50-$3/hr | T4, V100, A100 | Unlimited | Production |
| **Lambda Labs** | $0.50-$1/hr | RTX 6000, A100 | Unlimited | ML workloads |
| **Paperspace** | $0.45/hr | Various | Unlimited | Jupyter workflows |

---

## 💡 Recommended Workflow

### For Beginners (Free)

1. Start with **Google Colab** (free)
2. Test the code and small experiments
3. If you need more time, try **Kaggle** (30 hrs/week free)
4. Save results to Google Drive

### For Students

1. Check for **university GPU resources** (often free)
2. Apply for **educational cloud credits**
3. Use **Colab/Kaggle** for quick tests
4. Use university cluster for long training

### For Researchers

1. Use **AWS/GCP** with spot instances
2. Or **Lambda Labs** for good ML pricing
3. Set up proper experiment tracking (wandb)
4. Use S3/Cloud Storage for datasets
5. Stop instances when not in use!

---

## 🔧 Alternative: Docker Container

If you have access to a GPU machine but don't want to install everything:

```bash
# Pull pre-built PyTorch container
docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Run container
docker run --gpus all -it -v $(pwd):/workspace pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Inside container:
cd /workspace
pip install -r temp/requirements.txt
```

---

## ❌ What WON'T Work

1. **Running on CPU only**
   - mamba_ssm requires CUDA
   - Will not install without GPU

2. **Using AMD GPUs**
   - ROCm support not available for mamba_ssm
   - Need NVIDIA CUDA

3. **Using Apple Silicon (M1/M2)**
   - MPS not supported by mamba_ssm
   - Need CUDA

4. **Using WSL without GPU passthrough**
   - Need WSL2 with CUDA support
   - Requires Windows 11 and proper setup

5. **Using different architecture**
   - Orochi is built specifically on Mamba
   - Replacing architecture defeats the purpose

---

## 📝 Quick Start with Colab

Here's a complete Colab notebook template:

```python
# 1. Check GPU
!nvidia-smi

# 2. Clone repo
!git clone https://github.com/jnjnnjzch/foundation-mamba-biomed.git
%cd foundation-mamba-biomed

# 3. Install dependencies
!pip install -q torch torchvision torchaudio
!pip install -q -r temp/requirements.txt

# 4. Mount Drive (for saving results)
from google.colab import drive
drive.mount('/content/drive')

# 5. Run quick test
!python quick_test.py

# 6. Download checkpoint (example)
!mkdir -p checkpoints/2D
!huggingface-cli download eternalaudrey/mamba-fm-2d-ckpt --local-dir checkpoints/2D

# 7. Run experiment (modify paths as needed)
%cd temp/experiments/2D/scripts/SR
!python finetune_UniFMIR_SR.py

# 8. Save results to Drive
!cp -r results /content/drive/MyDrive/Orochi/
```

---

## 🎯 Bottom Line

**If you don't have an NVIDIA GPU:**
1. You CANNOT run Orochi locally
2. Start with **Google Colab** (free, easy)
3. For serious work, use **AWS/GCP/Lambda Labs** (paid but powerful)
4. Check your **university resources** (often free for students)

**Questions?** Check TROUBLESHOOTING.md or the compatibility check script.
