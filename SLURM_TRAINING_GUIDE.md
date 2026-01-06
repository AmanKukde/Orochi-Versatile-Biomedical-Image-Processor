# SLURM Training Guide

Quick reference for submitting training jobs on SLURM clusters.

## Available Training Scripts

| Script | Purpose | Duration | Freeze Settings |
|--------|---------|----------|-----------------|
| `train.sh` | ViT encoder training | 36 hours | Decoders frozen |
| `train_mamba.sh` | Mamba encoder training | 36 hours | None |
| `train_3dino_phase1.sh` | 3DINO bottleneck training | 36 hours | Encoder + Decoders frozen |
| `train_3dino_phase2.sh` | 3DINO encoder finetuning | 72 hours | Decoders frozen |

---

## Quick Start

### Train ViT Encoder

```bash
sbatch train.sh
```

### Train Mamba Encoder

```bash
sbatch train_mamba.sh
```

### Train 3DINO (Two-Phase)

**Phase 1: Train bottleneck only**
```bash
# 1. Update pretrained_encoder_path in configs/3dino_bottleneck.yaml
# 2. Submit job
sbatch train_3dino_phase1.sh
```

**Phase 2: Finetune encoder**
```bash
# 1. Wait for phase 1 to complete
# 2. Update PHASE1_CHECKPOINT in train_3dino_phase2.sh
# 3. Submit job
sbatch train_3dino_phase2.sh
```

---

## Before Running

### 1. Update Paths in Scripts

Each script has paths specific to your environment. Update these:

**In all scripts:**
- `cd /home/aman.kukde/...` → Your working directory
- `conda activate /home/aman.kukde/conda/envs/...` → Your conda environment
- `/group/jug/aman/...` → Your output directory

**In train_3dino_phase2.sh:**
- `PHASE1_CHECKPOINT=...` → Path to phase 1 checkpoint

### 2. Update Config Files

**For 3DINO training**, update these config files:

**configs/3dino_bottleneck.yaml:**
```yaml
pretrained_encoder_path: "/path/to/3dino/checkpoint.pth"  # UPDATE THIS
```

**configs/3dino_finetune.yaml:**
```yaml
pretrained_encoder_path: "/path/to/3dino/checkpoint.pth"  # UPDATE THIS
resume_checkpoint: "./checkpoints/3dino_bottleneck/best_checkpoint.pth"  # UPDATE AFTER PHASE 1
```

---

## Job Management

### Submit Job

```bash
sbatch train_3dino_phase1.sh
```

### Check Job Status

```bash
squeue -u $USER
```

### Cancel Job

```bash
scancel <job_id>
```

### View Logs (Live)

```bash
tail -f /group/jug/aman/MambaSplit_Runs/3dino_bottleneck/logs/*.log
```

### View Errors

```bash
tail -f /group/jug/aman/MambaSplit_Runs/3dino_bottleneck/logs/*.err
```

---

## Resource Configuration

All scripts use:
- **Nodes**: 1
- **GPUs**: 4 × 35GB
- **CPUs**: 4 per task
- **Memory**: 32GB
- **Partition**: dgx

To modify resources, edit the `#SBATCH` directives in the script:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:35gb:4
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
```

---

## Training Parameters

### Default Settings

All scripts use:
- **Batch size**: 3 per GPU
- **Gradient accumulation**: 8 steps
- **Effective batch size**: 3 × 8 × 4 GPUs = 96
- **AMP**: Enabled (automatic mixed precision)
- **Distributed**: Enabled (DDP)

### Modify Training Parameters

To change training parameters, edit the `torchrun` command in the script:

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=29500 \
    src/finetune_with_wandb.py \
    --model vit \
    --config configs/3dino_bottleneck.yaml \
    --batch_size 3 \              # Per-GPU batch size
    --gradient_accumulation_steps 8 \  # Accumulation steps
    --num_epochs 50 \             # Number of epochs
    --freeze_encoder \            # Freeze encoder
    --freeze_decoders \           # Freeze decoders
    --amp                         # Enable AMP
```

---

## Output Directories

Each script creates organized output directories:

```
/group/jug/aman/MambaSplit_Runs/
├── 3dino_bottleneck/
│   ├── checkpoints/
│   │   └── 3dino_bottleneck_phase1_20260106_123456/
│   ├── logs/
│   │   └── 3dino-bottleneck-phase1_12345.log
│   └── outputs/
│       └── 3dino_bottleneck_phase1_20260106_123456/
├── 3dino_finetune/
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
├── vit_finetune/
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
└── mamba_finetune/
    ├── checkpoints/
    ├── logs/
    └── outputs/
```

---

## 3DINO Two-Phase Training

### Phase 1: Bottleneck Training

**Goal**: Learn dimension adaptation (384-dim → 128-dim)

**Status**:
- ❄️ Encoder: FROZEN
- ❄️ Decoders: FROZEN
- 🔥 Bottleneck: TRAINABLE

**Steps**:
1. Update `pretrained_encoder_path` in `configs/3dino_bottleneck.yaml`
2. Submit job: `sbatch train_3dino_phase1.sh`
3. Monitor: `tail -f /group/jug/aman/MambaSplit_Runs/3dino_bottleneck/logs/*.log`
4. Wait for completion (~50 epochs)
5. Note best checkpoint path

**Expected output**:
```
/group/jug/aman/MambaSplit_Runs/3dino_bottleneck/checkpoints/
└── 3dino_bottleneck_phase1_20260106_123456/
    ├── best_epoch050_loss0.001234.pth
    ├── latest_epoch050.pth
    └── periodic_epoch045.pth
```

### Phase 2: Encoder Finetuning

**Goal**: Adapt 3DINO encoder to biomedical tasks

**Status**:
- 🔥 Encoder: TRAINABLE
- ❄️ Decoders: FROZEN
- 🔥 Bottleneck: TRAINABLE

**Steps**:
1. Update `PHASE1_CHECKPOINT` in `train_3dino_phase2.sh`:
   ```bash
   PHASE1_CHECKPOINT="/group/jug/aman/.../best_epoch050_loss0.001234.pth"
   ```
2. Update `resume_checkpoint` in `configs/3dino_finetune.yaml` (optional)
3. Submit job: `sbatch train_3dino_phase2.sh`
4. Monitor: `tail -f /group/jug/aman/MambaSplit_Runs/3dino_finetune/logs/*.log`
5. Wait for completion (~100 epochs)

---

## Troubleshooting

### Error: "Batch script contains DOS line breaks"

**Cause**: Script has Windows line endings (CRLF)

**Solution**: Convert to Unix line endings (LF)
```bash
sed -i 's/\r$//' train_3dino_phase1.sh
```

### Error: "Phase 1 checkpoint not found"

**Cause**: `PHASE1_CHECKPOINT` path in `train_3dino_phase2.sh` is incorrect

**Solution**: Update path to actual checkpoint from phase 1
```bash
# In train_3dino_phase2.sh
PHASE1_CHECKPOINT="/group/jug/aman/MambaSplit_Runs/3dino_bottleneck/checkpoints/3dino_bottleneck_phase1_20260106_123456/best_epoch050_loss0.001234.pth"
```

### Error: "pretrained_encoder_path not found"

**Cause**: Path in config file is incorrect or placeholder

**Solution**: Update config file with correct 3DINO checkpoint path
```yaml
# configs/3dino_bottleneck.yaml
pretrained_encoder_path: "/correct/path/to/3dino/checkpoint.pth"
```

### Job Pending Forever

**Cause**: Insufficient resources or wrong partition

**Check available partitions**:
```bash
sinfo
```

**Check job details**:
```bash
scontrol show job <job_id>
```

**Solution**: Adjust resource requirements or partition in script

### Out of Memory

**Cause**: Batch size too large

**Solution**: Reduce batch size or increase gradient accumulation
```bash
# In script, modify:
--batch_size 2 \                      # Reduce from 3 to 2
--gradient_accumulation_steps 12      # Increase to maintain effective batch size
```

### WandB Login Required

**Cause**: WandB not authenticated

**Solution**: Login to WandB before submitting
```bash
wandb login
```

Or disable WandB in config:
```yaml
# In config file
use_wandb: false
```

---

## Best Practices

### 1. Test on Small Dataset First

Before full training, test on subset:
```yaml
# In config file
subset: 100  # Use only 100 samples
num_epochs: 5  # Few epochs for testing
```

### 2. Monitor Training

Check logs regularly:
```bash
watch -n 10 tail -20 /group/jug/aman/MambaSplit_Runs/3dino_bottleneck/logs/*.log
```

### 3. Save Phase 1 Checkpoint Path

After phase 1 completes, save the best checkpoint path:
```bash
# Find best checkpoint
ls -lh /group/jug/aman/MambaSplit_Runs/3dino_bottleneck/checkpoints/*/best*.pth

# Copy full path for phase 2
```

### 4. Use Interactive Session for Debugging

For debugging, use interactive session:
```bash
srun --partition=dgx --gres=gpu:35gb:1 --pty bash
cd /home/aman.kukde/MambaSplit/Orochi-Versatile-Biomedical-Image-Processor/
conda activate /home/aman.kukde/conda/envs/mamba_biomed

# Run training directly
python src/finetune_with_wandb.py --config configs/3dino_bottleneck.yaml
```

### 5. Monitor GPU Usage

Check GPU memory:
```bash
watch -n 1 nvidia-smi
```

---

## Local/Interactive Training

For local training without SLURM, use `train_local.sh`:

```bash
# Make sure you're on a GPU node first
./train_local.sh vit
./train_local.sh mamba
./train_local.sh 3dino-phase1
./train_local.sh 3dino-phase2
```

This is useful for:
- Debugging
- Small experiments
- Testing config changes

---

## Summary

**For SLURM clusters**:
1. Choose appropriate script (`train_*.sh`)
2. Update paths in script
3. Update config files if needed
4. Submit: `sbatch train_*.sh`
5. Monitor: `tail -f /path/to/logs/*.log`

**For 3DINO training**:
1. Phase 1: `sbatch train_3dino_phase1.sh`
2. Wait for completion
3. Note best checkpoint path
4. Update `PHASE1_CHECKPOINT` in `train_3dino_phase2.sh`
5. Phase 2: `sbatch train_3dino_phase2.sh`

**For local/interactive**:
- Use `train_local.sh` with mode argument
