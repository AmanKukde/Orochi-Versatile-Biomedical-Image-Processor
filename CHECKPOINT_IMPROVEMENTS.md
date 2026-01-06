# Checkpoint Manager Improvements

## Overview

The finetuning script has been refactored with a robust `CheckpointManager` class that provides comprehensive checkpoint and artifact management with wandb integration.

## New Features

### 1. **Robust CheckpointManager Class** (`src/checkpoint_manager.py`)

A dedicated class for managing all checkpoint operations:

```python
from src.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir=config.checkpoint_dir,
    experiment_name=config.experiment_name,
    save_code=True,           # Save training script
    save_config=True,         # Save configuration
    max_checkpoints=5,        # Keep only 5 periodic checkpoints
    use_wandb=True           # Upload to wandb artifacts
)
```

### 2. **Automatic Best Model Tracking**

The CheckpointManager automatically tracks and saves the best model:

```python
# Automatically saves if loss improved
is_best = checkpoint_manager.save_best_if_improved(
    model, optimizer, scheduler,
    epoch, val_loss,
    metrics=metrics
)
```

**Features:**
- ✅ Tracks best validation loss
- ✅ Automatically saves when loss improves
- ✅ Uploads to wandb with "best" alias
- ✅ Prints improvement messages

### 3. **Code and Config Artifacts**

On initialization, the CheckpointManager saves:

**Code Artifacts:**
- `finetune_with_wandb.py` (training script)
- `vit_model.py` (model architecture)
- `vit_encoder.py` (encoder)
- `ours_mamba.py` (Mamba model)
- `losses.py` (loss functions)
- `utils.py` (utilities)

**Config Artifacts:**
- Complete configuration as JSON
- All hyperparameters
- Dataset settings
- Model architecture settings

**Benefits:**
- 🔁 **Reproducibility**: Exact code used for training
- 📊 **Experiment tracking**: Compare code versions
- 🐛 **Debugging**: Know exactly what code produced results

### 4. **Comprehensive Checkpoint Metadata**

Each checkpoint includes:

```python
{
    'epoch': int,
    'val_loss': float,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'timestamp': str,                    # ISO format
    'metrics': {                        # All metrics
        'train_loss': float,
        'train_mse': float,
        ...
        'val_loss': float,
        'val_psnr': float,
        ...
    },
    'metadata': {                       # Custom metadata
        'distributed': bool,
        'world_size': int,
        ...
    }
}
```

### 5. **Three Types of Checkpoints**

#### **Best Checkpoint**
- Saved when validation loss improves
- Filename: `best_epoch{XXX}_loss{X.XXXXXX}.pth`
- WandB aliases: `["latest", "best"]`
- Only one best checkpoint kept

#### **Latest Checkpoint**
- Saved every validation interval
- Filename: `latest_epoch{XXX}_loss{X.XXXXXX}.pth`
- WandB alias: `["latest"]`
- Overwrites previous latest

#### **Periodic Checkpoints**
- Saved every N validation intervals
- Filename: `periodic_epoch{XXX}_loss{X.XXXXXX}.pth`
- WandB alias: `["latest"]`
- Keeps last `max_checkpoints` (default: 5)
- Older ones automatically deleted

### 6. **Smart WandB Integration**

**Artifacts saved automatically:**

1. **Model Artifacts** (every save):
   ```
   model-{experiment_name}
   ├── checkpoint.pth
   └── metadata: {epoch, loss, metrics}
   ```

2. **Code Artifact** (once at start):
   ```
   code-{experiment_name}
   ├── finetune_with_wandb.py
   ├── vit_model.py
   ├── losses.py
   └── ... (all relevant code)
   ```

3. **Config Artifact** (once at start):
   ```
   config-{experiment_name}
   └── config.json
   ```

**Benefits:**
- 📦 All artifacts in one place
- 🔍 Easy to find specific versions
- 🔄 Download any checkpoint later
- 📈 Track checkpoint history

### 7. **Improved Logging**

**Console output:**
```
💾 Saved checkpoint: best_epoch025_loss0.001234.pth
☁️  Uploaded to wandb: model-experiment_name (latest, best)
✨ New best model! Loss: 0.001234 (prev: 0.001456)
```

**WandB logging:**
```python
# Structured metric names
{
    'train_loss': 0.0015,
    'train_mse': 0.0012,
    'train_ncc': 0.95,
    'val_loss': 0.0012,
    'val_psnr': 32.5,
    'epoch': 25,
    'best_loss': 0.0012
}
```

### 8. **Automatic Cleanup**

Old periodic checkpoints are automatically removed:

```python
# Keeps only 5 most recent periodic checkpoints
max_checkpoints=5

# When 6th checkpoint is saved:
# 1. Oldest checkpoint deleted
# 2. New checkpoint saved
# 3. Message printed: "🗑️ Removed old checkpoint: ..."
```

## Usage Examples

### Basic Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/finetune_with_wandb.py \
    --model vit \
    --config configs/vit_finetune.yaml \
    --distributed \
    --batch_size 3 \
    --gradient_accumulation_steps 8 \
    --freeze_decoders \
    --amp
```

**What happens:**
1. ✅ Code saved to wandb (once)
2. ✅ Config saved to wandb (once)
3. ✅ Best model tracked automatically
4. ✅ Latest checkpoint saved every validation
5. ✅ Periodic checkpoints every 5 validations
6. ✅ All uploaded to wandb artifacts

### Resume from Checkpoint

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/finetune_with_wandb.py \
    --model vit \
    --config configs/vit_finetune.yaml \
    --distributed \
    --resume checkpoints/best_epoch025_loss0.001234.pth
```

**What happens:**
1. ✅ Loads model, optimizer, scheduler state
2. ✅ Resumes from correct epoch
3. ✅ Maintains best loss tracking
4. ✅ Continues logging to same wandb run

### Download from WandB

```python
import wandb

# Initialize
run = wandb.init(project="your-project")

# Download best model
artifact = run.use_artifact('model-experiment_name:best')
artifact_dir = artifact.download()

# Load checkpoint
checkpoint = torch.load(f"{artifact_dir}/checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```

## Comparison: Before vs After

| Feature | Before | After |
|---------|--------|-------|
| **Best model tracking** | Manual | ✅ Automatic |
| **Code saving** | ❌ None | ✅ Automatic |
| **Config saving** | ❌ None | ✅ Automatic |
| **Checkpoint metadata** | Basic | ✅ Comprehensive |
| **WandB artifacts** | Basic | ✅ Full integration |
| **Old checkpoint cleanup** | ❌ Manual | ✅ Automatic |
| **Checkpoint types** | 2 | ✅ 3 (best/latest/periodic) |
| **Resume support** | Basic | ✅ Full with metadata |
| **Logging format** | Inconsistent | ✅ Structured |
| **Error handling** | Basic | ✅ Robust |

## Implementation Details

### File Structure

```
src/
├── finetune_with_wandb.py      # Main training script (refactored)
├── checkpoint_manager.py       # New CheckpointManager class
├── vit_model.py               # Model (unchanged)
├── losses.py                   # Losses (unchanged)
└── utils.py                    # Utils (unchanged)

checkpoints/
├── best_epoch025_loss0.001234.pth
├── latest_epoch030_loss0.001456.pth
├── periodic_epoch020_loss0.001567.pth
├── periodic_epoch015_loss0.001678.pth
└── config.json
```

### WandB Artifacts Structure

```
wandb artifacts
├── model-experiment_name:latest
│   └── checkpoint files
├── model-experiment_name:best
│   └── best checkpoint
├── code-experiment_name:v0
│   └── all code files
└── config-experiment_name:v0
    └── config.json
```

## Benefits Summary

1. **Reproducibility** 🔁
   - Exact code and config saved
   - Can reproduce any experiment

2. **Reliability** 💪
   - Best model always saved
   - Automatic cleanup prevents disk overflow
   - Robust error handling

3. **Convenience** ⚡
   - Automatic tracking
   - Easy resume
   - Download from anywhere

4. **Organization** 📁
   - Structured artifacts
   - Clear naming
   - Metadata included

5. **Debugging** 🐛
   - Full checkpoint history
   - All metrics saved
   - Code version tracking

## Migration Guide

### For existing users:

**No code changes needed!** The refactored script is **100% backward compatible**.

**Optional improvements:**
- Old checkpoint files still work with `--resume`
- WandB artifacts are opt-in (enabled by default)
- Can disable code/config saving if needed

**To disable artifact saving:**
```python
checkpoint_manager = CheckpointManager(
    ...,
    save_code=False,
    save_config=False
)
```

## Troubleshooting

### Issue: "Code artifact not saving"

**Solution:** Ensure wandb is initialized before creating CheckpointManager:
```python
if WANDB_AVAILABLE:
    wandb.init(...)
checkpoint_manager = CheckpointManager(...)  # After wandb.init
```

### Issue: "Too many checkpoints"

**Solution:** Adjust `max_checkpoints` parameter:
```python
checkpoint_manager = CheckpointManager(
    ...,
    max_checkpoints=3  # Keep only 3 periodic checkpoints
)
```

### Issue: "Checkpoint not uploaded to wandb"

**Solution:** Check wandb is enabled:
```python
checkpoint_manager = CheckpointManager(
    ...,
    use_wandb=True  # Must be True
)
```

## Future Improvements

Potential enhancements:

1. **Multi-version code tracking**: Save code diffs between runs
2. **Checkpoint compression**: Reduce artifact size
3. **Automatic testing**: Run validation suite on checkpoints
4. **Checkpoint comparison**: Compare metrics across checkpoints
5. **Smart cleanup**: Keep checkpoints based on metrics, not just time

## References

- **CheckpointManager class**: `src/checkpoint_manager.py`
- **Refactored training script**: `src/finetune_with_wandb.py`
- **WandB artifacts docs**: https://docs.wandb.ai/guides/artifacts
- **Original script backup**: `src/finetune_with_wandb.py.backup`
