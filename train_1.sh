#!/bin/bash
#SBATCH --job-name=vit-finetune-ddp
#SBATCH --partition=dgx

# =========================
# Resource configuration
# =========================
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # One task per MIG GPU
#SBATCH --gpus-per-task=1            # Enforce 1 GPU per task
#SBATCH --gres=gpu:35gb:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --time=36:00:00

# =========================
# Logging
# =========================
#SBATCH --output=/group/jug/aman/MambaSplit_Runs/vit_finetune/logs/%x_%j.out
#SBATCH --error=/group/jug/aman/MambaSplit_Runs/vit_finetune/logs/%x_%j.err
#SBATCH --mail-type=NONE

# =========================
# Environment setup
# =========================
source ~/.bashrc
conda activate /home/aman.kukde/conda/envs/mamba_biomed

cd /home/aman.kukde/MambaSplit/Orochi-Versatile-Biomedical-Image-Processor/

# =========================
# Experiment naming
# =========================
export EXP_NAME="vit_finetune_$(date +%Y%m%d_%H%M%S)"
export LOGDIR="/group/jug/aman/MambaSplit_Runs/vit_finetune/logs/${EXP_NAME}"
export CHECKPOINTDIR="/group/jug/aman/MambaSplit_Runs/vit_finetune/checkpoints/${EXP_NAME}"
export OUTPUTDIR="/group/jug/aman/MambaSplit_Runs/vit_finetune/outputs/${EXP_NAME}"

mkdir -p "${LOGDIR}" "${CHECKPOINTDIR}" "${OUTPUTDIR}"

# =========================
# DDP / NCCL / Performance
# =========================
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_DEVICE_MAX_CONNECTIONS=1

export TOKENIZERS_PARALLELISM=false
export WANDB_START_METHOD=thread

# =========================
# (Optional) PyTorch flags
# =========================
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================
# Launch training
# =========================
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/finetune_with_wandb.py \
    --model vit \
    --config configs/vit_finetune.yaml \
    --distributed \
    --wandb_project orochi-vit \
    --experiment_name "${EXP_NAME}" \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 100 \
    --freeze_decoders \
    --amp \
    --pretrained_decoders "/home/aman.kukde/MambaSplit/Orochi-Versatile-Biomedical-Image-Processor/pretrained_checkpoints/mamba_fm_3d.pth.tar"

echo "========================================"
echo " Training complete"
echo " Logs:        ${LOGDIR}"
echo " Checkpoints: ${CHECKPOINTDIR}"
echo " Outputs:     ${OUTPUTDIR}"
echo "========================================"

