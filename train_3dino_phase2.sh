#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/MambaSplit_Runs/3dino_finetune/logs/%x_%j.log
#SBATCH --error=/group/jug/aman/MambaSplit_Runs/3dino_finetune/logs/%x_%j.err

# Multi-node setup: adjust --nnodes, --nproc_per_node as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

#SBATCH --partition=dgx
#SBATCH --gres=gpu:35gb:4

#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=3dino-finetune-phase2
#SBATCH --time=72:00:00

# Set consistent naming for logs/checkpoints
source ~/.bashrc
cd /home/aman.kukde/MambaSplit/Orochi-Versatile-Biomedical-Image-Processor/

# Activate your conda env
conda activate /home/aman.kukde/conda/envs/mamba_biomed

export EXP_NAME="3dino_finetune_phase2_$(date +%Y%m%d_%H%M%S)"
export LOGDIR="/group/jug/aman/MambaSplit_Runs/3dino_finetune/logs/${EXP_NAME}"
export CHECKPOINTDIR="/group/jug/aman/MambaSplit_Runs/3dino_finetune/checkpoints/${EXP_NAME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directories
mkdir -p "${LOGDIR}"
mkdir -p "${CHECKPOINTDIR}"
mkdir -p "/group/jug/aman/MambaSplit_Runs/3dino_finetune/outputs/${EXP_NAME}"

echo "================================================================"
echo "3DINO-ViT Phase 2: Encoder Finetuning"
echo "================================================================"
echo "Encoder: TRAINABLE"
echo "Decoders: FROZEN"
echo "Bottleneck: TRAINABLE"
echo "================================================================"

# IMPORTANT: Update this path with your phase 1 checkpoint!
PHASE1_CHECKPOINT="/group/jug/aman/MambaSplit_Runs/3dino_bottleneck/checkpoints/best_checkpoint.pth"

if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CHECKPOINT"
    echo "Please update PHASE1_CHECKPOINT variable in this script"
    exit 1
fi

echo "Resuming from phase 1 checkpoint: $PHASE1_CHECKPOINT"
echo ""

# torchrun auto-detects SLURM vars:
# - SLURM_NNODES → --nnodes
# - SLURM_PROCID → --rank
# - SLURM_LOCALID → --local_rank
# - SLURM_NTASKS_PER_NODE → --nproc_per_node

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=29500 \
    src/finetune_with_wandb.py \
    --model vit \
    --config configs/3dino_finetune.yaml \
    --distributed \
    --wandb_project orochi-3dino \
    --experiment_name "${EXP_NAME}" \
    --batch_size 3 \
    --gradient_accumulation_steps 8 \
    --num_epochs 100 \
    --freeze_decoders \
    --amp \
    --resume "$PHASE1_CHECKPOINT" \
    --pretrained_decoders "/home/aman.kukde/MambaSplit/Orochi-Versatile-Biomedical-Image-Processor/pretrained_checkpoints/mamba_fm_3d.pth.tar"

echo "Phase 2 complete! Checkpoints in ${CHECKPOINTDIR}"
echo "Logs in ${LOGDIR}"
