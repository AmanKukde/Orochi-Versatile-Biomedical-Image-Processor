#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/MambaSplit_Runs/vit_finetune/logs/%x_%j.log
#SBATCH --error=/group/jug/aman/MambaSplit_Runs/vit_finetune/logs/%x_%j.err

# Multi-node setup: adjust --nnodes, --nproc_per_node as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

#SBATCH --partition=dgx
#SBATCH --gres=gpu:18gb:4

#SBATCH --mem=32GB
#SBATCH --cpus-per-task=12
#SBATCH --job-name=vit-finetune-ddp
#SBATCH --time=36:00:00

# Set consistent naming for logs/checkpoints
source ~/.bashrc
cd /home/aman.kukde/MambaSplit/Orochi-Versatile-Biomedical-Image-Processor/ # UPDATE: path to your finetune_with_wandb.py

# Activate your conda env (UPDATE: replace 'msr' with your env name if different)
conda activate /home/aman.kukde/conda/envs/mamba_biomed

export EXP_NAME="vit_finetune_$(date +%Y%m%d_%H%M%S)"
export LOGDIR="/group/jug/aman/MambaSplit_Runs/vit_finetune/logs/${EXP_NAME}"
export CHECKPOINTDIR="/group/jug/aman/MambaSplit_Runs/vit_finetune/checkpoints/${EXP_NAME}"

# Create output directories
mkdir -p "${LOGDIR}"
mkdir -p "${CHECKPOINTDIR}"
mkdir -p "/group/jug/aman/MambaSplit_Runs/vit_finetune/outputs/${EXP_NAME}"


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
    --config configs/vit_finetune.yaml \
    --distributed \
    --wandb_project orochi-vit \
    --experiment_name "${EXP_NAME}" \
    --batch_size 3 \
    --num_epochs 100 \
    --freeze_decoders \
    --pretrained_decoders "/home/aman.kukde/MambaSplit/Orochi-Versatile-Biomedical-Image-Processor/pretrained_checkpoints/mamba_fm_3d.pth.tar"

echo "Training complete! Checkpoints in ${CHECKPOINTDIR}"
echo "Logs in ${LOGDIR}"
§