#!/bin/bash
# Training script for Orochi with multiple encoder backends
#
# This script provides convenient commands for training with different encoders:
# - ViT encoder (standard Vision Transformer)
# - Mamba encoder (state-space models)
# - 3DINO-ViT encoder (pretrained biomedical ViT)
#
# Usage:
#   ./train_local.sh vit          # Train ViT encoder
#   ./train_local.sh mamba        # Train Mamba encoder
#   ./train_local.sh 3dino-phase1 # Train 3DINO bottleneck only
#   ./train_local.sh 3dino-phase2 # Finetune 3DINO encoder

set -e  # Exit on error

# Default settings
NUM_GPUS=4
BATCH_SIZE=3
GRAD_ACCUM=8
AMP=true
DISTRIBUTED=true

# Parse command line arguments
MODE=${1:-"help"}

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

show_help() {
    cat << EOF
${GREEN}Orochi Training Script${NC}

Usage: ./train_local.sh [MODE] [OPTIONS]

${BLUE}Available Modes:${NC}
  vit              Train ViT encoder (standard Vision Transformer)
  mamba            Train Mamba encoder (state-space models)
  3dino-phase1     Train 3DINO bottleneck only (encoder + decoders frozen)
  3dino-phase2     Finetune 3DINO encoder + bottleneck (decoders frozen)
  help             Show this help message

${BLUE}Environment Variables:${NC}
  NUM_GPUS         Number of GPUs to use (default: 4)
  BATCH_SIZE       Per-GPU batch size (default: 3)
  GRAD_ACCUM       Gradient accumulation steps (default: 8)
  AMP              Use automatic mixed precision (default: true)
  DISTRIBUTED      Use distributed training (default: true)

${BLUE}Examples:${NC}
  # Train ViT with default settings
  ./train_local.sh vit

  # Train with 2 GPUs
  NUM_GPUS=2 ./train_local.sh vit

  # Train 3DINO bottleneck (phase 1)
  ./train_local.sh 3dino-phase1

  # Finetune 3DINO encoder (phase 2)
  ./train_local.sh 3dino-phase2

${BLUE}Two-Phase 3DINO Training:${NC}
  Phase 1: Train bottleneck only (encoder + decoders frozen)
    ./train_local.sh 3dino-phase1

  Phase 2: Finetune encoder + bottleneck (decoders frozen)
    Update resume_checkpoint in configs/3dino_finetune.yaml
    ./train_local.sh 3dino-phase2

${YELLOW}Note:${NC}
  - Update pretrained paths in config files before training
  - Effective batch size = BATCH_SIZE × GRAD_ACCUM × NUM_GPUS
  - Default effective batch size = 3 × 8 × 4 = 96

EOF
}

train_vit() {
    print_header "Training ViT Encoder"

    CONFIG="configs/vit_finetune.yaml"

    if [ ! -f "$CONFIG" ]; then
        print_error "Config file not found: $CONFIG"
        exit 1
    fi

    print_success "Config: $CONFIG"
    print_success "GPUs: $NUM_GPUS"
    print_success "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)))"

    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        src/finetune_with_wandb.py \
        --model vit \
        --config $CONFIG \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --freeze_decoders"

    if [ "$DISTRIBUTED" = true ]; then
        CMD="$CMD --distributed"
    fi

    if [ "$AMP" = true ]; then
        CMD="$CMD --amp"
    fi

    print_success "Starting training..."
    echo ""
    eval $CMD
}

train_mamba() {
    print_header "Training Mamba Encoder"

    CONFIG="configs/mamba_finetune.yaml"

    if [ ! -f "$CONFIG" ]; then
        print_error "Config file not found: $CONFIG"
        exit 1
    fi

    print_success "Config: $CONFIG"
    print_success "GPUs: $NUM_GPUS"
    print_success "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)))"

    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        src/finetune_with_wandb.py \
        --model mamba \
        --config $CONFIG \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM"

    if [ "$DISTRIBUTED" = true ]; then
        CMD="$CMD --distributed"
    fi

    if [ "$AMP" = true ]; then
        CMD="$CMD --amp"
    fi

    print_success "Starting training..."
    echo ""
    eval $CMD
}

train_3dino_phase1() {
    print_header "3DINO-ViT Phase 1: Bottleneck Training"
    print_warning "Encoder: FROZEN"
    print_warning "Decoders: FROZEN"
    print_success "Bottleneck: TRAINABLE"
    echo ""

    CONFIG="configs/3dino_bottleneck.yaml"

    if [ ! -f "$CONFIG" ]; then
        print_error "Config file not found: $CONFIG"
        exit 1
    fi

    # Check if pretrained paths are set
    if grep -q "/path/to/3dino/checkpoint.pth" "$CONFIG"; then
        print_error "Please update pretrained_encoder_path in $CONFIG"
        exit 1
    fi

    print_success "Config: $CONFIG"
    print_success "GPUs: $NUM_GPUS"
    print_success "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)))"

    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        src/finetune_with_wandb.py \
        --model vit \
        --config $CONFIG \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --freeze_encoder \
        --freeze_decoders"

    if [ "$DISTRIBUTED" = true ]; then
        CMD="$CMD --distributed"
    fi

    if [ "$AMP" = true ]; then
        CMD="$CMD --amp"
    fi

    print_success "Starting phase 1 training..."
    echo ""
    eval $CMD

    echo ""
    print_success "Phase 1 complete!"
    print_warning "Next: Update resume_checkpoint in configs/3dino_finetune.yaml"
    print_warning "Then run: ./train_local.sh 3dino-phase2"
}

train_3dino_phase2() {
    print_header "3DINO-ViT Phase 2: Encoder Finetuning"
    print_success "Encoder: TRAINABLE"
    print_warning "Decoders: FROZEN"
    print_success "Bottleneck: TRAINABLE"
    echo ""

    CONFIG="configs/3dino_finetune.yaml"

    if [ ! -f "$CONFIG" ]; then
        print_error "Config file not found: $CONFIG"
        exit 1
    fi

    # Check if pretrained paths are set
    if grep -q "/path/to/3dino/checkpoint.pth" "$CONFIG"; then
        print_error "Please update pretrained_encoder_path in $CONFIG"
        exit 1
    fi

    # Check if resume checkpoint is set
    if ! grep -q "resume_checkpoint:" "$CONFIG" || grep -q "UPDATE THIS PATH" "$CONFIG"; then
        print_error "Please update resume_checkpoint in $CONFIG with phase 1 checkpoint"
        exit 1
    fi

    print_success "Config: $CONFIG"
    print_success "GPUs: $NUM_GPUS"
    print_success "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)))"

    # Get resume checkpoint from config
    RESUME_CKPT=$(grep "resume_checkpoint:" "$CONFIG" | cut -d'"' -f2)

    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
        src/finetune_with_wandb.py \
        --model vit \
        --config $CONFIG \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --freeze_decoders \
        --resume $RESUME_CKPT"

    if [ "$DISTRIBUTED" = true ]; then
        CMD="$CMD --distributed"
    fi

    if [ "$AMP" = true ]; then
        CMD="$CMD --amp"
    fi

    print_success "Starting phase 2 training..."
    echo ""
    eval $CMD
}

# Main execution
case "$MODE" in
    vit)
        train_vit
        ;;
    mamba)
        train_mamba
        ;;
    3dino-phase1)
        train_3dino_phase1
        ;;
    3dino-phase2)
        train_3dino_phase2
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo ""
        show_help
        exit 1
        ;;
esac
