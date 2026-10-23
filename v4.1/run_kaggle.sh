#!/usr/bin/env bash
# ==============================================================================
# TUS-GAN v4.1 (Cell C: CNN + Hard Masking) — Kaggle Dual GPU Execution Script
# ==============================================================================
# Automatically detects available GPUs and launches:
# 1. Pre-flight smoke tests (including dead-gradient verification)
# 2. Dual-GPU training via torchrun DDP (or single GPU fallback)
# 3. Post-training ultimate evaluation (Table 6 metrics & diagnostic plots)
# ==============================================================================

set -e

EPOCHS=250
BATCH_SIZE=512
DATA_PATH="tusgan_encode.npz"
CKPT_DIR="checkpoints"
LOG_DIR="runs/tusgan_v4_1_hard"
EVAL_DIR="evaluation_results"
SEED=42

echo "==============================================================================="
echo "🚀 TUS-GAN v4.1 (Cell C: CNN + Hard Masking) — Kaggle Training Runner"
echo "==============================================================================="
echo "Date       : $(date)"
echo "Data Path  : $DATA_PATH"
echo "Epochs     : $EPOCHS"
echo "Batch Size : $BATCH_SIZE"
echo "Seed       : $SEED"
echo "==============================================================================="

# 1. GPU Check
echo "🔍 Checking GPU Configuration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "⚠️ nvidia-smi not found. Checking PyTorch device count..."
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "Detected GPUs: $NUM_GPUS"

# 2. Locate Dataset
if [ ! -f "$DATA_PATH" ]; then
    if [ -f "data/tusgan_encode.npz" ]; then
        DATA_PATH="data/tusgan_encode.npz"
    elif [ -f "../data/tusgan_encode.npz" ]; then
        DATA_PATH="../data/tusgan_encode.npz"
    fi
fi
echo "Using Dataset: $DATA_PATH"

# 3. Run Pre-Flight Smoke Test
echo ""
echo "🧪 Running Pre-Flight Smoke Test..."
python smoke_test.py

# 4. Launch Training
echo ""
echo "==============================================================================="
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "▶️ Launching DDP Multi-GPU Training on $NUM_GPUS GPUs via torchrun..."
    echo "==============================================================================="
    torchrun --nproc_per_node=2 train.py \
        --data "$DATA_PATH" \
        --epochs "$EPOCHS" \
        --batch "$BATCH_SIZE" \
        --seed "$SEED" \
        --checkpoint_dir "$CKPT_DIR" \
        --log_dir "$LOG_DIR"
else
    echo "▶️ Launching Single-GPU Training via python train.py..."
    echo "==============================================================================="
    python train.py \
        --data "$DATA_PATH" \
        --epochs "$EPOCHS" \
        --batch "$BATCH_SIZE" \
        --seed "$SEED" \
        --checkpoint_dir "$CKPT_DIR" \
        --log_dir "$LOG_DIR"
fi

echo "✅ Training successfully finished!"

# 5. Run Ultimate Evaluation
echo ""
echo "==============================================================================="
echo "📊 Running Ultimate Evaluation on Final Checkpoint..."
echo "==============================================================================="
python evaluate_ultimate.py \
    --checkpoint "$CKPT_DIR/final.pt" \
    --data "$DATA_PATH" \
    --n_samples 30000 \
    --output_dir "$EVAL_DIR"

# 6. Package Deliverables
echo ""
echo "📦 Packaging results and checkpoints into archive..."
zip -r v4_1_kaggle_results.zip "$CKPT_DIR/final.pt" "$CKPT_DIR/grad_diagnostics_*.pt" "$EVAL_DIR" "$LOG_DIR" 2>/dev/null || true

echo "==============================================================================="
echo "🎉 Pipeline Completed Successfully!"
echo "   • Final Model    : $CKPT_DIR/final.pt"
echo "   • Metrics & Plots: $EVAL_DIR/"
echo "   • Download Bundle: v4_1_kaggle_results.zip"
echo "==============================================================================="
