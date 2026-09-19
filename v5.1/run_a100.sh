#!/usr/bin/env bash
# ==============================================================================
# TUS-GAN v5.1 (Cell D) - JarvisLabs NVIDIA A100-80GB Automated Execution Script
# ==============================================================================
# Usage:
#   bash run_a100.sh [single|multi]
#     single : Trains 1 run (Seed 42, 250 epochs)
#     multi  : Trains all 3 random seeds (42, 123, 999) required for Table 6
# ==============================================================================

set -e

MODE=${1:-"single"}
DATA_PATH="tusgan_encode.npz"
BATCH_SIZE=512
EPOCHS=250

echo "==============================================================================="
echo "🚀 TUS-GAN v5.1 (Cell D: Transformer + Soft Penalty) - A100 Execution"
echo "==============================================================================="
echo "Mode       : $MODE"
echo "Data Path  : $DATA_PATH"
echo "Batch Size : $BATCH_SIZE"
echo "Epochs     : $EPOCHS"
echo "Date       : $(date)"
echo "GPU Status :"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "NVIDIA SMI not available"
echo "==============================================================================="

# 1. Install dependencies if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Checking dependencies..."
    pip install -q -r requirements.txt
fi

# 2. Run Pre-Flight Smoke Test
echo "🧪 Running Pre-Flight Smoke Test..."
python smoke_test.py

run_training_and_eval() {
    local SEED=$1
    local CKPT_DIR="checkpoints_seed${SEED}"
    local LOG_FILE="train_seed${SEED}.log"
    local EVAL_DIR="evaluation_results_seed${SEED}"

    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "▶️ Starting Training: Seed $SEED ($EPOCHS Epochs, Batch $BATCH_SIZE)"
    echo "   Checkpoints: $CKPT_DIR"
    echo "   Log File   : $LOG_FILE"
    echo "-------------------------------------------------------------------------------"

    python train.py \
        --data "$DATA_PATH" \
        --epochs "$EPOCHS" \
        --batch "$BATCH_SIZE" \
        --seed "$SEED" \
        --checkpoint_dir "$CKPT_DIR" 2>&1 | tee "$LOG_FILE"

    echo "✅ Training complete for Seed $SEED."
    echo "📊 Running Evaluation for Table 6 metrics..."

    python evaluate_ultimate.py \
        --checkpoint "${CKPT_DIR}/final.pt" \
        --data "$DATA_PATH" \
        --n_samples 50000 \
        --output_dir "$EVAL_DIR"

    echo "🎉 Seed $SEED Evaluation finished. Summary in ${EVAL_DIR}/table6_cell_d_results.txt"
}

if [ "$MODE" = "single" ]; then
    run_training_and_eval 42
elif [ "$MODE" = "multi" ]; then
    echo "🔥 Running 3-Seed Factorial Protocol (Seeds 42, 123, 999) for Table 6..."
    for SEED in 42 123 999; do
        run_training_and_eval "$SEED"
    done
    echo "🏆 All 3 seeds finished! You can now compute mean ± SD for Table 6."
else
    echo "❌ Invalid mode '$MODE'. Use 'single' or 'multi'."
    exit 1
fi

echo "==============================================================================="
echo "✅ Execution completed successfully at $(date)"
echo "==============================================================================="
