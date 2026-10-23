#!/usr/bin/env bash
# ==============================================================================
# run_sweep.sh: Automated Runner for Section 6.8 Sensitivity Sweep
# ==============================================================================
# Usage:
#   bash run_sweep.sh [all|single] [lambda_value]
# Examples:
#   bash run_sweep.sh all            # Trains 0.1, 1.0, 100.0 and evaluates all 4
#   bash run_sweep.sh single 0.1     # Trains lambda=0.1 and evaluates
#   bash run_sweep.sh eval           # Only evaluates existing checkpoints & plots
# ==============================================================================

set -e

MODE=${1:-"help"}
TARGET_LAMBDA=${2:-"0.1"}
FORCE_CPU=${3:-""}
[ "$2" = "--force-cpu" ] && FORCE_CPU="--force-cpu"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_PATH="$ROOT_DIR/data/tusgan_encode.npz"

if [ ! -f "$DATA_PATH" ]; then
    if [ -f "$ROOT_DIR/v4/tusgan_encode.npz" ]; then
        DATA_PATH="$ROOT_DIR/v4/tusgan_encode.npz"
    elif [ -f "tusgan_encode.npz" ]; then
        DATA_PATH="tusgan_encode.npz"
    fi
fi

EPOCHS=250
BATCH_SIZE=512

show_help() {
    echo "==============================================================================="
    echo "🔬 TUS-GAN Section 6.8 Sensitivity Sweep Automation"
    echo "==============================================================================="
    echo "Usage:"
    echo "  bash run_sweep.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  eval                    Evaluate existing checkpoints and render Figure 10 (Fast, CPU-safe)"
    echo "  test                    Run quick end-to-end smoke test (1 epoch, 512 samples, ~15s)"
    echo "  single <lambda>         Train and evaluate a single lambda value (e.g. 0.1, 1.0, 100.0)"
    echo "  all                     Train all sweep values (0.1, 1.0, 100.0) and render Figure 10"
    echo "  help | -h | --help      Show this help documentation"
    echo ""
    echo "Safety Flags:"
    echo "  --force-cpu             Allow full training execution on CPU without NVIDIA GPU"
    echo "==============================================================================="
}

if [ "$MODE" = "help" ] || [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    show_help
    exit 0
fi

echo "==============================================================================="
echo "🔬 TUS-GAN Section 6.8 Sensitivity Sweep Automation"
echo "==============================================================================="
echo "Mode      : $MODE"
echo "Data Path : $DATA_PATH"
echo "Directory : $SCRIPT_DIR"
echo "Date      : $(date)"
echo "GPU Status:"
HAS_GPU=0
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    HAS_GPU=1
else
    echo "   (Running on CPU / No NVIDIA GPU detected)"
fi
echo "==============================================================================="

# Safety Guard for CPU training
check_gpu_safety() {
    if [ "$HAS_GPU" -eq 0 ] && [ "$FORCE_CPU" != "--force-cpu" ]; then
        echo ""
        echo "⚠️ SAFETY WARNING: No NVIDIA GPU was detected on this system."
        echo "   Running full GAN training ($EPOCHS epochs) on CPU requires >40 hours and will"
        echo "   heavily load all CPU cores."
        echo ""
        echo "   • To render publication figures from existing results (Instant):"
        echo "       bash run_sweep.sh eval"
        echo ""
        echo "   • To run a fast 15-second pipeline smoke test:"
        echo "       bash run_sweep.sh test"
        echo ""
        echo "   • If you intentionally want to train on CPU anyway, add --force-cpu:"
        echo "       bash run_sweep.sh $MODE $TARGET_LAMBDA --force-cpu"
        echo "==============================================================================="
        exit 1
    fi
}

train_and_eval_lambda() {
    local LAMBDA=$1
    local CKPT_DIR="$SCRIPT_DIR/checkpoints/lambda_${LAMBDA}"
    local LOG_FILE="$SCRIPT_DIR/train_lambda_${LAMBDA}.log"

    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "▶️ Training Cell A with λ_rule = $LAMBDA ($EPOCHS Epochs, Batch $BATCH_SIZE)"
    echo "   Checkpoints: $CKPT_DIR"
    echo "   Log File   : $LOG_FILE"
    echo "-------------------------------------------------------------------------------"

    python3 "$SCRIPT_DIR/train_sweep.py" \
        --data "$DATA_PATH" \
        --lambda_rule "$LAMBDA" \
        --epochs "$EPOCHS" \
        --batch "$BATCH_SIZE" \
        --output_dir "$CKPT_DIR" 2>&1 | tee "$LOG_FILE"

    echo "✅ Finished training for λ = $LAMBDA."
    echo "📊 Evaluating checkpoint for Table 9..."

    python3 "$SCRIPT_DIR/evaluate_sweep.py" \
        --checkpoint "$CKPT_DIR/final.pt" \
        --lambda_rule "$LAMBDA" \
        --data "$DATA_PATH" \
        --n_samples 50000

    echo "🎉 Evaluation completed for λ = $LAMBDA."
}

if [ "$MODE" = "test" ]; then
    echo "🧪 Running fast end-to-end smoke test (~15 seconds)..."
    TEST_DIR="$SCRIPT_DIR/test_outputs/smoke_test_run"
    mkdir -p "$TEST_DIR"
    python3 "$SCRIPT_DIR/train_sweep.py" \
        --data "$DATA_PATH" \
        --lambda_rule 1.0 \
        --epochs 1 \
        --batch 256 \
        --subset 512 \
        --output_dir "$TEST_DIR"

    python3 "$SCRIPT_DIR/evaluate_sweep.py" \
        --checkpoint "$TEST_DIR/final.pt" \
        --lambda_rule 1.0 \
        --data "$DATA_PATH" \
        --n_samples 100 \
        --skip_rf \
        --output_json "$TEST_DIR/smoke_eval.json" \
        --output_csv "$TEST_DIR/smoke_eval.csv"

    python3 "$SCRIPT_DIR/make_figure10_TRC.py" \
        --output_dir "$TEST_DIR" \
        --no_mirror

    echo "✅ All smoke tests passed successfully! Pipeline verified."
elif [ "$MODE" = "single" ]; then
    check_gpu_safety
    train_and_eval_lambda "$TARGET_LAMBDA"
elif [ "$MODE" = "all" ]; then
    check_gpu_safety
    echo "🚀 Sweeping all missing lambda values (0.1, 1.0, 100.0)..."
    
    # Check if lambda=10.0 (production champion) exists, link or evaluate it
    V4_FINAL="$ROOT_DIR/v4/checkpoints/final.pt"
    if [ -f "$V4_FINAL" ]; then
        echo "Found production champion at $V4_FINAL (λ=10.0). Evaluating if needed..."
        python3 "$SCRIPT_DIR/evaluate_sweep.py" \
            --checkpoint "$V4_FINAL" \
            --lambda_rule 10.0 \
            --data "$DATA_PATH" \
            --n_samples 50000
    fi

    for LAMBDA in 0.1 1.0 100.0; do
        train_and_eval_lambda "$LAMBDA"
    done

    echo "🎨 Rendering Figure 10..."
    python3 "$SCRIPT_DIR/make_figure10_TRC.py"
    echo "🏆 Full sensitivity sweep completed successfully!"
elif [ "$MODE" = "eval" ]; then
    echo "🎨 Rendering Figure 10 and refreshing summary tables..."
    python3 "$SCRIPT_DIR/make_figure10_TRC.py"
else
    echo "❌ Unknown mode '$MODE'."
    echo ""
    show_help
    exit 1
fi
