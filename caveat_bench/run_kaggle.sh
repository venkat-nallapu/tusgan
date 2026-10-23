#!/usr/bin/env bash
# ==============================================================================
# run_kaggle.sh — End-to-End Automated 250-Epoch Benchmark Runner for Kaggle
# ==============================================================================
# Usage on Kaggle (in a single notebook cell or terminal):
#   !bash run_kaggle.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=============================================================================="
echo "🚀 CAVEAT / ActVAE 250-EPOCH BENCHMARK — KAGGLE AUTOMATED RUNNER"
echo "=============================================================================="
echo "Start Time : $(date)"
echo "Directory  : ${SCRIPT_DIR}"
echo "=============================================================================="

# 1. GPU Check
echo "🔍 Checking GPU configuration..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
fi
echo "Detected GPUs: ${NUM_GPUS}"

# 2. Dependency Installation
echo ""
echo "📦 Installing Caveat, ActEval & dependencies..."
pip install -q acteval pytorch-lightning pot

if [ ! -d "caveat" ]; then
    echo "Cloning official Caveat framework..."
    git clone https://github.com/big-ucl/caveat.git
fi
pip install -q -e caveat --no-deps
echo "✓ Caveat package ready."

# 3. Verify Dataset Cohort
echo ""
echo "📂 Verifying ITUS 50,000-schedule benchmark cohort..."
for f in "data/real_train_schedules.csv" "data/real_test_schedules.csv" "data/synth_cell_a_schedules.csv"; do
    if [ ! -f "$f" ]; then
        echo "❌ Missing required file: $f"
        exit 1
    fi
done
echo "✓ All dataset files verified."

# 4. Train Unconstrained ActVAE Baseline (250 Epochs Parity)
echo ""
echo "=============================================================================="
echo "▶ STEP 1/4: Training Unconstrained ActVAE Baseline (250 Epochs, patience=50)..."
echo "=============================================================================="
python train_actvae.py --config configs/itus_cvae_unconstrained.yaml

# 5. Train Soft-Constrained ActVAE Baseline (250 Epochs Parity, lambda=10.0)
echo ""
echo "=============================================================================="
echo "▶ STEP 2/4: Training Soft-Constrained ActVAE Baseline (250 Epochs, lambda=10.0)..."
echo "=============================================================================="
python train_actvae.py --config configs/itus_cvae_soft.yaml

# 6. Run Official ActEval Multi-Domain Metric Suite (Table 8)
echo ""
echo "=============================================================================="
echo "▶ STEP 3/4: Running Official ActEval Multi-Domain Evaluation Suite..."
echo "=============================================================================="
python run_caveat_evaluation.py --data_dir data --output_dir outputs

# 7. Render Elsevier TRC Figure 9
echo ""
echo "=============================================================================="
echo "▶ STEP 4/4: Generating Elsevier TRC-Compliant Figure 9..."
echo "=============================================================================="
python make_figure9_TRC.py --data_dir data --output_dir outputs

# 8. Package Outputs for Auto-Download & Copy to /kaggle/working/
echo ""
echo "=============================================================================="
echo "📦 Packaging Deliverables..."
echo "=============================================================================="
mkdir -p /kaggle/working 2>/dev/null || true

zip -r caveat_retrained_250epochs_outputs.zip outputs/ -x "outputs/actvae_logs/**/checkpoints/**"
cp caveat_retrained_250epochs_outputs.zip /kaggle/working/ 2>/dev/null || true
cp outputs/table8_caveat_evaluation.csv /kaggle/working/ 2>/dev/null || true
cp outputs/figure9_diurnal_participation.pdf /kaggle/working/ 2>/dev/null || true
cp outputs/figure9_diurnal_participation.png /kaggle/working/ 2>/dev/null || true

echo "✓ Files copied to /kaggle/working/ for automatic Kaggle output retrieval."

# 9. Cloud Direct Download Backup Link
echo ""
echo "--- Cloud Direct Download Backup Link ---"
curl -s -F "file=@caveat_retrained_250epochs_outputs.zip" https://0x0.st || true

echo ""
echo "=============================================================================="
echo "🎉 ALL 250-EPOCH BENCHMARK TASKS COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "=============================================================================="
