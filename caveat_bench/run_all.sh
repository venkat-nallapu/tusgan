#!/usr/bin/env bash
# ==============================================================================
# run_all.sh — Complete Automated Pipeline for Caveat Benchmarking
# ==============================================================================
# Generates Table 8 and Figure 9 for Section 6.7 of TUS_GAN_V2.docx
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/venkat/projects/tusgan/.venv/bin/python"

echo "======================================================================"
echo "          TUS-GAN — CAVEAT & ActVAE BENCHMARKING SUITE"
echo "======================================================================"
echo "Working directory: ${SCRIPT_DIR}"
echo "Python: ${PYTHON_EXEC}"
echo ""

cd "${SCRIPT_DIR}"

# Step 1: Export Real Data and TUS-GAN Checkpoints
echo "▶ STEP 1: Exporting ITUS 2019 data and TUS-GAN checkpoints to Caveat format..."
"${PYTHON_EXEC}" tusgan_to_caveat.py \
    --npz_path ../data/tusgan_encode.npz \
    --output_dir data \
    --sample_size 50000 \
    --test_split 0.2 \
    --export_checkpoints

# Step 2: Train Unconstrained ActVAE
echo ""
echo "▶ STEP 2A: Training Unconstrained ActVAE Baseline..."
"${PYTHON_EXEC}" train_actvae.py --config configs/itus_cvae_unconstrained.yaml

# Step 2B: Train Soft-Constrained ActVAE
echo ""
echo "▶ STEP 2B: Training Soft-Constrained ActVAE Baseline (lambda_logic=10.0)..."
"${PYTHON_EXEC}" train_actvae.py --config configs/itus_cvae_soft.yaml

# Step 3: Run Full Caveat Evaluation
echo ""
echo "▶ STEP 3: Running Caveat ActEval Multi-Domain Metric Suite..."
"${PYTHON_EXEC}" run_caveat_evaluation.py --data_dir data --output_dir outputs

# Step 4: Generate Publication Figure 9
echo ""
echo "▶ STEP 4: Generating Elsevier TRC-Compliant Figure 9..."
"${PYTHON_EXEC}" make_figure9_TRC.py --data_dir data --output_dir outputs

echo ""
echo "======================================================================"
echo "✅ CAVEAT BENCHMARKING COMPLETED SUCCESSFULLY!"
echo "   - Table 8 saved to: outputs/table8_caveat_evaluation.csv"
echo "   - Figure 9 saved to: outputs/figure9_diurnal_participation.pdf & .png"
echo "======================================================================"
