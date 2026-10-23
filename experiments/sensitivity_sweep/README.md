# Section 6.8: Soft Penalty ($\lambda_{\text{rule}}$) Sensitivity Sweep

> **Target Publication**: Elsevier *Transportation Research Part C: Emerging Technologies*  
> **Experiment Focus**: Multi-objective sensitivity sweep across three orders of magnitude ($\lambda_{\text{rule}} \in \{0.1, 1.0, 10.0, 100.0\}$), proving that the soft constraint mechanism exhibits a broad stability plateau without knife-edge sensitivity to hyperparameter tuning.

---

## 1. Quick Start

### A. Instant Evaluation & Figure Generation (Zero GPU Required, < 5 seconds)
If precomputed checkpoints or benchmark JSON records exist, generate the publication 4-panel Figure 10 and summary tables instantly:
```bash
bash run_sweep.sh eval
```
Outputs are written to `outputs/` and mirrored to `../../publish/figures/figure10_sensitivity_sweep/`.

### B. Fast End-to-End Pipeline Smoke Test (< 30 seconds)
Verify data loading, 1 training epoch on a subset, checkpoint serialization, evaluation, and Figure 10 rendering:
```bash
bash run_sweep.sh test
```

### C. Interactive Jupyter Exploration
Launch Jupyter and open [`sensitivity_sweep.ipynb`](sensitivity_sweep.ipynb) for step-by-step interactive inspection.

---

## 2. Hardware Requirements & Safety Guards

| Environment | Training Time (per $\lambda$) | Sweep Time (3 models) | Recommendation |
|:------------|:------------------------------|:----------------------|:---------------|
| **NVIDIA GPU (A100 / RTX 3090 / T4)** | ~35–50 minutes | ~2–2.5 hours | **Recommended for full retraining** |
| **CPU Only (e.g. laptop)** | ~14–18 hours | >40 hours | ⚠️ **Use `eval` or `test` mode only** |

> [!CAUTION]
> **CPU Safety Guard**: By default, `run_sweep.sh all` and `run_sweep.sh single` will detect if an NVIDIA GPU is missing and **fail-fast** to prevent accidental CPU saturation. To explicitly force CPU training if intended, pass the `--force-cpu` flag:
> ```bash
> bash run_sweep.sh all --force-cpu
> ```

---

## 3. Directory Structure & File Inventory

```
experiments/sensitivity_sweep/
├── README.md                      # THIS FILE — Developer experience & execution guide
├── run_sweep.sh                   # Main entry point with CLI help, safety guards & smoke testing
├── train_sweep.py                 # Parameterized Cell A trainer with isolated checkpoints
├── evaluate_sweep.py              # Table 9 metrics evaluator (JSD, violations, RF accuracy, discard)
├── make_figure10_TRC.py           # Master 4-panel Elsevier TRC Figure 10 generator
├── sensitivity_sweep.ipynb        # Interactive Jupyter execution and visual validation notebook
├── results_sensitivity_sweep.csv  # Audited evaluation metrics (CSV)
├── results_sensitivity_sweep.json # Audited evaluation metrics (JSON)
├── outputs/                       # Generated Figure 10 (PDF & 300 DPI PNG)
└── test_outputs/                  # Scratch directory for smoke tests and diagnostics
```

---

## 4. Script Usage & CLI Reference

### `run_sweep.sh`
The primary orchestration script:
```bash
bash run_sweep.sh [command] [options]

Commands:
  eval                 Evaluate existing checkpoints and render Figure 10 (Fast, CPU-safe)
  test                 Run quick end-to-end smoke test (1 epoch, 512 samples, ~15s)
  single <lambda>      Train and evaluate a single lambda value (e.g. 0.1, 1.0, 100.0)
  all                  Train all sweep values (0.1, 1.0, 100.0) and render Figure 10
  help | -h | --help   Display command help

Safety Flags:
  --force-cpu          Bypass GPU check and allow training on CPU
```

### `train_sweep.py`
Trains a specific $\lambda_{\text{rule}}$ value:
```bash
python3 train_sweep.py \
    --data ../../data/tusgan_encode.npz \
    --lambda_rule 1.0 \
    --epochs 250 \
    --batch 512 \
    --output_dir checkpoints/lambda_1.0
```

### `evaluate_sweep.py`
Computes the 5 core Table 9 metrics for a given checkpoint:
```bash
python3 evaluate_sweep.py \
    --checkpoint checkpoints/lambda_1.0/final.pt \
    --lambda_rule 1.0 \
    --data ../../data/tusgan_encode.npz \
    --n_samples 50000
```

### `make_figure10_TRC.py`
Renders the Elsevier TRC-compliant 4-panel master exhibit:
```bash
python3 make_figure10_TRC.py [--output_dir outputs] [--json results_sensitivity_sweep.json] [--no_mirror]
```

---

## 5. Benchmark Results Summary (Table 9)

| $\lambda_{\text{rule}}$ | Raw Violations | Delivered Violations | Macro JSD ($\times 10^{-4}$) | RF Accuracy | RF AUC | Discard Rate | Gradient Ratio $\rho_{\text{ch0}}$ |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **0.1** | 2.46% | 0.00% | 0.81 | 65.12% | 0.7190 | 2.45% | 0.0071 |
| **1.0** | 0.82% | 0.00% | 0.82 | 65.28% | 0.7210 | 0.82% | 0.0709 |
| **10.0 (Cell A 🏆)** | **0.30%** | **0.00%** | **0.83** | **65.47%** | **0.7225** | **0.30%** | **0.7094** |
| **100.0** | 0.04% | 0.00% | 1.89 | 68.15% | 0.7480 | 0.04% | 7.0941 |

### Key Scientific Insights
1. **Stability Plateau**: $\lambda_{\text{rule}} \in [1.0, 10.0]$ maintains sub-$1.0 \times 10^{-4}$ Macro JSD while reducing raw violations by up to 92%.
2. **Mechanistic Parity ($\rho = 1.0$)**: Channel 0 gradient ratio crosses parity at $\lambda^* \approx 14.1$. For $\lambda \le 10.0$, the adversarial loss dominates ($\rho < 1$). Beyond $14.1$, the rule penalty overpowers the adversarial signal by $>7\times$ at $\lambda=100.0$, explaining the $2.3\times$ JSD degradation.
3. **Zero-Sum Budget Conservation**: Suppressed child work ($-4.6$ min/day) reallocates into **Education (+5.5 min/day)** and **Sleep/Self-Care (+5.9 min/day)** with zero distortion to adult schedules ($<1.7\%$).
