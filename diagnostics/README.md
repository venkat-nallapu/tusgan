# TUS-GAN — Diagnostic & Gradient Analysis Suite (`diagnostics/`)

This directory contains standalone diagnostic scripts designed to measure layer-wise gradient flow, calculate per-channel output gradient norms, and verify the mathematical validity of the **Dead-Gradient Theorem** across the 2x2 factorial experimental matrix (Table 6 & Figure 8).

---

## Tool Overview

| Script | Purpose | Output |
| :--- | :--- | :--- |
| [`run_advanced_gradient_analysis.py`](file:///home/venkat/projects/tusgan/diagnostics/gradient_analysis/run_advanced_gradient_analysis.py) | Full 2x2 factorial backward pass analysis across adult and child profiles | Prints gradient norm tables, generates 4-panel comparison, saves `.pt` diagnostics |
| [`run_minimal_gradient_analysis.py`](file:///home/venkat/projects/tusgan/diagnostics/gradient_analysis/run_minimal_gradient_analysis.py) | Lightweight, fast CPU warm-up gradient flow diagnostic | Quick console gradient flow verification |
| [`make_figure8_TRC.py`](file:///home/venkat/projects/tusgan/diagnostics/gradient_analysis/make_figure8_TRC.py) | Renders the publication Figure 8 per-channel gradient bar plot | Vector PDF (`figure8_TRC_formatted.pdf`) & high-res PNG |
| [`make_annotated_guide.py`](file:///home/venkat/projects/tusgan/diagnostics/gradient_analysis/make_annotated_guide.py) | Generates side-by-side pedagogical breakdown of Cell A vs. Cell C | Annotated diagnostic figure highlighting the $600\times$ learning signal |
| [`fix_guide.py`](file:///home/venkat/projects/tusgan/diagnostics/gradient_analysis/fix_guide.py) | Unicode character sanitizer for matplotlib PDF exports | Sanitizes scripts for font-compatible export |

---

## How to Run
```bash
# 1. Run the comprehensive gradient analysis
python run_advanced_gradient_analysis.py

# 2. Render Figure 8 for Elsevier TRC submission
python make_figure8_TRC.py
```

> [!NOTE]
> Root symlinks are maintained at the repository root for all diagnostic scripts.
