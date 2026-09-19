#!/usr/bin/env python3
"""
TUS-GAN — Figure 8: Per-Channel Output Gradient Norms
Formatted per Transportation Research Part C (Elsevier) guidelines.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Elsevier / Transportation Research Part C rcParams ──────
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            8,
    "axes.labelsize":       9,
    "axes.titlesize":       9,
    "xtick.labelsize":      7,
    "ytick.labelsize":      7,
    "legend.fontsize":      7,
    "axes.linewidth":       0.6,
    "grid.linewidth":       0.3,
    "grid.alpha":           0.4,
    "lines.linewidth":      1.0,
    "lines.markersize":     4,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
    "legend.frameon":       True,
    "legend.framealpha":    1.0,
    "legend.edgecolor":     "0.8",
    "legend.fancybox":      False,
    "legend.borderpad":     0.4,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
})

# ── Column widths (Elsevier standard) ──────────────────────
FULL_WIDTH = 7.48  # 190 mm in inches

# ── Color palette (original combo) ─────────────────────────
COL_ADULT = "#1f77b4"   # blue
COL_SOFT  = "#2ca02c"   # green
COL_HARD  = "#d62728"   # red

# ── No hatching (user preference) ──────────────────────────
HATCH_ADULT = ""
HATCH_CHILD = ""

# ── Data from the advanced gradient analysis run ───────────
RESULTS_FILE = "results/gradient_diagnostics_advanced.pt"
ACTIVITY_NAMES = ["Empl.", "Sleep", "Eat", "Dom.", "Trav.",
                  "Leis.", "Educ.", "Self.", "Other"]

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

data = torch.load(RESULTS_FILE, weights_only=False)

panel_keys = [
    ("Cell_A_v4",  "Convolutional + Soft penalty"),
    ("Cell_B_v5",  "Transformer + Hard masking"),
    ("Cell_C_v41", "Convolutional + Hard masking"),
    ("Cell_D_v51", "Transformer + Soft penalty"),
]

fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.5))

for idx, (cname, subtitle) in enumerate(panel_keys):
    ax = axes[idx // 2][idx % 2]
    cell = data[cname]
    enforcement = cell["enforcement"]

    channels = sorted(cell["adult_channels"].keys())
    n_ch = len(channels)
    adult_vals = [cell["adult_channels"][c] for c in channels]
    child_vals = [max(cell["child_channels"].get(c, 1e-16), 1e-16) for c in channels]

    x = np.arange(n_ch)
    w = 0.35

    # Adult bars
    ax.bar(x - w / 2, adult_vals, w,
           color=COL_ADULT, edgecolor="black", linewidth=0.4,
           hatch=HATCH_ADULT, label="Unconstrained (adult)", zorder=3)

    # Child bars
    child_color = COL_SOFT if enforcement == "Soft Penalty" else COL_HARD
    child_label = "Soft penalty (child)" if enforcement == "Soft Penalty" else "Hard masking (child)"
    ax.bar(x + w / 2, child_vals, w,
           color=child_color, edgecolor="black", linewidth=0.4,
           hatch=HATCH_CHILD, label=child_label, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIVITY_NAMES, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", zorder=0)

    # Panel label (a), (b), (c), (d) — NO set_title
    panel_letter = chr(ord("a") + idx)
    ax.text(-0.12, 1.05, f"({panel_letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")

    # Thin subtitle below panel label (acceptable in multi-panel figures)
    ax.text(0.50, 1.05, subtitle, transform=ax.transAxes,
            fontsize=7, va="top", ha="center", color="0.35")

    if idx % 2 == 0:
        ax.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")

    ax.legend(loc="upper right")

    # Mark Channel 0 with a thin vertical span (no pastel box)
    ax.axvspan(-0.5, 0.5, color="0.90", zorder=0)

plt.tight_layout()

# Save as vector PDF (preferred) and high-res PNG
pdf_path = os.path.join(OUT_DIR, "figure8_TRC_formatted.pdf")
png_path = os.path.join(OUT_DIR, "figure8_TRC_formatted.png")
plt.savefig(pdf_path, format="pdf")
plt.savefig(png_path, dpi=300)
plt.close()

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
