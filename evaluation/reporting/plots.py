# -*- coding: utf-8 -*-
"""
TRC Publication-Grade Figure Generation Suite
=============================================
Strictly conforms to Elsevier Transportation Research Part C (TRC) visual standards:
- Typography: Times New Roman / STIXGeneral serif font family.
- Pure white background (#FFFFFF) with delicate gridlines (#E0E0E0, alpha=0.5).
- Clean unframed legends and strict black axes spines.
- Generates dual publication-ready vector PDF and 300 DPI PNG outputs.
"""

import os
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Apply strict Elsevier TRC style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#111111",
    "axes.linewidth": 0.8,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "mathtext.fontset": "stix",
})

ICATUS_NAMES = [
    "1. Employment & Related",
    "2. Production for Own Use",
    "3. Unpaid Domestic Services",
    "4. Unpaid Caregiving",
    "5. Unpaid Volunteer/Community",
    "6. Learning & Education",
    "7. Socializing & Religious",
    "8. Culture, Leisure & Sports",
    "9. Self-Care & Maintenance",
]

MODEL_COLORS = {
    "Real ITUS (Survey)": "#111111",
    "v3": "#7F7F7F",                 # Gray
    "v4": "#0055D4",                 # Strong Royal Blue (Champion 🏆)
    "v4.1": "#D9534F",               # Coral Red (Ablation C)
    "v5": "#F0AD4E",                 # Amber Orange (Ablation B)
    "v5.1": "#2CA02C",               # Forest Green (Restoration D)
    "v6": "#9467BD",                 # Muted Purple (Next-Gen)
    "actvae_unconstrained": "#8C564B",  # Chestnut Brown
    "actvae_soft": "#E377C2",           # Pinkish Purple
    "caveat_unconstrained": "#8C564B",
    "caveat_soft": "#E377C2",
}

MODEL_LINESTYLES = {
    "Real ITUS (Survey)": "-",
    "v3": ":",
    "v4": "-",
    "v4.1": "--",
    "v5": "-.",
    "v5.1": "--",
    "v6": "-",
    "actvae_unconstrained": ":",
    "actvae_soft": "-.",
    "caveat_unconstrained": ":",
    "caveat_soft": "-.",
}


def plot_diurnal_overlays(
    real_codes: np.ndarray,
    model_codes: Dict[str, np.ndarray],
    output_dir: str,
):
    """
    Figure 1: 9-panel Diurnal Activity Participation Curve Overlays across 48 half-hour slots.
    """
    N_real = len(real_codes)
    slots = np.arange(48)
    time_ticks = [0, 8, 16, 24, 32, 40, 47]
    time_labels = ["04:00", "08:00", "12:00", "16:00", "20:00", "00:00", "03:30"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for d in range(1, 10):
        ax = axes[d - 1]
        real_curve = np.mean(real_codes == d, axis=0) * 100.0

        # Plot Real Ground Truth
        ax.plot(
            slots,
            real_curve,
            color=MODEL_COLORS["Real ITUS (Survey)"],
            linestyle="-",
            linewidth=2.0,
            label="Real Survey (ITUS 2019)",
            zorder=4,
        )

        # Plot Synthetic Models
        for m_name, f_codes in model_codes.items():
            fake_curve = np.mean(f_codes == d, axis=0) * 100.0
            color = MODEL_COLORS.get(m_name, "#333333")
            ls = MODEL_LINESTYLES.get(m_name, "--")
            lw = 1.8 if m_name == "v4" else 1.2
            alpha = 0.95 if m_name == "v4" else 0.75
            if "actvae" in m_name or "caveat" in m_name:
                m_label = "ActVAE (Soft)" if "soft" in m_name else "ActVAE (Unconstrained)"
            else:
                m_label = f"TUS-GAN {m_name}"
            ax.plot(
                slots,
                fake_curve,
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=alpha,
                label=m_label,
                zorder=3 if m_name == "v4" else 2,
            )

        ax.set_title(ICATUS_NAMES[d - 1], fontsize=10, fontweight="bold", pad=5)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels)
        ax.grid(True)
        ax.set_xlim(0, 47)
        ax.set_ylabel("Participation (%)")

        if d - 1 == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=7.5)

    for i in range(6, 9):
        axes[i].set_xlabel("Time of Day (Hour:Minute)")

    plt.tight_layout()

    png_path = os.path.join(output_dir, "figure_trc_diurnal_overlays.png")
    pdf_path = os.path.join(output_dir, "figure_trc_diurnal_overlays.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_spell_duration_survival(
    real_codes: np.ndarray,
    model_codes: Dict[str, np.ndarray],
    output_dir: str,
):
    """
    Figure 2: Spell Duration Survival Step Curves S(t) = P(Duration >= t)
    for Employment (ICATUS 1), Domestic (ICATUS 3), and Sleep (ICATUS 9).
    """
    target_activities = [(1, "Employment & Related"), (3, "Unpaid Domestic Services"), (9, "Self-Care & Maintenance")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    def extract_spells(codes_arr: np.ndarray, code_val: int) -> np.ndarray:
        durations = []
        for row in codes_arr:
            padded = np.pad(row == code_val, (1, 1), mode="constant")
            diff = np.diff(padded.astype(np.int32))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lens = (ends - starts) * 30  # minutes
            durations.extend(lens)
        return np.array(durations) if len(durations) > 0 else np.array([0])

    for idx, (act_code, act_name) in enumerate(target_activities):
        ax = axes[idx]
        r_spells = extract_spells(real_codes, act_code)
        t_vals = np.linspace(0, 720, 200)

        # Real survival curve S(t)
        r_surv = [np.mean(r_spells >= t) for t in t_vals]
        ax.step(t_vals / 60.0, r_surv, color="#111111", linewidth=2.0, label="Real Survey", where="post")

        for m_name, f_codes in model_codes.items():
            f_spells = extract_spells(f_codes, act_code)
            f_surv = [np.mean(f_spells >= t) for t in t_vals]
            color = MODEL_COLORS.get(m_name, "#555555")
            ls = MODEL_LINESTYLES.get(m_name, "--")
            lw = 1.8 if m_name == "v4" else 1.2
            if "actvae" in m_name or "caveat" in m_name:
                m_label = "ActVAE (Soft)" if "soft" in m_name else "ActVAE (Unconstrained)"
            else:
                m_label = f"{m_name}"
            ax.step(t_vals / 60.0, f_surv, color=color, linestyle=ls, linewidth=lw, label=m_label, where="post")

        ax.set_title(f"Activity {act_code}: {act_name}", fontweight="bold", pad=8)
        ax.set_xlabel("Spell Duration (Hours)")
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)

        if idx == 0:
            ax.set_ylabel(r"Survival Probability $S(t) = P(T \geq t)$")
            ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    plt.tight_layout()
    png_path = os.path.join(output_dir, "figure_trc_survival_curves.png")
    pdf_path = os.path.join(output_dir, "figure_trc_survival_curves.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison_bars(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
):
    """
    Figure 3: Multi-Domain Comparative Bar Chart comparing all 6 TRC performance axes.
    """
    models = list(results.keys())
    x_labels = [
        "ActVAE\n(Uncon)" if "unconstrained" in m else
        "ActVAE\n(Soft)" if ("soft" in m and "actvae" in m) else
        m for m in models
    ]
    x = np.arange(len(models))
    width = 0.55

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Metric 1: Macro JSD (Domain 1)
    ax = axes[0, 0]
    vals = [results[m]["domain1"]["macro_jsd"] * 1e4 for m in models]
    colors = [MODEL_COLORS.get(m, "#555555") for m in models]
    ax.bar(x, vals, width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_title("Macro JSD (×10⁻⁴) ↓", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.grid(True, axis="y")

    # Metric 2: Mean Slice Transition F-Norm (Domain 2)
    ax = axes[0, 1]
    vals = [results[m]["domain2"]["slice_fnorms"]["mean_slice_fnorm"] for m in models]
    ax.bar(x, vals, width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_title("Mean 4-Slice Transition F-Norm ↓", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.grid(True, axis="y")

    # Metric 3: Mean Spell Duration EMD in Minutes (Domain 2)
    ax = axes[0, 2]
    vals = [results[m]["domain2"]["mean_spell_emd_minutes"] for m in models]
    ax.bar(x, vals, width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_title("Mean Spell Duration EMD (min) ↓", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.grid(True, axis="y")

    # Metric 4: Child Labor Raw Violation Rate (%) (Domain 4)
    ax = axes[1, 0]
    vals = [results[m]["domain4"]["child_labor"]["child_labor_raw_rate"] for m in models]
    ax.bar(x, vals, width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_title("Child Labor Violation Rate (%) ↓", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.grid(True, axis="y")

    # Metric 5: Domestic Disparity Ratio Error (Domain 5)
    ax = axes[1, 1]
    vals = [results[m]["domain5"]["gender_parity"]["domestic_ratio_abs_error"] for m in models]
    ax.bar(x, vals, width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_title("Gender Domestic Ratio Error ↓", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.grid(True, axis="y")

    # Metric 6: Random Forest Adversarial ROC-AUC (Domain 6)
    ax = axes[1, 2]
    vals = [results[m]["domain6"]["adversarial_rf"]["rf_roc_auc"] for m in models]
    ax.bar(x, vals, width, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.axhline(0.5, color="#111111", linestyle="--", linewidth=1.2, label="Ideal (0.500)")
    ax.set_title("Random Forest ROC-AUC (Ideal=0.50) ↓", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "figure_trc_model_comparison_bars.png")
    pdf_path = os.path.join(output_dir, "figure_trc_model_comparison_bars.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
