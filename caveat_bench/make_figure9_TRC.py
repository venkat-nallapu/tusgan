# -*- coding: utf-8 -*-
"""
make_figure9_TRC.py: Generates Elsevier TRC-compliant Figure 9 diurnal participation
curves comparing Real ITUS 2019, TUS-GAN Cell D, ActVAE Unconstrained, and ActVAE Soft.

Target: Section 6.7 of TUS_GAN_V2.docx
Formatting: 190 mm full-page width, anti-AI styling, vector PDF + 300 DPI PNG.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# 1. CONSTANTS & ACTIVITY MAPPING
# ─────────────────────────────────────────────────────────────

DIVISION_NAMES = [
    "employment",
    "production_own_use",
    "unpaid_domestic",
    "unpaid_caregiving",
    "unpaid_volunteer",
    "learning_education",
    "socializing_religious",
    "culture_leisure",
    "self_care",
]

DIVISION_TITLES = [
    "1. Employment & Related",
    "2. Production for Own Use",
    "3. Unpaid Domestic Services",
    "4. Unpaid Caregiving Services",
    "5. Volunteer & Community Work",
    "6. Learning & Education",
    "7. Socializing & Religious Practice",
    "8. Culture, Leisure & Sports",
    "9. Self-Care & Maintenance",
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]


# ─────────────────────────────────────────────────────────────
# 2. PARTICIPATION CURVE CALCULATOR
# ─────────────────────────────────────────────────────────────

def calculate_diurnal_curve(df_schedules: pd.DataFrame, num_pids: int) -> dict:
    """
    Computes participation rate in [0.0, 1.0] across 48 half-hour slots for each division.
    """
    curves = {act: np.zeros(48, dtype=float) for act in DIVISION_NAMES}
    
    # Pre-filter by activity
    for act in DIVISION_NAMES:
        sub = df_schedules[df_schedules["act"] == act]
        counts = np.zeros(48, dtype=float)
        for _, row in sub.iterrows():
            start_slot = int(np.clip(row["start"] // 30, 0, 47))
            end_slot = int(np.clip(row["end"] // 30, 0, 48))
            counts[start_slot:end_slot] += 1.0
        curves[act] = counts / max(1, num_pids)

    return curves


# ─────────────────────────────────────────────────────────────
# 3. FIGURE RENDERER (TRC FORMATTED)
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 9 Diurnal Participation Overlay")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing schedules")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for output figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    publish_dir = "/home/venkat/projects/tusgan/publish/plan"

    print("🎨 Setting up Elsevier TRC Plot Parameters...")
    # Strict Elsevier TRC styling parameters
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.1,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.5,
        "figure.dpi": 300,
    })

    # Load schedules
    model_files = {
        "Real ITUS 2019": ("real_test_schedules.csv", "#000000", "-", 1.3, 4),
        "TUS-GAN Cell D": ("synth_cell_d_schedules.csv", "#0072B2", "-", 1.2, 3),
        "ActVAE (Unconstrained)": ("synth_vae_unconstrained_schedules.csv", "#D55E00", "--", 1.0, 2),
        "ActVAE + Soft (C1)": ("synth_vae_soft_schedules.csv", "#009E73", ":", 1.2, 2),
    }

    curves_by_model = {}
    for label, (filename, color, linestyle, lw, zorder) in model_files.items():
        filepath = os.path.join(args.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            n_pids = df["pid"].nunique()
            curves_by_model[label] = {
                "curves": calculate_diurnal_curve(df, n_pids),
                "color": color,
                "linestyle": linestyle,
                "linewidth": lw,
                "zorder": zorder,
            }
            print(f"   ✓ Calculated diurnal curves for: {label} ({n_pids:,} schedules)")
        else:
            print(f"   ⚠️ File not found for {label} at {filepath}")

    if "Real ITUS 2019" not in curves_by_model:
        raise FileNotFoundError("Missing real test schedules! Run tusgan_to_caveat.py first.")

    # 190 mm width = 7.48 inches, height = 4.8 inches
    fig, axes = plt.subplots(3, 3, figsize=(7.48, 4.8), sharex=True, sharey=True)
    time_slots = np.linspace(0, 24, 48)

    for idx, (ax, act_key, title, panel_lbl) in enumerate(zip(axes.flatten(), DIVISION_NAMES, DIVISION_TITLES, PANEL_LABELS)):
        ax.set_facecolor("#FFFFFF")
        
        for label, mdata in curves_by_model.items():
            ax.plot(
                time_slots,
                mdata["curves"][act_key],
                label=label,
                color=mdata["color"],
                linestyle=mdata["linestyle"],
                linewidth=mdata["linewidth"],
                zorder=mdata["zorder"]
            )

        # Panel formatting
        ax.text(0.03, 0.90, panel_lbl, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.text(0.12, 0.90, title, transform=ax.transAxes, fontsize=7.5, va="top")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
        ax.set_xticklabels(["04:00", "08:00", "12:00", "16:00", "20:00", "00:00", "04:00"], rotation=30)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Shared axis labels
    fig.supxlabel("Time of Day (Diurnal Cycle starting 04:00 AM)", fontsize=8.5, y=0.01)
    fig.supylabel("Population Participation Rate", fontsize=8.5, x=0.01)

    # Shared legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=7.5)

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.97])

    out_pdf = os.path.join(args.output_dir, "figure9_diurnal_participation.pdf")
    out_png = os.path.join(args.output_dir, "figure9_diurnal_participation.png")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n🖼️ Successfully generated Figure 9:")
    print(f"   - Vector PDF: {out_pdf}")
    print(f"   - High-Res PNG (300 DPI): {out_png}")

    # Mirror to publish/plan
    if os.path.exists(publish_dir):
        pub_pdf = os.path.join(publish_dir, "figure9_diurnal_participation.pdf")
        pub_png = os.path.join(publish_dir, "figure9_diurnal_participation.png")
        import shutil
        shutil.copy2(out_pdf, pub_pdf)
        shutil.copy2(out_png, pub_png)
        print(f"   - Mirrored to: {pub_pdf}")


if __name__ == "__main__":
    main()
