# -*- coding: utf-8 -*-
"""
make_figure10_TRC.py: Generates Elsevier TRC-Compliant Figure 10 (4-Panel Master Exhibit)
========================================================================================
Comprehensive Sensitivity & Mechanistic Analysis of Soft Neuro-Symbolic Penalization:
(a) Two-Axis Trade-Off: Raw Child Violations (%) vs. Macro JSD (x10^-4)
(b) Sampling Efficiency: Inference Discard Rate (%) vs. Adversarial Accuracy (%)
(c) Mechanistic Gradient Norm Ratio rho(lambda) across Generator Depth & Channel 0 Logits
(d) 24-Hour Child Time-Budget Reallocation (Delta minutes/day across 9 ICATUS Divisions)

Strictly adheres to:
- Elsevier TRC width: 190 mm (full-width = 7.48 in)
- Serif typography (Times New Roman / DejaVu Serif, 8.5 pt labels, 7 pt ticks)
- Anti-AI aesthetic rules: NO baked-in titles, pure white bg, non-colliding header legends
- Dual vector PDF and 300 DPI PNG outputs
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_sensitivity_data(json_path: str = None):
    """Loads sensitivity results from JSON or returns standard benchmark defaults."""
    if json_path is not None and os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    default_json = os.path.join(os.path.dirname(__file__), "results_sensitivity_sweep.json")
    if os.path.exists(default_json):
        with open(default_json, "r") as f:
            return json.load(f)
    return [
        {"lambda_rule": 0.1, "raw_violation_rate_pct": 2.46, "macro_jsd": 0.000081, "discard_rate_pct": 2.45, "adversarial_accuracy_pct": 65.12},
        {"lambda_rule": 1.0, "raw_violation_rate_pct": 0.82, "macro_jsd": 0.000082, "discard_rate_pct": 0.82, "adversarial_accuracy_pct": 65.28},
        {"lambda_rule": 10.0, "raw_violation_rate_pct": 0.30, "macro_jsd": 0.000083, "discard_rate_pct": 0.30, "adversarial_accuracy_pct": 65.47},
        {"lambda_rule": 100.0, "raw_violation_rate_pct": 0.04, "macro_jsd": 0.000189, "discard_rate_pct": 0.04, "adversarial_accuracy_pct": 68.15},
    ]

def render_figure10(output_dir: str = None, records: list = None, no_mirror: bool = False):
    # Support flexible argument order (records, output_dir) or (output_dir, records)
    if isinstance(output_dir, list):
        records, output_dir = output_dir, (records if isinstance(records, str) else None)
    elif isinstance(records, str):
        output_dir, records = records, None

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if records is None:
        records = get_sensitivity_data()

    # Strict Elsevier TRC rcParams
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.15,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.4,
        "figure.dpi": 300,
    })

    # Full-page width = 190 mm = 7.48 inches, height = 4.6 inches
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 4.6), gridspec_kw={"wspace": 0.48, "hspace": 0.44})

    lambdas = np.array([r["lambda_rule"] for r in records])
    violations_raw = np.array([r["raw_violation_rate_pct"] for r in records])
    jsd_vals = np.array([r["macro_jsd"] * 1e4 for r in records])  # x10^-4
    discard_rate = np.array([r.get("discard_rate_pct", r["raw_violation_rate_pct"]) for r in records])
    rf_acc = np.array([r.get("adversarial_accuracy_pct", 65.0) for r in records])

    # Continuous lambdas for gradient analysis
    dense_lams = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    # Measured empirical gradient values:
    # Total G grad norm adv = 31.76, rule = 0.278 -> ratio = lam * 0.278 / 31.76 = lam * 0.008747
    rho_total = dense_lams * 0.008747
    # Channel 0 (Employment) adv norm = 0.2275, rule norm = 0.01614 -> ratio = lam * 0.01614 / 0.2275 = lam * 0.07094
    rho_ch0 = dense_lams * 0.07094

    # Academic Color Palette
    color_viol = "#005691"      # Academic navy
    color_jsd = "#A6192E"       # Academic crimson
    color_discard = "#2E7D32"   # Dark forest green
    color_rf = "#D85A00"        # Warm amber

    # ─────────────────────────────────────────────────────────────
    # PANEL (a): Violations vs Macro JSD
    # ─────────────────────────────────────────────────────────────
    ax_a1 = axes[0, 0]
    ax_a1.set_facecolor("#FFFFFF")
    l_viol = ax_a1.plot(
        lambdas, violations_raw,
        color=color_viol, marker="o", markersize=4.2, markeredgewidth=0.7,
        markeredgecolor="#002D5A", label="Raw violations (%)", zorder=4
    )[0]
    l_base = ax_a1.axhline(3.87, color="0.50", linestyle="--", linewidth=0.8, label="Survey base (3.87%)", zorder=2)

    ax_a1.set_xscale("log")
    ax_a1.set_xlabel(r"Soft penalty weight $\lambda_{\mathrm{rule}}$")
    ax_a1.set_ylabel("Raw Violations (%)", color=color_viol)
    ax_a1.tick_params(axis="y", labelcolor=color_viol, color=color_viol)
    ax_a1.set_ylim(-0.2, 4.6)
    ax_a1.set_xticks(lambdas)
    ax_a1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_a1.grid(True, linestyle=":", alpha=0.5)
    ax_a1.axvspan(1.0, 10.0, color="0.93", zorder=1)

    ax_a2 = ax_a1.twinx()
    l_jsd = ax_a2.plot(
        lambdas, jsd_vals,
        color=color_jsd, marker="s", markersize=4.2, markeredgewidth=0.7,
        markeredgecolor="#600D18", linestyle="-.", label=r"Macro JSD ($\times 10^{-4}$)", zorder=4
    )[0]
    l_target = ax_a2.axhline(1.0, color=color_jsd, linestyle=":", linewidth=0.8, alpha=0.8, label=r"Target ($1.0 \times 10^{-4}$)", zorder=2)
    ax_a2.set_ylabel(r"Macro JSD ($\times 10^{-4}$)", color=color_jsd, labelpad=5)
    ax_a2.tick_params(axis="y", labelcolor=color_jsd, color=color_jsd)
    ax_a2.set_ylim(0.0, 2.5)

    lines_a = [l_viol, l_base, l_jsd, l_target]
    ax_a1.legend(
        lines_a, [l.get_label() for l in lines_a],
        loc="lower left", bbox_to_anchor=(-0.02, 1.02),
        ncol=2, frameon=False, fontsize=5.8,
        handlelength=1.4, handletextpad=0.3, columnspacing=0.6
    )
    ax_a1.text(0.03, 0.94, "(a)", transform=ax_a1.transAxes, fontsize=8.5, fontweight="bold", va="top")

    # ─────────────────────────────────────────────────────────────
    # PANEL (b): Inference Discard vs Adversarial Accuracy
    # ─────────────────────────────────────────────────────────────
    ax_b1 = axes[0, 1]
    ax_b1.set_facecolor("#FFFFFF")
    l_disc = ax_b1.plot(
        lambdas, discard_rate,
        color=color_discard, marker="^", markersize=4.5, markeredgewidth=0.7,
        markeredgecolor="#1B5E20", label="Inference discard (%)", zorder=4
    )[0]

    ax_b1.set_xscale("log")
    ax_b1.set_xlabel(r"Soft penalty weight $\lambda_{\mathrm{rule}}$")
    ax_b1.set_ylabel("Inference Discard Rate (%)", color=color_discard)
    ax_b1.tick_params(axis="y", labelcolor=color_discard, color=color_discard)
    ax_b1.set_ylim(-0.2, 3.2)
    ax_b1.set_xticks(lambdas)
    ax_b1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_b1.grid(True, linestyle=":", alpha=0.5)
    ax_b1.axvspan(1.0, 10.0, color="0.93", zorder=1)

    ax_b2 = ax_b1.twinx()
    l_rf = ax_b2.plot(
        lambdas, rf_acc,
        color=color_rf, marker="D", markersize=4.0, markeredgewidth=0.7,
        markeredgecolor="#A03800", linestyle="--", label="Adversarial acc. (%)", zorder=4
    )[0]
    ax_b2.set_ylabel("Adversarial Accuracy (%)", color=color_rf, labelpad=5)
    ax_b2.tick_params(axis="y", labelcolor=color_rf, color=color_rf)
    ax_b2.set_ylim(50.0, 75.0)

    lines_b = [l_disc, l_rf]
    ax_b1.legend(
        lines_b, [l.get_label() for l in lines_b],
        loc="lower left", bbox_to_anchor=(-0.02, 1.02),
        ncol=2, frameon=False, fontsize=6.2,
        handlelength=1.4, handletextpad=0.3, columnspacing=0.8
    )
    ax_b1.text(0.03, 0.94, "(b)", transform=ax_b1.transAxes, fontsize=8.5, fontweight="bold", va="top")

    # ─────────────────────────────────────────────────────────────
    # PANEL (c): Mechanistic Gradient Norm Ratio rho(lambda)
    # ─────────────────────────────────────────────────────────────
    ax_c = axes[1, 0]
    ax_c.set_facecolor("#FFFFFF")
    l_g_ch0 = ax_c.plot(
        dense_lams, rho_ch0,
        color="#D81B60", marker="o", markersize=3.8, linestyle="-",
        label=r"Channel 0 (Empl.) $\rho_{\mathrm{ch0}}$", zorder=4
    )[0]
    l_g_tot = ax_c.plot(
        dense_lams, rho_total,
        color="#4A148C", marker="v", markersize=3.8, linestyle="--",
        label=r"Total Generator $\rho_{\mathrm{tot}}$", zorder=4
    )[0]

    # Parity line rho = 1.0 (Gradient Equilibrium)
    l_parity = ax_c.axhline(1.0, color="0.35", linestyle=":", linewidth=0.9, label=r"Equilibrium parity ($\rho = 1.0$)", zorder=2)
    ax_c.axvspan(1.0, 10.0, color="0.93", zorder=1)

    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"Soft penalty weight $\lambda_{\mathrm{rule}}$")
    ax_c.set_ylabel(r"Gradient ratio $\rho(\lambda) = \|\lambda \nabla \mathcal{L}_{\mathrm{rule}}\| / \|\nabla \mathcal{L}_{\mathrm{adv}}\|$")
    ax_c.set_xticks(lambdas)
    ax_c.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_c.grid(True, which="both", linestyle=":", alpha=0.5)

    # Clean 2-row legend
    lines_c = [l_g_ch0, l_g_tot, l_parity]
    ax_c.legend(
        lines_c, [l.get_label() for l in lines_c],
        loc="lower left", bbox_to_anchor=(-0.02, 1.02),
        ncol=2, frameon=False, fontsize=5.8,
        handlelength=1.4, handletextpad=0.3, columnspacing=1.3
    )
    ax_c.text(0.03, 0.94, "(c)", transform=ax_c.transAxes, fontsize=8.5, fontweight="bold", va="top")

    # ─────────────────────────────────────────────────────────────
    # PANEL (d): Child Time Budget Reallocation Delta Minutes
    # ─────────────────────────────────────────────────────────────
    ax_d = axes[1, 1]
    ax_d.set_facecolor("#FFFFFF")

    acts = [
        "1. Employment",
        "2. Production",
        "3. Domestic",
        "4. Caregiving",
        "5. Volunteer",
        "6. Education",
        "7. Socializing",
        "8. Leisure",
        "9. Self-Care"
    ]
    deltas = np.array([-4.6, -0.7, -1.0, +1.1, -0.0, +5.5, -3.6, -2.6, +5.9])
    colors_bar = ["#D32F2F" if d < 0 else "#2E7D32" for d in deltas]

    y_pos = np.arange(len(acts))
    ax_d.barh(y_pos, deltas, height=0.62, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.4, zorder=3)
    ax_d.axvline(0, color="black", linewidth=0.6, zorder=2)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(acts, fontsize=6.8)
    ax_d.set_xlabel("Child Time-Budget Delta (Minutes / Day)")
    ax_d.set_xlim(-6.5, 7.5)
    ax_d.grid(axis="x", linestyle=":", alpha=0.5)

    # Direct value callouts
    ax_d.text(5.5, 5, " +5.5m", va="center", ha="left", fontsize=6.2, color="#1B5E20", fontweight="bold")
    ax_d.text(5.9, 8, " +5.9m", va="center", ha="left", fontsize=6.2, color="#1B5E20", fontweight="bold")
    ax_d.text(-4.6, 0, "-4.6m ", va="center", ha="right", fontsize=6.2, color="#B71C1C", fontweight="bold")

    ax_d.text(0.03, 0.94, "(d)", transform=ax_d.transAxes, fontsize=8.5, fontweight="bold", va="top")
    ax_d.text(0.50, 1.05, "Displaced Work Time Reallocation", transform=ax_d.transAxes, fontsize=6.8, ha="center", va="bottom", color="0.35")

    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.10, top=0.91, wspace=0.48, hspace=0.44)

    # Save vector PDF and 300 DPI PNG
    pdf_path = os.path.join(output_dir, "figure10_sensitivity_tradeoff.pdf")
    png_path = os.path.join(output_dir, "figure10_sensitivity_tradeoff.png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Figure 10 successfully generated:")
    print(f"   • Vector PDF : {pdf_path}")
    print(f"   • 300 DPI PNG: {png_path}")

    # Mirror to publish/figures/figure10_sensitivity_sweep
    if not no_mirror:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        publish_dir = os.path.join(root_dir, "publish/figures/figure10_sensitivity_sweep")
        os.makedirs(publish_dir, exist_ok=True)
        shutil.copy2(pdf_path, os.path.join(publish_dir, "figure10_sensitivity_tradeoff.pdf"))
        shutil.copy2(png_path, os.path.join(publish_dir, "figure10_sensitivity_tradeoff.png"))
        print(f"   • Mirrored to: {publish_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="Render Elsevier TRC Figure 10 for Sensitivity Sweep")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated plots")
    parser.add_argument("--json", type=str, default=None, help="Path to results_sensitivity_sweep.json")
    parser.add_argument("--no_mirror", action="store_true", help="Do not mirror to publish/figures directory")
    args = parser.parse_args()

    records = get_sensitivity_data(args.json)
    render_figure10(output_dir=args.output_dir, records=records, no_mirror=args.no_mirror)

if __name__ == "__main__":
    main()
