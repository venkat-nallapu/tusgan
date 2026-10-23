# -*- coding: utf-8 -*-
"""
TRC Publication Formatter & Table Generator
============================================
Converts multi-domain evaluation metrics across models into publication-ready
Markdown, LaTeX, CSV, and JSON summary tables following Elsevier TRC specifications.
"""

import json
import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def generate_markdown_summary(
    results: Dict[str, Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Builds a comprehensive, multi-section Markdown report covering all 6 TRC domains.
    """
    models = list(results.keys())

    md = []
    md.append("# TUS-GAN Comprehensive Multi-Model Evaluation Benchmark")
    md.append("## Elsevier *Transportation Research Part C: Emerging Technologies* (TRC) Standards")
    md.append("\nThis document presents the unified 6-domain benchmark evaluated on the Indian Time Use Survey (ITUS) 2019 microdata.\n")

    # Table 1: Model Metadata & Overview
    md.append("### 1. Model Architecture & Experimental Lineage")
    md.append("| Model | Factorial Cell | Backbone | Conditioning / Constraint | Checkpoint Epoch | Loaded State |")
    md.append("| :--- | :--- | :--- | :--- | :---: | :---: |")
    for m in models:
        info = results[m]["info"]
        md.append(f"| **{m}** | {info.get('cell', 'N/A')} | {info.get('backbone', 'N/A')} | {info.get('constraint', 'N/A')} | {info.get('checkpoint_epoch', 'N/A')} | `{info.get('state_key_loaded', 'N/A')}` |")
    md.append("\n")

    # Table 2: Domain 1 - Statistical & Aggregate Distributional Realism
    md.append("### 2. Domain 1: Statistical & Aggregate Distributional Realism")
    md.append("| Model | Macro JSD (9×48) ↓ | Max Slot JSD ↓ | Min Slot JSD ↓ | Participation TVD ↓ | Mean Abs Duration Error (min/day) ↓ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        d1 = results[m]["domain1"]
        md.append(
            f"| **{m}** | **{d1['macro_jsd']:.6f}** | {d1['max_slot_jsd']:.6f} | {d1['min_slot_jsd']:.6f} | "
            f"{d1['participation_tvd']:.4f} | {d1['mean_absolute_duration_bias_minutes']:.2f} |"
        )
    md.append("\n")

    # Table 3: Domain 2 - Temporal Pacing & Episode Dynamics
    md.append("### 3. Domain 2: Temporal Pacing & Episode Dynamics")
    md.append("| Model | Slice 0 F-Norm ↓ | Slice 1 F-Norm ↓ | Slice 2 F-Norm ↓ | Slice 3 F-Norm ↓ | Mean Slice F-Norm ↓ | Mean Spell EMD (min) ↓ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        d2 = results[m]["domain2"]
        fn = d2["slice_fnorms"]
        md.append(
            f"| **{m}** | {fn['slice_0_morning']:.4f} | {fn['slice_1_afternoon']:.4f} | "
            f"{fn['slice_2_evening']:.4f} | {fn['slice_3_night']:.4f} | "
            f"**{fn['mean_slice_fnorm']:.4f}** | **{d2['mean_spell_emd_minutes']:.2f}** |"
        )
    md.append("\n")

    # Table 4: Domain 3 - Sequence Structure & Tour Formation
    md.append("### 4. Domain 3: Sequence Structure & Tour Formation")
    md.append("| Model | Real Mean Episodes | Fake Mean Episodes | Episode Error Δμ ↓ | Real Switching Freq | Fake Switching Freq | Freq Error Δμ ↓ | Work Start EMD (min) ↓ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        d3 = results[m]["domain3"]
        ep = d3["episodes"]
        sw = d3["switching"]
        wa = d3["work_anchors"]
        md.append(
            f"| **{m}** | {ep['real_mean_episodes']:.2f} | {ep['fake_mean_episodes']:.2f} | "
            f"{ep['episode_diff']:.2f} | {sw['real_mean_switches']:.2f} | {sw['fake_mean_switches']:.2f} | "
            f"{sw['switch_diff']:.2f} | **{wa['work_start_emd_minutes']:.1f}** |"
        )
    md.append("\n")

    # Table 5: Domain 4 - Normative & Statutory Feasibility
    md.append("### 5. Domain 4: Normative & Statutory Feasibility")
    md.append("| Model | 24h Time Budget Feasibility | Child Labor Raw Rate (%) ↓ | Child Labor Delivered Rate (%) | Discard Overhead (%) ↓ | Under-6h Sleep (%) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        d4 = results[m]["domain4"]
        cl = d4["child_labor"]
        sl = d4["sleep"]
        md.append(
            f"| **{m}** | {d4['budget']['feasibility_rate']:.1f}% | "
            f"**{cl['child_labor_raw_rate']:.2f}%** | **{cl['child_labor_delivered_rate']:.2f}%** | "
            f"{cl['rejection_discard_rate']:.2f}% | {sl['under_6h_sleep_pct']:.2f}% |"
        )
    md.append("\n")

    # Table 6: Domain 5 - Demographic & Subgroup Equity Parity
    md.append("### 6. Domain 5: Demographic & Subgroup Equity Parity")
    md.append("| Model | Real Dom Ratio (F/M) | Fake Dom Ratio (F/M) | Dom Ratio Error ↓ | Female JSD ↓ | Male JSD ↓ | Rural JSD ↓ | Urban JSD ↓ |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        d5 = results[m]["domain5"]
        gp = d5["gender_parity"]
        sp = d5["spatial_parity"]
        md.append(
            f"| **{m}** | {gp['real_domestic_disparity_ratio']:.2f}× | {gp['fake_domestic_disparity_ratio']:.2f}× | "
            f"**{gp['domestic_ratio_abs_error']:.3f}** | {gp['female_conditional_jsd']:.6f} | "
            f"{gp['male_conditional_jsd']:.6f} | {sp['rural_conditional_jsd']:.6f} | {sp['urban_conditional_jsd']:.6f} |"
        )
    md.append("\n")

    # Table 7: Domain 6 - ML Indistinguishability & Privacy
    md.append("### 7. Domain 6: ML Indistinguishability & Privacy")
    md.append("| Model | RF ROC-AUC (Optimal=0.500) ↓ | RF Accuracy (Optimal=50.0%) ↓ | Exact Match Count | Exact Match Rate (%) | Mean Min Hamming (slots) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m in models:
        d6 = results[m]["domain6"]
        rf = d6["adversarial_rf"]
        pr = d6["privacy"]
        md.append(
            f"| **{m}** | **{rf['rf_roc_auc']:.4f}** | {rf['rf_accuracy']:.2f}% | "
            f"{pr['exact_match_count']} | **{pr['exact_match_rate_pct']:.4f}%** | {pr['mean_min_hamming_distance_slots']:.2f} |"
        )
    md.append("\n")

    report_text = "\n".join(md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def export_tabular_summary(
    results: Dict[str, Dict[str, Any]],
    csv_path: str,
    json_path: str,
):
    """
    Exports unified flat tabular CSV and raw nested JSON for downstream analysis.
    """
    rows = []
    for m, data in results.items():
        row = {
            "model": m,
            "backbone": data["info"]["backbone"],
            "constraint": data["info"]["constraint"],
            "cell": data["info"]["cell"],
            # Domain 1
            "d1_macro_jsd": data["domain1"]["macro_jsd"],
            "d1_max_slot_jsd": data["domain1"]["max_slot_jsd"],
            "d1_participation_tvd": data["domain1"]["participation_tvd"],
            "d1_mean_abs_dur_bias_min": data["domain1"]["mean_absolute_duration_bias_minutes"],
            # Domain 2
            "d2_mean_slice_fnorm": data["domain2"]["slice_fnorms"]["mean_slice_fnorm"],
            "d2_mean_spell_emd_min": data["domain2"]["mean_spell_emd_minutes"],
            "d2_emp_spell_emd_min": data["domain2"]["spell_emds_minutes"].get("Employment & Related", 0.0),
            "d2_sleep_spell_emd_min": data["domain2"]["spell_emds_minutes"].get("Self-Care & Maintenance", 0.0),
            # Domain 3
            "d3_mean_episodes": data["domain3"]["episodes"]["fake_mean_episodes"],
            "d3_episode_diff": data["domain3"]["episodes"]["episode_diff"],
            "d3_switching_diff": data["domain3"]["switching"]["switch_diff"],
            "d3_work_start_emd_min": data["domain3"]["work_anchors"]["work_start_emd_minutes"],
            # Domain 4
            "d4_feasibility_rate": data["domain4"]["budget"]["feasibility_rate"],
            "d4_child_labor_raw_rate": data["domain4"]["child_labor"]["child_labor_raw_rate"],
            "d4_child_labor_delivered_rate": data["domain4"]["child_labor"]["child_labor_delivered_rate"],
            "d4_rejection_discard_rate": data["domain4"]["child_labor"]["rejection_discard_rate"],
            "d4_under_6h_sleep_pct": data["domain4"]["sleep"]["under_6h_sleep_pct"],
            # Domain 5
            "d5_fake_dom_ratio": data["domain5"]["gender_parity"]["fake_domestic_disparity_ratio"],
            "d5_dom_ratio_error": data["domain5"]["gender_parity"]["domestic_ratio_abs_error"],
            "d5_female_jsd": data["domain5"]["gender_parity"]["female_conditional_jsd"],
            "d5_male_jsd": data["domain5"]["gender_parity"]["male_conditional_jsd"],
            "d5_rural_jsd": data["domain5"]["spatial_parity"]["rural_conditional_jsd"],
            "d5_urban_jsd": data["domain5"]["spatial_parity"]["urban_conditional_jsd"],
            # Domain 6
            "d6_rf_roc_auc": data["domain6"]["adversarial_rf"]["rf_roc_auc"],
            "d6_rf_accuracy": data["domain6"]["adversarial_rf"]["rf_accuracy"],
            "d6_exact_match_rate_pct": data["domain6"]["privacy"]["exact_match_rate_pct"],
            "d6_mean_min_hamming_slots": data["domain6"]["privacy"]["mean_min_hamming_distance_slots"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    def json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return str(obj)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_serializer)
