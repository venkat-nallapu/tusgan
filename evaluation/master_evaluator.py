# -*- coding: utf-8 -*-
"""
TUS-GAN Master Evaluation Engine
=================================
Elsevier Transportation Research Part C (TRC) Unified Multi-Model Benchmark.
Evaluates generative time-use schedules across 6 orthogonal research domains:
1. Statistical & Aggregate Distributional Realism
2. Temporal Pacing & Episode Dynamics
3. Sequence Structure & Tour Formation
4. Normative & Statutory Feasibility
5. Demographic & Subgroup Equity Parity
6. ML Indistinguishability & Privacy

Usage:
    python evaluation/master_evaluator.py --models v3 v4 v5 v5.1 --n_samples 30000
"""

import argparse
import os
import sys
import time

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Dict, Any, List
import numpy as np
import torch

from evaluation.metrics import (
    compute_slot_jsd,
    compute_participation_tvd,
    compute_daily_minutes,
    compute_transition_matrices_and_fnorm,
    compute_spell_emds,
    compute_episode_statistics,
    compute_work_anchor_timing,
    compute_transition_entropy,
    compute_time_budget_closure,
    compute_child_labor_audit,
    compute_sleep_constraint_audit,
    compute_gender_parity,
    compute_spatial_parity,
    compute_adversarial_rf,
    compute_privacy_and_novelty,
)
from evaluation.model_adapters import load_generator, generate_synthetic_codes, MODEL_SPECS
from evaluation.reporting import (
    generate_markdown_summary,
    export_tabular_summary,
    plot_diurnal_overlays,
    plot_spell_duration_survival,
    plot_model_comparison_bars,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Master TRC 6-Domain Evaluation Suite for TUS-GAN"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/tusgan_encode.npz",
        help="Path to pre-encoded ITUS 2019 dataset (tusgan_encode.npz)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["v3", "v4", "v5", "v5.1"],
        help="List of model variants to benchmark (e.g. v3 v4 v4.1 v5 v5.1 v6)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30000,
        help="Number of synthetic schedules to generate and benchmark (default: 30000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Inference batch size (default: 512)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results/master_benchmark",
        help="Directory to save benchmark reports, tables, and figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and inference (default: 42)",
    )
    parser.add_argument(
        "--rejection_sampling",
        action="store_true",
        help="Enable post-hoc rejection sampling for child labor enforcement",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda or cpu)",
    )
    return parser.parse_args()


def load_dataset(data_path: str):
    """Loads real dataset from .npz file."""
    if not os.path.exists(data_path):
        # Fallback to local tusgan_encode.npz if present in v3 or v4
        alt_paths = [
            "data/tusgan_encode.npz",
            "v4/tusgan_encode.npz",
            "v3/tusgan_encode.npz",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                data_path = alt
                break

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset at {data_path}")

    print(f"[Dataset] Loading ITUS microdata from: {data_path} ...")
    data = np.load(data_path)
    diary_tensor = data["diary_tensor"]  # (N, 9, 48, 1)
    cond_vector = data["cond_vector"]    # (N, 82)
    district_ids = data["district_ids"]  # (N,)
    state_ids = data["state_ids"]        # (N,)

    print(f"[Dataset] Loaded {len(diary_tensor):,} survey diaries.")
    return diary_tensor, cond_vector, district_ids, state_ids


def evaluate_single_model(
    model_name: str,
    real_codes: np.ndarray,
    cond_sub: np.ndarray,
    dist_sub: np.ndarray,
    state_sub: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
    seed: int = 42,
    rejection_sampling: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a single model variant across all 6 TRC domains.
    """
    print(f"\n{'='*70}")
    print(f" BENCHMARKING MODEL: {model_name.upper()}")
    print(f"{'='*70}")

    # 1. Load Model Generator
    t0 = time.time()
    generator, info = load_generator(model_name, device=device)
    print(f"Loaded {model_name} in {time.time()-t0:.2f}s (Backbone: {info['backbone']}, State: {info['state_key_loaded']})")

    # 2. Generate Synthetic Schedules
    t_gen0 = time.time()
    if hasattr(generator, "get_samples_and_cond"):
        fake_codes, model_cond = generator.get_samples_and_cond(n_samples=len(cond_sub), seed=seed)
    else:
        fake_codes = generate_synthetic_codes(
            generator=generator,
            cond_vector=cond_sub,
            district_ids=dist_sub,
            state_ids=state_sub,
            batch_size=batch_size,
            seed=seed,
            device=device,
            rejection_sampling=rejection_sampling,
        )
        model_cond = cond_sub
    gen_time = time.time() - t_gen0
    n_samples = len(fake_codes)
    speed = n_samples / max(gen_time, 1e-6)
    print(f"Generated {n_samples:,} schedules in {gen_time:.2f}s ({speed:.1f} schedules/sec)")

    # 3. Domain 1: Statistical & Aggregate Realism
    print("  -> Computing Domain 1: Statistical & Aggregate Realism...")
    d1_jsd = compute_slot_jsd(real_codes, fake_codes)
    d1_tvd = compute_participation_tvd(real_codes, fake_codes)
    d1_dur = compute_daily_minutes(real_codes, fake_codes)
    domain1 = {
        "macro_jsd": d1_jsd["macro_jsd"],
        "max_slot_jsd": d1_jsd["max_slot_jsd"],
        "min_slot_jsd": d1_jsd["min_slot_jsd"],
        "participation_tvd": d1_tvd["tvd"],
        "mean_absolute_duration_bias_minutes": d1_dur["mae_minutes"],
        "duration_breakdown": d1_dur["bias_mins"],
    }

    # 4. Domain 2: Temporal Pacing & Episode Dynamics
    print("  -> Computing Domain 2: Temporal Pacing & Episode Dynamics...")
    d2_trans = compute_transition_matrices_and_fnorm(real_codes, fake_codes)
    d2_spells = compute_spell_emds(real_codes, fake_codes)
    mean_spell_emd = float(np.mean(list(d2_spells["emd_mins"].values())))
    domain2 = {
        "slice_fnorms": {
            "slice_0_morning": d2_trans["fnorms_slices"][0],
            "slice_1_afternoon": d2_trans["fnorms_slices"][1],
            "slice_2_evening": d2_trans["fnorms_slices"][2],
            "slice_3_night": d2_trans["fnorms_slices"][3],
            "mean_slice_fnorm": d2_trans["macro_fnorm"],
        },
        "mean_spell_emd_minutes": mean_spell_emd,
        "spell_emds_minutes": d2_spells["emd_mins"],
        "real_matrices": d2_trans["real_matrices"],
        "fake_matrices": d2_trans["fake_matrices"],
    }

    # 5. Domain 3: Sequence Structure & Tour Formation
    print("  -> Computing Domain 3: Sequence Structure & Tour Formation...")
    d3_episodes = compute_episode_statistics(real_codes, fake_codes)
    d3_anchors = compute_work_anchor_timing(real_codes, fake_codes)
    d3_entropy_fake = compute_transition_entropy(fake_codes)
    d3_entropy_real = compute_transition_entropy(real_codes)
    domain3 = {
        "episodes": {
            "real_mean_episodes": d3_episodes["real_episodes_mean"],
            "fake_mean_episodes": d3_episodes["fake_episodes_mean"],
            "episode_diff": d3_episodes["episode_diff"],
        },
        "switching": {
            "real_mean_switches": d3_episodes["real_switches_mean"],
            "fake_mean_switches": d3_episodes["fake_switches_mean"],
            "switch_diff": d3_episodes["fake_switches_mean"] - d3_episodes["real_switches_mean"],
        },
        "work_anchors": {
            "work_start_emd_minutes": d3_anchors["work_start_w1_mins"],
            "real_work_start_mean_h": d3_anchors["real_work_start_mean_h"],
            "fake_work_start_mean_h": d3_anchors["fake_work_start_mean_h"],
        },
        "entropy": {
            "real_entropy": d3_entropy_real,
            "fake_entropy": d3_entropy_fake,
        },
    }

    # 6. Domain 4: Normative & Statutory Feasibility
    print("  -> Computing Domain 4: Normative & Statutory Feasibility...")
    d4_budget = compute_time_budget_closure(fake_codes)
    d4_child = compute_child_labor_audit(fake_codes, model_cond)
    d4_sleep = compute_sleep_constraint_audit(fake_codes)
    domain4 = {
        "budget": d4_budget,
        "child_labor": d4_child,
        "sleep": d4_sleep,
    }

    # 7. Domain 5: Demographic & Subgroup Equity Parity
    print("  -> Computing Domain 5: Demographic & Subgroup Equity Parity...")
    d5_gender = compute_gender_parity(real_codes, fake_codes, cond_sub, model_cond)
    d5_spatial = compute_spatial_parity(real_codes, fake_codes, cond_sub, model_cond)
    domain5 = {
        "gender_parity": d5_gender,
        "spatial_parity": d5_spatial,
    }

    # 8. Domain 6: ML Indistinguishability & Privacy
    print("  -> Computing Domain 6: ML Indistinguishability & Privacy...")
    d6_rf = compute_adversarial_rf(real_codes, fake_codes, n_sample=min(10000, n_samples), seed=seed)
    d6_priv = compute_privacy_and_novelty(real_codes, fake_codes, n_hamming_eval=min(1000, n_samples), seed=seed)
    domain6 = {
        "adversarial_rf": d6_rf,
        "privacy": d6_priv,
    }

    print(f"Completed evaluation for {model_name} in {time.time()-t0:.2f}s.")

    return {
        "model_name": model_name,
        "info": info,
        "generation_time_sec": gen_time,
        "generation_speed_sps": speed,
        "domain1": domain1,
        "domain2": domain2,
        "domain3": domain3,
        "domain4": domain4,
        "domain5": domain5,
        "domain6": domain6,
        "fake_codes": fake_codes,  # Kept in memory for unified plot generation
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print(" TUS-GAN ULTIMATE TRC 6-DOMAIN RESEARCH EVALUATOR")
    print(" Elsevier Transportation Research Part C: Emerging Technologies Benchmark")
    print("="*80)
    print(f"Target Models:      {args.models}")
    print(f"Sample Size:        {args.n_samples:,} schedules")
    print(f"Inference Device:   {args.device}")
    print(f"Output Directory:   {args.output_dir}")
    print(f"Rejection Sampling: {args.rejection_sampling}")

    # 1. Load Microdata
    diary_tensor, cond_vector, district_ids, state_ids = load_dataset(args.data)

    # 2. Create Deterministic Evaluation Subset
    rng = np.random.default_rng(args.seed)
    N_total = len(diary_tensor)
    N_eval = min(args.n_samples, N_total)

    eval_indices = rng.choice(N_total, size=N_eval, replace=False)
    real_sub = diary_tensor[eval_indices]
    cond_sub = cond_vector[eval_indices]
    dist_sub = district_ids[eval_indices]
    state_sub = state_ids[eval_indices]

    # Decode real schedules: (N, 9, 48, 1) -> argmax along axis 1 -> (N, 48)
    real_codes = np.argmax(real_sub.squeeze(-1), axis=1) + 1

    # 3. Benchmark All Models
    all_results = {}
    model_generated_codes = {}

    for model_name in args.models:
        spec = MODEL_SPECS.get(model_name)
        if spec is None:
            print(f"[Warning] Unknown model '{model_name}'. Skipping.")
            continue
        source_path = spec.get("default_checkpoint") or spec.get("schedules_csv")
        if not source_path or not os.path.exists(source_path):
            print(f"[Notice] Source artifact for '{model_name}' not found at {source_path}. Skipping.")
            continue

        try:
            m_res = evaluate_single_model(
                model_name=model_name,
                real_codes=real_codes,
                cond_sub=cond_sub,
                dist_sub=dist_sub,
                state_sub=state_sub,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed,
                rejection_sampling=args.rejection_sampling,
            )
            model_generated_codes[model_name] = m_res.pop("fake_codes")
            all_results[model_name] = m_res
        except Exception as e:
            print(f"[ERROR] Failed benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("[Error] No models were successfully evaluated. Exiting.")
        sys.exit(1)

    # 4. Generate Reports & Tables
    print(f"\n{'='*70}")
    print(" GENERATING TRC REPORTS & PUBLICATION FIGURES")
    print(f"{'='*70}")

    md_report_path = os.path.join(args.output_dir, "MASTER_BENCHMARK_REPORT.md")
    csv_report_path = os.path.join(args.output_dir, "master_benchmark_summary.csv")
    json_report_path = os.path.join(args.output_dir, "master_benchmark_summary.json")

    generate_markdown_summary(all_results, md_report_path)
    export_tabular_summary(all_results, csv_report_path, json_report_path)
    print(f"Saved Markdown report:  {md_report_path}")
    print(f"Saved Tabular CSV:      {csv_report_path}")
    print(f"Saved Raw JSON:         {json_report_path}")

    # 5. Generate Figures
    print("Generating Figure 1: 9-panel Diurnal Overlays...")
    plot_diurnal_overlays(real_codes, model_generated_codes, args.output_dir)

    print("Generating Figure 2: Spell Duration Survival Step Curves...")
    plot_spell_duration_survival(real_codes, model_generated_codes, args.output_dir)

    print("Generating Figure 3: Multi-Domain Comparative Bar Chart...")
    plot_model_comparison_bars(all_results, args.output_dir)

    print(f"All publication figures saved to: {args.output_dir}/")

    # 6. Print Executive Summary Table
    print("\n" + "="*80)
    print(" MASTER BENCHMARK EXECUTIVE SUMMARY")
    print("="*80)
    print(f"{'Model':<8} | {'Cell':<18} | {'Macro JSD':<12} | {'F-Norm':<10} | {'Spell EMD':<10} | {'Viol %':<8} | {'RF AUC':<8}")
    print("-" * 80)
    for m, data in all_results.items():
        cell = data["info"].get("cell", "N/A")
        jsd = data["domain1"]["macro_jsd"]
        fnorm = data["domain2"]["slice_fnorms"]["mean_slice_fnorm"]
        spell_emd = data["domain2"]["mean_spell_emd_minutes"]
        viol = data["domain4"]["child_labor"]["child_labor_raw_rate"]
        auc = data["domain6"]["adversarial_rf"]["rf_roc_auc"]
        print(f"{m:<8} | {cell:<18} | {jsd:<12.6f} | {fnorm:<10.4f} | {spell_emd:<10.2f} | {viol:<8.2f} | {auc:<8.4f}")
    print("="*80)
    print("Benchmark complete! All TRC artifacts generated successfully.\n")


if __name__ == "__main__":
    main()
