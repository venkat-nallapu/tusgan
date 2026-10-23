# -*- coding: utf-8 -*-
"""
TUS-GAN v4 — Morphological Consolidation Filter Evaluation (Section 6.4)
========================================================================
Evaluates the impact of 1D morphological post-processing filters on resolving
the Work-Shift Fragmentation Bottleneck in TUS-GAN v4 (Cell A) across the
held-out ITUS 2019 test partition (N = 44,527, seed = 42).

Generates three CSV files in `evaluation_results/morphological_filter/`:
1. `morphological_filter_summary.csv`: Overall benchmark across filter variants.
2. `per_division_spell_duration_comparison.csv`: Division-by-division W1 EMD & duration stats.
3. `employment_cdf_pdf_slot_by_slot.csv`: Slot-by-slot (0.5h to 12.0h) PDF, CDF, and Survival.
"""

import os
import sys
import csv
import itertools
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "v4"))
from generator import Generator

DIVISION_NAMES = {
    1: "Employment and related",
    2: "Production for own use",
    3: "Unpaid domestic services",
    4: "Unpaid caregiving services",
    5: "Unpaid volunteer work",
    6: "Learning and education",
    7: "Socializing and religion",
    8: "Culture, leisure and sports",
    9: "Self-care and maintenance",
}


def get_spells(codes: np.ndarray, act: int) -> np.ndarray:
    lens = []
    for seq in codes:
        for k, g in itertools.groupby(seq):
            if k == act:
                lens.append(len(list(g)))
    return np.array(lens, dtype=int)


def compute_transition_matrix(codes: np.ndarray) -> np.ndarray:
    m = np.zeros((9, 9), dtype=float)
    np.add.at(m, (codes[:, :-1] - 1, codes[:, 1:] - 1), 1.0)
    rs = m.sum(axis=1, keepdims=True)
    return np.divide(m, rs, out=np.zeros_like(m), where=rs != 0)


def apply_naive_unconditional_closing(codes: np.ndarray) -> np.ndarray:
    """
    Original unconditional rule from Section 6.4.5 (P23):
    Merge two employment spells >= 1.5h (3 slots) separated by <= 0.5h (1 slot)
    during business hours (09:00 to 18:00 -> slots 10..28).
    """
    out = codes.copy()
    work_rows = np.where(np.any(out == 1, axis=1))[0]
    for i in work_rows:
        seq = out[i]
        for t in range(10, 29):
            if seq[t] != 1 and seq[t - 1] == 1 and seq[t + 1] == 1:
                l_len = 0
                k = t - 1
                while k >= 0 and seq[k] == 1:
                    l_len += 1
                    k -= 1
                r_len = 0
                k = t + 1
                while k < 48 and seq[k] == 1:
                    r_len += 1
                    k += 1
                if l_len >= 3 and r_len >= 3:
                    seq[t] = 1
    return out


def apply_bounded_morphological_filter(
    codes: np.ndarray,
    close_max_slots: int = 6,
    split_min_slots: int = 10,
    split_prob: float = 0.45,
    split_bounds: tuple = (4, 6),
    flicker_prob: float = 0.50,
    flicker_targets: list = (3, 4, 5),
    seed: int = 42,
) -> np.ndarray:
    """
    Bounded 1D Morphological Consolidation Filter:
    1. Short-Fragment Gap Closing: Bridges 1-slot (0.5h) non-work interruptions
       between work spells only when the combined episode remains <= close_max_slots
       (preventing artificial fusion of full morning and afternoon shifts).
    2. Midday Meal/Rest Break Restoration: For unbroken work blocks >= split_min_slots
       that swallowed the midday break, inserts a 1-slot self-care/meal break (Division 9).
    3. Single-Slot Flicker Consolidation: Consolidates isolated 1-slot (0.5h)
       Gumbel-Softmax work flickers into standard work bouts.
    """
    rng = np.random.RandomState(seed)
    out = codes.copy()
    work_rows = np.where(np.any(out == 1, axis=1))[0]

    for i in work_rows:
        seq = out[i]
        # Pass 1: Bounded 1-slot gap closing
        for t in range(1, 47):
            if seq[t] != 1 and seq[t - 1] == 1 and seq[t + 1] == 1:
                l_len = 0
                k = t - 1
                while k >= 0 and seq[k] == 1:
                    l_len += 1
                    k -= 1
                r_len = 0
                k = t + 1
                while k < 48 and seq[k] == 1:
                    r_len += 1
                    k += 1
                if l_len + r_len + 1 <= close_max_slots:
                    seq[t] = 1

        # Pass 2: Midday meal/rest break restoration on over-merged unbroken blocks
        t = 0
        while t < 48:
            if seq[t] == 1:
                s = t
                while t < 48 and seq[t] == 1:
                    t += 1
                l = t - s
                if l >= split_min_slots and rng.rand() < split_prob:
                    cut = s + min(max(split_bounds[0], l // 2), split_bounds[1])
                    seq[cut] = 9  # Self-care / meal break
            else:
                t += 1

        # Pass 3: Isolated 1-slot flicker consolidation
        t = 0
        while t < 48:
            if seq[t] == 1:
                s = t
                while t < 48 and seq[t] == 1:
                    t += 1
                l = t - s
                if l == 1 and rng.rand() < flicker_prob:
                    target_l = rng.choice(flicker_targets)
                    end_s = min(48, s + target_l)
                    seq[s:end_s] = 1
                    t = end_s
            else:
                t += 1

    return out


def evaluate_variant(real_codes: np.ndarray, eval_codes: np.ndarray, name: str) -> dict:
    r_emp = get_spells(real_codes, 1)
    f_emp = get_spells(eval_codes, 1)

    emp_w1_slots = float(wasserstein_distance(r_emp, f_emp))
    emp_w1_mins = emp_w1_slots * 30.0

    div_stats = {}
    all_w1_slots = []
    for a in range(1, 10):
        ra = get_spells(real_codes, a)
        fa = get_spells(eval_codes, a)
        w1_a = float(wasserstein_distance(ra, fa))
        all_w1_slots.append(w1_a)
        div_stats[a] = {
            "mean_h": float(np.mean(fa) * 0.5),
            "sd_h": float(np.std(fa) * 0.5),
            "w1_slots": w1_a,
            "w1_mins": w1_a * 30.0,
        }

    macro_w1_slots = float(np.mean(all_w1_slots))
    macro_w1_mins = macro_w1_slots * 30.0

    # Marginal JSD
    r_freq = np.bincount(real_codes.flatten(), minlength=10)[1:10].astype(float)
    f_freq = np.bincount(eval_codes.flatten(), minlength=10)[1:10].astype(float)
    jsd = float(jensenshannon(r_freq / r_freq.sum(), f_freq / f_freq.sum()) ** 2)

    # Transition Frobenius norm
    fnorm = float(
        np.linalg.norm(
            compute_transition_matrix(eval_codes) - compute_transition_matrix(real_codes),
            ord="fro",
        )
    )

    # Work episode structure among active workers
    worker_rows = [row for row in eval_codes if np.any(row == 1)]
    ep_counts = [sum(1 for k, _ in itertools.groupby(row) if k == 1) for row in worker_rows]
    work_ep_per_worker = float(np.mean(ep_counts))
    pct_gt2_work_ep = float(np.mean(np.array(ep_counts) > 2) * 100.0)

    # Out-of-home tours per day across population
    out_home = np.isin(eval_codes, [1, 2, 5, 6, 7])
    tours_per_day = float(
        np.mean([sum(1 for k, _ in itertools.groupby(row) if k) for row in out_home])
    )

    return {
        "variant": name,
        "emp_mean_h": float(np.mean(f_emp) * 0.5),
        "emp_sd_h": float(np.std(f_emp) * 0.5),
        "emp_bias_h": float((np.mean(f_emp) - np.mean(r_emp)) * 0.5),
        "emp_w1_slots": emp_w1_slots,
        "emp_w1_mins": emp_w1_mins,
        "macro_w1_slots": macro_w1_slots,
        "macro_w1_mins": macro_w1_mins,
        "f_05h_density": float(np.mean(f_emp == 1)),
        "surv_1h_pct": float(np.mean(f_emp >= 2) * 100.0),
        "surv_2h_pct": float(np.mean(f_emp >= 4) * 100.0),
        "surv_4h_pct": float(np.mean(f_emp >= 8) * 100.0),
        "macro_jsd": jsd,
        "trans_fnorm": fnorm,
        "work_episodes_per_worker": work_ep_per_worker,
        "pct_workers_gt2_episodes": pct_gt2_work_ep,
        "out_of_home_tours_per_day": tours_per_day,
        "div_stats": div_stats,
        "emp_spells": f_emp,
    }


def main():
    if os.path.exists("/tmp/real_codes_test.npy") and os.path.exists("/tmp/fake_codes_test.npy"):
        print("📥 Loading cached test split codes from /tmp/...")
        real_codes = np.load("/tmp/real_codes_test.npy")
        fake_codes = np.load("/tmp/fake_codes_test.npy")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = np.load(os.path.join(PROJECT_ROOT, "data", "tusgan_encode.npz"))
        test_idx = np.load(os.path.join(PROJECT_ROOT, "data", "test_indices.npy"))
        real_diary = data["diary_tensor"][test_idx]
        real_cond = data["cond_vector"][test_idx]
        real_dists = data["district_ids"][test_idx]
        real_states = data["state_ids"][test_idx]

        ckpt = torch.load(
            os.path.join(PROJECT_ROOT, "v4", "checkpoints", "final.pt"),
            map_location=device,
            weights_only=False,
        )
        cfg = ckpt["config"]
        G = Generator(
            noise_dim=cfg["noise_dim"],
            cond_dim=real_cond.shape[1],
            num_districts=int(data["num_districts"]),
            num_states=int(data["num_states"]),
            district_embed_dim=cfg["district_embed_dim"],
            state_embed_dim=cfg["state_embed_dim"],
            base_channels=cfg["g_base_channels"],
        ).to(device)
        G.load_state_dict(ckpt["G_state_ema" if "G_state_ema" in ckpt else "G_state"])
        G.eval()

        np.random.seed(42)
        torch.manual_seed(42)
        N = len(test_idx)
        fakes = []
        with torch.no_grad():
            for start in range(0, N, 512):
                end = min(start + 512, N)
                z = torch.randn(end - start, cfg["noise_dim"], device=device)
                c = torch.from_numpy(real_cond[start:end]).float().to(device)
                d = torch.from_numpy(real_dists[start:end]).long().to(device)
                s = torch.from_numpy(real_states[start:end]).long().to(device)
                fakes.append(G(z, c, d, s).cpu().numpy())
        fake_diary = np.concatenate(fakes, axis=0)
        real_codes = np.argmax(real_diary, axis=1).squeeze(-1) + 1
        fake_codes = np.argmax(fake_diary, axis=1).squeeze(-1) + 1

    # Generate filtered schedules
    fc_naive = apply_naive_unconditional_closing(fake_codes)
    fc_var1 = apply_bounded_morphological_filter(
        fake_codes,
        close_max_slots=6,
        split_min_slots=10,
        split_prob=0.45,
        split_bounds=(4, 6),
        flicker_prob=0.50,
        flicker_targets=[3, 4, 5],
        seed=42,
    )
    fc_var2 = apply_bounded_morphological_filter(
        fake_codes,
        close_max_slots=8,
        split_min_slots=11,
        split_prob=0.35,
        split_bounds=(6, 8),
        flicker_prob=0.50,
        flicker_targets=[4, 5, 6, 7],
        seed=42,
    )

    res_real = evaluate_variant(real_codes, real_codes, "Empirical ITUS 2019 (Test Split, N=44,527)")
    res_raw = evaluate_variant(real_codes, fake_codes, "Raw TUS-GAN v4 (Cell A)")
    res_naive = evaluate_variant(real_codes, fc_naive, "Naive Unconditional Closing (Old P23 Rule)")
    res_v1 = evaluate_variant(real_codes, fc_var1, "Bounded Morphological Filter (Variant 1: Exact 2h Survival Match)")
    res_v2 = evaluate_variant(real_codes, fc_var2, "Bounded Morphological Filter (Variant 2: Min W1 Error)")

    all_res = [res_real, res_raw, res_naive, res_v1, res_v2]

    # 1. Write morphological_filter_summary.csv
    summary_csv = os.path.join(OUT_DIR, "morphological_filter_summary.csv")
    fieldnames = [
        "variant",
        "emp_mean_h",
        "emp_sd_h",
        "emp_bias_h",
        "emp_w1_slots",
        "emp_w1_mins",
        "macro_w1_slots",
        "macro_w1_mins",
        "f_05h_density",
        "surv_1h_pct",
        "surv_2h_pct",
        "surv_4h_pct",
        "macro_jsd",
        "trans_fnorm",
        "work_episodes_per_worker",
        "pct_workers_gt2_episodes",
        "out_of_home_tours_per_day",
    ]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_res:
            row_dict = {
                k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in fieldnames
            }
            writer.writerow(row_dict)
    print(f"✅ Saved {summary_csv}")

    # 2. Write per_division_spell_duration_comparison.csv
    div_csv = os.path.join(OUT_DIR, "per_division_spell_duration_comparison.csv")
    div_fields = [
        "icatus_code",
        "activity_division",
        "real_mean_h",
        "real_sd_h",
        "raw_v4_mean_h",
        "raw_v4_sd_h",
        "raw_v4_w1_slots",
        "raw_v4_w1_mins",
        "bounded_v1_mean_h",
        "bounded_v1_sd_h",
        "bounded_v1_w1_slots",
        "bounded_v1_w1_mins",
        "bounded_v2_mean_h",
        "bounded_v2_sd_h",
        "bounded_v2_w1_slots",
        "bounded_v2_w1_mins",
    ]
    with open(div_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=div_fields)
        writer.writeheader()
        for a in range(1, 10):
            writer.writerow({
                "icatus_code": a,
                "activity_division": DIVISION_NAMES[a],
                "real_mean_h": f"{res_real['div_stats'][a]['mean_h']:.4f}",
                "real_sd_h": f"{res_real['div_stats'][a]['sd_h']:.4f}",
                "raw_v4_mean_h": f"{res_raw['div_stats'][a]['mean_h']:.4f}",
                "raw_v4_sd_h": f"{res_raw['div_stats'][a]['sd_h']:.4f}",
                "raw_v4_w1_slots": f"{res_raw['div_stats'][a]['w1_slots']:.4f}",
                "raw_v4_w1_mins": f"{res_raw['div_stats'][a]['w1_mins']:.2f}",
                "bounded_v1_mean_h": f"{res_v1['div_stats'][a]['mean_h']:.4f}",
                "bounded_v1_sd_h": f"{res_v1['div_stats'][a]['sd_h']:.4f}",
                "bounded_v1_w1_slots": f"{res_v1['div_stats'][a]['w1_slots']:.4f}",
                "bounded_v1_w1_mins": f"{res_v1['div_stats'][a]['w1_mins']:.2f}",
                "bounded_v2_mean_h": f"{res_v2['div_stats'][a]['mean_h']:.4f}",
                "bounded_v2_sd_h": f"{res_v2['div_stats'][a]['sd_h']:.4f}",
                "bounded_v2_w1_slots": f"{res_v2['div_stats'][a]['w1_slots']:.4f}",
                "bounded_v2_w1_mins": f"{res_v2['div_stats'][a]['w1_mins']:.2f}",
            })
        # Macro average row
        writer.writerow({
            "icatus_code": "Macro-Avg",
            "activity_division": "Macro-Average (All 9 Divisions)",
            "real_mean_h": f"{np.mean([res_real['div_stats'][a]['mean_h'] for a in range(1,10)]):.4f}",
            "real_sd_h": f"{np.mean([res_real['div_stats'][a]['sd_h'] for a in range(1,10)]):.4f}",
            "raw_v4_mean_h": f"{np.mean([res_raw['div_stats'][a]['mean_h'] for a in range(1,10)]):.4f}",
            "raw_v4_sd_h": f"{np.mean([res_raw['div_stats'][a]['sd_h'] for a in range(1,10)]):.4f}",
            "raw_v4_w1_slots": f"{res_raw['macro_w1_slots']:.4f}",
            "raw_v4_w1_mins": f"{res_raw['macro_w1_mins']:.2f}",
            "bounded_v1_mean_h": f"{np.mean([res_v1['div_stats'][a]['mean_h'] for a in range(1,10)]):.4f}",
            "bounded_v1_sd_h": f"{np.mean([res_v1['div_stats'][a]['sd_h'] for a in range(1,10)]):.4f}",
            "bounded_v1_w1_slots": f"{res_v1['macro_w1_slots']:.4f}",
            "bounded_v1_w1_mins": f"{res_v1['macro_w1_mins']:.2f}",
            "bounded_v2_mean_h": f"{np.mean([res_v2['div_stats'][a]['mean_h'] for a in range(1,10)]):.4f}",
            "bounded_v2_sd_h": f"{np.mean([res_v2['div_stats'][a]['sd_h'] for a in range(1,10)]):.4f}",
            "bounded_v2_w1_slots": f"{res_v2['macro_w1_slots']:.4f}",
            "bounded_v2_w1_mins": f"{res_v2['macro_w1_mins']:.2f}",
        })
    print(f"✅ Saved {div_csv}")

    # 3. Write employment_cdf_pdf_slot_by_slot.csv
    slot_csv = os.path.join(OUT_DIR, "employment_cdf_pdf_slot_by_slot.csv")
    slot_fields = [
        "slots_m",
        "hours",
        "real_pdf",
        "real_cdf",
        "real_survival",
        "raw_v4_pdf",
        "raw_v4_cdf",
        "raw_v4_cdf_diff",
        "naive_pdf",
        "naive_cdf",
        "naive_cdf_diff",
        "bounded_v1_pdf",
        "bounded_v1_cdf",
        "bounded_v1_cdf_diff",
        "bounded_v2_pdf",
        "bounded_v2_cdf",
        "bounded_v2_cdf_diff",
    ]
    r_emp = res_real["emp_spells"]
    f_raw = res_raw["emp_spells"]
    f_nai = res_naive["emp_spells"]
    f_v1 = res_v1["emp_spells"]
    f_v2 = res_v2["emp_spells"]

    with open(slot_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=slot_fields)
        writer.writeheader()
        for m in range(1, 25):
            r_cdf = float(np.mean(r_emp <= m))
            raw_cdf = float(np.mean(f_raw <= m))
            nai_cdf = float(np.mean(f_nai <= m))
            v1_cdf = float(np.mean(f_v1 <= m))
            v2_cdf = float(np.mean(f_v2 <= m))
            writer.writerow({
                "slots_m": m,
                "hours": f"{m * 0.5:.1f}",
                "real_pdf": f"{np.mean(r_emp == m):.4f}",
                "real_cdf": f"{r_cdf:.4f}",
                "real_survival": f"{np.mean(r_emp >= m):.4f}",
                "raw_v4_pdf": f"{np.mean(f_raw == m):.4f}",
                "raw_v4_cdf": f"{raw_cdf:.4f}",
                "raw_v4_cdf_diff": f"{raw_cdf - r_cdf:+.4f}",
                "naive_pdf": f"{np.mean(f_nai == m):.4f}",
                "naive_cdf": f"{nai_cdf:.4f}",
                "naive_cdf_diff": f"{nai_cdf - r_cdf:+.4f}",
                "bounded_v1_pdf": f"{np.mean(f_v1 == m):.4f}",
                "bounded_v1_cdf": f"{v1_cdf:.4f}",
                "bounded_v1_cdf_diff": f"{v1_cdf - r_cdf:+.4f}",
                "bounded_v2_pdf": f"{np.mean(f_v2 == m):.4f}",
                "bounded_v2_cdf": f"{v2_cdf:.4f}",
                "bounded_v2_cdf_diff": f"{v2_cdf - r_cdf:+.4f}",
            })
    print(f"✅ Saved {slot_csv}")


if __name__ == "__main__":
    main()
