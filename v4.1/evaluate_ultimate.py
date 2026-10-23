# -*- coding: utf-8 -*-
"""
TUS-GAN — Ultimate Evaluation Suite (v4.1 / Cell C)
===================================================
Unified evaluation pipeline tailored for TUS-GAN v4.1 (CNN + Deterministic Hard Masking).
Evaluates all metrics required for Table 6 and Section 6.5 in the manuscript:
1. Statistical Realism: Jensen-Shannon Divergence (Macro JSD), Time-Use Minutes per division.
2. Temporal Realism: Time-Slice Transition Matrix Frobenius Norm, Spell-Duration EMD (slots & mins).
3. Statutory Compliance: Child Labor 3-Point Reporting (Empirical Base, Raw Model, Delivered).
4. Adversarial Indistinguishability: Random Forest Accuracy & ROC-AUC.
5. Publication Figures: TRC-formatted diurnal participation curves, transition heatmaps, ROC curve.
"""

import os
import argparse
import itertools
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from generator import Generator

# ─────────────────────────────────────────────────────────────
# 1. CONSTANTS & MAPPINGS
# ─────────────────────────────────────────────────────────────

DIVISION_LABELS = {
    1: "Employment & Related",
    2: "Production for Own Use",
    3: "Unpaid Domestic Services",
    4: "Unpaid Caregiving",
    5: "Unpaid Volunteer/Community",
    6: "Learning & Education",
    7: "Socializing & Religious",
    8: "Culture, Leisure & Sports",
    9: "Self-Care & Maintenance",
}

MINUTES_PER_SLOT = 30


# ─────────────────────────────────────────────────────────────
# 2. GENERATOR & DATA LOADERS
# ─────────────────────────────────────────────────────────────


def load_generator(checkpoint_path: str, data_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    data = np.load(data_path)
    actual_cond_dim = data["cond_vector"].shape[1]
    actual_num_districts = int(data["num_districts"])
    actual_num_states = int(data["num_states"])

    G = Generator(
        noise_dim=cfg.get("noise_dim", 128),
        cond_dim=actual_cond_dim,
        num_districts=actual_num_districts,
        num_states=actual_num_states,
        district_embed_dim=cfg.get("district_embed_dim", 16),
        state_embed_dim=cfg.get("state_embed_dim", 8),
        base_channels=cfg.get("g_base_channels", 256),
    ).to(device)

    g_state_key = "G_state_ema" if "G_state_ema" in ckpt else "G_state"
    G.load_state_dict(ckpt[g_state_key])
    G.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"✅ Loaded Generator from {checkpoint_path} (epoch {epoch}, weights: {g_state_key})")
    return G, cfg.get("noise_dim", 128)


def resolve_split_indices(split="test", split_indices_path=None, npz_path=None):
    """
    Resolves and loads the 90/10 train/test partition indices.
    """
    if split is None or str(split).lower() == "all":
        return None

    if split_indices_path and os.path.exists(split_indices_path):
        indices = np.load(split_indices_path)
        print(f"📁 Loaded explicit split indices from {split_indices_path} ({len(indices):,} records)")
        return indices

    fname = f"{str(split).lower()}_indices.npy"
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

    candidates = [
        fname,
        os.path.join(script_dir, fname),
        os.path.join(script_dir, "..", "data", fname),
        os.path.join("data", fname),
        os.path.join("..", "data", fname),
    ]
    if npz_path:
        npz_dir = os.path.dirname(os.path.abspath(npz_path))
        candidates.extend([
            os.path.join(npz_dir, fname),
            os.path.join(npz_dir, "..", "data", fname),
            os.path.join(npz_dir, "data", fname),
        ])

    for cand in candidates:
        if os.path.exists(cand):
            indices = np.load(cand)
            print(f"📁 Loaded split '{split}' from {cand} ({len(indices):,} records)")
            return indices

    print(f"⚠️ Warning: Split indices for '{split}' not found in candidates. Using full dataset.")
    return None


def load_real_data(data_path: str, split: str = "test", split_indices: str = None):
    if not os.path.exists(data_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            data_path,
            os.path.join(script_dir, data_path),
            os.path.join(script_dir, "tusgan_encode.npz"),
            os.path.join(script_dir, "..", "data", "tusgan_encode.npz"),
            os.path.join(script_dir, "..", "v4", "tusgan_encode.npz"),
            "tusgan_encode.npz",
            "data/tusgan_encode.npz",
            "v4/tusgan_encode.npz",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                data_path = cand
                break

    data = np.load(data_path)
    diary = data["diary_tensor"]
    cond = data["cond_vector"]
    district = data["district_ids"]
    state = data["state_ids"]

    split_idx = resolve_split_indices(split=split, split_indices_path=split_indices, npz_path=data_path)
    if split_idx is not None:
        diary = diary[split_idx]
        cond = cond[split_idx]
        district = district[split_idx]
        state = state[split_idx]
        split_info = f" (split='{split}', {len(diary):,} records)"
    else:
        split_info = " (full dataset)"

    print(f"✅ Loaded real data from {data_path}{split_info}")
    return diary, cond, district, state


@torch.no_grad()
def generate_synthetic(
    G, cond_vector, district_ids, state_ids, noise_dim, n_samples, device, batch_size=512
):
    N = cond_vector.shape[0]
    indices = np.random.choice(N, size=n_samples, replace=True)

    cond_all = torch.from_numpy(cond_vector[indices]).float()
    dist_all = torch.from_numpy(district_ids[indices]).long()
    state_all = torch.from_numpy(state_ids[indices]).long()

    all_fakes = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        z = torch.randn(end - start, noise_dim, device=device)
        c = cond_all[start:end].to(device)
        d = dist_all[start:end].to(device)
        s = state_all[start:end].to(device)

        # In v4.1, deterministic hard masking is intrinsic
        fake = G(z, c, d, s, temp=0.1, hard=True)
        all_fakes.append(fake.cpu().numpy())

    fake_tensor = np.concatenate(all_fakes, axis=0)
    return fake_tensor, cond_vector[indices]


def decode_to_codes(tensor_9ch: np.ndarray) -> np.ndarray:
    return np.argmax(tensor_9ch, axis=1).squeeze(-1) + 1


# ─────────────────────────────────────────────────────────────
# 3. METRIC COMPUTATIONS
# ─────────────────────────────────────────────────────────────


def compute_standard_metrics(real_codes: np.ndarray, fake_codes: np.ndarray):
    # 1. JSD across 9 x 48 activity-time slots
    jsd_per_slot = []
    for t in range(48):
        r_counts = np.bincount(real_codes[:, t], minlength=10)[1:]
        f_counts = np.bincount(fake_codes[:, t], minlength=10)[1:]
        p_real = r_counts / (r_counts.sum() + 1e-12)
        p_fake = f_counts / (f_counts.sum() + 1e-12)
        jsd = jensenshannon(p_real, p_fake) ** 2
        jsd_per_slot.append(jsd)
    overall_jsd = float(np.mean(jsd_per_slot))

    # 2. Time-Slice Transition Matrix Frobenius Norm
    fnorms = []
    for k in range(4):
        start, end = k * 12, (k + 1) * 12
        r_t = real_codes[:, start : end - 1] - 1
        r_tp1 = real_codes[:, start + 1 : end] - 1
        f_t = fake_codes[:, start : end - 1] - 1
        f_tp1 = fake_codes[:, start + 1 : end] - 1

        P_r = np.zeros((9, 9))
        P_f = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                P_r[i, j] = np.sum((r_t == i) & (r_tp1 == j))
                P_f[i, j] = np.sum((f_t == i) & (f_tp1 == j))
        P_r = P_r / (P_r.sum(axis=1, keepdims=True) + 1e-12)
        P_f = P_f / (P_f.sum(axis=1, keepdims=True) + 1e-12)
        fnorms.append(float(np.linalg.norm(P_r - P_f, "fro")))

    mean_fnorm = float(np.mean(fnorms))

    # 3. Time Use Minutes
    r_mins = {}
    f_mins = {}
    for act in range(1, 10):
        r_mins[act] = float(np.mean(np.sum(real_codes == act, axis=1)) * MINUTES_PER_SLOT)
        f_mins[act] = float(np.mean(np.sum(fake_codes == act, axis=1)) * MINUTES_PER_SLOT)

    return {
        "jsd": overall_jsd,
        "trans_fnorm": mean_fnorm,
        "fnorms_slices": fnorms,
        "real_mins": r_mins,
        "fake_mins": f_mins,
    }


def extract_spell_durations(codes: np.ndarray):
    N, T = codes.shape
    spells_by_activity = {act: [] for act in range(1, 10)}
    for i in range(N):
        seq = codes[i]
        curr_act = seq[0]
        curr_len = 1
        for t in range(1, T):
            if seq[t] == curr_act:
                curr_len += 1
            else:
                spells_by_activity[curr_act].append(curr_len * MINUTES_PER_SLOT)
                curr_act = seq[t]
                curr_len = 1
        spells_by_activity[curr_act].append(curr_len * MINUTES_PER_SLOT)
    return spells_by_activity


def evaluate_spells(real_codes: np.ndarray, fake_codes: np.ndarray):
    real_spells = extract_spell_durations(real_codes)
    fake_spells = extract_spell_durations(fake_codes)
    distances_min = {}
    distances_slot = {}
    for act in range(1, 10):
        r_dur = real_spells[act]
        f_dur = fake_spells[act]
        if len(r_dur) > 0 and len(f_dur) > 0:
            w_min = float(wasserstein_distance(r_dur, f_dur))
            w_slot = float(w_min / MINUTES_PER_SLOT)
            distances_min[act] = w_min
            distances_slot[act] = w_slot
        else:
            distances_min[act] = 0.0
            distances_slot[act] = 0.0
    mean_w1_min = float(np.mean(list(distances_min.values())))
    mean_w1_slot = float(np.mean(list(distances_slot.values())))
    return {
        "distances_min": distances_min,
        "distances_slot": distances_slot,
        "mean_w1_min": mean_w1_min,
        "mean_w1_slot": mean_w1_slot,
    }


def evaluate_child_labor(codes: np.ndarray, cond_vector: np.ndarray):
    is_child = cond_vector[:, 0] == 1.0
    child_codes = codes[is_child]
    n_children = len(child_codes)
    if n_children == 0:
        return 0.0, 0, 0
    child_work_slots = np.sum(child_codes == 1, axis=1)
    violating_schedules = np.sum(child_work_slots > 0)
    rate = float(violating_schedules / n_children) * 100.0
    return rate, int(violating_schedules), int(n_children)


def evaluate_adversarial(real_codes: np.ndarray, fake_codes: np.ndarray, n_eval: int = 10000):
    n_samples = min(len(real_codes), len(fake_codes), n_eval)
    idx_r = np.random.choice(len(real_codes), n_samples, replace=False)
    idx_f = np.random.choice(len(fake_codes), n_samples, replace=False)

    X_real = np.eye(9)[real_codes[idx_r] - 1].reshape(n_samples, -1)
    X_fake = np.eye(9)[fake_codes[idx_f] - 1].reshape(n_samples, -1)

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)[:, 1]

    acc = float(accuracy_score(y_te, preds))
    auc = float(roc_auc_score(y_te, probs))
    fpr, tpr, _ = roc_curve(y_te, probs)

    return acc, auc, fpr, tpr


# ─────────────────────────────────────────────────────────────
# 4. PLOTTING DIAGNOSTICS
# ─────────────────────────────────────────────────────────────


def plot_diagnostics(real_codes, fake_codes, fpr, tpr, auc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    time_hours = np.linspace(4, 28, 48) % 24

    # 1. Diurnal Participation Curves (9 Divisions)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for act in range(1, 10):
        r_part = np.mean(real_codes == act, axis=0)
        f_part = np.mean(fake_codes == act, axis=0)

        ax = axes[act - 1]
        ax.plot(range(48), r_part, label="Real (ITUS 2019)", color="#1F77B4", lw=2.0)
        ax.plot(range(48), f_part, label="Synthetic (v4.1 Cell C)", color="#D62728", lw=2.0, ls="--")
        ax.set_title(DIVISION_LABELS[act], fontsize=10, fontweight="bold")
        ax.set_xticks(range(0, 48, 8))
        ax.set_xticklabels([f"{int(time_hours[t]):02d}:00" for t in range(0, 48, 8)], fontsize=8)
        ax.set_ylabel("Participation Rate", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        if act == 1:
            ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cell_c_diurnal_participation.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "cell_c_diurnal_participation.pdf"))
    plt.close()

    # 2. Adversarial ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2B5C8F", lw=2, label=f"Random Forest (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("Adversarial Indistinguishability (v4.1 Cell C)", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9, frameon=False)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cell_c_adversarial_roc.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "cell_c_adversarial_roc.pdf"))
    plt.close()


# ─────────────────────────────────────────────────────────────
# 5. MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TUS-GAN v4.1 (Cell C) Ultimate Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt", help="Path to generator checkpoint")
    parser.add_argument("--data", type=str, default="tusgan_encode.npz", help="Path to npz dataset")
    parser.add_argument("--n_samples", type=int, default=30000, help="Number of synthetic samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Dataset split: 'test' (10%%, 44,527 records), 'train' (90%%, 400,741 records), or 'all'",
    )
    parser.add_argument(
        "--split_indices",
        type=str,
        default=None,
        help="Explicit path to split indices .npy file",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Evaluation Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Generator & Real Data
    G, noise_dim = load_generator(args.checkpoint, args.data, device)
    real_tensor, real_cond, real_dist, real_state = load_real_data(
        args.data, split=args.split, split_indices=args.split_indices
    )
    real_codes = decode_to_codes(real_tensor)

    # 2. Generate Synthetic Cohort
    print(f"🎲 Generating N = {args.n_samples:,} synthetic diaries under Cell C (CNN + Hard Mask)...")
    fake_tensor, fake_cond = generate_synthetic(
        G, real_cond, real_dist, real_state, noise_dim, args.n_samples, device
    )
    fake_codes = decode_to_codes(fake_tensor)

    # 3. Compute Metrics
    print("\n📊 Computing statistical and temporal metrics...")
    std_metrics = compute_standard_metrics(real_codes, fake_codes)
    print(f"  • Macro JSD                           : {std_metrics['jsd']:.6f}")
    print(f"  • Time-Slice Transition F-Norm        : {std_metrics['trans_fnorm']:.4f}")

    print("\n⏱️ Computing spell-duration Earth Mover's Distances...")
    spell_metrics = evaluate_spells(real_codes, fake_codes)
    print(f"  • Mean Spell Duration EMD (Minutes)   : {spell_metrics['mean_w1_min']:.2f} min")
    print(f"  • Mean Spell Duration EMD (Slots)     : {spell_metrics['mean_w1_slot']:.3f} slots")
    print(f"  • Employment Spell EMD (ICATUS 1)     : {spell_metrics['distances_min'][1]:.1f} min ({spell_metrics['distances_slot'][1]:.3f} slots)")
    print(f"  • Sleep Spell EMD (ICATUS 9)          : {spell_metrics['distances_min'][9]:.1f} min ({spell_metrics['distances_slot'][9]:.3f} slots)")

    print("\n⚖️ Auditing Statutory Rule Compliance (Child Labor C1)...")
    real_rate, r_v, r_t = evaluate_child_labor(real_codes, real_cond)
    fake_rate, f_v, f_t = evaluate_child_labor(fake_codes, fake_cond)
    print(f"  • Empirical Ground Truth Violation    : {real_rate:.2f}% ({r_v}/{r_t})")
    print(f"  • Raw Model Violation Rate (Cell C)   : {fake_rate:.2f}% ({f_v}/{f_t})")

    print("\n🛡️ Running Adversarial Validation (Random Forest Indistinguishability)...")
    acc, auc, fpr, tpr = evaluate_adversarial(real_codes, fake_codes, n_eval=10000)
    print(f"  • Adversarial Accuracy                : {acc * 100:.2f}%")
    print(f"  • Adversarial ROC-AUC                 : {auc:.4f}")

    # 4. Generate Diagnostic Plots
    print("\n📈 Rendering publication-quality diagnostic exhibits...")
    plot_diagnostics(real_codes, fake_codes, fpr, tpr, auc, args.output_dir)

    # 5. Save Results to Disk
    results = {
        "model": "TUS-GAN v4.1 (Cell C: CNN + Hard Masking)",
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "macro_jsd": std_metrics["jsd"],
        "transition_fnorm": std_metrics["trans_fnorm"],
        "transition_fnorms_slices": std_metrics["fnorms_slices"],
        "mean_spell_emd_min": spell_metrics["mean_w1_min"],
        "mean_spell_emd_slots": spell_metrics["mean_w1_slot"],
        "spell_emds_min": spell_metrics["distances_min"],
        "spell_emds_slots": spell_metrics["distances_slot"],
        "child_labor_empirical_rate": real_rate,
        "child_labor_raw_rate": fake_rate,
        "adversarial_accuracy": acc,
        "adversarial_auc": auc,
    }

    json_path = os.path.join(args.output_dir, "metrics_cell_c.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args.output_dir, "table6_cell_c_summary.csv")
    with open(csv_path, "w") as f:
        f.write("Cell,Backbone,Enforcement,JSD,Transition_FNorm,Child_Labor_Raw,Adversarial_AUC,Mean_Spell_EMD_min\n")
        f.write(f"C,CNN,Hard Masking,{std_metrics['jsd']:.6f},{std_metrics['trans_fnorm']:.4f},{fake_rate:.2f}%,{auc:.4f},{spell_metrics['mean_w1_min']:.2f}\n")

    print(f"\n✅ All evaluation artifacts successfully saved to {args.output_dir}/")
    print(f"   • Summary JSON : {json_path}")
    print(f"   • Summary CSV  : {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
