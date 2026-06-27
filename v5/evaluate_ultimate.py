# -*- coding: utf-8 -*-
"""
TUS-GAN — Ultimate Evaluation Script (v5)
=========================================
This script combines all previous metrics and advanced evaluations into one
unified pipeline for TUS-GAN v5.

It computes:
1. Statistical Realism (JSD, Time-Use Minutes, Frequency Distributions)
2. Temporal Realism (Transition Matrix Frobenius Norm, Spell-Duration EMD)
3. Logical Realism (Child Labor Constraint Violations)
4. Indistinguishability (Adversarial Random Forest Classifier AUC-ROC)
5. Visualizations (Bar charts, Heatmaps, Sample Step-Plots, EMD distributions, ROC)
"""

import os
import argparse
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

from generator import Generator

# ─────────────────────────────────────────────────────────────
# 1. CONSTANTS & ACTIVITY MAPPING
# ─────────────────────────────────────────────────────────────

DIVISION_LABELS = {
    1: "Employment & Related",
    2: "Production for Own Use",
    3: "Unpaid Domestic Services",
    4: "Unpaid Caregiving",
    5: "Unpaid Volunteer/Community",
    6: "Learning",
    7: "Socializing & Religious",
    8: "Culture, Leisure & Sports",
    9: "Self-care & Maintenance",
}

MINUTES_PER_SLOT = 30

# ─────────────────────────────────────────────────────────────
# 2. HELPER — LOAD GENERATOR
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


# ─────────────────────────────────────────────────────────────
# 3. HELPER — LOAD DATA & GENERATE
# ─────────────────────────────────────────────────────────────


def load_real_data(data_path: str):
    data = np.load(data_path)
    print(f"✅ Loaded real data from {data_path} ({data['diary_tensor'].shape[0]:,} diaries)")
    return (
        data["diary_tensor"],
        data["cond_vector"],
        data["district_ids"],
        data["state_ids"],
    )


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
        fake = G(z, c, d, s)
        all_fakes.append(fake.cpu().numpy())

    fake_tensor = np.concatenate(all_fakes, axis=0)
    print(f"✅ Generated {fake_tensor.shape[0]:,} synthetic diaries.")
    return fake_tensor, cond_vector[indices]


def decode_to_codes(tensor_9ch: np.ndarray) -> np.ndarray:
    return np.argmax(tensor_9ch, axis=1).squeeze(-1) + 1


# ─────────────────────────────────────────────────────────────
# 4. STANDARD METRICS (JSD, F-NORM, AVG MINUTES)
# ─────────────────────────────────────────────────────────────


def compute_numpy_transition_matrix(codes: np.ndarray) -> np.ndarray:
    c_t = codes[:, :-1] - 1
    c_tp1 = codes[:, 1:] - 1
    trans = np.zeros((9, 9), dtype=float)
    np.add.at(trans, (c_t, c_tp1), 1.0)
    row_sums = trans.sum(axis=1, keepdims=True)
    return np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums != 0)


def compute_standard_metrics(real_codes, fake_codes):
    divisions = np.arange(1, 10)

    real_flat, fake_flat = real_codes.flatten(), fake_codes.flatten()
    real_freq = np.array([(real_flat == d).sum() for d in divisions], dtype=float)
    fake_freq = np.array([(fake_flat == d).sum() for d in divisions], dtype=float)

    real_pct = real_freq / real_flat.size * 100
    fake_pct = fake_freq / fake_flat.size * 100

    real_prob = real_freq / real_freq.sum()
    fake_prob = fake_freq / fake_freq.sum()
    jsd = float(jensenshannon(real_prob, fake_prob) ** 2)

    real_minutes = np.array(
        [np.mean(np.sum(real_codes == d, axis=1)) * MINUTES_PER_SLOT for d in divisions]
    )
    fake_minutes = np.array(
        [np.mean(np.sum(fake_codes == d, axis=1)) * MINUTES_PER_SLOT for d in divisions]
    )

    trans_real = compute_numpy_transition_matrix(real_codes)
    trans_fake = compute_numpy_transition_matrix(fake_codes)
    trans_fnorm = float(np.linalg.norm(trans_fake - trans_real, ord="fro"))

    return dict(
        divisions=divisions,
        real_pct=real_pct,
        fake_pct=fake_pct,
        jsd=jsd,
        real_minutes=real_minutes,
        fake_minutes=fake_minutes,
        trans_fnorm=trans_fnorm,
    )


# ─────────────────────────────────────────────────────────────
# 5. SPELL-DURATION ANALYSIS (EMD)
# ─────────────────────────────────────────────────────────────


def extract_spells(sequences: np.ndarray):
    spells = {i: [] for i in range(1, 10)}
    for seq in sequences:
        for activity, group in itertools.groupby(seq):
            spells[activity].append(len(list(group)))
    return spells


def evaluate_spells(real_codes, fake_codes):
    real_spells = extract_spells(real_codes)
    fake_spells = extract_spells(fake_codes)
    distances = {}
    for act_idx in range(1, 10):
        r_dur, f_dur = real_spells[act_idx], fake_spells[act_idx]
        if len(r_dur) > 0 and len(f_dur) > 0:
            distances[act_idx] = wasserstein_distance(r_dur, f_dur)
        else:
            distances[act_idx] = float("nan")
    return distances, real_spells, fake_spells


# ─────────────────────────────────────────────────────────────
# 6. LOGICAL CONSTRAINTS (CHILD LABOR)
# ─────────────────────────────────────────────────────────────


def evaluate_logical_constraints(real_codes, fake_codes, real_cond, fake_cond):
    def check_violations(cond, codes):
        child_mask = cond[:, 0] == 1
        num_children = np.sum(child_mask)
        if num_children > 0:
            child_diaries = codes[child_mask]
            violations = np.sum(np.any(child_diaries == 1, axis=1))
            return violations, num_children, (violations / num_children) * 100
        return 0, 0, 0.0

    r_v, r_c, r_rate = check_violations(real_cond, real_codes)
    f_v, f_c, f_rate = check_violations(fake_cond, fake_codes)

    return {"real": (r_v, r_c, r_rate), "fake": (f_v, f_c, f_rate)}


# ─────────────────────────────────────────────────────────────
# 7. ADVERSARIAL VALIDATION CLASSIFIER
# ─────────────────────────────────────────────────────────────


def run_adversarial_validation(real_tensor, fake_tensor, real_cond, fake_cond):
    X_real = np.hstack([real_cond, real_tensor.reshape(real_tensor.shape[0], -1)])
    X_fake = np.hstack([fake_cond, fake_tensor.reshape(fake_tensor.shape[0], -1)])

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([np.ones(X_real.shape[0]), np.zeros(X_fake.shape[0])])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, clf.predict(X_test))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return auc, acc, fpr, tpr


# ─────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────


def plot_all_visuals(
    metrics, emd_data, roc_data, real_tensor, fake_tensor, real_codes, fake_codes, output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    divisions = metrics["divisions"]
    x = np.arange(len(divisions))
    w = 0.35

    # 1. Frequency Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, metrics["real_pct"], w, label="Real", color="#4C72B0")
    ax.bar(x + w / 2, metrics["fake_pct"], w, label="Synthetic", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{DIVISION_LABELS[d][:12]}" for d in divisions], fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(output_dir, "activity_distribution.png"), dpi=150)
    plt.close()

    # 2. Time Use
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, metrics["real_minutes"], w, label="Real", color="#4C72B0")
    ax.bar(x + w / 2, metrics["fake_minutes"], w, label="Synthetic", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n{DIVISION_LABELS[d][:12]}" for d in divisions], fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(output_dir, "time_use_comparison.png"), dpi=150)
    plt.close()

    # 3. Heatmap
    real_avg = real_tensor.mean(axis=0).squeeze(-1)
    fake_avg = fake_tensor.mean(axis=0).squeeze(-1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].imshow(real_avg, aspect="auto", cmap="viridis")
    axes[0].set_title("Average Real Diary")
    axes[1].imshow(fake_avg, aspect="auto", cmap="viridis")
    axes[1].set_title("Average Synthetic Diary")
    plt.savefig(os.path.join(output_dir, "heatmap_comparison.png"), dpi=150)
    plt.close()

    # 4. Spell-Durations
    real_spells, fake_spells = emd_data
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    bins = np.arange(1, 49) - 0.5
    for act_idx in range(1, 10):
        ax = axes[act_idx - 1]
        ax.hist(
            real_spells[act_idx], bins=bins, density=True, alpha=0.5, label="Real", color="#1f77b4"
        )
        ax.hist(
            fake_spells[act_idx],
            bins=bins,
            density=True,
            alpha=0.5,
            label="Synthetic",
            color="#ff7f0e",
        )
        ax.set_title(f"{DIVISION_LABELS[act_idx]}", fontsize=10)
        ax.set_xlim(0.5, 24.5)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spell_durations.png"), dpi=150)
    plt.close()

    # 5. ROC Curve
    fpr, tpr, auc = roc_data
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Classifier ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "adversarial_roc.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────
# 9. MAIN RUNNER
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="../v4/tusgan_encode.npz")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="evaluation_v5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting TUS-GAN Ultimate Evaluation v5 on {device}...\n")

    # Load & Generate
    real_tensor, real_cond, real_dists, real_states = load_real_data(args.data)
    G, noise_dim = load_generator(args.checkpoint, args.data, device)

    eval_n = min(args.n_samples, real_tensor.shape[0])
    fake_tensor, fake_cond = generate_synthetic(
        G, real_cond, real_dists, real_states, noise_dim, eval_n, device
    )

    real_tensor = real_tensor[:eval_n]
    real_cond = real_cond[:eval_n]

    real_codes = decode_to_codes(real_tensor)
    fake_codes = decode_to_codes(fake_tensor)

    # 1. Standard Metrics
    m = compute_standard_metrics(real_codes, fake_codes)

    # 2. Spell-Duration (EMD)
    emd_dists, real_spells, fake_spells = evaluate_spells(real_codes, fake_codes)
    avg_emd = np.nanmean(list(emd_dists.values()))

    # 3. Logical Constraints
    logic = evaluate_logical_constraints(real_codes, fake_codes, real_cond, fake_cond)
    r_v, r_c, r_r = logic["real"]
    f_v, f_c, f_r = logic["fake"]

    # 4. Adversarial Validation
    auc, acc, fpr, tpr = run_adversarial_validation(real_tensor, fake_tensor, real_cond, fake_cond)

    # 5. Terminal Output
    print("\n" + "=" * 80)
    print(" 🏆 ULTIMATE EVALUATION RESULTS (v5)")
    print("=" * 80)
    print(f" 📊 Statistical Realism:")
    print(f"    - Jensen-Shannon Divergence (JSD)  : {m['jsd']:.6f}")
    print(f" ⏳ Temporal Realism:")
    print(f"    - Transition Matrix F-Norm         : {m['trans_fnorm']:.6f}")
    print(f"    - Average Spell-Duration EMD       : {avg_emd:.6f}")
    print(f" 🛡️ Logical Rules (Child Labor):")
    print(f"    - Real Violations                  : {r_v}/{r_c} ({r_r:.2f}%)")
    print(f"    - Synthetic Violations             : {f_v}/{f_c} ({f_r:.2f}%)")
    print(f" ⚔️ Adversarial Indistinguishability:")
    print(f"    - Random Forest Accuracy           : {acc:.4f} (Ideal: 0.5)")
    print(f"    - Random Forest AUC-ROC            : {auc:.4f} (Ideal: 0.5)")
    print("=" * 80)

    # 6. Save Plots
    print(f"\n💾 Saving ultimate visualizations to {args.output_dir}/")
    plot_all_visuals(
        m,
        (real_spells, fake_spells),
        (fpr, tpr, auc),
        real_tensor,
        fake_tensor,
        real_codes,
        fake_codes,
        args.output_dir,
    )

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
