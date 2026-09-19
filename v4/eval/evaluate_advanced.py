# -*- coding: utf-8 -*-
"""
TUS-GAN — Advanced Evaluation Pipeline (v3)
============================================
Implements:
1. Spell-Duration Distribution Analysis:
   Extracts contiguous blocks of activities (e.g. consecutive hours of sleep or work)
   and computes the Wasserstein (Earth Mover's) Distance between real and synthetic distributions.
2. Adversarial Validation Classifier:
   Trains a Random Forest classifier to distinguish between real and synthetic diaries
   conditioned on demographics. Lower classifier performance (AUC-ROC close to 0.5)
   indicates higher realism and indistinguishability.
"""

import os
import argparse
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report

# Import Generator structure
from generator import Generator

# Mapping of channel indices (1-based) to activity names
ACTIVITY_NAMES = {
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

# ─────────────────────────────────────────────────────────────
# 1. HELPER — LOAD GENERATOR & DATASET
# ─────────────────────────────────────────────────────────────


def load_generator(checkpoint_path: str, device: torch.device):
    """Load the Generator model from a checkpoint, preferring EMA weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Infer architecture params from checkpoint config if saved
    cfg = ckpt.get("config", {})
    noise_dim = cfg.get("noise_dim", 128)
    cond_dim = cfg.get("cond_dim", 83)
    district_embed_dim = cfg.get("district_embed_dim", 16)
    state_embed_dim = cfg.get("state_embed_dim", 8)
    g_base_channels = cfg.get("g_base_channels", 256)

    # Instantiate Generator
    G = Generator(
        noise_dim=noise_dim,
        cond_dim=cond_dim,
        district_embed_dim=district_embed_dim,
        state_embed_dim=state_embed_dim,
        base_channels=g_base_channels,
    ).to(device)

    # Load state dict (prefer EMA weights)
    g_state_key = "G_state_ema" if "G_state_ema" in ckpt else "G_state"
    G.load_state_dict(ckpt[g_state_key])
    G.eval()
    print(f"✅ Loaded Generator from checkpoint (Weights: {g_state_key})")

    return G, noise_dim


def load_real_data(data_path: str):
    """Load the pre-encoded NPZ dataset."""
    if not os.path.exists(data_path):
        alt = os.path.join("wgan-gp", data_path)
        if os.path.exists(alt):
            data_path = alt
        else:
            raise FileNotFoundError(f"Dataset NPZ not found at: {data_path}")

    data = np.load(data_path)
    print(f"✅ Loaded real data from {data_path} ({data['diary_tensor'].shape[0]:,} records)")
    return (
        data["diary_tensor"],  # (N, 9, 48, 1)
        data["cond_vector"],  # (N, 83)
        data["district_ids"],  # (N,)
        data["state_ids"],  # (N,)
    )


@torch.no_grad()
def generate_synthetic(G, cond, dists, states, noise_dim, n_samples, device, batch_size=512):
    """Generate synthetic diaries matched to real demographics."""
    N = cond.shape[0]
    indices = np.random.choice(N, size=n_samples, replace=True)

    cond_all = torch.from_numpy(cond[indices]).float()
    dist_all = torch.from_numpy(dists[indices]).long()
    state_all = torch.from_numpy(states[indices]).long()

    all_fakes = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        z = torch.randn(end - start, noise_dim, device=device)
        c = cond_all[start:end].to(device)
        d = dist_all[start:end].to(device)
        s = state_all[start:end].to(device)
        fake = G(z, c, d, s)
        all_fakes.append(fake.cpu().numpy())

    return np.concatenate(all_fakes, axis=0), cond[indices], dists[indices], states[indices]


def decode_to_codes(tensor_9ch: np.ndarray) -> np.ndarray:
    """Decode a (N, 9, 48, 1) one-hot tensor into integer activity codes (N, 48) [1..9]."""
    return np.argmax(tensor_9ch, axis=1).squeeze(-1) + 1


# ─────────────────────────────────────────────────────────────
# 2. SPELL-DURATION ANALYSIS
# ─────────────────────────────────────────────────────────────


def extract_spells(sequences: np.ndarray):
    """
    Extract contiguous activity block lengths (in 30-min intervals)
    for each activity division.
    """
    spells = {i: [] for i in range(1, 10)}
    for seq in sequences:
        for activity, group in itertools.groupby(seq):
            length = len(list(group))
            spells[activity].append(length)
    return spells


def evaluate_spells(real_codes: np.ndarray, fake_codes: np.ndarray, output_dir: str):
    """Compute spell distributions, Wasserstein Distance, and save comparison plot."""
    print("\n⏳ Analyzing Activity Spell-Durations...")
    real_spells = extract_spells(real_codes)
    fake_spells = extract_spells(fake_codes)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    distances = {}

    for act_idx in range(1, 10):
        ax = axes[act_idx - 1]
        r_dur = real_spells[act_idx]
        f_dur = fake_spells[act_idx]

        # Calculate Wasserstein Distance (EMD)
        if len(r_dur) > 0 and len(f_dur) > 0:
            w_dist = wasserstein_distance(r_dur, f_dur)
        else:
            w_dist = float("nan")

        distances[act_idx] = w_dist

        # Plot histograms
        # Max spell is 48 steps (24 hours)
        bins = np.arange(1, 49) - 0.5
        ax.hist(
            r_dur,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Real",
            color="#1f77b4",
            edgecolor="k",
            align="mid",
        )
        ax.hist(
            f_dur,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Synthetic",
            color="#ff7f0e",
            edgecolor="k",
            align="mid",
        )

        ax.set_title(f"{ACTIVITY_NAMES[act_idx]}\nEMD (Wasserstein): {w_dist:.4f}", fontsize=10)
        ax.set_xlabel("Spell Duration (30-min intervals)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_xlim(0.5, 24.5)  # Focus visual on spells up to 12 hours for readability
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle(
        "Spell-Duration Distribution Comparison (Real vs. Synthetic)", fontsize=16, weight="bold"
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spell_duration_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"📊 Spell comparison plot saved → {plot_path}")
    print("\n📝 Spell-Duration Earth Mover's Distance (EMD) (Lower is better):")
    for act_idx, dist in distances.items():
        print(f"   - {ACTIVITY_NAMES[act_idx]:<25} : {dist:.4f}")

    return distances


# ─────────────────────────────────────────────────────────────
# 3. ADVERSARIAL VALIDATION CLASSIFIER
# ─────────────────────────────────────────────────────────────


def train_adversarial_validation(
    real_diaries: np.ndarray,
    fake_diaries: np.ndarray,
    real_cond: np.ndarray,
    fake_cond: np.ndarray,
    output_dir: str,
):
    """
    Trains a Random Forest classifier to distinguish between real and synthetic diaries
    conditioned on demographics.
    """
    print("\n⚔️ Training Adversarial Validation Classifier...")

    # Flatten diaries: shape (N, 9 * 48) = (N, 432)
    N_real = real_diaries.shape[0]
    N_fake = fake_diaries.shape[0]

    X_real_diaries = real_diaries.reshape(N_real, -1)
    X_fake_diaries = fake_diaries.reshape(N_fake, -1)

    # Combine with demographics: shape (N, 432 + 83)
    X_real = np.hstack([real_cond, X_real_diaries])
    X_fake = np.hstack([fake_cond, X_fake_diaries])

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([np.ones(N_real), np.zeros(N_fake)])  # 1 = Real, 0 = Synthetic

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    print("\n📊 Adversarial Validation Performance Summary:")
    print(f"   - Classifier Accuracy: {acc:.4f} (Ideal: 0.5000)")
    print(f"   - Classifier AUC-ROC:  {auc:.4f} (Ideal: 0.5000)")
    print("\n   Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Synthetic", "Real"]))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Classifier ROC (AUC = {auc:.3f})")
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Ideal / Random (AUC = 0.500)"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Adversarial Validation ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(linestyle="--", alpha=0.5)

    plot_path = os.path.join(output_dir, "adversarial_validation_roc.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 ROC Curve saved → {plot_path}")

    # Interpret results
    print("\n💡 Interpretation:")
    if auc < 0.60:
        print(
            "   🟢 SUCCESS: The classifier cannot distinguish synthetic diaries from real ones. High realism!"
        )
    elif auc < 0.80:
        print(
            "   🟡 MODERATE: The classifier has some ability to tell them apart. Inspect sequence logic."
        )
    else:
        print(
            "   🔴 WARNING: The classifier easily distinguishes synthetic from real. Check for artifact leaks."
        )

    return auc, acc


# ─────────────────────────────────────────────────────────────
# 4. LOGICAL CONSTRAINT VALIDATION
# ─────────────────────────────────────────────────────────────


def evaluate_logical_constraints(
    real_codes: np.ndarray, fake_codes: np.ndarray, real_cond: np.ndarray, fake_cond: np.ndarray
):
    """
    Check logical consistency constraints on real and synthetic diaries:
    1. Child Labor: No Employment (code 1) for individuals aged < 15 (cond_vector index 0 == 1).
    """
    print("\n🧐 Validating Logical Constraints...")

    # Real
    real_child_mask = real_cond[:, 0] == 1
    num_real_children = np.sum(real_child_mask)
    if num_real_children > 0:
        real_child_diaries = real_codes[real_child_mask]
        # Check if code 1 (Employment) is present in any slot
        real_violations = np.sum(np.any(real_child_diaries == 1, axis=1))
        real_violation_rate = (real_violations / num_real_children) * 100
    else:
        real_violations = 0
        real_violation_rate = 0.0

    # Fake
    fake_child_mask = fake_cond[:, 0] == 1
    num_fake_children = np.sum(fake_child_mask)
    if num_fake_children > 0:
        fake_child_diaries = fake_codes[fake_child_mask]
        fake_violations = np.sum(np.any(fake_child_diaries == 1, axis=1))
        fake_violation_rate = (fake_violations / num_fake_children) * 100
    else:
        fake_violations = 0
        fake_violation_rate = 0.0

    print(f"   - Child Labor Constraint (No work for Age < 15):")
    print(
        f"     • Real:      {real_violations}/{num_real_children} children violated ({real_violation_rate:.2f}%)"
    )
    print(
        f"     • Synthetic: {fake_violations}/{num_fake_children} children violated ({fake_violation_rate:.2f}%)"
    )

    return {
        "real_violations": real_violations,
        "real_children": num_real_children,
        "fake_violations": fake_violations,
        "fake_children": num_fake_children,
    }


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TUS-GAN Advanced Evaluation Pipeline")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="2019/img-encode/tusgan_encode.npz",
        help="Path to real dataset (.npz)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation charts",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")

    # 1. Load resources
    real_diary_tensor, real_cond, real_dists, real_states = load_real_data(args.data)
    G, noise_dim = load_generator(args.checkpoint, device)

    # Match evaluation sample size
    eval_n = min(args.n_samples, real_diary_tensor.shape[0])

    # 2. Generate matches
    fake_diary_tensor, fake_cond, _, _ = generate_synthetic(
        G, real_cond, real_dists, real_states, noise_dim, eval_n, device
    )

    # Slice real data down to match sample size exactly
    real_diary_tensor = real_diary_tensor[:eval_n]
    real_cond = real_cond[:eval_n]

    # Decode to integer sequences for spell analysis
    real_codes = decode_to_codes(real_diary_tensor)
    fake_codes = decode_to_codes(fake_diary_tensor)

    # Run spell evaluation
    evaluate_spells(real_codes, fake_codes, args.output_dir)

    # Run logical constraints evaluation
    evaluate_logical_constraints(real_codes, fake_codes, real_cond, fake_cond)

    # Run classifier validation
    train_adversarial_validation(
        real_diary_tensor, fake_diary_tensor, real_cond, fake_cond, args.output_dir
    )

    print("\n\u2728 Advanced evaluation complete!\n")
