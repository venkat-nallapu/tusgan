# -*- coding: utf-8 -*-
"""
TUS-GAN — Optimized Inference & Evaluation Script (v4)
======================================================
This script implements post-training optimizations (Truncation Trick & Rejection Sampling)
to squeeze the best possible statistical and logical realism out of the v4 checkpoint
without retraining the model.

Optimizations included:
1. Truncation Trick: Samples noise from a truncated normal distribution.
2. Rejection Sampling: Automatically discards generated samples that violate logical rules (Child Labor).
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import truncnorm

# ─────────────────────────────────────────────────────────────
# 1. CONSTANTS & ACTIVITY MAPPING
# ─────────────────────────────────────────────────────────────

DIVISION_LABELS = {
    1: "Employment",
    2: "Production",
    3: "Unpaid Domestic",
    4: "Unpaid Caregiving",
    5: "Unpaid Volunteer",
    6: "Learning",
    7: "Socializing & Religious",
    8: "Leisure & Sports",
    9: "Self-care & Maintenance",
}

MINUTES_PER_SLOT = 30


# ─────────────────────────────────────────────────────────────
# 2. HELPER — LOAD GENERATOR
# ─────────────────────────────────────────────────────────────

def load_generator(checkpoint_path: str, data_path: str, device: torch.device):
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from generator import Generator

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    data = np.load(data_path)
    actual_cond_dim = data["cond_vector"].shape[1]
    actual_num_districts = int(data["num_districts"])
    actual_num_states = int(data["num_states"])

    G = Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=actual_cond_dim,
        num_districts=actual_num_districts,
        num_states=actual_num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["g_base_channels"],
    ).to(device)

    g_state_key = "G_state_ema" if "G_state_ema" in ckpt else "G_state"
    G.load_state_dict(ckpt[g_state_key])
    G.eval()

    print(f"✅ Generator loaded from {checkpoint_path} (EMA weights)")
    return G, cfg


def load_real_data(data_path: str):
    data = np.load(data_path)
    return data["diary_tensor"], data["cond_vector"], data["district_ids"], data["state_ids"]


def decode_to_codes(tensor_9ch: np.ndarray) -> np.ndarray:
    return np.argmax(tensor_9ch, axis=1).squeeze(-1) + 1


# ─────────────────────────────────────────────────────────────
# 3. OPTIMIZED GENERATOR PIPELINE
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_synthetic_optimized(
    G, cond_vector, district_ids, state_ids, noise_dim, n_samples, device, batch_size=512, truncation=1.5
):
    print(f"\n🚀 Starting Optimized Inference Pipeline...")
    print(f"   - Truncation Trick active (cutoff={truncation})")
    print(f"   - Rejection Sampling active (enforcing 0.00% Child Labor)")
    
    valid_fakes = []
    total_valid = 0
    N = cond_vector.shape[0]

    while total_valid < n_samples:
        current_batch = min(batch_size, n_samples - total_valid + int(0.1 * n_samples))
        indices = np.random.choice(N, size=current_batch, replace=True)
        
        c = torch.from_numpy(cond_vector[indices]).float().to(device)
        d = torch.from_numpy(district_ids[indices]).long().to(device)
        s = torch.from_numpy(state_ids[indices]).long().to(device)
        
        # 1. Truncation Trick: Sample from truncated normal distribution
        z_np = truncnorm.rvs(-truncation, truncation, size=(current_batch, noise_dim))
        z = torch.from_numpy(z_np).float().to(device)

        # 2. Generation (Lower temp if supported)
        try:
            fake = G(z, c, d, s, temp=0.1)
        except TypeError:
            fake = G(z, c, d, s)

        fake_np = fake.cpu().numpy()
        c_np = c.cpu().numpy()

        # 3. Rejection Sampling (Child Labor Filter)
        codes = decode_to_codes(fake_np)
        child_mask = c_np[:, 0] == 1
        
        valid_mask = np.ones(current_batch, dtype=bool)
        has_employment = np.any(codes == 1, axis=1)
        invalid_children = child_mask & has_employment
        
        # Discard invalid samples
        valid_mask[invalid_children] = False

        good_fake = fake_np[valid_mask]
        
        if len(good_fake) > 0:
            valid_fakes.append(good_fake)
            total_valid += len(good_fake)
            print(f"   Generated {min(total_valid, n_samples)}/{n_samples} valid samples...", end="\r")

    final_fake = np.concatenate(valid_fakes, axis=0)[:n_samples]
    print(f"\n✅ Successfully collected {n_samples:,} strictly valid synthetic diaries.\n")
    return final_fake


# ─────────────────────────────────────────────────────────────
# 4. EVALUATION & VISUALIZATION
# ─────────────────────────────────────────────────────────────

def compute_numpy_transition_matrix(codes: np.ndarray) -> np.ndarray:
    c_t = codes[:, :-1] - 1
    c_tp1 = codes[:, 1:] - 1
    trans = np.zeros((9, 9), dtype=float)
    np.add.at(trans, (c_t, c_tp1), 1.0)
    row_sums = trans.sum(axis=1, keepdims=True)
    return np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums != 0)

def compute_metrics(real_codes, fake_codes):
    divisions = np.arange(1, 10)
    real_flat, fake_flat = real_codes.flatten(), fake_codes.flatten()
    
    real_freq = np.array([(real_flat == d).sum() for d in divisions], dtype=float)
    fake_freq = np.array([(fake_flat == d).sum() for d in divisions], dtype=float)

    real_pct = real_freq / real_flat.size * 100
    fake_pct = fake_freq / fake_flat.size * 100

    real_prob = real_freq / real_freq.sum()
    fake_prob = fake_freq / fake_freq.sum()
    jsd = float(jensenshannon(real_prob, fake_prob) ** 2)

    real_minutes = np.array([np.mean(np.sum(real_codes == d, axis=1)) * MINUTES_PER_SLOT for d in divisions])
    fake_minutes = np.array([np.mean(np.sum(fake_codes == d, axis=1)) * MINUTES_PER_SLOT for d in divisions])

    trans_real = compute_numpy_transition_matrix(real_codes)
    trans_fake = compute_numpy_transition_matrix(fake_codes)
    trans_fnorm = float(np.linalg.norm(trans_fake - trans_real, ord="fro"))

    return dict(
        divisions=divisions, real_pct=real_pct, fake_pct=fake_pct,
        jsd=jsd, real_minutes=real_minutes, fake_minutes=fake_minutes, trans_fnorm=trans_fnorm
    )

def print_metrics(m: dict):
    print("=" * 78)
    print("  OPTIMIZED EVALUATION RESULTS (v4)")
    print("=" * 78)
    print(f"\n  Jensen-Shannon Divergence (JSD):                {m['jsd']:.6f}")
    print(f"  Transition Matrix Frobenius Difference (F-norm): {m['trans_fnorm']:.6f}")
    print(f"  Logical Violations (Child Labor):                0.00% (Guaranteed by Rejection Sampling)")
    print("=" * 78)

def plot_sample_diaries(real_codes: np.ndarray, fake_codes: np.ndarray, output_dir: str):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharex=True, sharey=True)
    for i in range(5):
        axes[0, i].step(range(48), real_codes[i], where="post", color="#4C72B0", linewidth=1.2)
        axes[0, i].set_title(f"Real Diary {i + 1}", fontsize=9)
        axes[1, i].step(range(48), fake_codes[i], where="post", color="#DD8452", linewidth=1.2)
        axes[1, i].set_title(f"Optimized Synthetic Diary {i + 1}", fontsize=9)

    for ax in axes.flatten():
        ax.set_ylim(0.5, 9.5)
        ax.set_yticks(range(1, 10))
        ax.set_yticklabels([DIVISION_LABELS[d][:10] for d in range(1, 10)], fontsize=6)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.supxlabel("Time Slot (30 mins)")
    fig.supylabel("Activity Division")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "optimized_sample_diaries.png"), dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="tusgan_encode.npz")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--truncation", type=float, default=1.5, help="Standard deviations for truncation trick")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_optimized")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, cfg = load_generator(args.checkpoint, args.data, device)
    
    diary_tensor, cond_vector, district_ids, state_ids = load_real_data(args.data)
    
    fake_tensor = generate_synthetic_optimized(
        G, cond_vector, district_ids, state_ids, cfg["noise_dim"], args.n_samples, device, truncation=args.truncation
    )

    real_codes = decode_to_codes(diary_tensor)
    fake_codes = decode_to_codes(fake_tensor)

    metrics = compute_metrics(real_codes, fake_codes)
    print_metrics(metrics)
    
    plot_sample_diaries(real_codes, fake_codes, args.output_dir)
    print(f"\n📊 Plots saved to {args.output_dir}/optimized_sample_diaries.png")

if __name__ == "__main__":
    main()
