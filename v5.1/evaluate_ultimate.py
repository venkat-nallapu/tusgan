# -*- coding: utf-8 -*-
"""
TUS-GAN — Ultimate Evaluation Suite (v5.1 / Cell D)
===================================================
Unified evaluation pipeline tailored for TUS-GAN v5.1 (Transformer + Soft Penalty).
Evaluates all metrics required for Table 6 and Section 5 in the manuscript:
1. Statistical Realism (Jensen-Shannon Divergence, Time-Use Minutes).
2. Temporal Realism (Time-Slice Transition Matrix F-Norm, Spell-Duration EMD).
3. Logical Rules (Child Labor 3-Point Reporting: Empirical Base, Raw Model, Delivered).
4. Adversarial Indistinguishability (Random Forest Accuracy & ROC-AUC).
5. Diagnostic Figures (TRC-formatted diurnal participation curves, transition heatmaps, ROC).
"""

import os
import argparse
import itertools
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
    G, cond_vector, district_ids, state_ids, noise_dim, n_samples, device, batch_size=512, apply_hard_mask=False
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

        fake = G(z, c, d, s, temp=0.1, hard=True, return_soft=False, apply_hard_mask=apply_hard_mask)
        all_fakes.append(fake.cpu().numpy())

    fake_tensor = np.concatenate(all_fakes, axis=0)
    return fake_tensor, cond_vector[indices]


def decode_to_codes(tensor_9ch: np.ndarray) -> np.ndarray:
    # argmax along channel dimension (axis=1), returning 1 to 9
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
    overall_jsd = np.mean(jsd_per_slot)

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
        fnorms.append(np.linalg.norm(P_r - P_f, "fro"))

    mean_fnorm = np.mean(fnorms)

    # 3. Time Use Minutes
    r_mins = {}
    f_mins = {}
    for act in range(1, 10):
        r_mins[act] = np.mean(np.sum(real_codes == act, axis=1)) * MINUTES_PER_SLOT
        f_mins[act] = np.mean(np.sum(fake_codes == act, axis=1)) * MINUTES_PER_SLOT

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
    distances = {}
    for act in range(1, 10):
        r_dur = real_spells[act]
        f_dur = fake_spells[act]
        if len(r_dur) > 0 and len(f_dur) > 0:
            distances[act] = wasserstein_distance(r_dur, f_dur)
        else:
            distances[act] = float("nan")
    return distances, real_spells, fake_spells


def evaluate_child_violations(cond, codes):
    child_mask = cond[:, 0] == 1.0
    num_children = int(np.sum(child_mask))
    if num_children > 0:
        child_diaries = codes[child_mask]
        violations = int(np.sum(np.any(child_diaries == 1, axis=1)))
        return violations, num_children, (violations / num_children) * 100.0
    return 0, 0, 0.0


def run_adversarial_validation(real_tensor, fake_tensor, real_cond, fake_cond):
    X_real = np.hstack([real_cond, real_tensor.reshape(real_tensor.shape[0], -1)])
    X_fake = np.hstack([fake_cond, fake_tensor.reshape(fake_tensor.shape[0], -1)])

    y_real = np.ones(X_real.shape[0])
    y_fake = np.zeros(X_fake.shape[0])

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    return auc, acc, fpr, tpr


# ─────────────────────────────────────────────────────────────
# 4. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TUS-GAN v5.1 Ultimate Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pt", help="Path to checkpoint .pt")
    parser.add_argument("--data", type=str, default="tusgan_encode.npz", help="Path to real data .npz")
    parser.add_argument("--n_samples", type=int, default=50000, help="Number of diaries to generate for evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save exhibits")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Running TUS-GAN v5.1 Evaluation on {device}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data & Model
    real_tensor, real_cond, real_dists, real_states = load_real_data(args.data)
    G, noise_dim = load_generator(args.checkpoint, args.data, device)

    eval_n = min(args.n_samples, real_tensor.shape[0])

    # 2. Generate Raw Model Output (Soft Penalty Only, apply_hard_mask=False)
    print(f"\nGenerating {eval_n:,} raw synthetic diaries (soft penalty in training)...")
    fake_tensor_raw, fake_cond_raw = generate_synthetic(
        G, real_cond, real_dists, real_states, noise_dim, eval_n, device, apply_hard_mask=False
    )
    fake_codes_raw = decode_to_codes(fake_tensor_raw)

    # 3. Generate Delivered Output (Inference Guarantee, apply_hard_mask=True)
    print(f"Generating {eval_n:,} delivered synthetic diaries (with inference guarantee)...")
    fake_tensor_del, fake_cond_del = generate_synthetic(
        G, real_cond, real_dists, real_states, noise_dim, eval_n, device, apply_hard_mask=True
    )
    fake_codes_del = decode_to_codes(fake_tensor_del)

    real_sub = real_tensor[:eval_n]
    real_cond_sub = real_cond[:eval_n]
    real_codes = decode_to_codes(real_sub)

    # 4. Standard Metrics (computed on raw model to judge learned statistical manifold)
    m = compute_standard_metrics(real_codes, fake_codes_raw)
    emd_dists, _, _ = evaluate_spells(real_codes, fake_codes_raw)
    avg_emd = np.nanmean(list(emd_dists.values()))

    # 5. Three-Point Child Labor Violation Reporting (Section 5.4)
    r_v, r_c, r_rate = evaluate_child_violations(real_cond_sub, real_codes)
    raw_v, raw_c, raw_rate = evaluate_child_violations(fake_cond_raw, fake_codes_raw)
    del_v, del_c, del_rate = evaluate_child_violations(fake_cond_del, fake_codes_del)

    # 6. Adversarial Validation
    auc, acc, _, _ = run_adversarial_validation(real_sub, fake_tensor_raw, real_cond_sub, fake_cond_raw)

    # 7. Print Table 6 Style Summary
    print("\n" + "=" * 80)
    print(" 🏆 TUS-GAN v5.1 (CELL D) EVALUATION RESULTS — TABLE 6 METRICS")
    print("=" * 80)
    print(f" 📊 Statistical Fidelity:")
    print(f"    - Jensen-Shannon Divergence (JSD)       : {m['jsd']:.6f}  (Target: < 0.000100)")
    print(f" ⏳ Temporal Realism:")
    print(f"    - Transition Matrix F-Norm (4 Slices)   : {m['trans_fnorm']:.6f}")
    print(f"    - Average Spell-Duration EMD (mins)     : {avg_emd:.2f} mins")
    print(f"    - Employment Spell EMD                  : {emd_dists[1]:.2f} mins")
    print(f" 🛡️ Normative Constraint C1 (Child Labor 3-Point Reporting):")
    print(f"    - 1. Empirical Survey Base Rate         : {r_v:,}/{r_c:,} ({r_rate:.2f}%)")
    print(f"    - 2. Raw Model Output (Soft Penalty)    : {raw_v:,}/{raw_c:,} ({raw_rate:.2f}%)")
    print(f"    - 3. Delivered Output (Inference Mask)  : {del_v:,}/{del_c:,} ({del_rate:.2f}%)")
    print(f" ⚔️ Adversarial Indistinguishability (Random Forest):")
    print(f"    - Adversarial Accuracy                  : {acc*100:.2f}%  (Ideal: 50.0%)")
    print(f"    - Adversarial AUC-ROC                   : {auc:.4f}  (Ideal: 0.5000)")
    print("=" * 80)

    # Save metrics to text report
    report_path = os.path.join(args.output_dir, "table6_cell_d_results.txt")
    with open(report_path, "w") as f:
        f.write("TUS-GAN v5.1 (Cell D) Evaluation Summary\n")
        f.write(f"JSD: {m['jsd']:.6f}\n")
        f.write(f"Transition_FNorm: {m['trans_fnorm']:.6f}\n")
        f.write(f"Average_EMD: {avg_emd:.2f}\n")
        f.write(f"Employment_EMD: {emd_dists[1]:.2f}\n")
        f.write(f"Empirical_Child_Labor_Rate: {r_rate:.2f}%\n")
        f.write(f"Raw_Model_Child_Labor_Rate: {raw_rate:.2f}%\n")
        f.write(f"Delivered_Child_Labor_Rate: {del_rate:.2f}%\n")
        f.write(f"Adversarial_Accuracy: {acc:.4f}\n")
        f.write(f"Adversarial_AUC: {auc:.4f}\n")
    print(f"\n💾 Results exported to: {report_path}")


if __name__ == "__main__":
    main()
