# -*- coding: utf-8 -*-
"""
evaluate_sweep.py: Metric Evaluator for Section 6.8 Sensitivity Sweep
====================================================================
Computes the four essential Table 9 metrics for each lambda_rule checkpoint:
1. Raw child labor violation rate (%)
2. Delivered violation rate (%) after rejection sampling
3. Macro Jensen-Shannon Divergence (JSD)
4. Adversarial Random Forest Accuracy (%) & AUC-ROC
5. Inference rejection discard rate (%)

Outputs structured JSON and CSV records.
"""

import os
import sys
import argparse
import json
import csv
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import truncnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Import v4 models
V4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../v4"))
sys.path.insert(0, V4_DIR)
from generator import Generator

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


def load_generator_model(checkpoint_path: str, data_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

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
    print(f"✅ Loaded Generator from {checkpoint_path} (weights: {g_state_key})")
    return G, cfg


def decode_to_codes(tensor_9ch: np.ndarray) -> np.ndarray:
    return np.argmax(tensor_9ch, axis=1).squeeze(-1) + 1


@torch.no_grad()
def generate_raw_samples(G, cond_vec, dist_ids, state_ids, noise_dim, n_samples, device, batch_size=512):
    N = cond_vec.shape[0]
    indices = np.random.choice(N, size=n_samples, replace=True)
    c_all = torch.from_numpy(cond_vec[indices]).float()
    d_all = torch.from_numpy(dist_ids[indices]).long()
    s_all = torch.from_numpy(state_ids[indices]).long()

    fakes = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        B = end - start
        z = torch.randn(B, noise_dim, device=device)
        c = c_all[start:end].to(device)
        d = d_all[start:end].to(device)
        s = s_all[start:end].to(device)
        fake = G(z, c, d, s)
        fakes.append(fake.cpu().numpy())

    return np.concatenate(fakes, axis=0), cond_vec[indices]


@torch.no_grad()
def generate_rejection_samples(G, cond_vec, dist_ids, state_ids, noise_dim, n_samples, device, batch_size=512, truncation=1.5):
    """
    Rejection sampling: discards any child diary generating Employment (code 1).
    Returns strictly valid fake diaries, their corresponding conditions, and discard rate.
    """
    N = cond_vec.shape[0]
    valid_fakes = []
    valid_conds = []
    total_valid = 0
    total_generated = 0

    while total_valid < n_samples:
        current_batch = min(batch_size, n_samples - total_valid + 500)
        indices = np.random.choice(N, size=current_batch, replace=True)

        c = torch.from_numpy(cond_vec[indices]).float().to(device)
        d = torch.from_numpy(dist_ids[indices]).long().to(device)
        s = torch.from_numpy(state_ids[indices]).long().to(device)

        z_np = truncnorm.rvs(-truncation, truncation, size=(current_batch, noise_dim))
        z = torch.from_numpy(z_np).float().to(device)

        fake = G(z, c, d, s)
        fake_np = fake.cpu().numpy()
        c_np = c.cpu().numpy()

        codes = decode_to_codes(fake_np)
        child_mask = c_np[:, 0] == 1
        has_employment = np.any(codes == 1, axis=1)
        invalid = child_mask & has_employment

        valid_mask = ~invalid
        good_fake = fake_np[valid_mask]
        good_cond = c_np[valid_mask]

        total_generated += current_batch
        if len(good_fake) > 0:
            valid_fakes.append(good_fake)
            valid_conds.append(good_cond)
            total_valid += len(good_fake)

    final_fake = np.concatenate(valid_fakes, axis=0)[:n_samples]
    final_cond = np.concatenate(valid_conds, axis=0)[:n_samples]
    discard_rate = ((total_generated - n_samples) / max(1, total_generated)) * 100.0
    return final_fake, final_cond, discard_rate


def compute_macro_jsd(real_codes: np.ndarray, fake_codes: np.ndarray) -> float:
    divisions = np.arange(1, 10)
    real_flat = real_codes.flatten()
    fake_flat = fake_codes.flatten()

    real_freq = np.array([(real_flat == d).sum() for d in divisions], dtype=float)
    fake_freq = np.array([(fake_flat == d).sum() for d in divisions], dtype=float)

    real_prob = real_freq / real_freq.sum()
    fake_prob = fake_freq / fake_freq.sum()
    return float(jensenshannon(real_prob, fake_prob) ** 2)


def run_adversarial_validation(real_diaries: np.ndarray, fake_diaries: np.ndarray, real_cond: np.ndarray, fake_cond: np.ndarray, n_rf: int = 10000):
    """Random Forest classifier on flattened diary (432) + demographics (82)."""
    n_use = min(len(real_diaries), len(fake_diaries), n_rf)
    r_idx = np.random.choice(len(real_diaries), size=n_use, replace=False)
    f_idx = np.random.choice(len(fake_diaries), size=n_use, replace=False)

    real_X = np.hstack([real_diaries[r_idx].reshape(n_use, -1), real_cond[r_idx]])
    fake_X = np.hstack([fake_diaries[f_idx].reshape(n_use, -1), fake_cond[f_idx]])

    X = np.vstack([real_X, fake_X])
    y = np.concatenate([np.ones(n_use), np.zeros(n_use)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = float(roc_auc_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred) * 100.0)
    return acc, auc


def evaluate_checkpoint(checkpoint_path: str, data_path: str, lambda_rule: float, n_samples: int = 50000, skip_rf: bool = False, device_str: str = None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n===============================================================================")
    print(f"📊 Evaluating Sensitivity Checkpoint: λ_rule = {lambda_rule}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Data      : {data_path}")
    print(f"   Samples   : {n_samples:,} | Device: {device}")
    print(f"===============================================================================")

    G, cfg = load_generator_model(checkpoint_path, data_path, device)
    data = np.load(data_path)
    real_diaries = data["diary_tensor"]
    real_cond = data["cond_vector"]
    real_dist = data["district_ids"]
    real_state = data["state_ids"]
    noise_dim = cfg.get("noise_dim", 128)

    # 1. Raw Generation & Violations
    print("Generating raw unconstrained synthetic diaries...")
    fake_raw, fake_cond = generate_raw_samples(G, real_cond, real_dist, real_state, noise_dim, n_samples, device)
    fake_raw_codes = decode_to_codes(fake_raw)

    real_sample_idx = np.random.choice(len(real_diaries), size=min(n_samples, len(real_diaries)), replace=False)
    real_sample_codes = decode_to_codes(real_diaries[real_sample_idx])

    child_mask = fake_cond[:, 0] == 1
    num_children = int(np.sum(child_mask))
    if num_children > 0:
        child_codes = fake_raw_codes[child_mask]
        violations = int(np.sum(np.any(child_codes == 1, axis=1)))
        raw_violation_rate = (violations / num_children) * 100.0
    else:
        violations, raw_violation_rate = 0, 0.0

    print(f"   • Raw Child Labor Violations: {violations}/{num_children} ({raw_violation_rate:.2f}%)")

    # 2. Macro JSD
    jsd = compute_macro_jsd(real_sample_codes, fake_raw_codes)
    print(f"   • Macro Jensen-Shannon Divergence: {jsd:.6f}")

    # 3. Rejection Sampling & Discard Rate
    print("Running rejection sampling...")
    fake_opt, opt_cond, discard_rate = generate_rejection_samples(G, real_cond, real_dist, real_state, noise_dim, min(10000, n_samples), device)
    fake_opt_codes = decode_to_codes(fake_opt)
    opt_child_mask = opt_cond[:, 0] == 1
    num_opt_children = int(np.sum(opt_child_mask))
    if num_opt_children > 0:
        opt_child_codes = fake_opt_codes[opt_child_mask]
        delivered_violations = int(np.sum(np.any(opt_child_codes == 1, axis=1)))
        delivered_violation_rate = (delivered_violations / num_opt_children) * 100.0
    else:
        delivered_violations, delivered_violation_rate = 0, 0.0
    print(f"   • Delivered Child Violations: {delivered_violations}/{num_opt_children} ({delivered_violation_rate:.2f}%) [Discard rate: {discard_rate:.2f}%]")

    # 4. Adversarial Validation
    if not skip_rf:
        print("Training Adversarial Random Forest Classifier...")
        rf_acc, rf_auc = run_adversarial_validation(real_diaries, fake_raw, real_cond, fake_cond, n_rf=min(10000, n_samples))
        print(f"   • Adversarial Accuracy: {rf_acc:.2f}% | AUC-ROC: {rf_auc:.4f}")
    else:
        rf_acc, rf_auc = None, None

    result = {
        "lambda_rule": float(lambda_rule),
        "raw_violation_rate_pct": round(raw_violation_rate, 4),
        "delivered_violation_rate_pct": round(delivered_violation_rate, 4),
        "macro_jsd": round(jsd, 8),
        "adversarial_accuracy_pct": round(rf_acc, 2) if rf_acc is not None else None,
        "adversarial_auc_roc": round(rf_auc, 4) if rf_auc is not None else None,
        "discard_rate_pct": round(discard_rate, 4),
        "n_samples": n_samples,
        "checkpoint": checkpoint_path,
    }
    return result


def save_and_merge_results(new_record: dict, json_path: str, csv_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    records = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Update or append record matching lambda_rule
    updated = False
    for i, r in enumerate(records):
        if abs(r["lambda_rule"] - new_record["lambda_rule"]) < 1e-4:
            records[i] = new_record
            updated = True
            break
    if not updated:
        records.append(new_record)

    records.sort(key=lambda x: x["lambda_rule"])

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"✅ Updated JSON results: {json_path}")

    # Write CSV
    if records:
        keys = list(records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        print(f"✅ Updated CSV results: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sensitivity Sweep Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--lambda_rule", type=float, required=True, help="Soft penalty weight")
    parser.add_argument("--data", type=str, default="data/tusgan_encode.npz")
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--skip_rf", action="store_true", help="Skip Random Forest adversarial validation")
    parser.add_argument("--output_json", type=str, default="experiments/sensitivity_sweep/results_sensitivity_sweep.json")
    parser.add_argument("--output_csv", type=str, default="experiments/sensitivity_sweep/results_sensitivity_sweep.csv")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        for candidate in ["v4/tusgan_encode.npz", "../data/tusgan_encode.npz", "../../data/tusgan_encode.npz"]:
            if os.path.exists(candidate):
                data_path = candidate
                break

    result = evaluate_checkpoint(
        args.checkpoint,
        data_path,
        args.lambda_rule,
        n_samples=args.n_samples,
        skip_rf=args.skip_rf,
        device_str=args.device,
    )
    save_and_merge_results(result, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
