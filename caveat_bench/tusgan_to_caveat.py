# -*- coding: utf-8 -*-
"""
tusgan_to_caveat.py: Translates ITUS 2019 microdata and TUS-GAN checkpoints
into Caveat-compatible CSV schedule and attribute tables.

Target Exhibits: Table 8 & Figure 9 in TUS-GAN manuscript.
Reference: Shone & Hillel (2025), Transportation Research Part C.
"""

import os
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────────────────────
# 1. CONSTANTS & ACTIVITY CLASSIFICATION
# ─────────────────────────────────────────────────────────────

# ICATUS 2016 9 major divisions mapped to lowercase identifier strings
DIVISION_NAMES = {
    1: "employment",
    2: "production_own_use",
    3: "unpaid_domestic",
    4: "unpaid_caregiving",
    5: "unpaid_volunteer",
    6: "learning_education",
    7: "socializing_religious",
    8: "culture_leisure",
    9: "self_care",
}

# ─────────────────────────────────────────────────────────────
# 2. RUN-LENGTH ENCODING: 48 SLOTS → EPISODIC SCHEDULES
# ─────────────────────────────────────────────────────────────

def slots_to_episodes(diary_matrix: np.ndarray, pid_offset: int = 0) -> pd.DataFrame:
    """
    Converts (N, 48) integer activity matrix (values 0-8 or 1-9) into an episodic DataFrame
    with continuous timestamps [0, 1440] minutes.
    """
    records = []
    N = diary_matrix.shape[0]

    # Detect whether input is 0-indexed (0..8) or 1-indexed (1..9)
    min_val = diary_matrix.min()
    offset = 1 if min_val == 0 else 0

    for i in range(N):
        pid = pid_offset + i
        slots = diary_matrix[i]

        start_slot = 0
        current_act = int(slots[0]) + offset

        for t in range(1, 48):
            act_t = int(slots[t]) + offset
            if act_t != current_act:
                records.append({
                    "pid": pid,
                    "act": DIVISION_NAMES[current_act],
                    "start": int(start_slot * 30),
                    "end": int(t * 30),
                    "duration": int((t - start_slot) * 30),
                })
                start_slot = t
                current_act = act_t

        # Final episode up to minute 1440
        records.append({
            "pid": pid,
            "act": DIVISION_NAMES[current_act],
            "start": int(start_slot * 30),
            "end": 1440,
            "duration": int((48 - start_slot) * 30),
        })

    df = pd.DataFrame(records)
    df["act"] = df["act"].astype("category")
    return df


def extract_attributes(cond_vector: np.ndarray, state_ids: np.ndarray, district_ids: np.ndarray, pid_offset: int = 0) -> pd.DataFrame:
    """
    Extracts person-level demographic attributes into Caveat attribute DataFrame.
    """
    N = cond_vector.shape[0]
    records = []

    # Demographics layout in cond_vector:
    # cond[0]: Age < 15 indicator (Child)
    # cond[7:10]: Gender (Male, Female, Transgender)
    # cond[46:48]: Sector (Rural, Urban)
    for i in range(N):
        pid = pid_offset + i
        is_child = int(cond_vector[i, 0] > 0.5)
        
        # Gender index (0: Male, 1: Female, 2: Transgender)
        gender_slice = cond_vector[i, 7:10]
        gender = int(np.argmax(gender_slice)) if len(gender_slice) == 3 and np.sum(gender_slice) > 0 else 0
        
        # Sector index (0: Rural, 1: Urban)
        sector_slice = cond_vector[i, 46:48]
        sector = int(np.argmax(sector_slice)) if len(sector_slice) == 2 and np.sum(sector_slice) > 0 else 0

        records.append({
            "pid": pid,
            "is_child": is_child,
            "gender": gender,
            "sector": sector,
            "state": int(state_ids[i]),
            "district": int(district_ids[i]),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 3. GENERATOR MODEL LOADER & SAMPLER
# ─────────────────────────────────────────────────────────────

def load_generator_module(gen_file_path: str):
    """Dynamically loads Generator class from a specific version folder."""
    spec = importlib.util.spec_from_file_location("dynamic_generator", gen_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Generator


@torch.no_grad()
def generate_diaries(G, cond_vector, district_ids, state_ids, noise_dim, n_samples, device, batch_size=512, version_cell="A", shuffle=False):
    """
    Synthesizes n_samples diaries using conditioning profiles.
    Returns decoded integer activity codes (n_samples, 48).
    When shuffle=False and n_samples == N, preserves 1:1 alignment with conditioning profiles.
    """
    N = cond_vector.shape[0]
    if shuffle or n_samples != N:
        indices = np.random.choice(N, size=n_samples, replace=(n_samples > N))
        cond_all = torch.from_numpy(cond_vector[indices]).float()
        dist_all = torch.from_numpy(district_ids[indices]).long()
        state_all = torch.from_numpy(state_ids[indices]).long()
        used_cond = cond_vector[indices]
        used_dist = district_ids[indices]
        used_state = state_ids[indices]
    else:
        cond_all = torch.from_numpy(cond_vector).float()
        dist_all = torch.from_numpy(district_ids).long()
        state_all = torch.from_numpy(state_ids).long()
        used_cond = cond_vector
        used_dist = district_ids
        used_state = state_ids

    all_codes = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        B = end - start
        z = torch.randn(B, noise_dim, device=device)
        c = cond_all[start:end].to(device)
        d = dist_all[start:end].to(device)
        s = state_all[start:end].to(device)

        if version_cell == "D":
            # v5.1 supports apply_hard_mask
            try:
                fake = G(z, c, d, s, temp=0.1, apply_hard_mask=False)
            except TypeError:
                fake = G(z, c, d, s)
        elif version_cell in ["A", "B"]:
            try:
                fake = G(z, c, d, s, temp=0.1)
            except TypeError:
                fake = G(z, c, d, s)
        else:
            fake = G(z, c, d, s)

        # If tuple returned (e.g. return_soft), unpack
        if isinstance(fake, tuple):
            fake = fake[0]

        # Decode (B, 9, 48, 1) -> (B, 48)
        codes = torch.argmax(fake, dim=1).squeeze(-1).cpu().numpy()  # values 0..8
        all_codes.append(codes)

    return np.concatenate(all_codes, axis=0), used_cond, used_dist, used_state


# ─────────────────────────────────────────────────────────────
# 4. MAIN EXPORT PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TUS-GAN to Caveat Data & Checkpoint Exporter")
    parser.add_argument("--npz_path", type=str, default="../data/tusgan_encode.npz", help="Path to ITUS 2019 NPZ")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for CSVs")
    parser.add_argument("--sample_size", type=int, default=50000, help="Number of schedules to export for benchmarking")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction for evaluation test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--export_checkpoints", action="store_true", default=True, help="Also export synthetic schedules from trained checkpoints")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve dataset path
    npz_path = args.npz_path
    if not os.path.exists(npz_path):
        candidates = [
            "../data/tusgan_encode.npz",
            "data/tusgan_encode.npz",
            "/home/venkat/projects/tusgan/data/tusgan_encode.npz",
            "/home/venkat/projects/tusgan/v4/tusgan_encode.npz"
        ]
        for c in candidates:
            if os.path.exists(c):
                npz_path = c
                break

    print(f"📦 Loading real ITUS 2019 dataset from: {npz_path}")
    data = np.load(npz_path)
    real_diaries = data["diary_tensor"]  # (N, 9, 48, 1)
    cond_vector = data["cond_vector"]    # (N, 83)
    district_ids = data["district_ids"]  # (N,)
    state_ids = data["state_ids"]        # (N,)
    num_districts = int(data["num_districts"])
    num_states = int(data["num_states"])
    actual_cond_dim = cond_vector.shape[1]

    total_records = real_diaries.shape[0]
    eval_n = min(args.sample_size, total_records)
    print(f"   Total survey records: {total_records:,} | Benchmark subset: {eval_n:,}")

    # Subsample real data
    subset_indices = np.random.choice(total_records, size=eval_n, replace=False)
    real_sub = real_diaries[subset_indices]
    cond_sub = cond_vector[subset_indices]
    dist_sub = district_ids[subset_indices]
    state_sub = state_ids[subset_indices]

    # Decode real schedules to discrete integer codes 0..8
    real_codes = np.argmax(real_sub, axis=1).squeeze(-1)

    # Split into train and test sets
    n_test = int(eval_n * args.test_split)
    n_train = eval_n - n_test

    train_codes = real_codes[:n_train]
    test_codes = real_codes[n_train:]
    
    train_cond = cond_sub[:n_train]
    test_cond = cond_sub[n_train:]
    train_dist = dist_sub[:n_train]
    test_dist = dist_sub[n_train:]
    train_state = state_sub[:n_train]
    test_state = state_sub[n_train:]

    # Convert to Caveat CSVs
    print("🔄 Converting Real ITUS schedules to episodic CSV...")
    df_real_train = slots_to_episodes(train_codes, pid_offset=0)
    df_real_train_attrs = extract_attributes(train_cond, train_state, train_dist, pid_offset=0)

    df_real_test = slots_to_episodes(test_codes, pid_offset=n_train)
    df_real_test_attrs = extract_attributes(test_cond, test_state, test_dist, pid_offset=n_train)

    df_real_train.to_csv(os.path.join(args.output_dir, "real_train_schedules.csv"), index=False)
    df_real_train_attrs.to_csv(os.path.join(args.output_dir, "real_train_attributes.csv"), index=False)
    df_real_test.to_csv(os.path.join(args.output_dir, "real_test_schedules.csv"), index=False)
    df_real_test_attrs.to_csv(os.path.join(args.output_dir, "real_test_attributes.csv"), index=False)

    print(f"   ✓ Saved {len(df_real_train):,} train episodes → real_train_schedules.csv")
    print(f"   ✓ Saved {len(df_real_test):,} test episodes → real_test_schedules.csv")

    # ─────────────────────────────────────────────────────────────
    # EXPORT TUS-GAN CHECKPOINTS
    # ─────────────────────────────────────────────────────────────
    if args.export_checkpoints:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🚀 Exporting synthetic schedules from TUS-GAN checkpoints on device: {device}...")

        checkpoints_to_export = [
            ("Cell A (v4 Conv Soft)", "/home/venkat/projects/tusgan/v4/checkpoints/final.pt", "/home/venkat/projects/tusgan/v4/generator.py", "synth_cell_a_schedules.csv", "A"),
            ("Cell B (v5 Trans Hard)", "/home/venkat/projects/tusgan/v5/checkpoints/final.pt", "/home/venkat/projects/tusgan/v5/generator.py", "synth_cell_b_schedules.csv", "B"),
            ("Cell D (v5.1 Trans Soft)", "/home/venkat/projects/tusgan/v5.1/checkpoints_seed42/final.pt", "/home/venkat/projects/tusgan/v5.1/generator.py", "synth_cell_d_schedules.csv", "D"),
        ]

        # Use test set conditioning attributes to match test evaluation
        n_synth = n_test

        for model_name, ckpt_path, gen_file, out_csv, cell_tag in checkpoints_to_export:
            if not os.path.exists(ckpt_path):
                print(f"   ⚠️ Skipping {model_name}: Checkpoint not found at {ckpt_path}")
                continue

            print(f"\n   Sampling from {model_name}...")
            GenCls = load_generator_module(gen_file)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg = ckpt.get("config", {})

            G = GenCls(
                noise_dim=cfg.get("noise_dim", 128),
                cond_dim=actual_cond_dim,
                num_districts=num_districts,
                num_states=num_states,
                district_embed_dim=cfg.get("district_embed_dim", 16),
                state_embed_dim=cfg.get("state_embed_dim", 8),
                base_channels=cfg.get("g_base_channels", 256),
            ).to(device)

            g_state_key = "G_state_ema" if "G_state_ema" in ckpt else "G_state"
            G.load_state_dict(ckpt[g_state_key])
            G.eval()

            # Generate synthetic codes with 1:1 aligned conditioning profiles
            fake_codes, s_cond, s_dist, s_state = generate_diaries(
                G, test_cond, test_dist, test_state,
                noise_dim=cfg.get("noise_dim", 128),
                n_samples=n_synth,
                device=device,
                version_cell=cell_tag,
                shuffle=False
            )

            df_synth = slots_to_episodes(fake_codes, pid_offset=n_train)
            df_synth.to_csv(os.path.join(args.output_dir, out_csv), index=False)
            print(f"   ✓ Generated {len(df_synth):,} episodes for {model_name} → {out_csv}")

            # Export matching synthetic attributes file for 1:1 alignment
            attr_csv = out_csv.replace("_schedules.csv", "_attributes.csv")
            df_real_test_attrs.to_csv(os.path.join(args.output_dir, attr_csv), index=False)
            print(f"   ✓ Mirrored matching attributes → {attr_csv}")

    print("\n✅ Data export complete. Ready for ActVAE training and Caveat evaluation!\n")


if __name__ == "__main__":
    main()
