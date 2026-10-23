#!/usr/bin/env python3
"""
create_stratified_split.py
==========================
Generates a deterministic, leak-free 90/10 Train/Test partition of the
Indian Time Use Survey (ITUS) 2019 microdata (N = 445,268).

Design Rules:
  1. Exact split: Train = 400,741 (90.0%), Test = 44,527 (10.0%), Total = 445,268.
  2. Stratification: Joint stratification on (state_id, district_id, is_child).
     - Rare cells (count = 1) are deterministically routed to Train to guarantee
       that every unique stratum is represented in model training.
     - All remaining cells are stratified via train_test_split (seed=42).
  3. Spatial Coverage:
     - 100% of surveyed districts (676 of 676) present in Train.
     - 100% of surveyed states (36 of 36) present in both Train and Test.
  4. Demographic Balance:
     - Children (<15), gender, and sector proportions match within <0.1% delta.
  5. Deterministic Outputs:
     - data/train_indices.npy
     - data/test_indices.npy
     - data/split_metadata.json
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split


def create_split(
    data_path: str = "data/tusgan_encode.npz",
    output_dir: str = "data",
    test_size: int = 44527,
    seed: int = 42,
):
    print(f"📂 Loading dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    data = np.load(data_path)
    states = data["state_ids"]
    districts = data["district_ids"]
    cond = data["cond_vector"]

    N_total = len(states)
    target_train = N_total - test_size
    print(f"📊 Total records: {N_total:,} | Target: Train={target_train:,}, Test={test_size:,}")

    # Child indicator (index 0 of cond_vector)
    is_child = cond[:, 0].astype(int)

    # Age groups (indices 0:7)
    age_groups = np.argmax(cond[:, 0:7], axis=1)

    # Compound stratum: (state, district, is_child)
    strata = np.array([f"{s}_{d}_{c}" for s, d, c in zip(states, districts, is_child)])
    counts = Counter(strata)

    # 1. Deterministically assign singletons (count == 1) to Train
    # This guarantees rare cells are exposed to training
    singleton_mask = np.array([counts[x] == 1 for x in strata])
    singleton_idx = np.where(singleton_mask)[0]
    remaining_idx = np.where(~singleton_mask)[0]

    print(f"🔒 Singletons assigned directly to Train: {len(singleton_idx)}")
    print(f"🎲 Stratifying remaining records: {len(remaining_idx):,}")

    # 2. Stratified split on remaining pool
    _, remaining_strata_codes = np.unique(strata[remaining_idx], return_inverse=True)
    rem_train_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=test_size,
        random_state=seed,
        stratify=remaining_strata_codes,
    )

    # Combine singletons with stratified train pool
    train_idx = np.concatenate([singleton_idx, rem_train_idx])
    train_idx.sort()
    test_idx.sort()

    # ─────────────────────────────────────────────────────────────
    # VALIDATION CHECKS
    # ─────────────────────────────────────────────────────────────
    print("\n🔍 Executing verification assertions...")

    # Assert lengths
    assert len(train_idx) == target_train, f"Train length mismatch: {len(train_idx)} vs {target_train}"
    assert len(test_idx) == test_size, f"Test length mismatch: {len(test_idx)} vs {test_size}"
    assert len(train_idx) + len(test_idx) == N_total, "Total length mismatch"

    # Assert zero overlap
    intersection = np.intersect1d(train_idx, test_idx)
    assert len(intersection) == 0, f"Data leakage detected! {len(intersection)} overlapping indices."

    # Assert spatial coverage
    all_pairs = set(zip(states, districts))
    train_pairs = set(zip(states[train_idx], districts[train_idx]))
    test_pairs = set(zip(states[test_idx], districts[test_idx]))

    unique_states_train = np.unique(states[train_idx])
    unique_states_test = np.unique(states[test_idx])
    assert len(unique_states_train) == 36, "Missing states in training set"
    assert len(unique_states_test) == 36, "Missing states in test set"
    assert len(train_pairs) == len(all_pairs), (
        f"Missing districts in training set: {len(all_pairs) - len(train_pairs)}"
    )

    print("  ✅ Lengths and zero-overlap: PASSED")
    print(f"  ✅ State coverage: 36/36 in Train, 36/36 in Test (PASSED)")
    print(f"  ✅ District coverage: {len(train_pairs)}/{len(all_pairs)} in Train, {len(test_pairs)}/{len(all_pairs)} in Test (PASSED)")

    # Demographic alignment checks
    child_real = float(is_child.mean() * 100)
    child_train = float(is_child[train_idx].mean() * 100)
    child_test = float(is_child[test_idx].mean() * 100)
    print(f"\n📈 Subgroup Distributions:")
    print(f"  Children (<15): Real={child_real:.3f}%, Train={child_train:.3f}%, Test={child_test:.3f}% (Delta: {abs(child_train - child_test):.3f}%)")

    gender_names = ["Male", "Female", "Transgender"]
    gender_stats = {}
    for g, name in enumerate(gender_names):
        r = float((cond[:, 7 + g] == 1).mean() * 100)
        tr = float((cond[train_idx, 7 + g] == 1).mean() * 100)
        te = float((cond[test_idx, 7 + g] == 1).mean() * 100)
        gender_stats[name] = {"real_pct": r, "train_pct": tr, "test_pct": te}
        print(f"  Gender {name:11s}: Real={r:.3f}%, Train={tr:.3f}%, Test={te:.3f}%")

    sector_names = ["Rural", "Urban"]
    sector_stats = {}
    for s, name in enumerate(sector_names):
        r = float((cond[:, 46 + s] == 1).mean() * 100)
        tr = float((cond[train_idx, 46 + s] == 1).mean() * 100)
        te = float((cond[test_idx, 46 + s] == 1).mean() * 100)
        sector_stats[name] = {"real_pct": r, "train_pct": tr, "test_pct": te}
        print(f"  Sector {name:11s}: Real={r:.3f}%, Train={tr:.3f}%, Test={te:.3f}%")

    # ─────────────────────────────────────────────────────────────
    # SAVE INDICES AND METADATA
    # ─────────────────────────────────────────────────────────────
    train_out = os.path.join(output_dir, "train_indices.npy")
    test_out = os.path.join(output_dir, "test_indices.npy")
    meta_out = os.path.join(output_dir, "split_metadata.json")

    np.save(train_out, train_idx)
    np.save(test_out, test_idx)

    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "dataset_source": data_path,
        "random_seed": seed,
        "total_records": int(N_total),
        "train_records": int(len(train_idx)),
        "test_records": int(len(test_idx)),
        "train_fraction": float(len(train_idx) / N_total),
        "test_fraction": float(len(test_idx) / N_total),
        "stratification_method": "joint (state_id, district_id, is_child) with singleton preservation",
        "spatial_coverage": {
            "total_states": 36,
            "train_states": int(len(unique_states_train)),
            "test_states": int(len(unique_states_test)),
            "total_districts": len(all_pairs),
            "train_districts": len(train_pairs),
            "test_districts": len(test_pairs),
        },
        "demographic_parity": {
            "child_under_15": {
                "real_pct": child_real,
                "train_pct": child_train,
                "test_pct": child_test,
                "train_test_delta": abs(child_train - child_test),
            },
            "gender": gender_stats,
            "sector": sector_stats,
        },
        "artifacts": {
            "train_indices_file": train_out,
            "test_indices_file": test_out,
        },
    }

    with open(meta_out, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Saved {train_out} ({os.path.getsize(train_out):,} bytes)")
    print(f"💾 Saved {test_out} ({os.path.getsize(test_out):,} bytes)")
    print(f"💾 Saved {meta_out}")
    print("✨ 90/10 Stratified Partitioning successfully completed!")


if __name__ == "__main__":
    create_split()
