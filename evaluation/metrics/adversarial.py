# -*- coding: utf-8 -*-
"""
Domain 6: ML Indistinguishability & Privacy (Adversarial Validation)
===================================================================
Computes:
- Random Forest Adversarial Classifier:
    Trains an ensemble discriminator to distinguish real vs. synthetic schedules.
    Measures ROC-AUC (optimal = 0.5000) and Accuracy (optimal = 50.00%).
- Empirical Privacy & Zero-Memorization Audit:
    Exact Match Rate (EMR) checking whether synthetic agents reproduce training diaries.
    Subsample Nearest-Neighbor Hamming Distance (Novelty verification conforming to
    Shone & Hillel 2025 ActEval Domain 5).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

MINUTES_PER_SLOT = 30
NUM_SLOTS = 48
NUM_DIVISIONS = 9


def compute_adversarial_rf(
    real_codes: np.ndarray,
    fake_codes: np.ndarray,
    n_sample: int = 10000,
    seed: int = 42,
):
    """
    Trains a Random Forest classifier (100 estimators) on a balanced mixture of
    real (label 0) and synthetic (label 1) activity schedules.

    Features per schedule:
    - 48 categorical activity codes.
    - 9 total daily activity durations (minutes/day).
    Total features = 57.

    Returns:
    - roc_auc: Area Under the ROC Curve on the holdout test set (ideal: 0.5000).
    - test_accuracy: Classification accuracy on holdout test set (ideal: 50.00%).
    """
    n_real = min(len(real_codes), n_sample)
    n_fake = min(len(fake_codes), n_sample)
    n_eval = min(n_real, n_fake)

    rng = np.random.default_rng(seed)
    idx_real = rng.choice(len(real_codes), size=n_eval, replace=False)
    idx_fake = rng.choice(len(fake_codes), size=n_eval, replace=False)

    sub_real = real_codes[idx_real]
    sub_fake = fake_codes[idx_fake]

    # Feature extraction: [48 codes, 9 daily durations in minutes]
    def extract_features(codes: np.ndarray) -> np.ndarray:
        B, T = codes.shape
        durations = np.zeros((B, NUM_DIVISIONS), dtype=np.float32)
        for d in range(1, NUM_DIVISIONS + 1):
            durations[:, d - 1] = np.sum(codes == d, axis=1) * MINUTES_PER_SLOT
        return np.hstack([codes.astype(np.float32), durations])

    X_real = extract_features(sub_real)
    X_fake = extract_features(sub_fake)

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([np.zeros(n_eval, dtype=np.int32), np.ones(n_eval, dtype=np.int32)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=5,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = float(roc_auc_score(y_test, probs))
    acc = float(accuracy_score(y_test, preds)) * 100.0

    return {
        "rf_roc_auc": auc,
        "rf_accuracy": acc,
        "eval_sample_size": n_eval * 2,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def compute_privacy_and_novelty(
    real_codes: np.ndarray,
    fake_codes: np.ndarray,
    n_hamming_eval: int = 1000,
    seed: int = 42,
):
    """
    Evaluates privacy preservation and novelty:
    1. Exact Match Rate (EMR): Checks whether any synthetic schedule is an exact
       slot-by-slot replica of any real training diary across all 48 slots.
    2. Nearest-Neighbor Hamming Distance: Evaluates the minimum number of 30-min
       slots differing between synthetic schedules and the closest real schedules.
    """
    # 1. Exact Match via Fast 64-bit integer / byte hashing
    # Convert each row to bytes
    real_set = set(row.tobytes() for row in real_codes)
    fake_matches = sum(1 for row in fake_codes if row.tobytes() in real_set)
    emr_pct = float(fake_matches / len(fake_codes)) * 100.0

    # 2. Nearest-Neighbor Hamming Distance on representative subset
    rng = np.random.default_rng(seed)
    n_h = min(len(fake_codes), n_hamming_eval)
    n_r_sub = min(len(real_codes), 5000)

    sub_fake = fake_codes[rng.choice(len(fake_codes), size=n_h, replace=False)]
    sub_real = real_codes[rng.choice(len(real_codes), size=n_r_sub, replace=False)]

    # Pairwise Hamming distance calculation: sub_fake (n_h, 48) vs sub_real (n_r_sub, 48)
    # distance in slots: sum(fake_i != real_j)
    min_hamming_dists = []
    for f_row in sub_fake:
        diffs = np.sum(sub_real != f_row, axis=1)
        min_hamming_dists.append(int(np.min(diffs)))

    min_hamming_arr = np.array(min_hamming_dists)

    return {
        "exact_match_count": int(fake_matches),
        "exact_match_rate_pct": emr_pct,
        "mean_min_hamming_distance_slots": float(np.mean(min_hamming_arr)),
        "min_hamming_distance_slots": int(np.min(min_hamming_arr)),
        "max_min_hamming_distance_slots": int(np.max(min_hamming_arr)),
    }
