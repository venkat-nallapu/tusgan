# -*- coding: utf-8 -*-
"""
Domain 2: Temporal Pacing & Episode Dynamics
============================================
Computes:
- Time-slice transition matrix consistency (Frobenius norms across 4 quadrants: Morning, Afternoon, Evening, Night).
- Contiguous spell-duration extraction (Run-Length Encoding).
- 1D Wasserstein Earth Mover's Distance (W1) in discrete slots and clock minutes.
- Empirical spell survival curves S(t) = P(Duration >= t) for employment and sleep.
"""

import numpy as np
from scipy.stats import wasserstein_distance

MINUTES_PER_SLOT = 30
NUM_DIVISIONS = 9
NUM_SLOTS = 48


def compute_transition_matrices_and_fnorm(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Computes empirical transition matrices across 4 quadrants (12 slots each)
    and the Frobenius norm discrepancy ||P_fake - P_real||_F.
    """
    fnorms = []
    real_matrices = []
    fake_matrices = []

    quadrant_names = ["Morning (04-10h)", "Afternoon (10-16h)", "Evening (16-22h)", "Night (22-04h)"]

    for k in range(4):
        start, end = k * 12, (k + 1) * 12
        r_t = real_codes[:, start : end - 1] - 1
        r_tp1 = real_codes[:, start + 1 : end] - 1
        f_t = fake_codes[:, start : end - 1] - 1
        f_tp1 = fake_codes[:, start + 1 : end] - 1

        P_r = np.zeros((NUM_DIVISIONS, NUM_DIVISIONS))
        P_f = np.zeros((NUM_DIVISIONS, NUM_DIVISIONS))

        for i in range(NUM_DIVISIONS):
            for j in range(NUM_DIVISIONS):
                P_r[i, j] = np.sum((r_t == i) & (r_tp1 == j))
                P_f[i, j] = np.sum((f_t == i) & (f_tp1 == j))

        P_r = P_r / (P_r.sum(axis=1, keepdims=True) + 1e-12)
        P_f = P_f / (P_f.sum(axis=1, keepdims=True) + 1e-12)

        f_dist = float(np.linalg.norm(P_r - P_f, "fro"))
        fnorms.append(f_dist)
        real_matrices.append(P_r)
        fake_matrices.append(P_f)

    macro_fnorm = float(np.mean(fnorms))
    return {
        "macro_fnorm": macro_fnorm,
        "fnorms_slices": fnorms,
        "quadrant_names": quadrant_names,
        "real_matrices": real_matrices,
        "fake_matrices": fake_matrices,
    }


def extract_spells(codes: np.ndarray):
    """
    Extracts contiguous episode durations (in slots and minutes)
    for each activity division via Run-Length Encoding (RLE).
    """
    N, T = codes.shape
    spells_by_act_slots = {act: [] for act in range(1, NUM_DIVISIONS + 1)}
    spells_by_act_mins = {act: [] for act in range(1, NUM_DIVISIONS + 1)}

    for i in range(N):
        seq = codes[i]
        curr_act = seq[0]
        curr_len = 1
        for t in range(1, T):
            if seq[t] == curr_act:
                curr_len += 1
            else:
                spells_by_act_slots[curr_act].append(curr_len)
                spells_by_act_mins[curr_act].append(curr_len * MINUTES_PER_SLOT)
                curr_act = seq[t]
                curr_len = 1
        spells_by_act_slots[curr_act].append(curr_len)
        spells_by_act_mins[curr_act].append(curr_len * MINUTES_PER_SLOT)

    return spells_by_act_slots, spells_by_act_mins


def compute_spell_emds(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Computes 1D Wasserstein Earth Mover's Distance between real and synthetic
    spell durations for all 9 ICATUS divisions.
    """
    r_slots, r_mins = extract_spells(real_codes)
    f_slots, f_mins = extract_spells(fake_codes)

    emd_slots = {}
    emd_mins = {}
    real_means_hours = {}
    fake_means_hours = {}
    real_stds_hours = {}
    fake_stds_hours = {}

    for k in range(1, NUM_DIVISIONS + 1):
        r_s = r_slots[k]
        f_s = f_slots[k]
        if len(r_s) > 0 and len(f_s) > 0:
            w_s = float(wasserstein_distance(r_s, f_s))
            w_m = float(w_s * MINUTES_PER_SLOT)
            emd_slots[k] = w_s
            emd_mins[k] = w_m
            real_means_hours[k] = float(np.mean(r_s) * 0.5)
            fake_means_hours[k] = float(np.mean(f_s) * 0.5)
            real_stds_hours[k] = float(np.std(r_s) * 0.5)
            fake_stds_hours[k] = float(np.std(f_s) * 0.5)
        else:
            emd_slots[k] = 0.0
            emd_mins[k] = 0.0
            real_means_hours[k] = 0.0
            fake_means_hours[k] = 0.0
            real_stds_hours[k] = 0.0
            fake_stds_hours[k] = 0.0

    mean_emd_slots = float(np.mean(list(emd_slots.values())))
    mean_emd_mins = float(np.mean(list(emd_mins.values())))

    return {
        "mean_emd_slots": mean_emd_slots,
        "mean_emd_mins": mean_emd_mins,
        "emd_slots": emd_slots,
        "emd_mins": emd_mins,
        "real_means_hours": real_means_hours,
        "fake_means_hours": fake_means_hours,
        "real_stds_hours": real_stds_hours,
        "fake_stds_hours": fake_stds_hours,
    }


def compute_survival_curves(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Computes survival probability S(t) = P(T >= t) for employment (k=1)
    and sleep/self-care (k=9) across duration thresholds.
    """
    r_slots, _ = extract_spells(real_codes)
    f_slots, _ = extract_spells(fake_codes)

    r_work_h = np.array(r_slots[1]) * 0.5
    f_work_h = np.array(f_slots[1]) * 0.5
    r_sleep_h = np.array(r_slots[9]) * 0.5
    f_sleep_h = np.array(f_slots[9]) * 0.5

    work_thresholds = [1.0, 2.0, 4.0, 8.0]
    sleep_thresholds = [2.0, 4.0, 6.0, 8.0]

    work_surv_real = {t: float(np.mean(r_work_h >= t) * 100.0) if len(r_work_h) > 0 else 0.0 for t in work_thresholds}
    work_surv_fake = {t: float(np.mean(f_work_h >= t) * 100.0) if len(f_work_h) > 0 else 0.0 for t in work_thresholds}

    sleep_surv_real = {t: float(np.mean(r_sleep_h >= t) * 100.0) if len(r_sleep_h) > 0 else 0.0 for t in sleep_thresholds}
    sleep_surv_fake = {t: float(np.mean(f_sleep_h >= t) * 100.0) if len(f_sleep_h) > 0 else 0.0 for t in sleep_thresholds}

    # Early fragmentation gap at 1h and 2h for work
    work_gap_1h = float(work_surv_real[1.0] - work_surv_fake[1.0])
    work_gap_2h = float(work_surv_real[2.0] - work_surv_fake[2.0])

    return {
        "work_surv_real": work_surv_real,
        "work_surv_fake": work_surv_fake,
        "sleep_surv_real": sleep_surv_real,
        "sleep_surv_fake": sleep_surv_fake,
        "work_gap_1h": work_gap_1h,
        "work_gap_2h": work_gap_2h,
    }
