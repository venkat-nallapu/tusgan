# -*- coding: utf-8 -*-
"""
Domain 3: Sequence Structure, Tour Formation & Travel Realism
=============================================================
Computes:
- Daily activity episode counts per schedule (fragmentation vs. continuity).
- Activity switching frequencies (number of daily transitions).
- Shannon transition entropy (measuring sequence variety vs. mode collapse).
- Commute tour anchor alignment (morning work departure and evening return times).
"""

import numpy as np
from scipy.stats import wasserstein_distance

MINUTES_PER_SLOT = 30
NUM_SLOTS = 48


def compute_episode_statistics(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Computes the number of contiguous activity episodes per diary
    and the activity switching frequency.
    """
    N_r, T = real_codes.shape
    N_f = fake_codes.shape[0]

    # Episodes per diary = 1 + count of transitions where code[t] != code[t-1]
    r_switches = np.sum(real_codes[:, 1:] != real_codes[:, :-1], axis=1)
    f_switches = np.sum(fake_codes[:, 1:] != fake_codes[:, :-1], axis=1)

    r_episodes = r_switches + 1
    f_episodes = f_switches + 1

    r_mean, r_std = float(np.mean(r_episodes)), float(np.std(r_episodes))
    f_mean, f_std = float(np.mean(f_episodes)), float(np.std(f_episodes))
    episode_diff = f_mean - r_mean

    return {
        "real_episodes_mean": r_mean,
        "real_episodes_std": r_std,
        "fake_episodes_mean": f_mean,
        "fake_episodes_std": f_std,
        "episode_diff": float(episode_diff),
        "real_switches_mean": float(np.mean(r_switches)),
        "fake_switches_mean": float(np.mean(f_switches)),
    }


def compute_work_anchor_timing(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    For schedules that contain employment (ICATUS 1), extracts:
    - Primary work start time (earliest work slot in morning/daytime).
    - Primary work end time (latest work slot).
    - Wasserstein distance between real and synthetic work start times in clock minutes.
    """
    def extract_work_windows(codes):
        starts = []
        ends = []
        for i in range(len(codes)):
            work_slots = np.where(codes[i] == 1)[0]
            if len(work_slots) > 0:
                # Convert slot index (0..47) to clock hour (04:00 AM base)
                start_h = (4.0 + work_slots[0] * 0.5) % 24.0
                end_h = (4.0 + (work_slots[-1] + 1) * 0.5) % 24.0
                starts.append(start_h)
                ends.append(end_h)
        return np.array(starts), np.array(ends)

    r_starts, r_ends = extract_work_windows(real_codes)
    f_starts, f_ends = extract_work_windows(fake_codes)

    if len(r_starts) > 0 and len(f_starts) > 0:
        w_start_h = float(wasserstein_distance(r_starts, f_starts))
        w_start_min = float(w_start_h * 60.0)
        r_start_mean = float(np.mean(r_starts))
        f_start_mean = float(np.mean(f_starts))
    else:
        w_start_h = 0.0
        w_start_min = 0.0
        r_start_mean = 0.0
        f_start_mean = 0.0

    return {
        "work_start_w1_hours": w_start_h,
        "work_start_w1_mins": w_start_min,
        "real_work_start_mean_h": r_start_mean,
        "fake_work_start_mean_h": f_start_mean,
        "real_work_participants": int(len(r_starts)),
        "fake_work_participants": int(len(f_starts)),
    }


def compute_transition_entropy(codes: np.ndarray):
    """
    Computes the Shannon entropy of activity sequence transitions (bigrams).
    Higher entropy indicates diverse schedule generation without mode collapse.
    """
    transitions = []
    for i in range(len(codes)):
        for t in range(codes.shape[1] - 1):
            transitions.append((codes[i, t], codes[i, t + 1]))

    # Compute empirical transition distribution
    unique_trans, counts = np.unique(transitions, axis=0, return_counts=True)
    probs = counts / float(np.sum(counts))
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    return entropy
