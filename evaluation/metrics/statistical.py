# -*- coding: utf-8 -*-
"""
Domain 1: Statistical & Aggregate Distributional Realism
=========================================================
Computes:
- Slot-by-slot and Macro Jensen-Shannon Divergence (JSD) across the 9x48 activity-time matrix.
- Total Variation Distance (TVD) on activity participation rates.
- Average daily duration allocations (minutes/day) and duration bias (Δμ).
"""

import numpy as np
from scipy.spatial.distance import jensenshannon

MINUTES_PER_SLOT = 30
NUM_DIVISIONS = 9
NUM_SLOTS = 48

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


def compute_slot_jsd(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Computes Jensen-Shannon Divergence at each of the 48 half-hour slots,
    and returns the macro-average JSD across the entire 24-hour cycle.
    """
    jsd_per_slot = []
    for t in range(NUM_SLOTS):
        r_counts = np.bincount(real_codes[:, t], minlength=NUM_DIVISIONS + 1)[1:]
        f_counts = np.bincount(fake_codes[:, t], minlength=NUM_DIVISIONS + 1)[1:]
        p_real = r_counts / (r_counts.sum() + 1e-12)
        p_fake = f_counts / (f_counts.sum() + 1e-12)
        # JSD squared conforms to standard literature definition
        jsd = float(jensenshannon(p_real, p_fake) ** 2)
        jsd_per_slot.append(jsd)

    macro_jsd = float(np.mean(jsd_per_slot))
    return {
        "macro_jsd": macro_jsd,
        "jsd_per_slot": jsd_per_slot,
        "max_slot_jsd": float(np.max(jsd_per_slot)),
        "min_slot_jsd": float(np.min(jsd_per_slot)),
    }


def compute_participation_tvd(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Computes participation rates across the 9 ICATUS divisions and the
    overall Total Variation Distance (TVD).
    """
    real_rates = {}
    fake_rates = {}
    diffs = []

    for k in range(1, NUM_DIVISIONS + 1):
        r_part = np.mean(np.any(real_codes == k, axis=1)) * 100.0
        f_part = np.mean(np.any(fake_codes == k, axis=1)) * 100.0
        real_rates[k] = float(r_part)
        fake_rates[k] = float(f_part)
        diffs.append(abs(r_part - f_part) / 100.0)

    tvd = float(0.5 * np.sum(diffs))
    return {
        "tvd": tvd,
        "real_participation_rates": real_rates,
        "fake_participation_rates": fake_rates,
    }


def compute_daily_minutes(real_codes: np.ndarray, fake_codes: np.ndarray):
    """
    Computes average daily minutes per activity, duration bias Δμ,
    and Mean Absolute Error (MAE) across all 9 divisions.
    """
    real_mins = {}
    fake_mins = {}
    bias_mins = {}
    abs_errors = []

    for k in range(1, NUM_DIVISIONS + 1):
        r_m = float(np.mean(np.sum(real_codes == k, axis=1)) * MINUTES_PER_SLOT)
        f_m = float(np.mean(np.sum(fake_codes == k, axis=1)) * MINUTES_PER_SLOT)
        bias = f_m - r_m

        real_mins[k] = r_m
        fake_mins[k] = f_m
        bias_mins[k] = float(bias)
        abs_errors.append(abs(bias))

    mae_minutes = float(np.mean(abs_errors))
    return {
        "real_mins": real_mins,
        "fake_mins": fake_mins,
        "bias_mins": bias_mins,
        "mae_minutes": mae_minutes,
    }
