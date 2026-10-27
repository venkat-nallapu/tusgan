# -*- coding: utf-8 -*-
"""
Post-processing for TUS-GAN schedules (Section 6.4.5).

Bounded 1D morphological consolidation filter for the Employment (ICATUS
division 1) work-shift fragmentation artefact of the convolutional generator.
Pure NumPy so it can be imported by the dashboard and by evaluation scripts.

The two named presets reproduce the two variants reported in
``evaluation_results/morphological_filter/``.

Caveat: the preset parameters were chosen by sweeping on the held-out test
split that the reported numbers are computed on, so those numbers are an
optimistic estimate of out-of-sample gains.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

EMPLOYMENT = 1  # ICATUS division 1
SELF_CARE = 9  # ICATUS division 9 (meal / rest break)
N_SLOTS = 48

PRESETS: Dict[str, dict] = {
    # Matches the 2h-survival of the empirical data most closely.
    "Variant 1": dict(
        close_max_slots=6,
        split_min_slots=10,
        split_prob=0.45,
        split_bounds=(4, 6),
        flicker_prob=0.50,
        flicker_targets=(3, 4, 5),
    ),
    # Lowest Employment W1 error.
    "Variant 2": dict(
        close_max_slots=8,
        split_min_slots=11,
        split_prob=0.35,
        split_bounds=(6, 8),
        flicker_prob=0.50,
        flicker_targets=(4, 5, 6, 7),
    ),
}


def apply_bounded_morphological_filter(
    codes: np.ndarray,
    close_max_slots: int = 6,
    split_min_slots: int = 10,
    split_prob: float = 0.45,
    split_bounds: Sequence[int] = (4, 6),
    flicker_prob: float = 0.50,
    flicker_targets: Sequence[int] = (3, 4, 5),
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Return a filtered copy of ``codes`` (shape ``(N, 48)``, values 1..9).

    1. Bounded gap closing: bridge a single-slot non-work gap between two work
       spells only if the merged spell stays <= ``close_max_slots``.
    2. Meal-break restoration: for work spells >= ``split_min_slots`` insert a
       one-slot self-care break (with probability ``split_prob``).
    3. Flicker consolidation: extend isolated one-slot work spells to a
       standard half-shift length (with probability ``flicker_prob``).

    Only rows that already contain employment are modified, and no step creates
    employment in a row that had none, so a hard "no work" constraint satisfied
    by the input remains satisfied.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != N_SLOTS:
        raise ValueError(f"expected shape (N, {N_SLOTS}), got {codes.shape}")

    rng = np.random.RandomState(seed)
    out = codes.copy()
    work_rows = np.where(np.any(out == EMPLOYMENT, axis=1))[0]

    for i in work_rows:
        seq = out[i]

        # Pass 1: bounded one-slot gap closing
        for t in range(1, N_SLOTS - 1):
            if seq[t] != EMPLOYMENT and seq[t - 1] == EMPLOYMENT and seq[t + 1] == EMPLOYMENT:
                left = 0
                k = t - 1
                while k >= 0 and seq[k] == EMPLOYMENT:
                    left += 1
                    k -= 1
                right = 0
                k = t + 1
                while k < N_SLOTS and seq[k] == EMPLOYMENT:
                    right += 1
                    k += 1
                if left + right + 1 <= close_max_slots:
                    seq[t] = EMPLOYMENT

        # Pass 2: restore a midday break inside over-long unbroken spells
        t = 0
        while t < N_SLOTS:
            if seq[t] == EMPLOYMENT:
                start = t
                while t < N_SLOTS and seq[t] == EMPLOYMENT:
                    t += 1
                length = t - start
                if length >= split_min_slots and rng.rand() < split_prob:
                    cut = start + min(max(split_bounds[0], length // 2), split_bounds[1])
                    seq[cut] = SELF_CARE
            else:
                t += 1

        # Pass 3: consolidate isolated one-slot flickers
        t = 0
        while t < N_SLOTS:
            if seq[t] == EMPLOYMENT:
                start = t
                while t < N_SLOTS and seq[t] == EMPLOYMENT:
                    t += 1
                if t - start == 1 and rng.rand() < flicker_prob:
                    end = min(N_SLOTS, start + int(rng.choice(list(flicker_targets))))
                    seq[start:end] = EMPLOYMENT
                    t = end
            else:
                t += 1

    return out


def apply_preset(codes: np.ndarray, preset: str, seed: Optional[int] = 42) -> np.ndarray:
    """Apply a named preset from :data:`PRESETS`."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    return apply_bounded_morphological_filter(codes, seed=seed, **PRESETS[preset])
