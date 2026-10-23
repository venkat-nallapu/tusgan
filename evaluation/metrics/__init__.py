# -*- coding: utf-8 -*-
"""
TUS-GAN 6-Domain TRC Evaluation Metric Modules
"""

from .statistical import compute_slot_jsd, compute_participation_tvd, compute_daily_minutes
from .temporal import compute_transition_matrices_and_fnorm, compute_spell_emds, extract_spells
from .tours import compute_episode_statistics, compute_work_anchor_timing, compute_transition_entropy
from .feasibility import compute_time_budget_closure, compute_child_labor_audit, compute_sleep_constraint_audit
from .equity import compute_gender_parity, compute_spatial_parity
from .adversarial import compute_adversarial_rf, compute_privacy_and_novelty

__all__ = [
    "compute_slot_jsd",
    "compute_participation_tvd",
    "compute_daily_minutes",
    "compute_transition_matrices_and_fnorm",
    "compute_spell_emds",
    "extract_spells",
    "compute_episode_statistics",
    "compute_work_anchor_timing",
    "compute_transition_entropy",
    "compute_time_budget_closure",
    "compute_child_labor_audit",
    "compute_sleep_constraint_audit",
    "compute_gender_parity",
    "compute_spatial_parity",
    "compute_adversarial_rf",
    "compute_privacy_and_novelty",
]
