# -*- coding: utf-8 -*-
"""
Domain 5: Demographic & Subgroup Equity Parity
==============================================
Computes:
- Gender disparity ratios in domestic labor (ICATUS 3: Unpaid Domestic,
  ICATUS 4: Unpaid Caregiving) vs ground truth survey values.
- Rural-Urban disparity ratios in primary production (ICATUS 2: Production for Own Use)
  and formal employment (ICATUS 1: Employment).
- Conditional subgroup Jensen-Shannon Divergence (JSD) ensuring generative fidelity
  is equitably preserved across intersectional demographics (Female, Male, Rural, Urban).
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import jensenshannon

MINUTES_PER_SLOT = 30
NUM_DIVISIONS = 9
NUM_SLOTS = 48

# Conditioning Vector Index Constants (Verified from ITUS 2019 data)
IDX_MALE = 7
IDX_FEMALE = 8
IDX_TRANSGENDER = 9
IDX_RURAL = 46
IDX_URBAN = 47


def _compute_group_macro_jsd(real_codes: np.ndarray, fake_codes: np.ndarray) -> float:
    """Computes macro JSD across 48 slots for a specific demographic slice."""
    if len(real_codes) == 0 or len(fake_codes) == 0:
        return 0.0
    jsds = []
    for t in range(NUM_SLOTS):
        r_counts = np.bincount(real_codes[:, t], minlength=NUM_DIVISIONS + 1)[1:]
        f_counts = np.bincount(fake_codes[:, t], minlength=NUM_DIVISIONS + 1)[1:]
        p_real = r_counts / (r_counts.sum() + 1e-12)
        p_fake = f_counts / (f_counts.sum() + 1e-12)
        jsds.append(float(jensenshannon(p_real, p_fake) ** 2))
    return float(np.mean(jsds))


def compute_gender_parity(
    real_codes: np.ndarray,
    fake_codes: np.ndarray,
    cond_vector: np.ndarray,
    fake_cond_vector: Optional[np.ndarray] = None,
):
    """
    Evaluates gender equity in time-allocation patterns:
    - Domestic work burden (ICATUS 3 + 4) ratio between Females and Males.
    - Employment time (ICATUS 1) gap between Females and Males.
    - Subgroup conditional JSD for Female and Male cohorts.
    """
    f_cond = cond_vector if fake_cond_vector is None else fake_cond_vector

    male_mask_r = cond_vector[:, IDX_MALE] == 1.0
    female_mask_r = cond_vector[:, IDX_FEMALE] == 1.0
    male_mask_f = f_cond[:, IDX_MALE] == 1.0
    female_mask_f = f_cond[:, IDX_FEMALE] == 1.0

    # Real time allocations (minutes/day)
    r_male_codes = real_codes[male_mask_r]
    r_female_codes = real_codes[female_mask_r]

    f_male_codes = fake_codes[male_mask_f]
    f_female_codes = fake_codes[female_mask_f]

    # Real domestic minutes (ICATUS 3: Domestic + ICATUS 4: Caregiving)
    r_f_dom = np.mean(np.sum((r_female_codes == 3) | (r_female_codes == 4), axis=1)) * MINUTES_PER_SLOT if len(r_female_codes) > 0 else 0.0
    r_m_dom = np.mean(np.sum((r_male_codes == 3) | (r_male_codes == 4), axis=1)) * MINUTES_PER_SLOT if len(r_male_codes) > 0 else 0.0
    real_dom_ratio = float(r_f_dom / (r_m_dom + 1e-6))

    # Fake domestic minutes
    f_f_dom = np.mean(np.sum((f_female_codes == 3) | (f_female_codes == 4), axis=1)) * MINUTES_PER_SLOT if len(f_female_codes) > 0 else 0.0
    f_m_dom = np.mean(np.sum((f_male_codes == 3) | (f_male_codes == 4), axis=1)) * MINUTES_PER_SLOT if len(f_male_codes) > 0 else 0.0
    fake_dom_ratio = float(f_f_dom / (f_m_dom + 1e-6))

    # Real employment minutes (ICATUS 1)
    r_f_emp = np.mean(np.sum(r_female_codes == 1, axis=1)) * MINUTES_PER_SLOT if len(r_female_codes) > 0 else 0.0
    r_m_emp = np.mean(np.sum(r_male_codes == 1, axis=1)) * MINUTES_PER_SLOT if len(r_male_codes) > 0 else 0.0

    # Fake employment minutes
    f_f_emp = np.mean(np.sum(f_female_codes == 1, axis=1)) * MINUTES_PER_SLOT if len(f_female_codes) > 0 else 0.0
    f_m_emp = np.mean(np.sum(f_male_codes == 1, axis=1)) * MINUTES_PER_SLOT if len(f_male_codes) > 0 else 0.0

    # Conditional fidelity (JSD)
    female_jsd = _compute_group_macro_jsd(r_female_codes, f_female_codes)
    male_jsd = _compute_group_macro_jsd(r_male_codes, f_male_codes)

    return {
        "female_domestic_mins_real": float(r_f_dom),
        "female_domestic_mins_fake": float(f_f_dom),
        "male_domestic_mins_real": float(r_m_dom),
        "male_domestic_mins_fake": float(f_m_dom),
        "real_domestic_disparity_ratio": real_dom_ratio,
        "fake_domestic_disparity_ratio": fake_dom_ratio,
        "domestic_ratio_abs_error": float(abs(fake_dom_ratio - real_dom_ratio)),
        "female_employment_mins_real": float(r_f_emp),
        "female_employment_mins_fake": float(f_f_emp),
        "male_employment_mins_real": float(r_m_emp),
        "male_employment_mins_fake": float(f_m_emp),
        "female_conditional_jsd": female_jsd,
        "male_conditional_jsd": male_jsd,
    }


def compute_spatial_parity(
    real_codes: np.ndarray,
    fake_codes: np.ndarray,
    cond_vector: np.ndarray,
    fake_cond_vector: Optional[np.ndarray] = None,
):
    """
    Evaluates rural-urban spatial equity:
    - Primary production for own use (ICATUS 2) disparity.
    - Formal employment (ICATUS 1) allocation across sectors.
    - Subgroup conditional JSD for Rural and Urban cohorts.
    """
    f_cond = cond_vector if fake_cond_vector is None else fake_cond_vector

    rural_mask_r = cond_vector[:, IDX_RURAL] == 1.0
    urban_mask_r = cond_vector[:, IDX_URBAN] == 1.0
    rural_mask_f = f_cond[:, IDX_RURAL] == 1.0
    urban_mask_f = f_cond[:, IDX_URBAN] == 1.0

    r_rural_codes = real_codes[rural_mask_r]
    r_urban_codes = real_codes[urban_mask_r]

    f_rural_codes = fake_codes[rural_mask_f]
    f_urban_codes = fake_codes[urban_mask_f]

    # Real production for own use (ICATUS 2)
    r_r_prod = np.mean(np.sum(r_rural_codes == 2, axis=1)) * MINUTES_PER_SLOT if len(r_rural_codes) > 0 else 0.0
    r_u_prod = np.mean(np.sum(r_urban_codes == 2, axis=1)) * MINUTES_PER_SLOT if len(r_urban_codes) > 0 else 0.0
    real_prod_ratio = float(r_r_prod / (r_u_prod + 1e-6))

    # Fake production for own use (ICATUS 2)
    f_r_prod = np.mean(np.sum(f_rural_codes == 2, axis=1)) * MINUTES_PER_SLOT if len(f_rural_codes) > 0 else 0.0
    f_u_prod = np.mean(np.sum(f_urban_codes == 2, axis=1)) * MINUTES_PER_SLOT if len(f_urban_codes) > 0 else 0.0
    fake_prod_ratio = float(f_r_prod / (f_u_prod + 1e-6))

    # Conditional fidelity (JSD)
    rural_jsd = _compute_group_macro_jsd(r_rural_codes, f_rural_codes)
    urban_jsd = _compute_group_macro_jsd(r_urban_codes, f_urban_codes)

    return {
        "rural_production_mins_real": float(r_r_prod),
        "rural_production_mins_fake": float(f_r_prod),
        "urban_production_mins_real": float(r_u_prod),
        "urban_production_mins_fake": float(f_u_prod),
        "real_production_ratio": real_prod_ratio,
        "fake_production_ratio": fake_prod_ratio,
        "production_ratio_abs_error": float(abs(fake_prod_ratio - real_prod_ratio)),
        "rural_conditional_jsd": rural_jsd,
        "urban_conditional_jsd": urban_jsd,
    }
