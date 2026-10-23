# -*- coding: utf-8 -*-
"""
TUS-GAN Model Adapter & Checkpoint Loader
==========================================
Unified interface to load and run inference across any TUS-GAN generator
(v3, v4, v4.1, v5, v5.1, v6) with identical conditioning inputs.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# Root directory of tusgan
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_SPECS = {
    "v3": {
        "generator_path": os.path.join(REPO_ROOT, "v3", "generator.py"),
        "class_name": "Generator",
        "default_checkpoint": os.path.join(REPO_ROOT, "v3", "checkpoints", "final.pt"),
        "backbone": "CNN",
        "constraint": "None (Unconstrained Baseline)",
        "cell": "Baseline",
    },
    "v4": {
        "generator_path": os.path.join(REPO_ROOT, "v4", "generator.py"),
        "class_name": "Generator",
        "default_checkpoint": os.path.join(REPO_ROOT, "v4", "checkpoints", "final.pt"),
        "backbone": "CNN",
        "constraint": "Soft Penalty (λ=10)",
        "cell": "Cell A (Champion 🏆)",
    },
    "v4.1": {
        "generator_path": os.path.join(REPO_ROOT, "v4.1", "generator.py"),
        "class_name": "Generator",
        "default_checkpoint": os.path.join(REPO_ROOT, "v4.1", "checkpoints", "final.pt"),
        "backbone": "CNN",
        "constraint": "Hard Mask (-1e9)",
        "cell": "Cell C (Ablation)",
    },
    "v5": {
        "generator_path": os.path.join(REPO_ROOT, "v5", "generator.py"),
        "class_name": "Generator",
        "default_checkpoint": os.path.join(REPO_ROOT, "v5", "checkpoints", "final.pt"),
        "backbone": "Transformer",
        "constraint": "Hard Mask (-1e9)",
        "cell": "Cell B (Ablation)",
    },
    "v5.1": {
        "generator_path": os.path.join(REPO_ROOT, "v5.1", "generator.py"),
        "class_name": "Generator",
        "default_checkpoint": os.path.join(REPO_ROOT, "v5.1", "checkpoints_seed42", "final.pt"),
        "backbone": "Transformer",
        "constraint": "Soft Penalty (λ=10)",
        "cell": "Cell D (Restoration)",
    },
    "v6": {
        "generator_path": os.path.join(REPO_ROOT, "v6", "generator.py"),
        "class_name": "Generator",
        "default_checkpoint": os.path.join(REPO_ROOT, "v6", "checkpoints", "final.pt"),
        "backbone": "Deep Conformer (6L 8H)",
        "constraint": "Hybrid",
        "cell": "Next-Gen",
    },
    "actvae_unconstrained": {
        "type": "external_csv",
        "schedules_csv": os.path.join(REPO_ROOT, "caveat_bench", "data", "synth_vae_unconstrained_schedules.csv"),
        "attributes_csv": os.path.join(REPO_ROOT, "caveat_bench", "outputs", "actvae_logs", "cvae_unconstrained", "version_0", "synthetic_attributes.csv"),
        "default_checkpoint": os.path.join(REPO_ROOT, "caveat_bench", "data", "synth_vae_unconstrained_schedules.csv"),
        "backbone": "ActVAE (Conditional VAE)",
        "constraint": "None (Unconstrained Baseline)",
        "cell": "Caveat Baseline",
        "source": "Shone & Hillel (2025)",
    },
    "actvae_soft": {
        "type": "external_csv",
        "schedules_csv": os.path.join(REPO_ROOT, "caveat_bench", "data", "synth_vae_soft_schedules.csv"),
        "attributes_csv": os.path.join(REPO_ROOT, "caveat_bench", "outputs", "actvae_logs", "cvae_soft_constrained", "version_0", "synthetic_attributes.csv"),
        "default_checkpoint": os.path.join(REPO_ROOT, "caveat_bench", "data", "synth_vae_soft_schedules.csv"),
        "backbone": "ActVAE (Conditional VAE)",
        "constraint": "Soft Penalty (C1)",
        "cell": "Caveat Constrained",
        "source": "Shone & Hillel (2025)",
    },
}

# Aliases for convenience
MODEL_SPECS["caveat_unconstrained"] = MODEL_SPECS["actvae_unconstrained"]
MODEL_SPECS["caveat_soft"] = MODEL_SPECS["actvae_soft"]


class ActVAEAdapter(nn.Module):
    """
    Adapter for Shone & Hillel (2025) Caveat ActVAE benchmark models.
    Loads standardized episodic schedules and demographic attributes,
    decoding them to 48-slot (N, 48) integer code arrays and (N, 82) condition vectors.
    """
    def __init__(self, model_name: str, schedules_csv: str, attributes_csv: str, spec: Dict[str, Any]):
        super().__init__()
        self.model_name = model_name
        self.schedules_csv = schedules_csv
        self.attributes_csv = attributes_csv
        self.spec = spec
        self._cached_codes = None
        self._cached_cond = None

    def _load_data(self):
        if self._cached_codes is not None:
            return self._cached_codes, self._cached_cond

        import pandas as pd
        if not os.path.exists(self.schedules_csv):
            raise FileNotFoundError(f"Caveat schedule file not found at: {self.schedules_csv}")
        if not os.path.exists(self.attributes_csv):
            raise FileNotFoundError(f"Caveat attribute file not found at: {self.attributes_csv}")

        df_sched = pd.read_csv(self.schedules_csv)
        df_attr = pd.read_csv(self.attributes_csv)

        ACT_TO_ID = {
            "employment": 1,
            "production_own_use": 2,
            "unpaid_domestic": 3,
            "unpaid_caregiving": 4,
            "unpaid_volunteer": 5,
            "learning_education": 6,
            "socializing_religious": 7,
            "culture_leisure": 8,
            "self_care": 9,
        }

        # Convert episodic to (N, 48) slots
        pids = sorted(df_sched["pid"].unique())
        pid_map = {pid: i for i, pid in enumerate(pids)}
        N = len(pids)
        slots = np.zeros((N, 48), dtype=int)
        for row in df_sched.itertuples():
            if row.pid in pid_map:
                idx = pid_map[row.pid]
                s_slot = int(row.start // 30)
                e_slot = int(min(row.end // 30, 48))
                act_id = ACT_TO_ID.get(row.act, 9)
                slots[idx, s_slot:e_slot] = act_id

        # Build (N, 82) conditioning matrix matching df_attr
        cond = np.zeros((N, 82), dtype=np.float32)
        attr_pid_map = {row.pid: row for row in df_attr.itertuples()}
        for i, pid in enumerate(pids):
            if pid in attr_pid_map:
                a = attr_pid_map[pid]
                # is_child at index 0
                if getattr(a, "is_child", 0) == 1:
                    cond[i, 0] = 1.0
                # gender: male=0 (idx 7), female=1 (idx 8)
                g = getattr(a, "gender", 0)
                if g == 0:
                    cond[i, 7] = 1.0
                elif g == 1:
                    cond[i, 8] = 1.0
                # sector: rural=0 (idx 46), urban=1 (idx 47)
                s = getattr(a, "sector", 0)
                if s == 0:
                    cond[i, 46] = 1.0
                elif s == 1:
                    cond[i, 47] = 1.0

        self._cached_codes = slots
        self._cached_cond = cond
        return self._cached_codes, self._cached_cond

    def get_samples_and_cond(self, n_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        codes, cond = self._load_data()
        total = len(codes)
        rng = np.random.default_rng(seed)
        if n_samples <= total:
            indices = rng.choice(total, size=n_samples, replace=False)
        else:
            indices = rng.choice(total, size=n_samples, replace=True)
        return codes[indices], cond[indices]


def _load_class_from_file(filepath: str, class_name: str):
    """Dynamically loads a class from a file without polluting global sys.modules."""
    module_name = f"tusgan_dyn_{class_name}_{os.path.basename(os.path.dirname(filepath))}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def load_generator(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads any TUS-GAN generator model and restores its EMA weights.
    """
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(MODEL_SPECS.keys())}")

    spec = MODEL_SPECS[model_name]

    if spec.get("type") == "external_csv":
        adapter = ActVAEAdapter(
            model_name=model_name,
            schedules_csv=spec["schedules_csv"],
            attributes_csv=spec["attributes_csv"],
            spec=spec,
        )
        info = {
            "model_name": model_name,
            "type": "external_csv",
            "checkpoint_path": spec["schedules_csv"],
            "checkpoint_epoch": "N/A (Pre-trained)",
            "state_key_loaded": "CSV_Episodes",
            "backbone": spec["backbone"],
            "constraint": spec["constraint"],
            "cell": spec["cell"],
            "config": spec,
        }
        return adapter, info

    ckpt_path = checkpoint_path or spec["default_checkpoint"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found for {model_name} at: {ckpt_path}\n"
            f"If training is still in progress, please verify the checkpoint path."
        )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    noise_dim = cfg.get("noise_dim", 128)
    cond_dim = cfg.get("cond_dim", 82)
    num_districts = cfg.get("num_districts", 71)
    num_states = cfg.get("num_states", 36)
    base_channels = cfg.get("g_base_channels", 256)

    # Dynamic class import
    G_cls = _load_class_from_file(spec["generator_path"], spec["class_name"])

    # Instantiate generator with exact config parameters
    G = G_cls(
        noise_dim=noise_dim,
        cond_dim=cond_dim,
        num_districts=num_districts,
        num_states=num_states,
        base_channels=base_channels,
    )

    # Prefer EMA weights if present, fallback to raw generator state
    state_key = "G_state_ema" if "G_state_ema" in ckpt else "G_state"
    G.load_state_dict(ckpt[state_key])
    G.to(device)
    G.eval()

    info = {
        "model_name": model_name,
        "checkpoint_path": ckpt_path,
        "checkpoint_epoch": ckpt.get("epoch", -1),
        "state_key_loaded": state_key,
        "backbone": spec["backbone"],
        "constraint": spec["constraint"],
        "cell": spec["cell"],
        "config": cfg,
    }

    return G, info


def generate_synthetic_codes(
    generator: nn.Module,
    cond_vector: np.ndarray,
    district_ids: np.ndarray,
    state_ids: np.ndarray,
    noise_dim: int = 128,
    batch_size: int = 512,
    temp: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
    rejection_sampling: bool = False,
) -> np.ndarray:
    """
    Batched generator forward pass converting synthetic logits to discrete 1..9 ICATUS codes.
    If rejection_sampling=True, child labor violations (ICATUS 1 for Age < 15) are
    resampled until 0.00% delivered violation rate is achieved.
    """
    if isinstance(generator, ActVAEAdapter):
        return generator.get_samples_and_cond(len(cond_vector), seed=seed)[0]

    generator.eval()
    N = len(cond_vector)
    all_codes = []

    # Deterministic noise generation with PyTorch generator
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(seed)

    with torch.no_grad():
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            b_size = end_idx - start_idx

            c_b = torch.from_numpy(cond_vector[start_idx:end_idx]).float().to(device)
            d_b = torch.from_numpy(district_ids[start_idx:end_idx]).long().to(device)
            s_b = torch.from_numpy(state_ids[start_idx:end_idx]).long().to(device)
            z_b = torch.randn(b_size, noise_dim, generator=torch_rng, device=device)

            fake_tensor = generator(z_b, c_b, d_b, s_b, temp=temp, hard=True)
            # fake_tensor shape: (B, 9, 48, 1) -> argmax along dim 1 -> (B, 48)
            batch_codes = (torch.argmax(fake_tensor, dim=1).squeeze(-1) + 1).cpu().numpy()

            if rejection_sampling:
                # Identify child labor violations in this batch
                is_child = (c_b[:, 0] == 1.0).cpu().numpy()
                has_work = np.any(batch_codes == 1, axis=1)
                violations = is_child & has_work

                max_retries = 10
                retry = 0
                while np.any(violations) and retry < max_retries:
                    n_viol = int(np.sum(violations))
                    z_retry = torch.randn(n_viol, noise_dim, generator=torch_rng, device=device)
                    fake_retry = generator(
                        z_retry,
                        c_b[violations],
                        d_b[violations],
                        s_b[violations],
                        temp=temp,
                        hard=True,
                    )
                    retry_codes = (torch.argmax(fake_retry, dim=1).squeeze(-1) + 1).cpu().numpy()
                    batch_codes[violations] = retry_codes

                    # Re-check violations for newly sampled schedules
                    has_work_retry = np.any(retry_codes == 1, axis=1)
                    violations[violations] = has_work_retry
                    retry += 1

            all_codes.append(batch_codes)

    return np.vstack(all_codes)
