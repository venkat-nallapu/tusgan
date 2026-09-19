# -*- coding: utf-8 -*-
"""
train_actvae.py: Trains Conditional Variational Autoencoder (ActVAE) baselines
using the Caveat framework on the ITUS 2019 dataset.

Supports:
1. Unconstrained CVAE (reproduces empirical ~3.87% child labor violation)
2. Soft-Constrained CVAE (CondDiscCNN2DSoft with lambda_logic=10.0 penalty)
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Add caveat package to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CAVEAT_PKG = os.path.join(REPO_ROOT, "caveat")
if CAVEAT_PKG not in sys.path:
    sys.path.insert(0, CAVEAT_PKG)

import caveat.models
from caveat.models.discrete.cond_discrete_conv import CondDiscCNN2D
from caveat.runners import run_command


# ─────────────────────────────────────────────────────────────
# BUGFIX: Ensure labels tensor is cast to float for nn.Linear fc_hidden
# ─────────────────────────────────────────────────────────────
_orig_decode = CondDiscCNN2D.decode

def _safe_decode(self, z, labels, **kwargs):
    if labels is not None and labels.dtype != torch.float32:
        labels = labels.float()
    return _orig_decode(self, z=z, labels=labels, **kwargs)

CondDiscCNN2D.decode = _safe_decode

_orig_loss_fn = CondDiscCNN2D.loss_function

def _safe_loss_fn(self, *args, **kwargs):
    res = _orig_loss_fn(self, *args, **kwargs)
    if "recon_loss" not in res and "recon_act_nlll_loss" in res:
        res["recon_loss"] = res["recon_act_nlll_loss"]
    return res

CondDiscCNN2D.loss_function = _safe_loss_fn


# ─────────────────────────────────────────────────────────────
# SOFT-CONSTRAINED ActVAE SUBCLASS
# ─────────────────────────────────────────────────────────────

class CondDiscCNN2DSoft(CondDiscCNN2D):
    """
    Subclasses Caveat's CondDiscCNN2D to add a differentiable soft penalty
    for child labor prohibition (C1).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_logic = float(kwargs.get("lambda_logic", 10.0))
        print(f"⚖️ CondDiscCNN2DSoft initialized with lambda_logic = {self.lambda_logic}")

    def discretized_loss(
        self,
        log_probs,
        mu,
        log_var,
        target,
        weights,
        label_weights=(None, None),
        labels=None,
        **kwargs,
    ) -> dict:
        # Standard ELBO loss from Base
        loss_dict = super().discretized_loss(
            log_probs=log_probs,
            mu=mu,
            log_var=log_var,
            target=target,
            weights=weights,
            label_weights=label_weights,
            labels=labels,
            **kwargs,
        )

        # Apply soft penalty on child employment
        # log_probs shape: [B, steps, acts] where acts=9
        # Channel 0 is Employment in our ITUS mapping
        if labels is not None and self.lambda_logic > 0.0:
            # labels is typically [B, num_labels]; is_child is the first label index (0)
            is_child = labels[:, 0] > 0.5
            if is_child.any():
                # Convert log-probabilities to soft probabilities in [0, 1]
                probs = torch.exp(log_probs)
                child_work_probs = probs[is_child, :, 0]  # channel 0 = employment
                logic_loss = torch.mean(child_work_probs)
                loss_dict["loss"] = loss_dict["loss"] + self.lambda_logic * logic_loss
                loss_dict["logic_loss"] = logic_loss.detach()

        return loss_dict


# Register custom model in Caveat's model registry
caveat.models.library["CondDiscCNN2DSoft"] = CondDiscCNN2DSoft


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING RUNNER
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ActVAE Baselines for TUS-GAN Benchmarking")
    parser.add_argument("--config", type=str, required=True, help="Path to Caveat YAML config")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose training progress")
    parser.add_argument("--no_gen", action="store_true", default=False, help="Disable generation after training")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    
    print(f"📖 Loading Caveat configuration: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure paths in config are relative to caveat_bench
    for key in ["schedules_path", "attributes_path"]:
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(SCRIPT_DIR, config[key])

    if "logging_params" in config and "log_dir" in config["logging_params"]:
        if not os.path.isabs(config["logging_params"]["log_dir"]):
            config["logging_params"]["log_dir"] = os.path.join(SCRIPT_DIR, config["logging_params"]["log_dir"])

    print(f"🚀 Launching ActVAE Training ({config.get('logging_params', {}).get('name', 'experiment')})...")
    run_command(
        config=config,
        verbose=args.verbose,
        gen=not args.no_gen,
        test=False,
        infer=True,
    )
    print("\n✅ ActVAE Training Run Complete!")

    # Mirror generated synthetic_schedules.csv to data/ directory for downstream evaluation
    log_dir = Path(config.get("logging_params", {}).get("log_dir", "outputs/actvae_logs"))
    name = config.get("logging_params", {}).get("name", "")
    target_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(target_dir, exist_ok=True)

    dest_filename = "synth_vae_soft_schedules.csv" if "soft" in name.lower() else "synth_vae_unconstrained_schedules.csv"
    dest_path = os.path.join(target_dir, dest_filename)

    found_files = list(log_dir.glob(f"**/{name}/**/synthetic_schedules.csv")) or list(log_dir.glob("**/synthetic_schedules.csv"))
    if found_files:
        import shutil
        latest_synth = max(found_files, key=os.path.getmtime)
        shutil.copy2(latest_synth, dest_path)
        print(f"📦 Mirrored generated schedules: {latest_synth} → {dest_path}")
    else:
        print(f"⚠️ Note: synthetic_schedules.csv not automatically detected under {log_dir}")


if __name__ == "__main__":
    main()
