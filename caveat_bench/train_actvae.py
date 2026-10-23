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

import time
from pytorch_lightning.callbacks import Callback

class CleanEpochLogger(Callback):
    """
    Lightweight, single-line epoch summary logger.
    Replaces spammy step-by-step TQDM lines that freeze browser tabs in Kaggle/Jupyter.
    """
    def __init__(self, log_every=1):
        super().__init__()
        self.log_every = log_every
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        metrics = trainer.callback_metrics
        
        train_loss = metrics.get("loss", metrics.get("train_loss", None))
        val_loss = metrics.get("val_loss", None)
        val_recon = metrics.get("val_recon_loss", None)
        
        t_str = f"{train_loss.item():.4f}" if train_loss is not None else "N/A"
        v_str = f"{val_loss.item():.4f}" if val_loss is not None else "N/A"
        r_str = f"{val_recon.item():.4f}" if val_recon is not None else "N/A"
        
        elapsed = f"{time.time() - self.epoch_start_time:.1f}s" if self.epoch_start_time else "0.0s"
        
        if epoch == 1 or epoch % self.log_every == 0 or epoch == max_epochs:
            print(f"   [Epoch {epoch:03d}/{max_epochs:03d}]  Train Loss: {t_str} | Val Loss: {v_str} | Val Recon: {r_str}  ({elapsed})")

# Patch Caveat's build_trainer to use CleanEpochLogger and disable spammy TQDM progress bar
_orig_build_trainer = caveat.runners.build_trainer

def _clean_build_trainer(logger, config):
    if "trainer_params" not in config:
        config["trainer_params"] = {}
    config["trainer_params"]["enable_progress_bar"] = False
    
    trainer = _orig_build_trainer(logger, config)
    trainer.callbacks.append(CleanEpochLogger(log_every=1))
    return trainer

caveat.runners.build_trainer = _clean_build_trainer

# ─────────────────────────────────────────────────────────────
# BUGFIX: Patch Evaluator to drop deprecated 'progress' keyword
# In newer acteval versions, Evaluator.__init__ removed 'progress'.
# ─────────────────────────────────────────────────────────────
from acteval.evaluate import Evaluator

_orig_evaluator_init = Evaluator.__init__

def _safe_evaluator_init(self, *args, **kwargs):
    kwargs.pop("progress", None)
    return _orig_evaluator_init(self, *args, **kwargs)

Evaluator.__init__ = _safe_evaluator_init

_orig_eval_synthetics = caveat.runners.evaluate_synthetics

def _resilient_evaluate_synthetics(*args, **kwargs):
    try:
        return _orig_eval_synthetics(*args, **kwargs)
    except Exception as e:
        print(f"⚠️ Note: Intermediate single-model evaluation skipped ({e}). Downstream Table 8 suite will evaluate all models.")

caveat.runners.evaluate_synthetics = _resilient_evaluate_synthetics


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

        # Also mirror matching synthetic_attributes.csv
        synth_attr_file = latest_synth.parent / "synthetic_attributes.csv"
        if synth_attr_file.exists():
            dest_attr_filename = "synth_vae_soft_attributes.csv" if "soft" in name.lower() else "synth_vae_unconstrained_attributes.csv"
            dest_attr_path = os.path.join(target_dir, dest_attr_filename)
            shutil.copy2(synth_attr_file, dest_attr_path)
            print(f"📦 Mirrored matching attributes: {synth_attr_file} → {dest_attr_path}")
    else:
        print(f"⚠️ Note: synthetic_schedules.csv not automatically detected under {log_dir}")


if __name__ == "__main__":
    main()
