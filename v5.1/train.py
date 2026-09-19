# -*- coding: utf-8 -*-
"""
TUS-GAN — Training Script (v5.1 / Cell D)
=========================================
Trains the Transformer-Based Conditional Generator with Differentiable Soft Penalty (Cell D).

Key Characteristics:
  - Backbone: Transformer (Temporal Multi-Head Attention blocks)
  - Enforcement: Differentiable Soft Penalty (lambda_logic = 10.0 on soft probabilities)
  - Auxiliary Losses:
      * InfoNCE Contrastive Loss (Critic demographic-sequence alignment)
      * Time-Slice Transition Matrix Loss (Morning, Afternoon, Evening, Night)
      * Differentiable Spell-Duration Loss (Contiguous duration alignment)
  - Diagnostics: Automatic per-layer gradient norm logging at checkpoints (epochs 10, 50, 100, 250)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from generator import Generator
from critic import Critic, compute_gradient_penalty, critic_loss, generator_loss


# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────


class TUSDataset(Dataset):
    """
    Loads pre-encoded 9-channel ITUS dataset with state and district embeddings.
    """

    def __init__(self, npz_path: str, subset_size: int = None, device=None):
        if not os.path.exists(npz_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(script_dir, npz_path),
                os.path.join(script_dir, "..", "v4", "tusgan_encode.npz"),
                os.path.join(script_dir, "..", "v5", "tusgan_encode.npz"),
                "v4/tusgan_encode.npz",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    npz_path = cand
                    break

        data = np.load(npz_path)
        diary = torch.from_numpy(data["diary_tensor"]).float()  # (N, 9, 48, 1)
        cond = torch.from_numpy(data["cond_vector"]).float()    # (N, 83)
        district_ids = torch.from_numpy(data["district_ids"]).long()
        state_ids = torch.from_numpy(data["state_ids"]).long()

        if subset_size is not None and subset_size > 0:
            diary = diary[:subset_size]
            cond = cond[:subset_size]
            district_ids = district_ids[:subset_size]
            state_ids = state_ids[:subset_size]

        self.diary = diary
        self.cond = cond
        self.district_ids = district_ids
        self.state_ids = state_ids

        self.num_districts = int(data["num_districts"])
        self.num_states = int(data["num_states"])
        self.num_channels = self.diary.shape[1]
        self.cond_dim = self.cond.shape[1]

        print(f"✅ [v5.1] Dataset loaded: {len(self.diary):,} diaries")
        print(f"   Shape: {self.diary.shape}, Cond dim: {self.cond_dim}")
        print(f"   Districts: {self.num_districts}, States: {self.num_states}")

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.district_ids[idx], self.state_ids[idx]


# ─────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS & CONFIGURATION
# ─────────────────────────────────────────────────────────────


def get_config():
    return dict(
        data_path="../v4/tusgan_encode.npz",
        num_workers=2,
        subset=None,
        noise_dim=128,
        cond_dim=83,
        district_embed_dim=16,
        state_embed_dim=8,
        g_base_channels=256,
        d_base_channels=64,
        proj_dim=64,
        num_channels=9,
        epochs=250,
        batch_size=512,
        n_critic=5,
        lambda_gp=10.0,
        lr=0.0001,
        beta1=0.0,
        beta2=0.9,
        log_every=50,
        save_every=10,
        sample_every=10,
        n_samples=16,
        checkpoint_dir="v5.1/checkpoints",
        sample_dir="v5.1/samples",
        log_dir="v5.1/runs/tusgan_v5_1_soft",
        resume=None,
        # Transition & Sequence Regularization
        lambda_transition=1.0,
        ema_decay=0.999,
        gumbel_temp_start=1.0,
        gumbel_temp_min=0.1,
        gumbel_temp_decay=0.013,
        # Contrastive InfoNCE and Duration losses
        lambda_infonce=0.5,
        infonce_temp=0.1,
        lambda_duration=1.0,
        # Differentiable Neuro-Symbolic Soft Constraint (Cell D)
        lambda_logic=10.0,
        # Checkpoints for gradient diagnostics (Figure 8 in paper)
        grad_diag_epochs=[10, 50, 100, 250],
    )


# ─────────────────────────────────────────────────────────────
# 3. EMA & CHECKPOINTS
# ─────────────────────────────────────────────────────────────


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


def save_checkpoint(path, epoch, G, D, opt_G, opt_D, ema, cfg):
    G_net = G.module if isinstance(G, nn.DataParallel) else G
    D_net = D.module if isinstance(D, nn.DataParallel) else D

    ema.apply_shadow()
    g_state_ema = G_net.state_dict()
    ema.restore()

    torch.save(
        {
            "epoch": epoch,
            "G_state": G_net.state_dict(),
            "G_state_ema": g_state_ema,
            "D_state": D_net.state_dict(),
            "opt_G_state": opt_G.state_dict(),
            "opt_D_state": opt_D.state_dict(),
            "config": cfg,
        },
        path,
    )
    print(f"  [v5.1] Checkpoint saved → {path}")


def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)
    G_net = G.module if isinstance(G, nn.DataParallel) else G
    D_net = D.module if isinstance(D, nn.DataParallel) else D

    G_net.load_state_dict(ckpt["G_state"])
    D_net.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    print(f"  [v5.1] Resumed from {path} at epoch {ckpt['epoch']}")
    return ckpt["epoch"] + 1


# ─────────────────────────────────────────────────────────────
# 4. LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────


def compute_neuro_symbolic_loss(probs_fake, cond_vec):
    """
    Differentiable soft constraint loss for child labor prohibition (C1).
    probs_fake: (B, 9, 48) in [0, 1]
    cond_vec: (B, 83) where cond_vec[:, 0] == 1 indicates Age < 15.
    """
    loss = 0.0
    child_mask = cond_vec[:, 0] == 1
    if child_mask.any():
        child_work = probs_fake[child_mask, 0, :]  # Channel 0 = Employment
        loss = torch.mean(child_work)
    return loss


def compute_infonce_loss(feat_proj, cond_proj, temperature=0.1):
    logits = torch.matmul(feat_proj, cond_proj.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_f2c = F.cross_entropy(logits, labels)
    loss_c2f = F.cross_entropy(logits.T, labels)
    return (loss_f2c + loss_c2f) / 2.0


def compute_spell_duration_loss(probs_fake, real_avg_durations):
    total_time = probs_fake.sum(dim=2)
    x_t = probs_fake[:, :, 1:]
    x_tm1 = probs_fake[:, :, :-1]
    starts = (x_t * (1 - x_tm1)).sum(dim=2) + probs_fake[:, :, 0]
    starts = starts.clamp(min=0.5)
    avg_duration = total_time / starts
    avg_duration = avg_duration.clamp(max=48.0)
    batch_avg_duration = avg_duration.mean(dim=0)
    return F.mse_loss(batch_avg_duration, real_avg_durations)


def get_real_avg_durations(dataset, device):
    real_probs = (dataset.diary.squeeze(-1) + 1.0) / 2.0
    total_time = real_probs.sum(dim=2)
    x_t = real_probs[:, :, 1:]
    x_tm1 = real_probs[:, :, :-1]
    starts = (x_t * (1 - x_tm1)).sum(dim=2) + real_probs[:, :, 0]
    starts = starts.clamp(min=0.5)
    avg_duration = total_time / starts
    return avg_duration.mean(dim=0).to(device)


# ─────────────────────────────────────────────────────────────
# 5. GRADIENT DIAGNOSTIC LOGGER (Section 6.5 / Figure 8)
# ─────────────────────────────────────────────────────────────


def log_gradient_flow_diagnostics(G, D, real_avg_durations, device, epoch, cfg, writer=None):
    """
    Measures per-layer gradient norms split by constrained (Child) vs unconstrained (Adult)
    profiles for Cell D to confirm that soft penalties preserve healthy gradient flow.
    """
    G.eval()
    D.eval()
    diag_batch = 32
    grad_norms = {}

    scenarios = {"Adults_Unconstrained": 0.0, "Children_SoftPenalty": 1.0}
    for name, is_child_val in scenarios.items():
        G.zero_grad()
        z = torch.randn(diag_batch, cfg["noise_dim"], device=device)
        cond = torch.zeros(diag_batch, cfg["cond_dim"], device=device)
        cond[:, 0] = is_child_val
        dist = torch.zeros(diag_batch, dtype=torch.long, device=device)
        state = torch.zeros(diag_batch, dtype=torch.long, device=device)

        fake_hard, fake_soft = G(z, cond, dist, state, temp=1.0, hard=True, return_soft=True)
        scores, f_proj, c_proj = D(fake_hard, cond, dist, state, return_proj=True)

        loss_infonce = compute_infonce_loss(f_proj, c_proj, cfg["infonce_temp"])
        probs_fake_soft = (fake_soft.squeeze(-1) + 1.0) / 2.0
        loss_duration = compute_spell_duration_loss(probs_fake_soft, real_avg_durations)
        loss_logic = compute_neuro_symbolic_loss(probs_fake_soft, cond)

        loss_G = -torch.mean(scores) + cfg["lambda_infonce"] * loss_infonce + cfg["lambda_duration"] * loss_duration + cfg["lambda_logic"] * loss_logic
        loss_G.backward()

        layer_norms = {}
        for param_name, param in G.named_parameters():
            if "weight" in param_name and param.grad is not None:
                layer_norms[param_name] = param.grad.norm().item()
        grad_norms[name] = layer_norms

    out_path = os.path.join(cfg["checkpoint_dir"], f"grad_diagnostics_epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "grad_norms": grad_norms}, out_path)
    print(f"  📊 Gradient diagnostics recorded at epoch {epoch} → {out_path}")

    if writer:
        for scen, layers in grad_norms.items():
            for layer, val in layers.items():
                writer.add_scalar(f"GradNorm_{scen}/{layer}", val, epoch)

    G.train()
    D.train()


# ─────────────────────────────────────────────────────────────
# 6. MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────


def train(cfg):
    if cfg.get("seed") is not None:
        seed = cfg["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"🌱 Random seed set to {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["sample_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    dataset = TUSDataset(cfg["data_path"], subset_size=cfg.get("subset"))
    cfg["cond_dim"] = dataset.cond_dim
    cfg["num_districts"] = dataset.num_districts
    cfg["num_states"] = dataset.num_states
    cfg["num_channels"] = dataset.num_channels

    print("Computing real time-slice transition matrices...")
    P_real_slices = []
    with torch.no_grad():
        real_probs_all = (dataset.diary.squeeze(-1) + 1.0) / 2.0
        for k in range(4):
            start, end = k * 12, (k + 1) * 12
            x_t = real_probs_all[..., start : end - 1]
            x_tp1 = real_probs_all[..., start + 1 : end]
            trans_real = torch.einsum("b i t, b j t -> i j", x_t, x_tp1)
            row_sums = trans_real.sum(dim=1, keepdim=True)
            P_real_k = (trans_real / (row_sums + 1e-8)).to(device)
            P_real_slices.append(P_real_k)

    print("Computing real average spell durations...")
    real_avg_durations = get_real_avg_durations(dataset, device)

    num_workers = min(4, os.cpu_count() or 2)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    data_iter = iter(loader)

    G = Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=cfg["cond_dim"],
        num_districts=dataset.num_districts,
        num_states=dataset.num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["g_base_channels"],
    ).to(device)

    D = Critic(
        cond_dim=dataset.cond_dim,
        num_districts=dataset.num_districts,
        num_states=dataset.num_states,
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["d_base_channels"],
        proj_dim=cfg["proj_dim"],
    ).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg["epochs"], eta_min=1e-6)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg["epochs"], eta_min=1e-6)

    ema = EMA(G, decay=cfg["ema_decay"])

    start_epoch = 1
    if cfg["resume"]:
        start_epoch = load_checkpoint(cfg["resume"], G, D, opt_G, opt_D, device)

    writer = SummaryWriter(cfg["log_dir"])
    global_step = (start_epoch - 1) * (len(loader) // cfg["n_critic"])

    print(f"\n🚀 Starting Cell D Training (v5.1: Transformer + Soft Penalty) -> {cfg['epochs']} Epochs\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        current_temp = max(
            cfg["gumbel_temp_min"],
            cfg["gumbel_temp_start"] * np.exp(-cfg["gumbel_temp_decay"] * (epoch - 1)),
        )

        epoch_loss_D, epoch_loss_G, epoch_loss_trans, epoch_loss_infonce, epoch_loss_logic, epoch_loss_duration = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        epoch_gp, epoch_w_dist = 0.0, 0.0
        n_critic_steps, n_gen_steps = 0, 0
        n_gen_batches = max(1, len(loader) // cfg["n_critic"])

        for gen_step in range(n_gen_batches):
            # STEP A: Train Critic with InfoNCE
            for _ in range(cfg["n_critic"]):
                try:
                    real_diaries, cond_vec, dist_ids, state_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    real_diaries, cond_vec, dist_ids, state_ids = next(data_iter)

                real_diaries = real_diaries.to(device).float()
                cond_vec = cond_vec.to(device).float()
                dist_ids = dist_ids.to(device).long()
                state_ids = state_ids.to(device).long()
                B = real_diaries.size(0)

                z = torch.randn(B, cfg["noise_dim"], device=device)
                with torch.no_grad():
                    fake_diaries = G(z, cond_vec, dist_ids, state_ids, temp=current_temp, hard=True)

                real_scores, f_proj_real, c_proj_real = D(real_diaries, cond_vec, dist_ids, state_ids, return_proj=True)
                fake_scores = D(fake_diaries, cond_vec, dist_ids, state_ids, return_proj=False)

                gp = compute_gradient_penalty(
                    D, real_diaries, fake_diaries, cond_vec, dist_ids, state_ids, device, cfg["lambda_gp"]
                )
                loss_infonce = compute_infonce_loss(f_proj_real, c_proj_real, cfg["infonce_temp"])
                loss_D = critic_loss(real_scores, fake_scores, gp) + cfg["lambda_infonce"] * loss_infonce

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                epoch_w_dist += (real_scores.mean() - fake_scores.mean()).item()
                epoch_loss_D += loss_D.item()
                epoch_loss_infonce += loss_infonce.item()
                epoch_gp += gp.item()
                n_critic_steps += 1

            # STEP B: Train Generator (with Soft Penalty + InfoNCE + Spell Duration + Transitions)
            z_g = torch.randn(B, cfg["noise_dim"], device=device)
            fake_hard, fake_soft = G(z_g, cond_vec, dist_ids, state_ids, temp=current_temp, hard=True, return_soft=True)
            fake_scores, f_proj_fake, c_proj_fake = D(fake_hard, cond_vec, dist_ids, state_ids, return_proj=True)

            probs_fake_soft = (fake_soft.squeeze(-1) + 1.0) / 2.0

            # Time-slice transition loss
            loss_trans_slices = []
            for k in range(4):
                start, end = k * 12, (k + 1) * 12
                x_t_f = probs_fake_soft[..., start : end - 1]
                x_tp1_f = probs_fake_soft[..., start + 1 : end]
                trans_fake = torch.einsum("b i t, b j t -> i j", x_t_f, x_tp1_f)
                row_sums_f = trans_fake.sum(dim=1, keepdim=True)
                P_fake_k = trans_fake / (row_sums_f + 1e-8)
                loss_trans_slices.append(torch.mean((P_fake_k - P_real_slices[k]) ** 2))
            loss_trans = torch.mean(torch.stack(loss_trans_slices))

            loss_infonce_g = compute_infonce_loss(f_proj_fake, c_proj_fake, cfg["infonce_temp"])
            loss_duration = compute_spell_duration_loss(probs_fake_soft, real_avg_durations)
            loss_logic = compute_neuro_symbolic_loss(probs_fake_soft, cond_vec)

            loss_G = (
                generator_loss(fake_scores)
                + cfg["lambda_transition"] * loss_trans
                + cfg["lambda_infonce"] * loss_infonce_g
                + cfg["lambda_duration"] * loss_duration
                + cfg["lambda_logic"] * loss_logic
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            ema.update()

            epoch_loss_G += loss_G.item()
            epoch_loss_trans += loss_trans.item()
            epoch_loss_logic += loss_logic.item()
            epoch_loss_duration += loss_duration.item()
            n_gen_steps += 1
            global_step += 1

        sched_G.step()
        sched_D.step()

        # Checkpoint & Gradient Diagnostic logging
        if epoch in cfg["grad_diag_epochs"]:
            log_gradient_flow_diagnostics(G, D, real_avg_durations, device, epoch, cfg, writer)

        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, G, D, opt_G, opt_D, ema, cfg)

        avg_loss_D = epoch_loss_D / max(1, n_critic_steps)
        avg_loss_G = epoch_loss_G / max(1, n_gen_steps)
        avg_w = epoch_w_dist / max(1, n_critic_steps)
        print(f"Epoch [{epoch:>4}/{cfg['epochs']}] | W-dist: {avg_w:+.4f} | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f} | Logic: {epoch_loss_logic/max(1,n_gen_steps):.4f}")

    # Save final
    final_path = os.path.join(cfg["checkpoint_dir"], "final.pt")
    save_checkpoint(final_path, cfg["epochs"], G, D, opt_G, opt_D, ema, cfg)
    print(f"🎉 [v5.1] Cell D Training Complete. Final saved → {final_path}")


if __name__ == "__main__":
    # A100 / Ampere GPU Optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Disable optimized SDP attention backends because they do not support
    # double backward passes required for WGAN-GP's gradient penalty.
    # We force the math backend instead.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    parser = argparse.ArgumentParser(description="Train TUS-GAN v5.1 (Cell D: Transformer + Soft Penalty)")
    parser.add_argument("--data", type=str, default="tusgan_encode.npz", help="Path to npz dataset")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for quick debug")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    cfg = get_config()
    cfg["data_path"] = args.data
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch
    cfg["subset"] = args.subset
    cfg["seed"] = args.seed
    cfg["checkpoint_dir"] = args.checkpoint_dir
    cfg["sample_dir"] = os.path.join(args.checkpoint_dir, "..", "samples")
    cfg["log_dir"] = os.path.join(args.checkpoint_dir, "..", "runs")
    cfg["resume"] = args.resume

    train(cfg)
