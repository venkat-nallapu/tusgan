# -*- coding: utf-8 -*-
"""
TUS-GAN — Training Script for 9-Channel (v5 Ultimate Realism)
==============================================================
Trains the conditional Generator and Critic using WGAN-GP.

v5 Updates:
  - Contrastive InfoNCE Loss (Demographic Entanglement).
  - Differentiable Spell-Duration Loss (Fragmentation Penalty).
  - Hard Deterministic Logit Masking (Handled in Generator).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# Import our models
from generator import Generator
from critic import Critic, compute_gradient_penalty, critic_loss, generator_loss

# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────


class TUSDataset(Dataset):
    def __init__(self, npz_path: str, subset_size: int = None, device=None):
        if not os.path.exists(npz_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            npz_path = os.path.join(script_dir, os.path.basename(npz_path))

        data = np.load(npz_path)
        diary = torch.from_numpy(data["diary_tensor"]).float()  # (N, 9, 48, 1)
        cond = torch.from_numpy(data["cond_vector"]).float()  # (N, 83)
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

        print(f"✅ Dataset loaded: {len(self.diary):,} diaries")
        print(f"   Diary shape: {self.diary.shape} (9-channel representation)")
        print(f"   Cond dim: {self.cond_dim}")
        print(f"   Num districts: {self.num_districts}")
        print(f"   Num states: {self.num_states}")

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.district_ids[idx], self.state_ids[idx]


# ─────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────


def get_config():
    return dict(
        data_path="../v4/tusgan_encode.npz",  # Use the same encode file
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
        checkpoint_dir="checkpoints",
        sample_dir="samples",
        log_dir="runs/tusgan_v5_ultimate",
        resume=None,
        # v3 transition losses
        lambda_transition=1.0,
        ema_decay=0.999,
        gumbel_temp_start=1.0,
        gumbel_temp_min=0.1,
        gumbel_temp_decay=0.013,
        # v5 infoNCE & duration matching
        lambda_infonce=0.5,
        infonce_temp=0.1,
        lambda_duration=1.0,
    )


# ─────────────────────────────────────────────────────────────
# 3. CHECKPOINT HELPERS & EMA DEFINITION
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
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)

    G_net = G.module if isinstance(G, nn.DataParallel) else G
    D_net = D.module if isinstance(D, nn.DataParallel) else D

    G_net.load_state_dict(ckpt["G_state"])
    D_net.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    start_epoch = ckpt["epoch"] + 1
    print(f"  Resumed from {path} at epoch {ckpt['epoch']}")
    return start_epoch


# ─────────────────────────────────────────────────────────────
# 4. v5 ADVANCED LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────


def compute_infonce_loss(feat_proj, cond_proj, temperature=0.1):
    """
    Computes InfoNCE loss to maximize mutual information between
    the sequence features and the demographic conditioning.
    """
    # Compute similarity matrix (B, B)
    logits = torch.matmul(feat_proj, cond_proj.T) / temperature

    # Target is the diagonal (matching diary to its own demographic)
    labels = torch.arange(logits.size(0), device=logits.device)

    # Cross entropy over both dimensions
    loss_f2c = F.cross_entropy(logits, labels)
    loss_c2f = F.cross_entropy(logits.T, labels)
    return (loss_f2c + loss_c2f) / 2.0


def compute_spell_duration_loss(probs_fake, real_avg_durations):
    """
    Differentiable spell duration penalty.
    Calculates the approximate average continuous block length per activity
    and penalizes the MSE against the real dataset's average.
    """
    loss = 0.0
    # probs_fake shape: (B, 9, 48)
    total_time = probs_fake.sum(dim=2)  # (B, 9)

    # Number of starts: sum of max(0, x_t - x_{t-1})
    # Approximate using smooth ReLU equivalent or just x_t * (1 - x_{t-1})
    x_t = probs_fake[:, :, 1:]
    x_tm1 = probs_fake[:, :, :-1]
    starts = (x_t * (1 - x_tm1)).sum(dim=2)  # (B, 9)
    # Plus initial state
    starts = starts + probs_fake[:, :, 0]

    # Add small epsilon to avoid div by zero
    starts = starts.clamp(min=0.5)
    avg_duration = total_time / starts  # (B, 9)
    avg_duration = avg_duration.clamp(max=48.0)

    # Average across batch
    batch_avg_duration = avg_duration.mean(dim=0)  # (9,)

    loss = F.mse_loss(batch_avg_duration, real_avg_durations)
    return loss


def get_real_avg_durations(dataset, device):
    real_probs = (dataset.diary.squeeze(-1) + 1.0) / 2.0  # (N, 9, 48)
    total_time = real_probs.sum(dim=2)
    x_t = real_probs[:, :, 1:]
    x_tm1 = real_probs[:, :, :-1]
    starts = (x_t * (1 - x_tm1)).sum(dim=2) + real_probs[:, :, 0]
    avg_duration = total_time / (starts + 1e-5)

    return avg_duration.mean(dim=0).to(device)


# ─────────────────────────────────────────────────────────────
# 5. SAMPLE HELPER
# ─────────────────────────────────────────────────────────────


def save_samples(G, fixed_z, fixed_cond, fixed_dist, fixed_state, epoch, sample_dir, writer):
    G.eval()
    with torch.no_grad():
        fake = G(fixed_z, fixed_cond, fixed_dist, fixed_state)
    G.train()

    fake_np = fake.cpu().numpy()
    path = os.path.join(sample_dir, f"epoch_{epoch:04d}.npy")
    np.save(path, fake_np)
    return fake_np


# ─────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────────────────────


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["sample_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    data_path = cfg["data_path"]
    dataset = TUSDataset(data_path, subset_size=cfg.get("subset"))

    cfg["cond_dim"] = dataset.cond_dim
    cfg["num_districts"] = dataset.num_districts
    cfg["num_states"] = dataset.num_states
    cfg["num_channels"] = dataset.num_channels

    print("Computing real time-slice transition matrices from dataset...")
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
    print(f"Real Target Durations (30-min slots): {real_avg_durations.cpu().numpy()}")

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

    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"   Generator: {g_params:,}")
    print(f"   Critic:    {d_params:,}")
    print(f"   Total:     {g_params + d_params:,}")

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg["epochs"], eta_min=1e-6)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg["epochs"], eta_min=1e-6)

    ema = EMA(G, decay=cfg["ema_decay"])

    # Wrap in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs via DataParallel!")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    start_epoch = 1
    if cfg["resume"]:
        start_epoch = load_checkpoint(cfg["resume"], G, D, opt_G, opt_D, device)

    writer = SummaryWriter(cfg["log_dir"])

    fixed_batch = next(iter(DataLoader(dataset, batch_size=cfg["n_samples"], shuffle=True)))
    fixed_cond = fixed_batch[1].to(device).float()
    fixed_dist = fixed_batch[2].to(device).long()
    fixed_state = fixed_batch[3].to(device).long()
    fixed_z = torch.randn(cfg["n_samples"], cfg["noise_dim"], device=device)

    global_step = (start_epoch - 1) * (len(loader) // cfg["n_critic"])

    print(f"\n🚀 Starting TUS-GAN v5 Ultimate Realism Training\n")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        current_temp = max(
            cfg["gumbel_temp_min"],
            cfg["gumbel_temp_start"] * np.exp(-cfg["gumbel_temp_decay"] * (epoch - 1)),
        )

        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_loss_trans = 0.0
        epoch_loss_infonce = 0.0
        epoch_loss_duration = 0.0
        epoch_w_dist = 0.0
        epoch_gp = 0.0
        n_critic_steps = 0
        n_gen_steps = 0

        n_gen_batches = max(1, len(loader) // cfg["n_critic"])

        for gen_step in range(n_gen_batches):

            # --- Train Critic ---
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

                real_scores, f_proj_real, c_proj_real = D(
                    real_diaries, cond_vec, dist_ids, state_ids, return_proj=True
                )
                fake_scores = D(fake_diaries, cond_vec, dist_ids, state_ids, return_proj=False)

                gp = compute_gradient_penalty(
                    D,
                    real_diaries,
                    fake_diaries,
                    cond_vec,
                    dist_ids,
                    state_ids,
                    device,
                    cfg["lambda_gp"],
                )

                # InfoNCE Contrastive Loss (Train critic to align real diaries with demographics)
                loss_infonce = compute_infonce_loss(f_proj_real, c_proj_real, cfg["infonce_temp"])

                loss_D = (
                    critic_loss(real_scores, fake_scores, gp) + cfg["lambda_infonce"] * loss_infonce
                )

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                w_dist = (real_scores.mean() - fake_scores.mean()).item()
                epoch_w_dist += w_dist
                epoch_loss_D += loss_D.item()
                epoch_gp += gp.item()
                n_critic_steps += 1

            # --- Train Generator ---
            z_g = torch.randn(B, cfg["noise_dim"], device=device)
            fake_hard, fake_soft = G(z_g, cond_vec, dist_ids, state_ids, temp=current_temp, hard=True, return_soft=True)

            fake_scores, f_proj_fake, c_proj_fake = D(
                fake_hard, cond_vec, dist_ids, state_ids, return_proj=True
            )

            probs_fake = (fake_soft.squeeze(-1) + 1.0) / 2.0

            # 1. Time-Slice Transition Matrix Loss
            loss_trans_slices = []
            for k in range(4):
                start, end = k * 12, (k + 1) * 12
                x_t_f = probs_fake[..., start : end - 1]
                x_tp1_f = probs_fake[..., start + 1 : end]
                trans_fake = torch.einsum("b i t, b j t -> i j", x_t_f, x_tp1_f)
                row_sums_f = trans_fake.sum(dim=1, keepdim=True)
                P_fake_k = trans_fake / (row_sums_f + 1e-8)
                loss_trans_k = torch.mean((P_fake_k - P_real_slices[k]) ** 2)
                loss_trans_slices.append(loss_trans_k)
            loss_trans = torch.mean(torch.stack(loss_trans_slices))

            # 2. InfoNCE Loss for Generator (fool the critic by producing sequences perfectly aligned with condition)
            loss_infonce_G = compute_infonce_loss(f_proj_fake, c_proj_fake, cfg["infonce_temp"])

            # 3. Differentiable Spell-Duration Loss
            loss_duration = compute_spell_duration_loss(probs_fake, real_avg_durations)

            # Total Generator Loss
            loss_G = (
                generator_loss(fake_scores)
                + cfg["lambda_transition"] * loss_trans
                + cfg["lambda_infonce"] * loss_infonce_G
                + cfg["lambda_duration"] * loss_duration
            )

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            ema.update()

            epoch_loss_G += loss_G.item()
            epoch_loss_trans += loss_trans.item()
            epoch_loss_infonce += loss_infonce_G.item()
            epoch_loss_duration += loss_duration.item()
            n_gen_steps += 1
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                writer.add_scalar("Batch/loss_D", loss_D.item(), global_step)
                writer.add_scalar("Batch/loss_G", loss_G.item(), global_step)
                writer.add_scalar("Batch/loss_transition", loss_trans.item(), global_step)
                writer.add_scalar("Batch/loss_infonce", loss_infonce_G.item(), global_step)
                writer.add_scalar("Batch/loss_duration", loss_duration.item(), global_step)

        sched_G.step()
        sched_D.step()

        avg_loss_D = epoch_loss_D / n_critic_steps if n_critic_steps > 0 else 0
        avg_loss_G = epoch_loss_G / n_gen_steps if n_gen_steps > 0 else 0
        avg_loss_trans = epoch_loss_trans / n_gen_steps if n_gen_steps > 0 else 0
        avg_loss_duration = epoch_loss_duration / n_gen_steps if n_gen_steps > 0 else 0
        avg_loss_infonce = epoch_loss_infonce / n_gen_steps if n_gen_steps > 0 else 0
        avg_w_dist = epoch_w_dist / n_critic_steps if n_critic_steps > 0 else 0
        avg_gp = epoch_gp / n_critic_steps if n_critic_steps > 0 else 0

        writer.add_scalar("Epoch/loss_D", avg_loss_D, epoch)
        writer.add_scalar("Epoch/loss_G", avg_loss_G, epoch)
        writer.add_scalar("Epoch/loss_transition", avg_loss_trans, epoch)
        writer.add_scalar("Epoch/loss_duration", avg_loss_duration, epoch)
        writer.add_scalar("Epoch/loss_infonce", avg_loss_infonce, epoch)
        writer.add_scalar("Epoch/w_distance", avg_w_dist, epoch)
        writer.add_scalar("Epoch/gradient_penalty", avg_gp, epoch)
        writer.add_scalar("Epoch/gumbel_temp", current_temp, epoch)

        print(
            f"Epoch [{epoch:>4}/{cfg['epochs']}] | "
            f"W-dist: {avg_w_dist:+.4f} | "
            f"Loss_D: {avg_loss_D:.4f} | Loss_G: {avg_loss_G:.4f} | "
            f"L_Trans: {avg_loss_trans:.5f} | L_Dur: {avg_loss_duration:.5f} | L_Info: {avg_loss_infonce:.5f} | "
            f"Temp: {current_temp:.3f} | GP: {avg_gp:.4f}"
        )

        if epoch % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, G, D, opt_G, opt_D, ema, cfg)

        if epoch % cfg["sample_every"] == 0:
            ema.apply_shadow()
            fake_samples = save_samples(
                G, fixed_z, fixed_cond, fixed_dist, fixed_state, epoch, cfg["sample_dir"], writer
            )
            ema.restore()

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            axes[0].imshow(fake_samples[0, :, :, 0], aspect="auto", cmap="viridis")
            axes[0].set_title(f"Generated Synthetic Diary (Epoch {epoch})")
            axes[1].imshow(fixed_batch[0][0, :, :, 0].numpy(), aspect="auto", cmap="viridis")
            axes[1].set_title("Real Reference Diary")
            writer.add_figure("VisualComparison/DiaryHeatmap", fig, epoch)
            plt.close(fig)

    save_checkpoint(
        os.path.join(cfg["checkpoint_dir"], "final.pt"), cfg["epochs"], G, D, opt_G, opt_D, ema, cfg
    )
    writer.close()
    print("\n✅ TUS-GAN v5 Training complete!")


def parse_args():
    cfg = get_config()
    parser = argparse.ArgumentParser()
    # Simple parser for quick runs
    parser.add_argument("--data", type=str, default=cfg["data_path"])
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch", type=int, default=cfg["batch_size"])
    args, _ = parser.parse_known_args()

    cfg["data_path"] = args.data
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch

    print("\n" + "=" * 70)
    print("TUS-GAN 9-CHANNEL ULTIMATE REALISM TRAINING (v5)")
    print("=" * 70)
    print("\n📋 Configuration:")
    for k, v in cfg.items():
        print(f"   {k:<22}: {v}")
    print("=" * 70)

    return cfg


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Disable optimized SDP attention backends because they do not support
    # double backward passes required for WGAN-GP's gradient penalty.
    # We force the math backend instead.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    cfg = parse_args()
    train(cfg)
