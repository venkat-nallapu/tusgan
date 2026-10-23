# -*- coding: utf-8 -*-
"""
train_sweep.py: Parameterized Training Script for Section 6.8 Sensitivity Sweep
==============================================================================
Trains Cell A (TUS-GAN v4 CNN + AC-GAN + 4 Time-Slice Transitions) with variable
soft child-labor penalty weights: lambda_rule in {0.1, 1.0, 10.0, 100.0}.

Checkpoints and logs are completely isolated by lambda value to protect existing
production weights.
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import v4 models
V4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../v4"))
sys.path.insert(0, V4_DIR)
from generator import Generator
from critic import Critic, compute_gradient_penalty, critic_loss, generator_loss


class TUSDataset(Dataset):
    """Loads pre-encoded 9-channel ITUS 2019 dataset."""

    def __init__(self, npz_path: str, subset_size: int = None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found at: {npz_path}")

        data = np.load(npz_path)
        diary = torch.from_numpy(data["diary_tensor"]).float()
        cond = torch.from_numpy(data["cond_vector"]).float()
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

        print(f"✅ Loaded dataset: {len(self.diary):,} diaries (cond_dim={self.cond_dim})")

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.district_ids[idx], self.state_ids[idx]


class DemographicClassifier(nn.Module):
    """Auxiliary AC-GAN Classifier predicting demographic vector from diary."""

    def __init__(self, cond_dim: int, base_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(9, base_channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(base_channels * 2 * 12 * 1, cond_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class EMA:
    """Exponential Moving Average for generator shadow weights."""

    def __init__(self, model, decay: float = 0.999):
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
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def compute_neuro_symbolic_loss(probs_fake: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
    """
    Computes soft child labor penalty:
    Age < 15 is index 0 in cond_vec; Employment is channel 0.
    """
    child_mask = cond_vec[:, 0] == 1
    if child_mask.any():
        child_work = probs_fake[child_mask, 0, :]
        return torch.mean(child_work)
    return torch.tensor(0.0, device=probs_fake.device)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sweep_run(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    print(f"\n===============================================================================")
    print(f"🚀 Training Sensitivity Sweep: λ_rule = {cfg['lambda_logic']} | Seed = {cfg['seed']}")
    print(f"   Output Directory: {cfg['checkpoint_dir']}")
    print(f"   Device          : {device}")
    print(f"===============================================================================")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    # Save configuration
    with open(os.path.join(cfg["checkpoint_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    dataset = TUSDataset(cfg["data_path"], subset_size=cfg.get("subset"))
    cfg["cond_dim"] = dataset.cond_dim
    cfg["num_districts"] = dataset.num_districts
    cfg["num_states"] = dataset.num_states

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"Batches per epoch: {len(loader):,}")

    # Precompute real time-slice transition matrices
    print("Computing real time-slice transition matrices...")
    P_real_slices = []
    with torch.no_grad():
        real_probs = (dataset.diary.squeeze(-1) + 1.0) / 2.0
        for k in range(4):
            start, end = k * 12, (k + 1) * 12
            x_t = real_probs[..., start : end - 1]
            x_tp1 = real_probs[..., start + 1 : end]
            trans = torch.einsum("b i t, b j t -> i j", x_t, x_tp1)
            row_sums = trans.sum(dim=1, keepdim=True)
            P_real_k = (trans / (row_sums + 1e-8)).to(device)
            P_real_slices.append(P_real_k)

    # Initialize models
    G = Generator(
        noise_dim=cfg["noise_dim"],
        cond_dim=cfg["cond_dim"],
        num_districts=cfg["num_districts"],
        num_states=cfg["num_states"],
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["g_base_channels"],
    ).to(device)

    D = Critic(
        cond_dim=cfg["cond_dim"],
        num_districts=cfg["num_districts"],
        num_states=cfg["num_states"],
        district_embed_dim=cfg["district_embed_dim"],
        state_embed_dim=cfg["state_embed_dim"],
        base_channels=cfg["d_base_channels"],
    ).to(device)

    C = DemographicClassifier(cond_dim=cfg["cond_dim"], base_channels=32).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_C = torch.optim.Adam(C.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg["epochs"], eta_min=1e-6)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg["epochs"], eta_min=1e-6)
    sched_C = torch.optim.lr_scheduler.CosineAnnealingLR(opt_C, T_max=cfg["epochs"], eta_min=1e-6)

    ema = EMA(G, decay=cfg["ema_decay"])
    writer = SummaryWriter(cfg["log_dir"])
    criterion_C = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        temp = max(cfg["gumbel_temp_min"], cfg["gumbel_temp_start"] * np.exp(-cfg["gumbel_temp_decay"] * (epoch - 1)))
        G.train()
        D.train()
        C.train()

        epoch_loss_D, epoch_loss_G, epoch_logic = 0.0, 0.0, 0.0
        g_steps = 0
        for i, (real_diaries, cond_vec, dist_ids, state_ids) in enumerate(loader):
            B = real_diaries.size(0)
            real_diaries = real_diaries.to(device)
            cond_vec = cond_vec.to(device)
            dist_ids = dist_ids.to(device)
            state_ids = state_ids.to(device)

            # ── Critic Step ──
            z = torch.randn(B, cfg["noise_dim"], device=device)
            with torch.no_grad():
                fake_diaries = G(z, cond_vec, dist_ids, state_ids, temp=temp, hard=True)

            real_scores = D(real_diaries, cond_vec, dist_ids, state_ids)
            fake_scores = D(fake_diaries.detach(), cond_vec, dist_ids, state_ids)
            gp = compute_gradient_penalty(D, real_diaries, fake_diaries.detach(), cond_vec, dist_ids, state_ids, device, cfg["lambda_gp"])
            loss_D = critic_loss(real_scores, fake_scores, gp)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Classifier Step
            pred_c_real = C(real_diaries)
            loss_C = criterion_C(pred_c_real, cond_vec)
            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

            # ── Generator Step ──
            if i % cfg["n_critic"] == 0:
                z_g = torch.randn(B, cfg["noise_dim"], device=device)
                fake_g = G(z_g, cond_vec, dist_ids, state_ids, temp=temp, hard=True)
                fake_scores_g = D(fake_g, cond_vec, dist_ids, state_ids)

                probs_fake = (fake_g.squeeze(-1) + 1.0) / 2.0
                loss_trans_slices = []
                for k in range(4):
                    start, end = k * 12, (k + 1) * 12
                    x_t_f = probs_fake[..., start : end - 1]
                    x_tp1_f = probs_fake[..., start + 1 : end]
                    trans_f = torch.einsum("b i t, b j t -> i j", x_t_f, x_tp1_f)
                    row_sums_f = trans_f.sum(dim=1, keepdim=True)
                    P_f = trans_f / (row_sums_f + 1e-8)
                    loss_trans_slices.append(torch.mean((P_f - P_real_slices[k]) ** 2))
                loss_trans = torch.mean(torch.stack(loss_trans_slices))

                pred_c_fake = C(fake_g)
                loss_cond = criterion_C(pred_c_fake, cond_vec)
                loss_logic = compute_neuro_symbolic_loss(probs_fake, cond_vec)

                loss_G = (
                    generator_loss(fake_scores_g)
                    + cfg["lambda_transition"] * loss_trans
                    + cfg["lambda_cond"] * loss_cond
                    + cfg["lambda_logic"] * loss_logic
                )

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()
                ema.update()

                epoch_loss_G += loss_G.item()
                epoch_logic += loss_logic.item()
                g_steps += 1
                global_step += 1

            epoch_loss_D += loss_D.item()

        sched_G.step()
        sched_D.step()
        sched_C.step()

        if epoch % cfg["log_every_epoch"] == 0 or epoch == cfg["epochs"]:
            mean_g = epoch_loss_G / max(1, g_steps)
            mean_logic = epoch_logic / max(1, g_steps)
            print(f"Epoch [{epoch:03d}/{cfg['epochs']}] | Loss_D: {epoch_loss_D/len(loader):.4f} | Loss_G: {mean_g:.4f} | Logic_Penalty: {mean_logic:.6f} | Temp: {temp:.3f}")

        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            ema.apply_shadow()
            torch.save(
                {
                    "epoch": epoch,
                    "G_state": G.state_dict(),
                    "G_state_ema": G.state_dict(),
                    "D_state": D.state_dict(),
                    "C_state": C.state_dict(),
                    "config": cfg,
                },
                os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt" if epoch < cfg["epochs"] else "final.pt"),
            )
            ema.restore()

    writer.close()
    print(f"✅ Training completed! Model saved: {cfg['checkpoint_dir']}/final.pt\n")


def parse_args():
    parser = argparse.ArgumentParser(description="TUS-GAN Section 6.8 Sensitivity Sweep Trainer")
    parser.add_argument("--data", type=str, default="data/tusgan_encode.npz")
    parser.add_argument("--lambda_rule", type=float, default=10.0, help="Soft penalty weight (0.1, 1.0, 10.0, 100.0)")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every_epoch", type=int, default=10)
    args = parser.parse_args()

    # Resolve data path
    data_path = args.data
    if not os.path.exists(data_path):
        for candidate in ["v4/tusgan_encode.npz", "../data/tusgan_encode.npz", "../../data/tusgan_encode.npz"]:
            if os.path.exists(candidate):
                data_path = candidate
                break

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), f"checkpoints/lambda_{args.lambda_rule}")

    cfg = {
        "data_path": data_path,
        "checkpoint_dir": output_dir,
        "log_dir": os.path.join(output_dir, "runs"),
        "lambda_logic": args.lambda_rule,
        "lambda_rule": args.lambda_rule,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "seed": args.seed,
        "lr": args.lr,
        "subset": args.subset,
        "save_every": args.save_every,
        "log_every_epoch": args.log_every_epoch,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": min(4, os.cpu_count() or 2),
        "noise_dim": 128,
        "district_embed_dim": 16,
        "state_embed_dim": 8,
        "g_base_channels": 256,
        "d_base_channels": 64,
        "n_critic": 5,
        "lambda_gp": 10.0,
        "lambda_transition": 10.0,
        "lambda_cond": 1.0,
        "beta1": 0.0,
        "beta2": 0.9,
        "ema_decay": 0.999,
        "gumbel_temp_start": 1.0,
        "gumbel_temp_min": 0.1,
        "gumbel_temp_decay": 0.013,
    }
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train_sweep_run(cfg)
