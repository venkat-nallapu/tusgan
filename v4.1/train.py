# -*- coding: utf-8 -*-
"""
TUS-GAN — Training Script (v4.1 / Cell C)
=========================================
Trains the Convolutional Conditional Generator with Deterministic Hard Logit Masking (Cell C).
Corresponds to Cell C in the 2x2 Experimental Design of TUS-GAN V2 (Table 6).

Key Characteristics:
  - Backbone: Convolutional (Transposed Conv + CBN + Self-Attention)
  - Enforcement: Deterministic Hard Logit Masking (Pre-softmax logits set to -1e9 for child work)
  - Auxiliary Losses: Time-Slice Transition Matrix Loss + AC-GAN Conditioning Classifier
  - Multi-GPU: Full support for Kaggle Dual GPU (2x Tesla T4) via torchrun (DDP) or single-GPU fallback
  - Diagnostics: Automatic per-layer gradient norm logging at checkpoints (epochs 10, 50, 100, 250)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from generator import Generator
from critic import Critic, compute_gradient_penalty, critic_loss, generator_loss


def resolve_split_indices(split="train", split_indices_path=None, npz_path=None):
    """
    Resolves and loads the 90/10 train/test partition indices.
    """
    if split is None or str(split).lower() == "all":
        return None

    if split_indices_path and os.path.exists(split_indices_path):
        indices = np.load(split_indices_path)
        print(f"📁 Loaded explicit split indices from {split_indices_path} ({len(indices):,} records)")
        return indices

    fname = f"{str(split).lower()}_indices.npy"
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

    candidates = [
        fname,
        os.path.join(script_dir, fname),
        os.path.join(script_dir, "..", "data", fname),
        os.path.join("data", fname),
        os.path.join("..", "data", fname),
    ]
    if npz_path:
        npz_dir = os.path.dirname(os.path.abspath(npz_path))
        candidates.extend([
            os.path.join(npz_dir, fname),
            os.path.join(npz_dir, "..", "data", fname),
            os.path.join(npz_dir, "data", fname),
        ])

    for cand in candidates:
        if os.path.exists(cand):
            indices = np.load(cand)
            print(f"📁 Loaded split '{split}' from {cand} ({len(indices):,} records)")
            return indices

    print(f"⚠️ Warning: Split indices for '{split}' not found in candidates. Using full dataset.")
    return None


# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────


class TUSDataset(Dataset):
    """
    Loads pre-encoded 9-channel ITUS dataset with state and district embeddings.
    """

    def __init__(
        self,
        npz_path: str,
        subset_size: int = None,
        split: str = "train",
        split_indices: str = None,
    ):
        if not os.path.exists(npz_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                npz_path,
                os.path.join(script_dir, npz_path),
                os.path.join(script_dir, "tusgan_encode.npz"),
                os.path.join(script_dir, "..", "data", "tusgan_encode.npz"),
                os.path.join(script_dir, "..", "v4", "tusgan_encode.npz"),
                "tusgan_encode.npz",
                "data/tusgan_encode.npz",
                "v4/tusgan_encode.npz",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    npz_path = cand
                    break

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Could not locate dataset at {npz_path}")

        data = np.load(npz_path)
        diary = torch.from_numpy(data["diary_tensor"]).float()  # (N, 9, 48, 1)
        cond = torch.from_numpy(data["cond_vector"]).float()    # (N, 82)
        district_ids = torch.from_numpy(data["district_ids"]).long()
        state_ids = torch.from_numpy(data["state_ids"]).long()

        # Filter by train/test partition (default: 'train', 400,741 records)
        split_idx = resolve_split_indices(split=split, split_indices_path=split_indices, npz_path=npz_path)
        if split_idx is not None:
            diary = diary[split_idx]
            cond = cond[split_idx]
            district_ids = district_ids[split_idx]
            state_ids = state_ids[split_idx]
            split_info = f" (split='{split}', {len(diary):,} records)"
        else:
            split_info = " (full dataset)"

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
        self.split_info = split_info

    def __len__(self):
        return len(self.diary)

    def __getitem__(self, idx):
        return self.diary[idx], self.cond[idx], self.district_ids[idx], self.state_ids[idx]


# ─────────────────────────────────────────────────────────────
# 2. HYPERPARAMETERS & CONFIGURATION
# ─────────────────────────────────────────────────────────────


def get_config():
    return dict(
        data_path="tusgan_encode.npz",
        split="train",
        split_indices=None,
        num_workers=2,
        subset=None,
        seed=42,
        noise_dim=128,
        cond_dim=82,
        district_embed_dim=16,
        state_embed_dim=8,
        g_base_channels=256,
        d_base_channels=64,
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
        log_dir="runs/tusgan_v4_1_hard",
        resume=None,
        # Transition regularization
        lambda_transition=10.0,
        ema_decay=0.999,
        gumbel_temp_start=1.0,
        gumbel_temp_min=0.1,
        gumbel_temp_decay=0.013,
        # AC-GAN auxiliary classifier weight
        lambda_cond=1.0,
        # Epoch checkpoints for gradient flow diagnostics (Figure 8 in paper)
        grad_diag_epochs=[10, 50, 100, 250],
    )


# ─────────────────────────────────────────────────────────────
# 3. EMA & AUXILIARY CLASSIFIER
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
                if name in self.shadow:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()
                else:
                    self.shadow[name] = param.data.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class DemographicClassifier(nn.Module):
    """
    Auxiliary Classifier (AC-GAN) for demographic conditioning alignment.
    """

    def __init__(self, cond_dim=82, base_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(9, base_channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0))
        self.act2 = nn.LeakyReLU(0.2)
        flat_dim = base_channels * 2 * 12 * 1
        self.fc = nn.Linear(flat_dim, cond_dim)

    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.act2(self.conv2(h))
        h = h.view(h.size(0), -1)
        return self.fc(h)


def save_checkpoint(path, epoch, G, D, C, opt_G, opt_D, opt_C, ema, cfg):
    G_net = G.module if hasattr(G, "module") else G
    D_net = D.module if hasattr(D, "module") else D
    C_net = C.module if hasattr(C, "module") else C

    if ema is not None:
        ema.apply_shadow()
        g_state_ema = G_net.state_dict()
        ema.restore()
    else:
        g_state_ema = G_net.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "G_state": G_net.state_dict(),
            "G_state_ema": g_state_ema,
            "D_state": D_net.state_dict(),
            "C_state": C_net.state_dict(),
            "opt_G_state": opt_G.state_dict(),
            "opt_D_state": opt_D.state_dict(),
            "opt_C_state": opt_C.state_dict(),
            "config": cfg,
        },
        path,
    )
    print(f"  [v4.1] Checkpoint saved → {path}")


def load_checkpoint(path, G, D, C, opt_G, opt_D, opt_C, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    G_net = G.module if hasattr(G, "module") else G
    D_net = D.module if hasattr(D, "module") else D
    C_net = C.module if hasattr(C, "module") else C

    G_net.load_state_dict(ckpt["G_state"])
    D_net.load_state_dict(ckpt["D_state"])
    if "C_state" in ckpt and C_net is not None:
        C_net.load_state_dict(ckpt["C_state"])
    opt_G.load_state_dict(ckpt["opt_G_state"])
    opt_D.load_state_dict(ckpt["opt_D_state"])
    if "opt_C_state" in ckpt and opt_C is not None:
        opt_C.load_state_dict(ckpt["opt_C_state"])
    print(f"  [v4.1] Resumed from {path} at epoch {ckpt['epoch']}")
    return ckpt["epoch"] + 1


# ─────────────────────────────────────────────────────────────
# 4. GRADIENT DIAGNOSTIC LOGGER (Section 6.5 / Figure 8)
# ─────────────────────────────────────────────────────────────


def log_gradient_flow_diagnostics(G, D, device, epoch, cfg, writer=None):
    """
    Measures per-layer gradient norms split by constrained (Child) vs unconstrained (Adult)
    profiles to record empirical dead-gradient data for Figure 8 in the manuscript.
    """
    G_raw = G.module if hasattr(G, "module") else G
    D_raw = D.module if hasattr(D, "module") else D

    G_raw.eval()
    D_raw.eval()
    diag_batch = 32
    grad_norms = {}

    scenarios = {"Adults_Unconstrained": 0.0, "Children_HardMasked": 1.0}
    for name, is_child_val in scenarios.items():
        G_raw.zero_grad()
        z = torch.randn(diag_batch, cfg["noise_dim"], device=device)
        cond = torch.zeros(diag_batch, cfg["cond_dim"], device=device)
        cond[:, 0] = is_child_val
        dist_ids = torch.zeros(diag_batch, dtype=torch.long, device=device)
        state_ids = torch.zeros(diag_batch, dtype=torch.long, device=device)

        # Forward pass (with deterministic hard logit masking in G)
        fake = G_raw(z, cond, dist_ids, state_ids, temp=1.0, hard=True)
        scores = D_raw(fake, cond, dist_ids, state_ids)
        loss = -torch.mean(scores)
        loss.backward()

        layer_norms = {}
        for param_name, param in G_raw.named_parameters():
            if "weight" in param_name and param.grad is not None:
                layer_norms[param_name] = param.grad.norm().item()
        grad_norms[name] = layer_norms

    # Save to disk
    out_path = os.path.join(cfg["checkpoint_dir"], f"grad_diagnostics_epoch_{epoch:04d}.pt")
    torch.save({"epoch": epoch, "grad_norms": grad_norms}, out_path)
    print(f"  📊 Gradient diagnostics recorded at epoch {epoch} → {out_path}")

    # Inspect Channel 0 output conv gradient
    ch0_grad = 0.0
    if hasattr(G_raw, "out_conv") and G_raw.out_conv.weight.grad is not None:
        ch0_grad = G_raw.out_conv.weight.grad[0].norm().item()
    print(f"     [Channel 0 Logit Gradient] Children: {ch0_grad:.8f} (Expected Dead = 0.00000000)")

    if writer:
        for scen, layers in grad_norms.items():
            for layer, val in layers.items():
                writer.add_scalar(f"GradNorm_{scen}/{layer}", val, epoch)

    G_raw.train()
    D_raw.train()


# ─────────────────────────────────────────────────────────────
# 5. MAIN TRAINING LOOP (DDP + Single-GPU Compatible)
# ─────────────────────────────────────────────────────────────


def train(cfg):
    # DDP Initialization (for Kaggle dual GPU via torchrun)
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_main_process = (local_rank == 0)

    # Set random seeds for reproducibility
    seed = cfg.get("seed", 42) + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if is_main_process:
        print("=" * 80)
        print("🚀 TUS-GAN v4.1 (Cell C: CNN + Deterministic Hard Masking) — Training")
        print("=" * 80)
        print(f"🖥️ Execution Environment:")
        print(f"   Device         : {device}")
        print(f"   Distributed    : {is_distributed} (World Size: {world_size})")
        if torch.cuda.is_available():
            print(f"   GPU Count      : {torch.cuda.device_count()}")
            print(f"   Current GPU    : {torch.cuda.get_device_name(device)}")
            if torch.cuda.device_count() > 1 and not is_distributed:
                print("   💡 Note: 2 GPUs detected! To utilize both GPUs via DDP, run with:")
                print("      torchrun --nproc_per_node=2 train.py")
        print(f"   Epochs         : {cfg['epochs']}")
        print(f"   Batch Size     : {cfg['batch_size']} (Total across all GPUs)")
        print(f"   Seed           : {cfg.get('seed', 42)}")
        print("=" * 80)

        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
        os.makedirs(cfg["sample_dir"], exist_ok=True)
        os.makedirs(cfg["log_dir"], exist_ok=True)

    # Load dataset
    dataset = TUSDataset(
        cfg["data_path"],
        subset_size=cfg.get("subset"),
        split=cfg.get("split", "train"),
        split_indices=cfg.get("split_indices", None),
    )
    cfg["cond_dim"] = dataset.cond_dim
    cfg["num_districts"] = dataset.num_districts
    cfg["num_states"] = dataset.num_states
    cfg["num_channels"] = dataset.num_channels

    if is_main_process:
        print(f"✅ Dataset loaded: {len(dataset):,} diaries{dataset.split_info} (Cond Dim: {cfg['cond_dim']})")
        print("Computing real time-slice transition matrices...")

    # Real time-slice transition matrices
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

    # DataLoader & DistributedSampler setup
    batch_size_per_gpu = max(1, cfg["batch_size"] // world_size)
    num_workers = min(4, os.cpu_count() or 2)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if is_distributed else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model instantiation
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
    ).to(device)

    C = DemographicClassifier(cond_dim=cfg["cond_dim"], base_channels=32).to(device)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    opt_C = torch.optim.Adam(C.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=cfg["epochs"], eta_min=1e-6)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=cfg["epochs"], eta_min=1e-6)
    sched_C = torch.optim.lr_scheduler.CosineAnnealingLR(opt_C, T_max=cfg["epochs"], eta_min=1e-6)

    ema = EMA(G, decay=cfg["ema_decay"]) if is_main_process else None
    criterion_C = nn.BCEWithLogitsLoss()

    start_epoch = 1
    if cfg["resume"]:
        start_epoch = load_checkpoint(cfg["resume"], G, D, C, opt_G, opt_D, opt_C, device)

    # Wrap in DDP if multi-GPU
    if is_distributed:
        G = nn.parallel.DistributedDataParallel(G, device_ids=[local_rank], broadcast_buffers=False)
        D = nn.parallel.DistributedDataParallel(D, device_ids=[local_rank], broadcast_buffers=False)
        C = nn.parallel.DistributedDataParallel(C, device_ids=[local_rank], broadcast_buffers=False)

    writer = SummaryWriter(cfg["log_dir"]) if is_main_process else None

    if is_main_process:
        print(f"\n▶️ Starting Cell C Training Loop: Epoch {start_epoch} to {cfg['epochs']}")
        print(f"   Batch per GPU: {batch_size_per_gpu} | Batches per Epoch: {len(loader)}\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        current_temp = max(
            cfg["gumbel_temp_min"],
            cfg["gumbel_temp_start"] * np.exp(-cfg["gumbel_temp_decay"] * (epoch - 1)),
        )

        epoch_loss_D, epoch_loss_G, epoch_loss_trans, epoch_loss_C, epoch_gp, epoch_w_dist = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_critic_steps, n_gen_steps = 0, 0

        data_iter = iter(loader)
        n_gen_batches = max(1, len(loader) // cfg["n_critic"])

        for gen_step in range(n_gen_batches):
            # ─────────────────────────────────────────────────────────
            # STEP A: Train Critic & AC-GAN Classifier
            # ─────────────────────────────────────────────────────────
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

                real_scores = D(real_diaries, cond_vec, dist_ids, state_ids)
                fake_scores = D(fake_diaries, cond_vec, dist_ids, state_ids)

                # For WGAN-GP in DDP, pass the raw module to compute_gradient_penalty
                D_raw = D.module if hasattr(D, "module") else D
                gp = compute_gradient_penalty(
                    D_raw, real_diaries, fake_diaries, cond_vec, dist_ids, state_ids, device, cfg["lambda_gp"]
                )
                loss_D = critic_loss(real_scores, fake_scores, gp)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                # Demographic Classifier update
                pred_c = C(real_diaries)
                loss_C = criterion_C(pred_c, cond_vec)
                opt_C.zero_grad()
                loss_C.backward()
                opt_C.step()

                epoch_w_dist += (real_scores.mean() - fake_scores.mean()).item()
                epoch_loss_D += loss_D.item()
                epoch_loss_C += loss_C.item()
                epoch_gp += gp.item()
                n_critic_steps += 1

            # ─────────────────────────────────────────────────────────
            # STEP B: Train Generator (with Hard Logit Masking)
            # ─────────────────────────────────────────────────────────
            z_g = torch.randn(B, cfg["noise_dim"], device=device)
            fake_diaries = G(z_g, cond_vec, dist_ids, state_ids, temp=current_temp, hard=True)
            fake_scores = D(fake_diaries, cond_vec, dist_ids, state_ids)

            # Time-slice transition loss
            probs_fake = (fake_diaries.squeeze(-1) + 1.0) / 2.0
            loss_trans_slices = []
            for k in range(4):
                start, end = k * 12, (k + 1) * 12
                x_t_f = probs_fake[..., start : end - 1]
                x_tp1_f = probs_fake[..., start + 1 : end]
                trans_fake = torch.einsum("b i t, b j t -> i j", x_t_f, x_tp1_f)
                row_sums_f = trans_fake.sum(dim=1, keepdim=True)
                P_fake_k = trans_fake / (row_sums_f + 1e-8)
                loss_trans_slices.append(torch.mean((P_fake_k - P_real_slices[k]) ** 2))
            loss_trans = torch.mean(torch.stack(loss_trans_slices))

            pred_c_fake = C(fake_diaries)
            loss_cond = criterion_C(pred_c_fake, cond_vec)

            # Generator objective (No soft penalty in Cell C: tests pure hard masking)
            loss_G = generator_loss(fake_scores) + cfg["lambda_transition"] * loss_trans + cfg["lambda_cond"] * loss_cond

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if ema is not None:
                G_raw = G.module if hasattr(G, "module") else G
                ema.model = G_raw
                ema.update()

            epoch_loss_G += loss_G.item()
            epoch_loss_trans += loss_trans.item()
            n_gen_steps += 1

        sched_G.step()
        sched_D.step()
        sched_C.step()

        # Master Process: Logging & Diagnostics
        if is_main_process:
            avg_loss_D = epoch_loss_D / max(1, n_critic_steps)
            avg_loss_G = epoch_loss_G / max(1, n_gen_steps)
            avg_w = epoch_w_dist / max(1, n_critic_steps)

            print(f"Epoch [{epoch:>4}/{cfg['epochs']}] | W-dist: {avg_w:+.4f} | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f} | Temp: {current_temp:.3f}")

            if writer:
                writer.add_scalar("Loss/Critic", avg_loss_D, epoch)
                writer.add_scalar("Loss/Generator", avg_loss_G, epoch)
                writer.add_scalar("Metric/Wasserstein_Distance", avg_w, epoch)
                writer.add_scalar("Gumbel/Temperature", current_temp, epoch)

            # Gradient Diagnostic Logging (Epochs 10, 50, 100, 250)
            if epoch in cfg["grad_diag_epochs"]:
                log_gradient_flow_diagnostics(G, D, device, epoch, cfg, writer)

            # Periodic Checkpoint Saving
            if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
                ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt")
                save_checkpoint(ckpt_path, epoch, G, D, C, opt_G, opt_D, opt_C, ema, cfg)

    # Save Final Checkpoint
    if is_main_process:
        final_path = os.path.join(cfg["checkpoint_dir"], "final.pt")
        save_checkpoint(final_path, cfg["epochs"], G, D, C, opt_G, opt_D, opt_C, ema, cfg)
        print(f"\n🎉 [v4.1] Cell C Training Complete. Final saved → {final_path}")

    if is_distributed:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────
# 6. ENTRYPOINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TUS-GAN v4.1 (Cell C: Conv + Hard Masking)")
    parser.add_argument("--data", type=str, default="tusgan_encode.npz", help="Path to npz dataset")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for quick debug")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--sample_dir", type=str, default="samples", help="Directory to save samples")
    parser.add_argument("--log_dir", type=str, default="runs/tusgan_v4_1_hard", help="TensorBoard log directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Dataset split: 'train' (90%%, 400,741 records), 'test' (10%%, 44,527 records), or 'all'",
    )
    parser.add_argument(
        "--split_indices",
        type=str,
        default=None,
        help="Explicit path to split indices .npy file",
    )
    args = parser.parse_args()

    cfg = get_config()
    cfg["data_path"] = args.data
    cfg["split"] = args.split
    cfg["split_indices"] = args.split_indices
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch
    cfg["subset"] = args.subset
    cfg["seed"] = args.seed
    cfg["checkpoint_dir"] = args.checkpoint_dir
    cfg["sample_dir"] = args.sample_dir
    cfg["log_dir"] = args.log_dir
    cfg["resume"] = args.resume

    train(cfg)
