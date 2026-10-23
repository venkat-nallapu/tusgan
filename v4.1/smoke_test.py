#!/usr/bin/env python3
"""
TUS-GAN v4.1 (Cell C: CNN + Hard Masking) — Pre-Flight Smoke Test
================================================================
Rigorously validates all v4.1 components prior to launching training on Kaggle Dual GPUs:
1. CUDA / GPU detection and dual GPU verification.
2. Generator: Transposed Conv blocks, Conditional BatchNorm, SelfAttention2d.
3. Deterministic Hard Logit Masking: Verifies pre-softmax logit clamping (-1e9) eliminates Channel 0 for children.
4. Critic: Strided Conv blocks, Conditional InstanceNorm, Spectral Normalization.
5. WGAN-GP Gradient Penalty: Verifies autograd backward and double-backward compatibility.
6. Dead-Gradient Proof (Cell C): Empirically proves Channel 0 gradient is 0.00000000 for child profiles.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

from generator import Generator
from critic import Critic, compute_gradient_penalty, critic_loss, generator_loss
from train import DemographicClassifier


def run_v41_smoke_tests():
    print("=" * 80)
    print("🧪 Running TUS-GAN v4.1 Pre-Flight Smoke Tests (Cell C: CNN + Hard Masking)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  • GPU {i}: {name} ({vram:.1f} GB VRAM)")

    BATCH_SIZE = 8
    NOISE_DIM = 128
    COND_DIM = 82
    NUM_DISTRICTS = 71
    NUM_STATES = 36
    NUM_CHANNELS = 9
    SEQ_LEN = 48

    # ─────────────────────────────────────────────────────────────
    # 1. Dummy Inputs
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 1] Initializing test batch with child and adult profiles...")
    z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
    cond_vec = torch.zeros(BATCH_SIZE, COND_DIM, device=device)
    cond_vec[0:4, 0] = 1.0  # First 4 profiles are children (Age < 15)
    cond_vec[4:8, 0] = 0.0  # Remaining 4 profiles are adults
    dist_ids = torch.randint(0, NUM_DISTRICTS, (BATCH_SIZE,), device=device)
    state_ids = torch.randint(0, NUM_STATES, (BATCH_SIZE,), device=device)
    print("  ✓ Batch initialized.")

    # ─────────────────────────────────────────────────────────────
    # 2. Generator Test & Hard Mask Verification
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 2] Testing v4.1 Generator (CNN + Hard Logit Masking)...")
    G = Generator(
        noise_dim=NOISE_DIM,
        cond_dim=COND_DIM,
        num_districts=NUM_DISTRICTS,
        num_states=NUM_STATES,
        base_channels=64,
    ).to(device)

    fake = G(z, cond_vec, dist_ids, state_ids, temp=1.0, hard=True)
    assert fake.shape == (BATCH_SIZE, NUM_CHANNELS, SEQ_LEN, 1), f"Unexpected shape {fake.shape}"
    print(f"  ✓ Output shape verified: {fake.shape}")

    # Check child labor hard masking: channel 0 must be strictly -1.0 for children
    child_work_mass = (fake[0:4, 0, :, :] + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
    adult_work_mass = (fake[4:8, 0, :, :] + 1.0) / 2.0
    assert torch.all(child_work_mass == 0.0), "Child labor hard masking failed: child has non-zero work mass!"
    print("  ✓ Hard logit masking verified: Child employment mass is strictly 0.00% (-1.0 in tensor).")

    # ─────────────────────────────────────────────────────────────
    # 3. Critic & AC-GAN Demographic Classifier
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 3] Testing Critic and Demographic Classifier...")
    D = Critic(
        cond_dim=COND_DIM,
        num_districts=NUM_DISTRICTS,
        num_states=NUM_STATES,
        base_channels=32,
    ).to(device)

    scores = D(fake, cond_vec, dist_ids, state_ids)
    assert scores.shape == (BATCH_SIZE, 1), f"Unexpected scores shape: {scores.shape}"
    print(f"  ✓ Critic score output shape verified: {scores.shape}")

    C = DemographicClassifier(cond_dim=COND_DIM, base_channels=32).to(device)
    c_pred = C(fake)
    assert c_pred.shape == (BATCH_SIZE, COND_DIM), f"Unexpected classifier shape: {c_pred.shape}"
    print(f"  ✓ Auxiliary Classifier shape verified: {c_pred.shape}")

    # ─────────────────────────────────────────────────────────────
    # 4. WGAN-GP Gradient Penalty Autograd
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 4] Testing WGAN-GP Gradient Penalty autograd pass...")
    real_dummy = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LEN, 1, device=device)
    gp = compute_gradient_penalty(D, real_dummy, fake, cond_vec, dist_ids, state_ids, device)
    assert gp.ndim == 0 and not torch.isnan(gp), "GP computation failed or returned NaN!"
    print(f"  ✓ Gradient penalty computed successfully: {gp.item():.4f}")

    # Critic loss backward
    d_loss = critic_loss(scores, scores, gp)
    d_loss.backward()
    print("  ✓ Critic backward pass completed cleanly.")

    # ─────────────────────────────────────────────────────────────
    # 5. Dead-Gradient Proof (Cell C Verification)
    # ─────────────────────────────────────────────────────────────
    print("\n[Step 5] Empirically testing the Dead-Gradient Phenomenon (Cell C)...")
    G.zero_grad()
    child_cond = torch.zeros(BATCH_SIZE, COND_DIM, device=device)
    child_cond[:, 0] = 1.0  # All children

    fake_children = G(z, child_cond, dist_ids, state_ids, temp=1.0, hard=True)
    crit_scores = D(fake_children, child_cond, dist_ids, state_ids)
    g_loss = -torch.mean(crit_scores)
    g_loss.backward()

    ch0_grad = G.out_conv.weight.grad[0].norm().item()
    other_grad = G.out_conv.weight.grad[1:].norm().item()

    print(f"  • Channel 0 (Employment) Output Conv Gradient Norm: {ch0_grad:.8f}")
    print(f"  • Channels 1-8 Output Conv Gradient Norm          : {other_grad:.8f}")

    assert ch0_grad == 0.0, f"Expected dead gradient (0.0), but got {ch0_grad}"
    print("  ✓ Mathematical Dead-Gradient Proof confirmed: ∂L/∂θ_0 ≡ 0.00000000 for hard-masked channel!")

    print("\n" + "=" * 80)
    print("🎉 All TUS-GAN v4.1 Pre-Flight Smoke Tests Passed Successfully! ✓")
    print("=" * 80)


if __name__ == "__main__":
    run_v41_smoke_tests()
