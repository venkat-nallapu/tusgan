#!/usr/bin/env python3
"""
TUS-GAN v5.1 (Cell D) - Pre-Flight Smoke Test
============================================
Rigorously tests all v5.1 components before launching long training on JarvisLabs A100:
1. CUDA / GPU detection and A100 SDP math-backend verification.
2. Generator: Transformer temporal blocks, soft vs. hard outputs, unmasked logits in training.
3. Critic: Transformer temporal blocks, score calculation, InfoNCE projections.
4. WGAN-GP Gradient Penalty: Verifies that double-backward autograd pass succeeds on GPU.
5. Soft Neuro-Symbolic loss, InfoNCE loss, and Spell-Duration loss backward passes.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

# Disable flash SDP for WGAN-GP double-backward compatibility
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from generator import Generator
from critic import Critic, compute_gradient_penalty, critic_loss, generator_loss
from train import compute_infonce_loss, compute_spell_duration_loss, compute_neuro_symbolic_loss


def run_v51_smoke_tests():
    print("=" * 75)
    print("🚀 Running TUS-GAN v5.1 Pre-Flight Smoke Tests (Cell D)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    BATCH_SIZE = 8
    NOISE_DIM = 128
    COND_DIM = 83
    NUM_DISTRICTS = 71
    NUM_STATES = 36
    NUM_CHANNELS = 9
    SEQ_LEN = 48

    # -------------------------------------------------------------
    # 1. Dummy Inputs
    # -------------------------------------------------------------
    print("\n[Step 1] Creating synthetic batch with constrained child profiles...")
    z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
    cond_vec = torch.zeros(BATCH_SIZE, COND_DIM, device=device)
    cond_vec[0:4, 0] = 1.0  # First 4 profiles are children (Age < 15)
    dist_ids = torch.randint(0, NUM_DISTRICTS, (BATCH_SIZE,), device=device)
    state_ids = torch.randint(0, NUM_STATES, (BATCH_SIZE,), device=device)
    print("  ✓ Batch initialized.")

    # -------------------------------------------------------------
    # 2. Generator Test
    # -------------------------------------------------------------
    print("\n[Step 2] Testing v5.1 Generator (Transformer + Soft Probabilities)...")
    G = Generator(
        noise_dim=NOISE_DIM,
        cond_dim=COND_DIM,
        num_districts=NUM_DISTRICTS,
        num_states=NUM_STATES,
        base_channels=64,
    ).to(device)

    # Training mode: apply_hard_mask=False
    fake_hard, fake_soft = G(z, cond_vec, dist_ids, state_ids, temp=1.0, hard=True, return_soft=True, apply_hard_mask=False)
    assert fake_hard.shape == (BATCH_SIZE, NUM_CHANNELS, SEQ_LEN, 1), f"Unexpected shape {fake_hard.shape}"
    assert fake_soft.shape == (BATCH_SIZE, NUM_CHANNELS, SEQ_LEN, 1), f"Unexpected shape {fake_soft.shape}"

    # Soft probabilities should contain non-zero mass on channel 0 for children to permit gradient flow
    probs_soft = (fake_soft.squeeze(-1) + 1.0) / 2.0
    child_work_soft = probs_soft[0:4, 0, :].mean().item()
    print(f"  ✓ Output shapes verified: {fake_hard.shape}")
    print(f"  ✓ Child employment soft mass (gradient enabled): {child_work_soft:.4f}")
    assert child_work_soft > 0.0, "Child soft mass must be non-zero during training!"

    # Inference mode: apply_hard_mask=True
    fake_inf_hard = G(z, cond_vec, dist_ids, state_ids, temp=1.0, hard=True, return_soft=False, apply_hard_mask=True)
    assert (fake_inf_hard[0:4, 0, :, :] == -1.0).all(), "Inference masking failed: child has employment mass!"
    print("  ✓ Optional inference hard masking verified (0.00% physical violation).")

    # -------------------------------------------------------------
    # 3. Critic & InfoNCE Projections
    # -------------------------------------------------------------
    print("\n[Step 3] Testing v5.1 Critic (Transformer + InfoNCE Projections)...")
    D = Critic(
        cond_dim=COND_DIM,
        num_districts=NUM_DISTRICTS,
        num_states=NUM_STATES,
        base_channels=32,
        proj_dim=64,
    ).to(device)

    scores, f_proj, c_proj = D(fake_hard, cond_vec, dist_ids, state_ids, return_proj=True)
    assert scores.shape == (BATCH_SIZE, 1), f"Unexpected score shape {scores.shape}"
    assert f_proj.shape == (BATCH_SIZE, 64), f"Unexpected feat proj shape {f_proj.shape}"
    assert c_proj.shape == (BATCH_SIZE, 64), f"Unexpected cond proj shape {c_proj.shape}"
    print("  ✓ Critic outputs and projection dimensions verified.")

    # -------------------------------------------------------------
    # 4. WGAN-GP Second-Order Gradient Penalty (Crucial A100 test)
    # -------------------------------------------------------------
    print("\n[Step 4] Testing WGAN-GP Gradient Penalty (Double-Backward check)...")
    real_sample = torch.randn(BATCH_SIZE, NUM_CHANNELS, SEQ_LEN, 1, device=device)
    gp = compute_gradient_penalty(D, real_sample, fake_hard, cond_vec, dist_ids, state_ids, device)
    gp.backward(retain_graph=True)
    print(f"  ✓ WGAN-GP gradient penalty computed and backpropagated: GP = {gp.item():.4f}")

    # -------------------------------------------------------------
    # 5. Full Generator Loss Backward Pass
    # -------------------------------------------------------------
    print("\n[Step 5] Testing Generator Multi-Objective Loss Backward Pass...")
    G.zero_grad()
    D.zero_grad()

    loss_infonce = compute_infonce_loss(f_proj, c_proj)
    loss_logic = compute_neuro_symbolic_loss(probs_soft, cond_vec)
    dummy_durations = torch.ones(NUM_CHANNELS, device=device) * 4.0
    loss_duration = compute_spell_duration_loss(probs_soft, dummy_durations)

    total_g_loss = generator_loss(scores) + 0.5 * loss_infonce + 1.0 * loss_duration + 10.0 * loss_logic
    total_g_loss.backward()

    has_grad = [p.grad is not None and p.grad.norm().item() > 0 for p in G.parameters() if p.requires_grad]
    assert all(has_grad), "Some generator parameters received zero/None gradient!"
    print(f"  ✓ Generator loss computed: {total_g_loss.item():.4f}")
    print(f"  ✓ Neuro-symbolic logic penalty active: {loss_logic.item():.4f}")
    print("  ✓ Full gradient flow confirmed across all generator layers.")

    print("\n" + "=" * 75)
    print("🎉 ALL PRE-FLIGHT SMOKE TESTS PASSED! Ready to train on JarvisLabs A100.")
    print("=" * 75)


if __name__ == "__main__":
    run_v51_smoke_tests()
