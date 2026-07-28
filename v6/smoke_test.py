"""
TUS-GAN v5 - Advanced Smoke Tests
=================================
This script rigorously tests the new v5 architectural components:
1. Transformer temporal blocks (shape and gradient flow).
2. Deterministic Child Labor Masking (Employment logit forced to -inf).
3. InfoNCE Contrastive Loss (Proper calculation and backward pass).
4. Differentiable Spell-Duration Loss (Proper calculation).

Run this script directly to verify all v5 components before full training.
"""

import torch
import torch.nn.functional as F
import numpy as np

from generator import Generator
from critic import Critic
from train import compute_infonce_loss, compute_spell_duration_loss


def run_advanced_smoke_tests():
    print("🚀 Starting TUS-GAN v5 Advanced Smoke Tests...\n")

    # ---------------------------------------------------------
    # 1. Initialization and Parameter Configuration
    # ---------------------------------------------------------
    print("Step 1: Initializing configuration and dummy data.")
    BATCH_SIZE = 8
    NOISE_DIM = 128
    COND_DIM = 83
    NUM_DISTRICTS = 71
    NUM_STATES = 36
    NUM_CHANNELS = 9
    SEQ_LEN = 48

    device = torch.device("cpu")

    # Dummy inputs
    z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
    cond_vec = torch.zeros(BATCH_SIZE, COND_DIM, device=device)

    # Trigger Child Labor constraint on the first 2 instances
    # Age < 15 is index 0 in the cond_vec.
    cond_vec[0:2, 0] = 1.0

    dist_ids = torch.randint(0, NUM_DISTRICTS, (BATCH_SIZE,), device=device)
    state_ids = torch.randint(0, NUM_STATES, (BATCH_SIZE,), device=device)

    print("  ✓ Dummy data created.")

    # ---------------------------------------------------------
    # 2. Generator Test (Transformers & Deterministic Masking)
    # ---------------------------------------------------------
    print("\nStep 2: Testing Generator (Transformer Block & Masking)...")

    G = Generator(
        noise_dim=NOISE_DIM,
        cond_dim=COND_DIM,
        num_districts=NUM_DISTRICTS,
        num_states=NUM_STATES,
        base_channels=64,  # Reduced for fast smoke test
    ).to(device)

    # Forward pass
    fake_hard, fake_soft = G(z, cond_vec, dist_ids, state_ids, temp=1.0, hard=True, return_soft=True)

    # Check shape
    expected_shape = (BATCH_SIZE, NUM_CHANNELS, SEQ_LEN, 1)
    assert (
        fake_hard.shape == expected_shape
    ), f"Expected {expected_shape}, got {fake_hard.shape}"
    assert (
        fake_soft.shape == expected_shape
    ), f"Expected {expected_shape}, got {fake_soft.shape}"
    
    # Soft outputs should NOT be strictly -1.0 or 1.0 everywhere
    assert not (fake_soft == -1.0).all() and not (fake_soft == 1.0).all(), "Soft outputs are hard!"
    print(f"  ✓ Output shapes and softness verified.")

    # The first two individuals are children, so their Employment channel (index 0) must be exactly -1.0
    # (Since gumbel_softmax scales [0,1] outputs to [-1, 1], and the masked logit is -1e9 -> prob 0 -> output -1.0)
    children_employment = fake_hard[0:2, 0, :, :]
    assert (
        children_employment == -1.0
    ).all(), "Masking failed: Child generated employment activity!"
    print("  ✓ Deterministic Child Labor logit masking passed.")

    # ---------------------------------------------------------
    # 3. Critic Test (Transformers & InfoNCE Projection)
    # ---------------------------------------------------------
    print("\nStep 3: Testing Critic (Transformer Block & InfoNCE Head)...")

    D = Critic(
        cond_dim=COND_DIM,
        num_districts=NUM_DISTRICTS,
        num_states=NUM_STATES,
        base_channels=32,
        proj_dim=64,
    ).to(device)

    # Forward pass returning projection features for InfoNCE
    scores, f_proj, c_proj = D(fake_hard, cond_vec, dist_ids, state_ids, return_proj=True)

    assert scores.shape == (BATCH_SIZE, 1), f"Expected scores shape (8, 1), got {scores.shape}"
    assert f_proj.shape == (
        BATCH_SIZE,
        64,
    ), f"Expected feature projection shape (8, 64), got {f_proj.shape}"
    assert c_proj.shape == (
        BATCH_SIZE,
        64,
    ), f"Expected condition projection shape (8, 64), got {c_proj.shape}"

    # Verify L2 Normalization of projections
    f_norm = torch.norm(f_proj, p=2, dim=1)
    c_norm = torch.norm(c_proj, p=2, dim=1)
    assert torch.allclose(
        f_norm, torch.ones_like(f_norm)
    ), "Feature projection is not L2 normalized!"
    assert torch.allclose(
        c_norm, torch.ones_like(c_norm)
    ), "Condition projection is not L2 normalized!"
    print("  ✓ Critic output shapes and L2 normalization passed.")

    # ---------------------------------------------------------
    # 4. InfoNCE Contrastive Loss Test
    # ---------------------------------------------------------
    print("\nStep 4: Testing InfoNCE Contrastive Loss Calculation...")

    loss_infonce = compute_infonce_loss(f_proj, c_proj, temperature=0.1)
    assert (
        not torch.isnan(loss_infonce) and loss_infonce.item() > 0
    ), "InfoNCE loss is NaN or invalid."
    print(f"  ✓ InfoNCE Loss calculated successfully: {loss_infonce.item():.4f}")

    # ---------------------------------------------------------
    # 5. Differentiable Spell-Duration Loss Test
    # ---------------------------------------------------------
    print("\nStep 5: Testing Differentiable Spell-Duration Loss...")

    probs_fake = (fake_soft.squeeze(-1) + 1.0) / 2.0  # Scale back to [0, 1]

    # Dummy real average durations for 9 channels
    real_avg_durations = torch.tensor(
        [12.5, 4.2, 5.5, 3.1, 1.2, 8.4, 4.0, 6.7, 16.0], device=device
    )

    loss_duration = compute_spell_duration_loss(probs_fake, real_avg_durations)
    assert (
        not torch.isnan(loss_duration) and loss_duration.item() >= 0
    ), "Duration loss is NaN or invalid."
    print(f"  ✓ Spell-Duration Loss calculated successfully: {loss_duration.item():.4f}")

    # ---------------------------------------------------------
    # 6. Gradient Flow Test
    # ---------------------------------------------------------
    print("\nStep 6: Testing Gradient Flow End-to-End...")

    # Optimize dummy loss
    total_loss = loss_infonce + loss_duration + scores.mean()
    total_loss.backward()

    # Check if generator weights received gradients
    has_grads = any(p.grad is not None for p in G.parameters())
    assert has_grads, "Generator did not receive gradients!"

    has_grads_d = any(p.grad is not None for p in D.parameters())
    assert has_grads_d, "Critic did not receive gradients!"
    print("  ✓ Backpropagation successful, gradients are flowing.")

    print("\n🎉 ALL ADVANCED SMOKE TESTS PASSED SUCCESSFULLY! 🎉")


if __name__ == "__main__":
    # Disable optimized SDP attention backends because they do not support
    # double backward passes required for WGAN-GP's gradient penalty.
    # We force the math backend instead.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    run_advanced_smoke_tests()
