#!/usr/bin/env python3
"""
TUS-GAN — Advanced Gradient Flow Analysis (Corrected)
=====================================================
Performs rigorous per-layer and per-CHANNEL gradient norm measurement
across all 4 cells of the 2x2 factorial experimental matrix.

CORRECTED: Respects each version's actual forward() API signatures:
  - v4  Generator: forward(z, cond, dist, state, temp, hard)         — NO return_soft
  - v4  Critic:    forward(diary, cond, dist, state)                  — NO return_proj
  - v4.1 Generator: forward(z, cond, dist, state, temp, hard, return_soft)
  - v4.1 Critic:    forward(diary, cond, dist, state)                 — NO return_proj
  - v5  Generator: forward(z, cond, dist, state, temp, hard, return_soft)
  - v5  Critic:    forward(diary, cond, dist, state, return_proj)
  - v5.1 Generator: forward(z, cond, dist, state, temp, hard, return_soft, apply_hard_mask)
  - v5.1 Critic:    forward(diary, cond, dist, state, return_proj)

Produces:
  1. Per-layer L2 gradient norm table (Adult vs Child)
  2. Per-CHANNEL output-layer gradient norms (the critical diagnostic)
  3. Publication-quality 4-panel comparison plot (Figure 8)
  4. Saved .pt diagnostics file for reproducibility
"""

import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
NOISE_DIM = 128
COND_DIM = 83
BASE_CH_G = 64
BASE_CH_D = 32
WARMUP_STEPS = 10
LAMBDA_LOGIC = 10.0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Batch: {BATCH_SIZE}, Warmup Steps: {WARMUP_STEPS}")

# ─── Module Loading ──────────────────────────────────────────
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v4_gen_mod   = load_mod("v4_gen",   "v4/generator.py")
v4_crit_mod  = load_mod("v4_crit",  "v4/critic.py")
v41_gen_mod  = load_mod("v41_gen",  "v4.1/generator.py")
v41_crit_mod = load_mod("v41_crit", "v4.1/critic.py")
v5_gen_mod   = load_mod("v5_gen",   "v5/generator.py")
v5_crit_mod  = load_mod("v5_crit",  "v5/critic.py")
v51_gen_mod  = load_mod("v51_gen",  "v5.1/generator.py")
v51_crit_mod = load_mod("v51_crit", "v5.1/critic.py")


# ─── Loss Helpers ────────────────────────────────────────────
def soft_penalty_loss(probs, cond):
    """Differentiable soft penalty: mean probability of Employment (ch 0) for children."""
    child_mask = (cond[:, 0] == 1)
    if child_mask.any():
        return probs[child_mask, 0, :].mean()
    return torch.tensor(0.0, device=probs.device)


# ─── Cell Definitions ────────────────────────────────────────
# Each cell specifies how to call its G and D correctly.

class CellConfig:
    def __init__(self, name, label, backbone, enforcement, G, D):
        self.name = name
        self.label = label
        self.backbone = backbone
        self.enforcement = enforcement
        self.G = G
        self.D = D


def build_cells():
    cells = []

    # Cell A (v4): Conv + Soft Penalty
    G_a = v4_gen_mod.Generator(base_channels=BASE_CH_G).to(DEVICE)
    D_a = v4_crit_mod.Critic(base_channels=BASE_CH_D).to(DEVICE)
    cells.append(CellConfig("Cell_A_v4", "(a) Cell A: Conv + Soft Penalty [v4]",
                            "Convolutional", "Soft Penalty", G_a, D_a))

    # Cell B (v5): Transformer + Hard Masking
    G_b = v5_gen_mod.Generator(base_channels=BASE_CH_G).to(DEVICE)
    D_b = v5_crit_mod.Critic(base_channels=BASE_CH_D).to(DEVICE)
    cells.append(CellConfig("Cell_B_v5", "(b) Cell B: Transformer + Hard Mask [v5]",
                            "Transformer", "Hard Masking", G_b, D_b))

    # Cell C (v4.1): Conv + Hard Masking
    G_c = v41_gen_mod.Generator(base_channels=BASE_CH_G).to(DEVICE)
    D_c = v41_crit_mod.Critic(base_channels=BASE_CH_D).to(DEVICE)
    cells.append(CellConfig("Cell_C_v41", "(c) Cell C: Conv + Hard Mask [v4.1]",
                            "Convolutional", "Hard Masking", G_c, D_c))

    # Cell D (v5.1): Transformer + Soft Penalty
    G_d = v51_gen_mod.Generator(base_channels=BASE_CH_G).to(DEVICE)
    D_d = v51_crit_mod.Critic(base_channels=BASE_CH_D).to(DEVICE)
    cells.append(CellConfig("Cell_D_v51", "(d) Cell D: Transformer + Soft Penalty [v5.1]",
                            "Transformer", "Soft Penalty", G_d, D_d))

    return cells


def make_batch(is_child_ratio=0.5):
    """Create a synthetic batch with specified child ratio."""
    z = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
    cond = torch.zeros(BATCH_SIZE, COND_DIM, device=DEVICE)
    n_child = int(BATCH_SIZE * is_child_ratio)
    cond[:n_child, 0] = 1.0  # Age < 15 flag
    dist = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    state = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
    real = torch.randn(BATCH_SIZE, 9, 48, 1, device=DEVICE)
    return z, cond, dist, state, real


# ─── Version-Aware Forward Passes ────────────────────────────
def gen_forward(cell, z, cond, dist, state, need_soft=False):
    """
    Call generator forward with correct API per version.
    Returns (hard_output, soft_output_or_None)
    """
    if cell.name == "Cell_A_v4":
        # v4 generator: NO return_soft parameter
        out = cell.G(z, cond, dist, state, temp=1.0, hard=True)
        return out, out  # Use same output as "soft" proxy
    elif cell.name == "Cell_C_v41":
        # v4.1 generator: HAS return_soft
        if need_soft:
            hard, soft = cell.G(z, cond, dist, state, temp=1.0, hard=True, return_soft=True)
            return hard, soft
        else:
            out = cell.G(z, cond, dist, state, temp=1.0, hard=True)
            return out, None
    elif cell.name == "Cell_B_v5":
        # v5 generator: HAS return_soft
        if need_soft:
            hard, soft = cell.G(z, cond, dist, state, temp=1.0, hard=True, return_soft=True)
            return hard, soft
        else:
            out = cell.G(z, cond, dist, state, temp=1.0, hard=True, return_soft=False)
            return out, None
    elif cell.name == "Cell_D_v51":
        # v5.1 generator: HAS return_soft + apply_hard_mask
        if need_soft:
            hard, soft = cell.G(z, cond, dist, state, temp=1.0, hard=True,
                                return_soft=True, apply_hard_mask=False)
            return hard, soft
        else:
            out = cell.G(z, cond, dist, state, temp=1.0, hard=True,
                         return_soft=False, apply_hard_mask=False)
            return out, None
    raise ValueError(f"Unknown cell: {cell.name}")


def critic_forward(cell, diary, cond, dist, state):
    """Call critic forward with correct API per version. Returns score tensor."""
    if cell.name in ("Cell_A_v4", "Cell_C_v41"):
        # v4 and v4.1 critics: NO return_proj
        return cell.D(diary, cond, dist, state)
    else:
        # v5 and v5.1 critics: HAS return_proj
        return cell.D(diary, cond, dist, state, return_proj=False)


def compute_g_loss(cell, fake_hard, fake_soft, scores, cond):
    """Compute generator loss matching the cell's enforcement method."""
    loss = -scores.mean()
    if cell.enforcement == "Soft Penalty":
        probs = (fake_soft.squeeze(-1) + 1.0) / 2.0
        loss = loss + LAMBDA_LOGIC * soft_penalty_loss(probs, cond)
    return loss


# ─── Warmup Training ─────────────────────────────────────────
def warmup_train(cell, num_steps=WARMUP_STEPS):
    """Run a few training steps to move weights off random initialization."""
    cell.G.train()
    cell.D.train()
    opt_G = torch.optim.Adam(cell.G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(cell.D.parameters(), lr=1e-4)

    for step in range(num_steps):
        z, cond, dist, state, real = make_batch(is_child_ratio=0.5)

        # --- Critic step ---
        opt_D.zero_grad()
        with torch.no_grad():
            fake_hard, _ = gen_forward(cell, z, cond, dist, state, need_soft=False)
        real_scores = critic_forward(cell, real, cond, dist, state)
        fake_scores = critic_forward(cell, fake_hard, cond, dist, state)
        loss_D = fake_scores.mean() - real_scores.mean()
        loss_D.backward()
        opt_D.step()

        # --- Generator step ---
        opt_G.zero_grad()
        z2 = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
        fake_hard, fake_soft = gen_forward(cell, z2, cond, dist, state, need_soft=True)
        scores = critic_forward(cell, fake_hard, cond, dist, state)
        loss_G = compute_g_loss(cell, fake_hard, fake_soft, scores, cond)
        loss_G.backward()
        opt_G.step()

    print(f"  Warmup complete ({num_steps} steps)")


# ─── Gradient Measurement ────────────────────────────────────
def measure_gradients(cell, is_child):
    """
    Measure per-layer AND per-channel-output-layer gradient norms for one profile type.
    """
    cell.G.eval()
    cell.D.eval()
    cell.G.zero_grad()

    z, cond, dist, state, _ = make_batch(is_child_ratio=0.0)
    # Override cond to be all-adult or all-child
    cond[:, 0] = 1.0 if is_child else 0.0

    fake_hard, fake_soft = gen_forward(cell, z, cond, dist, state, need_soft=True)
    scores = critic_forward(cell, fake_hard, cond, dist, state)
    loss = compute_g_loss(cell, fake_hard, fake_soft, scores, cond)
    loss.backward()

    # --- Per-layer gradient norms ---
    layer_norms = {}
    for name, param in cell.G.named_parameters():
        if param.grad is not None and "weight" in name:
            # Pick representative layers
            short = None
            if "fc" in name and "weight" in name:
                short = "fc_input"
            elif "up1" in name and "conv_t" in name:
                short = "up1_conv"
            elif "up2" in name and "conv_t" in name:
                short = "up2_conv"
            elif "out_conv" in name:
                short = "out_conv"
            elif "transformer1" in name and "linear1" in name and "layers.0" in name:
                short = "transf_L0"
            elif "transformer1" in name and "linear1" in name and "layers.1" in name:
                short = "transf_L1"
            elif "attn" in name and "value" in name:
                short = "attn_value"
            elif "attn" in name and "query" in name:
                short = "attn_query"

            if short and short not in layer_norms:
                layer_norms[short] = param.grad.norm().item()

    # --- Per-CHANNEL output gradient norms (THE critical diagnostic) ---
    # Find the out_conv layer and measure gradient per output channel
    channel_norms = {}
    for name, param in cell.G.named_parameters():
        if "out_conv" in name and "weight" in name and param.grad is not None:
            # out_conv weight shape: (out_channels, in_channels, kH, kW)
            # For 9-channel output, out_channels = 9
            for ch in range(min(9, param.grad.shape[0])):
                ch_grad = param.grad[ch]
                channel_norms[f"Ch{ch}"] = ch_grad.norm().item()
            break

    return layer_norms, channel_norms, loss.item()


# ─── Main Execution ──────────────────────────────────────────
print("\n" + "=" * 80)
print("   ADVANCED GRADIENT FLOW ANALYSIS — ALL 4 CELLS")
print("=" * 80)

cells = build_cells()
all_results = {}

for cell in cells:
    print(f"\n{'─'*60}")
    print(f"  {cell.label}")
    print(f"  Backbone: {cell.backbone} | Enforcement: {cell.enforcement}")
    print(f"{'─'*60}")

    warmup_train(cell)

    adult_layers, adult_channels, adult_loss = measure_gradients(cell, is_child=False)
    child_layers, child_channels, child_loss = measure_gradients(cell, is_child=True)

    all_results[cell.name] = {
        "label": cell.label,
        "backbone": cell.backbone,
        "enforcement": cell.enforcement,
        "adult_layers": adult_layers,
        "child_layers": child_layers,
        "adult_channels": adult_channels,
        "child_channels": child_channels,
        "adult_loss": adult_loss,
        "child_loss": child_loss,
    }

# ─── Print Tables ────────────────────────────────────────────
print("\n")
print("=" * 100)
print("  TABLE 1: PER-LAYER L2 GRADIENT NORMS (Adult vs Child)")
print("=" * 100)
print(f"{'Cell':<40} | {'Layer':<14} | {'Adult ||grad||':<14} | {'Child ||grad||':<14} | {'Ratio (C/A)':<12} | {'Status'}")
print("-" * 100)

for cname, data in all_results.items():
    for layer in data["adult_layers"]:
        a = data["adult_layers"][layer]
        c = data["child_layers"].get(layer, 0.0)
        ratio = c / max(a, 1e-15)
        if data["enforcement"] == "Soft Penalty":
            if ratio > 1.5:
                status = "BOOSTED (Soft Penalty active)"
            elif ratio > 0.3:
                status = "HEALTHY"
            else:
                status = "WEAK"
        else:
            if ratio > 0.8:
                status = "EQUAL (no extra signal)"
            elif ratio > 0.3:
                status = "REDUCED"
            else:
                status = "DEAD / NEAR-ZERO"
        print(f"{data['label']:<40} | {layer:<14} | {a:<14.6e} | {c:<14.6e} | {ratio:<12.2f} | {status}")
    print("-" * 100)

print("\n")
print("=" * 100)
print("  TABLE 2: PER-CHANNEL OUTPUT GRADIENT NORMS (THE CRITICAL DIAGNOSTIC)")
print("  Channel 0 = Employment (Forbidden for Children)")
print("=" * 100)
ACTIVITY_NAMES = {0: "Employment*", 1: "Sleep", 2: "Eating", 3: "Domestic", 4: "Travel",
                  5: "Leisure", 6: "Education", 7: "SelfCare", 8: "Other"}
print(f"{'Cell':<40} | {'Channel':<16} | {'Adult Grad':<14} | {'Child Grad':<14} | {'Ratio':<10} | {'Verdict'}")
print("-" * 100)

for cname, data in all_results.items():
    for ch_key in sorted(data["adult_channels"].keys()):
        ch_idx = int(ch_key.replace("Ch", ""))
        a = data["adult_channels"][ch_key]
        c = data["child_channels"].get(ch_key, 0.0)
        ratio = c / max(a, 1e-15)
        act_name = ACTIVITY_NAMES.get(ch_idx, f"Activity_{ch_idx}")

        if ch_idx == 0:
            if c < 1e-8 and data["enforcement"] == "Hard Masking":
                verdict = "DEAD (Jacobian collapse)"
            elif ratio > 1.0:
                verdict = "ACTIVE (Soft penalty drives learning)"
            else:
                verdict = f"ratio={ratio:.2f}"
        else:
            verdict = ""
        print(f"{data['label']:<40} | {ch_key} ({act_name:<11}) | {a:<14.6e} | {c:<14.6e} | {ratio:<10.2f} | {verdict}")
    print("-" * 100)

# ─── Save Diagnostic File ────────────────────────────────────
save_path = os.path.join(OUT_DIR, "gradient_diagnostics_advanced.pt")
torch.save(all_results, save_path)
print(f"\nDiagnostics saved to: {save_path}")

# ─── Publication Plot ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

panel_configs = [
    ("Cell_A_v4",  "(a) Cell A: Conv + Soft Penalty [v4]"),
    ("Cell_B_v5",  "(b) Cell B: Transformer + Hard Mask [v5]"),
    ("Cell_C_v41", "(c) Cell C: Conv + Hard Mask [v4.1]"),
    ("Cell_D_v51", "(d) Cell D: Transformer + Soft Penalty [v5.1]"),
]

for idx, (cname, title) in enumerate(panel_configs):
    ax = axes[idx // 2][idx % 2]
    data = all_results[cname]

    # Use per-CHANNEL output norms (the real diagnostic)
    channels = sorted(data["adult_channels"].keys())
    adult_vals = [data["adult_channels"][c] for c in channels]
    child_vals = [max(data["child_channels"].get(c, 1e-15), 1e-15) for c in channels]
    x_labels = [f"{c}\n({ACTIVITY_NAMES.get(int(c[2:]), '')})" for c in channels]

    x = np.arange(len(channels))
    w = 0.35

    ax.bar(x - w / 2, adult_vals, w, label="Adults (Unconstrained)", color="#1f77b4", alpha=0.9)
    if data["enforcement"] == "Soft Penalty":
        ax.bar(x + w / 2, child_vals, w, label="Children (Soft Penalty)", color="#2ca02c", alpha=0.9)
    else:
        ax.bar(x + w / 2, child_vals, w, label="Children (Hard Masked)", color="#d62728", alpha=0.9)

    # Highlight Channel 0
    rect = plt.Rectangle((x[0] - 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-16),
                          1.0, max(adult_vals + child_vals) * 2,
                          alpha=0.08, color="#FF6600", zorder=0)
    ax.add_patch(rect)

    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#003366")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("L2 Gradient Norm (Log Scale)" if idx % 2 == 0 else "", fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper right")

    # Annotate Channel 0
    ch0_child = data["child_channels"].get("Ch0", 1e-15)
    ch0_adult = data["adult_channels"].get("Ch0", 1e-15)
    if data["enforcement"] == "Hard Masking" and ch0_child < 1e-8:
        ax.annotate("Ch0 DEAD\n(grad = 0)", xy=(0.175, max(ch0_child, 1e-14)),
                    xytext=(2, ch0_child * 0.01 if ch0_child > 1e-12 else 1e-13),
                    fontsize=8, fontweight="bold", color="#B71C1C",
                    arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1.5, headwidth=5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#C62828"))
    elif data["enforcement"] == "Soft Penalty" and ch0_child > ch0_adult * 0.5:
        ax.annotate("Ch0 ACTIVE\n(learning rule)", xy=(0.175, ch0_child),
                    xytext=(2, ch0_child * 5),
                    fontsize=8, fontweight="bold", color="#1B5E20",
                    arrowprops=dict(facecolor="#2ca02c", shrink=0.05, width=1.5, headwidth=5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#E8F5E9", ec="#2E7D32"))

plt.suptitle("Figure 8: Per-Channel Output Gradient Norms Across 2x2 Factorial Design\n"
             "(Channel 0 = Employment, highlighted in orange = the constrained channel)",
             fontsize=13, fontweight="bold")
plt.tight_layout()

plot_png = os.path.join(OUT_DIR, "figure8_advanced_per_channel_gradient.png")
plot_pdf = os.path.join(OUT_DIR, "figure8_advanced_per_channel_gradient.pdf")
plt.savefig(plot_png, dpi=300)
plt.savefig(plot_pdf, format="pdf")
plt.close()
print(f"\nFigure 8 saved:")
print(f"  PNG: {plot_png}")
print(f"  PDF: {plot_pdf}")
print("\nDone.")
