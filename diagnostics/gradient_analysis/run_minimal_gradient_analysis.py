"""
TUS-GAN — Minimal Training & Gradient Flow Analysis (Cells A, B, C, D)
======================================================================
Executes minimal training (warm-up steps on batch) and analyzes layer-wise
gradient flow across all 4 cells of the 2x2 experimental matrix:
  - Cell A (v4):   Convolutional + Soft Penalty
  - Cell B (v5):   Transformer   + Hard Masking
  - Cell C (v4.1): Convolutional + Hard Masking
  - Cell D (v5.1): Transformer   + Soft Penalty

Generates Figure 8 (4-panel gradient flow comparison plot) and prints the exact gradient norms.
"""

import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cpu")
print(f"🖥️ Running Gradient Flow Analysis on Device: {device}")

# ─────────────────────────────────────────────────────────────
# 1. LOAD MODULES CLEANLY VIA IMPORTLIB
# ─────────────────────────────────────────────────────────────
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v4_gen = load_module("v4_gen", "v4/generator.py")
v4_crit = load_module("v4_crit", "v4/critic.py")

v5_gen = load_module("v5_gen", "v5/generator.py")
v5_crit = load_module("v5_crit", "v5/critic.py")

v4_1_gen = load_module("v4_1_gen", "v4.1/generator.py")
v4_1_crit = load_module("v4_1_crit", "v4.1/critic.py")

v5_1_gen = load_module("v5_1_gen", "v5.1/generator.py")
v5_1_crit = load_module("v5_1_crit", "v5.1/critic.py")

# ─────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def compute_neuro_symbolic_loss(probs_fake, cond_vec):
    loss = 0.0
    child_mask = cond_vec[:, 0] == 1
    if child_mask.any():
        child_work = probs_fake[child_mask, 0, :]
        loss = torch.mean(child_work)
    return loss

def train_minimal_steps(cell_key, G, D, num_steps=5, batch_size=16):
    """
    Executes a minimal warm-up training pass (5 steps) to calibrate network weights.
    """
    print(f"  Training {cell_key} for {num_steps} warm-up steps...")
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0001)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0001)
    
    for step in range(num_steps):
        # Step D
        opt_D.zero_grad()
        real_fake_z = torch.randn(batch_size, 128, device=device)
        cond = torch.zeros(batch_size, 83, device=device)
        cond[batch_size // 2:, 0] = 1.0  # 50% child, 50% adult
        dist = torch.zeros(batch_size, dtype=torch.long, device=device)
        state = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        real = torch.randn(batch_size, 9, 48, 1, device=device)
        with torch.no_grad():
            if cell_key in ["Cell B (v5)", "Cell D (v5.1)"]:
                fake_res = G(real_fake_z, cond, dist, state, temp=1.0, hard=True, return_soft=False)
                fake = fake_res
            else:
                fake = G(real_fake_z, cond, dist, state, temp=1.0, hard=True)
        
        if cell_key in ["Cell B (v5)", "Cell D (v5.1)"]:
            scores_real = D(real, cond, dist, state, return_proj=False)
            scores_fake = D(fake, cond, dist, state, return_proj=False)
        else:
            scores_real = D(real, cond, dist, state)
            scores_fake = D(fake, cond, dist, state)
            
        loss_D = scores_fake.mean() - scores_real.mean()
        loss_D.backward()
        opt_D.step()
        
        # Step G
        opt_G.zero_grad()
        z_g = torch.randn(batch_size, 128, device=device)
        
        if cell_key == "Cell A (v4: Conv + Soft Penalty)":
            fake = G(z_g, cond, dist, state, temp=1.0, hard=True)
            scores = D(fake, cond, dist, state)
            probs = (fake.squeeze(-1) + 1.0) / 2.0
            loss_logic = compute_neuro_symbolic_loss(probs, cond)
            loss_G = -scores.mean() + 10.0 * loss_logic
            
        elif cell_key == "Cell B (v5: Transf + Hard Mask)":
            fake = G(z_g, cond, dist, state, temp=1.0, hard=True, return_soft=False)
            scores = D(fake, cond, dist, state, return_proj=False)
            loss_G = -scores.mean()
            
        elif cell_key == "Cell C (v4.1: Conv + Hard Mask)":
            fake = G(z_g, cond, dist, state, temp=1.0, hard=True)
            scores = D(fake, cond, dist, state)
            loss_G = -scores.mean()
            
        elif cell_key == "Cell D (v5.1: Transf + Soft Penalty)":
            fake_hard, fake_soft = G(z_g, cond, dist, state, temp=1.0, hard=True, return_soft=True)
            scores = D(fake_hard, cond, dist, state, return_proj=False)
            probs = (fake_soft.squeeze(-1) + 1.0) / 2.0
            loss_logic = compute_neuro_symbolic_loss(probs, cond)
            loss_G = -scores.mean() + 10.0 * loss_logic
            
        loss_G.backward()
        opt_G.step()

# ─────────────────────────────────────────────────────────────
# 3. MEASURE GRADIENT NORMS
# ─────────────────────────────────────────────────────────────
def measure_cell_gradients(cell_key, G, D, batch_size=32):
    """
    Measures per-layer gradient norms separately for Adult (Unconstrained) and Child (Constrained) profiles.
    """
    G.eval()
    D.eval()
    grad_results = {}
    
    scenarios = {
        "Adults (Unconstrained)": 0.0,
        "Children (Constrained)": 1.0
    }
    
    for scen_name, is_child in scenarios.items():
        G.zero_grad()
        z = torch.randn(batch_size, 128, device=device)
        cond = torch.zeros(batch_size, 83, device=device)
        cond[:, 0] = is_child
        dist = torch.zeros(batch_size, dtype=torch.long, device=device)
        state = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        if cell_key == "Cell A (v4: Conv + Soft Penalty)":
            fake = G(z, cond, dist, state, temp=1.0, hard=True)
            scores = D(fake, cond, dist, state)
            probs = (fake.squeeze(-1) + 1.0) / 2.0
            loss = -scores.mean() + 10.0 * compute_neuro_symbolic_loss(probs, cond)
            
        elif cell_key == "Cell B (v5: Transf + Hard Mask)":
            fake = G(z, cond, dist, state, temp=1.0, hard=True, return_soft=False)
            scores = D(fake, cond, dist, state, return_proj=False)
            loss = -scores.mean()
            
        elif cell_key == "Cell C (v4.1: Conv + Hard Mask)":
            fake = G(z, cond, dist, state, temp=1.0, hard=True)
            scores = D(fake, cond, dist, state)
            loss = -scores.mean()
            
        elif cell_key == "Cell D (v5.1: Transf + Soft Penalty)":
            fake_hard, fake_soft = G(z, cond, dist, state, temp=1.0, hard=True, return_soft=True)
            scores = D(fake_hard, cond, dist, state, return_proj=False)
            probs = (fake_soft.squeeze(-1) + 1.0) / 2.0
            loss = -scores.mean() + 10.0 * compute_neuro_symbolic_loss(probs, cond)
            
        loss.backward()
        
        layer_dict = {}
        for param_name, param in G.named_parameters():
            if 'weight' in param_name and param.grad is not None:
                # Select clean representative layer names
                if 'attn' in param_name and 'gamma' not in param_name:
                    short = 'attn_query' if 'query' in param_name else ('attn_key' if 'key' in param_name else 'attn_val')
                    layer_dict[short] = param.grad.norm().item()
                elif 'transformer1' in param_name and 'linear1' in param_name and 'layers.0' in param_name:
                    layer_dict['transf_layer0'] = param.grad.norm().item()
                elif 'transformer1' in param_name and 'linear1' in param_name and 'layers.1' in param_name:
                    layer_dict['transf_layer1'] = param.grad.norm().item()
                elif 'up1.conv_t' in param_name:
                    layer_dict['up1_conv_t'] = param.grad.norm().item()
                elif 'up2.conv_t' in param_name:
                    layer_dict['up2_conv_t'] = param.grad.norm().item()
                elif 'fc' in param_name:
                    layer_dict['fc_input'] = param.grad.norm().item()
                elif 'out_conv' in param_name:
                    layer_dict['out_conv'] = param.grad.norm().item()
                    
        grad_results[scen_name] = layer_dict
        
    return grad_results

# ─────────────────────────────────────────────────────────────
# 4. EXECUTE 2x2 FACTORIAL ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("🚀 RUNNING 2x2 FACTORIAL GRADIENT FLOW ANALYSIS")
print("="*75)

models = {
    "Cell A (v4: Conv + Soft Penalty)": (v4_gen.Generator(base_channels=64).to(device), v4_crit.Critic(base_channels=32).to(device)),
    "Cell B (v5: Transf + Hard Mask)": (v5_gen.Generator(base_channels=64).to(device), v5_crit.Critic(base_channels=32).to(device)),
    "Cell C (v4.1: Conv + Hard Mask)": (v4_1_gen.Generator(base_channels=64).to(device), v4_1_crit.Critic(base_channels=32).to(device)),
    "Cell D (v5.1: Transf + Soft Penalty)": (v5_1_gen.Generator(base_channels=64).to(device), v5_1_crit.Critic(base_channels=32).to(device)),
}

all_results = {}
for cell_key, (G, D) in models.items():
    print(f"\n▶ Testing {cell_key}...")
    train_minimal_steps(cell_key, G, D, num_steps=5)
    grads = measure_cell_gradients(cell_key, G, D)
    all_results[cell_key] = grads

# ─────────────────────────────────────────────────────────────
# 5. PRINT DETAILED SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*85)
print(f"{'Cell':<38} | {'Layer':<15} | {'Adult Norm':<12} | {'Child Norm':<12} | {'Status'}")
print("="*85)

for cell_key, data in all_results.items():
    adult_dict = data["Adults (Unconstrained)"]
    child_dict = data["Children (Constrained)"]
    for layer in adult_dict.keys():
        ad_v = adult_dict[layer]
        ch_v = child_dict.get(layer, 0.0)
        # Check ratio
        ratio = ch_v / max(1e-12, ad_v)
        if ratio > 0.05:
            status = f"🟢 HEALTHY ({ratio*100:.1f}%)"
        elif ratio > 0.001:
            status = f"🟡 REDUCED ({ratio*100:.2f}%)"
        else:
            status = f"🔴 DEAD GRADIENT ({ratio*100:.4f}%)"
        print(f"{cell_key:<38} | {layer:<15} | {ad_v:<12.5e} | {ch_v:<12.5e} | {status}")
    print("-" * 85)

# ─────────────────────────────────────────────────────────────
# 6. PLOT 4-PANEL GRADIENT FLOW COMPARISON (FIGURE 8)
# ─────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
axes = axes.flatten()

panel_titles = [
    '(a) Cell A: Convolutional + Soft Penalty (v4)', 
    '(b) Cell B: Transformer + Hard Masking (v5)', 
    '(c) Cell C: Convolutional + Hard Masking (v4.1)', 
    '(d) Cell D: Transformer + Soft Penalty (v5.1)'
]

for idx, (cell_key, data) in enumerate(all_results.items()):
    ax = axes[idx]
    adult_dict = data["Adults (Unconstrained)"]
    child_dict = data["Children (Constrained)"]
    
    layers = list(adult_dict.keys())
    adult_vals = [adult_dict[l] for l in layers]
    child_vals = [max(1e-10, child_dict.get(l, 1e-10)) for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, adult_vals, width, label='Adults (Unconstrained)', color='#1f77b4', alpha=0.9)
    if "Soft" in cell_key:
        ax.bar(x + width/2, child_vals, width, label='Children (Soft Penalty Active)', color='#2ca02c', alpha=0.9)
    else:
        ax.bar(x + width/2, child_vals, width, label='Children (Hard Masked)', color='#d62728', alpha=0.9)
        
    ax.set_yscale('log')
    ax.set_title(panel_titles[idx], fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha='right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=8.5, loc='upper right')
    if idx % 2 == 0:
        ax.set_ylabel('L2 Gradient Norm (Log Scale)', fontsize=10)

plt.tight_layout()
out_plot_png = "results/figure8_gradient_flow_2x2_comparison.png"
out_plot_pdf = "results/figure8_gradient_flow_2x2_comparison.pdf"
plt.savefig(out_plot_png, dpi=300)
plt.savefig(out_plot_pdf, format='pdf')
print(f"\n✅ Figure 8 comparison plot successfully generated and saved to:\n   - {out_plot_png}\n   - {out_plot_pdf}")
