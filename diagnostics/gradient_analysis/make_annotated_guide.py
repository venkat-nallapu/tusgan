import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT_DIR = "/home/venkat/.gemini/antigravity-cli/brain/bb3b38d3-413d-4133-8b29-e91990a6d585"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
fig.patch.set_facecolor("#FFFFFF")

# -------------------------------------------------------------
# SUBPLOT 1: Comparing (a) vs (c) - The Convolutional Pair
# -------------------------------------------------------------
ax1 = axes[0]
layers_conv = ['fc_input', 'up1_conv_t', 'up2_conv_t', 'out_conv']
a_soft = [1.62e-6, 7.50e-6, 3.43e-4, 1.16e-1]      # (a) Cell A: Soft
c_hard = [1.01e-7, 5.11e-8, 1.50e-6, 1.94e-4]      # (c) Cell C: Hard
adult_conv = [6.97e-12, 3.00e-12, 1.04e-10, 1.34e-7]

x = np.arange(len(layers_conv))
width = 0.28

ax1.bar(x - width, adult_conv, width, label='Baseline (Adults)', color='#1f77b4', alpha=0.85)
ax1.bar(x, a_soft, width, label='Cell A (Soft Penalty [Soft])', color='#2ca02c', alpha=0.9)
ax1.bar(x + width, c_hard, width, label='Cell C (Hard Masking [Hard])', color='#d62728', alpha=0.9)

ax1.set_yscale('log')
ax1.set_title('HOW TO DIFFERENTIATE (a) vs (c)\n[Convolutional Backbone]', fontsize=11.5, fontweight='bold', color='#003366')
ax1.set_xticks(x)
ax1.set_xticklabels(layers_conv, fontsize=9.5)
ax1.set_ylabel('Gradient Norm (Log Scale)', fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.legend(fontsize=9, loc='upper left')

# Highlight annotation on out_conv
ax1.annotate('600× STRONGER LEARNING SIGNAL!\n(Soft Penalty actively teaches output layer)',
             xy=(3, 1.16e-1), xytext=(1.0, 1.0e-2),
             arrowprops=dict(facecolor='#2ca02c', edgecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#1B5E20',
             bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='#2E7D32', lw=1.2))

ax1.annotate('597× WEAKER\n(Silenced by Mask)',
             xy=(3.28, 1.94e-4), xytext=(2.2, 5.0e-8),
             arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#B71C1C',
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#C62828', lw=1.2))

# -------------------------------------------------------------
# SUBPLOT 2: Comparing (b) vs (d) - The Transformer Pair
# -------------------------------------------------------------
ax2 = axes[1]
layers_transf = ['fc_input', 'transf_layer0', 'transf_layer1', 'out_conv']
adult_tf = [2.13, 0.261, 0.286, 0.485]
b_hard = [1.93, 0.248, 0.263, 0.614]   # (b) Cell B: Hard
d_soft = [2.27, 0.970, 0.793, 1.096]   # (d) Cell D: Soft

x2 = np.arange(len(layers_transf))

ax2.bar(x2 - width, adult_tf, width, label='Baseline (Adults)', color='#1f77b4', alpha=0.85)
ax2.bar(x2, d_soft, width, label='Cell D (Soft Penalty [Soft])', color='#2ca02c', alpha=0.9)
ax2.bar(x2 + width, b_hard, width, label='Cell B (Hard Masking [Hard])', color='#d62728', alpha=0.9)

ax2.set_title('HOW TO DIFFERENTIATE (b) vs (d)\n[Transformer Backbone]', fontsize=11.5, fontweight='bold', color='#003366')
ax2.set_xticks(x2)
ax2.set_xticklabels(layers_transf, fontsize=9.5)
ax2.set_ylabel('Gradient Norm (Linear Scale)', fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.legend(fontsize=9, loc='upper left')

# Highlight annotation on Transformer layers
ax2.annotate('3.7× HIGHER GRADIENT BOOST!\n(Attention heads actively learn the rule)',
             xy=(1, 0.970), xytext=(0.5, 1.8),
             arrowprops=dict(facecolor='#2ca02c', edgecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#1B5E20',
             bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='#2E7D32', lw=1.2))

ax2.annotate('NO EXTRA SIGNAL\n(Red bar ≤ Blue bar)',
             xy=(1.28, 0.248), xytext=(1.8, 0.05),
             arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
             fontsize=8.5, fontweight='bold', color='#B71C1C',
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#C62828', lw=1.2))

plt.suptitle('Side-by-Side Diagnostic Guide: What Actually Distinguishes Soft Penalties from Hard Masking', fontsize=13, fontweight='bold')
plt.tight_layout()
out_file = os.path.join(OUT_DIR, "figure8_annotated_difference_guide.png")
plt.savefig(out_file, dpi=300)
plt.close()
print(f"Saved: {out_file}")
