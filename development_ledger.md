# Development Ledger - TUS-GAN

## 2026-06-27: Implementation of GLM-5.2 Code Review Fixes (v5)

### Generator (`v5/generator.py`)
- **Added Positional Encoding:** Added a learnable `pos_embed` to `TemporalTransformerBlock` to ensure the self-attention mechanism processes temporal order accurately.
- **Overhauled Weight Initialization:** Replaced the global normal initialization with targeted Kaiming Normal for convolutional layers and Xavier Uniform for linear layers to stabilize the Transformer attention weights and prevent breaking BatchNorm running statistics.
- **Returned Soft Targets:** Updated the `forward` pass to optionally return the continuous Gumbel-Softmax probabilities alongside the hard outputs, enabling differentiable auxiliary losses.

### Critic (`v5/critic.py`)
- **Added Positional Encoding:** Mirrored the Generator's temporal order tracking by adding a learnable `pos_embed` to the Critic's `TemporalTransformerBlock`.
- **Added Explicit Temporal Pooling:** Introduced an `nn.AdaptiveAvgPool1d(1)` before the InfoNCE projection head (`feat_proj`) instead of flat reshaping, improving contrastive representation learning.
- **Removed Final Spectral Norm:** Dropped the `nn.utils.spectral_norm` wrapper from the final real/fake scoring linear layer to maximize discriminator capacity.
- **Fixed Weight Initialization:** Removed silent exception swallowing (`try...except`) and correctly skipped spectral-norm-wrapped layers (`weight_orig`) during custom initialization.

### Training Script (`v5/train.py`)
- **Auxiliary Losses on Soft Targets:** Now retrieves `return_soft=True` from the Generator and passes these differentiable probabilities to the Transition Matrix and Spell-Duration losses, fixing the broken gradient flow.
- **Stabilized Duration Loss:** Clamped `starts` (min=0.5) and `avg_duration` (max=48.0) in `compute_spell_duration_loss` to prevent mathematically exploding NaN/Inf gradients.
- **Reduced Loss Lambdas:** Scaled down `lambda_transition` (to 1.0), `lambda_infonce` (to 0.5), and `lambda_duration` (to 1.0) to prevent auxiliary losses from overpowering WGAN-GP adversarial realism.
- **Prevented GPU OOM:** Removed the blanket `self.diary.to(device)` pre-loading in the dataset constructor, offloading memory management back to the DataLoader's pin-memory buffer.

### Smoke Tests (`v5/smoke_test.py`)
- **Updated Generator Validation:** Upgraded test assertions to explicitly request and validate the dimensions and continuous (soft) values of `fake_soft`, verifying that deterministic child labor masking and continuous gradient streams remain intact.

### Documentation (`tusgan-v5.md`)
- **Updated Graphical Representation:** Reflected the integration of Positional Encoding in both Transformer Backbones. Visualized the split between Hard Outputs (for Critic scoring) and Soft Probabilities (for Auxiliary Differentiable Losses). Added the Explicit Temporal Pooling step before the Critic's Projection Head.
- **Refined Component Descriptions:** Detailed the exact mechanisms regarding temporal pooling, positional embeddings, and soft probabilities in the architecture breakdown.
