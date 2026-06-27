TUS-GAN v5 — Code Review: Fixes & Rationale
This document outlines all critical and high-priority fixes required across the TUS-GAN v5 codebase, along with the underlying architectural and mathematical reasons for each change.

1. generator.py
Fix 1.1: Add Positional Encoding to TemporalTransformerBlock
Severity: 🔴 Critical
Reason: The nn.TransformerEncoder has no inherent concept of sequence order. Without positional encodings, the attention mechanism treats time step 1 (6:00 AM) and time step 48 (5:30 AM next day) as identical positions in a set. This completely destroys temporal coherence.

class TemporalTransformerBlock(nn.Module):    def __init__(self, channels: int, max_seq_len: int = 48, num_layers: int = 1, nhead: int = 4):        super().__init__()        # ADD THIS: Learned positional embedding        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, channels) * 0.02)                encoder_layer = nn.TransformerEncoderLayer(            d_model=channels, nhead=nhead, dim_feedforward=channels * 4,            dropout=0.1, activation="gelu", batch_first=True,        )        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)    def forward(self, x: torch.Tensor) -> torch.Tensor:        B, C, H, W = x.shape        seq = x.squeeze(-1).permute(0, 2, 1)                # ADD THIS: Inject positional information        seq = seq + self.pos_embed[:, :H, :]                 out_seq = self.transformer(seq)        out = out_seq.permute(0, 2, 1).unsqueeze(-1)        return x + out
Fix 1.2: Overhaul Weight Initialization
Severity: 🟠 High
Reason: Applying nn.init.normal_(std=0.02) to all layers is a legacy DCGAN practice. It breaks BatchNorm2d running statistics and destabilizes Transformer attention weights (which rely on Xavier/Glorot initialization to maintain variance across layers).

def _init_weights(self):    for m in self.modules():        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')        elif isinstance(m, nn.Linear) and not hasattr(m, '_is_cbn_affine'):            nn.init.xavier_uniform_(m.weight)        # DELIBERATELY SKIP: nn.BatchNorm2d and CBN affine layers        # Let PyTorch's default init (weight=1, bias=0) handle them.                if hasattr(m, 'bias') and m.bias is not None:            nn.init.zeros_(m.bias)
Fix 1.3: Return Soft Probabilities for Auxiliary Losses
Severity: 🔴 Critical
Reason: When hard=True, the straight-through Gumbel-Softmax estimator outputs hard one-hot vectors. While this is great for forward passes (discrete realism), taking the MSE or Transition loss of these hard vectors yields sparse, zero, or useless gradients. The auxiliary losses must be computed on the continuous soft probabilities.

def forward(self, z, cond_vector, district_ids, state_ids, temp=1.0, hard=True, return_soft=False):    # ... [Embedding and Backbone code remains the same] ...        logits = self.out_conv(h)  # (B, 9, 48, 1)    # Deterministic Logit Masking    is_child = cond_vector[:, 0] == 1    if is_child.any():        logits[is_child, 0, :, :] = -1e9    # Compute both hard and soft outputs    y_hard = F.gumbel_softmax(logits, tau=temp, hard=True, dim=1)    y_soft = F.gumbel_softmax(logits, tau=temp, hard=False, dim=1)        output_hard = 2.0 * y_hard - 1.0    output_soft = 2.0 * y_soft - 1.0        # ADD THIS: Option to return soft targets for differentiable losses    if return_soft:        return output_hard, output_soft    return output_hard
2. critic.py
Fix 2.1: Add Positional Encoding
Severity: 🔴 Critical
Reason: Identical to Fix 1.1. The Critic's Transformer must understand time order to accurately penalize unrealistic chronological transitions (e.g., sleeping -> working -> sleeping -> working within a few hours).

Action: Apply the exact same pos_embed fix from 1.1 to the Critic's TemporalTransformerBlock.

Fix 2.2: Explicit Temporal Pooling Before InfoNCE Projection
Severity: 🟡 Medium
Reason: Currently, the code flattens the entire spatial/temporal tensor (x.view(x.size(0), -1)) and feeds it directly into the feat_proj MLP. This forces the MLP to arbitrarily learn how to compress time, making the InfoNCE loss extremely difficult to optimize. Explicit pooling yields much better contrastive representations.

# In Critic.__init__:self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # Collapse time explicitlyself.feat_proj = nn.Sequential(    nn.Linear(base_channels * 4, base_channels * 2),  # Note: Input dim changed    nn.LeakyReLU(0.2, inplace=True),    nn.Linear(base_channels * 2, proj_dim),)# Ensure self.output input dim is also updated to base_channels * 4# In Critic.forward (inside return_proj block):# OLD: x = x.view(x.size(0), -1)# NEW:x_pooled = self.temporal_pool(x.squeeze(-1)).squeeze(-1)  # (B, C*4)score = self.output(x_pooled)# ... then use x_pooled for feat_proj ...
Fix 2.3: Remove Spectral Norm from Final Output Layer
Severity: 🟡 Medium
Reason: Spectral Normalization enforces a strict Lipschitz constraint (preventing the output scale from growing too fast). While great for intermediate feature extraction layers, applying it to the final 1-dimensional scoring layer artificially restricts the Critic's ability to output extreme scores, reducing its capacity to differentiate real from fake.

# OLD:self.output = nn.utils.spectral_norm(nn.Linear(flat_dim, 1))# NEW:self.output = nn.Linear(flat_dim, 1)
Fix 2.4: Fix Silent Exception Swallowing & Spectral Norm Init
Severity: 🟠 High
Reason: The try...except Exception: pass block hides real initialization bugs. Furthermore, calling nn.init.normal_ on a layer wrapped in spectral_norm modifies the wrapper, not the actual weights used in the forward pass (which are renamed to weight_orig).

def _init_weights(self):    for m in self.modules():        # Skip spectral norm wrappers entirely        if hasattr(m, 'weight_orig'):            continue                     if isinstance(m, (nn.Conv2d, nn.Linear)):            if m.weight is not None:                nn.init.xavier_uniform_(m.weight)        if hasattr(m, 'bias') and m.bias is not None:            nn.init.zeros_(m.bias)
3. train.py
Fix 3.1: Compute Auxiliary Losses on Soft Probabilities
Severity: 🔴 Critical
Reason: Building on Fix 1.3, the training loop currently computes the Transition and Duration losses on the hard one-hot outputs. This results in zero gradients flowing back through these auxiliary terms, rendering your carefully designed loss functions useless.

# OLD:fake_diaries = G(z_g, cond_vec, dist_ids, state_ids, temp=current_temp, hard=True)probs_fake = (fake_diaries.squeeze(-1) + 1.0) / 2.0# NEW:# Get both hard (for critic) and soft (for losses)fake_hard, fake_soft = G(z_g, cond_vec, dist_ids, state_ids,                          temp=current_temp, hard=True, return_soft=True)# Use hard for adversarial scoringfake_scores, f_proj_fake, c_proj_fake = D(fake_hard, cond_vec, dist_ids, state_ids, return_proj=True)# Use soft for differentiable auxiliary losses!probs_fake = (fake_soft.squeeze(-1) + 1.0) / 2.0
Fix 3.2: Stabilize Spell-Duration Loss
Severity: 🟠 High
Reason: The calculation total_time / (starts + 1e-5) will mathematically explode (producing NaN or Inf gradients) if an activity is never initiated (starts == 0) but has slight floating-point noise in total_time.

def compute_spell_duration_loss(probs_fake, real_avg_durations):    total_time = probs_fake.sum(dim=2)    x_t = probs_fake[:, :, 1:]    x_tm1 = probs_fake[:, :, :-1]    starts = (x_t * (1 - x_tm1)).sum(dim=2) + probs_fake[:, :, 0]        # ADD THIS: Clamp to prevent division by zero/exploding gradients    starts = starts.clamp(min=0.5)    avg_duration = total_time / starts        # ADD THIS: Duration logically cannot exceed the 48-slot sequence length    avg_duration = avg_duration.clamp(max=48.0)     batch_avg_duration = avg_duration.mean(dim=0)    return F.mse_loss(batch_avg_duration, real_avg_durations)
Fix 3.3: Reduce Auxiliary Loss Lambdas
Severity: 🟡 Medium
Reason: Your lambda values (lambda_transition=10.0, lambda_duration=15.0) are far too high. If these approximate/smoothed losses dominate the pure WGAN-GP adversarial loss, the Generator will optimize for the auxiliary metrics while ignoring adversarial realism, leading to mode collapse.

# RECOMMENDED TUNING:lambda_transition=1.0,   # Down from 10.0lambda_infonce=0.5,      # Down from 5.0lambda_duration=1.0,     # Down from 15.0
Fix 3.4: Remove GPU Pre-loading in Dataset
Severity: 🟠 High
Reason: Loading the entire dataset to GPU memory in __init__ (self.diary = self.diary.to(device)) will trigger an Out-Of-Memory (OOM) error on almost any standard GPU when scaling beyond a few thousand samples. Let the DataLoader and pin_memory handle memory transfers efficiently.

# In TUSDataset.__init__:# DELETE THESE LINES:# if device is not None:#     self.diary = self.diary.to(device)#     self.cond = self.cond.to(device)#     self.district_ids = self.district_ids.to(device)#     self.state_ids = self.state_ids.to(device)
4. smoke_test.py
Fix 4.1: Update Test to Validate Soft Outputs
Severity: 🟡 Medium
Reason: To ensure Fix 1.3 and 3.1 work correctly, the smoke test must explicitly verify that the generator can return soft outputs and that gradients flow through them.

# In Step 2 (Generator Test), add:fake_hard, fake_soft = G(z, cond_vec, dist_ids, state_ids, temp=1.0, hard=True, return_soft=True)assert fake_hard.shape == expected_shapeassert fake_soft.shape == expected_shape# Soft outputs should NOT be strictly -1.0 or 1.0assert not (fake_soft == -1.0).all() and not (fake_soft == 1.0).all(), "Soft outputs are hard!"
5. evaluate_ultimate.py
Status: ✅ No Fixes Required
Reason: This evaluation script is exceptionally well-written. Because it runs strictly under torch.no_grad() and operates on already-generated numpy arrays/tensors, the differentiability issues (Gumbel-Softmax, Duration loss division) do not apply here. The metrics (JSD, EMD, F-Norm, Random Forest AUC) are mathematically sound for post-hoc evaluation.

Summary Priority Matrix

```
File	Issue	Impact if Ignored	Priority
generator.py	Missing Pos Encoding	Temporal chaos (morning=night)	P0
generator.py	No Soft Output Return	Auxiliary losses do nothing	P0
train.py	Aux losses on hard tensors	Zero gradients for Duration/Trans	P0
critic.py	Missing Pos Encoding	Cannot learn chronological logic	P0
train.py	Duration loss division by 0	Training crashes with NaN loss	P1
critic.py	Broken weight init wrapper	Weights don't actually update	P1
generator.py	Bad weight init	Instability/slow convergence	P1
train.py	Dataset pre-loaded to GPU	Out-of-Memory crashes	P1
train.py	Lambdas too high	Mode collapse / loss imbalance	P2
critic.py	Spectral norm on output	Artificially restricted scoring	P2
critic.py	Flat temporal pooling	Slower/harder InfoNCE convergence	P2
```

