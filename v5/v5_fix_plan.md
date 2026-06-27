# TUS-GAN v5 - Implementation Plan for Code Review Fixes

I have reviewed the `code_reveiw.md` file and cross-referenced it with the actual source code in `generator.py`, `critic.py`, `train.py`, and `smoke_test.py`. 

**Status: ✅ The code review is 100% accurate.** 

All identified issues are present in the current v5 codebase. Below is the proposed plan to implement these changes safely and efficiently.

## Phase 1: Core Architectural Fixes (Generator & Critic)

1. **`generator.py` Fixes**
   - **Fix 1.1:** Update `TemporalTransformerBlock` to include a learnable positional embedding `self.pos_embed` and add it to the sequence before the transformer layer.
   - **Fix 1.2:** Overhaul `_init_weights`. Replace the blanket `nn.init.normal_` with targeted Kaiming Normal for Convs, Xavier Uniform for Linears, and skip BN/CBN affine layers.
   - **Fix 1.3:** Modify the `forward` pass to accept a `return_soft=False` parameter. When true, it should return both hard outputs and soft Gumbel-Softmax probabilities.

2. **`critic.py` Fixes**
   - **Fix 2.1:** Mirror Fix 1.1 by adding the exact same learnable positional embedding to the Critic's `TemporalTransformerBlock`.
   - **Fix 2.2:** Add `nn.AdaptiveAvgPool1d(1)` for explicit temporal pooling before the InfoNCE projection head (`self.feat_proj`). Update dimensions accordingly.
   - **Fix 2.3:** Remove `nn.utils.spectral_norm` from the final `self.output` linear layer to prevent artificially restricting the WGAN-GP scoring capacity.
   - **Fix 2.4:** Remove the `try...except` block in `_init_weights`. Specifically skip layers with `weight_orig` (spectral norm wrappers) and use Xavier Uniform safely.

## Phase 2: Training & Loss Function Stability

3. **`train.py` Fixes**
   - **Fix 3.1:** Update the Generator training step to request `return_soft=True` from `G`. Pass the hard outputs to the Critic, but use the **soft outputs** to calculate the Transition and Spell-Duration losses.
   - **Fix 3.2:** Update `compute_spell_duration_loss` to include `.clamp(min=0.5)` on `starts` and `.clamp(max=48.0)` on `avg_duration` to prevent NaN/Inf gradients.
   - **Fix 3.3:** Adjust hyperparameters in `get_config()`:
     - `lambda_transition` -> `1.0`
     - `lambda_infonce` -> `0.5`
     - `lambda_duration` -> `1.0`
   - **Fix 3.4:** Remove the block in `TUSDataset.__init__` that pre-loads the entire dataset to the GPU (`to(device)`), relying on the DataLoader's `pin_memory` instead.

## Phase 3: Validation

4. **`smoke_test.py` Updates**
   - **Fix 4.1:** Update the Generator smoke test to pass `return_soft=True`.
   - Add assertions to verify that the soft outputs are returned correctly, have the expected shapes, and are genuinely soft (not strictly -1.0 or 1.0).
