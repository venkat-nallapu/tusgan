# TUS-GAN — Architecture Specification Library (`docs/architecture_specs/`)

This directory houses the formal engineering and design documents for every version and factorial cell of the TUS-GAN lineage.

---

## Architecture Lineage & Factorial Matrix

| Document | Version / Cell | Backbone | Key Innovation | Constraint Policy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`tusgan-v2.md`](file:///home/venkat/projects/tusgan/docs/architecture_specs/tusgan-v2.md) | **v2** | CNN | 9-channel one-hot + State & District learned embeddings | None | Legacy Prototype |
| [`tusgan-v3.md`](file:///home/venkat/projects/tusgan/docs/architecture_specs/tusgan-v3.md) | **v3** | CNN | Gumbel-Softmax + EMA generator + Global transition matrix loss | None | Baseline ($JSD = 0.000161$) |
| [`tusgan-v4.md`](file:///home/venkat/projects/tusgan/docs/architecture_specs/tusgan-v4.md) | **v4 (Cell A)** | CNN | Decoupled AC-GAN + 4-Quadrant Time-Slice transitions | Differentiable Soft Penalty ($\lambda = 10$) | **Production Champion 🏆** ($JSD = 0.000083$) |
| [`tusgan-v4one.md`](file:///home/venkat/projects/tusgan/docs/architecture_specs/tusgan-v4one.md) | **v4.1 (Cell C)** | CNN | Factorial isolate: CNN backbone with $-10^9$ logit masking | Hard Logit Masking | Factorial Dead-Gradient Proof |
| [`tusgan-v5.md`](file:///home/venkat/projects/tusgan/docs/architecture_specs/tusgan-v5.md) | **v5 (Cell B)** | Transformer | Temporal Transformer (2L, 4H) + InfoNCE dual projection heads | Hard Logit Masking | Factorial Cell B ($JSD = 0.001557$) |
| [`tusgan-v5one.md`](file:///home/venkat/projects/tusgan/docs/architecture_specs/tusgan-v5one.md) | **v5.1 (Cell D)** | Transformer | Unmasked transformer logits + continuous soft loss backpropagation | Differentiable Soft Penalty ($\lambda = 10$) | Gradient Restoration Proof ($JSD = 0.004534$) |
| [`tusgan-v6.md`](file:///home/venkat/projects/tusgan/docs/architecture_specs/tusgan-v6.md) | **v6** | Conformer | Deep Conformer (6L, 8H) + Sinusoidal PE + Rebalanced InfoNCE | Hybrid: Soft Training + Hard Inference | Next-Gen Architecture |

---

> [!TIP]
> Backward-compatible symlinks are maintained at the repository root for all `tusgan-v*.md` files.
