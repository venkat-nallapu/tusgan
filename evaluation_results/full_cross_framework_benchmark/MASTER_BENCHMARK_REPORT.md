# TUS-GAN Comprehensive Multi-Model Evaluation Benchmark
## Elsevier *Transportation Research Part C: Emerging Technologies* (TRC) Standards

This document presents the unified 6-domain benchmark evaluated on the Indian Time Use Survey (ITUS) 2019 microdata.

### 1. Model Architecture & Experimental Lineage
| Model | Factorial Cell | Backbone | Conditioning / Constraint | Checkpoint Epoch | Loaded State |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **v4** | Cell A (Champion 🏆) | CNN | Soft Penalty (λ=10) | 250 | `G_state_ema` |
| **v4.1** | Cell C (Ablation) | CNN | Hard Mask (-1e9) | 250 | `G_state_ema` |
| **v5** | Cell B (Ablation) | Transformer | Hard Mask (-1e9) | 250 | `G_state_ema` |
| **v5.1** | Cell D (Restoration) | Transformer | Soft Penalty (λ=10) | 250 | `G_state_ema` |
| **actvae_soft** | Caveat Constrained | ActVAE (Conditional VAE) | Soft Penalty (C1) | N/A (Pre-trained) | `CSV_Episodes` |
| **actvae_unconstrained** | Caveat Baseline | ActVAE (Conditional VAE) | None (Unconstrained Baseline) | N/A (Pre-trained) | `CSV_Episodes` |


### 2. Domain 1: Statistical & Aggregate Distributional Realism
| Model | Macro JSD (9×48) ↓ | Max Slot JSD ↓ | Min Slot JSD ↓ | Participation TVD ↓ | Mean Abs Duration Error (min/day) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v4** | **0.000545** | 0.004195 | 0.000059 | 0.0732 | 2.54 |
| **v4.1** | **0.002586** | 0.012341 | 0.000406 | 0.1534 | 11.09 |
| **v5** | **0.001254** | 0.003932 | 0.000133 | 0.0432 | 7.46 |
| **v5.1** | **0.002056** | 0.004995 | 0.000405 | 0.0729 | 10.33 |
| **actvae_soft** | **0.093012** | 0.193623 | 0.001296 | 0.7435 | 55.40 |
| **actvae_unconstrained** | **0.094954** | 0.208078 | 0.001296 | 0.7337 | 55.92 |


### 3. Domain 2: Temporal Pacing & Episode Dynamics
| Model | Slice 0 F-Norm ↓ | Slice 1 F-Norm ↓ | Slice 2 F-Norm ↓ | Slice 3 F-Norm ↓ | Mean Slice F-Norm ↓ | Mean Spell EMD (min) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v4** | 0.1254 | 0.2007 | 0.1511 | 0.3320 | **0.2023** | **11.69** |
| **v4.1** | 0.3118 | 0.4740 | 0.3490 | 0.6676 | **0.4506** | **21.97** |
| **v5** | 0.2285 | 0.2720 | 0.1817 | 0.3495 | **0.2579** | **14.95** |
| **v5.1** | 0.1890 | 0.2494 | 0.2082 | 0.3136 | **0.2400** | **17.02** |
| **actvae_soft** | 1.3976 | 1.4921 | 1.1264 | 2.0173 | **1.5084** | **72.78** |
| **actvae_unconstrained** | 1.3974 | 1.5191 | 1.1183 | 2.0173 | **1.5130** | **72.80** |


### 4. Domain 3: Sequence Structure & Tour Formation
| Model | Real Mean Episodes | Fake Mean Episodes | Episode Error Δμ ↓ | Real Switching Freq | Fake Switching Freq | Freq Error Δμ ↓ | Work Start EMD (min) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v4** | 12.31 | 12.40 | 0.09 | 11.31 | 11.40 | 0.09 | **17.7** |
| **v4.1** | 12.31 | 12.88 | 0.56 | 11.31 | 11.88 | 0.56 | **31.2** |
| **v5** | 12.31 | 11.64 | -0.68 | 11.31 | 10.64 | -0.68 | **6.8** |
| **v5.1** | 12.31 | 11.34 | -0.98 | 11.31 | 10.34 | -0.98 | **14.7** |
| **actvae_soft** | 12.31 | 5.59 | -6.73 | 11.31 | 4.59 | -6.73 | **55.1** |
| **actvae_unconstrained** | 12.31 | 5.59 | -6.72 | 11.31 | 4.59 | -6.72 | **57.9** |


### 5. Domain 4: Normative & Statutory Feasibility
| Model | 24h Time Budget Feasibility | Child Labor Raw Rate (%) ↓ | Child Labor Delivered Rate (%) | Discard Overhead (%) ↓ | Under-6h Sleep (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v4** | 100.0% | **0.19%** | **0.00%** | 0.19% | 0.05% |
| **v4.1** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.09% |
| **v5** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.18% |
| **v5.1** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.13% |
| **actvae_soft** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.00% |
| **actvae_unconstrained** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.00% |


### 6. Domain 5: Demographic & Subgroup Equity Parity
| Model | Real Dom Ratio (F/M) | Fake Dom Ratio (F/M) | Dom Ratio Error ↓ | Female JSD ↓ | Male JSD ↓ | Rural JSD ↓ | Urban JSD ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v4** | 7.59× | 7.91× | **0.319** | 0.000526 | 0.000791 | 0.000921 | 0.000594 |
| **v4.1** | 7.59× | 9.28× | **1.695** | 0.002210 | 0.003867 | 0.003652 | 0.002443 |
| **v5** | 7.59× | 8.03× | **0.441** | 0.001496 | 0.001395 | 0.001254 | 0.001308 |
| **v5.1** | 7.59× | 10.25× | **2.661** | 0.002623 | 0.002273 | 0.002373 | 0.002018 |
| **actvae_soft** | 7.59× | 487602850.23× | **487602842.639** | 0.131034 | 0.103351 | 0.102687 | 0.098090 |
| **actvae_unconstrained** | 7.59× | 499412476.47× | **499412468.883** | 0.133873 | 0.103828 | 0.104111 | 0.100236 |


### 7. Domain 6: ML Indistinguishability & Privacy
| Model | RF ROC-AUC (Optimal=0.500) ↓ | RF Accuracy (Optimal=50.0%) ↓ | Exact Match Count | Exact Match Rate (%) | Mean Min Hamming (slots) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v4** | **0.6255** | 58.97% | 29 | **0.0967%** | 9.45 |
| **v4.1** | **0.7532** | 67.88% | 54 | **0.1800%** | 9.29 |
| **v5** | **0.6790** | 63.53% | 27 | **0.0900%** | 9.11 |
| **v5.1** | **0.7016** | 64.92% | 54 | **0.1800%** | 8.23 |
| **actvae_soft** | **1.0000** | 99.87% | 187 | **0.6233%** | 3.70 |
| **actvae_unconstrained** | **1.0000** | 99.85% | 253 | **0.8433%** | 3.73 |

