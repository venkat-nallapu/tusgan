# TUS-GAN Comprehensive Multi-Model Evaluation Benchmark
## Elsevier *Transportation Research Part C: Emerging Technologies* (TRC) Standards

This document presents the unified 6-domain benchmark evaluated on the Indian Time Use Survey (ITUS) 2019 microdata.

### 1. Model Architecture & Experimental Lineage
| Model | Factorial Cell | Backbone | Conditioning / Constraint | Checkpoint Epoch | Loaded State |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **v3** | Baseline | CNN | None (Unconstrained Baseline) | 350 | `G_state_ema` |
| **v4** | Cell A (Champion 🏆) | CNN | Soft Penalty (λ=10) | 250 | `G_state_ema` |
| **v4.1** | Cell C (Ablation) | CNN | Hard Mask (-1e9) | 250 | `G_state_ema` |
| **v5** | Cell B (Ablation) | Transformer | Hard Mask (-1e9) | 250 | `G_state_ema` |
| **v5.1** | Cell D (Restoration) | Transformer | Soft Penalty (λ=10) | 250 | `G_state_ema` |


### 2. Domain 1: Statistical & Aggregate Distributional Realism
| Model | Macro JSD (9×48) ↓ | Max Slot JSD ↓ | Min Slot JSD ↓ | Participation TVD ↓ | Mean Abs Duration Error (min/day) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v3** | **0.000321** | 0.000935 | 0.000096 | 0.0348 | 4.33 |
| **v4** | **0.000546** | 0.004019 | 0.000044 | 0.0742 | 2.42 |
| **v4.1** | **0.002564** | 0.012429 | 0.000512 | 0.1537 | 10.95 |
| **v5** | **0.001266** | 0.003843 | 0.000194 | 0.0407 | 7.48 |
| **v5.1** | **0.002027** | 0.004753 | 0.000429 | 0.0717 | 10.19 |


### 3. Domain 2: Temporal Pacing & Episode Dynamics
| Model | Slice 0 F-Norm ↓ | Slice 1 F-Norm ↓ | Slice 2 F-Norm ↓ | Slice 3 F-Norm ↓ | Mean Slice F-Norm ↓ | Mean Spell EMD (min) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **v3** | 0.1781 | 0.0775 | 0.0989 | 0.6377 | **0.2481** | **8.15** |
| **v4** | 0.1372 | 0.2079 | 0.1468 | 0.3428 | **0.2087** | **11.60** |
| **v4.1** | 0.3202 | 0.4683 | 0.3478 | 0.6742 | **0.4526** | **22.02** |
| **v5** | 0.2167 | 0.2745 | 0.2096 | 0.4159 | **0.2792** | **14.93** |
| **v5.1** | 0.1877 | 0.2547 | 0.2096 | 0.2846 | **0.2342** | **16.96** |


### 4. Domain 3: Sequence Structure & Tour Formation
| Model | Real Mean Episodes | Fake Mean Episodes | Episode Error Δμ ↓ | Real Switching Freq | Fake Switching Freq | Freq Error Δμ ↓ | Work Start EMD (min) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v3** | 12.31 | 11.96 | -0.35 | 11.31 | 10.96 | -0.35 | **6.2** |
| **v4** | 12.31 | 12.40 | 0.09 | 11.31 | 11.40 | 0.09 | **18.5** |
| **v4.1** | 12.31 | 12.88 | 0.57 | 11.31 | 11.88 | 0.57 | **30.8** |
| **v5** | 12.31 | 11.63 | -0.68 | 11.31 | 10.63 | -0.68 | **6.8** |
| **v5.1** | 12.31 | 11.37 | -0.94 | 11.31 | 10.37 | -0.94 | **13.1** |


### 5. Domain 4: Normative & Statutory Feasibility
| Model | 24h Time Budget Feasibility | Child Labor Raw Rate (%) ↓ | Child Labor Delivered Rate (%) | Discard Overhead (%) ↓ | Under-6h Sleep (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v3** | 100.0% | **3.39%** | **0.00%** | 3.39% | 0.09% |
| **v4** | 100.0% | **0.33%** | **0.00%** | 0.33% | 0.06% |
| **v4.1** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.07% |
| **v5** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.16% |
| **v5.1** | 100.0% | **0.00%** | **0.00%** | 0.00% | 0.12% |


### 6. Domain 5: Demographic & Subgroup Equity Parity
| Model | Real Dom Ratio (F/M) | Fake Dom Ratio (F/M) | Dom Ratio Error ↓ | Female JSD ↓ | Male JSD ↓ | Rural JSD ↓ | Urban JSD ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v3** | 7.59× | 8.51× | **0.924** | 0.000437 | 0.000416 | 0.000381 | 0.000433 |
| **v4** | 7.59× | 7.94× | **0.355** | 0.000526 | 0.000794 | 0.000506 | 0.000947 |
| **v4.1** | 7.59× | 9.27× | **1.679** | 0.002112 | 0.003896 | 0.002159 | 0.003704 |
| **v5** | 7.59× | 7.98× | **0.390** | 0.001472 | 0.001450 | 0.001464 | 0.001288 |
| **v5.1** | 7.59× | 10.19× | **2.603** | 0.002568 | 0.002249 | 0.002076 | 0.002337 |


### 7. Domain 6: ML Indistinguishability & Privacy
| Model | RF ROC-AUC (Optimal=0.500) ↓ | RF Accuracy (Optimal=50.0%) ↓ | Exact Match Count | Exact Match Rate (%) | Mean Min Hamming (slots) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v3** | **0.5862** | 55.65% | 62 | **0.2067%** | 9.21 |
| **v4** | **0.6323** | 59.25% | 40 | **0.1333%** | 9.49 |
| **v4.1** | **0.7606** | 68.90% | 64 | **0.2133%** | 9.29 |
| **v5** | **0.6702** | 61.95% | 48 | **0.1600%** | 9.14 |
| **v5.1** | **0.7059** | 64.72% | 69 | **0.2300%** | 8.26 |

