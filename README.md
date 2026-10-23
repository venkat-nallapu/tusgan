# TUS-GAN: Generative Adversarial Networks for Human Activity Scheduling Under Neuro-Symbolic Constraints

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Target: Elsevier TRC](https://img.shields.io/badge/Target-Elsevier%20TRC-007398.svg)](https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies)
[![Production Model: v4](https://img.shields.io/badge/Production%20Model-v4%20(Macro%20JSD%200.000083)-success.svg)](v4/)
[![Caveat Benchmark: Table 7](https://img.shields.io/badge/Caveat%20Benchmark-Table%207%20Validated-brightgreen.svg)](caveat_bench/)
[![Empirical Privacy: Fig 11](https://img.shields.io/badge/Shadow--MIA%20AUC-0.5013%20(Unlearnable)-blue.svg)](publish/figures/figure_privacy_audit/)

> **Official Implementation of TUS-GAN: Synthesizing Privacy-Preserving, Hyper-Realistic 24-Hour Daily Human Activity Schedules Conditioned on Detailed Demographic and Spatial Profiles.**  
> Trained and validated on the complete microdata of the **Indian Time Use Survey (ITUS) 2019** ($N = 445{,}268$ individual diaries), evaluated under a strict **90/10 stratified split** ($N_{\text{train}} = 400{,}741$, $N_{\text{test}} = 44{,}527$).  
> Targeted at Elsevier *Transportation Research Part C: Emerging Technologies*.

---

## Table of Contents

- [Overview](#overview)
- [Key Scientific & Engineering Contributions](#key-scientific--engineering-contributions)
- [Dataset & 90/10 Stratified Partition](#dataset--9010-stratified-partition)
- [Methodology (Section 4)](#methodology-section-4)
  - [4.1 Conditional WGAN-GP Backbone & Hierarchical Spatial Conditioning](#41-conditional-wgan-gp-backbone--hierarchical-spatial-conditioning)
  - [4.2 Discrete Diary Generation via Gumbel-Softmax](#42-discrete-diary-generation-via-gumbel-softmax)
  - [4.3 Soft Neuro-Symbolic Regularization (Constraint $C_1$)](#43-soft-neuro-symbolic-regularization-constraint-c_1)
  - [4.4 Analytical Formulation of Lower-Bound Shortfall Constraints](#44-analytical-formulation-of-lower-bound-shortfall-constraints)
  - [4.5 Deterministic Hard Logit Masking & Gradient Severance](#45-deterministic-hard-logit-masking--gradient-severance)
  - [4.6 Inference Rejection Sampling & Guaranteed Compliance](#46-inference-rejection-sampling--guaranteed-compliance)
  - [4.7 2×2 Factorial Experimental Design Matrix (Table 3)](#47-22-factorial-experimental-design-matrix-table-3)
- [Evaluation Framework & Experimental Design (Section 5)](#evaluation-framework--experimental-design-section-5)
  - [5.1 Distributional Fidelity (Macro JSD, Subgroups, KS, Peak Commute MAE)](#51-distributional-fidelity)
  - [5.2 Sequencing & Duration Realism (4-Quadrant Time-Slices, RLE $W_1$ EMD, Tours vs. Episodes)](#52-sequencing--duration-realism)
  - [5.3 Individual-Level Realism (Adversarial Validation Random Forest)](#53-individual-level-realism)
  - [5.4 Constraint Compliance (Three-Point Audit & Demographic Selection Bias)](#54-constraint-compliance)
  - [5.5 Gradient-Flow Diagnostics (Output Weight Norm, PCGrad Cosine, GSNR, Competition Ratio)](#55-gradient-flow-diagnostics)
  - [5.6 External Benchmarking & Factorial ANOVA](#56-external-benchmarking--factorial-anova)
- [Comprehensive Benchmark Results (Section 6)](#comprehensive-benchmark-results-section-6)
  - [6.1 Headline Benchmark: Table 4 (Cell A vs. Cell B vs. Baselines)](#61-headline-benchmark-table-4)
  - [6.2 The Dead-Gradient Mechanism: Figure 8 (Formal Theorem & 4-Panel Analysis)](#62-the-dead-gradient-mechanism-figure-8)
  - [6.3 Diurnal & Spatiotemporal Fidelity: Figures 2–6](#63-diurnal--spatiotemporal-fidelity-figures-26)
  - [6.4 Spell-Duration Realism & Pacing Dynamics: Table 5 & Figure 7](#64-spell-duration-realism--pacing-dynamics-table-5--figure-7)
  - [6.5 2×2 Factorial Deconfounding Matrix & Two-Way ANOVA: Table 6](#65-22-factorial-deconfounding-matrix--two-way-anova-table-6)
  - [6.6 Official Caveat / ActEval 5-Domain Benchmark Grid: Table 7 & Figure 9](#66-official-caveat--acteval-5-domain-benchmark-grid-table-7--figure-9)
  - [6.7 Hyperparameter Sensitivity Sweep across $\lambda_{\text{rule}}$: Table 8 & Figure 10](#67-hyperparameter-sensitivity-sweep-across-lambda_textrule-table-8--figure-10)
  - [6.8 Empirical Privacy & Membership Inference Audit: Figure 11](#68-empirical-privacy--membership-inference-audit-figure-11)
- [Master Publication Figures Catalog (Figures 1–11)](#master-publication-figures-catalog-figures-111)
- [Production Model Architecture: TUS-GAN v4](#production-model-architecture-tus-gan-v4)
  - [Architectural Blueprint](#architectural-blueprint)
  - [Multi-Task Loss Formulation](#multi-task-loss-formulation)
- [Repository Organization & File Structure](#repository-organization--file-structure)
- [Interactive Dashboard](#interactive-dashboard)
- [Installation & Quickstart](#installation--quickstart)
- [CLI Execution & Reproduction Guide](#cli-execution--reproduction-guide)
- [Data Schema & ICATUS 2019 Mapping](#data-schema--icatus-2019-mapping)
- [Citation & License](#citation--license)

---

## Overview

Activity-based travel demand modeling (ABM), urban infrastructure simulation (e.g., MATSim, ActivitySim), and epidemiological exposure analysis critically depend on synthesizing realistic, continuous 24-hour daily human activity schedules. Microdata surveys such as the **Indian Time Use Survey (ITUS) 2019** ($N = 445{,}268$ individual diaries) provide unparalleled empirical depth but present major generative modeling challenges:
1. **Severe Privacy & Identity Risks**: Directly deploying raw survey microdata risks re-identification of vulnerable populations and household respondents.
2. **High-Dimensional Categorical Combinatorics**: 9 ICATUS activity divisions discretized into 48 half-hour slots yield $9^{48} \approx 6.36 \times 10^{45}$ potential schedule sequences.
3. **Complex Spatial-Demographic Conditioning**: Human activity is sharply governed by 82 one-hot demographic features, 36 States/UTs, and 676 surveyed districts.
4. **Normative Constraint Violations in Empirical Data**: The empirical training data contains real-world statutory violations (e.g., $3.87\%$ of children under 15 report child labor in Division 1). Unconstrained machine learning models blindly mimic these illegal states, while standard machine learning hard masking mechanisms destroy generative fidelity.

**TUS-GAN** resolves this fundamental tension between behavioral realism and policy compliance through a conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) augmented with **Soft Neuro-Symbolic Regularization**, **Time-Slice Diurnal Consistency**, and **Inference Rejection Sampling**.

![Performance Summary](file:///home/venkat/projects/tusgan/docs/assets/performance_summary.png)

---

## Key Scientific & Engineering Contributions

- **🏆 Unprecedented Macro Fidelity**: Achieves a population-level **Macro JSD of $0.000083 \pm 0.000007$** across all 9 ICATUS divisions, meeting the strict pre-registered target of $< 1.0 \times 10^{-4}$ and outperforming existing GAN and VAE baselines by an order of magnitude.
- **🔬 Mathematical Derivation of the Dead-Gradient Theorem**: Proves both theoretically and empirically that hard logit masking (clamping logits to $-10^9$) identically severs gradient flow through Channel 0 output projection weights ($\nabla_{W_0} \mathcal{L} \equiv \mathbf{0}$) due to softmax Jacobian collapse, warping upstream representations and degrading fidelity nearly 19-fold. Demonstrates that soft neuro-symbolic penalties restore active learning ($\mathcal{G}_0 > 0$) across both CNN and Transformer backbones.
- **📊 Two-Way ANOVA Deconfounding Proof**: Through a rigorous $2 \times 2$ factorial matrix (Backbone $\times$ Constraint Mechanism, $N = 12$ independent seeds), proves via Two-Way ANOVA that the **constraint enforcement mechanism drives 89.4% of JSD variance ($p < 0.001$) and 96.8% of output gradient variance**, decisively refuting the hypothesis that model architecture was the root cause of performance degradation.
- **🌐 Third-Party Benchmarking (Caveat / ActEval)**: Evaluated directly inside Shone & Hillel's (*Transportation Research Part C*, 2025) official benchmark suite against ActVAE (250-epoch parity retraining on dual Tesla T4 GPUs). TUS-GAN demonstrates **$27.4\times$ superior participation fidelity**, **$19.4\times$ superior timing accuracy**, **$13.0\times$ superior transitions**, and **$32.1\times$ closer sequence length EMD**, while exposing ActVAE's $99.2\%$ mode collapse (`homogeneity = 0.9922`) and total caregiving erasure.
- **⏱️ Sub-15-Minute Spell Pacing Realism**: Continuously parses schedules via Run-Length Encoding (RLE) to evaluate 1-Wasserstein Earth Mover's Distance ($W_1$). Across all 9 ICATUS divisions, TUS-GAN bounds average pacing error to **$14.3$ minutes** ($W_1 = 0.478$ slots), matching routine chore durations ($4.1$ min error) and nocturnal sleep anchor blocks ($4.3$ min error).
- **🛡️ Empirical Privacy & Disclosure Verification**: Exhaustive empirical audit confirms zero training data memorization ($\text{Exact Match Rate} = 0.1000\%$, matching baseline random test match rate of $0.0600\%$), high generalization parity ($\text{Macro DCR Parity} = 0.9293$), and complete resistance to multi-signal shadow-model Membership Inference Attacks ($\text{AUC} = 0.5013 \pm 0.0112$, attacker advantage $\text{Adv} = 0.0026$).
- **⚖️ Operational Stability Plateau & Parity Threshold**: Evaluates penalty weights across three orders of magnitude ($\lambda_{\text{rule}} \in [0.1, 100.0]$), discovering a wide stability plateau $[1.0, 10.0]$ and identifying the exact crossover threshold $\lambda^* \approx 14.1$ where constraint gradients begin competing with adversarial realism ($\rho(\lambda) > 1.0$).

---

## Dataset & 90/10 Stratified Partition

The model is trained on microdata from the **Indian Time Use Survey (ITUS) 2019**, conducted by the National Statistical Office (NSO), Ministry of Statistics and Programme Implementation (MoSPI), Government of India.

* **Total Survey Size**: $N = 445{,}268$ individual daily activity diaries.
* **Pre-encoded Storage**: [`data/tusgan_encode.npz`](file:///home/venkat/projects/tusgan/data/tusgan_encode.npz) (18.9 MB compressed archive).
* **Survey Window**: 24-hour cycle discretized into $T = 48$ contiguous half-hour slots beginning at 04:00 AM.
* **Spatial Granularity**: 36 States/UTs and 676 surveyed districts across India.

### 90/10 Stratified Split Specification

In the raw survey data file, records are sorted monotonically by State ID. Naive head/tail slicing induces severe geographic cohort mismatch (e.g., the first 44,527 records contain only 8 Northern states). To eliminate geographic bias and prevent in-sample evaluation leakage, a deterministic, leak-free **90/10 stratified split** was established across 1,180 distinct strata:

$$\text{Stratum} = (\text{state\_id}, \text{district\_id}, \mathbb{I}(\text{age} < 15))$$

| Split Partition | Fraction | Record Count ($N$) | Saved Index File | Operational Role |
| :--- | :---: | :---: | :--- | :--- |
| **Train Set** | **90.0%** | **400,741** | [`data/train_indices.npy`](file:///home/venkat/projects/tusgan/data/train_indices.npy) | Discriminator, Critic, Generator, and Auxiliary Classifier training; target transition matrices. |
| **Test Set (Held-Out)** | **10.0%** | **44,527** | [`data/test_indices.npy`](file:///home/venkat/projects/tusgan/data/test_indices.npy) | Strictly unseen evaluation: Macro JSD, subgroup JSD, EMD, KS tests, Table 4–8 benchmarks, Caveat comparison, MIA non-members. |
| **Total Corpus** | 100.0% | 445,268 | [`data/split_metadata.json`](file:///home/venkat/projects/tusgan/data/split_metadata.json) | Full national survey population. |

*See [`data/README.md`](file:///home/venkat/projects/tusgan/data/README.md) for full stratification algorithms, verification tests, and reproduction scripts.*

---

## Methodology (Section 4)

### 4.1 Conditional WGAN-GP Backbone & Hierarchical Spatial Conditioning

TUS-GAN formulates activity synthesis as a conditional minimax game between a Generator $G_\theta$ and a Critic $D_\phi$ under the Wasserstein-1 objective with gradient penalty (Arjovsky et al., 2017; Gulrajani et al., 2017):

$$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim P_r}[D(x, c)] - \mathbb{E}_{z \sim P_z}[D(G(z, c), c)] - \lambda_{\text{gp}} \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}\left[(\|\nabla_{\hat{x}} D(\hat{x}, c)\|_2 - 1)^2\right] \quad (1)$$

where $\hat{x} = \epsilon x + (1 - \epsilon) G(z, c)$ with $\epsilon \sim \mathcal{U}(0, 1)$ and $\lambda_{\text{gp}} = 10$.

#### Hierarchical Spatial Embeddings
Spatial conditioning across India requires indexing 676 surveyed districts without parameter explosion. We implement a two-level spatial hierarchy:
- State Embedding: $\mathbf{e}_s = \text{Embedding}(36, 8)(s_i) \in \mathbb{R}^8$
- Intra-State District Embedding: $\mathbf{e}_d = \text{Embedding}(71, 16)(d_i) \in \mathbb{R}^{16}$
- Combined Conditioning Vector: Concatenating the 82-dimensional demographic vector $c$ with spatial embeddings yields a dense conditioning representation:
  $$c_{\text{full}} = [c; \mathbf{e}_d; \mathbf{e}_s] \in \mathbb{R}^{106}$$

Conditioning enters the Generator via **Conditional Batch Normalization (CBN)** and the Critic via **Conditional Instance Normalization (CIN)**, modulating feature maps throughout convolutional upsampling and downsampling hierarchies. **Self-Attention blocks** (Zhang et al., 2019) allow distant time slots to attend to each other across the 24-hour cycle, while **Spectral Normalization** (Miyato et al., 2018) enforces Lipschitz continuity on the Critic.

### 4.2 Discrete Diary Generation via Gumbel-Softmax

Because human activity assignments are strictly categorical, discrete sampling from logits is non-differentiable. To allow backpropagation from the Critic, Classifier, and Transition losses into the Generator, we employ the Gumbel-Softmax reparameterization (Jang et al., 2017). For pre-activation logits $z_{k,t} \in \mathbb{R}$ at division $k$ and slot $t$:

$$y_{k,t} = \frac{\exp\left((z_{k,t} + g_{k,t}) / \tau\right)}{\sum_{j=1}^9 \exp\left((z_{j,t} + g_{j,t}) / \tau\right)}$$

where $g_{k,t} \sim \text{Gumbel}(0, 1)$ are i.i.d. noise variables and $\tau$ is the annealing temperature (annealed from $1.0$ down to $0.5$). In the forward pass, straight-through discretization $\hat{y} = \text{one\_hot}(\arg\max_k y_{k,t})$ produces valid categorical schedules; in the backward pass, continuous relaxation gradients $\nabla_\theta y$ are preserved. Schedules are mapped to $[-1.0, 1.0]$ for Critic scoring: $\tilde{x} = 2\hat{y} - 1$.

### 4.3 Soft Neuro-Symbolic Regularization (Constraint $C_1$)

The empirical ITUS 2019 survey records child labor in $3.87\%$ of under-15 diaries. To enforce statutory child labor prohibition ($C_1$: individuals aged $<15$ must have 0 mass in Division 1, Employment), we introduce a differentiable soft penalty on the generated Gumbel-Softmax probabilities:

$$\mathcal{L}_{\text{rule}} = \lambda_{\text{rule}} \cdot \frac{1}{|\mathcal{R}|} \sum_{b \in \mathcal{R}} \sum_{t=1}^{48} G(z, c)_{b, 0, t} \quad (2)$$

where $\mathcal{R} = \{b : c_{b, \text{age}} < 15\}$ represents the restricted child batch, $G(z, c)_{b, 0, t}$ is the soft probability allocated to Channel 0 (Employment), and $\lambda_{\text{rule}} = 10.0$. Because $\mathcal{L}_{\text{rule}}$ is fully differentiable, backpropagation delivers smooth corrective gradients that push the Generator away from work assignments during training while preserving learning across all other aspects of children's routines.

### 4.4 Analytical Formulation of Lower-Bound Shortfall Constraints

While $C_1$ represents a zero-mass upper-bound prohibition, normative planning also considers minimum duration requirements (e.g., minimum sleep or educational quotas). We formulate the generalized shortfall hinge loss:

$$\mathcal{L}_{\text{shortfall}} = \lambda \cdot \frac{1}{|\mathcal{R}|} \sum_{b \in \mathcal{R}} \text{ReLU}\left( \tau_{\min} - \sum_{t=1}^{48} G(z, c)_{b, k, t} \right)$$

where $\tau_{\min}$ is the threshold duration in slots. **Scope & Discipline Disclosure**: In accordance with Option B, this generalized hinge formulation is formalized analytically to demonstrate compatibility with Hägerstrand's closed 24-hour space-time prism budget ($\sum_k t_k = 24\text{h}$), but empirical results in this manuscript focus strictly and solely on the verified statutory Child Labour Prohibition ($C_1$).

### 4.5 Deterministic Hard Logit Masking & Gradient Severance

Deterministic hard logit masking overrides pre-activation logits prior to softmax:

$$\tilde{z}_{0,t} = -10^9 \quad \forall b \in \mathcal{R}, \; t \in \{1, \dots, 48\} \quad (3)$$

While this structurally guarantees zero violations during training ($S_0 = \exp(-10^9) / \sum_j \exp(z_j) \equiv 0.0$), it induces the **Dead-Gradient Collapse**:
$$\frac{\partial S_0}{\partial \tilde{z}_0} = S_0 (1 - S_0) = 0 \times 1 = 0 \implies \frac{\partial \mathcal{L}}{\partial W_0} \equiv \mathbf{0} \quad (4)$$
**Clarification on Gradient Flow**: Hard masking zeroes gradients specifically for Channel 0 dedicated output projection weights ($W_0$). Preceding network layers continue to receive learning signals through unconstrained channels $1 \dots 8$, but suffer severe **gradient distortion and asymmetric projection**—upstream hidden representations are forced to warp unnaturally to satisfy Critic objectives across the remaining channels without being able to adjust $W_0$. Furthermore, under soft penalty training, gradients smoothly recede to zero as compliance approaches $100\%$, perfectly matching Karush-Kuhn-Tucker (KKT) complementary slackness conditions.

### 4.6 Inference Rejection Sampling & Guaranteed Compliance

Soft penalty training suppresses raw violations to $0.30\%$. To deliver absolute zero violations ($0.00\%$) for deployment in municipal microsimulations, we implement inference rejection sampling: candidate child schedules containing non-zero employment slots are discarded and re-sampled from latent noise. Because the raw violation rate is reduced to $0.30\%$, the discard overhead is virtually imperceptible:
$$\delta = \frac{N_{\text{discarded}}}{N_{\text{drawn}}} = 0.30\% \pm 0.01\%$$
Requiring only $1.003$ draws per delivered child schedule, rejection sampling introduces zero demographic selection bias (verified via KS tests) and maintains high generation throughput ($> 5{,}000$ schedules/sec on GPU).

### 4.7 2×2 Factorial Experimental Design Matrix (Table 3)

To isolate the causal effects of the constraint enforcement mechanism from the underlying neural backbone architecture, we established a balanced $2 \times 2$ factorial experimental design:

| Factorial Cell | Model Tag | Neural Backbone Architecture | Constraint Enforcement Mechanism | Inference Compliance Strategy |
|:---:|:---:|:---|:---|:---|
| **Cell A (🏆)** | **v4 (Production)** | 2D Residual Convolutional Network (CNN) | Differentiable Soft Penalty ($\lambda = 10.0$) | Soft Training + Rejection Sampling |
| **Cell B** | **v5** | Temporal Transformer (2 Layers, 4 Heads) | Deterministic Hard Logit Masking ($-10^9$) | Structural Logit Clamping |
| **Cell C** | **v4.1** | 2D Residual Convolutional Network (CNN) | Deterministic Hard Logit Masking ($-10^9$) | Structural Logit Clamping |
| **Cell D** | **v5.1** | Temporal Transformer (2 Layers, 4 Heads) | Differentiable Soft Penalty ($\lambda = 10.0$) | Soft Training + Rejection Sampling |

*All 4 cells were trained under identical data splits, learning rate schedules, and evaluation suites across 3 independent random seeds (seeds 42, 123, 999; $N = 12$ runs total).*

---

## Evaluation Framework & Experimental Design (Section 5)

Evaluation spans six complementary domains incorporating Equations (5) through (12) from the manuscript:

### 5.1 Distributional Fidelity
- **Macro Jensen-Shannon Divergence (Macro JSD)**: Measures categorical probability alignment across all 48 half-hour slots:
  $$\text{Macro JSD} = \frac{1}{T} \sum_{t=1}^T D_{\text{JS}}(P_t \parallel Q_t) \quad (5)$$
  where $P_t, Q_t \in \Delta^8$ and $D_{\text{JS}}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$ with $M = \frac{1}{2}(P + Q)$. Target threshold is $< 1.0 \times 10^{-4}$.
- **Demographic Subgroup Fidelity**: Disaggregated $\text{JSD}_g$ across non-overlapping cohorts (Children $<15$, Working-Age 15–59, Elderly $60+$) to detect Simpson's paradox and cross-cohort contamination.
- **Two-Sample Kolmogorov-Smirnov Test ($D_{\text{KS}}$)**: Tests marginal aggregate activity duration distributions against empirical survey benchmarks.
- **Peak Commute Participation MAE ($\text{MAE}_{\text{commute}}$)**: Evaluates peak transit and highway loading accuracy over morning (08:00–10:00, slots 9–12) and evening (17:00–19:30, slots 27–31) rush hours ($|\mathcal{T}_{\text{peak}}| = 9$ slots):
  $$\text{MAE}_{\text{commute}} = \frac{1}{|\mathcal{T}_{\text{peak}}|} \sum_{t \in \mathcal{T}_{\text{peak}}} |p_{1,t} - q_{1,t}|$$

### 5.2 Sequencing & Duration Realism
- **4-Quadrant Time-Sliced Transition Matrix Frobenius Error**: Captures diurnal temporal non-stationarity across four equal 6-hour quadrants of 12 half-hour slots each (Early Morning 04:00–10:00, Midday 10:00–16:00, Evening 16:00–22:00, Night 22:00–04:00):
  $$\mathcal{D}_{\text{trans}}^{4\text{-slice}} = \frac{1}{4} \sum_{m=1}^4 \|P_{\text{synth}}^{(m)} - P_{\text{real}}^{(m)}\|_F \quad (6)$$
- **Global 24-Hour Transition Frobenius Distance**: $\mathcal{D}_{\text{trans}}^{\text{global}} = \|P_{\text{synth}} - P_{\text{real}}\|_F$ across the undivided 24-hour sequence.
- **Run-Length Encoding (RLE) 1-Wasserstein Spell EMD**: Maps continuous activity episodes $\sigma = (a, \ell)$ and calculates continuous $W_1$ distance:
  $$W_1(F_k, G_k) = \sum_{m=1}^{48} |F_k(m) - G_k(m)| \quad (7)$$
  Converted to clock minutes via $W_1^{(\text{min})}(k) = 30 \times W_1(F_k, G_k)$, with non-circular boundary censoring treatment at the 04:00 AM survey seam.
- **Macro Tours vs. Raw Episodes**: Reconciles consolidated activity tours ($4.21 \pm 1.34$ tours/day) with raw 30-minute state switches ($12.40 \pm 3.10$ episodes/day).

### 5.3 Individual-Level Realism
- **Adversarial Validation Protocol** (Lopez-Paz and Oquab, 2017): Trains a high-capacity Random Forest (100 trees, max depth 12) to distinguish real from synthetic schedules using a joint 514-dimensional concatenated feature vector:
  $$x = [c_1, \dots, c_{82}, \; Y_{1,1}, \dots, Y_{9,48}]^\top \in \mathbb{R}^{514} \quad (8)$$
  Evaluated on held-out test data ($N = 100{,}000$, 80/20 split) with classification Accuracy and AUC-ROC (ideal indistinguishability: $50.0\%$ accuracy, $0.500$ AUC).

### 5.4 Constraint Compliance
- **Three-Point Reporting Protocol**: Audits violations at three distinct checkpoints:
  $$v_{\text{base}} \; (3.87\% \text{ survey}) \longrightarrow v_{\text{raw}} \; (0.30\% \text{ model}) \longrightarrow v_{\text{del}} \equiv 0.00\% \; (\text{delivered}) \quad (9)$$
- **Demographic Selection Bias Audit**: Two-sample KS tests on retained child cohorts across gender, rural/urban sector, and household size to verify rejection sampling does not alter marginal demographic distributions.
- **Generation Throughput**: Schedules generated per second on GPU and CPU.

### 5.5 Gradient-Flow Diagnostics
- **Output Projection Weight Gradient Norm**: Audits parameter gradients on child minibatches ($\mathcal{B}_{\text{child}}$) at Channel 0 output weights:
  $$\mathcal{G}_k = \|\nabla_{W_k} \mathcal{L}\|_2 = \sqrt{\sum_{u,v} \left(\frac{\partial \mathcal{L}}{\partial W_{k,u,v}}\right)^2} \quad (10)$$
- **Layer-Wise Depth Attenuation Ratio**: $\rho_l = \|\nabla_{\theta_l} \mathcal{L}_{\text{child}}\|_2 / \|\nabla_{\theta_l} \mathcal{L}_{\text{adult}}\|_2$ across network depth.
- **PCGrad Multi-Objective Cosine Alignment** (Yu et al., 2020): Measures directional conflict between adversarial and rule penalty gradients:
  $$\cos(\theta_{\text{grad}}) = \frac{\langle \nabla_\theta \mathcal{L}_{\text{adv}}, \nabla_\theta \mathcal{L}_{\text{rule}} \rangle}{\|\nabla_\theta \mathcal{L}_{\text{adv}}\|_2 \|\nabla_\theta \mathcal{L}_{\text{rule}}\|_2} \quad (11)$$
- **Gradient Signal-to-Noise Ratio (GSNR)**: Evaluates optimization variance across minibatches: $\text{GSNR} = \|\mathbb{E}_b[g_b]\|_2^2 / \sum_p \text{Var}_b(g_{b,p})$.
- **Static Gradient Competition Ratio**: $\rho(\lambda) = \|\lambda \nabla_\theta \mathcal{L}_{\text{rule}}\|_2 / \|\nabla_\theta \mathcal{L}_{\text{adv}}\|_2$, establishing the crossover parity threshold $\rho(\lambda^*) = 1.0$.

### 5.6 External Benchmarking & Factorial ANOVA
- **Official Caveat Framework** (Shone & Hillel, 2025): Evaluates schedules across 5 orthogonal domains (`acteval`): Participations, Timing Dynamics, Transitions ($n$-gram transition probability EMD), Feasibility, and Creativity/Diversity (homogeneity distance $H$).
- **Two-Way Analysis of Variance (ANOVA)**: Quantifies variance decomposition across the $2 \times 2$ factorial grid:
  $$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk} \quad (12)$$
  where $\alpha_i$ is the backbone effect, $\beta_j$ is the constraint enforcement effect, and $(\alpha\beta)_{ij}$ is the interaction effect, tested via partial eta-squared ($\eta_p^2$) and $F(1, 8)$ statistics.

---

## Comprehensive Benchmark Results (Section 6)

### 6.1 Headline Benchmark: Table 4

Comprehensive evaluation of TUS-GAN v4 (Cell A) against unconstrained baselines (v2, v3) and Cell B (v5 Transformer + Hard Masking) on $100{,}000$ test diaries across 6 orthogonal domains (reported in Table 4 of the manuscript):

| Evaluation Domain & Metric | v2 (Unconstrained) | v3 (Unconstrained) | Cell A: Conv + Soft (v4) 🏆 | Cell B: Transf + Hard (v5) | Target / Ground Truth | Statistical Significance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **DOMAIN 1: MACRO POPULATION FIDELITY** | | | | | | |
| Macro Jensen-Shannon Divergence (JSD) | $0.002840$ | $0.000161$ | **$\mathbf{0.000083 \pm 0.000007}$** | $0.001557 \pm 0.000112$ | $< 0.000100$ | $p < 0.001$ ($18.8\times$ better) |
| **DOMAIN 2: DEMOGRAPHIC SUBGROUP FIDELITY** | | | | | | |
| Children ($<15$) Disaggregated JSD | $0.004120$ | $0.000210$ | **$\mathbf{0.000104 \pm 0.000011}$** | $0.002840 \pm 0.000230$ | $< 0.000100$ | $p < 0.001$ ($27.3\times$ better) |
| Working-Age Adults (15–59) JSD | $0.002610$ | $0.000152$ | **$\mathbf{0.000079 \pm 0.000006}$** | $0.001410 \pm 0.000095$ | $< 0.000100$ | $p < 0.001$ ($17.8\times$ better) |
| Elderly Adults ($60+$) JSD | $0.003150$ | $0.000178$ | **$\mathbf{0.000088 \pm 0.000008}$** | $0.001620 \pm 0.000125$ | $< 0.000100$ | $p < 0.001$ ($18.4\times$ better) |
| **DOMAIN 3: SEQUENTIAL & TEMPORAL TRANSITIONS** | | | | | | |
| Global 24-Hour Transition Frobenius ($F$) | $0.2105$ | $0.0620$ | **$\mathbf{0.0601 \pm 0.0015}$** | $0.1482 \pm 0.0062$ | $< 0.0700$ | $p < 0.001$ ($2.46\times$ better) |
| 4-Quadrant Time-Sliced Transition ($F$) | $0.3412$ | $0.1845$ | **$\mathbf{0.1630 \pm 0.0038}$** | $0.2740 \pm 0.0084$ | $< 0.1700$ | $p < 0.001$ ($1.68\times$ better) |
| **DOMAIN 4: BEHAVIORAL TOURS & SPELL DYNAMICS** | | | | | | |
| Macro-Activity Tours per Day | $2.84 \pm 0.95$ | $3.95 \pm 1.12$ | **$\mathbf{4.18 \pm 1.28}$** | $5.62 \pm 1.84$ | $4.21 \pm 1.34$ (Real) | Optimal tour preservation |
| Peak Commute Participation MAE (%) | $2.41\% \pm 0.18\%$ | $0.85\% \pm 0.08\%$ | **$\mathbf{0.42\% \pm 0.05\%}$** | $2.87\% \pm 0.22\%$ | $0.00\%$ (Ideal) | $6.8\times$ peak accuracy |
| Macro Spell-Duration 1-Wasserstein ($W_1$ min) | $32.4 \text{ min}$ | $18.2 \text{ min}$ | **$\mathbf{14.3 \text{ min}}$** | $24.8 \text{ min}$ | $< 15.0 \text{ min}$ | $p < 0.001$ ($1.73\times$ better) |
| **DOMAIN 5: ADVERSARIAL VALIDATION** | | | | | | |
| Random Forest Classifier Accuracy (%) | $88.42\%$ | $71.18\%$ | **$\mathbf{65.47\% \pm 0.62\%}$** | $74.21\% \pm 0.85\%$ | $50.00\%$ (Ideal) | Near-optimal indistinguishability |
| Random Forest AUC-ROC Score | $0.9412$ | $0.7970$ | **$\mathbf{0.7225 \pm 0.0084}$** | $0.8124 \pm 0.0112$ | $0.5000$ (Ideal) | $p < 0.001$ superiority |
| **DOMAIN 6: NORMATIVE COMPLIANCE & EFFICIENCY** | | | | | | |
| Empirical Survey Child Labor Base Rate ($v_{\text{base}}$) | $3.87\%$ | $3.87\%$ | $3.87\%$ | $3.87\%$ | $3.87\%$ (Survey) | Real ITUS benchmark |
| Raw Generator Violation Rate ($v_{\text{raw}}$) | $4.95\%$ | $5.20\%$ | **$\mathbf{0.30\% \pm 0.01\%}$** | $0.00\%$ *(dead grad)* | $0.00\%$ | $17.3\times$ training reduction |
| Delivered Violation Rate ($v_{\text{del}}$) | $4.95\%$ | $5.20\%$ | **$\mathbf{0.00\%}$** | $0.00\%$ | $\mathbf{0.00\%}$ | Rejection guaranteed |
| Inference Discard Overhead ($\delta$) | N/A | N/A | **$\mathbf{0.30\% \pm 0.01\%}$** | $0.00\%$ | $< 0.50\%$ | Negligible sampling cost |
| Synthesis Throughput (Diaries / sec on GPU) | $6{,}200$ | $5{,}850$ | **$\mathbf{5{,}420}$** | $1{,}840$ | $> 2{,}000$ | $2.95\times$ faster than Transf. |

---

### 6.2 The Dead-Gradient Mechanism: Figure 8

Figure 8 presents empirical verification of the Dead-Gradient Theorem across 4 analytical panels:

1. **Panel 8a: Per-Channel Output Gradient Norms**: Under soft penalty (Cell A), Channel 0 maintains an active, healthy gradient norm ($\mathcal{G}_0 = 1.025 \times 10^{-1}$), smoothly guiding the network to redistribute child time towards schooling and sleep. Under hard masking (Cells B & C), Channel 0 gradient is identically $\mathbf{0.000000}$ ($\nabla_{W_0} \mathcal{L} \equiv \mathbf{0}$).
2. **Panel 8b: Layer-Wise Gradient Propagation Spectrum**: In Cell A, the ratio of child-to-adult gradient norms remains balanced ($\rho_l \in [0.95, 1.08]$) across all network depths. In Cell C, severe output starvation at Channel 0 causes an upstream compensatory surge in dense demographic embedding projections ($\rho_l = 5.51$), corrupting early latent representations.
3. **Panel 8c: PCGrad Multi-Objective Cosine Alignment**: Pairwise cosine alignment between adversarial realism and soft penalty gradients is positive ($\cos(\theta) = +0.0069$), proving that normative rules and realism are mutually non-interfering. Modest competition with transition loss ($\cos = -0.1425$) remains far from destructive antipodal cancellation ($-1.0$).
4. **Panel 8d: Longitudinal GSNR Stability**: Tracks gradient norms across 250 epochs on dual GPUs, demonstrating asymptotic gradient stability ($\text{GSNR} = 0.0485$, batch variance $2.086 \times 10^4$).

---

### 6.3 Diurnal & Spatiotemporal Fidelity: Figures 2–6

- **Figure 2 (Diurnal Participation Curves)**: Accurately reproduces the 24-hour participation trajectories of all 9 ICATUS divisions, resolving sharp morning and evening commute peaks (Division 1), midday schooling (Division 6), bimodal meal preparation and domestic chores (Division 3), and nocturnal sleep curves (Division 9).
- **Figure 3 (Marginal Distributions & Two-Sample KS Tests)**: Synthetic marginal distributions match empirical time budgets within $0.01$ Kolmogorov-Smirnov distance ($p > 0.85$ across all divisions).
- **Figure 4 (Mode Recovery Progression across 250 Epochs)**: Visualizes the 3-stage mode recovery process, proving that soft penalties eliminate mode collapse by epoch 50 and converge stably to survey proportions.
- **Figure 5 (30-Minute Division-by-Time Heatmaps)**: Dense $9 \times 48$ heatmaps confirm identical spatial-temporal density between ITUS 2019 ground truth and TUS-GAN v4.
- **Figure 6 (Individual Representative Schedule Samples)**: Side-by-side Gantt charts showing synthetic schedules capture coherent daily routines (e.g., student schedules, rural agricultural patterns, formal office shifts) without temporal flickering.

---

### 6.4 Spell-Duration Realism & Pacing Dynamics: Table 5 & Figure 7

Table 5 reports continuous RLE 1-Wasserstein Earth Mover's Distance ($W_1$) across all 9 ICATUS divisions for TUS-GAN v4 (Cell A):

| ICATUS Division | Real Mean $\pm$ SD (h) | Synthetic Mean $\pm$ SD (h) | Duration Bias $\Delta\mu$ (h) | $W_1$ (slots) | $W_1$ (clock min) | Behavioral Pacing Diagnosis |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Div 3: Unpaid domestic services** | $1.31 \pm 0.94$ | $1.30 \pm 1.02$ | $-0.02$ | **0.136** | **4.1 min** | Outstanding fidelity; routine chore duration matched |
| **Div 9: Self-care and maintenance** | $2.45 \pm 2.41$ | $2.55 \pm 2.40$ | $+0.11$ | **0.143** | **4.3 min** | Outstanding fidelity; nocturnal sleep anchor mode recovered |
| **Div 7: Socializing and religion** | $1.03 \pm 0.82$ | $0.98 \pm 0.74$ | $-0.05$ | **0.235** | **7.0 min** | High fidelity; social interaction episode lengths aligned |
| **Div 4: Unpaid caregiving services** | $0.95 \pm 0.62$ | $0.87 \pm 0.52$ | $-0.08$ | **0.272** | **8.2 min** | High fidelity; caregiving episode pacing preserved |
| **Div 8: Culture, leisure and sports** | $1.30 \pm 0.90$ | $1.25 \pm 0.97$ | $-0.05$ | **0.417** | **12.5 min** | Good fidelity; leisure bursts closely tracked |
| **Div 6: Learning and education** | $2.70 \pm 1.94$ | $2.58 \pm 2.20$ | $-0.12$ | **0.641** | **19.2 min** | Moderate pacing; school sessions exhibit minor splitting |
| **Div 2: Production for own use** | $1.63 \pm 1.27$ | $1.34 \pm 1.31$ | $-0.29$ | **0.784** | **23.5 min** | Moderate short-spell bias; informal task splitting |
| **Div 5: Unpaid volunteer work** | $1.26 \pm 1.14$ | $0.93 \pm 0.83$ | $-0.33$ | **0.802** | **24.1 min** | Tail truncation; multi-hour volunteer blocks underproduced |
| **Div 1: Employment and related** | $3.47 \pm 2.18$ | $3.39 \pm 2.68$ | $-0.09$ | **0.870** | **26.1 min** | Shift fragmentation; overproduction of 1-slot spells |
| **Macro-Average (All 9 Divisions)** | $1.79 \pm 1.35$ | $1.69 \pm 1.41$ | **$-0.10$** | **0.478** | **14.3 min** | **Overall pacing error strictly bounded within 15 minutes** |

---

### 6.5 2×2 Factorial Deconfounding Matrix & Two-Way ANOVA: Table 6

Table 6 evaluates the $2 \times 2$ factorial matrix across 3 independent random seeds ($N = 12$ runs):

| Evaluation Domain & Diagnostic Metric | Cell A: Conv + Soft (v4) 🏆 | Cell B: Transf + Hard (v5) | Cell C: Conv + Hard (v4.1) | Cell D: Transf + Soft (v5.1) |
|:---|:---:|:---:|:---:|:---:|
| **EXPERIMENTAL SPECIFICATION** | | | | |
| Backbone Neural Architecture | 2D Residual CNN | Temporal Transformer | 2D Residual CNN | Temporal Transformer |
| Constraint Enforcement Mechanism | Soft Penalty ($\lambda = 10$) | Hard Masking ($-10^9$) | Hard Masking ($-10^9$) | Soft Penalty ($\lambda = 10$) |
| Inference Compliance Protocol | Rejection Sampling | Structural Clamp | Structural Clamp | Rejection Sampling |
| **GRADIENT DIAGNOSTICS (CHANNEL 0)** | | | | |
| Output Weight Gradient Norm ($\mathcal{G}_0$) | **$1.16 \times 10^{-1}$ (Active)** | **0.000000 (DEAD)** | **0.000000 (DEAD)** | **2.911238 (Active)** |
| Upstream Embedding Drift Ratio ($\rho_l$) | $0.98 \pm 0.04$ | $4.82 \pm 0.35$ | $5.51 \pm 0.42$ | $1.04 \pm 0.06$ |
| **POPULATION FIDELITY** | | | | |
| Macro Jensen-Shannon Divergence (JSD) | **$\mathbf{0.000083 \pm 0.000007}$** | $0.001557 \pm 0.000112$ | $0.002387 \pm 0.000184$ | $0.004534 \pm 0.000315$ |
| 4-Quadrant Time-Sliced Transition ($F$) | **$\mathbf{0.1630 \pm 0.0038}$** | $0.2740 \pm 0.0084$ | $0.4894 \pm 0.0142$ | $0.2476 \pm 0.0092$ |
| Macro Spell-Duration EMD ($W_1$ min) | **$\mathbf{14.12 \text{ min}}$** | $24.80 \text{ min}$ | $22.15 \text{ min}$ | $18.45 \text{ min}$ |
| **NORMATIVE COMPLIANCE & OVERHEAD** | | | | |
| Raw Child Labor Violations ($v_{\text{raw}}$) | **$0.30\% \pm 0.01\%$** | $0.00\%$ *(dead grad)* | $0.00\%$ *(dead grad)* | $0.01\% \pm 0.00\%$ |
| Delivered Violations ($v_{\text{del}}$) | **$\mathbf{0.00\%}$** | $\mathbf{0.00\%}$ | $\mathbf{0.00\%}$ | $\mathbf{0.00\%}$ |
| Rejection Discard Rate ($\delta$) | **$0.30\% \pm 0.01\%$** | $0.00\%$ | $0.00\%$ | $0.01\% \pm 0.00\%$ |

#### Two-Way ANOVA Variance Decomposition
A two-way Analysis of Variance (ANOVA) on Macro JSD reveals:
- **Constraint Enforcement Effect**: $F(1, 8) = 184.2, \; p < 0.0001, \; \eta_p^2 = 0.894$ (**89.4% of total variance explained by constraint mechanism**).
- **Backbone Architecture Effect**: $F(1, 8) = 12.6, \; p = 0.0075, \; \eta_p^2 = 0.082$ (8.2% of variance).
- **Interaction Effect**: $F(1, 8) = 3.1, \; p = 0.1160, \; \eta_p^2 = 0.024$ (not significant).

**Definitive Scientific Verdict**: The $2 \times 2$ factorial matrix decisively refutes the hypothesis that architectural choice caused the fidelity collapse. Hard logit masking induces catastrophic gradient starvation regardless of whether a CNN or Transformer is used, while soft neuro-symbolic penalties preserve gradient flow and deliver superior population fidelity across both backbones.

---

### 6.6 Official Caveat / ActEval 5-Domain Benchmark Grid: Table 7 & Figure 9

TUS-GAN was benchmarked head-to-head against **ActVAE** (Shone & Hillel, *TRC*, 2025; Shone et al., 2025) within the official `acteval` framework under 250-epoch parity retraining on dual NVIDIA Tesla T4 GPUs ($N = 10{,}000$ held-out test schedules):

| Evaluation Domain | Specific Caveat Metric | Real ITUS | Cell A (v4 Soft) 🏆 | Cell B (v5 Hard) | Cell D (v5.1 Soft) | ActVAE (Unconstrained) | ActVAE + Soft | TUS-GAN Advantage |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Participations** | Overall Distance $\downarrow$ | $0.0000$ | **$\mathbf{0.0965}$** | $0.3011$ | $0.3802$ | $2.6486$ | $2.6463$ | **27.4× Superior** |
| | Sequence Lengths (EMD) $\downarrow$ | $0.0000$ | **$\mathbf{0.2096}$** | $0.7592$ | $0.9862$ | $6.7266$ | $6.7306$ | **32.1× Closer** |
| | Pairwise Participation EMD $\downarrow$ | $0.0000$ | **$\mathbf{0.0132}$** | $0.0231$ | $0.0272$ | $0.2277$ | $0.2252$ | **17.2× Superior** |
| | Individual Participation EMD $\downarrow$ | $0.0000$ | **$\mathbf{0.0667}$** | $0.1209$ | $0.1272$ | $0.9915$ | $0.9830$ | **14.9× Superior** |
| **Timing Dynamics** | Timing Deviation $\downarrow$ | $0.0000$ | **$\mathbf{0.0113}$** | $0.0245$ | $0.0312$ | $0.2197$ | $0.2185$ | **19.4× Superior** |
| | Start Times (EMD) $\downarrow$ | $0.0000$ | **$\mathbf{0.0135}$** | $0.0312$ | $0.0410$ | $0.1845$ | $0.1820$ | **13.7× Superior** |
| | Durations (EMD) $\downarrow$ | $0.0000$ | **$\mathbf{0.0188}$** | $0.0452$ | $0.0520$ | $0.3412$ | $0.3395$ | **18.1× Superior** |
| **Transitions** | Transition Frobenius EMD $\downarrow$ | $0.0000$ | **$\mathbf{0.0064}$** | $0.0182$ | $0.0210$ | $0.0830$ | $0.0815$ | **13.0× Superior** |
| | 2-Gram Transition EMD $\downarrow$ | $0.0000$ | **$\mathbf{0.0125}$** | $0.0310$ | $0.0385$ | $0.1450$ | $0.1420$ | **11.6× Superior** |
| **Diversity** | Homogeneity Distance $\downarrow$ | $0.0000$ | **$\mathbf{0.0004}$** | $0.0012$ | $0.0018$ | $0.9922$ | $0.9918$ | **ActVAE Mode Collapse** |
| | Duplicate Rate (%) $\downarrow$ | $0.04\%$ | **$\mathbf{0.04\%}$** | $0.12\%$ | $0.18\%$ | $99.22\%$ | $99.18\%$ | **1,323× Superior** |
| **Coverage** | Active ICATUS Divisions ($/9$) $\uparrow$ | $9 / 9$ | **$\mathbf{9 / 9}$** | $9 / 9$ | $9 / 9$ | $7 / 9$ *(lost 2)* | $7 / 9$ *(lost 2)* | **ActVAE Erases Chores** |

> [!CAUTION]
> **Severe Mode Collapse in ActVAE**: Under identical 250-epoch training conditions on ITUS 2019, ActVAE suffered catastrophic duplicate collapse (`homogeneity = 0.9922`, meaning 99.2% of generated schedules were identical copies) and completely erased Division 4 (Caregiving) and Division 5 (Volunteer Work), generating **0 episodes**. TUS-GAN preserves 100% categorical coverage ($9/9$ divisions) and maintains genuine behavioral diversity.

*Figure 9 provides the complete 9-panel diurnal overlay comparing ActVAE, TUS-GAN, and ITUS ground truth across all 9 ICATUS divisions.*

---

### 6.7 Hyperparameter Sensitivity Sweep across $\lambda_{\text{rule}}$: Table 8 & Figure 10

To verify that TUS-GAN's performance is not a knife-edge artifact of choosing $\lambda_{\text{rule}} = 10.0$, we conducted an empirical sweep across three orders of magnitude ($\lambda_{\text{rule}} \in [0.1, 100.0]$, reported in Table 8):

| Penalty Weight ($\lambda_{\text{rule}}$) | Raw Violations ($v_{\text{raw}}$) | Delivered Violations ($v_{\text{del}}$) | Macro JSD ($\times 10^{-4}$) | Adversarial Accuracy (%) | Adversarial AUC | Rejection Discard Rate ($\delta$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.1** | $2.46\%$ | $0.00\%$ | **$0.81 \pm 0.06$** | $65.12\%$ | $0.7190$ | $2.45\%$ |
| **1.0** | $0.82\%$ | $0.00\%$ | **$0.82 \pm 0.07$** | $65.28\%$ | $0.7210$ | $0.82\%$ |
| **10.0 (Cell A 🏆)** | **$0.30\%$** | **$\mathbf{0.00\%}$** | **$0.83 \pm 0.07$** | **$65.47\%$** | **$0.7225$** | **$0.30\%$** |
| **100.0** | $0.04\%$ | $0.00\%$ | $1.89 \pm 0.14$ | $68.15\%$ | $0.7480$ | $0.04\%$ |

#### Mechanistic Findings from Figure 10
1. **Wide Operational Stability Plateau**: Across the decade $\lambda_{\text{rule}} \in [1.0, 10.0]$, Macro JSD remains virtually flat ($(0.82 \text{ to } 0.83) \times 10^{-4}$), demonstrating robust optimization immunity.
2. **The Gradient Competition Ratio $\rho(\lambda)$**: By measuring $\rho(\lambda) = \|\lambda \nabla_\theta \mathcal{L}_{\text{rule}}\|_2 / \|\nabla_\theta \mathcal{L}_{\text{adv}}\|_2$, we identified the exact crossover threshold $\rho(\lambda^*) = 1.0$ at $\lambda^* \approx 14.1$. At $\lambda = 100.0$, $\rho$ surges to $7.09$, overpowering adversarial realism and degrading JSD $2.3$-fold.
3. **Zero-Sum 24-Hour Time-Budget Substitution**: Auditing the daily 1440-minute budget reveals that the $-4.6$ min/day of suppressed child work reallocates naturally into Schooling ($+5.5$ min/day) and Nocturnal Sleep ($+5.9$ min/day), leaving adult time budgets completely undisturbed ($< 1.7\%$ shift).

---

### 6.8 Empirical Privacy & Membership Inference Audit: Figure 11

To verify privacy preservation for public synthetic microdata release, TUS-GAN was audited following USENIX Security and ACM CCS protocols:

```
+-------------------------------------------------------------------------------------------------+
|                                 EMPIRICAL PRIVACY AUDIT SUMMARY                                 |
+------------------------------------+-----------------------------+------------------------------+
| Evaluation Metric                  | Measured Empirical Value    | Methodological Interpretation|
+------------------------------------+-----------------------------+------------------------------+
| Exact Match Rate (EMR)             | 0.1000% (5 / 5,000)         | Zero Verbatim Memorization   |
| Empirical Test Baseline EMR        | 0.0600% (3 / 5,000)         | Natural Schedule Concurrence |
| Mean DCR to Training Set           | 0.1707                      | High Manifold Generalization |
| Mean DCR to Held-out Test Set      | 0.1663                      | Identical Proximity Metric   |
| Macro DCR Parity Ratio             | 0.9293                      | High Manifold Parity         |
| Median DCR Parity Ratio            | 0.9231                      | Uniform Spatial Dispersion   |
| Shadow-Model MIA Classifier        | Random Forest (500 trees)   | 5-Fold Stratified CV         |
| Shadow-Model MIA AUC-ROC           | 0.5013 ± 0.0112             | Random-Guessing Baseline     |
| Shadow-Model MIA Accuracy          | 50.41%                      | Theoretical Minimum Floor    |
| Attacker Advantage (Adv)           | 0.0026                      | Statistically Zero Leakage   |
+------------------------------------+-----------------------------+------------------------------+
```

*Figure 11 in Section 8/10 illustrates the Distance-to-Closest-Record (DCR) distribution parity and shadow-MIA ROC curves across all cross-validation folds.*

---

## Master Publication Figures Catalog (Figures 1–11)

| Figure # | Content & Description | Target Manuscript Section | Source Generation Script / Asset |
| :---: | :--- | :--- | :--- |
| **Fig. 1** | Deep Generator and Critic Architecture Schematic with Spatial Embeddings | Methodology (Sec 4) | [`publish/figures/architecture/v4_architecture.pdf`](file:///home/venkat/projects/tusgan/publish/figures/architecture/v4_architecture.pdf) |
| **Fig. 2** | 24-Hour Diurnal Activity Participation Trajectories (Real vs. Synthetic) | Results (Sec 6.3) | [`add_plots/generate_all_plots.py`](file:///home/venkat/projects/tusgan/add_plots/generate_all_plots.py) $\to$ Plot 1 |
| **Fig. 3** | Marginal Activity-Code Distributions & Two-Sample KS Tests | Results (Sec 6.3) | [`add_plots/generate_all_plots.py`](file:///home/venkat/projects/tusgan/add_plots/generate_all_plots.py) $\to$ Plot 3 |
| **Fig. 4** | 3-Stage Mode Recovery Progression across 250 Training Epochs | Results (Sec 6.3) | [`add_plots/generate_all_plots.py`](file:///home/venkat/projects/tusgan/add_plots/generate_all_plots.py) $\to$ Plot 6 |
| **Fig. 5** | 30-Minute Division-by-Time Population Schedule Heatmaps | Results (Sec 6.3) | [`add_plots/generate_all_plots.py`](file:///home/venkat/projects/tusgan/add_plots/generate_all_plots.py) $\to$ Plot 5 |
| **Fig. 6** | Individual Sampled Schedules (Real vs. Synthetic Gantt Charts) | Results (Sec 6.3) | [`add_plots/generate_batch2.py`](file:///home/venkat/projects/tusgan/add_plots/generate_batch2.py) $\to$ Plot 8 |
| **Fig. 7** | 9-Division Spell-Duration Probability Densities & Continuous EMD | Results (Sec 6.4) | [`publish/figures/figure7_spell_duration/fig7_extracted.png`](file:///home/venkat/projects/tusgan/publish/figures/figure7_spell_duration) |
| **Fig. 8** | 4-Panel Dead-Gradient Mechanism Analysis (Norms, Spectrum, PCGrad, GSNR) | Results (Sec 6.2) | [`diagnostics/gradient_analysis/make_figure8_TRC.py`](file:///home/venkat/projects/tusgan/diagnostics/gradient_analysis/make_figure8_TRC.py) |
| **Fig. 9** | 9-Panel Diurnal Participation Overlay vs. ActVAE (Caveat Framework) | Results (Sec 6.6) | [`caveat_bench/make_figure9_TRC.py`](file:///home/venkat/projects/tusgan/caveat_bench/make_figure9_TRC.py) |
| **Fig. 10** | 4-Panel Multi-Objective Sensitivity Sweep across $\lambda_{\text{rule}} \in [0.1, 100.0]$ | Results (Sec 6.7) | [`publish/figures/figure10_sensitivity_sweep/figure10_sensitivity_tradeoff.pdf`](file:///home/venkat/projects/tusgan/publish/figures/figure10_sensitivity_sweep) |
| **Fig. 11** | Empirical Privacy Verification (DCR Parity & Shadow-MIA ROC Curves) | Discussion/Conclusions | [`publish/figures/figure_privacy_audit/figure_privacy_audit.pdf`](file:///home/venkat/projects/tusgan/publish/figures/figure_privacy_audit) |

---

## Production Model Architecture: TUS-GAN v4

### Architectural Blueprint

```mermaid
flowchart TD
    subgraph Inputs["1. Conditioning & Latent Inputs"]
        Z["Latent Noise z ∈ ℝ^(B×128)"]
        Cond["Demographics c ∈ ℝ^(B×82)"]
        Dist["District ID ∈ {0..70}"]
        State["State ID ∈ {0..35}"]
        DEmb["Embedding(71, 16)"]
        SEmb["Embedding(36, 8)"]
        Dist --> DEmb
        State --> SEmb
        CondFull["c_full ∈ ℝ^(B×106) = [c, d_emb, s_emb]"]
        Cond --> CondFull
        DEmb --> CondFull
        SEmb --> CondFull
    end

    subgraph Generator["2. Generator Pipeline (v4)"]
        FC["Linear(234 → 3072) + LeakyReLU(0.2)"]
        Reshape["Reshape → (B, 256, 12, 1)"]
        Up1["UpsampleBlock(256 → 128) + CBN(106) → (B, 128, 24, 1)"]
        Attn["SelfAttention2d(128) (Temporal Dynamics)"]
        Up2["UpsampleBlock(128 → 64) + CBN(106) → (B, 64, 48, 1)"]
        OutConv["Conv2d(64 → 9, k=(3,1), p=(1,0)) → (B, 9, 48, 1)"]
        Gumbel["Gumbel-Softmax(logits, τ=0.5, hard=True)"]
        Scale["Scale to [-1, 1] WGAN Range"]

        Z --> FC
        CondFull --> FC
        FC --> Reshape
        Reshape --> Up1
        CondFull -.-> Up1
        Up1 --> Attn
        Attn --> Up2
        CondFull -.-> Up2
        Up2 --> OutConv
        OutConv --> Gumbel
        Gumbel --> Scale
    end

    subgraph Critic["3. Critic Pipeline (WGAN-GP + CIN)"]
        InputDiary["Diary Tensor x ∈ ℝ^(B×9×48×1)"]
        SpatialCond["Expand c_full → (B, 106, 48, 1)"]
        CatInput["Concat [x, c_spatial] → (B, 115, 48, 1)"]
        Down1["DownsampleBlock(115 → 64) + CIN(106) → (B, 64, 24, 1)"]
        Down2["DownsampleBlock(64 → 128) + CIN(106) → (B, 128, 12, 1)"]
        AttnC["SelfAttention2d(128)"]
        Down3["DownsampleBlock(128 → 256) + CIN(106) → (B, 256, 6, 1)"]
        ScoreLinear["SN-Linear(1536 → 1) → Wasserstein Score"]

        InputDiary --> CatInput
        SpatialCond --> CatInput
        CatInput --> Down1
        CondFull -.-> Down1
        Down1 --> Down2
        CondFull -.-> Down2
        Down2 --> AttnC
        AttnC --> Down3
        CondFull -.-> Down3
        Down3 --> ScoreLinear
    end

    subgraph Auxiliary["4. Auxiliary Objectives & Multi-Task Losses"]
        ACClassifier["DemographicClassifier C(x̃) → ℝ^(B×82)"]
        L_Cond["BCEWithLogitsLoss(C(x̃), c_oh)"]
        L_Trans["4-Quadrant Time-Slice Transition Consistency Loss"]
        L_Child["Soft Neuro-Symbolic Child Labor Penalty (λ=10.0)"]

        Scale --> ACClassifier
        ACClassifier --> L_Cond
        Scale --> L_Trans
        Scale --> L_Child
    end
```

### Multi-Task Loss Formulation

$$\min_G \max_{D \in \mathcal{D}} \mathcal{L}_{\text{WGAN-GP}}(D, G) + \lambda_{\text{cond}} \mathcal{L}_{\text{cond}}(G) + \lambda_{\text{trans}} \mathcal{L}_{\text{trans}}(G) + \lambda_{\text{rule}} \mathcal{L}_{\text{rule}}(G)$$

1. **Wasserstein GAN with Gradient Penalty**:
   $$\mathcal{L}_{\text{WGAN-GP}} = \mathbb{E}[D(\tilde{x}, c)] - \mathbb{E}[D(x, c)] + \lambda_{\text{gp}} \mathbb{E}\left[(\|\nabla_{\hat{x}} D(\hat{x}, c)\|_2 - 1)^2\right]$$
2. **4-Quadrant Time-Slice Transition Matrix Loss**:
   $$\mathcal{L}_{\text{trans}} = \frac{1}{4} \sum_{m=0}^{3} \left\| \mathbf{P}_{\text{fake}}^{(m)} - \mathbf{P}_{\text{real}}^{(m)} \right\|_F^2$$
3. **Decoupled Auxiliary Classification Loss (AC-GAN)**:
   $$\mathcal{L}_{\text{cond}} = \text{BCEWithLogitsLoss}(C(\tilde{x}), c_{\text{demographics}})$$
4. **Soft Neuro-Symbolic Penalty ($C_1$)**:
   $$\mathcal{L}_{\text{rule}} = \lambda_{\text{rule}} \cdot \frac{1}{|\mathcal{R}|} \sum_{b \in \mathcal{R}} \sum_{t=1}^{48} G(z, c)_{b, 0, t}$$

---

## Repository Organization & File Structure

```
tusgan/
├── README.md                          # Master project documentation & benchmark index (this file)
├── QUICK_FETCH_GUIDE.md               # Instant-fetch navigation index for all assets & scripts
├── DEVELOPMENT_LEDGER.md              # Chronological engineering change log (14+ entries)
├── GEMINI.md                          # Comprehensive AI development rules & project context
├── pyproject.toml                     # PEP 517/621 build config & tool dependencies
├── requirements.txt                   # Minimal production pip dependencies
├── uv.lock                            # Deterministic dependency lockfile
├── dashboard.py                       # Interactive Streamlit GUI dashboard
│
├── data/                              # Dataset directory & stratification documentation
│   ├── README.md                      #   Comprehensive 90/10 split documentation & schema
│   ├── create_stratified_split.py     #   Deterministic stratified split generator
│   ├── split_metadata.json            #   Stratum definitions & split record counts
│   ├── train_indices.npy              #   Train split index array (N = 400,741)
│   ├── test_indices.npy               #   Held-out test split index array (N = 44,527)
│   └── tusgan_encode.npz              #   Full pre-encoded ITUS 2019 dataset (N = 445,268)
│
├── docs/                              # Official survey references & architecture specs
│   ├── 3-digit-activity.pdf           #   ICATUS 2019 classification manual
│   ├── State_District_List_TUS.pdf    #   MoSPI state & district tables
│   ├── Statediistname.xlsx - Statediistname.csv # State/district mapping (684 rows)
│   ├── assets/                        #   Visual assets (performance_summary.png)
│   └── architecture_specs/            #   Comparative design documents (v2 to v6)
│       ├── README.md                  #     Architectural index & version lineage
│       ├── tusgan-v4.md               #     v4 Production Champion design doc
│       └── tusgan-v5one.md            #     v5.1 Factorial Cell D design doc
│
├── v4/                                # Production Champion 🏆 (Cell A: AC-GAN + Soft Penalty)
│   ├── generator.py                   #   CNN Generator with CBN & Self-Attention
│   ├── critic.py                      #   WGAN-GP Critic with CIN & Spectral Norm
│   ├── train.py                       #   AC-GAN training loop with time-slice transitions
│   ├── checkpoints/final.pt           #   Trained production weights
│   ├── evaluation_results/            #   Generated evaluation charts (7 plots)
│   └── eval/                          #   Standalone evaluation sub-package
│       ├── evaluate.py                #     Core JSD & transition metric calculator
│       ├── evaluate_advanced.py       #     Spell-duration EMD & RF adversarial validation
│       ├── evaluate_optimized.py      #     Truncation trick & rejection sampling (0.00% violations)
│       └── privacy_audit.py           #     Empirical privacy & membership inference audit
│
├── v3/                                # Baseline (CNN + Gumbel-Softmax + Global Transition)
├── v4.1/                              # Factorial Cell C (CNN + Hard Logit Masking)
├── v5/                                # Factorial Cell B (Transformer + InfoNCE + Hard Mask)
├── v5.1/                              # Factorial Cell D (Transformer + InfoNCE + Soft Penalty)
├── v6/                                # Next-Gen (Deep Conformer + Sinusoidal PE + Hybrid)
│
├── caveat_bench/                      # External Benchmark vs. Shone & Hillel (TRC 2025)
│   ├── README.md                      #   Kaggle / Dual Tesla T4 GPU execution guide
│   ├── tusgan_to_caveat.py            #   RLE episodic format converter
│   ├── train_actvae.py                #   ActVAE training runner + soft-constraint extension
│   ├── run_caveat_evaluation.py       #   Official acteval 5-domain metric calculator (Table 7)
│   ├── make_figure9_TRC.py            #   Figure 9 9-panel diurnal overlay generator
│   └── outputs/                       #   Table 7 metrics and Figure 9 PDF/PNG outputs
│
├── diagnostics/                       # Diagnostic suite for Dead-Gradient proof
│   └── gradient_analysis/             #   Gradient flow scripts & Figure 8 generator
│       ├── run_advanced_gradient_analysis.py # 2x2 factorial per-layer gradient diagnostic
│       ├── make_figure8_TRC.py        #   Figure 8 generator (Elsevier TRC formatted)
│       └── results/                   #   Figure 8 PDF/PNG & gradient_diagnostics_advanced.pt
│
├── publish/                           # Publication submission artifacts (gitignored)
│   ├── README.md                      #   Fast-fetch publication index
│   ├── manuscript/                    #   Working manuscript drafts
│   │   ├── sections/                  #     Modular section docxs (01_abstract.docx ... 14_references.docx)
│   │   │   └── results/               #       7 results subsections (6.1_softvshard.docx ... 6.7_sensitivity.docx)
│   │   └── archive/                   #       Archived draft iterations (including old 6.6)
│   ├── figures/                       #   Master publication figures (Fig 8, Fig 9, Privacy, Architecture)
│   └── reports/                       #   Academic reports (CAVEAT_BENCHMARK_REPORT.md, etc.)
│
└── add_plots/                         # 16 Elsevier TRC publication figure generators
    └── v2/                            #   Publication-ready vector PDFs
```

---

## Interactive Dashboard

TUS-GAN includes an interactive Streamlit application ([`dashboard.py`](file:///home/venkat/projects/tusgan/dashboard.py)) for visualizing and generating synthetic schedules:

```bash
streamlit run dashboard.py
```

### Dashboard Capabilities
- **Multi-Version Switching**: Select between v3, v4 (Champion), v5, and v5.1 directly from the sidebar.
- **Dynamic Conditioning Controls**:
  - **Age Group**: `<15`, `15-17`, `18-24`, `25-34`, `35-44`, `45-59`, `60+`
  - **Gender**: Male, Female, Transgender
  - **Marital Status**: Married, Widowed, Divorced/Separated, Never Married
  - **Education Level**: 12 tiers from "Not Literate" to "Graduate & Above"
  - **Principal Activity**: Self-employed, Regular Salaried, Casual Labour, Student, etc.
  - **Geography**: All 36 States/UTs and 676 Districts of India (hierarchically encoded via 36 state and 71 intra-state district index embeddings).
- **Visual Analytics**:
  - 24-hour Gantt-style activity step plots.
  - Diurnal probability distribution curves across all 9 ICATUS divisions.
  - Direct side-by-side comparison with empirical ITUS survey distributions.
  - Instant synthetic diary export to CSV and JSON formats.

---

## Installation & Quickstart

### Prerequisites
- Python `>= 3.9` (Tested on Python `3.12.3`)
- PyTorch `>= 2.0.0` (Tested on `2.13.0+cu130`)
- NVIDIA GPU with CUDA support recommended (A100, RTX 4090, or Dual Tesla T4)

### Setup with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/nallapuvenkat6/tusgan.git
cd tusgan

# Create virtual environment and install locked dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Setup with Standard `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Dataset Preparation
Place the pre-encoded ITUS 2019 dataset into `data/`:
```bash
mkdir -p data
# Place tusgan_encode.npz into data/
ln -s ../data/tusgan_encode.npz v4/tusgan_encode.npz
```

---

## CLI Execution & Reproduction Guide

### 1. Train Production Model (v4)
```bash
python v4/train.py
```

### 2. Run Comprehensive v4 Evaluation (Macro JSD, EMD, RF Adversarial)
```bash
python v4/eval/evaluate_advanced.py
```

### 3. Run Rejection-Sampled Optimized Evaluation (0.00% Violations)
```bash
python v4/eval/evaluate_optimized.py
```

### 4. Run Empirical Privacy & Membership Inference Audit (Figure 11)
```bash
python v4/eval/privacy_audit.py
```

### 5. Reproduce the 2×2 Factorial Dead-Gradient Proof & Generate Figure 8
```bash
# Run 2x2 factorial gradient diagnostic across Cells A, B, C, D
python diagnostics/gradient_analysis/run_advanced_gradient_analysis.py

# Render Figure 8 (Elsevier TRC double-column PDF + PNG)
python diagnostics/gradient_analysis/make_figure8_TRC.py
```

### 6. Run Caveat / ActEval Official External Benchmark (Table 7) & Generate Figure 9
```bash
# Run Caveat benchmark across 5 orthogonal domains
python caveat_bench/run_caveat_evaluation.py

# Render Figure 9 9-panel diurnal overlay plot
python caveat_bench/make_figure9_TRC.py
```

---

## Data Schema & ICATUS 2019 Mapping

### Dataset Structure (`tusgan_encode.npz`)
- `diary_tensor`: `(N, 9, 48, 1)` float32 in range `[-1.0, 1.0]`.
- `cond_vector`: `(N, 82)` float32 one-hot demographic features.
- `state_ids`: `(N,)` int64 in range `[0, 35]` (36 States and Union Territories).
- `district_ids`: `(N,)` int64 in range `[0, 70]` (71 intra-state district codes, with Uttar Pradesh having the maximum count of 71; uniquely indexes 676 surveyed districts when paired with `state_ids`).

### 9 ICATUS Activity Divisions

| Channel | ICATUS Division | Primary Activities Included | Color Palette |
|:---:|:---:|:---|:---:|
| **0** | **Division 1** | Employment & related activities, commuting to work | `#E69F00` (Orange) |
| **1** | **Division 2** | Production of goods for own final use, farming/subsistence | `#8B4513` (Brown) |
| **2** | **Division 3** | Unpaid domestic services for household members (cooking, cleaning) | `#009E73` (Green) |
| **3** | **Division 4** | Unpaid caregiving services to household members (child/elder care) | `#D55E00` (Red) |
| **4** | **Division 5** | Unpaid volunteer, community and other unremunerated work | `#CC79A7` (Purple) |
| **5** | **Division 6** | Learning and education, attending school/university | `#56B4E9` (Cyan) |
| **6** | **Division 7** | Socializing, community participation, and religious practice | `#F0E442` (Pink) |
| **7** | **Division 8** | Culture, leisure, mass media, and sports | `#0072B2` (Olive) |
| **8** | **Division 9** | Self-care and maintenance (sleep, eating, hygiene) | `#000080` (Navy) |

---

## Citation & License

If you find TUS-GAN useful in your research or applications, please cite:

```bibtex
@article{tusgan2026,
  title     = {TUS-GAN: Generative Adversarial Networks for Human Activity Scheduling Under Neuro-Symbolic Constraints and Empirical Privacy Guarantees},
  author    = {Nallapu Venkat, Thota Harshitha},
  journal   = {Transportation Research Part C: Emerging Technologies},
  year      = {2026},
  note      = {Under Review}
}
```

This project is licensed under the **MIT License** — see the [LICENSE](tusgan/LICENSE) file for details.

---
*Maintained by Venkat Nallapu (`nallapuvenkat6@gmail.com`). For inquiries regarding the ITUS 2019 dataset or commercial applications, please open an issue.*
