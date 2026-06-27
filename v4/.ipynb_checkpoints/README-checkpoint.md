# TUS-GAN v4: Advanced Sequence Modeling & Neuro-Symbolic Activity Synthesis

Welcome to the **TUS-GAN v4** repository. This project implements a conditional Generative Adversarial Network (GAN) designed to synthesize highly realistic 24-hour time-use diaries based on conditional demographics (Age, Gender, Education, Marital Status, principal activities, and geographical identifiers like State and District) using the **Indian Time Use Survey (ITUS) 2019** dataset.

This README provides a comprehensive review of the **TUS-GAN v3** performance analysis and details the **v4** technical upgrades engineered to push generative fidelity to near-indistinguishability.

---

## 📊 1. TUS-GAN v3: Performance Analysis

TUS-GAN v3 made substantial progress by introducing **Gumbel-Softmax** categorical discretization and a **Global Temporal Consistency Loss**. Below are the metrics achieved after training v3 for **350 epochs** with a batch size of 1024 on an NVIDIA A100:

### Key Metrics Table

| Metric | Achieved Value (v3) | Ideal Target | Status / Interpretation |
| :--- | :---: | :---: | :--- |
| **Jensen-Shannon Divergence (JSD)** | **`0.000161`** | `0.00000` | 🟢 Excellent: Population-level activity distributions match nearly perfectly. |
| **Transition Matrix Frobenius Norm (F-Norm)** | **`0.061843`** | `0.00000` | 🟢 Excellent: Sequence transition rules match overall average daily flow. |
| **Adversarial Validation Accuracy** | **`72.75%`** | `50.00%` | 🟡 Moderate: Random Forest classifier still distinguishes synthetic data from real. |
| **Adversarial Validation AUC-ROC** | **`0.7970`** | `0.5000` | 🟡 Moderate: Joint probability representation has minor high-dimensional leaks. |

---

### Activity Spell-Duration Earth Mover's Distance (EMD)

Wasserstein (EMD) distances measure how closely the continuous block lengths (spells of consecutive 30-min intervals) match the real behavior:

*   **Sleep / Personal Care:** `0.3192` (Very clean, realistic blocks)
*   **Study / Education:** `0.1217` (High alignment)
*   **Social & Leisure:** `0.4252` (Good shape representation)
*   **Employment / Job:** `0.4863` (Highly structured)
*   **Caregiving (Family) / Travel:** `0.6375` (Sporadic/bursty mismatch)

---

### 🛑 Identified Drawbacks in TUS-GAN v3

Despite massive improvements over v2, deep statistical evaluation identified three remaining drawbacks:
1.  **High-Dimensional Condition Leakage (AUC-ROC = 0.7970)**
    *   *The Leak:* The Random Forest classifier exploits minor logical or demographic inconsistencies in 432-dimensional space (e.g., matching a child's demographics to an adult's commute pattern). This happens because conditioning inputs concatenated to noise can get diluted in the deep layers of the Generator.
2.  **Time-Agnostic Transition Loss**
    *   *The Leak:* The transition matrix regularizer in v3 was **global** (averaged over the full 24 hours). The model missed time-of-day specific transition dynamics (e.g., morning commutes vs. night travel, or child-care clustering), causing higher EMD on bursty activities like Travel and Caregiving.
3.  **Absence of Logical Guardrails**
    *   *The Leak:* The model relied purely on distribution modeling. Because it lacks physical/logical rules, it occasionally generated impossible combinations, such as children under 15 working full-time shifts.

---

## 🚀 2. TUS-GAN v4: Technical Upgrades

To resolve these leaks, **TUS-GAN v4** introduces three key architectural upgrades:

```mermaid
graph TD
    Z[Latent Noise (128)] & Cond[Demographics (82)] --> G[Conditional Generator]
    G --> |Fake Diaries (B, 9, 48, 1)| D[Conditional Critic]
    G --> |Fake Diaries (B, 9, 48, 1)| C[Decoupled Demographic Classifier]
    G --> |Softmax Probabilities| LossLogic[Neuro-Symbolic Constraint Loss]
    
    D --> |Adversarial Loss| LossG[Total Generator Loss]
    C --> |AC-GAN Loss| LossG
    LossLogic --> |Child Labor Penalty| LossG
    
    Real[Real Diaries (B, 9, 48, 1)] --> D
    Real --> C
```

### 1. Decoupled Demographic Classifier (AC-GAN Loss)
Rather than coupling classification directly inside the Critic (which can cause instability in WGAN-GP), v4 trains an independent, decoupled classifier network $C$ on real diaries to predict demographics from activity patterns.
*   **The Classifier Loss:**
    $$\mathcal{L}_{C} = \text{BCEWithLogitsLoss}(C(x_{\text{real}}), c)$$
*   **The Generator Loss Component:**
    $$\mathcal{L}_{\text{cond}} = \text{BCEWithLogitsLoss}(C(x_{\text{fake}}), c)$$
*   **Why it works:** The Generator is penalized if it generates diaries that do not clearly print the target demographics into the sequence. This eliminates condition dilution and drives the validation AUC closer to the ideal `0.50`.

### 2. Time-Slice Transition Loss
Instead of one global $9 \times 9$ transition matrix, v4 splits the 48 time slots (24 hours) into **4 distinct 6-hour segments** (Morning, Afternoon, Evening, Night).
*   For each slice $k \in \{0, 1, 2, 3\}$, transition matrices $P_{\text{real}, k}$ are precomputed from the dataset.
*   Runtime transition matrices $P_{\text{fake}, k}$ are calculated and compared using:
    $$\mathcal{L}_{\text{transition\_slices}} = \frac{1}{4} \sum_{k=0}^{3} \| P_{\text{fake}, k} - P_{\text{real}, k} \|_F^2$$
*   **Why it works:** Forces the model to learn time-of-day constraints, ensuring commute behaviors and caregiving blocks are generated in realistic time windows.

### 3. Neuro-Symbolic Logical Constraint Loss
Inserts prior domain knowledge directly into backpropagation. We define a differentiable penalty targeting logical violations.
*   **Child Labor Constraint:** Prevent individuals under 15 ($c_0 = 1$) from working ($x_{\text{Employment}}$, channel 0).
*   **Mathematical formulation:**
    $$\mathcal{L}_{\text{logic}} = \text{mean}\big(\mathbb{I}_{c_0=1} \cdot \sum_{t=1}^{48} p_{\text{Employment}, t}\big)$$
*   **Why it works:** Guarantees near-zero logical violations in synthetic diaries.

---

## 🏗️ 3. Directory Layout & Isolated Executions

To ensure development hygiene, v4 operations are fully isolated in the `v4/` directory:

```text
v4/
├── checkpoints/              # Saves model checkpoints (G, D, C, optimizers)
├── samples/                  # Saves intermediate synthetic NPY samples
├── runs/                     # TensorBoard logging events
├── evaluation_results/       # Pre-computed evaluation charts
├── critic.py                 # Conditional Critic with CIN
├── generator.py              # Conditional Generator with CBN & Gumbel
├── train.py                  # Upgraded WGAN-GP training loop with AC-GAN & Logic
├── evaluate.py               # Statistical evaluation pipeline
├── evaluate_advanced.py      # Spell EMD, Validation Classifier & Logic Constraint checks
└── tusgan_encode.npz         # Local copy of encoded ITUS 2019 dataset
```

---

## 💻 4. Running the Pipelines

Activate your Python environment (`.venv-v3`) and run commands from the project root:

### 1. Model Training
```bash
python3 v4/train.py \
    --data v4/tusgan_encode.npz \
    --epochs 250 \
    --batch 512 \
    --lr 0.0001 \
    --lambda_transition 10.0 \
    --lambda_cond 1.0 \
    --lambda_logic 10.0
```

### 2. Advanced Evaluation
Computes Wasserstein spell EMDs, logical constraint compliance rates, and trains the Random Forest classifier:
```bash
python3 v4/evaluate_advanced.py \
    --checkpoint v4/checkpoints/final.pt \
    --data v4/tusgan_encode.npz \
    --n-samples 10000 \
    --output-dir v4/evaluation_results
```

### 3. Interactive Dashboard
Explore the results of both v3 and v4 models dynamically:
```bash
streamlit run dashboard.py
```
*(Select the model version dropdown in the sidebar to toggle between **TUS-GAN v3** and **TUS-GAN v4** weights dynamically)*.
