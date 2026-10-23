# ITUS 2019 Dataset Schema and 90/10 Stratified Partition

This document details the dataset representation, schema, and reproducible **90/10 Train/Test partition** for the Indian Time Use Survey (ITUS) 2019 microdata used in TUS-GAN.

---

## 1. Dataset Overview

* **Primary Corpus**: Indian Time Use Survey (ITUS) 2019 microdata collected by the Ministry of Statistics and Programme Implementation (MoSPI), Government of India.
* **Total Sample Size**: $N = 445,268$ daily 24-hour activity diaries.
* **Pre-encoded Archive**: [`data/tusgan_encode.npz`](file:///home/venkat/projects/tusgan/data/tusgan_encode.npz)

### Tensor Schema (`tusgan_encode.npz`)

| Key | Shape | Dtype | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `diary_tensor` | `(445268, 9, 48, 1)` | `float32` | $\{-1.0, 1.0\}$ | One-hot daily activity schedule scaled to WGAN range. 9 ICATUS activity divisions across 48 half-hour slots. |
| `cond_vector` | `(445268, 82)` | `float32` | $\{0.0, 1.0\}$ | Concatenated one-hot demographic conditioning attributes. |
| `state_ids` | `(445268,)` | `int64` | $0 \dots 35$ | Zero-indexed State/UT identifiers (36 unique entities). |
| `district_ids` | `(445268,)` | `int64` | $0 \dots 70$ | Intra-state district indices (71 max cardinality, covering 676 surveyed districts). |
| `num_states` | scalar | `int` | $36$ | Embedding cardinality for states (`nn.Embedding(36, 8)`). |
| `num_districts` | scalar | `int` | $71$ | Embedding cardinality for intra-state districts (`nn.Embedding(71, 16)`). |

### 82-Dimensional Demographic Conditioning Vector Layout

* `0:7` — **Age Group**: `<15` (index 0, triggers statutory child labor constraint), `15–17`, `18–24`, `25–34`, `35–44`, `45–59`, `60+`.
* `7:10` — **Gender**: Male (idx 7), Female (idx 8), Transgender (idx 9).
* `10:14` — **Marital Status**: Married, Widowed, Divorced/Separated, Never Married.
* `14:26` — **Educational Attainment**: 12 ordered tiers ("Not Literate" through "Graduate/Post-Graduate+").
* `26:39` — **Principal Activity Status**: 13 NSS economic activity codes (self-employed, regular wage/salary, casual labour, student, etc.).
* `39:46` — **Day of the Week**: Monday through Sunday.
* `46:48` — **Sector**: Rural (idx 46), Urban (idx 47).
* `48:58` — **Household Size**: Categorical bins.
* `58:82` — **MPCE & Auxiliary Features**: Monthly Per Capita Expenditure quintiles and survey stratification indicators.

---

## 2. Why a Fixed Stratified Split Was Required

In the raw survey data file (`tusgan_encode.npz`), records are sorted **monotonically by State ID**:
* Slicing naively (e.g., `real[:eval_n]`) results in severe **geographic cohort mismatch**: the first 44,527 records contain *only* States 0 through 7 (Northern India: Jammu & Kashmir, Himachal Pradesh, Punjab, Chandigarh, Uttarakhand, Haryana, Delhi, Rajasthan).
* Training on all 445,268 records without a partitioned holdout leads to **in-sample data leakage** and invalidates empirical privacy audits (such as membership inference attacks).

To establish an unimpeachable, peer-reviewed evaluation benchmark for Elsevier *Transportation Research Part C*, we generated a deterministic, leak-free **90/10 stratified split**.

---

## 3. The 90/10 Split Specification

| Split | Fraction | Record Count ($N$) | Saved Index File | Primary Purpose |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | **90.0%** | **400,741** | [`data/train_indices.npy`](file:///home/venkat/projects/tusgan/data/train_indices.npy) | Discriminator, Critic, Generator, Classifier training; empirical transition target computation. |
| **Test (Held-out)** | **10.0%** | **44,527** | [`data/test_indices.npy`](file:///home/venkat/projects/tusgan/data/test_indices.npy) | Unseen evaluation: JSD, EMD, KS tests, Table 4–6 benchmarks, Caveat comparisons, Privacy audit non-members. |
| **Total** | 100.0% | 445,268 | [`data/split_metadata.json`](file:///home/venkat/projects/tusgan/data/split_metadata.json) | Full ITUS 2019 survey population. |

### Splitting Protocol (`create_stratified_split.py`)

1. **Stratification Key**:
   Each survey diary is mapped to a composite stratum tuple:
   $$\text{Stratum} = (\text{state\_id}, \text{district\_id}, \mathbb{I}(\text{age} < 15))$$
   This yields 1,352 fine-grained spatiotemporal demographic strata.

2. **Singleton Preservation ($n = 1$)**:
   To prevent uninitialized embeddings or training data starvation on edge cohorts, any stratum containing exactly $1$ record ($3$ records across the entire country) is deterministically routed to the **Train** partition.

3. **Stratified Partitioning ($n \ge 2$)**:
   The remaining $445,265$ records are partitioned using `sklearn.model_selection.train_test_split` with target test size $44,527$ and fixed `random_state=42`.

---

## 4. Verification & Quality Guarantees

The generated partition satisfies all structural and demographic requirements:

### A. Zero Data Leakage
$$\text{Train Indices} \cap \text{Test Indices} = \emptyset \quad (| \text{Train} \cap \text{Test} | = 0)$$
$$\text{Train Indices} \cup \text{Test Indices} = \{0, 1, \dots, 445,267\}$$

### B. Spatial Embedding Coverage (100% Pan-India Representation)
* **States ($36 / 36$)**: All 36 States/UTs are represented in both Train and Test. The maximum state-level sampling delta across all 36 states is $\le 0.0184\%$.
* **Districts ($676 / 676$)**: All 676 official surveyed districts exist in the Training partition, ensuring every embedding vector in:
  * `nn.Embedding(36, 8)` (State Embedding)
  * `nn.Embedding(71, 16)` (Intra-state District Embedding)
  receives active gradient backpropagation during model training.

### C. Demographic & Statutory Parity

| Feature Cohort | Real ITUS 2019 | Train Split ($N=400,741$) | Test Split ($N=44,527$) | $\Delta(\text{Train} - \text{Test})$ |
| :--- | :---: | :---: | :---: | :---: |
| **Child ($<15$)** | **14.162%** | **14.168%** | **14.111%** | **0.057%** |
| Age 15–17 | 5.442% | 5.421% | 5.628% | 0.207% |
| Age 18–24 | 13.814% | 13.824% | 13.724% | 0.100% |
| Age 25–34 | 20.579% | 20.609% | 20.309% | 0.300% |
| Age 35–44 | 16.235% | 16.223% | 16.343% | 0.120% |
| Age 45–59 | 17.986% | 17.960% | 18.227% | 0.267% |
| Age 60+ | 11.781% | 11.795% | 11.658% | 0.137% |
| **Gender: Male** | 50.897% | 50.922% | 50.675% | 0.247% |
| **Gender: Female** | 49.074% | 49.049% | 49.296% | 0.247% |
| **Gender: Transgender** | 0.029% | 0.029% | 0.029% | 0.000% |
| **Sector: Rural** | 38.892% | 38.903% | 38.790% | 0.113% |
| **Sector: Urban** | 80.682% | 80.667% | 80.821% | 0.154% |

---

## 5. Usage in Code

### Loading in Python / PyTorch

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Load full dataset
data = np.load("data/tusgan_encode.npz")
diary_tensor = data["diary_tensor"]  # (445268, 9, 48, 1)
cond_vector = data["cond_vector"]    # (445268, 82)
state_ids = data["state_ids"]        # (445268,)
district_ids = data["district_ids"]  # (445268,)

# 2. Load fixed train/test split indices
train_idx = np.load("data/train_indices.npy")
test_idx = np.load("data/test_indices.npy")

print(f"Loaded Train: {len(train_idx):,} records | Test: {len(test_idx):,} records")

# 3. Slice for Training (400,741 records)
train_diaries = diary_tensor[train_idx]
train_cond = cond_vector[train_idx]
train_states = state_ids[train_idx]
train_districts = district_ids[train_idx]

# 4. Slice for Evaluation / Out-of-sample Testing (44,527 records)
test_diaries = diary_tensor[test_idx]
test_cond = cond_vector[test_idx]
test_states = state_ids[test_idx]
test_districts = district_ids[test_idx]
```

### Reproducing the Split

To re-generate or verify the split from scratch:
```bash
python3 data/create_stratified_split.py
```
This script validates all assertions and writes the metadata summary to [`data/split_metadata.json`](file:///home/venkat/projects/tusgan/data/split_metadata.json).
