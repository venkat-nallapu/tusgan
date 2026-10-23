# TUS-GAN v4.1 (Cell C: CNN + Hard Masking) — Kaggle Training Guide

> **Hardware**: Kaggle Dual Tesla T4 GPUs (`GPU T4 x2`)  
> **Training**: 250 Epochs, Batch Size 512, DDP via `torchrun`  
> **Target**: Table 6 (Cell C: Convolutional Backbone + Hard Masking) & Figure 8 (Dead-Gradient Validation)

---

## 📦 Package Contents

| File | Purpose |
|:-----|:--------|
| `v4_1_training_kaggle.ipynb` | Interactive Jupyter Notebook ready for Kaggle |
| `train.py` | Training script with DDP multi-GPU, WGAN-GP, and gradient logging |
| `generator.py` | Transposed Conv generator with deterministic hard logit masking (`-1e9`) |
| `critic.py` | Strided Conv critic with Spectral Norm & Conditional InstanceNorm |
| `smoke_test.py` | Pre-flight verification + empirical dead-gradient confirmation |
| `evaluate_ultimate.py` | Comprehensive Table 6 evaluation suite & diagnostic figure generator |
| `run_kaggle.sh` | 1-click end-to-end bash runner |
| `requirements.txt` | Dependency list |
| `tusgan_encode.npz` | Pre-encoded ITUS 2019 dataset (445,268 diaries) |

---

## 🚀 Step-by-Step Kaggle Execution Pipeline

### Step 1: Create a Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com/) and click **"+ Create"** → **"New Notebook"**.
2. In the right-hand panel under **Notebook settings**:
   - **Accelerator**: Select **`GPU T4 x2`** (Dual Tesla T4).
   - **Persistence**: Select **`Variables and Files`** (so your files remain if you reload).
   - **Internet**: Ensure **`Internet on`** is toggled on.

### Step 2: Upload Files to Kaggle
You have two easy options:

#### Option A: Direct Notebook Import (Fastest & Recommended)
1. In the top menu of Kaggle, click **File** → **Upload notebook** → select `v4_1_training_kaggle.ipynb`.
2. Upload the supporting `.py` files and `tusgan_encode.npz` into the notebook directory:
   - On the right panel under **Input**, click **Upload Dataset**, select `tusgan_v4_1_kaggle.zip`, name it `tusgan-v4-1`, and upload.
   - In the first cell of the notebook, unzip it into `/kaggle/working/`:
     ```python
     !unzip -q /kaggle/input/tusgan-v4-1/tusgan_v4_1_kaggle.zip -d /kaggle/working/
     %cd /kaggle/working
     ```

#### Option B: Terminal / Command-line
If using Kaggle terminal or script mode, simply run:
```bash
bash run_kaggle.sh
```

### Step 3: Run Pre-Flight Smoke Test
In the notebook, execute:
```python
!python smoke_test.py
```
This confirms:
- Both T4 GPUs are detected (16GB VRAM each).
- Deterministic logit masking forces child work mass to strictly 0.00%.
- Gradient penalty double-backward pass succeeds.
- Dead-gradient is mathematically proven: $\partial \mathcal{L} / \partial \theta_0 = 0.00000000$.

### Step 4: Launch Full 250-Epoch Training on Dual GPU
Run:
```python
!torchrun --nproc_per_node=2 train.py \
    --data tusgan_encode.npz \
    --epochs 250 \
    --batch 512 \
    --seed 42 \
    --checkpoint_dir checkpoints \
    --log_dir runs/tusgan_v4_1_hard
```
- **Dual GPU Acceleration**: PyTorch DDP splits the 512 batch (256 per GPU), maximizing throughput.
- **Estimated Runtime**: ~3.5 to 4.5 hours for full 250 epochs.
- **Checkpoints**: Saved every 10 epochs to `checkpoints/epoch_XXXX.pt`, plus `checkpoints/final.pt`.
- **Gradient Diagnostics**: Saved at epochs 10, 50, 100, 250 to `checkpoints/grad_diagnostics_epoch_XXXX.pt`.

### Step 5: Run Ultimate Evaluation
Once training completes, evaluate the final model:
```python
!python evaluate_ultimate.py \
    --checkpoint checkpoints/final.pt \
    --data tusgan_encode.npz \
    --n_samples 30000 \
    --output_dir evaluation_results
```
This generates:
- `evaluation_results/metrics_cell_c.json` (Full metrics)
- `evaluation_results/table6_cell_c_summary.csv` (Table 6 row)
- `evaluation_results/cell_c_diurnal_participation.png` & `.pdf`
- `evaluation_results/cell_c_adversarial_roc.png` & `.pdf`

### Step 6: Download Your Results
In the last cell, package all deliverables:
```python
!zip -r /kaggle/working/tusgan_v4_1_trained_outputs.zip checkpoints/final.pt checkpoints/grad_diagnostics_*.pt evaluation_results/ runs/
```
In the Kaggle right-hand panel under **Output** (`/kaggle/working`), click the three dots next to `tusgan_v4_1_trained_outputs.zip` and select **Download**!
