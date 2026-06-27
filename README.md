# TUS-GAN: Conditional Time-Use Diary Synthesis

This repository contains the development, training, and evaluation pipelines for **TUS-GAN**, a conditional Generative Adversarial Network designed to synthesize realistic 24-hour daily activity diaries based on demographics using the **Indian Time Use Survey (ITUS) 2019** dataset.

---

## 📂 Project Directory Structure

The workspace is organized into separate directories for each development version to isolate weights, datasets, training logs, and analysis:

*   **[v3/](file:///home/venkat/projects/tusgan-v3/v3/)**: Baseline model introducing categorical Gumbel-Softmax discretization and Global Transition Matrix Loss. Achieved high population-level distribution alignment but had small logical leaks.
*   **[v4/](file:///home/venkat/projects/tusgan-v3/v4/)**: Upgraded model incorporating decoupled Demographic AC-GAN Loss, Time-Slice Transition Loss (for time-of-day dynamics), and Neuro-Symbolic Logic Constraints (preventing Child Labor violations).
    *   *See [v4/README.md](file:///home/venkat/projects/tusgan-v3/v4/README.md) for a detailed v3 vs v4 performance comparison and mathematical formulations.*
*   **`[dashboard.py](file:///home/venkat/projects/tusgan-v3/dashboard.py)`**: Interactive Streamlit application to generate individual daily diaries in real-time. Supports switching between v3 and v4 models.
*   **`[DEVELOPMENT_LEDGER.md](file:///home/venkat/projects/tusgan-v3/DEVELOPMENT_LEDGER.md)`**: Change ledger recording all architectural modifications, training runs, and benchmark metrics chronological-order.

---

## 🚀 Running the Interactive Dashboard

You can interactively explore synthesized diaries and model metrics across versions using the unified dashboard:

```bash
# Ensure you are using the virtual environment
source .venv-v3/bin/activate

# Launch Streamlit
streamlit run dashboard.py
```

*Inside the dashboard sidebar, you can select the active model version dropdown to toggle between **TUS-GAN v3** and **TUS-GAN v4** weights dynamically.*

---

## 📈 Version Documentation Quick-Links

*   **TUS-GAN v3 Details**: [tusgan-v3.md](file:///home/venkat/projects/tusgan-v3/tusgan-v3.md)
*   **TUS-GAN v4 Upgrades**: [v4/README.md](file:///home/venkat/projects/tusgan-v3/v4/README.md) and [tusgan-v4.md](file:///home/venkat/projects/tusgan-v3/tusgan-v4.md)
