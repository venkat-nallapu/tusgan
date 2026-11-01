"""
TUS-GAN Dashboard — Example A: Research Workbench
=================================================
Design philosophy:
- Clean scientific instrument layout (top conditioning bar + tabbed analytical views)
- Scales from 1 to 50,000 schedules using vectorized NumPy conditioning & chunked inference
- Multi-agent 24-hour cohort Gantt raster + interactive single-diary inspector
- Zero emojis, sentence-case labels, Material Symbols icons, fragment-isolated post-processing

Run:
    streamlit run dashboard_example_a.py
"""

import glob
import io
import os
from datetime import datetime, timezone

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dashboard import (
    ACT_LABELS,
    ACTIVITY_COLORS,
    ACTIVITY_LABELS,
    AGE_LABELS,
    DATA_NPZ_PATH,
    DISTRICT_NAMES_BY_STATE,
    DOW_LABELS,
    EDU_LABELS,
    FILTER_HELP,
    FILTER_OFF,
    GENDER_LABELS,
    MARITAL_LABELS,
    SECTOR_LABELS,
    STATE_NAMES,
    TIME_SLOTS,
    count_spells,
    load_model_and_data,
)
from evaluation.postprocess import PRESETS, apply_preset

MAX_SAMPLES = 50_000
CHUNK_SIZE = 1024
MAX_RASTER_ROWS = 100

MODEL_VERSIONS = {
    "TUS-GAN v4 (Production)": ("v4", "TUS-GAN v4"),
    "TUS-GAN v3 (Baseline)": ("v3", "TUS-GAN v3"),
    "TUS-GAN v5 (Transformer)": ("v5", "TUS-GAN v5"),
}


def resolve_version_paths(version_key: str):
    root = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(root, version_key, "checkpoints", "final.pt")
    data_path = os.path.join(root, version_key, "tusgan_encode.npz")
    if not os.path.exists(data_path) and os.path.exists(DATA_NPZ_PATH):
        data_path = DATA_NPZ_PATH
    if version_key == "v5":
        eval_dir = os.path.join(root, "v5", "results")
        if not os.path.exists(eval_dir):
            eval_dir = os.path.join(root, "v5", "evaluation_v5_100k")
    else:
        eval_dir = os.path.join(root, version_key, "evaluation_results")
    return ckpt_path, data_path, eval_dir


def build_conditioned_batch(
    all_cond: np.ndarray,
    num_samples: int,
    *,
    age_idx: int,
    gender_idx: int,
    marital_idx: int,
    edu_idx: int,
    act_idx: int,
    dow_idx: int,
    sector_idx: int,
) -> np.ndarray:
    """Vectorized construction of ``(num_samples, 82)`` conditioning vectors."""
    row_idx = np.random.randint(0, all_cond.shape[0], size=num_samples)
    batch = all_cond[row_idx].copy()
    if batch.shape[1] >= 47:
        batch[:, 0:47] = 0.0
        batch[:, 0 + min(age_idx, 6)] = 1.0
        batch[:, 7 + min(gender_idx, 2)] = 1.0
        batch[:, 10 + min(marital_idx, 3)] = 1.0
        batch[:, 14 + min(edu_idx, 10)] = 1.0
        batch[:, 25 + min(act_idx, 12)] = 1.0
        batch[:, 38 + min(dow_idx, 6)] = 1.0
        batch[:, 45 + min(sector_idx, 1)] = 1.0
    return batch


def sample_batch(
    G,
    cfg,
    data,
    *,
    num_samples: int,
    district_id: int,
    state_id: int,
    gumbel_temp: float,
    gumbel_hard: bool,
    persona_indices: dict,
) -> np.ndarray:
    """Sample ``num_samples`` diaries (up to 50,000) in memory-safe mini-batches."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = G.to(device)
    all_cond = data["cond_vector"]
    cond_batch = build_conditioned_batch(all_cond, num_samples, **persona_indices)
    decoded = np.empty((num_samples, 48), dtype=np.int64)

    show_prog = num_samples > CHUNK_SIZE
    progress = st.progress(0.0, text=f"Synthesizing {num_samples:,} diaries...") if show_prog else None

    with torch.no_grad():
        for start in range(0, num_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_samples)
            b = end - start
            z = torch.randn(b, cfg["noise_dim"], device=device)
            cv = torch.from_numpy(cond_batch[start:end]).float().to(device)
            di = torch.full((b,), int(district_id), dtype=torch.long, device=device)
            si = torch.full((b,), int(state_id), dtype=torch.long, device=device)
            fake = G(z, cv, di, si, temp=gumbel_temp, hard=gumbel_hard)  # (b, 9, 48, 1)
            fake_np = fake.squeeze(-1).cpu().numpy()  # (b, 9, 48)
            decoded[start:end] = np.argmax(fake_np, axis=1) + 1
            if progress is not None:
                progress.progress(
                    end / num_samples,
                    text=f"Synthesizing diaries ({end:,} / {num_samples:,})",
                )

    if progress is not None:
        progress.empty()
    return decoded


# ---------------------------------------------------------------------------
# Vectorized batch metrics & fast serialization (supports N = 50,000)
# ---------------------------------------------------------------------------


def count_spells_batch(arr: np.ndarray, code: int = 1) -> int:
    """Total number of contiguous spells of ``code`` across an (N, 48) array."""
    is_c = arr == code
    starts = np.empty_like(is_c)
    starts[:, 0] = is_c[:, 0]
    starts[:, 1:] = is_c[:, 1:] & ~is_c[:, :-1]
    return int(starts.sum())


def mean_episodes_batch(arr: np.ndarray) -> float:
    """Mean number of contiguous activity episodes per 24h schedule across (N, 48)."""
    transitions = (arr[:, 1:] != arr[:, :-1]).sum(axis=1)
    return float((1 + transitions).mean())


def count_flickers_batch(arr: np.ndarray, code: int = 1) -> int:
    """Count isolated 1-slot (0.5h) spells of ``code`` across an (N, 48) array."""
    is_c = arr == code
    padded = np.pad(is_c, ((0, 0), (1, 1)), constant_values=False)
    isolated = padded[:, 1:-1] & ~padded[:, :-2] & ~padded[:, 2:]
    return int(isolated.sum())


@st.cache_data(max_entries=6, show_spinner="Formatting CSV export...")
def fast_diaries_to_csv_bytes(decoded: np.ndarray, meta_items: tuple, filter_name: str) -> bytes:
    """Vectorized DataFrame-to-CSV export capable of serializing 50,000 rows in <0.4s."""
    n = decoded.shape[0]
    meta = dict(meta_items)
    cols = {"sample_id": np.arange(1, n + 1)}
    for k, v in meta.items():
        cols[k] = v
    cols["employment_filter"] = filter_name
    df_meta = pd.DataFrame(cols)
    df_slots = pd.DataFrame(decoded, columns=TIME_SLOTS)
    df_full = pd.concat([df_meta, df_slots], axis=1)
    return df_full.to_csv(index=False).encode("utf-8")


@st.cache_data(max_entries=4, show_spinner="Formatting JSON export...")
def fast_diaries_to_json_bytes(decoded: np.ndarray, meta_items: tuple, filter_name: str) -> bytes:
    """Serialize batch to JSON bytes (compact for >1,000 schedules)."""
    n = decoded.shape[0]
    meta = dict(meta_items)
    cols = {"sample_id": np.arange(1, n + 1)}
    for k, v in meta.items():
        cols[k] = v
    cols["employment_filter"] = filter_name
    df_meta = pd.DataFrame(cols)
    df_slots = pd.DataFrame(decoded, columns=TIME_SLOTS)
    df_full = pd.concat([df_meta, df_slots], axis=1)
    indent = 2 if n <= 500 else None
    return df_full.to_json(orient="records", indent=indent).encode("utf-8")


# ---------------------------------------------------------------------------
# Publication-grade plot renderers (cached)
# ---------------------------------------------------------------------------


@st.cache_data(max_entries=128, show_spinner=False)
def render_cohort_raster_png(decoded_batch: np.ndarray, title: str) -> bytes:
    """Multi-agent 24-hour Gantt raster (caps display at MAX_RASTER_ROWS for large N)."""
    total_n = decoded_batch.shape[0]
    shown_slice = decoded_batch[:MAX_RASTER_ROWS]
    n = shown_slice.shape[0]

    fig_h = max(2.2, min(7.2, 0.28 * n + 1.2))
    fig, ax = plt.subplots(figsize=(13.5, fig_h), dpi=130)

    # Fast RGB raster construction (instant even for 100x48)
    rgb_grid = np.zeros((n, 48, 3), dtype=np.float32)
    for code, hex_col in ACTIVITY_COLORS.items():
        h = hex_col.lstrip("#")
        rgb = tuple(int(h[j : j + 2], 16) / 255.0 for j in (0, 2, 4))
        rgb_grid[shown_slice == code] = rgb

    ax.imshow(rgb_grid, aspect="auto", interpolation="nearest", extent=[0, 48, n - 0.5, -0.5])

    # Subtle row separators when N <= 30
    if n <= 30:
        for r in range(n + 1):
            ax.axhline(r - 0.5, color="white", linewidth=0.8)
    for t in range(49):
        ax.axvline(t, color="white", linewidth=0.25, alpha=0.6)

    if n <= 20:
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"Diary {i + 1}" for i in range(n)], fontsize=8, color="#3f3f46")
    else:
        step = max(1, n // 10)
        ticks = list(range(0, n, step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"Diary {i + 1}" for i in ticks], fontsize=8, color="#3f3f46")

    tick_pos = list(range(0, 48, 4))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([TIME_SLOTS[i] for i in tick_pos], fontsize=8.5, color="#3f3f46")

    full_title = (
        f"{title} (showing first {n} of {total_n:,} schedules)"
        if total_n > n
        else title
    )
    ax.set_title(full_title, fontsize=10, fontweight="bold", loc="left", color="#18181b", pad=8)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d4d4d8")
    ax.tick_params(axis="y", length=0)

    patches = [
        mpatches.Patch(color=ACTIVITY_COLORS[k], label=f"{k}. {ACTIVITY_LABELS[k]}")
        for k in sorted(ACTIVITY_LABELS)
    ]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        fontsize=7.5,
        frameon=False,
        handlelength=1.2,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=256, show_spinner=False)
def render_single_diary_png(decoded: np.ndarray, kind: str, title: str) -> bytes:
    """Clean single-diary strip or step transition plot."""
    if kind == "strip":
        fig, ax = plt.subplots(figsize=(13.5, 1.35), dpi=130)
        for i, code in enumerate(decoded):
            ax.barh(
                0,
                1.0,
                left=i,
                height=0.75,
                color=ACTIVITY_COLORS.get(int(code), "#d4d4d8"),
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlim(0, 48)
        ax.set_yticks([])
        tick_pos = list(range(0, 48, 4))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([TIME_SLOTS[i] for i in tick_pos], fontsize=8.5, color="#3f3f46")
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", color="#18181b", pad=6)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#d4d4d8")
    else:
        fig, ax = plt.subplots(figsize=(13.5, 3.2), dpi=130)
        x = np.arange(48)
        ax.step(x, decoded, where="post", color="#18181b", linewidth=1.6, zorder=3)
        for i in range(48):
            colour = ACTIVITY_COLORS.get(int(decoded[i]), "#f4f4f5")
            ax.axvspan(i, i + 1, color=colour, alpha=0.22, zorder=1)
        ax.set_xlim(0, 48)
        ax.set_ylim(0.5, 9.5)
        ax.set_yticks(range(1, 10))
        ax.set_yticklabels(
            [f"{k}. {ACTIVITY_LABELS[k]}" for k in range(1, 10)], fontsize=8, color="#3f3f46"
        )
        tick_pos = list(range(0, 48, 4))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([TIME_SLOTS[i] for i in tick_pos], fontsize=8.5, color="#3f3f46")
        ax.grid(axis="y", linestyle=":", alpha=0.4, color="#a1a1aa")
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", color="#18181b", pad=6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#d4d4d8")
        ax.spines["bottom"].set_color("#d4d4d8")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_breakdown_df(decoded: np.ndarray) -> pd.DataFrame:
    """Structured DataFrame for st.dataframe with ProgressColumn support."""
    rows = []
    for code in sorted(ACTIVITY_LABELS):
        slots = int((decoded == code).sum())
        hours = slots * 0.5
        episodes = count_spells(decoded, code=code)
        pct = (slots / 48.0) * 100.0
        rows.append(
            {
                "Code": code,
                "Activity division": ACTIVITY_LABELS[code],
                "Hours": hours,
                "Slots (30m)": slots,
                "Episodes": episodes,
                "Share of day": pct,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fragment: Interactive Results Workbench
# ---------------------------------------------------------------------------


@st.fragment
def render_workbench_results(eval_dir: str, version_label: str):
    gen = st.session_state["generated"]
    raw = gen["decoded"]
    meta = gen["meta"]
    meta_items = tuple(meta.items())

    # Top control & export bar
    with st.container(border=True):
        col_filter, col_export = st.columns([2.2, 1], vertical_alignment="center")
        with col_filter:
            choice = st.segmented_control(
                "Employment post-processing (bounded morphological filter)",
                [FILTER_OFF, *PRESETS],
                default=FILTER_OFF,
                required=True,
                key="morph_choice",
                help=FILTER_HELP,
            )
        shown = raw if choice == FILTER_OFF else apply_preset(raw, choice)
        suffix = "" if choice == FILTER_OFF else "_" + choice.lower().replace(" ", "")
        with col_export:
            st.caption(f"Export batch ({len(shown):,} schedules)")
            with st.container(horizontal=True):
                st.download_button(
                    "CSV",
                    data=fast_diaries_to_csv_bytes(shown, meta_items, choice),
                    file_name=f"tusgan_diaries_{gen['timestamp']}{suffix}.csv",
                    mime="text/csv",
                    icon=":material/table_view:",
                    key="dl_csv",
                    width="stretch",
                )
                st.download_button(
                    "JSON",
                    data=fast_diaries_to_json_bytes(shown, meta_items, choice),
                    file_name=f"tusgan_diaries_{gen['timestamp']}{suffix}.json",
                    mime="application/json",
                    icon=":material/data_object:",
                    key="dl_json",
                    width="stretch",
                )

    # Vectorized Batch KPI cards
    emp_hours = (shown == 1).sum(axis=1) * 0.5
    raw_emp_spells = count_spells_batch(raw, 1)
    shown_emp_spells = count_spells_batch(shown, 1)
    mean_episodes = mean_episodes_batch(shown)
    n_modified = int((shown != raw).any(axis=1).sum())

    with st.container(horizontal=True):
        st.metric(
            "Synthesized cohort",
            f"{len(shown):,} schedules",
            f"{meta['state_name']} · {meta['district_name']}",
            delta_color="off",
            border=True,
        )
        st.metric(
            "Mean employment",
            f"{emp_hours.mean():.2f} h/day",
            f"{int((emp_hours > 0).sum()):,}/{len(shown):,} active workers",
            delta_color="off",
            border=True,
        )
        spell_delta = (
            f"{shown_emp_spells - raw_emp_spells:+d} spells ({n_modified:,} diaries consolidated)"
            if choice != FILTER_OFF
            else "Raw generator output"
        )
        st.metric(
            "Employment spells (batch)",
            f"{shown_emp_spells:,} spells",
            spell_delta,
            delta_color="off",
            border=True,
        )
        st.metric(
            "Mean daily episodes",
            f"{mean_episodes:.1f} episodes",
            f"Gumbel τ = {meta['gumbel_temperature']:.2f}",
            delta_color="off",
            border=True,
        )

    tab_cohort, tab_diff, tab_eval = st.tabs(
        [
            ":material/view_timeline: Cohort & schedule inspector",
            ":material/compare_arrows: Morphological filter analysis",
            ":material/analytics: Model evaluation gallery",
        ]
    )

    with tab_cohort:
        with st.container(border=True):
            st.markdown("**24-hour cohort activity raster**")
            st.caption(
                f"Profile: {meta['gender']}, {meta['age_group']}, {meta['principal_activity']}, "
                f"{meta['sector']} ({meta['day_of_week']}) — {meta['model_version']}"
            )
            st.image(
                render_cohort_raster_png(
                    shown,
                    f"Synthetic cohort schedules (N = {len(shown):,}, filter: {choice})",
                ),
                width="stretch",
            )

        with st.container(border=True):
            header_col, picker_col = st.columns([1.2, 2], vertical_alignment="center")
            with header_col:
                st.markdown("**Individual schedule inspector**")
                st.caption("Select any diary from the batch to inspect its step trajectory and time budget.")
            with picker_col:
                if len(shown) <= 12:
                    sample_idx = st.segmented_control(
                        "Select diary",
                        options=list(range(len(shown))),
                        default=0,
                        format_func=lambda idx: f"#{idx + 1}",
                        required=True,
                        key="sample_picker_seg",
                    )
                else:
                    sample_idx = (
                        int(
                            st.number_input(
                                f"Select diary index (1 to {len(shown):,})",
                                min_value=1,
                                max_value=len(shown),
                                value=1,
                                step=1,
                                key="sample_picker_num",
                            )
                        )
                        - 1
                    )

            decoded_one = shown[sample_idx]
            raw_one = raw[sample_idx]
            if choice != FILTER_OFF:
                st.caption(
                    f"Diary #{sample_idx + 1:,} employment spells: "
                    f"{count_spells(raw_one, 1)} raw → {count_spells(decoded_one, 1)} filtered"
                )

            left_plot, right_table = st.columns([1.55, 1], vertical_alignment="top")
            with left_plot:
                st.image(
                    render_single_diary_png(
                        decoded_one,
                        "strip",
                        f"Diary #{sample_idx + 1:,} — 24-hour timeline strip",
                    ),
                    width="stretch",
                )
                st.image(
                    render_single_diary_png(
                        decoded_one,
                        "step",
                        f"Diary #{sample_idx + 1:,} — state transition step trajectory",
                    ),
                    width="stretch",
                )
            with right_table:
                df_break = build_breakdown_df(decoded_one)
                st.dataframe(
                    df_break,
                    hide_index=True,
                    width="stretch",
                    height=368,
                    column_config={
                        "Code": st.column_config.NumberColumn("ID", width="small"),
                        "Activity division": st.column_config.TextColumn("ICATUS division"),
                        "Hours": st.column_config.NumberColumn("Hours", format="%.1f h"),
                        "Slots (30m)": st.column_config.NumberColumn("Slots"),
                        "Episodes": st.column_config.NumberColumn("Spells"),
                        "Share of day": st.column_config.ProgressColumn(
                            "Share of 24h",
                            format="%.1f%%",
                            min_value=0.0,
                            max_value=100.0,
                        ),
                    },
                )

    with tab_diff:
        with st.container(border=True):
            st.markdown("**Before vs. after morphological consolidation**")
            st.caption(
                "Compares the raw generator output against the bounded 1D morphological filter "
                f"(manuscript section 6.4.5) across all {len(raw):,} synthesized schedules."
            )
            v1_batch = apply_preset(raw, list(PRESETS.keys())[0])
            v2_batch = apply_preset(raw, list(PRESETS.keys())[1])
            comp_df = pd.DataFrame(
                [
                    {
                        "Configuration": "Raw generator (Off)",
                        "Modified diaries": 0,
                        "Total employment spells": count_spells_batch(raw, 1),
                        "Mean employment (h/day)": float(((raw == 1).sum(axis=1) * 0.5).mean()),
                        "Single-slot (0.5h) work flickers": count_flickers_batch(raw, 1),
                    },
                    {
                        "Configuration": list(PRESETS.keys())[0],
                        "Modified diaries": int((v1_batch != raw).any(axis=1).sum()),
                        "Total employment spells": count_spells_batch(v1_batch, 1),
                        "Mean employment (h/day)": float(((v1_batch == 1).sum(axis=1) * 0.5).mean()),
                        "Single-slot (0.5h) work flickers": count_flickers_batch(v1_batch, 1),
                    },
                    {
                        "Configuration": list(PRESETS.keys())[1],
                        "Modified diaries": int((v2_batch != raw).any(axis=1).sum()),
                        "Total employment spells": count_spells_batch(v2_batch, 1),
                        "Mean employment (h/day)": float(((v2_batch == 1).sum(axis=1) * 0.5).mean()),
                        "Single-slot (0.5h) work flickers": count_flickers_batch(v2_batch, 1),
                    },
                ]
            )
            st.dataframe(
                comp_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Mean employment (h/day)": st.column_config.NumberColumn(format="%.2f h"),
                },
            )

    with tab_eval:
        render_evaluation_gallery(eval_dir, version_label)


def render_evaluation_gallery(eval_dir: str, version_label: str):
    if not os.path.isdir(eval_dir):
        st.warning(f"Evaluation directory `{eval_dir}` not found.", icon=":material/warning:")
        return

    images = sorted(glob.glob(os.path.join(eval_dir, "*.png")))
    if not images:
        st.warning(f"No evaluation plots found in `{eval_dir}`.", icon=":material/warning:")
        return

    st.caption(f"Pre-computed benchmark diagnostics for {version_label} (`{eval_dir}`).")
    cols = st.columns(2)
    for idx, img_path in enumerate(images):
        name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").capitalize()
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.image(img_path, width="stretch")


# ---------------------------------------------------------------------------
# Main App Entry Point
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="TUS-GAN Workbench (Example A)",
        page_icon=":material/schedule:",
        layout="wide",
    )

    # Sidebar: global model & sampler controls only
    with st.sidebar:
        st.markdown("### :material/neurology: TUS-GAN workbench")
        st.caption("Example A · Top command bar + cohort raster")

        version_choice = st.selectbox(
            "Model checkpoint",
            list(MODEL_VERSIONS.keys()),
            index=0,
        )
        version_key, version_label = MODEL_VERSIONS[version_choice]
        ckpt_path, data_path, eval_dir = resolve_version_paths(version_key)
        G, cfg, data = load_model_and_data(version_key, ckpt_path, data_path)

        st.markdown("**Sampler parameters**")
        num_samples = st.number_input(
            "Batch size (diaries, up to 50k)",
            min_value=1,
            max_value=MAX_SAMPLES,
            value=10,
            step=10,
        )
        gumbel_temp = st.slider(
            "Gumbel temperature (τ)", min_value=0.05, max_value=2.0, value=0.10, step=0.05
        )
        gumbel_hard = st.toggle("Hard one-hot discretization", value=True)

        st.caption(
            f"Dataset: {data['cond_vector'].shape[0]:,} ITUS diaries · "
            f"{int(data['num_states'])} states · {int(data['num_districts'])} district codes"
        )

    # Page header
    st.subheader("Synthetic time-use schedule workbench")
    st.caption(
        "Conditioned 24-hour activity synthesis (48 half-hour slots, 9 ICATUS divisions) "
        "trained on Indian Time Use Survey (ITUS) 2019 microdata."
    )

    # Top conditioning command bar
    num_states = int(data["num_states"])
    num_districts = int(data["num_districts"])
    state_options = {sid: STATE_NAMES.get(sid, f"State {sid}") for sid in range(num_states)}

    with st.container(border=True):
        st.markdown("**:material/tune: Conditioning profile & geography**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age_sel = st.selectbox("Age group", AGE_LABELS, index=3)
            gender_sel = st.selectbox("Gender", GENDER_LABELS, index=0)
        with c2:
            marital_sel = st.selectbox("Marital status", MARITAL_LABELS, index=0)
            edu_sel = st.selectbox("Education level", list(EDU_LABELS.values()), index=8)
        with c3:
            act_sel = st.selectbox("Principal activity", list(ACT_LABELS.values()), index=3)
            dow_sel = st.selectbox("Day of week", DOW_LABELS, index=0)
        with c4:
            state_id = st.selectbox(
                "State / UT",
                range(num_states),
                index=min(28, num_states - 1),
                format_func=lambda sid: f"{state_options[sid]} (#{sid})",
            )
            state_district_map = DISTRICT_NAMES_BY_STATE.get(state_id, {})
            available_district_ids = (
                sorted(state_district_map.keys())
                if state_district_map
                else list(range(num_districts))
            )
            district_options = {
                did: state_district_map.get(did, f"District {did}")
                for did in available_district_ids
            }
            district_id = st.selectbox(
                "District",
                available_district_ids,
                index=0,
                format_func=lambda did: f"{district_options[did]} (#{did})",
            )

        b1, b2, b3 = st.columns([1.2, 1.2, 1], vertical_alignment="center")
        with b1:
            sector_sel = st.segmented_control(
                "Sector", SECTOR_LABELS, default="Urban", required=True
            )
        with b2:
            caregiving_sel = st.toggle("Active household caregiving requirement", value=False)
        with b3:
            generate = st.button(
                "Synthesize schedules",
                type="primary",
                icon=":material/bolt:",
                width="stretch",
            )

    if generate or "generated" not in st.session_state:
        persona_indices = {
            "age_idx": AGE_LABELS.index(age_sel),
            "gender_idx": GENDER_LABELS.index(gender_sel),
            "marital_idx": MARITAL_LABELS.index(marital_sel),
            "edu_idx": list(EDU_LABELS.values()).index(edu_sel),
            "act_idx": list(ACT_LABELS.values()).index(act_sel),
            "dow_idx": DOW_LABELS.index(dow_sel),
            "sector_idx": SECTOR_LABELS.index(sector_sel),
        }
        decoded = sample_batch(
            G,
            cfg,
            data,
            num_samples=int(num_samples),
            district_id=district_id,
            state_id=state_id,
            gumbel_temp=gumbel_temp,
            gumbel_hard=gumbel_hard,
            persona_indices=persona_indices,
        )
        st.session_state["generated"] = {
            "decoded": decoded,
            "title": f"{gender_sel}, {age_sel} — {state_options[state_id]}",
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "meta": {
                "model_version": version_label,
                "age_group": age_sel,
                "gender": gender_sel,
                "marital_status": marital_sel,
                "education": edu_sel,
                "principal_activity": act_sel,
                "day_of_week": dow_sel,
                "sector": sector_sel,
                "caregiving_required": caregiving_sel,
                "state_id": state_id,
                "state_name": state_options[state_id],
                "district_id": district_id,
                "district_name": district_options[district_id],
                "gumbel_temperature": gumbel_temp,
                "gumbel_hard": gumbel_hard,
            },
        }

    render_workbench_results(eval_dir, version_label)


if __name__ == "__main__":
    main()
