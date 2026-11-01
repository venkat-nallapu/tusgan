"""
TUS-GAN workbench
=================
Streamlit front end for the conditioned time-use schedule generator.

Model and sampler controls sit in the sidebar. A single conditioning form sits
above the results canvas, and three analysis views follow it: the cohort raster
with a single-diary inspector, the filter comparison, and the evaluation gallery.

Inference runs in fixed-size chunks over a vectorised conditioning batch, so one
diary and fifty thousand diaries travel the same code path with a flat memory
profile. Sampling is seeded, so a profile plus a seed reproduces a batch exactly.

Run:
    streamlit run dashboard_example_a.py
"""

import glob
import html
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
SLOTS_PER_DAY = 48

# Offset and width of each one-hot block inside the conditioning vector.
COND_LAYOUT = {
    "age": (0, 7),
    "gender": (7, 3),
    "marital": (10, 4),
    "education": (14, 11),
    "activity": (25, 13),
    "day_of_week": (38, 7),
    "sector": (45, 2),
}
COND_WIDTH = 47

MODEL_VERSIONS = {
    "TUS-GAN v4 (production)": ("v4", "TUS-GAN v4"),
    "TUS-GAN v3 (baseline)": ("v3", "TUS-GAN v3"),
    "TUS-GAN v5 (transformer)": ("v5", "TUS-GAN v5"),
}

CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1380px;}
  .tw-title {font-size: 1.5rem; font-weight: 600; letter-spacing: -0.01em;
             color: #18181b; margin: 0; line-height: 1.3;}
  .tw-sub {font-size: 0.85rem; color: #71717a; margin: 0.25rem 0 0; line-height: 1.5;}
  .tw-label {font-size: 0.7rem; font-weight: 600; letter-spacing: 0.07em;
             text-transform: uppercase; color: #71717a; margin: 0 0 0.3rem;}
  .tw-strip {display: flex; flex-wrap: wrap; border-top: 1px solid #e4e4e7;
             border-bottom: 1px solid #e4e4e7; margin: 0.1rem 0 0.3rem;}
  .tw-stat {flex: 1 1 190px; padding: 0.6rem 1.1rem 0.6rem 0;}
  .tw-stat + .tw-stat {border-left: 1px solid #e4e4e7; padding-left: 1.1rem;}
  .tw-stat .k {font-size: 0.7rem; font-weight: 600; letter-spacing: 0.07em;
               text-transform: uppercase; color: #71717a;}
  .tw-stat .v {font-size: 1.28rem; font-weight: 600; color: #18181b;
               font-variant-numeric: tabular-nums; line-height: 1.35;}
  .tw-stat .n {font-size: 0.78rem; color: #71717a; line-height: 1.35;}
</style>
"""


# ---------------------------------------------------------------------------
# Paths, loading and sampling
# ---------------------------------------------------------------------------


def resolve_version_paths(version_key: str):
    """Checkpoint, conditioning data and evaluation directory for a version."""
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


@st.cache_resource(show_spinner="Loading checkpoint and conditioning data")
def load_bundle(version_key: str, ckpt_path: str, data_path: str):
    """Load the generator and dataset once per checkpoint, not once per rerun."""
    return load_model_and_data(version_key, ckpt_path, data_path)


def build_conditioned_batch(all_cond: np.ndarray, num_samples: int, *, persona: dict, rng) -> np.ndarray:
    """Build ``(num_samples, 82)`` conditioning vectors from a fixed persona.

    Donor rows supply the dimensions outside ``COND_LAYOUT``; the one-hot blocks
    are rewritten. Out-of-range indices raise instead of being silently clamped.
    """
    if all_cond.ndim != 2 or all_cond.shape[1] < COND_WIDTH:
        raise ValueError(
            f"conditioning vectors must be at least {COND_WIDTH} wide, got shape {all_cond.shape}"
        )

    batch = all_cond[rng.integers(0, all_cond.shape[0], size=num_samples)].copy()
    batch[:, :COND_WIDTH] = 0.0
    for name, (offset, width) in COND_LAYOUT.items():
        index = int(persona[name])
        if not 0 <= index < width:
            raise ValueError(f"{name} index {index} is outside the valid range 0..{width - 1}")
        batch[:, offset + index] = 1.0
    return batch


def sample_batch(
    G,
    cfg,
    data: dict,
    *,
    num_samples: int,
    district_id: int,
    state_id: int,
    gumbel_temp: float,
    gumbel_hard: bool,
    persona: dict,
    seed: int,
) -> np.ndarray:
    """Sample ``num_samples`` diaries in memory-safe mini-batches."""
    if not 1 <= num_samples <= MAX_SAMPLES:
        raise ValueError(f"batch size must fall in 1..{MAX_SAMPLES}, got {num_samples}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = G.to(device)
    G.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    cond_batch = build_conditioned_batch(
        data["cond_vector"], num_samples, persona=persona, rng=np.random.default_rng(seed)
    )
    decoded = np.empty((num_samples, SLOTS_PER_DAY), dtype=np.int64)

    show_progress = num_samples > CHUNK_SIZE
    progress = st.progress(0.0, text=f"Synthesising {num_samples:,} diaries") if show_progress else None

    with torch.no_grad():
        for start in range(0, num_samples, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_samples)
            size = end - start
            z = torch.randn(size, int(cfg["noise_dim"]), device=device, generator=generator)
            cond = torch.from_numpy(cond_batch[start:end]).float().to(device)
            districts = torch.full((size,), int(district_id), dtype=torch.long, device=device)
            states = torch.full((size,), int(state_id), dtype=torch.long, device=device)
            fake = G(z, cond, districts, states, temp=gumbel_temp, hard=gumbel_hard)
            decoded[start:end] = torch.argmax(fake.squeeze(-1), dim=1).cpu().numpy() + 1
            if progress is not None:
                progress.progress(end / num_samples, text=f"Synthesising diaries ({end:,} / {num_samples:,})")

    if progress is not None:
        progress.empty()
    return decoded


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------


def employment_hours(diaries: np.ndarray) -> np.ndarray:
    """Employment hours per diary, at 0.5 h per slot."""
    return (diaries == 1).sum(axis=1) * 0.5


def count_spells_batch(diaries: np.ndarray, code: int = 1) -> int:
    """Contiguous spells of ``code`` across a whole (N, 48) batch."""
    is_code = diaries == code
    starts = np.zeros_like(is_code)
    starts[:, 0] = is_code[:, 0]
    starts[:, 1:] = is_code[:, 1:] & ~is_code[:, :-1]
    return int(starts.sum())


def mean_episodes(diaries: np.ndarray) -> float:
    """Mean number of activity episodes per diary."""
    transitions = (diaries[:, 1:] != diaries[:, :-1]).sum(axis=1)
    return float((1 + transitions).mean())


def count_flickers_batch(diaries: np.ndarray, code: int = 1) -> int:
    """Isolated single-slot (0.5 h) spells of ``code`` across a batch."""
    is_code = diaries == code
    padded = np.pad(is_code, ((0, 0), (1, 1)), constant_values=False)
    isolated = padded[:, 1:-1] & ~padded[:, :-2] & ~padded[:, 2:]
    return int(isolated.sum())


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def _diaries_frame(diaries: np.ndarray, meta_items: tuple, filter_name: str) -> pd.DataFrame:
    columns = {"sample_id": np.arange(1, diaries.shape[0] + 1)}
    columns.update(dict(meta_items))
    columns["employment_filter"] = filter_name
    return pd.concat(
        [pd.DataFrame(columns), pd.DataFrame(diaries, columns=TIME_SLOTS)], axis=1
    )


@st.cache_data(max_entries=6, show_spinner="Preparing CSV export")
def diaries_to_csv_bytes(diaries: np.ndarray, meta_items: tuple, filter_name: str) -> bytes:
    return _diaries_frame(diaries, meta_items, filter_name).to_csv(index=False).encode("utf-8")


@st.cache_data(max_entries=4, show_spinner="Preparing JSON export")
def diaries_to_json_bytes(diaries: np.ndarray, meta_items: tuple, filter_name: str) -> bytes:
    frame = _diaries_frame(diaries, meta_items, filter_name)
    indent = 2 if len(frame) <= 500 else None
    return frame.to_json(orient="records", indent=indent).encode("utf-8")


# ---------------------------------------------------------------------------
# Plot renderers
# ---------------------------------------------------------------------------


def _hex_to_rgb(value: str) -> tuple:
    raw = value.lstrip("#")
    return tuple(int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


@st.cache_data(max_entries=64, show_spinner=False)
def render_cohort_raster_png(rows: np.ndarray, total_n: int, title: str) -> bytes:
    """Multi-diary 24-hour raster. Callers pass an already-sliced array."""
    n = rows.shape[0]
    if n == 0:
        raise ValueError("cannot render an empty raster")

    lookup = np.full((16, 3), 0.84, dtype=np.float32)
    for code, colour in ACTIVITY_COLORS.items():
        if 0 <= int(code) < 16:
            lookup[int(code)] = _hex_to_rgb(colour)
    rgb = lookup[np.clip(rows, 0, 15)]

    fig, ax = plt.subplots(figsize=(13.5, max(2.2, min(7.2, 0.28 * n + 1.2))), dpi=130)
    ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=[0, SLOTS_PER_DAY, n - 0.5, -0.5])

    if n <= 30:
        for row in range(n + 1):
            ax.axhline(row - 0.5, color="white", linewidth=0.8)
    for slot in range(SLOTS_PER_DAY + 1):
        ax.axvline(slot, color="white", linewidth=0.25, alpha=0.6)

    if n <= 20:
        ticks = list(range(n))
    else:
        step = max(1, n // 10)
        ticks = list(range(0, n, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"Diary {i + 1}" for i in ticks], fontsize=8, color="#3f3f46")

    x_ticks = list(range(0, SLOTS_PER_DAY, 4))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([TIME_SLOTS[i] for i in x_ticks], fontsize=8.5, color="#3f3f46")

    caption = f"{title} (first {n} of {total_n:,} schedules)" if total_n > n else title
    ax.set_title(caption, fontsize=10, fontweight="bold", loc="left", color="#18181b", pad=8)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d4d4d8")
    ax.tick_params(axis="y", length=0)

    handles = [
        mpatches.Patch(color=ACTIVITY_COLORS[code], label=f"{code}. {ACTIVITY_LABELS[code]}")
        for code in sorted(ACTIVITY_LABELS)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0, -0.06),
        ncol=3,
        fontsize=7.5,
        frameon=False,
        handlelength=1.2,
    )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


@st.cache_data(max_entries=128, show_spinner=False)
def render_single_diary_png(diary: np.ndarray, kind: str, title: str) -> bytes:
    """Single-diary timeline strip, or its state-transition step trajectory."""
    if kind == "strip":
        fig, ax = plt.subplots(figsize=(13.5, 1.35), dpi=130)
        for slot, code in enumerate(diary):
            ax.barh(
                0,
                1.0,
                left=slot,
                height=0.75,
                color=ACTIVITY_COLORS.get(int(code), "#d4d4d8"),
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#d4d4d8")
    else:
        fig, ax = plt.subplots(figsize=(13.5, 3.2), dpi=130)
        for slot in range(SLOTS_PER_DAY):
            colour = ACTIVITY_COLORS.get(int(diary[slot]), "#f4f4f5")
            ax.axvspan(slot, slot + 1, color=colour, alpha=0.22, zorder=1)
        ax.step(np.arange(SLOTS_PER_DAY), diary, where="post", color="#18181b", linewidth=1.6, zorder=3)
        ax.set_xlim(0, SLOTS_PER_DAY)
        ax.set_ylim(0.5, 9.5)
        ax.set_yticks(range(1, 10))
        ax.set_yticklabels([f"{k}. {ACTIVITY_LABELS[k]}" for k in range(1, 10)], fontsize=8, color="#3f3f46")
        ax.grid(axis="y", linestyle=":", alpha=0.4, color="#a1a1aa")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#d4d4d8")
        ax.spines["bottom"].set_color("#d4d4d8")

    x_ticks = list(range(0, SLOTS_PER_DAY, 4))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([TIME_SLOTS[i] for i in x_ticks], fontsize=8.5, color="#3f3f46")
    ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", color="#18181b", pad=6)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------


def section_label(text: str) -> None:
    st.markdown(f'<div class="tw-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def stat_strip(items) -> None:
    """Compact readout strip: (label, value, note) per column."""
    cells = "".join(
        '<div class="tw-stat">'
        f'<div class="k">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(value)}</div>'
        f'<div class="n">{html.escape(note)}</div>'
        "</div>"
        for label, value, note in items
    )
    st.markdown(f'<div class="tw-strip">{cells}</div>', unsafe_allow_html=True)


def pretty_name(stem: str) -> str:
    words = stem.replace("-", " ").replace("_", " ").split()
    return " ".join(word if word.isupper() else word[:1].upper() + word[1:] for word in words)


def build_breakdown_df(diary: np.ndarray) -> pd.DataFrame:
    rows = []
    for code in sorted(ACTIVITY_LABELS):
        slots = int((diary == code).sum())
        rows.append(
            {
                "Code": code,
                "Activity division": ACTIVITY_LABELS[code],
                "Hours": slots * 0.5,
                "Slots": slots,
                "Spells": count_spells(diary, code=code),
                "Share of day": slots / SLOTS_PER_DAY * 100.0,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(max_entries=4, show_spinner="Comparing filter presets")
def build_filter_comparison(raw: np.ndarray) -> pd.DataFrame:
    """Raw output against each registered preset, over the whole batch."""
    raw_spells = count_spells_batch(raw, 1)
    rows = [
        {
            "Configuration": "Raw generator",
            "Diaries modified": 0,
            "Employment spells": raw_spells,
            "Change vs raw": 0,
            "Mean employment (h/day)": float(employment_hours(raw).mean()),
            "Half-hour flickers": count_flickers_batch(raw, 1),
        }
    ]
    for preset in PRESETS:
        variant = apply_preset(raw, preset)
        spells = count_spells_batch(variant, 1)
        rows.append(
            {
                "Configuration": preset,
                "Diaries modified": int((variant != raw).any(axis=1).sum()),
                "Employment spells": spells,
                "Change vs raw": spells - raw_spells,
                "Mean employment (h/day)": float(employment_hours(variant).mean()),
                "Half-hour flickers": count_flickers_batch(variant, 1),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Results canvas
# ---------------------------------------------------------------------------


@st.fragment
def render_results(eval_dir: str, version_label: str, version_key: str) -> None:
    state = st.session_state.get("generated")
    if not state or state.get("version_key") != version_key:
        st.info("No batch for this checkpoint yet. Synthesise schedules to begin.")
        return

    raw = state["decoded"]
    meta = state["meta"]
    meta_items = tuple(meta.items())

    # Filter and export toolbar
    col_filter, col_export = st.columns([2.5, 1], vertical_alignment="bottom")
    with col_filter:
        choice = st.segmented_control(
            "Employment post-processing",
            [FILTER_OFF, *PRESETS],
            default=FILTER_OFF,
            required=True,
            key="morph_choice",
            help=FILTER_HELP,
        ) or FILTER_OFF
    shown = raw if choice == FILTER_OFF else apply_preset(raw, choice)
    suffix = "" if choice == FILTER_OFF else "_" + choice.lower().replace(" ", "")

    with col_export:
        st.caption(f"Export {len(shown):,} schedules")
        with st.container(horizontal=True):
            st.download_button(
                "CSV",
                data=diaries_to_csv_bytes(shown, meta_items, choice),
                file_name=f"tusgan_diaries_{state['timestamp']}{suffix}.csv",
                mime="text/csv",
                icon=":material/table_view:",
                key="dl_csv",
                width="stretch",
            )
            st.download_button(
                "JSON",
                data=diaries_to_json_bytes(shown, meta_items, choice),
                file_name=f"tusgan_diaries_{state['timestamp']}{suffix}.json",
                mime="application/json",
                icon=":material/data_object:",
                key="dl_json",
                width="stretch",
            )

    # Batch readout
    hours = employment_hours(shown)
    spells = count_spells_batch(shown, 1)
    if choice == FILTER_OFF:
        spell_note = "raw generator output"
    else:
        modified = int((shown != raw).any(axis=1).sum())
        spell_note = f"{spells - count_spells_batch(raw, 1):+d} vs raw · {modified:,} diaries changed"
    stat_strip(
        [
            ("Schedules", f"{len(shown):,}", f"{meta['state_name']} · {meta['district_name']}"),
            ("Mean employment", f"{hours.mean():.2f} h", f"{int((hours > 0).sum()):,} of {len(shown):,} in work"),
            ("Employment spells", f"{spells:,}", spell_note),
            ("Mean episodes", f"{mean_episodes(shown):.1f}", f"τ {meta['gumbel_temperature']:.2f} · seed {meta['seed']}"),
        ]
    )

    tab_cohort, tab_filter, tab_eval = st.tabs(
        ["Cohort and inspector", "Filter analysis", "Evaluation gallery"]
    )

    with tab_cohort:
        with st.container(border=True):
            st.markdown("**24-hour cohort raster**")
            st.caption(
                f"{meta['gender']}, {meta['age_group']}, {meta['principal_activity']}, "
                f"{meta['sector']} · {meta['day_of_week']} · {meta['model_version']}"
            )
            st.image(
                render_cohort_raster_png(
                    shown[:MAX_RASTER_ROWS],
                    len(shown),
                    f"Synthetic cohort (N = {len(shown):,}, filter: {choice})",
                ),
                width="stretch",
            )

        with st.container(border=True):
            head_left, head_right = st.columns([1.8, 1], vertical_alignment="center")
            with head_left:
                st.markdown("**Single-diary inspector**")
                st.caption("Step trajectory and time budget for one schedule in the batch.")
            with head_right:
                # The key carries the batch size, so a smaller batch starts a fresh
                # widget instead of restoring an index that no longer exists.
                if len(shown) <= 12:
                    sample_idx = st.segmented_control(
                        "Diary",
                        options=list(range(len(shown))),
                        default=0,
                        required=True,
                        format_func=lambda idx: f"#{idx + 1}",
                        key=f"inspector_seg_{len(shown)}",
                        label_visibility="collapsed",
                    )
                    sample_idx = 0 if sample_idx is None else int(sample_idx)
                else:
                    sample_idx = int(
                        st.number_input(
                            "Diary index",
                            min_value=1,
                            max_value=len(shown),
                            value=1,
                            step=1,
                            key=f"inspector_num_{len(shown)}",
                            label_visibility="collapsed",
                        )
                    ) - 1

            diary = shown[sample_idx]
            if choice != FILTER_OFF:
                st.caption(
                    f"Diary #{sample_idx + 1:,}: employment spells "
                    f"{count_spells(raw[sample_idx], 1)} raw to {count_spells(diary, 1)} filtered"
                )

            plot_col, table_col = st.columns([1.55, 1], vertical_alignment="top")
            with plot_col:
                st.image(
                    render_single_diary_png(
                        diary, "strip", f"Diary #{sample_idx + 1:,} · 24-hour timeline"
                    ),
                    width="stretch",
                )
                st.image(
                    render_single_diary_png(
                        diary, "step", f"Diary #{sample_idx + 1:,} · state transitions"
                    ),
                    width="stretch",
                )
            with table_col:
                st.dataframe(
                    build_breakdown_df(diary),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Code": st.column_config.NumberColumn("ID", width="small"),
                        "Activity division": st.column_config.TextColumn("ICATUS division"),
                        "Hours": st.column_config.NumberColumn("Hours", format="%.1f h"),
                        "Slots": st.column_config.NumberColumn("Slots", width="small"),
                        "Spells": st.column_config.NumberColumn("Spells", width="small"),
                        "Share of day": st.column_config.ProgressColumn(
                            "Share of 24 h", format="%.1f%%", min_value=0.0, max_value=100.0
                        ),
                    },
                )

    with tab_filter:
        with st.container(border=True):
            st.markdown("**Before and after morphological consolidation**")
            st.caption(
                "Fewer employment spells and fewer half-hour flickers mean smoother work blocks. "
                f"Counts cover all {len(raw):,} schedules in the batch."
            )
            st.dataframe(
                build_filter_comparison(raw),
                hide_index=True,
                width="stretch",
                column_config={
                    "Change vs raw": st.column_config.NumberColumn(format="%+d"),
                    "Mean employment (h/day)": st.column_config.NumberColumn(format="%.2f h"),
                },
            )

    with tab_eval:
        render_evaluation_gallery(eval_dir, version_label)


def render_evaluation_gallery(eval_dir: str, version_label: str) -> None:
    if not os.path.isdir(eval_dir):
        st.info(f"No evaluation directory was found for {version_label}.")
        return

    images = sorted(glob.glob(os.path.join(eval_dir, "*.png")))
    if not images:
        st.info(f"The evaluation directory for {version_label} holds no plots.")
        return

    st.caption(f"Benchmark diagnostics shipped with {version_label}.")
    columns = st.columns(2)
    for index, image_path in enumerate(images):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"**{pretty_name(os.path.splitext(os.path.basename(image_path))[0])}**")
                st.image(image_path, width="stretch")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="TUS-GAN workbench",
        page_icon=":material/schedule:",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("#### TUS-GAN workbench")
        st.caption("Conditioned synthesis of 24-hour time-use schedules")
        st.divider()

        version_choice = st.selectbox("Model checkpoint", list(MODEL_VERSIONS))
        version_key, version_label = MODEL_VERSIONS[version_choice]
        ckpt_path, data_path, eval_dir = resolve_version_paths(version_key)

        if not os.path.exists(ckpt_path):
            st.error(f"No checkpoint found for {version_label} at {os.path.relpath(ckpt_path)}.")
            st.stop()
        try:
            G, cfg, data = load_bundle(version_key, ckpt_path, data_path)
        except Exception as exc:  # surface load failures as a message, not a traceback
            st.error(f"Could not load {version_label}: {exc}")
            st.stop()

        st.caption(
            f"{data['cond_vector'].shape[0]:,} ITUS diaries · "
            f"{int(data['num_states'])} states · {int(data['num_districts'])} district codes"
        )
        st.divider()

        st.markdown("**Sampler**")
        num_samples = st.number_input(
            "Batch size", min_value=1, max_value=MAX_SAMPLES, value=10, step=10
        )
        gumbel_temp = st.slider(
            "Gumbel temperature (τ)", min_value=0.05, max_value=2.0, value=0.10, step=0.05
        )
        gumbel_hard = st.toggle("Hard one-hot discretisation", value=True)
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**31 - 1,
            value=0,
            step=1,
            help="The same profile and seed reproduce the same batch.",
        )

    st.markdown('<div class="tw-title">Synthetic time-use schedules</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tw-sub">48 half-hour slots · 9 ICATUS divisions · trained on '
        "Indian Time Use Survey 2019 microdata</p>",
        unsafe_allow_html=True,
    )

    num_states = int(data["num_states"])
    num_districts = int(data["num_districts"])
    state_options = {sid: STATE_NAMES.get(sid, f"State {sid}") for sid in range(num_states)}

    with st.form("conditioning", border=True):
        section_label("Persona")
        row_a, row_b, row_c = st.columns(3)
        with row_a:
            age_sel = st.selectbox("Age group", AGE_LABELS, index=min(3, len(AGE_LABELS) - 1))
            gender_sel = st.selectbox("Gender", GENDER_LABELS, index=0)
        with row_b:
            marital_sel = st.selectbox("Marital status", MARITAL_LABELS, index=0)
            edu_options = list(EDU_LABELS.values())
            edu_sel = st.selectbox("Education level", edu_options, index=min(8, len(edu_options) - 1))
        with row_c:
            act_options = list(ACT_LABELS.values())
            act_sel = st.selectbox("Principal activity", act_options, index=min(3, len(act_options) - 1))
            dow_sel = st.selectbox("Day of week", DOW_LABELS, index=0)

        section_label("Geography and household")
        row_d, row_e, row_f = st.columns(3)
        with row_d:
            state_id = st.selectbox(
                "State or union territory",
                list(range(num_states)),
                index=max(0, min(28, num_states - 1)),
                format_func=lambda sid: f"{state_options[sid]} (#{sid})",
            )
            district_map = DISTRICT_NAMES_BY_STATE.get(state_id, {})
            district_ids = (
                sorted(district_map) if district_map else list(range(num_districts))
            ) or [0]
            district_id = st.selectbox(
                "District",
                district_ids,
                index=0,
                format_func=lambda did: f"{district_map.get(did, f'District {did}')} (#{did})",
            )
        with row_e:
            sector_sel = st.segmented_control(
                "Sector", SECTOR_LABELS, default=SECTOR_LABELS[1], required=True
            ) or SECTOR_LABELS[1]
            caregiving_sel = st.toggle("Caregiving requirement (metadata)", value=False)
        with row_f:
            st.write("")
            submitted = st.form_submit_button(
                "Synthesise schedules", type="primary", icon=":material/bolt:", width="stretch"
            )

    needs_batch = (
        submitted
        or "generated" not in st.session_state
        or st.session_state["generated"].get("version_key") != version_key
    )
    if needs_batch:
        persona = {
            "age": AGE_LABELS.index(age_sel),
            "gender": GENDER_LABELS.index(gender_sel),
            "marital": MARITAL_LABELS.index(marital_sel),
            "education": edu_options.index(edu_sel),
            "activity": act_options.index(act_sel),
            "day_of_week": DOW_LABELS.index(dow_sel),
            "sector": SECTOR_LABELS.index(sector_sel),
        }
        try:
            decoded = sample_batch(
                G,
                cfg,
                data,
                num_samples=int(num_samples),
                district_id=district_id,
                state_id=state_id,
                gumbel_temp=gumbel_temp,
                gumbel_hard=gumbel_hard,
                persona=persona,
                seed=int(seed),
            )
        except Exception as exc:
            st.error(f"Sampling failed: {exc}")
            st.stop()

        st.session_state["generated"] = {
            "version_key": version_key,
            "decoded": decoded,
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
                "district_name": district_map.get(district_id, f"District {district_id}"),
                "gumbel_temperature": gumbel_temp,
                "gumbel_hard": gumbel_hard,
                "seed": int(seed),
            },
        }

    render_results(eval_dir, version_label, version_key)


if __name__ == "__main__":
    main()