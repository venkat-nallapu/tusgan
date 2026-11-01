"""TUS-GAN schedule workbench.

Generates 24-hour activity schedules from a TUS-GAN checkpoint, conditioned on a
person profile, a day type and a location. Schedules can be filtered, inspected
and exported.

Run with:
    streamlit run dashboard_example_a.py
"""

import glob
import io
import os
import re
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

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

ROOT = os.path.dirname(os.path.abspath(__file__))

MAX_SAMPLES = 50_000
CHUNK_SIZE = 1024
MAX_RASTER_ROWS = 100
PICKER_BUTTON_LIMIT = 12

N_SLOTS = 48
SLOT_HOURS = 0.5
EMPLOYMENT = 1
DEFAULT_STATE_ID = 28

MODEL_VERSIONS = {
    "v4 (production)": "v4",
    "v3 (baseline)": "v3",
    "v5 (transformer)": "v5",
}

# One-hot blocks at the start of the conditioning vector, as (offset, width).
# Columns after COND_ONE_HOT_WIDTH are copied from randomly drawn real records.
COND_BLOCKS = {
    "age": (0, 7),
    "gender": (7, 3),
    "marital": (10, 4),
    "education": (14, 11),
    "activity": (25, 13),
    "day": (38, 7),
    "sector": (45, 2),
}
COND_ONE_HOT_WIDTH = 47

INK = "#27272a"
MUTED = "#52525b"
RULE = "#d4d4d8"
PLOT_DPI = 140


def resolve_version_paths(version_key: str):
    ckpt_path = os.path.join(ROOT, version_key, "checkpoints", "final.pt")
    data_path = os.path.join(ROOT, version_key, "tusgan_encode.npz")
    if not os.path.exists(data_path) and os.path.exists(DATA_NPZ_PATH):
        data_path = DATA_NPZ_PATH
    if version_key == "v5":
        eval_dir = os.path.join(ROOT, "v5", "results")
        if not os.path.exists(eval_dir):
            eval_dir = os.path.join(ROOT, "v5", "evaluation_v5_100k")
    else:
        eval_dir = os.path.join(ROOT, version_key, "evaluation_results")
    return ckpt_path, data_path, eval_dir


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def build_conditioned_batch(all_cond: np.ndarray, num_samples: int, persona: dict) -> np.ndarray:
    """Draw real conditioning rows, then overwrite the one-hot profile block."""
    if all_cond.shape[1] < COND_ONE_HOT_WIDTH:
        raise ValueError(
            f"Conditioning vector has {all_cond.shape[1]} columns; "
            f"at least {COND_ONE_HOT_WIDTH} are required."
        )
    rows = np.random.randint(0, all_cond.shape[0], size=num_samples)
    batch = all_cond[rows]
    batch[:, :COND_ONE_HOT_WIDTH] = 0.0
    for name, (offset, width) in COND_BLOCKS.items():
        batch[:, offset + min(persona[name], width - 1)] = 1.0
    return batch


def sample_batch(
    G,
    cfg,
    data,
    *,
    num_samples: int,
    state_id: int,
    district_id: int,
    gumbel_temp: float,
    gumbel_hard: bool,
    persona: dict,
) -> np.ndarray:
    """Generate ``num_samples`` schedules in mini-batches. Returns an (N, 48) array of codes 1-9."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = G.to(device)
    cond = build_conditioned_batch(data["cond_vector"], num_samples, persona)
    decoded = np.empty((num_samples, N_SLOTS), dtype=np.int64)

    progress = None
    if num_samples > CHUNK_SIZE:
        progress = st.progress(0.0, text=f"Generating {num_samples:,} schedules")

    try:
        with torch.no_grad():
            for start in range(0, num_samples, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, num_samples)
                b = end - start
                z = torch.randn(b, cfg["noise_dim"], device=device)
                cv = torch.from_numpy(cond[start:end]).float().to(device)
                di = torch.full((b,), int(district_id), dtype=torch.long, device=device)
                si = torch.full((b,), int(state_id), dtype=torch.long, device=device)
                fake = G(z, cv, di, si, temp=gumbel_temp, hard=gumbel_hard)  # (b, 9, 48, 1)
                scores = fake.squeeze(-1).cpu().numpy()  # (b, 9, 48)
                decoded[start:end] = np.argmax(scores, axis=1) + 1
                if progress is not None:
                    progress.progress(
                        end / num_samples,
                        text=f"Generating schedules ({end:,} of {num_samples:,})",
                    )
    finally:
        if progress is not None:
            progress.empty()
    return decoded


# ---------------------------------------------------------------------------
# Batch metrics (vectorized)
# ---------------------------------------------------------------------------


def employment_hours(arr: np.ndarray) -> np.ndarray:
    return (arr == EMPLOYMENT).sum(axis=1) * SLOT_HOURS


def count_spells_batch(arr: np.ndarray, code: int = EMPLOYMENT) -> int:
    """Number of contiguous runs of ``code`` across an (N, 48) array."""
    hit = arr == code
    starts = hit.copy()
    starts[:, 1:] &= ~hit[:, :-1]
    return int(starts.sum())


def count_flickers_batch(arr: np.ndarray, code: int = EMPLOYMENT) -> int:
    """Number of runs of ``code`` that last exactly one slot."""
    hit = np.pad(arr == code, ((0, 0), (1, 1)), constant_values=False)
    isolated = hit[:, 1:-1] & ~hit[:, :-2] & ~hit[:, 2:]
    return int(isolated.sum())


def mean_episodes_batch(arr: np.ndarray) -> float:
    """Mean number of activity episodes per schedule, counting every activity type."""
    changes = (arr[:, 1:] != arr[:, :-1]).sum(axis=1)
    return float((1 + changes).mean())


def activity_breakdown(schedule: np.ndarray) -> pd.DataFrame:
    rows = []
    for code in sorted(ACTIVITY_LABELS):
        slots = int((schedule == code).sum())
        rows.append(
            {
                "Code": code,
                "Activity division": ACTIVITY_LABELS[code],
                "Hours": slots * SLOT_HOURS,
                "Slots": slots,
                "Episodes": count_spells(schedule, code=code),
                "Share of day": 100.0 * slots / N_SLOTS,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Export
#
# Arguments that start with an underscore are not hashed by st.cache_data. The
# cache is keyed on gen_id (unique per generation) and the filter name instead,
# so large arrays are never hashed or sampled.
# ---------------------------------------------------------------------------


def export_frame(decoded: np.ndarray, meta: dict, filter_name: str) -> pd.DataFrame:
    head = pd.DataFrame({"sample_id": np.arange(1, decoded.shape[0] + 1)})
    for key, value in meta.items():
        head[key] = value
    head["employment_filter"] = filter_name
    slots = pd.DataFrame(decoded, columns=TIME_SLOTS)
    return pd.concat([head, slots], axis=1)


@st.cache_data(max_entries=4, show_spinner="Preparing CSV")
def csv_bytes(_decoded: np.ndarray, _meta: dict, gen_id: str, filter_name: str) -> bytes:
    return export_frame(_decoded, _meta, filter_name).to_csv(index=False).encode("utf-8")


@st.cache_data(max_entries=4, show_spinner="Preparing JSON")
def json_bytes(_decoded: np.ndarray, _meta: dict, gen_id: str, filter_name: str) -> bytes:
    indent = 2 if _decoded.shape[0] <= 500 else 0
    frame = export_frame(_decoded, _meta, filter_name)
    return frame.to_json(orient="records", indent=indent).encode("utf-8")


def filename_suffix(choice: str) -> str:
    if choice == FILTER_OFF:
        return ""
    return "_" + re.sub(r"[^a-z0-9]+", "-", choice.lower()).strip("-")


# ---------------------------------------------------------------------------
# Plots
#
# Figures are built with matplotlib.figure.Figure rather than pyplot, because
# pyplot keeps global state that is not safe across Streamlit's script threads.
# ---------------------------------------------------------------------------


def build_rgb_lut() -> np.ndarray:
    lut = np.full((max(ACTIVITY_COLORS) + 1, 3), 0.85, dtype=np.float32)
    for code, colour in ACTIVITY_COLORS.items():
        lut[code] = to_rgb(colour)
    return lut


RGB_LUT = build_rgb_lut()


def to_rgb_image(codes: np.ndarray) -> np.ndarray:
    return RGB_LUT[np.clip(codes, 0, len(RGB_LUT) - 1)]


def style_time_axis(ax) -> None:
    ticks = list(range(0, N_SLOTS, 4))
    ax.set_xlim(0, N_SLOTS)
    ax.set_xticks(ticks)
    ax.set_xticklabels([TIME_SLOTS[i] for i in ticks], fontsize=8.5, color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)


def figure_to_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=PLOT_DPI, bbox_inches="tight")
    return buf.getvalue()


@st.cache_data(max_entries=32, show_spinner=False)
def raster_png(_decoded: np.ndarray, gen_id: str, filter_name: str) -> bytes:
    """One row per schedule, one column per half hour, coloured by activity."""
    n = _decoded.shape[0]
    fig = Figure(figsize=(12, max(2.4, min(7.0, 0.26 * n + 1.4))))
    ax = fig.subplots()
    ax.imshow(
        to_rgb_image(_decoded),
        aspect="auto",
        interpolation="nearest",
        extent=(0, N_SLOTS, n - 0.5, -0.5),
    )
    if n <= 30:
        ax.hlines(np.arange(n + 1) - 0.5, 0, N_SLOTS, color="white", linewidth=0.8)
    ax.vlines(range(0, N_SLOTS + 1, 4), -0.5, n - 0.5, color="white", linewidth=0.5, alpha=0.6)
    ax.set_ylim(n - 0.5, -0.5)

    style_time_axis(ax)
    step = 1 if n <= 20 else max(1, n // 10)
    rows = list(range(0, n, step))
    ax.set_yticks(rows)
    ax.set_yticklabels([str(i + 1) for i in rows], fontsize=8, color=MUTED)
    ax.set_ylabel("Schedule", fontsize=8.5, color=MUTED)
    ax.tick_params(axis="y", length=0)

    ax.legend(
        handles=[
            Patch(color=ACTIVITY_COLORS[k], label=f"{k}. {ACTIVITY_LABELS[k]}")
            for k in sorted(ACTIVITY_LABELS)
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        handlelength=1.2,
    )
    return figure_to_png(fig)


@st.cache_data(max_entries=128, show_spinner=False)
def diary_png(_decoded: np.ndarray, kind: str, gen_id: str, filter_name: str, idx: int) -> bytes:
    """Single schedule as a colour strip (kind='strip') or a step trace (kind='step')."""
    if kind == "strip":
        fig = Figure(figsize=(12, 1.1))
        ax = fig.subplots()
        ax.imshow(
            to_rgb_image(_decoded[np.newaxis, :]),
            aspect="auto",
            interpolation="nearest",
            extent=(0, N_SLOTS, 0.5, -0.5),
        )
        ax.vlines(range(N_SLOTS + 1), -0.5, 0.5, color="white", linewidth=0.4)
        ax.set_ylim(0.5, -0.5)
        ax.set_yticks([])
        style_time_axis(ax)
        return figure_to_png(fig)

    codes = sorted(ACTIVITY_LABELS)
    low, high = codes[0] - 0.5, codes[-1] + 0.5
    fig = Figure(figsize=(12, 3.2))
    ax = fig.subplots()
    ax.bar(
        np.arange(N_SLOTS) + 0.5,
        high - low,
        bottom=low,
        width=1.0,
        color=[ACTIVITY_COLORS.get(int(c), "#f4f4f5") for c in _decoded],
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )
    ax.step(
        np.arange(N_SLOTS + 1),
        np.append(_decoded, _decoded[-1]),
        where="post",
        color=INK,
        linewidth=1.6,
        zorder=3,
    )
    ax.set_ylim(low, high)
    ax.set_yticks(codes)
    ax.set_yticklabels([f"{k}. {ACTIVITY_LABELS[k]}" for k in codes], fontsize=8, color=MUTED)
    ax.grid(axis="y", linestyle=":", alpha=0.4, color="#a1a1aa")
    style_time_axis(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(RULE)
    return figure_to_png(fig)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def filtered_view(gen: dict, choice: str) -> np.ndarray:
    """Return the schedules with the chosen filter applied, computed once per generation."""
    if choice == FILTER_OFF:
        return gen["decoded"]
    if choice not in gen["filtered"]:
        gen["filtered"][choice] = apply_preset(gen["decoded"], choice)
    return gen["filtered"][choice]


def render_benchmarks(eval_dir: str, model_version: str) -> None:
    shown_dir = os.path.relpath(eval_dir, ROOT)
    if not os.path.isdir(eval_dir):
        st.warning(f"No benchmark folder for {model_version}. Expected `{shown_dir}`.")
        return
    images = sorted(glob.glob(os.path.join(eval_dir, "*.png")))
    if not images:
        st.warning(f"No PNG plots for {model_version} in `{shown_dir}`.")
        return

    st.caption(f"Pre-computed benchmark plots for {model_version}.")
    columns = st.columns(2)
    for i, path in enumerate(images):
        name = os.path.splitext(os.path.basename(path))[0].replace("_", " ").capitalize()
        with columns[i % 2]:
            st.image(path, caption=name, width="stretch")


@st.fragment
def render_results() -> None:
    gen = st.session_state["generated"]
    raw, meta, gen_id = gen["decoded"], gen["meta"], gen["id"]

    filter_col, export_col = st.columns([3, 2], vertical_alignment="bottom")
    with filter_col:
        choice = st.segmented_control(
            "Employment filter",
            [FILTER_OFF, *PRESETS],
            default=FILTER_OFF,
            required=True,
            key="morph_choice",
            help=FILTER_HELP,
        )
    shown = filtered_view(gen, choice)
    n = len(shown)

    with export_col:
        csv_col, json_col = st.columns(2)
        stem = f"tusgan_{gen['version_key']}_{n}schedules_{gen['timestamp']}{filename_suffix(choice)}"
        with csv_col:
            st.download_button(
                "Download CSV",
                data=csv_bytes(shown, meta, gen_id, choice),
                file_name=f"{stem}.csv",
                mime="text/csv",
                on_click="ignore",
                key="dl_csv",
                width="stretch",
            )
        with json_col:
            st.download_button(
                "Download JSON",
                data=json_bytes(shown, meta, gen_id, choice),
                file_name=f"{stem}.json",
                mime="application/json",
                on_click="ignore",
                key="dl_json",
                width="stretch",
            )

    hours = employment_hours(shown)
    raw_episodes = count_spells_batch(raw)
    episodes = count_spells_batch(shown)
    changed = int((shown != raw).any(axis=1).sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Schedules", f"{n:,}")
        st.caption(f"{meta['district_name']}, {meta['state_name']}")
    with k2:
        st.metric("Mean employment", f"{hours.mean():.2f} h/day")
        st.caption(f"{int((hours > 0).sum()):,} of {n:,} schedules include employment")
    with k3:
        st.metric("Employment episodes", f"{episodes:,}")
        if choice == FILTER_OFF:
            st.caption("Unfiltered output")
        else:
            st.caption(f"{episodes - raw_episodes:+,} vs unfiltered, {changed:,} schedules changed")
    with k4:
        st.metric("Mean episodes per schedule", f"{mean_episodes_batch(shown):.1f}")
        st.caption("All activity types")

    tab_schedules, tab_filter, tab_bench = st.tabs(
        ["Schedules", "Filter comparison", "Model benchmarks"]
    )

    with tab_schedules:
        rows = min(n, MAX_RASTER_ROWS)
        st.markdown("**Activity by half hour**")
        st.caption(
            f"First {rows} of {n:,} schedules, one row each."
            if n > rows
            else f"All {n:,} schedules, one row each."
        )
        st.image(raster_png(shown[:rows], gen_id, choice), width="stretch")

        st.divider()
        head_col, pick_col = st.columns([1, 1], vertical_alignment="bottom")
        with head_col:
            st.markdown("**Single schedule**")
        pick_key = f"pick_{gen_id}"
        with pick_col:
            if n == 1:
                idx = 0
            elif n <= PICKER_BUTTON_LIMIT:
                idx = st.segmented_control(
                    "Schedule",
                    list(range(n)),
                    default=0,
                    required=True,
                    format_func=lambda i: str(i + 1),
                    key=pick_key,
                )
            else:
                idx = (
                    int(
                        st.number_input(
                            f"Schedule number (1 to {n:,})",
                            min_value=1,
                            max_value=n,
                            value=1,
                            step=1,
                            key=pick_key,
                        )
                    )
                    - 1
                )

        schedule = shown[idx]
        if choice != FILTER_OFF:
            st.caption(
                f"Employment episodes in schedule {idx + 1:,}: "
                f"{count_spells(raw[idx], 1)} unfiltered, {count_spells(schedule, 1)} filtered."
            )

        plot_col, table_col = st.columns([1.55, 1])
        with plot_col:
            st.caption("Timeline")
            st.image(diary_png(schedule, "strip", gen_id, choice, idx), width="stretch")
            st.caption("Activity code over the day")
            st.image(diary_png(schedule, "step", gen_id, choice, idx), width="stretch")
        with table_col:
            st.caption("Time budget")
            st.dataframe(
                activity_breakdown(schedule),
                hide_index=True,
                width="stretch",
                height=368,
                column_config={
                    "Code": st.column_config.NumberColumn("Code", width="small"),
                    "Activity division": st.column_config.TextColumn("ICATUS division"),
                    "Hours": st.column_config.NumberColumn("Hours", format="%.1f h"),
                    "Share of day": st.column_config.ProgressColumn(
                        "Share of day", format="%.1f%%", min_value=0.0, max_value=100.0
                    ),
                },
            )

    with tab_filter:
        st.markdown("**Unfiltered output against each filter preset**")
        st.caption(
            "The bounded 1D morphological filter (manuscript section 6.4.5), "
            f"applied to all {len(raw):,} schedules."
        )
        variants = [("Unfiltered", raw)] + [(name, filtered_view(gen, name)) for name in PRESETS]
        table = pd.DataFrame(
            [
                {
                    "Configuration": name,
                    "Schedules changed": int((arr != raw).any(axis=1).sum()),
                    "Employment episodes": count_spells_batch(arr),
                    "Mean employment (h/day)": float(employment_hours(arr).mean()),
                    "Single-slot employment episodes": count_flickers_batch(arr),
                }
                for name, arr in variants
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Mean employment (h/day)": st.column_config.NumberColumn(format="%.2f h"),
            },
        )

    with tab_bench:
        render_benchmarks(gen["eval_dir"], gen["model_version"])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def select_index(label: str, labels, default: int = 0) -> int:
    """Selectbox over a label list that returns the chosen position."""
    labels = list(labels)
    return st.selectbox(
        label,
        range(len(labels)),
        index=min(default, len(labels) - 1),
        format_func=lambda i: labels[i],
    )


def main() -> None:
    st.set_page_config(page_title="TUS-GAN workbench", page_icon=":material/schedule:", layout="wide")

    with st.sidebar:
        st.subheader("Model")
        version_choice = st.selectbox("Checkpoint", list(MODEL_VERSIONS))
        version_key = MODEL_VERSIONS[version_choice]
        version_label = f"TUS-GAN {version_key}"
        ckpt_path, data_path, eval_dir = resolve_version_paths(version_key)
        try:
            G, cfg, data = load_model_and_data(version_key, ckpt_path, data_path)
        except FileNotFoundError as exc:
            st.error(f"Could not load {version_label}. File not found: {exc.filename or exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Could not load {version_label}: {exc}")
            st.stop()

        st.subheader("Sampler")
        num_samples = st.number_input(
            "Schedules to generate",
            min_value=1,
            max_value=MAX_SAMPLES,
            value=10,
            step=10,
            help=f"Between 1 and {MAX_SAMPLES:,}.",
        )
        gumbel_temp = st.slider("Gumbel temperature (τ)", 0.05, 2.0, 0.10, 0.05)
        gumbel_hard = st.toggle("Hard one-hot output", value=True)

        st.caption(
            f"Reference data: {data['cond_vector'].shape[0]:,} ITUS diaries, "
            f"{int(data['num_states'])} states and UTs, {int(data['num_districts'])} district codes."
        )

    st.header("Schedule workbench")
    st.caption(
        "Generate 24-hour activity schedules for a person profile, day type and location. "
        "Each schedule has 48 half-hour slots across 9 ICATUS activity divisions, learned from "
        "the 2019 Indian Time Use Survey."
    )

    num_states = int(data["num_states"])
    num_districts = int(data["num_districts"])
    state_names = {sid: STATE_NAMES.get(sid, f"State {sid}") for sid in range(num_states)}
    edu_labels = list(EDU_LABELS.values())
    act_labels = list(ACT_LABELS.values())
    urban = SECTOR_LABELS.index("Urban") if "Urban" in SECTOR_LABELS else 0

    with st.container(border=True):
        person_col, day_col, place_col = st.columns(3)
        with person_col:
            st.markdown("**Person**")
            age = select_index("Age group", AGE_LABELS, default=3)
            gender = select_index("Gender", GENDER_LABELS)
            marital = select_index("Marital status", MARITAL_LABELS)
            education = select_index("Education level", edu_labels, default=8)
        with day_col:
            st.markdown("**Day**")
            activity = select_index("Principal activity", act_labels, default=3)
            day = select_index("Day of week", DOW_LABELS)
            sector = st.segmented_control(
                "Sector",
                range(len(SECTOR_LABELS)),
                default=urban,
                required=True,
                format_func=lambda i: SECTOR_LABELS[i],
            )
        with place_col:
            st.markdown("**Place**")
            state_id = st.selectbox(
                "State or UT",
                range(num_states),
                index=min(DEFAULT_STATE_ID, num_states - 1),
                format_func=lambda sid: state_names[sid],
            )
            district_map = DISTRICT_NAMES_BY_STATE.get(state_id, {})
            district_ids = sorted(district_map) if district_map else list(range(num_districts))
            district_id = st.selectbox(
                "District",
                district_ids,
                format_func=lambda did: district_map.get(did, f"District {did}"),
                key=f"district_{state_id}",
            )
        generate = st.button("Generate schedules", type="primary")

    persona = {
        "age": age,
        "gender": gender,
        "marital": marital,
        "education": education,
        "activity": activity,
        "day": day,
        "sector": sector,
    }
    signature = (
        version_key,
        int(num_samples),
        float(gumbel_temp),
        bool(gumbel_hard),
        tuple(persona.items()),
        int(state_id),
        int(district_id),
    )

    if generate or "generated" not in st.session_state:
        decoded = sample_batch(
            G,
            cfg,
            data,
            num_samples=int(num_samples),
            state_id=state_id,
            district_id=district_id,
            gumbel_temp=gumbel_temp,
            gumbel_hard=gumbel_hard,
            persona=persona,
        )
        st.session_state["generated"] = {
            "id": uuid.uuid4().hex,
            "decoded": decoded,
            "filtered": {},
            "signature": signature,
            "version_key": version_key,
            "model_version": version_label,
            "eval_dir": eval_dir,
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "meta": {
                "model_version": version_label,
                "age_group": AGE_LABELS[age],
                "gender": GENDER_LABELS[gender],
                "marital_status": MARITAL_LABELS[marital],
                "education": edu_labels[education],
                "principal_activity": act_labels[activity],
                "day_of_week": DOW_LABELS[day],
                "sector": SECTOR_LABELS[sector],
                "state_id": state_id,
                "state_name": state_names[state_id],
                "district_id": district_id,
                "district_name": district_map.get(district_id, f"District {district_id}"),
                "gumbel_temperature": gumbel_temp,
                "gumbel_hard": gumbel_hard,
            },
        }

    gen = st.session_state["generated"]
    meta = gen["meta"]

    if gen["signature"] != signature:
        st.info(
            "The settings have changed since these schedules were generated. "
            "Select Generate schedules to apply them."
        )

    st.caption(
        f"{meta['model_version']}. {meta['gender']}, {meta['age_group']}, "
        f"{meta['principal_activity']}, {meta['sector']}, {meta['day_of_week']}. "
        f"{meta['district_name']}, {meta['state_name']}. "
        f"Gumbel τ {meta['gumbel_temperature']:.2f}, "
        f"{'hard' if meta['gumbel_hard'] else 'soft'} output."
    )
    render_results()


if __name__ == "__main__":
    main()