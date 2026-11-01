"""
TUS-GAN Dashboard — Example B: Split-Pane Simulation Inspector
==============================================================
Design philosophy:
- Dedicated sidebar persona configurator (keeps main canvas 100% focused on data)
- Scales from 1 to 50,000 schedules using vectorized NumPy conditioning & chunked inference
- Top segmented navigation + compact export popover (no giant download buttons at bottom)
- Side-by-side master-detail split canvas: visual timelines on the left, quantitative
  time-budget dataframe and episode metrics on the right (zero vertical scrolling)

Run:
    streamlit run dashboard_example_b.py
"""

import glob
import itertools
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import (
    ACT_LABELS,
    ACTIVITY_LABELS,
    AGE_LABELS,
    DISTRICT_NAMES_BY_STATE,
    DOW_LABELS,
    EDU_LABELS,
    FILTER_HELP,
    FILTER_OFF,
    GENDER_LABELS,
    MARITAL_LABELS,
    SECTOR_LABELS,
    STATE_NAMES,
    count_spells,
    load_model_and_data,
)
from dashboard_example_a import (
    MAX_SAMPLES,
    MODEL_VERSIONS,
    build_breakdown_df,
    fast_diaries_to_csv_bytes,
    fast_diaries_to_json_bytes,
    render_cohort_raster_png,
    render_single_diary_png,
    resolve_version_paths,
    sample_batch,
)
from evaluation.postprocess import PRESETS, apply_preset


@st.fragment
def render_split_inspector(eval_dir: str, version_label: str):
    gen = st.session_state["generated"]
    raw = gen["decoded"]
    meta = gen["meta"]
    meta_items = tuple(meta.items())

    # Top navigation & actions bar
    nav_col, filter_col, exp_col = st.columns([1.3, 1.8, 0.7], vertical_alignment="bottom")
    with nav_col:
        view_mode = st.segmented_control(
            "Workspace view",
            ["Schedule inspector", "Cohort matrix", "Evaluation suite"],
            default="Schedule inspector",
            required=True,
            key="view_mode_b",
        )
    with filter_col:
        choice = st.segmented_control(
            "Employment post-processing",
            [FILTER_OFF, *PRESETS],
            default=FILTER_OFF,
            required=True,
            key="morph_choice_b",
            help=FILTER_HELP,
        )

    shown = raw if choice == FILTER_OFF else apply_preset(raw, choice)
    suffix = "" if choice == FILTER_OFF else "_" + choice.lower().replace(" ", "")

    with exp_col:
        with st.popover(f"Export ({len(shown):,})", icon=":material/download:", width="stretch"):
            st.markdown(f"**Export {len(shown):,} schedules**")
            st.caption(f"Model: {meta['model_version']} · Filter: {choice}")
            st.download_button(
                "Download CSV",
                data=fast_diaries_to_csv_bytes(shown, meta_items, choice),
                file_name=f"tusgan_diaries_{gen['timestamp']}{suffix}.csv",
                mime="text/csv",
                icon=":material/table_view:",
                key="dl_csv_b",
                width="stretch",
            )
            st.download_button(
                "Download JSON",
                data=fast_diaries_to_json_bytes(shown, meta_items, choice),
                file_name=f"tusgan_diaries_{gen['timestamp']}{suffix}.json",
                mime="application/json",
                icon=":material/data_object:",
                key="dl_json_b",
                width="stretch",
            )

    if view_mode == "Evaluation suite":
        if not os.path.isdir(eval_dir):
            st.warning(f"Evaluation directory `{eval_dir}` not found.", icon=":material/warning:")
            return
        images = sorted(glob.glob(os.path.join(eval_dir, "*.png")))
        if not images:
            st.warning(f"No evaluation plots found in `{eval_dir}`.", icon=":material/warning:")
            return
        cols = st.columns(2)
        for idx, img_path in enumerate(images):
            name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").capitalize()
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.image(img_path, width="stretch")
        return

    if view_mode == "Cohort matrix":
        with st.container(border=True):
            st.markdown(f"**Multi-agent 24-hour cohort schedule matrix (N = {len(shown):,})**")
            st.caption(
                f"{meta['gender']} · {meta['age_group']} · {meta['principal_activity']} · "
                f"{meta['sector']} · {meta['district_name']}, {meta['state_name']}"
            )
            st.image(
                render_cohort_raster_png(
                    shown,
                    f"Batch schedule raster ({meta['model_version']}, filter: {choice})",
                ),
                width="stretch",
            )
        return

    # Default view: Side-by-Side Master-Detail Schedule Inspector
    with st.container(border=True):
        bar_left, bar_right = st.columns([1.8, 1], vertical_alignment="center")
        with bar_left:
            if len(shown) <= 15:
                sample_idx = st.segmented_control(
                    "Active agent schedule",
                    options=list(range(len(shown))),
                    default=0,
                    format_func=lambda idx: f"Agent #{idx + 1}",
                    required=True,
                    key="agent_sel_b",
                )
            else:
                sample_idx = (
                    int(
                        st.number_input(
                            f"Active agent index (1 to {len(shown):,})",
                            min_value=1,
                            max_value=len(shown),
                            value=1,
                            step=1,
                            key="agent_num_b",
                        )
                    )
                    - 1
                )
        with bar_right:
            n_changed = int((shown != raw).any(axis=1).sum())
            st.caption(
                f"**Persona**: {meta['gender']}, {meta['age_group']} ({meta['sector']})  \n"
                f"**Location**: {meta['district_name']}, {meta['state_name']} · "
                f"**Modified by filter**: {n_changed:,}/{len(shown):,} diaries"
            )

    decoded_one = shown[sample_idx]
    raw_one = raw[sample_idx]

    # Single-agent KPI row
    emp_hrs = (decoded_one == 1).sum() * 0.5
    sleep_hrs = (decoded_one == 9).sum() * 0.5
    total_episodes = sum(1 for _ in itertools.groupby(decoded_one))
    raw_emp_spells = count_spells(raw_one, 1)
    filt_emp_spells = count_spells(decoded_one, 1)

    with st.container(horizontal=True):
        st.metric(
            f"Agent #{sample_idx + 1:,} employment",
            f"{emp_hrs:.1f} h",
            f"{filt_emp_spells} work spells (raw: {raw_emp_spells})"
            if choice != FILTER_OFF
            else f"{filt_emp_spells} work spells",
            delta_color="off",
            border=True,
        )
        st.metric(
            "Self-care & sleep",
            f"{sleep_hrs:.1f} h",
            f"{count_spells(decoded_one, 9)} episodes",
            delta_color="off",
            border=True,
        )
        st.metric(
            "Daily state transitions",
            f"{total_episodes} episodes",
            f"{total_episodes - 1} activity switches / 24h",
            delta_color="off",
            border=True,
        )
        st.metric(
            "Model & discretization",
            meta["model_version"],
            f"τ = {meta['gumbel_temperature']:.2f} ({'hard' if meta['gumbel_hard'] else 'soft'})",
            delta_color="off",
            border=True,
        )

    # Master-detail split pane
    pane_left, pane_right = st.columns([1.6, 1], vertical_alignment="top")
    with pane_left:
        with st.container(border=True):
            st.markdown(f"**Agent #{sample_idx + 1:,} — 24-hour diurnal trajectory**")
            st.image(
                render_single_diary_png(
                    decoded_one,
                    "strip",
                    "Chronological activity strip (04:00 to 03:30)",
                ),
                width="stretch",
            )
            st.image(
                render_single_diary_png(
                    decoded_one,
                    "step",
                    "ICATUS division transition step plot",
                ),
                width="stretch",
            )

    with pane_right:
        with st.container(border=True):
            st.markdown(f"**Agent #{sample_idx + 1:,} — 24-hour time allocation**")
            df_break = build_breakdown_df(decoded_one)
            st.dataframe(
                df_break,
                hide_index=True,
                width="stretch",
                height=385,
                column_config={
                    "Code": st.column_config.NumberColumn("ID", width="small"),
                    "Activity division": st.column_config.TextColumn("Activity"),
                    "Hours": st.column_config.NumberColumn("Hours", format="%.1f h"),
                    "Slots (30m)": None,
                    "Episodes": st.column_config.NumberColumn("Spells"),
                    "Share of day": st.column_config.ProgressColumn(
                        "Share",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                },
            )


def main():
    st.set_page_config(
        page_title="TUS-GAN Inspector (Example B)",
        page_icon=":material/analytics:",
        layout="wide",
    )

    # Sidebar: Complete structured persona & model configurator
    with st.sidebar:
        st.markdown("### :material/tune: Simulation configurator")
        st.caption("Example B · Sidebar controls + split-pane inspector")

        version_choice = st.selectbox("Model architecture", list(MODEL_VERSIONS.keys()), index=0)
        version_key, version_label = MODEL_VERSIONS[version_choice]
        ckpt_path, data_path, eval_dir = resolve_version_paths(version_key)
        G, cfg, data = load_model_and_data(version_key, ckpt_path, data_path)

        generate = st.button(
            "Synthesize batch",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )

        with st.expander("Demographic persona", expanded=True, icon=":material/person:"):
            age_sel = st.selectbox("Age group", AGE_LABELS, index=3)
            gender_sel = st.segmented_control(
                "Gender", GENDER_LABELS, default="Male", required=True
            )
            marital_sel = st.selectbox("Marital status", MARITAL_LABELS, index=0)
            edu_sel = st.selectbox("Education level", list(EDU_LABELS.values()), index=8)
            act_sel = st.selectbox("Principal activity", list(ACT_LABELS.values()), index=3)
            dow_sel = st.selectbox("Day of week", DOW_LABELS, index=0)
            sector_sel = st.segmented_control(
                "Sector", SECTOR_LABELS, default="Urban", required=True
            )
            caregiving_sel = st.toggle("Caregiving required", value=False)

        num_states = int(data["num_states"])
        num_districts = int(data["num_districts"])
        state_options = {sid: STATE_NAMES.get(sid, f"State {sid}") for sid in range(num_states)}

        with st.expander("Spatial conditioning", expanded=True, icon=":material/location_on:"):
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

        with st.expander("Gumbel-Softmax sampler", expanded=False, icon=":material/settings:"):
            num_samples = st.number_input(
                "Number of diaries (up to 50k)",
                min_value=1,
                max_value=MAX_SAMPLES,
                value=10,
                step=10,
            )
            gumbel_temp = st.slider("Temperature (τ)", 0.05, 2.0, 0.10, 0.05)
            gumbel_hard = st.toggle("Hard discretization", value=True)

    st.subheader("TUS-GAN microsimulation schedule inspector")

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

    render_split_inspector(eval_dir, version_label)


if __name__ == "__main__":
    main()
