from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from batteryops.audit import build_public_readiness_audit
from batteryops.dashboard import (
    BATTERYOPS_COLORS,
    DashboardData,
    build_alert_coverage_figure,
    build_anomaly_timeline_figure,
    build_artifact_inventory_frame,
    build_asset_error_figure,
    build_asset_priority_table,
    build_asset_risk_driver_figure,
    build_asset_risk_driver_frame,
    build_cockpit_decision_frame,
    build_confusion_matrix_figure,
    build_feature_importance_figure,
    build_flagged_cycle_table,
    build_fleet_risk_concentration_figure,
    build_fleet_risk_figure,
    build_focus_incident_row,
    build_health_overview_figure,
    build_incident_markdown,
    build_incident_marker_figure,
    build_recent_cycle_table,
    build_replay_figure,
    build_report_confidence_figure,
    build_rul_confidence_figure,
    build_rul_scatter_figure,
    build_similar_cases_figure,
    build_similar_cases_frame,
    build_threshold_tradeoff_figure,
    build_triage_handoff_frame,
    load_dashboard_data,
)

REPLAY_STEP_SIZES = {"1 cycle": 1, "4 cycles": 4, "10 cycles": 10}
FOCUS_ASSET_STATE_KEY = "selected_focus_asset"


def load_demo_payload() -> tuple[dict[str, Any], pd.DataFrame]:
    """Return the summary and focus-asset timeline used by the Streamlit app."""
    data = load_dashboard_data()
    return data.summary, data.timeline


@st.cache_data(show_spinner=False)
def _load_public_readiness_audit() -> dict[str, Any]:
    return build_public_readiness_audit()


def main() -> None:
    """Render the public-facing BatteryOps dashboard."""
    st.set_page_config(
        page_title="BatteryOps",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()

    data = load_dashboard_data(_requested_focus_asset())
    selected_asset_id = _render_sidebar(data)
    if selected_asset_id != data.focus_asset_id:
        data = load_dashboard_data(selected_asset_id)
    _render_header(data)

    cockpit_tab, replay_tab, evidence_tab, similar_tab, eval_tab, provenance_tab = st.tabs(
        [
            "Fleet Cockpit",
            "Asset Replay",
            "Incident Evidence",
            "Similar Cases",
            "Model Evaluation",
            "Data & Provenance",
        ]
    )

    with cockpit_tab:
        _render_overview_tab(data)
    with replay_tab:
        _render_replay_tab(data)
    with evidence_tab:
        _render_report_tab(data)
    with similar_tab:
        _render_similar_cases_tab(data)
    with eval_tab:
        _render_evaluation_tab(data)
    with provenance_tab:
        _render_provenance_tab(data)


def _render_header(data: DashboardData) -> None:
    st.title("BatteryOps")
    st.caption(
        "Zero-cost ML engineering demo for battery telemetry triage on public NASA data."
    )
    st.markdown(
        """
        <div class="hero-card">
          <p class="hero-kicker">Public NASA data · local models · zero paid services</p>
          <h3>Fleet-level battery triage with validated artifacts and explainable alerts</h3>
          <p>
            BatteryOps packages inspectable ML system evidence in one local app:
            reproducible preprocessing, saved model artifacts, validated bundle IDs,
            threshold tradeoffs, incident retrieval, and no API keys or paid runtime.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Runtime provenance: {data.dataset_status.source_label}")
    _render_reviewer_proof_strip(data)
    _render_focus_asset_picker(data)

    _render_header_metrics(data)


def _render_focus_asset_picker(data: DashboardData) -> None:
    asset_options = sorted(data.fleet_timeline["asset_id"].astype(str).unique())
    if not asset_options:
        return

    if len(asset_options) == 1:
        st.caption(f"Focus asset: {data.focus_asset_id}")
        return

    session_asset = st.session_state.get(FOCUS_ASSET_STATE_KEY)
    current_asset_id = (
        session_asset
        if isinstance(session_asset, str) and session_asset in asset_options
        else data.focus_asset_id
    )
    st.selectbox(
        "Focus asset",
        asset_options,
        index=(
            asset_options.index(current_asset_id)
            if current_asset_id in asset_options
            else 0
        ),
        help="Choose which asset drives the cockpit and replay views.",
        key=FOCUS_ASSET_STATE_KEY,
    )


def _render_reviewer_proof_strip(data: DashboardData) -> None:
    audit = _load_public_readiness_audit()
    asset_count = int(data.metrics.get("asset_count", data.fleet_timeline["asset_id"].nunique()))
    cycle_count = int(data.metrics.get("cycle_count", len(data.fleet_timeline)))
    precision = float(data.metrics.get("alert_precision", 0.0)) * 100
    recall = float(data.metrics.get("alert_recall", 0.0)) * 100
    fingerprint = escape(_short_fingerprint(data.bundle_fingerprint))
    st.markdown(
        f"""
        <div class="proof-strip">
          <div class="proof-item">
            <span>Validated bundle</span>
            <strong>{asset_count} assets, {cycle_count} cycles</strong>
            <small>Fingerprint {fingerprint}</small>
          </div>
          <div class="proof-item">
            <span>Evaluation evidence</span>
            <strong>{precision:.1f}% precision / {recall:.1f}% recall</strong>
            <small>Held-out proxy alert metrics</small>
          </div>
          <div class="proof-item">
            <span>Public readiness</span>
            <strong>{int(audit["pass_count"])}/{len(audit["checks"])} checks pass</strong>
            <small>Includes screenshot and secret-pattern checks</small>
          </div>
          <div class="proof-item">
            <span>Cost boundary</span>
            <strong>$0 runtime</strong>
            <small>No APIs, keys, database, or paid inference</small>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header_metrics(data: DashboardData) -> None:
    inspect_queue = int(data.fleet_timeline["predicted_alert"].sum())
    metric_specs: tuple[tuple[str, str | int | float, str, list[float] | None], ...] = (
        (
            "Fleet assets",
            int(data.metrics.get("asset_count", data.fleet_timeline["asset_id"].nunique())),
            "Assets represented in the validated demo bundle.",
            None,
        ),
        (
            "Saved cycles",
            int(data.metrics.get("cycle_count", len(data.fleet_timeline))),
            "Cycle-level snapshots available for triage and evaluation.",
            None,
        ),
        (
            "Inspect queue",
            inspect_queue,
            "Cycles currently marked inspect-soon by the saved anomaly threshold.",
            data.fleet_timeline["anomaly_score"].tail(12).tolist(),
        ),
        (
            "Alert precision",
            f"{float(data.metrics.get('alert_precision', 0.0)) * 100:.1f}%",
            "Held-out proxy alert precision for the loaded artifact.",
            None,
        ),
        (
            "Proxy RUL MAE",
            f"{float(data.metrics.get('rul_proxy_mae', data.metrics.get('rul_mae', 0.0))):.2f}",
            "Held-out mean absolute error for the degradation-threshold horizon proxy.",
            None,
        ),
        (
            "Runtime cost",
            "$0",
            "No paid APIs, hosted databases, or metered services are used.",
            None,
        ),
    )
    _render_metric_grid(metric_specs, columns_per_row=3)


def _render_metric_grid(
    metric_specs: tuple[tuple[str, str | int | float, str, list[float] | None], ...],
    *,
    columns_per_row: int,
) -> None:
    for start in range(0, len(metric_specs), columns_per_row):
        row_specs = metric_specs[start : start + columns_per_row]
        row_columns = st.columns(len(row_specs))
        for column, (label, value, help_text, _chart_data) in zip(
            row_columns,
            row_specs,
            strict=True,
        ):
            column.metric(label, value, help=help_text, border=True)


def _render_replay_metrics(current_row: pd.Series, replay_cycle: int) -> None:
    metric_specs: tuple[tuple[str, str | int | float, str, list[float] | None], ...] = (
        (
            "Replay cycle",
            replay_cycle,
            "Current cycle shown by the replay cursor for this asset.",
            None,
        ),
        (
            "Anomaly score",
            f"{float(current_row['anomaly_score']):.3f}",
            "Saved anomaly score for the cycle currently in focus.",
            None,
        ),
        (
            "Proxy RUL",
            f"{float(current_row['predicted_rul_cycles']):.1f} cycles",
            "Heuristic proxy RUL estimate from the artifact (for this cycle).",
            None,
        ),
        (
            "Alert marker",
            "Threshold breach" if bool(current_row["predicted_alert"]) else "Monitor",
            "Whether the current replay cycle exceeds the current threshold.",
            None,
        ),
    )
    _render_metric_grid(metric_specs, columns_per_row=2)


def _render_sidebar(data: DashboardData) -> str:
    with st.sidebar:
        st.subheader("Runtime Context")
        st.caption(
            "Local-first by default: no backend service, no auth layer, no paid integrations."
        )

        with st.container(border=True):
            st.markdown("**Zero-cost contract**")
            st.write("Runtime cost: `$0`")
            st.write("External APIs: `none`")
            st.write("API keys or secrets: `none`")

        with st.container(border=True):
            st.markdown("**Runtime provenance**")
            st.write(f"Runtime source: `{data.dataset_status.runtime_source_label}`")
            st.caption(data.dataset_status.detail)
            if data.dataset_status.full_dataset_available:
                st.success("Loaded from the processed local parquet cache.")
            else:
                if data.dataset_status.runtime_source_label == "checked-in demo bundle":
                    st.info(
                        "This session is using checked-in demo metadata without a processed "
                        "local cache."
                    )
                else:
                    st.info("This session is using synthetic demo fallback artifacts.")

        with st.container(border=True):
            st.markdown("**Saved artifact scope**")
            selected_asset_id = data.focus_asset_id
            st.write(f"Focus asset: `{selected_asset_id}`.")
            st.caption(
                "Incident Evidence, Similar Cases, and Model Evaluation stay tied to the "
                "same loaded bundle."
            )
            asset_count = int(
                data.metrics.get("asset_count", data.fleet_timeline["asset_id"].nunique())
            )
            cycle_count = int(data.metrics.get("cycle_count", len(data.fleet_timeline)))
            incident_count = int(data.metrics.get("incident_case_count", len(data.incidents)))
            st.write(
                f"{asset_count} assets, {cycle_count} saved cycle snapshots, "
                f"{incident_count} incident windows."
            )
            st.write(
                f"Default anomaly threshold: `{float(data.anomaly_threshold):.2f}` "
                f"for asset `{selected_asset_id}`."
            )

        with st.container(border=True):
            st.markdown("**Tab guide**")
            st.write(
                "The six tabs walk from fleet triage to replay, incident evidence, retrieval, "
                "offline evaluation, and provenance."
            )
        with st.expander("Responsible-use limits", expanded=False):
            st.write(
                "BatteryOps is a portfolio-grade ML engineering demo built on public data and "
                "saved artifacts. It is not production EV safety software, a "
                "calibration benchmark, or a validated risk model."
            )
    return selected_asset_id


def _render_overview_tab(data: DashboardData) -> None:
    st.subheader("Fleet Cockpit")
    st.caption(
        "Fleet triage, selected-asset health, and the incident queue from the validated bundle."
    )
    metric_specs: tuple[tuple[str, str | int | float, str, list[float] | None], ...] = (
        (
            "Focus asset",
            str(data.summary["asset_id"]),
            "Asset currently highlighted in cockpit and replay views.",
            None,
        ),
        (
            "Latest cycle",
            int(data.summary["latest_cycle"]),
            "Most recent saved cycle for the selected asset.",
            None,
        ),
        (
            "Health index",
            f"{float(data.summary['health_score']):.1f}%",
            "Bounded 0..100 index normalized per asset from the source capacity proxy.",
            data.timeline["health_index_pct"].tail(12).tolist(),
        ),
        (
            "Selected-asset state",
            str(data.summary["alert_level"]),
            "Heuristic triage label for the latest saved cycle.",
            data.timeline["anomaly_score"].tail(12).tolist(),
        ),
    )
    _render_metric_grid(metric_specs, columns_per_row=4)

    fleet_left, fleet_right = st.columns((1.18, 0.82))
    fleet_left.plotly_chart(
        build_fleet_risk_figure(
            data.fleet_timeline,
            focus_asset_id=data.focus_asset_id,
            anomaly_threshold=data.anomaly_threshold,
        ),
        width="stretch",
        key="overview_fleet_risk_map",
    )
    with fleet_right:
        with st.container(border=True):
            st.markdown("**Cockpit decision ledger**")
            st.caption("Deterministic decisions from the same saved bundle and threshold.")
            decision_frame = build_cockpit_decision_frame(
                data.fleet_timeline,
                data.focus_asset_id,
                data.anomaly_threshold,
            )
            _render_decision_cards(decision_frame)

    with st.container(border=True):
        st.markdown("**Fleet priority queue**")
        st.caption(
            "Highest-priority assets from the saved bundle. Higher score means review first."
        )
        _render_priority_queue(build_asset_priority_table(data.fleet_timeline, row_count=6))

    concentration_left, drivers_right = st.columns((1.05, 0.95))
    concentration_left.plotly_chart(
        build_fleet_risk_concentration_figure(data.fleet_timeline),
        width="stretch",
        key="overview_fleet_risk_concentration",
    )
    risk_drivers = build_asset_risk_driver_frame(data.fleet_timeline, data.focus_asset_id)
    drivers_right.plotly_chart(
        build_asset_risk_driver_figure(risk_drivers),
        width="stretch",
        key="overview_asset_risk_drivers",
    )
    with st.expander("Selected-asset risk driver table", expanded=False):
        if risk_drivers.empty:
            st.info("Risk-driver details are not available for the selected asset.")
        else:
            st.dataframe(risk_drivers, width="stretch", hide_index=True)

    top_left, top_right = st.columns((1.1, 0.9))
    report_asset = str(data.report.get("asset_id", "")).strip()
    incident_cycle = (
        int(data.summary["focus_incident_cycle"]) if report_asset == data.focus_asset_id else None
    )
    top_left.plotly_chart(
        build_health_overview_figure(data.timeline, incident_cycle=incident_cycle),
        width="stretch",
        key="overview_health_trend",
    )
    top_right.plotly_chart(
        build_rul_confidence_figure(data.timeline),
        width="stretch",
        key="overview_rul_confidence",
    )

    bottom_left, bottom_right = st.columns((1.05, 0.95))
    with bottom_left:
        with st.container(border=True):
            st.markdown("**Latest telemetry rows**")
            st.caption("Ten saved rows from the selected asset timeline.")
            st.dataframe(
                build_recent_cycle_table(data.timeline, row_count=10),
                width="stretch",
                hide_index=True,
            )
    with bottom_right:
        with st.container(border=True):
            st.markdown("**Current triage note**")
            st.write(str(data.summary["triage_note"]))
            st.caption(
                "This note comes from the deterministic incident report stored with the current "
                "artifact bundle."
            )
        with st.container(border=True):
            st.markdown("**Triage handoff**")
            st.caption(
                "Work-order style next steps derived from the selected asset, saved threshold, "
                "and local artifact evidence."
            )
            _render_triage_handoff(
                build_triage_handoff_frame(
                    data.fleet_timeline,
                    data.focus_asset_id,
                    data.anomaly_threshold,
                )
            )
        with st.container(border=True):
            st.markdown("**Public readiness checks**")
            _render_public_readiness_checks()


def _render_replay_tab(data: DashboardData) -> None:
    st.subheader("Asset Replay")
    st.caption(
        "The replay advances a local inspection cursor through the saved timeline. Cursor "
        "movement does not retrain or fetch anything."
    )
    max_cycle = int(data.timeline["cycle"].max())
    if st.session_state.get("replay_asset_id") != data.focus_asset_id:
        st.session_state["replay_asset_id"] = data.focus_asset_id
        st.session_state["replay_cycle"] = max_cycle
    elif "replay_cycle" not in st.session_state:
        st.session_state["replay_cycle"] = max_cycle
    current_replay_cycle = int(st.session_state["replay_cycle"])
    current_replay_cycle = max(1, min(current_replay_cycle, max_cycle))
    st.session_state["replay_cycle"] = current_replay_cycle

    step_label = st.radio(
        "Step size",
        list(REPLAY_STEP_SIZES),
        horizontal=True,
        help="Controls how many cycles each forward or back button moves.",
    )
    step_size = REPLAY_STEP_SIZES[step_label]
    slider_cycle = int(
        st.slider(
            "Replay cycle cursor",
            min_value=1,
            max_value=max_cycle,
            value=current_replay_cycle,
            help="Move the cursor manually or use the replay controls below.",
        )
    )
    if slider_cycle != current_replay_cycle:
        st.session_state["replay_cycle"] = slider_cycle
    current_replay_cycle = int(st.session_state["replay_cycle"])
    at_start = current_replay_cycle <= 1
    at_latest = current_replay_cycle >= max_cycle
    replay_position = (
        f"Replay position: latest saved cycle ({current_replay_cycle} of {max_cycle})."
        if at_latest
        else f"Replay position: cycle {current_replay_cycle} of {max_cycle}."
    )
    st.caption(replay_position)

    controls = st.columns(4)
    if controls[0].button("Reset to start", width="stretch", disabled=at_start):
        _set_replay_cycle(1)
    if controls[1].button(
        f"Back {_cycle_count_label(step_size)}",
        width="stretch",
        disabled=at_start,
    ):
        _set_replay_cycle(max(1, int(st.session_state["replay_cycle"]) - step_size))
    if controls[2].button(
        f"Forward {_cycle_count_label(step_size)}",
        width="stretch",
        disabled=at_latest,
    ):
        _set_replay_cycle(min(max_cycle, int(st.session_state["replay_cycle"]) + step_size))
    if controls[3].button("Jump to latest", width="stretch", disabled=at_latest):
        _set_replay_cycle(max_cycle)

    replay_cycle = int(st.session_state["replay_cycle"])
    current_row = _current_cycle_row(data.timeline, replay_cycle)
    _render_replay_metrics(current_row, replay_cycle)

    replay_left, replay_right = st.columns((1.35, 0.65))
    replay_left.plotly_chart(
        build_replay_figure(
            data.timeline,
            replay_cycle=replay_cycle,
            threshold=data.anomaly_threshold,
        ),
        width="stretch",
        key="replay_telemetry",
    )
    with replay_right:
        st.plotly_chart(
            build_report_confidence_figure(data.report),
            width="stretch",
            key="replay_confidence_gauge",
        )
        with st.container(border=True):
            st.markdown("**Replay interpretation**")
            st.write(
                "Top chart shows capacity and resistance drift for the selected asset. Bottom "
                "chart overlays anomaly score, threshold, and alert markers from the "
                "saved timeline."
            )
            st.write(
                f"The current cursor implies a proxy RUL band of "
                f"{float(current_row['predicted_rul_lower_cycles']):.1f} to "
                f"{float(current_row['predicted_rul_upper_cycles']):.1f} cycles."
            )
            st.caption(
                "Report heuristic is a deterministic narrative aid from saved evidence and "
                "retrieval context, not a calibrated confidence estimate."
            )


def _render_anomaly_tab(data: DashboardData) -> None:
    st.subheader("Anomaly Timeline")
    st.caption(
        "Adjust the heuristic threshold to tune how aggressively the alert queue "
        "surfaces for this asset."
    )
    threshold = st.slider(
        "Alert threshold",
        min_value=0.0,
        max_value=1.0,
        value=_slider_threshold_value(data.anomaly_threshold),
        step=0.01,
        help="Cycles at or above this threshold are flagged for inspection.",
    )
    flagged_cycles = build_flagged_cycle_table(data.timeline, threshold)

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Flagged cycles",
        len(flagged_cycles),
        help="How many saved cycles exceed the selected threshold.",
        border=True,
    )
    metric_cols[1].metric(
        "Observed incidents",
        int(data.timeline["actual_alert"].sum()),
        help="Cycles in the saved artifact labeled with observed incident windows.",
        border=True,
    )
    metric_cols[2].metric(
        "Latest score vs threshold",
        f"{float(data.timeline.iloc[-1]['anomaly_score']):.2f} / {threshold:.2f}",
        help="Latest saved anomaly score against the chosen threshold.",
        border=True,
    )

    chart_col, table_col = st.columns((1.2, 0.8))
    chart_col.plotly_chart(
        build_anomaly_timeline_figure(data.timeline, threshold),
        width="stretch",
        key="anomaly_timeline",
    )
    with table_col:
        st.markdown("**Flagged cycle queue**")
        if flagged_cycles.empty:
            st.info("No saved cycles cross the selected threshold.")
        else:
            st.dataframe(flagged_cycles, width="stretch", hide_index=True)


def _render_report_tab(data: DashboardData) -> None:
    st.subheader("Incident Evidence")
    st.caption(
        "Highest-priority saved incident, threshold context, source evidence, and diagnostics."
    )
    threshold = st.slider(
        "Alert threshold",
        min_value=0.0,
        max_value=1.0,
        value=_slider_threshold_value(data.anomaly_threshold),
        step=0.01,
        help="Cycles at or above this threshold are flagged for inspection.",
    )
    flagged_cycles = build_flagged_cycle_table(data.timeline, threshold)
    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Flagged cycles",
        len(flagged_cycles),
        help="How many selected-asset cycles exceed the selected threshold.",
        border=True,
    )
    metric_cols[1].metric(
        "Observed incidents",
        int(data.timeline["actual_alert"].sum()),
        help="Cycles in the saved artifact labeled with observed incident windows.",
        border=True,
    )
    metric_cols[2].metric(
        "Report heuristic",
        f"{float(data.summary['confidence_score']) * 100:.0f}%",
        help="Deterministic narrative score from saved evidence, not a calibrated probability.",
        border=True,
    )
    report_asset = str(data.report.get("asset_id", "")).strip()
    incident_cycle = int(data.report.get("cycle_id", data.summary["focus_incident_cycle"]))
    report_timeline = (
        data.fleet_timeline.loc[data.fleet_timeline["asset_id"].astype(str) == report_asset]
        .sort_values("cycle")
        .reset_index(drop=True)
        if report_asset
        else data.timeline
    )
    if report_timeline.empty:
        report_timeline = data.timeline
    focus_incident = build_focus_incident_row(data.report, data.incidents)

    left, right = st.columns((1.12, 0.88))
    with left:
        if report_asset and report_asset != data.focus_asset_id:
            st.info(
                f"The saved report is tied to asset `{report_asset}`, while the selected "
                f"focus asset is `{data.focus_asset_id}`."
            )
        with st.container(border=True):
            st.markdown("**Report summary**")
            st.write(str(data.report.get("summary", "No report summary was available.")))
        st.plotly_chart(
            build_incident_marker_figure(
                report_timeline,
                incident_cycle=incident_cycle,
                threshold=threshold,
            ),
            width="stretch",
            key="report_incident_marker",
        )
    with right:
        st.plotly_chart(
            build_report_confidence_figure(data.report),
            width="stretch",
            key="report_confidence_gauge",
        )
        st.caption(
            "Report heuristic is a deterministic narrative aid, not a calibrated confidence score."
        )
        with st.container(border=True):
            st.markdown("**Saved incident metadata**")
            st.write(f"Focus cycle: `{incident_cycle}`")
            st.write(
                "Incident types: "
                + (
                    ", ".join(str(item) for item in data.report.get("incident_types", []))
                    or "not recorded"
                )
            )
            st.write(
                f"Evidence source coverage: {float(data.summary['grounding_coverage']) * 100:.0f}%"
            )
            if focus_incident is not None:
                st.write(f"Minimum voltage: {float(focus_incident['min_voltage_v']):.2f} V")
                st.write(f"Peak temperature: {float(focus_incident['max_temperature_c']):.2f} C")
                st.write(
                    f"Peak absolute current: {float(focus_incident['max_abs_current_a']):.2f} A"
                )
            else:
                st.write("Incident telemetry snapshot is not available.")
        incident_markdown = build_incident_markdown(data.report, focus_incident, data.metrics)
        st.download_button(
            "Download incident brief",
            data=incident_markdown,
            file_name=f"{data.report.get('report_id', 'batteryops-incident')}.md",
            mime="text/markdown",
            width="stretch",
        )

    evidence_col, tests_col = st.columns(2)
    with evidence_col:
        st.markdown("**Evidence**")
        evidence = data.report.get("evidence", [])
        if isinstance(evidence, list) and evidence:
            for item in evidence:
                if isinstance(item, dict):
                    st.write(f"- {item.get('text', '')}")
        else:
            st.info("No evidence bullets were saved with the current report artifact.")
    with tests_col:
        st.markdown("**Recommended next diagnostic tests**")
        tests = data.report.get("recommended_tests", [])
        if isinstance(tests, list) and tests:
            for item in tests:
                st.write(f"- {item}")
        else:
            st.info("No recommended tests were saved with the current report artifact.")
        st.markdown("**Flagged cycle queue**")
        if flagged_cycles.empty:
            st.info("No selected-asset cycles cross the selected threshold.")
        else:
            st.dataframe(flagged_cycles, width="stretch", hide_index=True)


def _render_similar_cases_tab(data: DashboardData) -> None:
    st.subheader("Similar Cases")
    st.caption(
        "Similarity lookup scores historical incidents stored with the loaded bundle. Lower "
        "distance means a closer historical match."
    )
    cases = build_similar_cases_frame(data.report)

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Retrieved cases",
        len(cases),
        help="Number of historical incident windows returned from the saved retrieval bundle.",
        border=True,
    )
    metric_cols[1].metric(
        "Closest distance",
        f"{float(cases['distance'].min()):.2f}" if not cases.empty else "N/A",
        help="Lower distance indicates a more similar historical case.",
        border=True,
    )
    metric_cols[2].metric(
        "Best type overlap",
        int(cases["type_overlap"].max()) if not cases.empty else 0,
        help="Shared incident types between saved report and the best historical match.",
        border=True,
    )

    chart_col, table_col = st.columns((1.1, 0.9))
    chart_col.plotly_chart(
        build_similar_cases_figure(cases),
        width="stretch",
        key="similar_cases_scatter",
    )
    with table_col:
        with st.container(border=True):
            st.markdown("**Retrieved case table**")
            st.caption("Local retrieval results from the saved bundle.")
            if cases.empty:
                st.info("No similar cases were returned by the local retrieval bundle.")
            else:
                display_cases = cases.rename(
                    columns={
                        "case_label": "Case",
                        "incident_label": "Incident types",
                        "distance": "Distance",
                        "severity_score": "Severity",
                        "type_overlap": "Shared types",
                    }
                ).copy()
                display_cases[["Distance", "Severity"]] = display_cases[
                    ["Distance", "Severity"]
                ].round(3)
                st.dataframe(display_cases, width="stretch", hide_index=True)


def _render_evaluation_tab(data: DashboardData) -> None:
    st.subheader("Model Evaluation")
    st.caption(
        "Offline proxy evaluation for the loaded artifact: holdout error, alert quality, "
        "threshold tradeoffs, feature signals, and failure-analysis slices."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Proxy RUL MAE",
        f"{float(data.metrics.get('rul_proxy_mae', data.metrics.get('rul_mae', 0.0))):.2f}",
        help="Held-out mean absolute error for the degradation-threshold horizon proxy.",
        border=True,
    )
    metric_cols[1].metric(
        "Alert precision",
        f"{float(data.metrics.get('alert_precision', 0.0)) * 100:.1f}%",
        help="Share of threshold-breaching alerts that matched an observed incident cycle.",
        border=True,
    )
    metric_cols[2].metric(
        "Alert recall",
        f"{float(data.metrics.get('alert_recall', 0.0)) * 100:.1f}%",
        help="Share of observed incident cycles recovered by the alert model.",
        border=True,
    )
    metric_cols[3].metric(
        "Evidence coverage",
        f"{float(data.metrics.get('evidence_source_coverage', 0.0)) * 100:.0f}%",
        help="Share of report evidence bullets backed by explicit source fields and values.",
        border=True,
    )

    top_left, top_right = st.columns(2)
    top_left.plotly_chart(
        build_confusion_matrix_figure(data.fleet_timeline),
        width="stretch",
        key="evaluation_confusion",
    )
    top_right.plotly_chart(
        build_rul_scatter_figure(data.fleet_timeline),
        width="stretch",
        key="evaluation_rul_scatter",
    )
    with st.expander("Threshold, feature, and asset analysis", expanded=True):
        st.plotly_chart(
            build_alert_coverage_figure(data.fleet_timeline),
            width="stretch",
            key="evaluation_alert_coverage",
        )
        detail_left, detail_right = st.columns(2)
        detail_left.plotly_chart(
            build_threshold_tradeoff_figure(data.evaluation_report),
            width="stretch",
            key="evaluation_threshold_tradeoff",
        )
        detail_right.plotly_chart(
            build_asset_error_figure(data.evaluation_report),
            width="stretch",
            key="evaluation_asset_error",
        )
        st.plotly_chart(
            build_feature_importance_figure(data.model_card),
            width="stretch",
            key="evaluation_feature_importance",
        )


def _render_provenance_tab(data: DashboardData) -> None:
    st.subheader("Data & Provenance")
    st.caption("Bundle health, model-card summary, and zero-cost runtime contract.")

    quality = data.data_quality_report
    model_card = data.model_card
    metric_specs: tuple[tuple[str, str | int | float, str, list[float] | None], ...] = (
        (
            "Bundle status",
            "Verified" if data.bundle_fingerprint else "Fallback",
            "The local artifact bundle loaded successfully and can be audited from disk.",
            None,
        ),
        (
            "Quality checks",
            _quality_status_label(quality),
            "Pass/warn summary for saved data-quality checks.",
            None,
        ),
        (
            "External APIs",
            "0",
            "No external API calls are required for runtime or training.",
            None,
        ),
        (
            "Health scale",
            "0-100%",
            "Normalized per asset; higher means healthier in the saved telemetry bundle.",
            None,
        ),
    )
    _render_metric_grid(metric_specs, columns_per_row=4)
    st.caption(
        f"Validated bundle ID: `{_short_fingerprint(data.bundle_fingerprint)}`. "
        "The ID is used for auditability, not as a user-facing score."
    )

    left, right = st.columns((1.0, 1.0))
    with left:
        with st.container(border=True):
            st.markdown("**Data quality checks**")
            checks = quality.get("checks", []) if isinstance(quality, dict) else []
            if checks:
                _render_quality_checks(pd.DataFrame(checks))
            else:
                st.info("No data-quality report was available for this bundle.")
        with st.container(border=True):
            st.markdown("**Zero-cost contract**")
            st.write("- Runtime cost: `$0`")
            st.write("- Paid APIs: `none`")
            st.write("- API keys or secrets: `none`")
            st.write("- Hosted database: `none`")
        with st.container(border=True):
            st.markdown("**Artifact inventory**")
            inventory = build_artifact_inventory_frame(data.artifact_inventory)
            if inventory.empty:
                st.info("Validated artifact inventory is not available for this session.")
            else:
                st.dataframe(inventory, width="stretch", hide_index=True)
    with right:
        with st.container(border=True):
            st.markdown("**Public readiness audit**")
            _render_public_readiness_checks()
        with st.container(border=True):
            st.markdown("**Model card summary**")
            models = model_card.get("models", {}) if isinstance(model_card, dict) else {}
            if models:
                for model_name, model_payload in models.items():
                    if isinstance(model_payload, dict):
                        st.write(
                            f"- `{model_name}`: {model_payload.get('estimator', 'unknown')}"
                        )
            else:
                st.info("No model card was available for this bundle.")
        with st.container(border=True):
            st.markdown("**Limitations**")
            limitations = model_card.get("limitations", []) if isinstance(model_card, dict) else []
            if limitations:
                for limitation in limitations:
                    st.write(f"- {limitation}")
            else:
                st.write("- Demo metrics are proxy evidence, not production safety claims.")


def _quality_status_label(data_quality_report: dict[str, Any]) -> str:
    checks = data_quality_report.get("checks", []) if isinstance(data_quality_report, dict) else []
    if not checks:
        return "Unavailable"
    warn_count = sum(
        1 for item in checks if isinstance(item, dict) and item.get("status") != "pass"
    )
    return "Pass" if warn_count == 0 else f"{warn_count} warnings"


def _query_focus_asset() -> str | None:
    raw_asset = st.query_params.get("focus_asset")
    if isinstance(raw_asset, list):
        raw_asset = raw_asset[0] if raw_asset else None
    if raw_asset is None:
        return None
    requested = str(raw_asset).strip()
    return requested or None


def _requested_focus_asset() -> str | None:
    session_asset = st.session_state.get(FOCUS_ASSET_STATE_KEY)
    if isinstance(session_asset, str) and session_asset.strip():
        return session_asset.strip()
    return _query_focus_asset()


def _render_decision_cards(decision_frame: pd.DataFrame) -> None:
    if decision_frame.empty:
        st.info("Cockpit decisions are not available for this bundle.")
        return

    cards: list[str] = []
    for row in decision_frame.to_dict("records"):
        layer = escape(str(row.get("Layer", "")))
        state = escape(str(row.get("State", "")))
        evidence = escape(str(row.get("Evidence", "")))
        next_action = escape(str(row.get("Next action", "")))
        cards.append(
            f'<div class="decision-card">'
            f'<div class="card-eyebrow">{layer}</div>'
            f'<div class="card-title">{state}</div>'
            f"<p>{evidence}</p>"
            f"<small>{next_action}</small>"
            "</div>"
        )
    st.markdown(
        f'<div class="decision-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_priority_queue(priority_frame: pd.DataFrame) -> None:
    if priority_frame.empty:
        st.info("The priority queue is not available for this bundle.")
        return

    cards: list[str] = []
    for rank, row in enumerate(priority_frame.to_dict("records"), start=1):
        asset = escape(str(row.get("Asset", "Unknown asset")))
        state = escape(str(row.get("Latest state", "Unknown state")))
        priority_score = _format_scalar(row.get("Priority score"))
        health = _format_scalar(row.get("Latest health (%)"))
        anomaly = _format_scalar(row.get("Max anomaly"), decimals=3)
        inspect_cycles = _format_scalar(row.get("Inspect cycles"), decimals=0)
        proxy_rul = _format_scalar(row.get("Proxy RUL"))
        cards.append(
            f'<div class="priority-card">'
            f'<div class="priority-rank">#{rank}</div>'
            '<div class="priority-body">'
            '<div class="priority-heading">'
            f"<strong>{asset}</strong>"
            f"<span>{state}</span>"
            "</div>"
            f'<div class="priority-score">Priority {priority_score}</div>'
            '<div class="metric-pairs">'
            f"<span><b>Health</b>{health}%</span>"
            f"<span><b>Anomaly</b>{anomaly}</span>"
            f"<span><b>Inspect cycles</b>{inspect_cycles}</span>"
            f"<span><b>Proxy RUL</b>{proxy_rul}</span>"
            "</div>"
            "</div>"
            "</div>"
        )
    st.markdown(
        f'<div class="priority-list">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_triage_handoff(handoff_frame: pd.DataFrame) -> None:
    if handoff_frame.empty:
        st.info("No triage handoff can be built for the selected asset.")
        return

    cards: list[str] = []
    for row in handoff_frame.to_dict("records"):
        step = escape(str(row.get("Review step", "")))
        decision = escape(str(row.get("Decision", "")))
        evidence = escape(str(row.get("Evidence", "")))
        why = escape(str(row.get("Why it matters", "")))
        cards.append(
            f'<div class="handoff-card">'
            f'<div class="card-eyebrow">{step}</div>'
            f'<div class="card-title">{decision}</div>'
            f"<p>{evidence}</p>"
            f"<small>{why}</small>"
            "</div>"
        )
    st.markdown(
        f'<div class="handoff-list">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_quality_checks(checks: pd.DataFrame) -> None:
    if checks.empty:
        st.info("No data-quality report was available for this bundle.")
        return

    cards: list[str] = []
    for row in checks.to_dict("records"):
        name = escape(_humanize_identifier(str(row.get("name", "Quality check"))))
        status = str(row.get("status", "")).title() or "Unknown"
        detail = escape(str(row.get("detail", "No detail recorded.")))
        cards.append(
            f'<div class="readiness-item">'
            f"<div><strong>{name}</strong><p>{detail}</p></div>"
            f'<span class="status-badge">{escape(status)}</span>'
            "</div>"
        )
    st.markdown(
        f'<div class="readiness-list">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _render_public_readiness_checks() -> None:
    audit = _load_public_readiness_audit()
    st.caption(
        f"{audit['pass_count']}/{len(audit['checks'])} local checks passing; "
        f"{audit['warning_count']} warnings."
    )
    cards: list[str] = []
    for check in audit["checks"]:
        if not isinstance(check, dict):
            continue
        area = escape(str(check.get("area", "Review item")))
        status = str(check.get("status", "")).title() or "Unknown"
        evidence = escape(str(check.get("evidence", "No evidence recorded.")))
        cards.append(
            f'<div class="readiness-item">'
            f"<div><strong>{area}</strong><p>{evidence}</p></div>"
            f'<span class="status-badge">{escape(status)}</span>'
            "</div>"
        )
    st.markdown(
        f'<div class="readiness-list">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def _humanize_identifier(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def _format_scalar(value: Any, *, decimals: int = 1) -> str:
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return escape(str(value))


def _short_fingerprint(fingerprint: str | None) -> str:
    if not fingerprint:
        return "Unavailable"
    return fingerprint[:12]


def _current_cycle_row(timeline: pd.DataFrame, replay_cycle: int) -> pd.Series:
    current = timeline.loc[timeline["cycle"] <= replay_cycle]
    if current.empty:
        return timeline.sort_values("cycle").iloc[0]
    return current.sort_values("cycle").iloc[-1]


def _set_replay_cycle(target_cycle: int) -> None:
    st.session_state["replay_cycle"] = target_cycle
    st.rerun()


def _cycle_count_label(cycle_count: int) -> str:
    return f"{cycle_count} cycle" if cycle_count == 1 else f"{cycle_count} cycles"


def _slider_threshold_value(threshold: float) -> float:
    return min(1.0, max(0.0, round(float(threshold), 2)))


def _apply_theme() -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{
            background:
              linear-gradient(180deg, {BATTERYOPS_COLORS["bg"]} 0%, #f8fafc 100%);
          }}
          .block-container {{
            max-width: 1400px;
            padding-top: 1.8rem;
            padding-bottom: 3.25rem;
          }}
          [data-testid="stToolbar"],
          [data-testid="stDecoration"],
          .stDeployButton {{
            display: none !important;
          }}
          h1, h2, h3, h4, [data-testid="stMetricLabel"], [data-baseweb="tab"] {{
            font-family: "IBM Plex Sans", "Aptos", "Segoe UI", sans-serif;
          }}
          code, pre, .stCodeBlock {{
            font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
          }}
          .hero-card {{
            background: {BATTERYOPS_COLORS["navy"]};
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-left: 6px solid {BATTERYOPS_COLORS["amber"]};
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 22px 52px rgba(15, 39, 66, 0.14);
          }}
          .hero-card h3 {{
            margin: 0.2rem 0 0.5rem 0;
            font-size: 1.5rem;
          }}
          .hero-card p {{
            margin: 0;
            color: rgba(255, 255, 255, 0.86);
          }}
          .hero-kicker {{
            text-transform: uppercase;
            letter-spacing: 0;
            font-size: 0.72rem;
            font-weight: 700;
            color: rgba(244, 185, 66, 0.95);
          }}
          .proof-strip {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.85rem 0 1.05rem 0;
          }}
          .proof-item {{
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid {BATTERYOPS_COLORS["border"]};
            border-radius: 8px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 12px 26px rgba(15, 39, 66, 0.05);
            min-height: 5.7rem;
          }}
          .proof-item span {{
            display: block;
            color: {BATTERYOPS_COLORS["muted"]};
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
          }}
          .proof-item strong {{
            display: block;
            color: {BATTERYOPS_COLORS["ink"]};
            font-size: 1.02rem;
            line-height: 1.25;
          }}
          .proof-item small {{
            display: block;
            color: {BATTERYOPS_COLORS["muted"]};
            line-height: 1.3;
            margin-top: 0.25rem;
          }}
          .decision-grid,
          .handoff-list,
          .readiness-list,
          .priority-list {{
            display: grid;
            gap: 0.62rem;
          }}
          .priority-list {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }}
          .decision-card,
          .handoff-card,
          .readiness-item,
          .priority-card {{
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(15, 39, 66, 0.10);
            border-radius: 8px;
            padding: 0.72rem 0.82rem;
          }}
          .decision-card,
          .handoff-card {{
            border-left: 4px solid {BATTERYOPS_COLORS["blue"]};
          }}
          .card-eyebrow {{
            color: {BATTERYOPS_COLORS["muted"]};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
          }}
          .card-title {{
            color: {BATTERYOPS_COLORS["ink"]};
            font-size: 1.02rem;
            line-height: 1.25;
            font-weight: 700;
            margin-bottom: 0.22rem;
          }}
          .decision-card p,
          .handoff-card p,
          .readiness-item p {{
            color: {BATTERYOPS_COLORS["ink"]};
            margin: 0.05rem 0 0.22rem 0;
            line-height: 1.35;
          }}
          .decision-card small,
          .handoff-card small {{
            display: block;
            color: {BATTERYOPS_COLORS["muted"]};
            line-height: 1.32;
          }}
          .priority-card {{
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 0.65rem;
            align-items: start;
          }}
          .priority-rank {{
            background: {BATTERYOPS_COLORS["navy"]};
            color: white;
            border-radius: 999px;
            min-width: 2.15rem;
            height: 2.15rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.82rem;
          }}
          .priority-heading {{
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            align-items: center;
          }}
          .priority-heading strong {{
            color: {BATTERYOPS_COLORS["ink"]};
            font-size: 1.02rem;
            line-height: 1.22;
          }}
          .priority-heading span,
          .status-badge {{
            color: {BATTERYOPS_COLORS["navy"]};
            background: rgba(29, 95, 167, 0.10);
            border: 1px solid rgba(29, 95, 167, 0.18);
            border-radius: 999px;
            padding: 0.14rem 0.45rem;
            font-size: 0.73rem;
            font-weight: 700;
            white-space: nowrap;
          }}
          .priority-score {{
            color: {BATTERYOPS_COLORS["red"]};
            font-weight: 700;
            margin: 0.2rem 0 0.35rem 0;
          }}
          .metric-pairs {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.32rem 0.55rem;
          }}
          .metric-pairs span {{
            color: {BATTERYOPS_COLORS["ink"]};
            font-size: 0.82rem;
          }}
          .metric-pairs b {{
            display: block;
            color: {BATTERYOPS_COLORS["muted"]};
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0;
          }}
          .readiness-item {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 0.75rem;
            align-items: start;
          }}
          .readiness-item strong {{
            color: {BATTERYOPS_COLORS["ink"]};
          }}
          .readiness-item p {{
            color: {BATTERYOPS_COLORS["muted"]};
            margin-bottom: 0;
          }}
          @media (max-width: 980px) {{
            .proof-strip {{
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .priority-list {{
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .metric-pairs {{
              grid-template-columns: 1fr;
            }}
          }}
          @media (max-width: 620px) {{
            .proof-strip {{
              grid-template-columns: 1fr;
            }}
            .priority-list {{
              grid-template-columns: 1fr;
            }}
            .priority-heading,
            .readiness-item {{
              display: block;
            }}
            .priority-heading span,
            .status-badge {{
              display: inline-block;
              margin-top: 0.3rem;
            }}
          }}
          div[data-baseweb="tab-list"] {{
            gap: 0.35rem;
            padding: 0.25rem;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid {BATTERYOPS_COLORS["border"]};
            border-radius: 8px;
            box-shadow: 0 10px 26px rgba(15, 39, 66, 0.05);
          }}
          button[data-baseweb="tab"] {{
            border-radius: 6px;
            padding: 0.55rem 0.9rem;
            border: 0;
            font-weight: 600;
            letter-spacing: 0;
          }}
          button[data-baseweb="tab"][aria-selected="true"] {{
            background: {BATTERYOPS_COLORS["navy"]};
            color: white;
          }}
          [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.84);
            border-radius: 8px;
            min-height: 6.45rem;
            box-shadow: 0 14px 32px rgba(15, 39, 66, 0.06);
          }}
          [data-testid="stDataFrame"] {{
            border: 1px solid rgba(15, 39, 66, 0.10);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 12px 28px rgba(15, 39, 66, 0.05);
          }}
          [data-testid="stPlotlyChart"] {{
            border: 1px solid rgba(15, 39, 66, 0.08);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.8);
            box-shadow: 0 12px 30px rgba(15, 39, 66, 0.05);
          }}
          div[data-testid="stAlert"] {{
            border-radius: 8px;
            box-shadow: 0 12px 30px rgba(15, 39, 66, 0.05);
          }}
          [data-testid="stMetricLabel"],
          [data-testid="stMetricLabel"] * {{
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.2;
          }}
          [data-testid="stMetricValue"],
          [data-testid="stMetricValue"] * {{
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.05;
          }}
          [data-testid="stMetricValue"] > div,
          [data-testid="stMetricValue"] p {{
            font-size: 1.95rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
