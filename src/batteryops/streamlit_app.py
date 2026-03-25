from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from batteryops.dashboard import (
    BATTERYOPS_COLORS,
    DashboardData,
    build_alert_coverage_figure,
    build_anomaly_timeline_figure,
    build_confusion_matrix_figure,
    build_flagged_cycle_table,
    build_focus_incident_row,
    build_health_overview_figure,
    build_incident_marker_figure,
    build_recent_cycle_table,
    build_replay_figure,
    build_report_confidence_figure,
    build_rul_confidence_figure,
    build_rul_scatter_figure,
    build_similar_cases_figure,
    build_similar_cases_frame,
    load_dashboard_data,
)

REPLAY_SPEEDS = {"1x": 1, "4x": 4, "10x": 10}


def load_demo_payload() -> tuple[dict[str, Any], pd.DataFrame]:
    """Return the summary and focus-asset timeline used by the Streamlit app."""
    data = load_dashboard_data()
    return data.summary, data.timeline


def main() -> None:
    """Render the public-facing BatteryOps dashboard."""
    st.set_page_config(page_title="BatteryOps", page_icon="🔋", layout="wide")
    _apply_theme()

    data = load_dashboard_data()
    selected_asset_id = _render_sidebar(data)
    if selected_asset_id != data.focus_asset_id:
        data = load_dashboard_data(selected_asset_id)
    _render_header(data)

    overview_tab, replay_tab, anomaly_tab, report_tab, similar_tab, eval_tab = st.tabs(
        [
            "Overview",
            "Live Telemetry Replay",
            "Anomaly Timeline",
            "Incident Report",
            "Similar Cases",
            "Evaluation Dashboard",
        ]
    )

    with overview_tab:
        _render_overview_tab(data)
    with replay_tab:
        _render_replay_tab(data)
    with anomaly_tab:
        _render_anomaly_tab(data)
    with report_tab:
        _render_report_tab(data)
    with similar_tab:
        _render_similar_cases_tab(data)
    with eval_tab:
        _render_evaluation_tab(data)


def _render_header(data: DashboardData) -> None:
    st.title("BatteryOps")
    st.caption(
        "Local-first telemetry triage demo built on public NASA battery data and saved artifacts."
    )
    st.markdown(
        """
        <div class="hero-card">
          <p class="hero-kicker">Public NASA battery data</p>
          <h3>Deterministic telemetry triage from saved NASA battery artifacts</h3>
          <p>
            BatteryOps boots from the validated checked-in demo bundle and renders from saved
            artifacts, so the app stays quick, deterministic, and easy to inspect offline.
            Local processed parquet remains optional workspace state for preprocessing and
            retraining, not a runtime dependency for this demo.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Runtime provenance: {data.dataset_status.source_label}")

    _render_header_metrics(data)

    st.warning(
        "BatteryOps is a recruiter-facing local demo built on public data and saved artifacts. "
        "It is not production EV safety software, a calibration benchmark, or a "
        "validated risk model."
    )


def _render_header_metrics(data: DashboardData) -> None:
    metric_specs: tuple[tuple[str, str | int | float, str, list[float] | None], ...] = (
        (
            "Focus asset",
            str(data.summary["asset_id"]),
            "Asset currently highlighted in Overview, Replay, and Anomaly Timeline.",
            None,
        ),
        (
            "Latest cycle",
            int(data.summary["latest_cycle"]),
            "Most recent saved cycle available for the selected asset.",
            None,
        ),
        (
            "Proxy RUL",
            f"{int(data.summary['estimated_rul_cycles'])} cycles",
            "Heuristic proxy remaining useful life, not a calibrated forecast.",
            data.timeline["predicted_rul_cycles"].tail(12).tolist(),
        ),
        (
            "Report heuristic",
            f"{float(data.summary['confidence_score']) * 100:.0f}%",
            "Deterministic narrative score from the saved artifact. Not a calibrated "
            "probability.",
            None,
        ),
        (
            "Capacity-retention proxy",
            f"{float(data.summary['health_score']):.1f}%",
            "Saved capacity-retention proxy for the current cycle compared with early-cycle "
            "baseline.",
            data.timeline["capacity_ah"].tail(12).tolist(),
        ),
        (
            "Heuristic alert state",
            str(data.summary["alert_level"]),
            "Heuristic triage label for the latest saved cycle.",
            data.timeline["anomaly_score"].tail(12).tolist(),
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
        for column, (label, value, help_text, chart_data) in zip(
            row_columns,
            row_specs,
            strict=True,
        ):
            if chart_data is not None:
                column.metric(label, value, help=help_text, border=True, chart_data=chart_data)
            else:
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
            "Local-first by default: no backend service, no auth layer, and no paid integrations."
        )

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
            asset_options = sorted(data.fleet_timeline["asset_id"].astype(str).unique())
            selected_asset_id = data.focus_asset_id
            if len(asset_options) > 1:
                selected_asset_id = st.selectbox(
                    "Focus asset",
                    asset_options,
                    index=(
                        asset_options.index(data.focus_asset_id)
                        if data.focus_asset_id in asset_options
                        else 0
                    ),
                    help="Choose which asset drives the overview, replay, and anomaly tabs.",
                    key="focus_asset_id",
                )
            else:
                st.write(f"Focus asset: `{selected_asset_id}`.")
            st.caption(
                "Incident Report, Similar Cases, and Evaluation stay tied to the "
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
                "The six tabs walk from current asset snapshot to replay, threshold tuning, "
                "incident context, similar cases, and offline evaluation."
            )
    return selected_asset_id


def _render_overview_tab(data: DashboardData) -> None:
    st.subheader("System Snapshot")
    st.caption(
        "Overview summarizes the selected asset's degradation trend, anomaly posture, "
        "proxy RUL signal, and deterministic triage note from the saved artifact."
    )
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
            st.markdown("**Saved caveats**")
            caveats = data.report.get("confidence_caveats", [])
            if isinstance(caveats, list) and caveats:
                for item in caveats:
                    st.write(f"- {item}")
            else:
                st.write("- No caveats were saved with the current artifact bundle.")


def _render_replay_tab(data: DashboardData) -> None:
    st.subheader("Live Telemetry Replay")
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

    speed_label = st.radio(
        "Replay speed",
        list(REPLAY_SPEEDS),
        horizontal=True,
        help="Controls how many cycles each forward/back step advances or retreats.",
    )
    speed = REPLAY_SPEEDS[speed_label]
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

    controls = st.columns(4)
    if controls[0].button("Reset to start", width="stretch"):
        _set_replay_cycle(1)
    if controls[1].button(f"Step back {speed}", width="stretch"):
        _set_replay_cycle(max(1, int(st.session_state["replay_cycle"]) - speed))
    if controls[2].button(f"Advance {speed}", width="stretch"):
        _set_replay_cycle(min(max_cycle, int(st.session_state["replay_cycle"]) + speed))
    if controls[3].button("Jump to latest", width="stretch"):
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
        value=float(data.anomaly_threshold),
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
    st.subheader("Incident Report")
    st.caption(
        "This tab displays the deterministic incident report for the highest-priority saved "
        "case in the active artifact."
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
                threshold=data.anomaly_threshold,
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
    st.subheader("Evaluation Dashboard")
    st.caption(
        "Offline proxy evaluation for the loaded artifact: held-out RUL proxy error, "
        "alert quality, evidence coverage, and fleet-level alert coverage."
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
    st.plotly_chart(
        build_alert_coverage_figure(data.fleet_timeline),
        width="stretch",
        key="evaluation_alert_coverage",
    )


def _current_cycle_row(timeline: pd.DataFrame, replay_cycle: int) -> pd.Series:
    current = timeline.loc[timeline["cycle"] == replay_cycle]
    if current.empty:
        return timeline.sort_values("cycle").iloc[-1]
    return current.iloc[-1]


def _set_replay_cycle(target_cycle: int) -> None:
    st.session_state["replay_cycle"] = target_cycle
    st.rerun()


def _apply_theme() -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{
            background:
              radial-gradient(circle at top right, rgba(29, 95, 167, 0.08), transparent 30%),
              linear-gradient(180deg, {BATTERYOPS_COLORS["bg"]} 0%, #f8fafc 100%);
          }}
          .block-container {{
            max-width: 1400px;
            padding-top: 1.8rem;
            padding-bottom: 3.25rem;
          }}
          h1, h2, h3, h4, [data-testid="stMetricLabel"], [data-baseweb="tab"] {{
            font-family: "IBM Plex Sans", "Aptos", "Segoe UI", sans-serif;
          }}
          code, pre, .stCodeBlock {{
            font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
          }}
          .hero-card {{
            background:
              radial-gradient(circle at top right, rgba(255, 255, 255, 0.14), transparent 26%),
              linear-gradient(135deg, rgba(15, 39, 66, 0.98), rgba(29, 95, 167, 0.94));
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 20px;
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
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
            color: rgba(244, 185, 66, 0.95);
          }}
          div[data-baseweb="tab-list"] {{
            gap: 0.35rem;
            padding: 0.25rem;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid {BATTERYOPS_COLORS["border"]};
            border-radius: 999px;
            box-shadow: 0 10px 26px rgba(15, 39, 66, 0.05);
          }}
          button[data-baseweb="tab"] {{
            border-radius: 999px;
            padding: 0.55rem 0.9rem;
            border: 0;
            font-weight: 600;
            letter-spacing: 0.01em;
          }}
          button[data-baseweb="tab"][aria-selected="true"] {{
            background: {BATTERYOPS_COLORS["navy"]};
            color: white;
          }}
          [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.84);
            border-radius: 16px;
            min-height: 8.6rem;
            box-shadow: 0 14px 32px rgba(15, 39, 66, 0.06);
          }}
          [data-testid="stDataFrame"] {{
            border: 1px solid rgba(15, 39, 66, 0.10);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 12px 28px rgba(15, 39, 66, 0.05);
          }}
          [data-testid="stPlotlyChart"] {{
            border: 1px solid rgba(15, 39, 66, 0.08);
            border-radius: 16px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.8);
            box-shadow: 0 12px 30px rgba(15, 39, 66, 0.05);
          }}
          div[data-testid="stAlert"] {{
            border-radius: 14px;
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
            font-size: clamp(1.85rem, 2.15vw, 2.7rem);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
