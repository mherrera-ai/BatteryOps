"""Dashboard data loading and Plotly figure helpers for the BatteryOps app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from batteryops.reports.demo import (
    DemoBundleStatus,
    build_demo_report,
    build_demo_timeline,
    inspect_demo_bundle,
)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_EXPECTED = (
    PROCESSED_DIR / "telemetry_samples.parquet",
    PROCESSED_DIR / "cycle_features.parquet",
    PROCESSED_DIR / "incident_windows.parquet",
)

BATTERYOPS_COLORS = {
    "navy": "#0f2742",
    "blue": "#1d5fa7",
    "teal": "#1f8a70",
    "amber": "#f4b942",
    "red": "#c94c4c",
    "ink": "#13212f",
    "muted": "#6b7785",
    "surface": "#ffffff",
    "border": "#d7dee6",
    "grid": "#e7edf3",
    "bg": "#f3f6f8",
}

DISPLAY_LABEL_CAPACITY = "Capacity proxy (Ah)"
DISPLAY_LABEL_RESISTANCE = "Internal resistance (Ohm)"
DISPLAY_LABEL_ANOMALY_SCORE = "Anomaly score"
DISPLAY_LABEL_PREDICTED_RUL = "Predicted RUL"
DISPLAY_LABEL_PREDICTED_RUL_CYCLES = "Predicted RUL (cycles)"
DISPLAY_LABEL_THRESHOLD_BREACH = "Threshold breach"
DISPLAY_LABEL_OBSERVED_INCIDENT = "Observed incident"
DISPLAY_LABEL_DETERMINISTIC_HEURISTIC = "Report heuristic (deterministic)"


@dataclass(frozen=True)
class DatasetStatus:
    """User-facing status for the dashboard's local data sources."""

    source_label: str
    runtime_source_label: str
    full_dataset_available: bool
    detail: str


@dataclass(frozen=True)
class DashboardData:
    """Normalized payload used by the recruiter-facing Streamlit dashboard."""

    summary: dict[str, Any]
    timeline: pd.DataFrame
    fleet_timeline: pd.DataFrame
    incidents: pd.DataFrame
    report: dict[str, Any]
    metrics: dict[str, Any]
    anomaly_threshold: float
    focus_asset_id: str
    dataset_status: DatasetStatus


def demo_artifacts_present() -> bool:
    """Return True when the checked-in demo bundle is coherent and usable."""
    return inspect_demo_bundle().healthy


def load_dashboard_data(focus_asset_id: str | None = None) -> DashboardData:
    """Load a demo-first dashboard payload with graceful local fallbacks."""
    bundle_status = inspect_demo_bundle()
    report = _load_report(bundle_status)
    fleet_timeline = _load_fleet_timeline(report, bundle_status)
    focus_asset_id = _resolve_focus_asset_id(report, fleet_timeline, focus_asset_id)
    timeline = (
        fleet_timeline.loc[fleet_timeline["asset_id"] == focus_asset_id]
        .sort_values("cycle")
        .reset_index(drop=True)
    )
    metrics = _load_metrics(bundle_status)
    incidents = _load_incident_cases(bundle_status)
    anomaly_threshold = _resolve_anomaly_threshold(fleet_timeline)
    fleet_timeline = _attach_rul_confidence_band(fleet_timeline, metrics, report)
    timeline = _attach_rul_confidence_band(timeline, metrics, report)
    summary = _build_summary(timeline, report, metrics, anomaly_threshold, focus_asset_id)
    dataset_status = _build_dataset_status(metrics, bundle_status)
    return DashboardData(
        summary=summary,
        timeline=timeline,
        fleet_timeline=fleet_timeline,
        incidents=incidents,
        report=report,
        metrics=metrics,
        anomaly_threshold=anomaly_threshold,
        focus_asset_id=focus_asset_id,
        dataset_status=dataset_status,
    )


def build_health_overview_figure(
    timeline: pd.DataFrame,
    incident_cycle: int | None = None,
) -> go.Figure:
    """Plot capacity-proxy drift and resistance drift for the focus asset."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=timeline["cycle"],
            y=timeline["capacity_ah"],
            mode="lines+markers",
            name=DISPLAY_LABEL_CAPACITY,
            line=dict(color=BATTERYOPS_COLORS["blue"], width=3),
            marker=dict(size=7),
            legendrank=1,
            hovertemplate=_format_metric_hover(DISPLAY_LABEL_CAPACITY, ".3f", "Ah"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=timeline["cycle"],
            y=timeline["internal_resistance_ohm"],
            mode="lines+markers",
            name=DISPLAY_LABEL_RESISTANCE,
            line=dict(color=BATTERYOPS_COLORS["amber"], width=2),
            marker=dict(size=6),
            legendrank=2,
            hovertemplate=_format_metric_hover(DISPLAY_LABEL_RESISTANCE, ".5f", "Ω"),
        ),
        secondary_y=True,
    )
    if incident_cycle is not None:
        fig.add_vline(
            x=incident_cycle,
            line_color=BATTERYOPS_COLORS["red"],
            line_dash="dot",
            annotation_text="Saved incident cycle",
            annotation_position="top left",
        )
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text=DISPLAY_LABEL_CAPACITY, secondary_y=False)
    fig.update_yaxes(title_text=DISPLAY_LABEL_RESISTANCE, secondary_y=True)
    fig.update_layout(hovermode="x unified")
    return _style_figure(
        fig,
        title="Health Trend: capacity proxy and resistance drift",
        height=410,
    )


def build_rul_confidence_figure(timeline: pd.DataFrame) -> go.Figure:
    """Plot the degradation-horizon proxy and a heuristic band."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["cycle"],
            y=timeline["predicted_rul_upper_cycles"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeline["cycle"],
            y=timeline["predicted_rul_lower_cycles"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(15, 39, 66, 0.12)",
            name="Heuristic band",
            legendrank=1,
            hovertemplate="Cycle %{x}<br>Heuristic lower bound %{y:.1f} cycles<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeline["cycle"],
            y=timeline["predicted_rul_cycles"],
            mode="lines+markers",
            name=DISPLAY_LABEL_PREDICTED_RUL,
            line=dict(color=BATTERYOPS_COLORS["navy"], width=3),
            marker=dict(size=7),
            legendrank=2,
            hovertemplate="Cycle %{x}<br>Heuristic proxy %{y:.1f} cycles<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text=DISPLAY_LABEL_PREDICTED_RUL_CYCLES)
    return _style_figure(fig, title="Predicted RUL with heuristic band", height=410)


def build_replay_figure(
    timeline: pd.DataFrame,
    replay_cycle: int,
    threshold: float,
) -> go.Figure:
    """Render a stepped telemetry replay up to the selected cycle."""
    replay_frame = timeline.loc[timeline["cycle"] <= replay_cycle].copy()
    current_row = replay_frame.iloc[-1]
    alert_rows = replay_frame.loc[replay_frame["anomaly_score"] >= threshold]
    actual_rows = replay_frame.loc[replay_frame["actual_alert"]]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        specs=[[{"secondary_y": True}], [{}]],
        row_heights=[0.58, 0.42],
    )
    fig.add_trace(
        go.Scatter(
            x=replay_frame["cycle"],
            y=replay_frame["capacity_ah"],
            mode="lines+markers",
            name=DISPLAY_LABEL_CAPACITY,
            line=dict(color=BATTERYOPS_COLORS["blue"], width=3),
            marker=dict(size=6),
            legendrank=1,
            hovertemplate=_format_metric_hover(DISPLAY_LABEL_CAPACITY, ".3f", "Ah"),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=replay_frame["cycle"],
            y=replay_frame["internal_resistance_ohm"],
            mode="lines",
            name=DISPLAY_LABEL_RESISTANCE,
            line=dict(color=BATTERYOPS_COLORS["amber"], width=2),
            legendrank=2,
            hovertemplate=_format_metric_hover(DISPLAY_LABEL_RESISTANCE, ".5f", "Ω"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=replay_frame["cycle"],
            y=replay_frame["anomaly_score"],
            mode="lines+markers",
            name=DISPLAY_LABEL_ANOMALY_SCORE,
            line=dict(color=BATTERYOPS_COLORS["navy"], width=3),
            marker=dict(size=6),
            legendrank=3,
            hovertemplate="Cycle %{x}<br>Anomaly score %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if not alert_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=alert_rows["cycle"],
                y=alert_rows["anomaly_score"],
                mode="markers",
                name=DISPLAY_LABEL_THRESHOLD_BREACH,
                marker=dict(
                    color=BATTERYOPS_COLORS["red"],
                    size=11,
                    symbol="diamond",
                    line=dict(width=1, color=BATTERYOPS_COLORS["surface"]),
                ),
                legendrank=4,
                hovertemplate="Cycle %{x}<br>Threshold breach %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if not actual_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_rows["cycle"],
                y=actual_rows["anomaly_score"],
                mode="markers",
                name=DISPLAY_LABEL_OBSERVED_INCIDENT,
                marker=dict(
                    color=BATTERYOPS_COLORS["amber"],
                    size=12,
                    symbol="x",
                    line=dict(width=2, color=BATTERYOPS_COLORS["ink"]),
                ),
                legendrank=5,
                hovertemplate="Cycle %{x}<br>Observed incident %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_vline(
        x=replay_cycle,
        line_color=BATTERYOPS_COLORS["red"],
        line_dash="dot",
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=BATTERYOPS_COLORS["red"],
        row=2,
        col=1,
        annotation_text=f"Threshold = {threshold:.2f}",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=[current_row["cycle"]],
            y=[current_row["capacity_ah"]],
            mode="markers",
            name=f"{DISPLAY_LABEL_CAPACITY} cursor",
            marker=dict(
                color=BATTERYOPS_COLORS["red"],
                size=13,
                symbol="circle-open",
                line=dict(width=3),
            ),
            showlegend=False,
            customdata=np.array([current_row["cycle"]]),
            hovertemplate=(
                "Replay cycle %{customdata[0]}<br>"
                "Cursor on the current focus cycle<extra></extra>"
            ),
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_xaxes(title_text="Cycle", row=2, col=1)
    fig.update_yaxes(title_text=DISPLAY_LABEL_CAPACITY, row=1, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text=DISPLAY_LABEL_RESISTANCE,
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(title_text=DISPLAY_LABEL_ANOMALY_SCORE, row=2, col=1)
    fig = _style_figure(fig, title="Telemetry Replay by Cycle", height=620)
    fig.update_layout(
        margin=dict(l=72, r=72, t=72, b=96),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    fig.update_xaxes(automargin=True, title_standoff=10, row=2, col=1)
    fig.update_yaxes(automargin=True, title_standoff=14, row=1, col=1, secondary_y=False)
    fig.update_yaxes(automargin=True, title_standoff=14, row=1, col=1, secondary_y=True)
    fig.update_yaxes(automargin=True, title_standoff=14, row=2, col=1)
    return fig


def build_anomaly_timeline_figure(timeline: pd.DataFrame, threshold: float) -> go.Figure:
    """Plot anomaly scores, threshold, and alert markers across the asset history."""
    flagged = timeline.loc[timeline["anomaly_score"] >= threshold]
    incidents = timeline.loc[timeline["actual_alert"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["cycle"],
            y=timeline["anomaly_score"],
            mode="lines+markers",
            name=DISPLAY_LABEL_ANOMALY_SCORE,
            line=dict(color=BATTERYOPS_COLORS["navy"], width=3),
            marker=dict(size=7),
            legendrank=1,
            hovertemplate="Cycle %{x}<br>Anomaly score %{y:.3f}<extra></extra>",
        )
    )
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged["cycle"],
                y=flagged["anomaly_score"],
                mode="markers",
                name=DISPLAY_LABEL_THRESHOLD_BREACH,
                marker=dict(
                    color=BATTERYOPS_COLORS["red"],
                    size=11,
                    symbol="diamond",
                ),
                legendrank=2,
                hovertemplate="Cycle %{x}<br>Threshold breach %{y:.3f}<extra></extra>",
            )
        )
    if not incidents.empty:
        fig.add_trace(
            go.Scatter(
                x=incidents["cycle"],
                y=incidents["anomaly_score"],
                mode="markers",
                name=DISPLAY_LABEL_OBSERVED_INCIDENT,
                marker=dict(
                    color=BATTERYOPS_COLORS["amber"],
                    size=12,
                    symbol="x",
                    line=dict(width=2, color=BATTERYOPS_COLORS["ink"]),
                ),
                legendrank=3,
                hovertemplate="Cycle %{x}<br>Observed incident score %{y:.3f}<extra></extra>",
            )
        )

    fig.add_hrect(
        y0=threshold,
        y1=max(float(timeline["anomaly_score"].max()), threshold),
        fillcolor="rgba(201, 76, 76, 0.10)",
        line_width=0,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=BATTERYOPS_COLORS["red"],
        annotation_text=f"Threshold = {threshold:.2f}",
        annotation_position="top left",
    )
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text=DISPLAY_LABEL_ANOMALY_SCORE)
    fig.update_layout(hovermode="x unified")
    return _style_figure(fig, title="Anomaly Timeline: thresholded proxy", height=470)


def build_report_confidence_figure(report: dict[str, Any]) -> go.Figure:
    """Show the deterministic report heuristic as a compact engineering-style gauge."""
    confidence_pct = float(report.get("confidence_score", 0.0)) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            number={"suffix": "%", "font": {"size": 40, "color": BATTERYOPS_COLORS["ink"]}},
            title={"text": DISPLAY_LABEL_DETERMINISTIC_HEURISTIC},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": BATTERYOPS_COLORS["navy"]},
                "steps": [
                    {"range": [0, 40], "color": "rgba(201, 76, 76, 0.16)"},
                    {"range": [40, 70], "color": "rgba(244, 185, 66, 0.18)"},
                    {"range": [70, 100], "color": "rgba(31, 138, 112, 0.18)"},
                ],
            },
        )
    )
    return _style_figure(fig, height=300)


def build_incident_marker_figure(
    timeline: pd.DataFrame,
    incident_cycle: int | None,
    threshold: float,
) -> go.Figure:
    """Render the anomaly timeline centered on the saved incident report cycle."""
    fig = build_anomaly_timeline_figure(timeline, threshold)
    if incident_cycle is not None:
        fig.add_vline(
            x=incident_cycle,
            line_color=BATTERYOPS_COLORS["red"],
            line_dash="dot",
            annotation_text=f"Saved incident cycle {incident_cycle}",
            annotation_position="top right",
        )
    return fig


def build_similar_cases_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Convert the report's similar cases into a recruiter-friendly table."""
    cases = pd.DataFrame(report.get("similar_cases", []))
    if cases.empty:
        return pd.DataFrame(
            columns=[
                "case_label",
                "asset_id",
                "cycle_id",
                "incident_label",
                "distance",
                "severity_score",
                "type_overlap",
            ]
        )

    focus_types = set(_incident_tokens(report.get("incident_types", [])))
    normalized_types = cases["incident_types"].apply(_incident_tokens)
    cases["incident_label"] = normalized_types.apply(_format_incident_label)
    cases["type_overlap"] = normalized_types.apply(lambda tokens: len(set(tokens) & focus_types))
    cases["distance"] = pd.to_numeric(cases["distance"], errors="coerce").round(6)
    cases["severity_score"] = pd.to_numeric(cases["severity_score"], errors="coerce").round(6)
    cases["case_label"] = (
        cases["asset_id"].astype(str) + " · cycle " + cases["cycle_id"].astype(int).astype(str)
    )
    ordered_columns = [
        "case_label",
        "asset_id",
        "cycle_id",
        "incident_label",
        "distance",
        "severity_score",
        "type_overlap",
    ]
    return (
        cases[ordered_columns]
        .sort_values(by=["distance", "severity_score"], ascending=[True, False])
        .copy()
    )


def build_similar_cases_figure(cases: pd.DataFrame) -> go.Figure:
    """Plot retrieval similarity against case severity."""
    fig = go.Figure()
    if cases.empty:
        fig.add_annotation(
            text="No similar historical cases were available in the local artifact bundle.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=BATTERYOPS_COLORS["muted"], size=14),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return _style_figure(fig, title="Similar Case Retrieval", height=400)

    fig.add_trace(
        go.Scatter(
            x=cases["distance"],
            y=cases["severity_score"],
            mode="markers+text",
            text=cases["case_label"],
            textposition="top center",
            marker=dict(
                size=cases["type_overlap"].astype(float) * 6 + 12,
                color=cases["type_overlap"],
                colorscale=[
                    [0.0, BATTERYOPS_COLORS["amber"]],
                    [1.0, BATTERYOPS_COLORS["navy"]],
                ],
                line=dict(width=1, color=BATTERYOPS_COLORS["surface"]),
                colorbar=dict(title="Shared incident types"),
            ),
            customdata=np.stack([cases["incident_label"]], axis=-1),
            hovertemplate=(
                "%{text}<br>Retrieval distance %{x:.2f}<br>Severity score %{y:.2f}<br>"
                "Incident types: %{customdata[0]}<extra></extra>"
            ),
            legendrank=1,
        )
    )
    fig.update_xaxes(title_text="Retrieval distance (lower is closer)")
    fig.update_yaxes(title_text="Severity score")
    return _style_figure(fig, title="Similar Case Retrieval", height=430)


def build_confusion_matrix_figure(fleet_timeline: pd.DataFrame) -> go.Figure:
    """Summarize alert performance across the saved demo bundle."""
    predicted = fleet_timeline["predicted_alert"].astype(bool)
    actual = fleet_timeline["actual_alert"].astype(bool)
    confusion = np.array(
        [
            [
                int((~predicted & ~actual).sum()),
                int((predicted & ~actual).sum()),
            ],
            [
                int((~predicted & actual).sum()),
                int((predicted & actual).sum()),
            ],
        ]
    )
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=confusion,
                x=["Monitor", "Inspect soon"],
                y=["Monitor", "Incident"],
                colorscale=[
                    [0.0, "#edf3f8"],
                    [0.5, "#88aacd"],
                    [1.0, BATTERYOPS_COLORS["navy"]],
                ],
                text=confusion,
                texttemplate="%{text}",
                showscale=False,
            )
        ]
    )
    fig.update_xaxes(title_text="Predicted alert state")
    fig.update_yaxes(title_text="Observed alert state")
    return _style_figure(fig, title="Alert Confusion Matrix (fleet)", height=360)


def build_rul_scatter_figure(fleet_timeline: pd.DataFrame) -> go.Figure:
    """Compare proxy RUL predictions against the degradation-threshold target."""
    max_rul = float(
        max(
            fleet_timeline["actual_rul_cycles"].max(),
            fleet_timeline["predicted_rul_cycles"].max(),
        )
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fleet_timeline["actual_rul_cycles"],
            y=fleet_timeline["predicted_rul_cycles"],
            mode="markers",
            marker=dict(
                color=fleet_timeline["anomaly_score"],
                colorscale="Blues",
                size=9,
                showscale=True,
                colorbar=dict(title="Anomaly score"),
                line=dict(width=1, color=BATTERYOPS_COLORS["surface"]),
            ),
            customdata=np.stack([fleet_timeline["asset_id"], fleet_timeline["cycle"]], axis=-1),
            hovertemplate=(
                "Asset %{customdata[0]}<br>Cycle %{customdata[1]}"
                "<br>Actual proxy RUL %{x:.1f} cycles"
                "<br>Predicted proxy RUL %{y:.1f} cycles<extra></extra>"
            ),
            name="Fleet snapshot",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_rul],
            y=[0, max_rul],
            mode="lines",
            name="Ideal fit",
            line=dict(color=BATTERYOPS_COLORS["red"], dash="dash"),
        )
    )
    fig.update_xaxes(title_text="Actual proxy RUL (cycles)")
    fig.update_yaxes(title_text=DISPLAY_LABEL_PREDICTED_RUL_CYCLES)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _style_figure(fig, title="Proxy RUL Agreement (fleet)", height=360)


def build_alert_coverage_figure(fleet_timeline: pd.DataFrame) -> go.Figure:
    """Compare predicted versus observed alert counts by asset."""
    coverage = (
        fleet_timeline.groupby("asset_id", as_index=False)
        .agg(
            predicted_alerts=("predicted_alert", "sum"),
            observed_incidents=("actual_alert", "sum"),
        )
        .sort_values("asset_id")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=coverage["asset_id"],
            y=coverage["predicted_alerts"],
            name="Predicted inspect cycles",
            marker_color=BATTERYOPS_COLORS["navy"],
            hovertemplate="Asset %{x}<br>Predicted inspect cycles: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=coverage["asset_id"],
            y=coverage["observed_incidents"],
            name="Observed incident cycles",
            marker_color=BATTERYOPS_COLORS["amber"],
            hovertemplate="Asset %{x}<br>Observed incident cycles: %{y}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Asset")
    fig.update_yaxes(title_text="Cycle count")
    return _style_figure(fig, title="Alert Coverage by Asset", height=360)


def build_focus_incident_row(report: dict[str, Any], incidents: pd.DataFrame) -> pd.Series | None:
    """Find the saved incident inside the local incident artifact bundle."""
    if incidents.empty:
        return None

    matches = incidents.loc[
        (incidents["asset_id"] == str(report.get("asset_id", "")))
        & (incidents["cycle_id"].astype(int) == int(report.get("cycle_id", 0)))
    ]
    if matches.empty:
        return None
    return matches.sort_values("severity_score", ascending=False).iloc[0]


def build_recent_cycle_table(timeline: pd.DataFrame, row_count: int = 8) -> pd.DataFrame:
    """Format the latest cycles for a recruiter-friendly detail table."""
    frame = (
        timeline.tail(row_count)[
            [
                "cycle",
                "capacity_ah",
                "internal_resistance_ohm",
                "anomaly_score",
                "predicted_rul_cycles",
                "status",
            ]
        ]
        .rename(
            columns={
                "cycle": "Cycle",
                "capacity_ah": DISPLAY_LABEL_CAPACITY,
                "internal_resistance_ohm": DISPLAY_LABEL_RESISTANCE,
                "anomaly_score": "Anomaly score",
                "predicted_rul_cycles": DISPLAY_LABEL_PREDICTED_RUL_CYCLES,
                "status": "Alert state",
            }
        )
        .copy()
    )
    numeric_columns = [
        DISPLAY_LABEL_CAPACITY,
        DISPLAY_LABEL_RESISTANCE,
        DISPLAY_LABEL_ANOMALY_SCORE,
        DISPLAY_LABEL_PREDICTED_RUL_CYCLES,
    ]
    frame[numeric_columns] = frame[numeric_columns].round(3)
    frame["Alert state"] = frame["Alert state"].map(_format_alert_state)
    return frame.reset_index(drop=True)


def build_flagged_cycle_table(timeline: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return the cycles that would be flagged at the selected threshold."""
    flagged = timeline.loc[timeline["anomaly_score"] >= threshold].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "Cycle",
                "Anomaly score",
                DISPLAY_LABEL_PREDICTED_RUL_CYCLES,
                "Observed incident",
                "Alert state",
            ]
        )
    flagged = flagged.rename(
        columns={
            "cycle": "Cycle",
            "anomaly_score": DISPLAY_LABEL_ANOMALY_SCORE,
            "predicted_rul_cycles": DISPLAY_LABEL_PREDICTED_RUL_CYCLES,
            "actual_alert": "Observed incident",
            "status": "Alert state",
        }
    )[
        [
            "Cycle",
            DISPLAY_LABEL_ANOMALY_SCORE,
            DISPLAY_LABEL_PREDICTED_RUL_CYCLES,
            "Observed incident",
            "Alert state",
        ]
    ]
    flagged[[DISPLAY_LABEL_ANOMALY_SCORE, DISPLAY_LABEL_PREDICTED_RUL_CYCLES]] = flagged[
        [DISPLAY_LABEL_ANOMALY_SCORE, DISPLAY_LABEL_PREDICTED_RUL_CYCLES]
    ].round(3)
    flagged["Alert state"] = flagged["Alert state"].map(_format_alert_state)
    flagged["Observed incident"] = flagged["Observed incident"].map(
        lambda value: "Yes" if bool(value) else "No"
    )
    return flagged.reset_index(drop=True)


def _build_summary(
    timeline: pd.DataFrame,
    report: dict[str, Any],
    metrics: dict[str, Any],
    anomaly_threshold: float,
    focus_asset_id: str,
) -> dict[str, Any]:
    latest = timeline.sort_values("cycle").iloc[-1]
    confidence_score = float(report.get("confidence_score", 0.0))
    alert_level = "Inspect soon" if bool(latest["predicted_alert"]) else "Monitor"
    triage_note = str(report.get("summary", "")).strip()
    if not triage_note:
        triage_note = (
            "The current artifact bundle did not include a saved report. "
            "No valid demo report artifact was found, so BatteryOps is showing "
            "telemetry-only fallback outputs."
        )
    report_asset_id = str(report.get("asset_id", "")).strip()
    if report_asset_id and report_asset_id != focus_asset_id:
        triage_note = (
            triage_note
            + f" The saved incident report in the bundle is tied to asset {report_asset_id}."
        )
    return {
        "asset_id": focus_asset_id,
        "latest_cycle": int(latest["cycle"]),
        "estimated_rul_cycles": int(round(float(latest["predicted_rul_cycles"]))),
        "health_score": _resolve_health_score(timeline),
        "alert_level": alert_level,
        "triage_note": triage_note,
        "confidence_score": confidence_score,
        "grounding_coverage": float(
            metrics.get("evidence_source_coverage", metrics.get("report_grounding_coverage", 0.0))
        ),
        "data_source": str(metrics.get("data_source", "demo_fallback")),
        "anomaly_threshold": anomaly_threshold,
        "focus_incident_cycle": int(report.get("cycle_id", latest["cycle"])),
    }


def _resolve_health_score(timeline: pd.DataFrame) -> float:
    """Return a bounded capacity-retention heuristic for the current focus asset."""
    capacity = pd.to_numeric(timeline["capacity_ah"], errors="coerce")
    if capacity.empty:
        return 0.0

    non_zero_capacity = capacity.replace(0.0, np.nan).dropna()
    if non_zero_capacity.empty:
        return 0.0

    baseline_window = non_zero_capacity.head(min(10, len(non_zero_capacity)))
    baseline_capacity = float(baseline_window.max())
    latest_capacity = float(non_zero_capacity.iloc[-1])
    if baseline_capacity <= 0.0 or not np.isfinite(baseline_capacity):
        return 0.0

    retention = np.clip(latest_capacity / baseline_capacity, 0.0, 1.0)
    return round(float(retention * 100.0), 1)


def _load_report(bundle_status: DemoBundleStatus) -> dict[str, Any]:
    report = build_demo_report()
    if report is not None:
        return report

    return {
        "report_id": "demo-report-missing",
        "asset_id": "NASA-B0005-demo",
        "cycle_id": 0,
        "incident_types": [],
        "summary": "",
        "evidence": [],
        "similar_cases": [],
        "recommended_tests": [],
        "confidence_score": 0.0,
        "confidence_caveats": [
            (
                "No valid demo report artifact was found, so the dashboard is using "
                "telemetry-only fallback outputs."
            ),
            bundle_status.reason,
        ],
        "grounding_coverage": 0.0,
    }


def _load_metrics(bundle_status: DemoBundleStatus) -> dict[str, Any]:
    if not bundle_status.healthy or bundle_status.metrics is None:
        return {}
    return dict(bundle_status.metrics)


def _load_fleet_timeline(
    report: dict[str, Any],
    bundle_status: DemoBundleStatus,
) -> pd.DataFrame:
    if bundle_status.healthy and bundle_status.artifact_dir is not None:
        timeline_path = bundle_status.artifact_dir / "demo_cycle_predictions.parquet"
        if timeline_path.exists():
            frame = pd.read_parquet(timeline_path)
            if not frame.empty:
                normalized = _normalize_cycle_table(frame)
                reported_asset_id = str(report.get("asset_id", "")).strip()
                if reported_asset_id and reported_asset_id in set(
                    normalized["asset_id"].astype(str)
                ):
                    return normalized

    timeline = build_demo_timeline().copy()
    timeline["asset_id"] = str(report.get("asset_id", "NASA-B0005-demo"))
    timeline["cycle_id"] = timeline["cycle"].astype(int)
    return _normalize_cycle_table(timeline)


def _load_incident_cases(bundle_status: DemoBundleStatus) -> pd.DataFrame:
    if not bundle_status.healthy or bundle_status.artifact_dir is None:
        return pd.DataFrame()
    incident_path = bundle_status.artifact_dir / "demo_incident_cases.parquet"
    if not incident_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(incident_path)


def _normalize_cycle_table(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "cycle" not in normalized and "cycle_id" in normalized:
        normalized["cycle"] = pd.to_numeric(normalized["cycle_id"], errors="coerce").astype(int)
    if "cycle_id" not in normalized and "cycle" in normalized:
        normalized["cycle_id"] = pd.to_numeric(normalized["cycle"], errors="coerce").astype(int)
    if "asset_id" not in normalized:
        normalized["asset_id"] = "NASA-B0005-demo"

    numeric_columns = [
        "capacity_ah",
        "internal_resistance_ohm",
        "anomaly_score",
        "predicted_rul_cycles",
        "actual_rul_cycles",
    ]
    for column in numeric_columns:
        if column not in normalized:
            continue
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "predicted_alert" not in normalized:
        status_series = (
            normalized["status"].astype(str)
            if "status" in normalized
            else pd.Series("", index=normalized.index, dtype=str)
        )
        normalized["predicted_alert"] = status_series.eq("inspect soon")
    normalized["predicted_alert"] = normalized["predicted_alert"].astype(bool)

    if "status" not in normalized:
        normalized["status"] = np.where(normalized["predicted_alert"], "inspect soon", "monitor")
    normalized["status"] = normalized["status"].astype(str)

    if "actual_alert" not in normalized:
        normalized["actual_alert"] = False
    normalized["actual_alert"] = normalized["actual_alert"].astype(bool)

    if "predicted_rul_cycles" not in normalized:
        normalized["predicted_rul_cycles"] = _estimate_rul_from_capacity(normalized)
    normalized["predicted_rul_cycles"] = normalized["predicted_rul_cycles"].fillna(
        _estimate_rul_from_capacity(normalized)
    )

    if "actual_rul_cycles" not in normalized:
        asset_cycle_max = normalized.groupby("asset_id")["cycle"].transform("max")
        normalized["actual_rul_cycles"] = (asset_cycle_max - normalized["cycle"]).clip(lower=0)

    return normalized.sort_values(["asset_id", "cycle"]).reset_index(drop=True)


def _estimate_rul_from_capacity(frame: pd.DataFrame) -> pd.Series:
    rolling_drop = (
        -pd.to_numeric(frame["capacity_ah"], errors="coerce")
        .diff()
        .rolling(5, min_periods=1)
        .mean()
    )
    bounded_drop = rolling_drop.fillna(0.0).clip(lower=0.001)
    estimated = (pd.to_numeric(frame["capacity_ah"], errors="coerce") - 1.4) / bounded_drop
    return estimated.fillna(0.0).clip(lower=0.0)


def _resolve_focus_asset_id(
    report: dict[str, Any],
    fleet_timeline: pd.DataFrame,
    preferred_focus_asset_id: str | None = None,
) -> str:
    if preferred_focus_asset_id is not None:
        preferred = str(preferred_focus_asset_id).strip()
        if preferred and preferred in set(fleet_timeline["asset_id"].astype(str)):
            return preferred
    reported_asset = str(report.get("asset_id", "")).strip()
    if reported_asset and reported_asset in set(fleet_timeline["asset_id"].astype(str)):
        return reported_asset
    return str(fleet_timeline.sort_values(["anomaly_score", "cycle"]).iloc[-1]["asset_id"])


def _resolve_anomaly_threshold(fleet_timeline: pd.DataFrame) -> float:
    alerted = fleet_timeline.loc[fleet_timeline["predicted_alert"], "anomaly_score"]
    if not alerted.empty:
        return round(float(alerted.min()), 3)
    return round(float(fleet_timeline["anomaly_score"].quantile(0.8)), 3)


def _attach_rul_confidence_band(
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    report: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return frame

    enriched = frame.copy()
    confidence = float(report.get("confidence_score", 0.65) or 0.65)
    mae = float(metrics.get("rul_proxy_mae", metrics.get("rul_mae", 3.0)) or 3.0)
    relative_band = enriched["predicted_rul_cycles"] * max(0.08, (1.0 - confidence) * 0.45)
    absolute_band = pd.Series(mae * 1.5, index=enriched.index, dtype=float)
    margin = np.maximum(relative_band, absolute_band)
    enriched["predicted_rul_lower_cycles"] = (enriched["predicted_rul_cycles"] - margin).clip(
        lower=0.0
    )
    enriched["predicted_rul_upper_cycles"] = enriched["predicted_rul_cycles"] + margin
    return enriched


def _build_dataset_status(
    metrics: dict[str, Any],
    bundle_status: DemoBundleStatus,
) -> DatasetStatus:
    processed_present = all(path.exists() for path in PROCESSED_EXPECTED)
    demo_bundle_present = bundle_status.healthy
    data_source = str(metrics.get("data_source", "demo_fallback"))
    asset_count = int(metrics.get("asset_count", 0) or 0)
    cycle_count = int(metrics.get("cycle_count", 0) or 0)
    if demo_bundle_present:
        source_label = (
            f"Processed NASA artifacts in the checked-in demo bundle ({asset_count} assets)"
            if data_source == "processed"
            else "Checked-in demo bundle"
        )
        runtime_source_label = "checked-in demo bundle"
        if data_source == "processed":
            detail = (
                "The checked-in demo bundle was regenerated from processed NASA data "
                f"covering {asset_count} assets and {cycle_count} cycle snapshots."
            )
            if processed_present:
                detail += (
                    " The local processed parquet cache is also available in this checkout "
                    "for regeneration workflows."
                )
            else:
                detail += (
                    " The local processed parquet cache is not present in this checkout, "
                    "but the saved bundle is still sufficient for the public demo."
                )
        else:
            detail = (
                "The checked-in demo bundle is coherent, but the metrics indicate a "
                "deterministic fallback data source rather than processed NASA parquet."
            )
    else:
        source_label = "Synthetic demo fallback"
        runtime_source_label = "synthetic demo fallback"
        detail = (
            "Processed NASA parquet was not found locally. "
            f"No complete demo bundle was found ({bundle_status.reason})."
        )
        if processed_present:
            detail += (
                " Local processed NASA parquet is available for regeneration, but the app "
                "is using synthetic fallback data because the demo bundle is incomplete."
            )
        else:
            detail += (
                " Local processed NASA parquet is also absent, so the app is using synthetic "
                "fallback data."
            )
    return DatasetStatus(
        source_label=source_label,
        runtime_source_label=runtime_source_label,
        full_dataset_available=processed_present,
        detail=detail,
    )


def _style_figure(
    fig: go.Figure,
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    if title is not None:
        fig.update_layout(
            title={
                "text": title,
                "x": 0.0,
                "xanchor": "left",
                "font": {"size": 18},
            }
        )
    fig.update_layout(
        template="simple_white",
        height=height,
        paper_bgcolor=BATTERYOPS_COLORS["surface"],
        plot_bgcolor=BATTERYOPS_COLORS["surface"],
        font=dict(color=BATTERYOPS_COLORS["ink"]),
        margin=dict(l=24, r=24, t=68, b=28),
        hoverlabel=dict(
            bgcolor=BATTERYOPS_COLORS["surface"],
            font=dict(color=BATTERYOPS_COLORS["ink"]),
            align="left",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.75)",
            font=dict(size=12),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BATTERYOPS_COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=BATTERYOPS_COLORS["grid"], zeroline=False)
    return fig


def _format_metric_hover(label: str, value_format: str, unit: str) -> str:
    return f"Cycle %{{x}}<br>{label} %{{y:{value_format}}} {unit}<extra></extra>"


def _incident_tokens(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token).strip() for token in raw_value if str(token).strip()]
    return [token.strip() for token in str(raw_value).split(",") if token.strip()]


def _format_incident_label(tokens: list[str]) -> str:
    if not tokens:
        return "No incident type"
    return ", ".join(token.replace("_", " ").title() for token in tokens)


def _format_alert_state(raw_value: object) -> str:
    normalized = " ".join(str(raw_value).replace("_", " ").split()).lower()
    if not normalized:
        return "Not available"
    return normalized[0].upper() + normalized[1:]
