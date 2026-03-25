from __future__ import annotations

from batteryops.dashboard import (
    build_confusion_matrix_figure,
    build_flagged_cycle_table,
    build_health_overview_figure,
    build_recent_cycle_table,
    build_replay_figure,
    build_rul_scatter_figure,
    build_similar_cases_frame,
    load_dashboard_data,
)


def test_replay_figure_uses_readable_layout() -> None:
    data = load_dashboard_data()

    figure = build_replay_figure(
        data.timeline,
        replay_cycle=int(data.timeline["cycle"].max()),
        threshold=float(data.anomaly_threshold),
    )

    assert figure.layout.title.text == "Telemetry Replay by Cycle"
    assert figure.layout.legend.y == -0.2
    assert figure.layout.margin.l == 72
    assert figure.layout.margin.r == 72
    assert figure.layout.margin.t == 72
    assert figure.layout.margin.b == 96
    assert figure.layout.yaxis.title.text == "Capacity proxy (Ah)"
    assert figure.layout.yaxis2.title.text == "Internal resistance (Ohm)"


def test_alert_tables_use_polished_display_labels() -> None:
    data = load_dashboard_data()

    recent_cycles = build_recent_cycle_table(data.timeline, row_count=6)
    flagged_cycles = build_flagged_cycle_table(
        data.timeline,
        threshold=float(data.anomaly_threshold),
    )

    assert "Capacity proxy (Ah)" in recent_cycles.columns
    assert set(recent_cycles["Alert state"]).issubset({"Inspect soon", "Monitor"})
    if not flagged_cycles.empty:
        assert set(flagged_cycles["Alert state"]).issubset({"Inspect soon", "Monitor"})
        assert set(flagged_cycles["Observed incident"]).issubset({"Yes", "No"})


def test_health_and_evaluation_figures_use_cleaner_labels() -> None:
    data = load_dashboard_data()

    health_figure = build_health_overview_figure(data.timeline)
    confusion_figure = build_confusion_matrix_figure(data.fleet_timeline)
    rul_figure = build_rul_scatter_figure(data.fleet_timeline)

    assert health_figure.layout.title.text == "Health Trend: capacity proxy and resistance drift"
    assert health_figure.layout.yaxis.title.text == "Capacity proxy (Ah)"
    assert health_figure.layout.yaxis2.title.text == "Internal resistance (Ohm)"
    assert list(confusion_figure.data[0].x) == ["Monitor", "Inspect soon"]
    assert list(confusion_figure.data[0].y) == ["Monitor", "Incident"]
    assert rul_figure.layout.yaxis.scaleanchor == "x"


def test_similar_cases_frame_is_sorted_and_labeled_for_recruiter_review() -> None:
    report = {
        "incident_types": ["low_voltage", "high_temperature"],
        "similar_cases": [
            {
                "window_id": "A-10-1",
                "asset_id": "A",
                "cycle_id": 10,
                "incident_types": "low_voltage,high_temperature",
                "distance": 0.30,
                "severity_score": 3.1,
            },
            {
                "window_id": "B-18-1",
                "asset_id": "B",
                "cycle_id": 18,
                "incident_types": "high_temperature,low_voltage",
                "distance": 0.10,
                "severity_score": 2.5,
            },
            {
                "window_id": "C-05-1",
                "asset_id": "C",
                "cycle_id": 5,
                "incident_types": "high_current",
                "distance": 0.10,
                "severity_score": 1.2,
            },
        ],
    }

    frame = build_similar_cases_frame(report)

    assert list(frame.columns) == [
        "case_label",
        "asset_id",
        "cycle_id",
        "incident_label",
        "distance",
        "severity_score",
        "type_overlap",
    ]
    assert list(frame["asset_id"]) == ["B", "C", "A"]
    assert list(frame["cycle_id"]) == [18, 5, 10]
    assert list(frame["incident_label"]) == [
        "High Temperature, Low Voltage",
        "High Current",
        "Low Voltage, High Temperature",
    ]
    assert list(frame["type_overlap"]) == [2, 0, 2]
