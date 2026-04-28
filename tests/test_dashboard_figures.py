from __future__ import annotations

from batteryops.dashboard import (
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
    build_health_overview_figure,
    build_recent_cycle_table,
    build_replay_figure,
    build_rul_scatter_figure,
    build_similar_cases_figure,
    build_similar_cases_frame,
    build_threshold_tradeoff_figure,
    build_triage_handoff_frame,
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
    assert figure.layout.yaxis.title.text == "Health index (%)"
    assert figure.layout.yaxis2.title.text == "Internal resistance (Ohm)"


def test_alert_tables_use_polished_display_labels() -> None:
    data = load_dashboard_data()

    recent_cycles = build_recent_cycle_table(data.timeline, row_count=6)
    flagged_cycles = build_flagged_cycle_table(
        data.timeline,
        threshold=float(data.anomaly_threshold),
    )

    assert "Health index (%)" in recent_cycles.columns
    assert set(recent_cycles["Alert state"]).issubset({"Inspect soon", "Monitor"})
    if not flagged_cycles.empty:
        assert set(flagged_cycles["Alert state"]).issubset({"Inspect soon", "Monitor"})
        assert set(flagged_cycles["Observed incident"]).issubset({"Yes", "No"})


def test_health_and_evaluation_figures_use_cleaner_labels() -> None:
    data = load_dashboard_data()

    health_figure = build_health_overview_figure(data.timeline)
    confusion_figure = build_confusion_matrix_figure(data.fleet_timeline)
    rul_figure = build_rul_scatter_figure(data.fleet_timeline)

    assert health_figure.layout.title.text == "Health Trend: normalized health and resistance drift"
    assert health_figure.layout.yaxis.title.text == "Health index (%)"
    assert health_figure.layout.yaxis2.title.text == "Internal resistance (Ohm)"
    assert list(confusion_figure.data[0].x) == ["Monitor", "Inspect soon"]
    assert list(confusion_figure.data[0].y) == ["Incident", "Monitor"]
    assert rul_figure.layout.yaxis.scaleanchor == "x"


def test_fleet_risk_map_and_priority_table_summarize_asset_queue() -> None:
    data = load_dashboard_data()

    figure = build_fleet_risk_figure(
        data.fleet_timeline,
        focus_asset_id=data.focus_asset_id,
        anomaly_threshold=float(data.anomaly_threshold),
    )
    priority = build_asset_priority_table(data.fleet_timeline, row_count=6)

    assert figure.layout.title.text == "Fleet Risk Map"
    assert figure.layout.xaxis.title.text == "Latest health index (%)"
    assert figure.layout.yaxis.title.text == "Maximum anomaly score"
    assert len(figure.data) >= 1
    assert list(priority.columns) == [
        "Asset",
        "Priority score",
        "Latest health (%)",
        "Max anomaly",
        "Inspect cycles",
        "Observed incidents",
        "Proxy RUL",
        "Latest state",
    ]
    assert len(priority) == 6
    assert priority["Priority score"].is_monotonic_decreasing
    assert set(priority["Latest state"]).issubset({"Inspect soon", "Monitor"})


def test_fleet_risk_concentration_and_decision_ledger_explain_queue() -> None:
    data = load_dashboard_data()

    figure = build_fleet_risk_concentration_figure(data.fleet_timeline, row_count=6)
    ledger = build_cockpit_decision_frame(
        data.fleet_timeline,
        data.focus_asset_id,
        float(data.anomaly_threshold),
    )

    assert figure.layout.title.text == "Fleet Risk Concentration"
    assert figure.layout.xaxis.title.text == "Priority score"
    assert figure.data[0].orientation == "h"
    assert len(figure.data[0].x) == 6
    assert list(ledger.columns) == ["Layer", "State", "Evidence", "Next action"]
    assert list(ledger["Layer"]) == [
        "Fleet pressure",
        "Highest-risk asset",
        "Selected asset",
        "Runtime boundary",
    ]
    assert "$0 local bundle" in set(ledger["State"])
    assert ledger["Evidence"].str.len().min() > 20


def test_asset_risk_driver_breakdown_explains_selected_asset_priority() -> None:
    data = load_dashboard_data()

    drivers = build_asset_risk_driver_frame(data.fleet_timeline, data.focus_asset_id)
    figure = build_asset_risk_driver_figure(drivers)

    assert list(drivers.columns) == ["Driver", "Weight", "Evidence", "Contribution"]
    assert set(drivers["Driver"]) == {
        "Anomaly pressure",
        "Inspect rate",
        "Health loss",
        "Observed incident rate",
    }
    assert figure.layout.title.text == "Selected-Asset Risk Drivers"
    assert figure.layout.xaxis.title.text == "Priority-score contribution"
    assert drivers["Contribution"].sum() > 0


def test_triage_handoff_turns_asset_state_into_review_actions() -> None:
    data = load_dashboard_data()

    handoff = build_triage_handoff_frame(
        data.fleet_timeline,
        data.focus_asset_id,
        float(data.anomaly_threshold),
    )

    assert list(handoff.columns) == ["Review step", "Decision", "Evidence", "Why it matters"]
    assert list(handoff["Review step"]) == [
        "Queue rank",
        "Alert trigger",
        "Health context",
        "Diagnostics handoff",
    ]
    alert_decision = handoff.loc[handoff["Review step"] == "Alert trigger", "Decision"].iloc[0]
    assert alert_decision in {"Inspect soon", "Monitor"}
    assert handoff["Decision"].astype(str).str.len().min() > 6
    assert handoff["Evidence"].str.len().min() > 10


def test_artifact_inventory_frame_shows_hash_prefixes() -> None:
    data = load_dashboard_data()

    inventory = build_artifact_inventory_frame(data.artifact_inventory)

    assert list(inventory.columns) == ["Artifact", "Size KB", "SHA-256 prefix"]
    assert "training_manifest.json" in set(inventory["Artifact"])
    assert inventory["SHA-256 prefix"].str.len().min() == 12
    assert data.bundle_fingerprint is not None


def test_model_evaluation_detail_figures_render_from_artifact_reports() -> None:
    data = load_dashboard_data()

    threshold_figure = build_threshold_tradeoff_figure(data.evaluation_report)
    asset_error_figure = build_asset_error_figure(data.evaluation_report)
    feature_figure = build_feature_importance_figure(data.model_card)

    assert threshold_figure.layout.title.text == "Alert Threshold Tradeoff"
    assert asset_error_figure.layout.title.text == "Per-Asset Proxy RUL Error"
    assert feature_figure.layout.title.text == "Model Input Signal Ranking"
    assert len(threshold_figure.data) >= 3
    assert len(feature_figure.data) >= 1


def test_similar_cases_frame_is_sorted_and_labeled_for_review() -> None:
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

    figure = build_similar_cases_figure(frame)

    assert figure.layout.title.text == "Closest Saved Incidents"
    assert figure.data[0].orientation == "h"
    assert figure.layout.xaxis.title.text == "Retrieval distance (lower is closer)"
