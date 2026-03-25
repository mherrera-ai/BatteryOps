from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import batteryops
import batteryops.dashboard as dashboard_module
import batteryops.reports.demo as demo_module
from batteryops.dashboard import build_similar_cases_frame, load_dashboard_data
from batteryops.reports.demo import DEMO_ARTIFACT_FILENAMES
from batteryops.streamlit_app import load_demo_payload


def _write_demo_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True)
    for filename in (
        DEMO_ARTIFACT_FILENAMES["rul_model"],
        DEMO_ARTIFACT_FILENAMES["anomaly_model"],
        DEMO_ARTIFACT_FILENAMES["incident_retrieval"],
    ):
        (bundle_dir / filename).write_bytes(b"placeholder")

    timeline = pd.DataFrame(
        [
            {
                "asset_id": "battery00",
                "cycle_id": 10,
                "capacity_ah": 2.00,
                "internal_resistance_ohm": 0.050,
                "anomaly_score": 0.18,
                "status": "monitor",
                "predicted_rul_cycles": 14.0,
            },
            {
                "asset_id": "battery00",
                "cycle_id": 11,
                "capacity_ah": 1.96,
                "internal_resistance_ohm": 0.051,
                "anomaly_score": 0.32,
                "status": "monitor",
                "predicted_rul_cycles": 12.0,
            },
            {
                "asset_id": "battery00",
                "cycle_id": 12,
                "capacity_ah": 1.92,
                "internal_resistance_ohm": 0.053,
                "anomaly_score": 0.91,
                "status": "inspect soon",
                "predicted_rul_cycles": 10.0,
            },
        ]
    )
    timeline.to_parquet(bundle_dir / DEMO_ARTIFACT_FILENAMES["demo_cycle_predictions"], index=False)

    incident_cases = pd.DataFrame(
        [
            {
                "window_id": "battery00-12-1",
                "asset_id": "battery00",
                "cycle_id": 12,
                "duration_s": 320.0,
                "sample_count": 21,
                "incident_types": "high_temperature,low_voltage",
                "max_temperature_c": 48.0,
                "min_voltage_v": 3.65,
                "max_abs_current_a": 5.4,
                "severity_score": 2.1,
            }
        ]
    )
    incident_cases.to_parquet(
        bundle_dir / DEMO_ARTIFACT_FILENAMES["demo_incident_cases"],
        index=False,
    )

    metrics = {
        "data_source": "processed",
        "asset_count": 1,
        "cycle_count": 3,
        "incident_case_count": 1,
        "evidence_source_coverage": 1.0,
        "report_grounding_coverage": 1.0,
        "alert_precision": 0.75,
        "alert_recall": 0.5,
        "rul_proxy_mae": 3.2,
        "rul_mae": 3.2,
    }
    (bundle_dir / DEMO_ARTIFACT_FILENAMES["demo_metrics"]).write_text(json.dumps(metrics))

    report = {
        "report_id": "battery00-12-incident",
        "asset_id": "battery00",
        "cycle_id": 12,
        "incident_types": ["high_temperature", "low_voltage"],
        "summary": "Saved incident context for the demo bundle.",
        "evidence": ["capacity drift", "temperature spike"],
        "similar_cases": [
            {
                "window_id": "battery01-11-1",
                "asset_id": "battery01",
                "cycle_id": 11,
                "incident_types": "high_temperature,low_voltage",
                "distance": 0.18,
                "severity_score": 2.4,
            }
        ],
        "recommended_tests": ["Inspect cooling path", "Check terminal voltage stability"],
        "confidence_score": 0.86,
    }
    (bundle_dir / DEMO_ARTIFACT_FILENAMES["demo_report"]).write_text(json.dumps(report))

    manifest = {
        "artifacts": {
            name: f"artifacts/demo/{filename}"
            for name, filename in DEMO_ARTIFACT_FILENAMES.items()
        },
        "metrics": metrics,
        "report_id": report["report_id"],
    }
    (bundle_dir / DEMO_ARTIFACT_FILENAMES["training_manifest"]).write_text(json.dumps(manifest))


def test_package_and_dashboard_payload_smoke_uses_hermetic_demo_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "demo_bundle"
    _write_demo_bundle(bundle_dir)
    monkeypatch.setattr(
        demo_module,
        "ARTIFACT_DIR_CANDIDATES",
        (bundle_dir,),
    )
    monkeypatch.setattr(
        dashboard_module,
        "PROCESSED_EXPECTED",
        (
            tmp_path / "telemetry_samples.parquet",
            tmp_path / "cycle_features.parquet",
            tmp_path / "incident_windows.parquet",
        ),
    )

    status = demo_module.inspect_demo_bundle()
    summary, timeline = load_demo_payload()
    summary_again, timeline_again = load_demo_payload()
    data = load_dashboard_data()
    similar_cases = build_similar_cases_frame(data.report)

    assert status.healthy is True
    assert status.artifact_dir == bundle_dir
    assert batteryops.__version__ == "0.1.0"
    assert summary["asset_id"] == data.focus_asset_id
    assert int(summary["latest_cycle"]) == int(timeline["cycle"].max())
    assert int(summary["estimated_rul_cycles"]) >= 0
    assert float(data.anomaly_threshold) > 0.0
    assert data.metrics["data_source"] == "processed"
    assert "Processed NASA" in data.dataset_status.source_label
    assert "demo bundle" in data.dataset_status.source_label
    assert data.dataset_status.runtime_source_label == "checked-in demo bundle"
    assert data.dataset_status.full_dataset_available is False
    assert "regenerated from processed NASA data" in data.dataset_status.detail
    assert "not present in this checkout" in data.dataset_status.detail
    assert summary == summary_again
    pd.testing.assert_frame_equal(timeline, timeline_again)
    assert timeline["asset_id"].nunique() == 1
    assert {
        "capacity_ah",
        "internal_resistance_ohm",
        "anomaly_score",
        "predicted_rul_cycles",
        "predicted_rul_lower_cycles",
        "predicted_rul_upper_cycles",
        "status",
    }.issubset(timeline.columns)
    assert not similar_cases.empty
    assert {"case_label", "distance", "severity_score", "type_overlap"}.issubset(
        similar_cases.columns
    )
    assert (timeline["predicted_rul_lower_cycles"] <= timeline["predicted_rul_cycles"]).all()
    assert (timeline["predicted_rul_cycles"] <= timeline["predicted_rul_upper_cycles"]).all()
