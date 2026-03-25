from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import batteryops.dashboard as dashboard_module
import batteryops.reports.demo as demo_module
from batteryops.reports.demo import DEMO_ARTIFACT_FILENAMES, DemoBundleStatus


def _missing_bundle_status(reason: str) -> DemoBundleStatus:
    return DemoBundleStatus(
        artifact_dir=None,
        healthy=False,
        reason=reason,
        missing_files=tuple(DEMO_ARTIFACT_FILENAMES.values()),
    )


def test_load_dashboard_data_uses_synthetic_fallback_when_demo_bundle_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle_status = _missing_bundle_status("training_manifest.json is missing.")
    monkeypatch.setattr(dashboard_module, "inspect_demo_bundle", lambda: bundle_status)
    monkeypatch.setattr(demo_module, "inspect_demo_bundle", lambda: bundle_status)
    monkeypatch.setattr(
        dashboard_module,
        "PROCESSED_EXPECTED",
        (
            tmp_path / "telemetry_samples.parquet",
            tmp_path / "cycle_features.parquet",
            tmp_path / "incident_windows.parquet",
        ),
    )

    data = dashboard_module.load_dashboard_data()

    assert data.metrics == {}
    assert data.dataset_status.source_label == "Synthetic demo fallback"
    assert data.dataset_status.runtime_source_label == "synthetic demo fallback"
    assert data.dataset_status.full_dataset_available is False
    assert "Processed NASA parquet was not found locally." in data.dataset_status.detail
    assert "No complete demo bundle was found" in data.dataset_status.detail
    assert "training_manifest.json is missing." in data.dataset_status.detail
    assert data.report["report_id"] == "demo-report-missing"
    assert data.summary["asset_id"] == "NASA-B0005-demo"
    assert "No valid demo report artifact was found" in data.summary["triage_note"]
    assert len(data.timeline) == 80
    assert data.timeline["asset_id"].nunique() == 1
    assert float(data.anomaly_threshold) > 0.0


def test_inspect_demo_bundle_rejects_stale_manifest_backed_bundle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "artifacts" / "demo"
    bundle_dir.mkdir(parents=True)
    for model_filename in ("rul_model.joblib", "anomaly_model.joblib", "incident_retrieval.joblib"):
        (bundle_dir / model_filename).write_bytes(b"placeholder")

    timeline = pd.DataFrame(
        [
            {
                "asset_id": "battery00",
                "cycle_id": 12,
                "capacity_ah": 1.98,
                "internal_resistance_ohm": 0.051,
                "anomaly_score": 0.87,
                "status": "monitor",
                "predicted_rul_cycles": 9.5,
            }
        ]
    )
    timeline.to_parquet(bundle_dir / "demo_cycle_predictions.parquet", index=False)

    incident_cases = pd.DataFrame(
        [
            {
                "window_id": "battery00-12-1",
                "asset_id": "battery00",
                "cycle_id": 12,
                "duration_s": 320.0,
                "sample_count": 21,
                "incident_types": "high_temperature",
                "max_temperature_c": 48.0,
                "min_voltage_v": 3.65,
                "max_abs_current_a": 5.4,
                "severity_score": 2.1,
            }
        ]
    )
    incident_cases.to_parquet(bundle_dir / "demo_incident_cases.parquet", index=False)

    metrics = {
        "data_source": "processed",
        "asset_count": 9,
        "cycle_count": 9,
        "incident_case_count": 9,
    }
    (bundle_dir / "demo_metrics.json").write_text(json.dumps(metrics))
    (bundle_dir / "demo_report.json").write_text(
        json.dumps(
            {
                "report_id": "battery00-12-incident",
                "asset_id": "battery00",
                "cycle_id": 12,
            }
        )
    )
    (bundle_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    name: f"artifacts/demo/{filename}"
                    for name, filename in DEMO_ARTIFACT_FILENAMES.items()
                },
                "metrics": metrics,
                "report_id": "battery00-12-incident",
            }
        )
    )

    monkeypatch.setattr(demo_module, "ARTIFACT_DIR_CANDIDATES", (bundle_dir,))

    status = demo_module.inspect_demo_bundle()

    assert status.healthy is False
    assert status.artifact_dir == bundle_dir
    assert status.reason == (
        "The demo bundle metrics are stale for: asset_count, cycle_count, incident_case_count"
    )
    assert status.missing_files == ("demo_metrics.json",)
