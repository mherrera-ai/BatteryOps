from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from batteryops.models import train_baselines
from batteryops.reports.demo import DEMO_ARTIFACT_FILENAMES

EXPECTED_ARTIFACT_KEYS = set(DEMO_ARTIFACT_FILENAMES)


def _normalize_artifact_record(record: object) -> object:
    if isinstance(record, dict):
        normalized = dict(record)
        normalized["path"] = Path(str(normalized["path"])).name
        return normalized
    return Path(str(record)).name


def _load_bundle_snapshot(artifact_paths: dict[str, Path]) -> dict[str, object]:
    return {
        "cycle_predictions": pd.read_parquet(artifact_paths["demo_cycle_predictions"]),
        "incident_cases": pd.read_parquet(artifact_paths["demo_incident_cases"]),
        "demo_metrics": json.loads(artifact_paths["demo_metrics"].read_text()),
        "demo_report": json.loads(artifact_paths["demo_report"].read_text()),
        "training_manifest": json.loads(artifact_paths["training_manifest"].read_text()),
    }


def _normalize_manifest(manifest: dict[str, object]) -> dict[str, object]:
    artifacts = manifest.get("artifacts", {})
    assert isinstance(artifacts, dict)
    return {
        "schema_version": manifest["schema_version"],
        "artifacts": {
            name: _normalize_artifact_record(record) for name, record in artifacts.items()
        },
        "metrics": manifest["metrics"],
        "report_id": manifest["report_id"],
        "provenance": manifest["provenance"],
    }


def test_train_baselines_uses_demo_fallback_when_processed_inputs_are_missing(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"

    result = train_baselines(
        processed_dir=processed_dir,
        artifact_dir=artifact_dir,
        allow_demo_fallback=True,
    )

    assert result.data_source == "demo_fallback"
    assert result.metrics["data_source"] == "demo_fallback"
    assert result.metrics["asset_count"] == 6
    assert result.metrics["cycle_count"] > 0
    assert {
        "rul_model",
        "anomaly_model",
        "incident_retrieval",
        "demo_cycle_predictions",
        "demo_incident_cases",
        "demo_metrics",
        "demo_report",
        "training_manifest",
    }.issubset(result.artifact_paths)
    assert all(path.exists() for path in result.artifact_paths.values())

    cycle_predictions = pd.read_parquet(result.artifact_paths["demo_cycle_predictions"])
    metrics = json.loads(result.artifact_paths["demo_metrics"].read_text())
    manifest = json.loads(result.artifact_paths["training_manifest"].read_text())

    assert set(cycle_predictions["asset_id"]) == {
        "DEMO-01",
        "DEMO-02",
        "DEMO-03",
        "DEMO-04",
        "DEMO-05",
        "DEMO-06",
    }
    assert cycle_predictions["predicted_alert"].dtype == bool
    assert metrics["data_source"] == "demo_fallback"
    assert manifest["metrics"]["data_source"] == "demo_fallback"
    assert manifest["report_id"].startswith("DEMO-")
    assert manifest["schema_version"] == 2
    assert manifest["provenance"]["training"]["deterministic"] is True
    assert manifest["provenance"]["inputs"]["data_source"] == "demo_fallback"
    assert manifest["provenance"]["bundle_fingerprint"]["algorithm"] == "sha256"
    assert manifest["provenance"]["runtime"]["batteryops"]


def test_train_baselines_emits_a_complete_demo_bundle_contract(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"

    result = train_baselines(
        processed_dir=processed_dir,
        artifact_dir=artifact_dir,
        allow_demo_fallback=True,
    )

    assert set(result.artifact_paths) == EXPECTED_ARTIFACT_KEYS
    assert all(path.exists() for path in result.artifact_paths.values())
    assert all(path.parent == artifact_dir for path in result.artifact_paths.values())

    manifest = json.loads(result.artifact_paths["training_manifest"].read_text())
    assert set(manifest["artifacts"]) == EXPECTED_ARTIFACT_KEYS
    assert manifest["schema_version"] == 2
    assert manifest["metrics"] == result.metrics
    assert (
        manifest["report_id"]
        == json.loads(result.artifact_paths["demo_report"].read_text())["report_id"]
    )
    assert manifest["artifacts"]["training_manifest"] == str(
        result.artifact_paths["training_manifest"]
    )
    for key, record in manifest["artifacts"].items():
        if key == "training_manifest":
            continue
        assert isinstance(record, dict)
        assert Path(str(record["path"])).parent == artifact_dir
        assert Path(str(record["path"])).name == DEMO_ARTIFACT_FILENAMES[key]
        assert int(record["size_bytes"]) > 0
        assert len(str(record["sha256"])) == 64

    provenance = manifest["provenance"]
    assert provenance["bundle_fingerprint"]["algorithm"] == "sha256"
    assert len(str(provenance["bundle_fingerprint"]["value"])) == 64
    assert provenance["inputs"]["cycle_features"]["row_count"] > 0
    assert provenance["inputs"]["incident_windows"]["row_count"] > 0
    assert provenance["training"]["feature_columns"]
    assert provenance["runtime"]["batteryops"]

    cycle_predictions = pd.read_parquet(result.artifact_paths["demo_cycle_predictions"])
    incident_cases = pd.read_parquet(result.artifact_paths["demo_incident_cases"])
    assert {
        "asset_id",
        "cycle_id",
        "predicted_rul_cycles",
        "predicted_alert",
        "actual_rul_cycles",
        "actual_alert",
    }.issubset(cycle_predictions.columns)
    assert {"window_id", "severity_score", "incident_types"}.issubset(incident_cases.columns)
    assert cycle_predictions["predicted_alert"].dtype == bool


def test_train_baselines_is_deterministic_for_demo_fallback(
    tmp_path: Path,
) -> None:
    first = train_baselines(
        processed_dir=tmp_path / "processed_1",
        artifact_dir=tmp_path / "artifacts_1",
        allow_demo_fallback=True,
    )
    second = train_baselines(
        processed_dir=tmp_path / "processed_2",
        artifact_dir=tmp_path / "artifacts_2",
        allow_demo_fallback=True,
    )

    assert first.data_source == second.data_source == "demo_fallback"
    assert first.metrics == second.metrics
    assert first.artifact_paths.keys() == second.artifact_paths.keys()

    first_snapshot = _load_bundle_snapshot(first.artifact_paths)
    second_snapshot = _load_bundle_snapshot(second.artifact_paths)

    pd.testing.assert_frame_equal(
        first_snapshot["cycle_predictions"],
        second_snapshot["cycle_predictions"],
    )
    pd.testing.assert_frame_equal(
        first_snapshot["incident_cases"],
        second_snapshot["incident_cases"],
    )
    assert first_snapshot["demo_metrics"] == second_snapshot["demo_metrics"]
    assert first_snapshot["demo_report"] == second_snapshot["demo_report"]
    assert _normalize_manifest(first_snapshot["training_manifest"]) == _normalize_manifest(
        second_snapshot["training_manifest"]
    )
