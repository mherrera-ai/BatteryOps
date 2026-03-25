from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from batteryops.provenance import (
    bundle_fingerprint,
    bundle_inventory,
    hash_file,
    runtime_environment_snapshot,
)

CHECKOUT_ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "demo"
LOCAL_ARTIFACT_DIR = Path.cwd() / "artifacts" / "demo"
ARTIFACT_DIR_CANDIDATES = (LOCAL_ARTIFACT_DIR, CHECKOUT_ARTIFACT_DIR)
DEMO_ARTIFACT_FILENAMES = {
    "training_manifest": "training_manifest.json",
    "rul_model": "rul_model.joblib",
    "anomaly_model": "anomaly_model.joblib",
    "incident_retrieval": "incident_retrieval.joblib",
    "demo_cycle_predictions": "demo_cycle_predictions.parquet",
    "demo_incident_cases": "demo_incident_cases.parquet",
    "demo_metrics": "demo_metrics.json",
    "demo_report": "demo_report.json",
}
TIMELINE_REQUIRED_COLUMNS = {
    "asset_id",
    "cycle_id",
    "capacity_ah",
    "internal_resistance_ohm",
    "anomaly_score",
    "status",
    "predicted_rul_cycles",
}
INCIDENT_REQUIRED_COLUMNS = {
    "asset_id",
    "cycle_id",
    "window_id",
    "duration_s",
    "sample_count",
    "incident_types",
    "max_temperature_c",
    "min_voltage_v",
    "max_abs_current_a",
    "severity_score",
}
METRICS_REQUIRED_KEYS = {
    "data_source",
    "asset_count",
    "cycle_count",
    "incident_case_count",
}
PROVENANCE_OPTIONAL_RUNTIME_KEYS = {
    "batteryops",
    "python",
    "platform",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
}


@dataclass(frozen=True)
class DemoBundleStatus:
    """Validation outcome for the checked-in demo artifact bundle."""

    artifact_dir: Path | None
    healthy: bool
    reason: str
    missing_files: tuple[str, ...]
    metrics: dict[str, Any] | None = None
    bundle_fingerprint: str | None = None
    artifact_inventory: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] | None = None


def resolve_demo_artifact_path(filename: str) -> Path:
    """Resolve a demo artifact path from the first complete local bundle."""
    bundle_dir = resolve_demo_bundle_dir()
    if bundle_dir is not None:
        candidate = bundle_dir / filename
        if candidate.exists():
            return candidate

    for artifact_dir in ARTIFACT_DIR_CANDIDATES:
        candidate = artifact_dir / filename
        if candidate.exists():
            return candidate
    return ARTIFACT_DIR_CANDIDATES[0] / filename


def demo_artifacts_present() -> bool:
    """Return True when a complete demo bundle is available locally."""
    return resolve_demo_bundle_dir() is not None


def inspect_demo_bundle() -> DemoBundleStatus:
    """Inspect the available demo bundles and report the first healthy one."""
    first_failure: DemoBundleStatus | None = None
    for artifact_dir in ARTIFACT_DIR_CANDIDATES:
        status = _inspect_bundle_dir(artifact_dir)
        if status.healthy:
            return status
        if first_failure is None:
            first_failure = status

    return first_failure or DemoBundleStatus(
        artifact_dir=None,
        healthy=False,
        reason="No demo artifact bundle was found.",
        missing_files=tuple(DEMO_ARTIFACT_FILENAMES.values()),
    )


def resolve_demo_bundle_dir() -> Path | None:
    """Return the first local artifact directory that passes bundle validation."""
    status = inspect_demo_bundle()
    return status.artifact_dir if status.healthy else None


def build_demo_timeline() -> pd.DataFrame:
    """Create the demo timeline, preferring a validated saved bundle when present."""
    artifact_timeline = _load_artifact_timeline()
    if artifact_timeline is not None:
        return artifact_timeline

    cycles = np.arange(1, 81)
    capacity = 2.05 - (0.0055 * cycles) + (0.025 * np.sin(cycles / 6))
    internal_resistance = 0.045 + (0.00045 * cycles) + (0.001 * np.cos(cycles / 8))
    anomaly_score = (internal_resistance - internal_resistance.min()) * 40 + (
        capacity.max() - capacity
    ) * 6
    status = np.where(anomaly_score >= 0.9, "inspect soon", "monitor")

    return pd.DataFrame(
        {
            "cycle": cycles,
            "capacity_ah": np.round(capacity, 4),
            "internal_resistance_ohm": np.round(internal_resistance, 5),
            "anomaly_score": np.round(anomaly_score, 4),
            "status": status,
        }
    )


def build_demo_report() -> dict[str, object] | None:
    """Load the saved deterministic incident report if it exists."""
    bundle_dir = resolve_demo_bundle_dir()
    if bundle_dir is None:
        return None

    report_path = bundle_dir / DEMO_ARTIFACT_FILENAMES["demo_report"]
    if not report_path.exists():
        return None

    report = _load_json_file(report_path)
    if report is None:
        return None
    if not report.get("report_id") or not report.get("asset_id"):
        return None
    if "cycle_id" not in report:
        return None
    return report


def _load_artifact_timeline() -> pd.DataFrame | None:
    bundle_dir = resolve_demo_bundle_dir()
    if bundle_dir is None:
        return None

    timeline_path = bundle_dir / DEMO_ARTIFACT_FILENAMES["demo_cycle_predictions"]
    if not timeline_path.exists():
        return None

    frame = pd.read_parquet(timeline_path)
    if frame.empty:
        return None

    report = build_demo_report()
    required_columns = TIMELINE_REQUIRED_COLUMNS
    if not required_columns.issubset(frame.columns):
        return None

    if report is None:
        return None

    asset_id = str(report.get("asset_id", "")).strip()
    if not asset_id:
        return None

    asset_frame = frame.loc[frame["asset_id"] == asset_id].sort_values("cycle_id")
    if asset_frame.empty:
        return None

    return pd.DataFrame(
        {
            "cycle": asset_frame["cycle_id"].astype(int),
            "capacity_ah": asset_frame["capacity_ah"].astype(float),
            "internal_resistance_ohm": asset_frame["internal_resistance_ohm"].astype(float),
            "anomaly_score": asset_frame["anomaly_score"].astype(float),
            "status": asset_frame["status"].astype(str),
            "predicted_rul_cycles": asset_frame["predicted_rul_cycles"].astype(float),
        }
    )


def _inspect_bundle_dir(artifact_dir: Path) -> DemoBundleStatus:
    resolved_artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / DEMO_ARTIFACT_FILENAMES["training_manifest"]
    if not artifact_dir.exists():
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=f"{artifact_dir} does not exist.",
            missing_files=tuple(DEMO_ARTIFACT_FILENAMES.values()),
        )
    if not manifest_path.exists():
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=f"{manifest_path.name} is missing.",
            missing_files=(manifest_path.name,),
        )

    manifest = _load_json_file(manifest_path)
    if manifest is None:
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=f"{manifest_path.name} could not be parsed as JSON.",
            missing_files=(manifest_path.name,),
        )

    manifest_provenance = manifest.get("provenance")
    if manifest_provenance is not None and not isinstance(manifest_provenance, dict):
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="training_manifest.json includes an invalid provenance block.",
            missing_files=(manifest_path.name,),
        )
    if isinstance(manifest_provenance, dict):
        runtime_claim = manifest_provenance.get("runtime")
        if runtime_claim is not None and not isinstance(runtime_claim, dict):
            return DemoBundleStatus(
                artifact_dir=resolved_artifact_dir,
                healthy=False,
                reason="training_manifest.json includes an invalid runtime provenance block.",
                missing_files=(manifest_path.name,),
            )
        if isinstance(runtime_claim, dict):
            unsupported_runtime_keys = tuple(
                key for key in runtime_claim if key not in PROVENANCE_OPTIONAL_RUNTIME_KEYS
            )
            if unsupported_runtime_keys:
                return DemoBundleStatus(
                    artifact_dir=resolved_artifact_dir,
                    healthy=False,
                    reason=(
                        "training_manifest.json includes unsupported runtime provenance keys: "
                        + ", ".join(unsupported_runtime_keys)
                    ),
                    missing_files=(manifest_path.name,),
                )

    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict):
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="training_manifest.json does not include a valid artifact mapping.",
            missing_files=tuple(DEMO_ARTIFACT_FILENAMES.values()),
        )
    manifest_metrics = manifest.get("metrics")
    if not isinstance(manifest_metrics, dict) or not METRICS_REQUIRED_KEYS.issubset(
        manifest_metrics
    ):
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="training_manifest.json is missing required bundle provenance metrics.",
            missing_files=(manifest_path.name,),
        )

    expected_missing: list[str] = []
    provenance_mismatches: list[str] = []
    for manifest_key, filename in DEMO_ARTIFACT_FILENAMES.items():
        declared_record = manifest_artifacts.get(manifest_key)
        if isinstance(declared_record, dict):
            declared_path = declared_record.get("path")
        else:
            declared_path = declared_record
        if not isinstance(declared_path, str) or Path(declared_path).name != filename:
            expected_missing.append(filename)
            continue

        candidate_path = artifact_dir / filename
        if not candidate_path.exists():
            expected_missing.append(filename)
            continue

        if isinstance(declared_record, dict):
            expected_size = declared_record.get("size_bytes")
            if expected_size is not None and int(expected_size) != candidate_path.stat().st_size:
                provenance_mismatches.append(filename)
                continue
            expected_sha256 = declared_record.get("sha256")
            if expected_sha256 is not None and str(expected_sha256).strip().lower() != hash_file(
                candidate_path
            ):
                provenance_mismatches.append(filename)
                continue

    if expected_missing:
        unique_missing = tuple(dict.fromkeys(expected_missing))
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=(
                "The demo bundle is incomplete. Missing or mismatched artifacts: "
                + ", ".join(unique_missing)
            ),
            missing_files=unique_missing,
        )
    if provenance_mismatches:
        unique_mismatches = tuple(dict.fromkeys(provenance_mismatches))
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=(
                "training_manifest.json artifact provenance does not match: "
                + ", ".join(unique_mismatches)
            ),
            missing_files=unique_mismatches,
        )

    report = _load_json_file(artifact_dir / DEMO_ARTIFACT_FILENAMES["demo_report"])
    metrics = _load_json_file(artifact_dir / DEMO_ARTIFACT_FILENAMES["demo_metrics"])
    if report is None:
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="demo_report.json could not be parsed as JSON.",
            missing_files=(DEMO_ARTIFACT_FILENAMES["demo_report"],),
        )
    if metrics is None:
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="demo_metrics.json could not be parsed as JSON.",
            missing_files=(DEMO_ARTIFACT_FILENAMES["demo_metrics"],),
        )
    if not METRICS_REQUIRED_KEYS.issubset(metrics):
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="demo_metrics.json is missing required provenance fields.",
            missing_files=(DEMO_ARTIFACT_FILENAMES["demo_metrics"],),
        )
    mismatched_metric_keys = tuple(
        key for key in METRICS_REQUIRED_KEYS if metrics.get(key) != manifest_metrics.get(key)
    )
    if mismatched_metric_keys:
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=(
                "training_manifest.json and demo_metrics.json disagree for: "
                + ", ".join(mismatched_metric_keys)
            ),
            missing_files=(
                DEMO_ARTIFACT_FILENAMES["training_manifest"],
                DEMO_ARTIFACT_FILENAMES["demo_metrics"],
            ),
        )

    timeline_path = artifact_dir / DEMO_ARTIFACT_FILENAMES["demo_cycle_predictions"]
    try:
        timeline = pd.read_parquet(timeline_path)
    except Exception as exc:  # pragma: no cover - defensive against corrupt bundles
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=f"{timeline_path.name} could not be read as parquet: {exc!s}",
            missing_files=(timeline_path.name,),
        )
    if timeline.empty or not TIMELINE_REQUIRED_COLUMNS.issubset(timeline.columns):
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=(f"{timeline_path.name} is missing required timeline columns or is empty."),
            missing_files=(timeline_path.name,),
        )

    incident_path = artifact_dir / DEMO_ARTIFACT_FILENAMES["demo_incident_cases"]
    try:
        incident_cases = pd.read_parquet(incident_path)
    except Exception as exc:  # pragma: no cover - defensive against corrupt bundles
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=f"{incident_path.name} could not be read as parquet: {exc!s}",
            missing_files=(incident_path.name,),
        )
    if incident_cases.empty or not INCIDENT_REQUIRED_COLUMNS.issubset(incident_cases.columns):
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=(f"{incident_path.name} is missing required incident-case columns or is empty."),
            missing_files=(incident_path.name,),
        )

    if str(report.get("report_id", "")).strip() != str(manifest.get("report_id", "")).strip():
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="The report artifact does not match the manifest report_id.",
            missing_files=(DEMO_ARTIFACT_FILENAMES["demo_report"],),
        )

    if str(report.get("asset_id", "")).strip() not in set(timeline["asset_id"].astype(str)):
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="The report asset_id is not present in the demo timeline artifact.",
            missing_files=(DEMO_ARTIFACT_FILENAMES["demo_cycle_predictions"],),
        )
    report_cycle_rows = timeline.loc[
        (timeline["asset_id"].astype(str) == str(report.get("asset_id", "")).strip())
        & (pd.to_numeric(timeline["cycle_id"], errors="coerce") == int(report.get("cycle_id", -1)))
    ]
    if report_cycle_rows.empty:
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason="The report cycle_id is not present for the report asset in the demo timeline.",
            missing_files=(DEMO_ARTIFACT_FILENAMES["demo_cycle_predictions"],),
        )
    actual_metrics = {
        "asset_count": int(timeline["asset_id"].nunique()),
        "cycle_count": int(len(timeline)),
        "incident_case_count": int(len(incident_cases)),
    }
    stale_metric_keys = tuple(
        key
        for key, expected_value in actual_metrics.items()
        if int(metrics.get(key, -1)) != expected_value
    )
    if stale_metric_keys:
        return DemoBundleStatus(
            artifact_dir=resolved_artifact_dir,
            healthy=False,
            reason=("The demo bundle metrics are stale for: " + ", ".join(stale_metric_keys)),
            missing_files=(DEMO_ARTIFACT_FILENAMES["demo_metrics"],),
        )

    inventory = bundle_inventory(artifact_dir, tuple(DEMO_ARTIFACT_FILENAMES.values()))
    payload_inventory = tuple(
        item for item in inventory if item.name != DEMO_ARTIFACT_FILENAMES["training_manifest"]
    )
    fingerprint = bundle_fingerprint(payload_inventory)
    live_provenance = {
        "bundle_fingerprint": fingerprint,
        "runtime": runtime_environment_snapshot(),
        "artifacts": [
            {
                "name": item.name,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in inventory
        ],
    }

    if isinstance(manifest_provenance, dict):
        manifest_fingerprint = manifest_provenance.get("bundle_fingerprint")
        if isinstance(manifest_fingerprint, dict):
            if manifest_fingerprint.get("algorithm") not in (None, "sha256"):
                return DemoBundleStatus(
                    artifact_dir=resolved_artifact_dir,
                    healthy=False,
                    reason=(
                        "training_manifest.json declares an unsupported bundle "
                        "fingerprint algorithm."
                    ),
                    missing_files=(manifest_path.name,),
                )
            manifest_fingerprint = manifest_fingerprint.get("value")
        if manifest_fingerprint is not None and str(manifest_fingerprint).strip() != fingerprint:
            return DemoBundleStatus(
                artifact_dir=resolved_artifact_dir,
                healthy=False,
                reason=(
                    "training_manifest.json bundle fingerprint does not match the "
                    "checked-in artifacts."
                ),
                missing_files=(manifest_path.name,),
            )

        manifest_artifact_records = manifest_provenance.get("artifacts")
        if manifest_artifact_records is not None:
            if not isinstance(manifest_artifact_records, dict):
                return DemoBundleStatus(
                    artifact_dir=resolved_artifact_dir,
                    healthy=False,
                    reason="training_manifest.json includes an invalid artifact provenance block.",
                    missing_files=(manifest_path.name,),
                )
            manifest_provenance_mismatches: list[str] = []
            for manifest_key, filename in DEMO_ARTIFACT_FILENAMES.items():
                declared_record = manifest_artifact_records.get(manifest_key)
                if declared_record is None:
                    continue
                if not isinstance(declared_record, dict):
                    manifest_provenance_mismatches.append(filename)
                    continue
                matching = next((item for item in inventory if item.name == filename), None)
                if matching is None:
                    manifest_provenance_mismatches.append(filename)
                    continue
                declared_size = declared_record.get("size_bytes")
                if declared_size is not None and int(declared_size) != matching.size_bytes:
                    manifest_provenance_mismatches.append(filename)
                    continue
                declared_sha256 = declared_record.get("sha256")
                if (
                    declared_sha256 is not None
                    and str(declared_sha256).strip().lower() != matching.sha256
                ):
                    manifest_provenance_mismatches.append(filename)
            if manifest_provenance_mismatches:
                unique_provenance_mismatches = tuple(
                    dict.fromkeys(manifest_provenance_mismatches)
                )
                return DemoBundleStatus(
                    artifact_dir=resolved_artifact_dir,
                    healthy=False,
                    reason=(
                        "training_manifest.json artifact provenance does not match: "
                        + ", ".join(unique_provenance_mismatches)
                    ),
                    missing_files=unique_provenance_mismatches,
                )

    return DemoBundleStatus(
        artifact_dir=resolved_artifact_dir,
        healthy=True,
        reason="The demo bundle manifest and artifact set are complete.",
        missing_files=(),
        metrics=dict(manifest_metrics),
        bundle_fingerprint=fingerprint,
        artifact_inventory=tuple(
            {
                "name": item.name,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in inventory
        ),
        provenance=live_provenance,
    )


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded
