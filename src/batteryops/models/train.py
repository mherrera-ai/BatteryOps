from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingRegressor,
    IsolationForest,
)
from sklearn.model_selection import GroupKFold  # type: ignore[import-untyped]

from batteryops import __version__
from batteryops.eval import (
    alert_lead_time,
    alert_precision,
    alert_recall,
    false_positive_rate,
    report_grounding_coverage,
    rul_mae,
)
from batteryops.provenance import (
    bundle_fingerprint,
    canonical_json_digest,
    file_provenance,
    runtime_environment_snapshot,
)
from batteryops.reports import generate_incident_report
from batteryops.retrieval import (
    IncidentRetrievalBundle,
    fit_retrieval_index,
    retrieve_similar_cases,
)

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_ARTIFACT_DIR = Path("artifacts/demo")
PROCESSED_FILES = {
    "cycle_features": "cycle_features.parquet",
    "incident_windows": "incident_windows.parquet",
}
MODEL_FEATURE_COLUMNS = [
    "sample_count",
    "duration_s",
    "voltage_min_v",
    "voltage_max_v",
    "voltage_mean_v",
    "temperature_max_c",
    "temperature_mean_c",
    "current_abs_max_a",
    "throughput_ah",
    "net_current_ah",
    "capacity_ah",
    "internal_resistance_ohm",
    "incident_window_count",
    "incident_severity_max",
    "is_reference_cycle",
]
REPORT_COLUMNS = [
    "asset_id",
    "cycle_id",
    "incident_types",
    "duration_s",
    "sample_count",
    "max_temperature_c",
    "min_voltage_v",
    "max_abs_current_a",
    "severity_score",
]
TRAINING_MANIFEST_SCHEMA_VERSION = 2
TRAINING_RANDOM_SEED = 42
FINAL_ANOMALY_CONTAMINATION = 0.2
HOLDOUT_ANOMALY_CONTAMINATION = 0.15
FINAL_ALERT_THRESHOLD_QUANTILE = 0.8
HOLDOUT_ALERT_THRESHOLD_QUANTILE = 0.85


@dataclass
class RulModelArtifact:
    model: HistGradientBoostingRegressor
    feature_columns: list[str]
    fill_values: dict[str, float]


@dataclass
class AnomalyModelArtifact:
    model: IsolationForest
    feature_columns: list[str]
    fill_values: dict[str, float]
    threshold: float
    score_min: float
    score_max: float


@dataclass
class TrainingResult:
    artifact_paths: dict[str, Path]
    metrics: dict[str, float | int | str]
    data_source: str


def train_baselines(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    allow_demo_fallback: bool = True,
) -> TrainingResult:
    """Train baseline models and save demo-ready artifacts."""
    resolved_processed = Path(processed_dir)
    resolved_artifacts = Path(artifact_dir)
    resolved_artifacts.mkdir(parents=True, exist_ok=True)

    cycle_features, incident_windows, data_source = load_training_inputs(
        resolved_processed,
        allow_demo_fallback=allow_demo_fallback,
    )
    cycle_table = prepare_cycle_training_table(cycle_features, incident_windows)
    validation_table, evaluation_mode = evaluate_holdout_predictions(cycle_table)
    feature_matrix, fill_values = build_feature_matrix(cycle_table)

    rul_model = HistGradientBoostingRegressor(random_state=TRAINING_RANDOM_SEED)
    rul_model.fit(feature_matrix, cycle_table["actual_rul_cycles"])
    cycle_table["predicted_rul_cycles"] = np.maximum(rul_model.predict(feature_matrix), 0.0)

    anomaly_model = IsolationForest(
        random_state=TRAINING_RANDOM_SEED,
        n_estimators=200,
        contamination=FINAL_ANOMALY_CONTAMINATION,
    )
    anomaly_model.fit(feature_matrix)
    raw_scores = -anomaly_model.score_samples(feature_matrix)
    score_min = float(np.min(raw_scores))
    score_max = float(np.max(raw_scores))
    cycle_table["anomaly_score"] = normalize_scores(raw_scores, score_min, score_max)
    threshold = float(cycle_table["anomaly_score"].quantile(FINAL_ALERT_THRESHOLD_QUANTILE))
    cycle_table["predicted_alert"] = cycle_table["anomaly_score"] >= threshold
    cycle_table["status"] = np.where(cycle_table["predicted_alert"], "inspect soon", "monitor")

    retrieval_bundle = fit_retrieval_index(incident_windows)
    report = build_demo_report(incident_windows, retrieval_bundle, data_source)
    evidence_source_coverage = round(report_grounding_coverage(report), 4)

    metrics: dict[str, float | int | str] = {
        "data_source": data_source,
        "evaluation_mode": evaluation_mode,
        "asset_count": int(cycle_table["asset_id"].nunique()),
        "cycle_count": int(len(cycle_table)),
        "incident_case_count": int(len(incident_windows)),
        "alert_lead_time": round(alert_lead_time(validation_table), 4),
        "alert_precision": round(
            alert_precision(
                validation_table["actual_alert"],
                validation_table["predicted_alert"],
            ),
            4,
        ),
        "alert_recall": round(
            alert_recall(
                validation_table["actual_alert"],
                validation_table["predicted_alert"],
            ),
            4,
        ),
        "false_positive_rate": round(
            false_positive_rate(
                validation_table["actual_alert"],
                validation_table["predicted_alert"],
            ),
            4,
        ),
        "rul_proxy_mae": round(
            rul_mae(
                validation_table["actual_rul_cycles"],
                validation_table["predicted_rul_cycles"],
            ),
            4,
        ),
        "rul_mae": round(
            rul_mae(
                validation_table["actual_rul_cycles"],
                validation_table["predicted_rul_cycles"],
            ),
            4,
        ),
        "evidence_source_coverage": evidence_source_coverage,
        "report_grounding_coverage": evidence_source_coverage,
    }

    artifact_paths = save_artifacts(
        resolved_artifacts,
        resolved_processed,
        cycle_features,
        cycle_table,
        incident_windows,
        RulModelArtifact(rul_model, MODEL_FEATURE_COLUMNS, fill_values),
        AnomalyModelArtifact(
            anomaly_model,
            MODEL_FEATURE_COLUMNS,
            fill_values,
            threshold,
            score_min,
            score_max,
        ),
        retrieval_bundle,
        report,
        metrics,
        allow_demo_fallback=allow_demo_fallback,
    )
    return TrainingResult(
        artifact_paths=artifact_paths,
        metrics=metrics,
        data_source=data_source,
    )


def load_training_inputs(
    processed_dir: Path,
    allow_demo_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load processed parquet inputs or fall back to deterministic demo data."""
    cycle_path = processed_dir / PROCESSED_FILES["cycle_features"]
    incident_path = processed_dir / PROCESSED_FILES["incident_windows"]
    if cycle_path.exists() and incident_path.exists():
        cycle_features = pd.read_parquet(cycle_path)
        incident_windows = pd.read_parquet(incident_path)
        if not cycle_features.empty and not incident_windows.empty:
            return cycle_features, incident_windows, "processed"

    if not allow_demo_fallback:
        raise FileNotFoundError(
            "Processed parquet files were not available. Run batteryops.data.preprocess first."
        )

    return build_demo_training_inputs()


def build_demo_training_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Create deterministic training inputs for a fast local demo path."""
    cycle_rows: list[dict[str, object]] = []
    incident_rows: list[dict[str, object]] = []
    start_date = pd.Timestamp("2025-01-01")

    for asset_idx in range(6):
        asset_id = f"DEMO-{asset_idx + 1:02d}"
        asset_group = "demo_pack"
        base_capacity = 2.35 - (0.06 * asset_idx)
        base_resistance = 0.045 + (0.0015 * asset_idx)
        wear_rate = 0.012 + (0.001 * asset_idx)
        warning_cycle = 15 + asset_idx
        cycle_count = 28

        for cycle_id in range(1, cycle_count + 1):
            cycle_start = start_date + pd.Timedelta(days=(asset_idx * 40) + cycle_id)
            accelerated_wear = max(cycle_id - warning_cycle, 0)
            capacity_ah = max(
                base_capacity
                - (wear_rate * cycle_id)
                - (0.0045 * accelerated_wear**1.2)
                + (0.008 * np.sin((cycle_id + asset_idx) / 4)),
                0.85,
            )
            internal_resistance = (
                base_resistance
                + (0.0008 * cycle_id)
                + (0.00055 * accelerated_wear)
                + (0.0002 * np.cos(cycle_id / 3))
            )
            voltage_min = (
                3.95 - (0.016 * cycle_id) - (0.025 * accelerated_wear) - (0.01 * asset_idx)
            )
            voltage_max = 4.18 - (0.0015 * accelerated_wear)
            temperature_max = 26.0 + (0.22 * cycle_id) + (0.4 * asset_idx) + accelerated_wear
            temperature_mean = temperature_max - 2.3
            current_abs_max = (
                2.6 + (0.08 * ((cycle_id + asset_idx) % 4)) + (0.18 * accelerated_wear)
            )
            throughput = capacity_ah * (0.92 + (0.01 * (cycle_id % 3)))
            incident_types: list[str] = []
            if voltage_min < 3.22:
                incident_types.append("low_voltage")
            if temperature_max >= 40.0:
                incident_types.append("high_temperature")
            if current_abs_max >= 4.0:
                incident_types.append("high_current")

            severity = float(len(incident_types) + min(accelerated_wear / 2.0, 3.0))
            incident_count = 1 if incident_types else 0
            cycle_rows.append(
                {
                    "dataset_code": "demo",
                    "asset_id": asset_id,
                    "asset_group": asset_group,
                    "cycle_kind": "reference_cycle",
                    "cycle_id": cycle_id,
                    "cycle_start_time": cycle_start,
                    "step_comment": "reference discharge",
                    "sample_count": 240 + (cycle_id * 3),
                    "duration_s": 3600.0 + (cycle_id * 35.0),
                    "internal_resistance_ohm": internal_resistance,
                    "voltage_min_v": voltage_min,
                    "voltage_max_v": voltage_max,
                    "voltage_mean_v": (voltage_min + voltage_max) / 2.0,
                    "temperature_max_c": temperature_max,
                    "temperature_mean_c": temperature_mean,
                    "current_abs_max_a": current_abs_max,
                    "throughput_ah": throughput,
                    "net_current_ah": -throughput * 0.93,
                    "discharge_capacity_proxy_ah": capacity_ah,
                    "charge_sample_count": 110,
                    "discharge_sample_count": 110,
                    "rest_sample_count": 20,
                    "incident_sample_count": incident_count * 8,
                    "is_reference_cycle": cycle_id % 3 == 0,
                }
            )

            if not incident_types:
                continue

            duration_s = 120.0 + (accelerated_wear * 35.0)
            incident_rows.append(
                {
                    "dataset_code": "demo",
                    "asset_id": asset_id,
                    "cycle_id": cycle_id,
                    "window_id": f"{asset_id}-{cycle_id}-1",
                    "cycle_start_time": cycle_start,
                    "window_start_time": cycle_start + pd.Timedelta(minutes=25),
                    "window_end_time": cycle_start + pd.Timedelta(minutes=25, seconds=duration_s),
                    "window_start_sample_s": 1500.0,
                    "window_end_sample_s": 1500.0 + duration_s,
                    "duration_s": duration_s,
                    "sample_count": 7 + accelerated_wear,
                    "incident_types": ",".join(incident_types),
                    "max_temperature_c": temperature_max,
                    "min_voltage_v": voltage_min,
                    "max_abs_current_a": current_abs_max,
                    "severity_score": severity,
                }
            )

    return pd.DataFrame(cycle_rows), pd.DataFrame(incident_rows), "demo_fallback"


def prepare_cycle_training_table(
    cycle_features: pd.DataFrame,
    incident_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cycle features with incident counts and degradation proxy targets."""
    incident_cycle = (
        incident_windows.groupby(["asset_id", "cycle_id"], as_index=False)
        .agg(
            incident_window_count=("window_id", "count"),
            incident_severity_max=("severity_score", "max"),
        )
        .reset_index(drop=True)
    )

    cycle_table = cycle_features.merge(
        incident_cycle,
        on=["asset_id", "cycle_id"],
        how="left",
    )
    cycle_table["incident_window_count"] = (
        cycle_table["incident_window_count"].fillna(0).astype(int)
    )
    cycle_table["incident_severity_max"] = cycle_table["incident_severity_max"].fillna(0.0)
    cycle_table["actual_alert"] = cycle_table["incident_window_count"] > 0

    asset_cycle_count = cycle_table.groupby("asset_id")["cycle_id"].transform("max")
    cycle_table["is_reference_cycle"] = cycle_table["is_reference_cycle"].astype(int)
    cycle_table["capacity_ah"] = cycle_table["discharge_capacity_proxy_ah"].fillna(
        cycle_table["throughput_ah"]
    )
    resistance_proxy = (cycle_table["voltage_max_v"] - cycle_table["voltage_min_v"]) / cycle_table[
        "current_abs_max_a"
    ].replace(0, np.nan)
    if "internal_resistance_ohm" in cycle_table:
        cycle_table["internal_resistance_ohm"] = cycle_table["internal_resistance_ohm"].fillna(
            resistance_proxy
        )
    else:
        cycle_table["internal_resistance_ohm"] = resistance_proxy
    cycle_table["internal_resistance_ohm"] = cycle_table["internal_resistance_ohm"].fillna(
        cycle_table["internal_resistance_ohm"].median()
    )
    failure_cycle = {
        str(asset_id): _resolve_failure_cycle_proxy(frame)
        for asset_id, frame in cycle_table.groupby("asset_id", sort=False)
    }
    cycle_table["failure_cycle"] = (
        cycle_table["asset_id"].map(failure_cycle).fillna(asset_cycle_count)
    )
    cycle_table["actual_rul_cycles"] = (
        pd.to_numeric(cycle_table["failure_cycle"], errors="coerce") - cycle_table["cycle_id"]
    ).clip(lower=0)
    return cycle_table.sort_values(["asset_id", "cycle_id"]).reset_index(drop=True)


def build_feature_matrix(
    cycle_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Prepare filled numeric model features."""
    feature_frame = cycle_table[MODEL_FEATURE_COLUMNS].copy()
    fill_values = {
        column: float(feature_frame[column].median())
        if not feature_frame[column].dropna().empty
        else 0.0
        for column in feature_frame.columns
    }
    return feature_frame.fillna(fill_values), fill_values


def evaluate_holdout_predictions(
    cycle_table: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Evaluate the baselines on held-out assets when possible."""
    feature_matrix, _ = build_feature_matrix(cycle_table)
    evaluated = cycle_table[["asset_id", "cycle_id", "actual_alert", "actual_rul_cycles"]].copy()
    evaluated["predicted_rul_cycles"] = 0.0
    evaluated["predicted_alert"] = False
    evaluated["anomaly_score"] = 0.0

    groups = cycle_table["asset_id"].astype(str)
    if groups.nunique() < 2:
        fitted = _fit_models_for_split(
            feature_matrix,
            cycle_table["actual_rul_cycles"],
            feature_matrix,
        )
        evaluated["predicted_rul_cycles"] = fitted["predicted_rul_cycles"]
        evaluated["predicted_alert"] = fitted["predicted_alert"]
        evaluated["anomaly_score"] = fitted["anomaly_score"]
        return evaluated, "single-asset in-sample degradation proxy"

    splitter = GroupKFold(n_splits=groups.nunique())
    for train_index, test_index in splitter.split(feature_matrix, groups=groups):
        split_predictions = _fit_models_for_split(
            feature_matrix.iloc[train_index],
            cycle_table["actual_rul_cycles"].iloc[train_index],
            feature_matrix.iloc[test_index],
        )
        test_labels = evaluated.index[test_index]
        evaluated.loc[test_labels, "predicted_rul_cycles"] = split_predictions[
            "predicted_rul_cycles"
        ]
        evaluated.loc[test_labels, "predicted_alert"] = split_predictions["predicted_alert"]
        evaluated.loc[test_labels, "anomaly_score"] = split_predictions["anomaly_score"]

    return evaluated, "leave-one-asset-out degradation proxy"


def _fit_models_for_split(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    rul_model = HistGradientBoostingRegressor(random_state=TRAINING_RANDOM_SEED)
    rul_model.fit(train_features, train_target)
    predicted_rul = np.maximum(rul_model.predict(test_features), 0.0)

    anomaly_model = IsolationForest(
        random_state=TRAINING_RANDOM_SEED,
        n_estimators=200,
        contamination=HOLDOUT_ANOMALY_CONTAMINATION,
    )
    anomaly_model.fit(train_features)
    train_scores = -anomaly_model.score_samples(train_features)
    test_scores = -anomaly_model.score_samples(test_features)
    score_min = float(np.min(train_scores))
    score_max = float(np.max(train_scores))
    normalized_train = normalize_scores(train_scores, score_min, score_max)
    normalized_test = normalize_scores(test_scores, score_min, score_max).clip(0.0, 1.0)
    threshold = float(np.quantile(normalized_train, HOLDOUT_ALERT_THRESHOLD_QUANTILE))
    return {
        "predicted_rul_cycles": predicted_rul,
        "predicted_alert": normalized_test >= threshold,
        "anomaly_score": normalized_test,
    }


def normalize_scores(raw_scores: np.ndarray, score_min: float, score_max: float) -> np.ndarray:
    """Normalize anomaly scores into a 0..1 band."""
    if np.isclose(score_max, score_min):
        return np.zeros_like(raw_scores, dtype=float)
    return (raw_scores - score_min) / (score_max - score_min)


def _resolve_failure_cycle_proxy(asset_cycles: pd.DataFrame) -> int:
    ordered = asset_cycles.sort_values("cycle_id").reset_index(drop=True)
    if ordered.empty:
        return 0

    baseline_window = min(5, len(ordered))
    base_capacity = float(ordered["capacity_ah"].head(baseline_window).median())
    base_resistance = float(ordered["internal_resistance_ohm"].head(baseline_window).median())
    degraded_capacity = ordered["capacity_ah"] <= (base_capacity * 0.8)
    degraded_resistance = ordered["internal_resistance_ohm"] >= (base_resistance * 1.5)
    degraded = degraded_capacity | degraded_resistance

    for index in range(len(ordered) - 1):
        if bool(degraded.iloc[index]) and bool(degraded.iloc[index + 1]):
            return int(ordered.iloc[index]["cycle_id"])
    return int(ordered["cycle_id"].max())


def build_demo_report(
    incident_windows: pd.DataFrame,
    retrieval_bundle: IncidentRetrievalBundle,
    data_source: str,
) -> dict[str, object]:
    """Create a representative report from a well-supported incident case."""
    top_incident = _select_report_incident(incident_windows)
    similar_cases = retrieve_similar_cases(
        retrieval_bundle,
        top_incident,
        top_k=3,
        exclude_window_id=str(top_incident["window_id"]),
    )
    return generate_incident_report(top_incident[REPORT_COLUMNS], similar_cases, data_source)


def _select_report_incident(incident_windows: pd.DataFrame) -> pd.Series:
    scored = incident_windows.copy()
    contradiction_penalty = (
        scored["incident_types"]
        .astype(str)
        .apply(lambda value: int("low_voltage" in value and "high_voltage" in value))
    )
    scored["report_priority"] = (
        pd.to_numeric(scored["severity_score"], errors="coerce").fillna(0.0)
        + pd.to_numeric(scored["max_temperature_c"], errors="coerce").fillna(0.0) / 20.0
        + pd.to_numeric(scored["max_abs_current_a"], errors="coerce").fillna(0.0) / 10.0
        - contradiction_penalty * 2.0
    )
    return scored.sort_values(["report_priority", "cycle_id"]).iloc[-1]


def _dataframe_fingerprint(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized[column] = series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        elif pd.api.types.is_timedelta64_dtype(series):
            normalized[column] = series.dt.total_seconds()

    payload = json.loads(
        normalized.to_json(
            orient="split",
            index=False,
            date_format="iso",
            date_unit="ns",
            double_precision=15,
        )
    )
    return canonical_json_digest(payload)


def _build_input_provenance(
    name: str,
    frame: pd.DataFrame,
    *,
    source_path: Path | None,
) -> dict[str, object]:
    input_provenance: dict[str, object] = {
        "filename": source_path.name if source_path is not None else f"{name}.generated",
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "dataframe_sha256": _dataframe_fingerprint(frame),
    }
    if source_path is not None and source_path.exists():
        file_metadata = file_provenance(source_path)
        input_provenance.update(
            {
                "size_bytes": file_metadata.size_bytes,
                "sha256": file_metadata.sha256,
            }
        )
    return input_provenance


def _build_training_provenance(
    processed_dir: Path,
    cycle_features: pd.DataFrame,
    incident_windows: pd.DataFrame,
    metrics: dict[str, float | int | str],
    *,
    allow_demo_fallback: bool,
) -> dict[str, object]:
    data_source = str(metrics["data_source"])
    cycle_source_path = (
        processed_dir / PROCESSED_FILES["cycle_features"] if data_source == "processed" else None
    )
    incident_source_path = (
        processed_dir / PROCESSED_FILES["incident_windows"] if data_source == "processed" else None
    )
    return {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "generated_by": f"batteryops {__version__}",
        "runtime": runtime_environment_snapshot(),
        "inputs": {
            "data_source": data_source,
            "allow_demo_fallback": allow_demo_fallback,
            "cycle_features": _build_input_provenance(
                "cycle_features",
                cycle_features,
                source_path=cycle_source_path,
            ),
            "incident_windows": _build_input_provenance(
                "incident_windows",
                incident_windows,
                source_path=incident_source_path,
            ),
        },
        "training": {
            "deterministic": True,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "random_seed": TRAINING_RANDOM_SEED,
            "evaluation_mode": str(metrics["evaluation_mode"]),
            "rul_model": {
                "estimator": "HistGradientBoostingRegressor",
                "random_state": TRAINING_RANDOM_SEED,
            },
            "anomaly_model": {
                "estimator": "IsolationForest",
                "random_state": TRAINING_RANDOM_SEED,
                "n_estimators": 200,
                "final_fit_contamination": FINAL_ANOMALY_CONTAMINATION,
                "holdout_fit_contamination": HOLDOUT_ANOMALY_CONTAMINATION,
                "final_threshold_quantile": FINAL_ALERT_THRESHOLD_QUANTILE,
                "holdout_threshold_quantile": HOLDOUT_ALERT_THRESHOLD_QUANTILE,
            },
        },
    }


def save_artifacts(
    artifact_dir: Path,
    processed_dir: Path,
    cycle_features: pd.DataFrame,
    cycle_table: pd.DataFrame,
    incident_windows: pd.DataFrame,
    rul_artifact: RulModelArtifact,
    anomaly_artifact: AnomalyModelArtifact,
    retrieval_bundle: IncidentRetrievalBundle,
    report: dict[str, object],
    metrics: dict[str, float | int | str],
    *,
    allow_demo_fallback: bool,
) -> dict[str, Path]:
    """Persist models, retrieval index, metrics, and demo-ready tables."""
    paths = {
        "rul_model": artifact_dir / "rul_model.joblib",
        "anomaly_model": artifact_dir / "anomaly_model.joblib",
        "incident_retrieval": artifact_dir / "incident_retrieval.joblib",
        "demo_cycle_predictions": artifact_dir / "demo_cycle_predictions.parquet",
        "demo_incident_cases": artifact_dir / "demo_incident_cases.parquet",
        "demo_metrics": artifact_dir / "demo_metrics.json",
        "demo_report": artifact_dir / "demo_report.json",
        "training_manifest": artifact_dir / "training_manifest.json",
    }

    joblib.dump(rul_artifact, paths["rul_model"])
    joblib.dump(anomaly_artifact, paths["anomaly_model"])
    joblib.dump(retrieval_bundle, paths["incident_retrieval"])

    cycle_columns = [
        "asset_id",
        "cycle_id",
        "cycle_start_time",
        "capacity_ah",
        "internal_resistance_ohm",
        "anomaly_score",
        "predicted_alert",
        "status",
        "predicted_rul_cycles",
        "actual_rul_cycles",
        "actual_alert",
    ]
    cycle_table[cycle_columns].to_parquet(paths["demo_cycle_predictions"], index=False)
    incident_windows.to_parquet(paths["demo_incident_cases"], index=False)
    paths["demo_metrics"].write_text(json.dumps(metrics, indent=2))
    paths["demo_report"].write_text(json.dumps(report, indent=2))

    artifact_records: dict[str, str | dict[str, object]] = {
        "training_manifest": str(paths["training_manifest"]),
    }
    payload_inventory = []
    for name, path in paths.items():
        if name == "training_manifest":
            continue
        file_metadata = file_provenance(path)
        payload_inventory.append(file_metadata)
        artifact_records[name] = {
            "path": str(path),
            "size_bytes": file_metadata.size_bytes,
            "sha256": file_metadata.sha256,
        }

    training_provenance = _build_training_provenance(
        processed_dir,
        cycle_features,
        incident_windows,
        metrics,
        allow_demo_fallback=allow_demo_fallback,
    )
    training_provenance["bundle_fingerprint"] = {
        "algorithm": "sha256",
        "value": bundle_fingerprint(tuple(payload_inventory)),
    }

    manifest = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "artifacts": artifact_records,
        "metrics": metrics,
        "report_id": report["report_id"],
        "provenance": training_provenance,
    }
    paths["training_manifest"].write_text(json.dumps(manifest, indent=2))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BatteryOps baseline models.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing cycle_features.parquet and incident_windows.parquet.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory where demo artifacts will be written.",
    )
    parser.add_argument(
        "--no-demo-fallback",
        action="store_true",
        help=(
            "Fail instead of training from deterministic demo data when processed "
            "parquet is missing."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_baselines(
        processed_dir=args.processed_dir,
        artifact_dir=args.artifact_dir,
        allow_demo_fallback=not args.no_demo_fallback,
    )
    print(json.dumps({"metrics": result.metrics}, indent=2))
    for name, path in result.artifact_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
