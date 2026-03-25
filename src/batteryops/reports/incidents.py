from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from batteryops.eval.metrics import report_grounding_coverage


def generate_incident_report(
    incident: pd.Series | dict[str, object],
    similar_cases: pd.DataFrame,
    data_source: str,
) -> dict[str, object]:
    """Create a deterministic report from incident telemetry and retrieved cases."""
    incident_row = (
        incident if isinstance(incident, pd.Series) else pd.Series(incident, dtype=object)
    )
    incident_types = _incident_types(incident_row.get("incident_types", ""))
    evidence = _build_evidence(incident_row)
    similar_case_items = _format_similar_cases(similar_cases)
    diagnostics = _recommended_tests(incident_types)
    caveats = _caveats(data_source, similar_cases)

    report = {
        "report_id": (
            f"{incident_row.get('asset_id', 'unknown')}-"
            f"{incident_row.get('cycle_id', 'unknown')}-incident"
        ),
        "asset_id": str(incident_row.get("asset_id", "unknown")),
        "cycle_id": int(incident_row.get("cycle_id", 0)),
        "incident_types": incident_types,
        "summary": _summary(incident_row, incident_types),
        "evidence": evidence,
        "similar_cases": similar_case_items,
        "recommended_tests": diagnostics,
        "confidence_score": _confidence_score(evidence, similar_cases, data_source),
        "confidence_caveats": caveats,
    }
    report["grounding_coverage"] = report_grounding_coverage(report)
    return report


def _summary(incident_row: pd.Series, incident_types: list[str]) -> str:
    labels = ", ".join(incident_types) if incident_types else "an unspecified incident"
    duration_s = float(incident_row.get("duration_s", 0.0) or 0.0)
    severity = float(incident_row.get("severity_score", 0.0) or 0.0)
    asset_id = incident_row.get("asset_id", "unknown")
    cycle_id = int(incident_row.get("cycle_id", 0) or 0)
    return (
        f"Asset {asset_id} showed {labels} during cycle {cycle_id}. "
        f"The observed window lasted {duration_s:.0f} seconds with severity {severity:.2f}."
    )


def _build_evidence(incident_row: pd.Series) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for field, template in (
        (
            "min_voltage_v",
            "Minimum observed voltage reached {value:.2f} V during the incident window.",
        ),
        (
            "max_temperature_c",
            "Peak observed temperature reached {value:.2f} C during the incident window.",
        ),
        (
            "max_abs_current_a",
            "Peak absolute current reached {value:.2f} A during the incident window.",
        ),
        (
            "duration_s",
            "The incident window persisted for {value:.0f} seconds.",
        ),
        (
            "sample_count",
            "The incident window contains {value:.0f} recorded samples.",
        ),
    ):
        value = incident_row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        numeric_value = float(value)
        evidence.append(
            {
                "text": template.format(value=numeric_value),
                "source_field": field,
                "value": numeric_value,
                "grounded": True,
            }
        )

    return evidence


def _format_similar_cases(similar_cases: pd.DataFrame) -> list[dict[str, object]]:
    if similar_cases.empty:
        return []

    items: list[dict[str, object]] = []
    for _, row in similar_cases.iterrows():
        items.append(
            {
                "window_id": str(row.get("window_id", "")),
                "asset_id": str(row.get("asset_id", "")),
                "cycle_id": int(row.get("cycle_id", 0)),
                "incident_types": _incident_types(row.get("incident_types", "")),
                "severity_score": float(row.get("severity_score", 0.0)),
                "distance": float(row.get("distance", 0.0)),
                "summary": (
                    f"Asset {row.get('asset_id', '')} cycle {int(row.get('cycle_id', 0))} "
                    f"showed {row.get('incident_types', 'unknown')} "
                    f"with retrieval distance {float(row.get('distance', 0.0)):.2f}."
                ),
            }
        )

    return items


def _recommended_tests(incident_types: Iterable[str]) -> list[str]:
    incident_set = set(incident_types)
    tests: list[str] = ["Repeat a reference charge-discharge cycle to benchmark capacity drift."]
    if "low_voltage" in incident_set:
        tests.append("Run a controlled low-current discharge to confirm cutoff behavior.")
    if "high_temperature" in incident_set:
        tests.append("Inspect thermistor placement and repeat the cycle with thermal logging.")
    if "high_current" in incident_set:
        tests.append("Verify current sensing calibration and repeat a stepped load test.")
    if "high_voltage" in incident_set:
        tests.append("Check charger regulation and confirm the constant-voltage transition.")
    return tests


def _confidence_score(
    evidence: list[dict[str, object]],
    similar_cases: pd.DataFrame,
    data_source: str,
) -> float:
    grounding = 0.0
    if evidence:
        grounding = sum(bool(item.get("grounded")) for item in evidence) / len(evidence)

    similarity_component = 0.2
    if not similar_cases.empty and "distance" in similar_cases:
        mean_distance = float(similar_cases["distance"].head(3).mean())
        similarity_component = max(0.0, 1.0 - min(mean_distance / 3.0, 1.0))

    source_component = 0.6 if data_source == "processed" else 0.4
    raw_score = 0.25 + (0.35 * grounding) + (0.25 * similarity_component) + source_component * 0.15
    return round(float(min(max(raw_score, 0.05), 0.95)), 2)


def _caveats(data_source: str, similar_cases: pd.DataFrame) -> list[str]:
    caveats = [
        "The report is deterministic and the reported heuristic score is not a calibrated "
        "probability, "
        "based on baseline models plus nearest-neighbor retrieval.",
    ]
    if data_source != "processed":
        caveats.append(
            "Artifacts were generated from deterministic demo data because processed NASA parquet "
            "was not available locally."
        )
    if similar_cases.empty:
        caveats.append("No comparable historical incident was available for retrieval context.")
    return caveats


def _incident_types(raw_types: object) -> list[str]:
    if raw_types is None:
        return []
    return [token.strip() for token in str(raw_types).split(",") if token.strip()]
