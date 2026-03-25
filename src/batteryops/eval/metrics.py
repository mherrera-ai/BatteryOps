from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def alert_lead_time(cycle_predictions: pd.DataFrame) -> float:
    """Average number of cycles between first alert and first incident."""
    required = {"asset_id", "cycle_id", "actual_alert", "predicted_alert"}
    if cycle_predictions.empty or not required.issubset(cycle_predictions.columns):
        return 0.0

    leads: list[float] = []
    for _, frame in cycle_predictions.groupby("asset_id", sort=False):
        ordered = frame.sort_values("cycle_id")
        actual_cycles = ordered.loc[ordered["actual_alert"], "cycle_id"]
        if actual_cycles.empty:
            continue

        first_actual = int(actual_cycles.min())
        prior_alerts = ordered.loc[
            ordered["predicted_alert"] & (ordered["cycle_id"] <= first_actual),
            "cycle_id",
        ]
        if prior_alerts.empty:
            leads.append(0.0)
            continue

        leads.append(float(first_actual - int(prior_alerts.min())))

    return float(np.mean(leads)) if leads else 0.0


def false_positive_rate(
    actual_alert: Sequence[bool] | pd.Series,
    predicted_alert: Sequence[bool] | pd.Series,
) -> float:
    """Fraction of non-incident rows that were incorrectly alerted."""
    actual = np.asarray(actual_alert, dtype=bool)
    predicted = np.asarray(predicted_alert, dtype=bool)

    negatives = ~actual
    negative_count = int(negatives.sum())
    if negative_count == 0:
        return 0.0

    return float((predicted & negatives).sum() / negative_count)


def alert_precision(
    actual_alert: Sequence[bool] | pd.Series,
    predicted_alert: Sequence[bool] | pd.Series,
) -> float:
    """Precision for inspect-soon alerts."""
    actual = np.asarray(actual_alert, dtype=bool)
    predicted = np.asarray(predicted_alert, dtype=bool)
    predicted_count = int(predicted.sum())
    if predicted_count == 0:
        return 0.0
    return float((predicted & actual).sum() / predicted_count)


def alert_recall(
    actual_alert: Sequence[bool] | pd.Series,
    predicted_alert: Sequence[bool] | pd.Series,
) -> float:
    """Recall for inspect-soon alerts."""
    actual = np.asarray(actual_alert, dtype=bool)
    predicted = np.asarray(predicted_alert, dtype=bool)
    actual_count = int(actual.sum())
    if actual_count == 0:
        return 0.0
    return float((predicted & actual).sum() / actual_count)


def rul_mae(
    actual_rul: Sequence[float] | pd.Series,
    predicted_rul: Sequence[float] | pd.Series,
) -> float:
    """Mean absolute error for cycle-based RUL targets."""
    actual = np.asarray(actual_rul, dtype=float)
    predicted = np.asarray(predicted_rul, dtype=float)
    if len(actual) == 0:
        return 0.0
    return float(np.mean(np.abs(actual - predicted)))


def report_grounding_coverage(report: dict[str, object]) -> float:
    """Share of evidence bullets backed by explicit source fields and values."""
    evidence = report.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return 0.0

    grounded = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        grounded_flag = bool(item.get("grounded"))
        has_field = bool(item.get("source_field"))
        has_value = item.get("value") is not None
        if grounded_flag and has_field and has_value:
            grounded += 1

    return grounded / len(evidence)
