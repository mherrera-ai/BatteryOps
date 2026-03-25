from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ABSOLUTE_LOW_VOLTAGE_V = 3.2
ABSOLUTE_HIGH_VOLTAGE_V = 8.85
ABSOLUTE_HIGH_TEMPERATURE_C = 40.0
ABSOLUTE_HIGH_CURRENT_A = 16.5
MIN_HISTORY_CYCLES = 4


@dataclass(frozen=True)
class SampleIncidentRule:
    """Raw-sample excursions used for cycle-level counts."""

    column: str
    label: str


@dataclass(frozen=True)
class CycleIncidentRule:
    """Cycle-level deviations used to define recruiter-facing incident cases."""

    metric: str
    label: str
    direction: str
    mad_scale: float
    floor: float
    absolute_limit: float | None = None


SAMPLE_INCIDENT_RULES = (
    SampleIncidentRule("flag_low_voltage", "low_voltage"),
    SampleIncidentRule("flag_high_voltage", "high_voltage"),
    SampleIncidentRule("flag_high_temperature", "high_temperature"),
    SampleIncidentRule("flag_high_current", "high_current"),
)

CYCLE_INCIDENT_RULES = (
    CycleIncidentRule(
        metric="voltage_min_v",
        label="low_voltage",
        direction="low",
        mad_scale=1.5,
        floor=0.05,
        absolute_limit=ABSOLUTE_LOW_VOLTAGE_V,
    ),
    CycleIncidentRule(
        metric="voltage_max_v",
        label="high_voltage",
        direction="high",
        mad_scale=1.0,
        floor=0.08,
    ),
    CycleIncidentRule(
        metric="temperature_max_c",
        label="high_temperature",
        direction="high",
        mad_scale=1.25,
        floor=1.5,
        absolute_limit=ABSOLUTE_HIGH_TEMPERATURE_C,
    ),
    CycleIncidentRule(
        metric="current_abs_max_a",
        label="high_current",
        direction="high",
        mad_scale=0.5,
        floor=0.2,
    ),
)


def add_incident_flags(samples: pd.DataFrame) -> pd.DataFrame:
    """Annotate raw-sample excursions used for downstream cycle summaries."""
    flagged = samples.copy()
    current_abs = flagged["current_a"].fillna(0.0).abs()
    incident_voltage = _incident_voltage(flagged)
    charge_voltage = _charge_voltage(flagged)

    flagged["flag_low_voltage"] = incident_voltage.fillna(np.inf) <= ABSOLUTE_LOW_VOLTAGE_V
    flagged["flag_high_voltage"] = charge_voltage.fillna(-np.inf) >= ABSOLUTE_HIGH_VOLTAGE_V
    flagged["flag_high_temperature"] = (
        flagged["temperature_c"].fillna(-np.inf) >= ABSOLUTE_HIGH_TEMPERATURE_C
    )
    flagged["flag_high_current"] = current_abs >= ABSOLUTE_HIGH_CURRENT_A
    flagged["incident_flag"] = flagged[[rule.column for rule in SAMPLE_INCIDENT_RULES]].any(axis=1)
    return flagged


def build_cycle_features(samples: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized telemetry into cycle-level features."""
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "dataset_code",
                "asset_id",
                "asset_group",
                "cycle_kind",
                "cycle_id",
                "cycle_start_time",
                "step_comment",
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
                "discharge_capacity_proxy_ah",
                "charge_sample_count",
                "discharge_sample_count",
                "rest_sample_count",
                "incident_sample_count",
                "is_reference_cycle",
            ]
        )

    flagged = add_incident_flags(samples).sort_values(
        ["dataset_code", "asset_id", "cycle_id", "sample_time_s", "sample_index"]
    )

    rows: list[dict[str, object]] = []
    group_cols = ["dataset_code", "asset_id", "asset_group", "cycle_kind", "cycle_id"]

    for _, frame in flagged.groupby(group_cols, sort=False):
        cycle_frame = frame.reset_index(drop=True)
        time_values = cycle_frame["sample_time_s"].astype(float).to_numpy()
        current_values = cycle_frame["current_a"].fillna(0.0).astype(float).to_numpy()
        voltage_values = _bounded_numeric_series(cycle_frame["voltage_v"], lower=0.5, upper=20.0)
        temperature_values = _bounded_numeric_series(
            cycle_frame["temperature_c"],
            lower=-20.0,
            upper=120.0,
        )
        discharge_mask = cycle_frame["mode_code"].eq(-1).to_numpy()

        rows.append(
            {
                "dataset_code": str(cycle_frame["dataset_code"].iloc[0]),
                "asset_id": str(cycle_frame["asset_id"].iloc[0]),
                "asset_group": str(cycle_frame["asset_group"].iloc[0]),
                "cycle_kind": str(cycle_frame["cycle_kind"].iloc[0]),
                "cycle_id": int(cycle_frame["cycle_id"].iloc[0]),
                "cycle_start_time": cycle_frame["cycle_start_time"].iloc[0],
                "step_comment": str(cycle_frame["step_comment"].iloc[0]),
                "sample_count": int(len(cycle_frame)),
                "duration_s": (
                    float(time_values.max() - time_values.min()) if len(time_values) else 0.0
                ),
                "voltage_min_v": (
                    float(voltage_values.min()) if not voltage_values.empty else np.nan
                ),
                "voltage_max_v": (
                    float(voltage_values.max()) if not voltage_values.empty else np.nan
                ),
                "voltage_mean_v": (
                    float(voltage_values.mean()) if not voltage_values.empty else np.nan
                ),
                "temperature_max_c": (
                    float(temperature_values.max()) if not temperature_values.empty else np.nan
                ),
                "temperature_mean_c": (
                    float(temperature_values.mean()) if not temperature_values.empty else np.nan
                ),
                "current_abs_max_a": (
                    float(np.abs(current_values).max()) if len(current_values) else 0.0
                ),
                "throughput_ah": _integrate(np.abs(current_values), time_values),
                "net_current_ah": _integrate(current_values, time_values),
                "discharge_capacity_proxy_ah": _integrate(
                    np.abs(current_values[discharge_mask]),
                    time_values[discharge_mask],
                ),
                "charge_sample_count": int(cycle_frame["mode_code"].eq(1).sum()),
                "discharge_sample_count": int(cycle_frame["mode_code"].eq(-1).sum()),
                "rest_sample_count": int(cycle_frame["mode_code"].eq(0).sum()),
                "incident_sample_count": int(cycle_frame["incident_flag"].sum()),
                "is_reference_cycle": _is_reference_cycle(cycle_frame),
            }
        )

    return pd.DataFrame(rows)


def build_incident_windows(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarize cycle-level deviations into recruiter-facing incident cases."""
    if samples.empty:
        return pd.DataFrame(
            columns=[
                "dataset_code",
                "asset_id",
                "cycle_id",
                "window_id",
                "cycle_start_time",
                "window_start_time",
                "window_end_time",
                "window_start_sample_s",
                "window_end_sample_s",
                "duration_s",
                "sample_count",
                "incident_types",
                "max_temperature_c",
                "min_voltage_v",
                "max_abs_current_a",
                "severity_score",
            ]
        )

    cycle_features = (
        build_cycle_features(samples).sort_values(["asset_id", "cycle_id"]).reset_index(drop=True)
    )
    if cycle_features.empty:
        return build_incident_windows(pd.DataFrame())

    cycle_bounds = (
        samples.groupby(["dataset_code", "asset_id", "cycle_id"], as_index=False)
        .agg(
            window_start_time=("sample_timestamp", "min"),
            window_end_time=("sample_timestamp", "max"),
            window_start_sample_s=("sample_time_s", "min"),
            window_end_sample_s=("sample_time_s", "max"),
        )
        .reset_index(drop=True)
    )
    cycle_flags = _build_cycle_incident_flags(cycle_features)
    incident_cycles = cycle_features.join(cycle_flags)
    incident_cycles = incident_cycles.loc[incident_cycles["incident_types"] != ""].copy()
    if incident_cycles.empty:
        return build_incident_windows(pd.DataFrame())

    incident_cycles = incident_cycles.merge(
        cycle_bounds,
        on=["dataset_code", "asset_id", "cycle_id"],
        how="left",
    )

    rows: list[dict[str, object]] = []
    for _, cycle in incident_cycles.iterrows():
        incident_types = [
            token.strip() for token in str(cycle["incident_types"]).split(",") if token.strip()
        ]
        severity_score = float(len(incident_types) + min(float(cycle["severity_bonus"]), 3.0))
        rows.append(
            {
                "dataset_code": str(cycle["dataset_code"]),
                "asset_id": str(cycle["asset_id"]),
                "cycle_id": int(cycle["cycle_id"]),
                "window_id": f"{cycle['asset_id']}-{int(cycle['cycle_id'])}-1",
                "cycle_start_time": cycle["cycle_start_time"],
                "window_start_time": cycle["window_start_time"],
                "window_end_time": cycle["window_end_time"],
                "window_start_sample_s": float(cycle["window_start_sample_s"]),
                "window_end_sample_s": float(cycle["window_end_sample_s"]),
                "duration_s": float(cycle["duration_s"]),
                "sample_count": int(cycle["sample_count"]),
                "incident_types": ",".join(incident_types),
                "max_temperature_c": float(cycle["temperature_max_c"]),
                "min_voltage_v": float(cycle["voltage_min_v"]),
                "max_abs_current_a": float(cycle["current_abs_max_a"]),
                "severity_score": round(severity_score, 3),
            }
        )

    return pd.DataFrame(rows)


def _integrate(values: np.ndarray, time_values: np.ndarray) -> float:
    if len(values) < 2 or len(time_values) < 2:
        return 0.0
    return float(np.trapezoid(values, time_values) / 3600.0)


def _is_reference_cycle(frame: pd.DataFrame) -> bool:
    labels = frame["step_comment"].fillna("").astype(str).str.lower()
    mission = frame["mission_label"].fillna("").astype(str).str.lower()
    return bool(labels.str.contains("reference").any() or mission.str.contains("reference").any())


def _incident_voltage(samples: pd.DataFrame) -> pd.Series:
    load_voltage = samples["voltage_load_v"]
    has_only_general_voltage = load_voltage.isna() & samples["voltage_charger_v"].isna()
    general_voltage = samples["voltage_v"].where(has_only_general_voltage)
    return load_voltage.where(~load_voltage.isna(), general_voltage)


def _charge_voltage(samples: pd.DataFrame) -> pd.Series:
    charger_voltage = samples["voltage_charger_v"]
    has_only_general_voltage = charger_voltage.isna() & samples["voltage_load_v"].isna()
    general_voltage = samples["voltage_v"].where(has_only_general_voltage)
    return charger_voltage.where(~charger_voltage.isna(), general_voltage)


def _bounded_numeric_series(
    values: pd.Series,
    lower: float,
    upper: float,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.loc[numeric.between(lower, upper)]


def _build_cycle_incident_flags(cycle_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, frame in cycle_features.groupby("asset_id", sort=False):
        ordered = frame.sort_values("cycle_id").reset_index(drop=True)
        flags = pd.DataFrame(index=ordered.index)
        severity_bonus = pd.Series(0.0, index=ordered.index, dtype=float)
        history_index = pd.Series(np.arange(len(ordered)), index=ordered.index, dtype=int)

        for rule in CYCLE_INCIDENT_RULES:
            values = pd.to_numeric(ordered[rule.metric], errors="coerce")
            baseline = values.expanding(min_periods=MIN_HISTORY_CYCLES).median().shift(1)
            mad = values.expanding(min_periods=MIN_HISTORY_CYCLES).apply(
                _median_absolute_deviation,
                raw=False,
            ).shift(1)
            margin = (mad.fillna(0.0) * rule.mad_scale).clip(lower=rule.floor)
            absolute_flag = pd.Series(False, index=ordered.index, dtype=bool)
            if rule.absolute_limit is not None:
                if rule.direction == "low":
                    absolute_flag = (history_index < MIN_HISTORY_CYCLES) & values.le(
                        rule.absolute_limit
                    )
                else:
                    absolute_flag = (history_index < MIN_HISTORY_CYCLES) & values.ge(
                        rule.absolute_limit
                    )
            if rule.direction == "low":
                dynamic_flag = values.lt(baseline - margin)
                rule_bonus = ((baseline - values) / margin.replace(0.0, np.nan)).clip(lower=0.0)
            else:
                dynamic_flag = values.gt(baseline + margin)
                rule_bonus = ((values - baseline) / margin.replace(0.0, np.nan)).clip(lower=0.0)
            final_flag = (absolute_flag | dynamic_flag.fillna(False)).fillna(False)
            flags[f"flag_{rule.label}"] = final_flag
            severity_bonus = severity_bonus.add(
                rule_bonus.where(final_flag, 0.0).fillna(0.0),
                fill_value=0.0,
            )

        flags["severity_bonus"] = severity_bonus.round(3)
        flags["incident_types"] = flags.apply(_format_incident_types, axis=1)
        rows.append(flags)

    return pd.concat(rows, ignore_index=True)


def _median_absolute_deviation(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    median = float(clean.median())
    return float(np.median(np.abs(clean.to_numpy() - median)))


def _format_incident_types(row: pd.Series) -> str:
    incident_types: list[str] = []
    for rule in CYCLE_INCIDENT_RULES:
        if bool(row.get(f"flag_{rule.label}", False)):
            incident_types.append(rule.label)
    return ",".join(incident_types)
