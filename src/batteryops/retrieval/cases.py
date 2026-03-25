from __future__ import annotations

from dataclasses import dataclass

import joblib  # type: ignore[import-untyped]
import pandas as pd
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

INCIDENT_FEATURE_COLUMNS = [
    "duration_s",
    "sample_count",
    "max_temperature_c",
    "min_voltage_v",
    "max_abs_current_a",
    "severity_score",
    "type_low_voltage",
    "type_high_voltage",
    "type_high_temperature",
    "type_high_current",
]

INCIDENT_TYPES = [
    "low_voltage",
    "high_voltage",
    "high_temperature",
    "high_current",
]


@dataclass
class IncidentRetrievalBundle:
    """Saved retrieval components for similar-incident search."""

    scaler: StandardScaler
    neighbors: NearestNeighbors
    feature_columns: list[str]
    fill_values: dict[str, float]
    case_features: pd.DataFrame
    case_metadata: pd.DataFrame


def fit_retrieval_index(incident_windows: pd.DataFrame) -> IncidentRetrievalBundle:
    """Fit a nearest-neighbor index over standardized incident vectors."""
    if incident_windows.empty:
        raise ValueError("Cannot build retrieval index from an empty incident table.")

    case_features = build_incident_feature_table(incident_windows)
    fill_values = {
        column: float(case_features[column].median())
        if not case_features[column].dropna().empty
        else 0.0
        for column in case_features.columns
    }
    filled = case_features.fillna(fill_values)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)
    neighbors = NearestNeighbors(
        metric="euclidean",
        n_neighbors=min(len(case_features), max(len(case_features), 1)),
    )
    neighbors.fit(scaled)

    metadata_columns = [
        "window_id",
        "asset_id",
        "cycle_id",
        "incident_types",
        "severity_score",
        "duration_s",
        "max_temperature_c",
        "min_voltage_v",
        "max_abs_current_a",
    ]
    metadata = incident_windows[metadata_columns].reset_index(drop=True).copy()
    return IncidentRetrievalBundle(
        scaler=scaler,
        neighbors=neighbors,
        feature_columns=list(case_features.columns),
        fill_values=fill_values,
        case_features=filled.reset_index(drop=True),
        case_metadata=metadata,
    )


def retrieve_similar_cases(
    bundle: IncidentRetrievalBundle,
    query_incident: pd.Series | dict[str, object],
    top_k: int = 3,
    exclude_window_id: str | None = None,
) -> pd.DataFrame:
    """Return the nearest historical incidents to the provided query."""
    if top_k <= 0 or bundle.case_metadata.empty:
        return pd.DataFrame(columns=list(bundle.case_metadata.columns) + ["distance"])

    query_row = (
        query_incident
        if isinstance(query_incident, pd.Series)
        else pd.Series(query_incident, dtype=object)
    )
    query_features = build_incident_feature_table(pd.DataFrame([query_row]))
    query_vector = query_features.reindex(columns=bundle.feature_columns).fillna(bundle.fill_values)
    scaled_query = bundle.scaler.transform(query_vector)

    neighbor_count = min(len(bundle.case_metadata), top_k + 1)
    distances, indices = bundle.neighbors.kneighbors(scaled_query, n_neighbors=neighbor_count)
    result = bundle.case_metadata.iloc[indices[0]].copy()
    result["distance"] = distances[0]

    window_id = exclude_window_id
    if window_id is None and "window_id" in query_row:
        window_id = str(query_row["window_id"])
    if window_id is not None and "window_id" in result:
        result = result.loc[result["window_id"] != window_id]

    return result.nsmallest(top_k, "distance").reset_index(drop=True)


def save_retrieval_bundle(bundle: IncidentRetrievalBundle, path: str) -> None:
    """Persist the retrieval components with joblib."""
    joblib.dump(bundle, path)


def load_retrieval_bundle(path: str) -> IncidentRetrievalBundle:
    """Load the retrieval components from a joblib artifact."""
    return joblib.load(path)


def build_incident_feature_table(incident_windows: pd.DataFrame) -> pd.DataFrame:
    """Convert incident rows into fixed-width numeric vectors."""
    frame = incident_windows.copy()
    incident_tokens = frame.get("incident_types", pd.Series(dtype=str)).fillna("").astype(str)

    feature_table = pd.DataFrame(
        {
            "duration_s": _numeric_column(frame, "duration_s"),
            "sample_count": _numeric_column(frame, "sample_count"),
            "max_temperature_c": _numeric_column(frame, "max_temperature_c"),
            "min_voltage_v": _numeric_column(frame, "min_voltage_v"),
            "max_abs_current_a": _numeric_column(frame, "max_abs_current_a"),
            "severity_score": _numeric_column(frame, "severity_score"),
        }
    )

    for incident_type in INCIDENT_TYPES:
        feature_table[f"type_{incident_type}"] = incident_tokens.str.contains(
            incident_type,
            regex=False,
        ).astype(float)

    return feature_table.reindex(columns=INCIDENT_FEATURE_COLUMNS)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame else pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(values, errors="coerce")
