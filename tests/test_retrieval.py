from __future__ import annotations

import pandas as pd

from batteryops.retrieval import fit_retrieval_index, retrieve_similar_cases


def test_retrieve_similar_cases_prefers_shared_incident_shape() -> None:
    incidents = pd.DataFrame(
        [
            {
                "window_id": "A-10-1",
                "asset_id": "A",
                "cycle_id": 10,
                "incident_types": "low_voltage,high_temperature",
                "duration_s": 210.0,
                "sample_count": 8,
                "max_temperature_c": 41.2,
                "min_voltage_v": 3.12,
                "max_abs_current_a": 4.1,
                "severity_score": 3.5,
            },
            {
                "window_id": "B-18-1",
                "asset_id": "B",
                "cycle_id": 18,
                "incident_types": "low_voltage,high_temperature",
                "duration_s": 220.0,
                "sample_count": 9,
                "max_temperature_c": 41.8,
                "min_voltage_v": 3.09,
                "max_abs_current_a": 4.0,
                "severity_score": 3.7,
            },
            {
                "window_id": "C-05-1",
                "asset_id": "C",
                "cycle_id": 5,
                "incident_types": "high_current",
                "duration_s": 90.0,
                "sample_count": 4,
                "max_temperature_c": 29.0,
                "min_voltage_v": 3.76,
                "max_abs_current_a": 4.6,
                "severity_score": 1.6,
            },
        ]
    )

    bundle = fit_retrieval_index(incidents)
    matches = retrieve_similar_cases(bundle, incidents.iloc[0], top_k=2, exclude_window_id="A-10-1")

    assert len(matches) == 2
    assert matches.iloc[0]["window_id"] == "B-18-1"
    assert float(matches.iloc[0]["distance"]) <= float(matches.iloc[1]["distance"])

