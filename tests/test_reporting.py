from __future__ import annotations

import pandas as pd

from batteryops.eval import report_grounding_coverage
from batteryops.reports import generate_incident_report


def test_generate_incident_report_contains_required_sections() -> None:
    incident = pd.Series(
        {
            "asset_id": "DEMO-06",
            "cycle_id": 24,
            "incident_types": "low_voltage,high_temperature",
            "duration_s": 240.0,
            "sample_count": 11,
            "max_temperature_c": 42.5,
            "min_voltage_v": 3.08,
            "max_abs_current_a": 4.2,
            "severity_score": 4.0,
        }
    )
    similar_cases = pd.DataFrame(
        [
            {
                "window_id": "DEMO-02-22-1",
                "asset_id": "DEMO-02",
                "cycle_id": 22,
                "incident_types": "low_voltage,high_temperature",
                "severity_score": 3.8,
                "distance": 0.42,
            }
        ]
    )

    report = generate_incident_report(incident, similar_cases, data_source="processed")

    assert "summary" in report
    assert report["similar_cases"]
    assert report["recommended_tests"]
    assert report["confidence_score"] >= 0.0
    assert report_grounding_coverage(report) == 1.0

