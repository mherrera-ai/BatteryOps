from __future__ import annotations

import re
from pathlib import Path


def _streamlit_source() -> str:
    source_path = Path(__file__).resolve().parents[1] / "src" / "batteryops" / "streamlit_app.py"
    return source_path.read_text(encoding="utf-8")


def _extract_tab_labels(source: str) -> list[str]:
    match = re.search(r"st\.tabs\(\s*\[\s*(.*?)\s*\]\s*\)", source, re.S)
    assert match is not None, "The tab declaration block is missing from streamlit_app.py"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_streamlit_tabs_and_section_headings_are_present() -> None:
    source = _streamlit_source()
    assert _extract_tab_labels(source) == [
        "Overview",
        "Live Telemetry Replay",
        "Anomaly Timeline",
        "Incident Report",
        "Similar Cases",
        "Evaluation Dashboard",
    ]
    assert re.search(r'subheader\("System Snapshot"\)', source) is not None
    assert re.search(r'subheader\("Live Telemetry Replay"\)', source) is not None
    assert re.search(r'subheader\("Anomaly Timeline"\)', source) is not None
    assert re.search(r'subheader\("Incident Report"\)', source) is not None
    assert re.search(r'subheader\("Similar Cases"\)', source) is not None
    assert re.search(r'subheader\("Evaluation Dashboard"\)', source) is not None


def test_streamlit_presentation_language_remains_local_first_and_safe() -> None:
    source = re.sub(r"\s+", " ", _streamlit_source().lower())
    assert "local-first" in source
    assert "runtime provenance" in source
    assert "public nasa battery data" in source
    assert "deterministic telemetry triage from saved nasa battery artifacts" in source
    assert "validated checked-in demo bundle" in source
    assert "renders from saved artifacts" in source
    assert "optional workspace state for preprocessing and retraining" in source
    assert "not a runtime dependency for this demo" in source
    assert "quick, deterministic, and easy to inspect offline" in source
    assert "recruiter-facing local demo" in source
    assert "not production ev safety software" in source
    assert "calibration benchmark" in source
    assert "validated risk model" in source


def test_streamlit_sidebar_copy_and_controls_stay_recruiter_visible() -> None:
    source = _streamlit_source()
    for label in (
        "Runtime Context",
        "Runtime provenance",
        "Saved artifact scope",
        "Tab guide",
        "Report summary",
        "Latest telemetry rows",
        "Retrieved case table",
    ):
        assert label in source
    for label in (
        "Replay speed",
        "Replay cycle cursor",
        "Alert threshold",
        "Flagged cycle queue",
        "Recommended next diagnostic tests",
        "Focus asset",
        "Proxy RUL MAE",
        "Proxy RUL",
        "Capacity-retention proxy",
    ):
        assert label in source
    assert "{speed}" in source
    assert "Step back" in source
    assert "Advance " in source
    assert "Jump to latest" in source


def test_streamlit_theme_tokens_preserve_presentation_direction() -> None:
    source = _streamlit_source()
    assert '"IBM Plex Sans"' in source
    assert '"IBM Plex Mono"' in source
    assert ".hero-card" in source
    assert ".hero-kicker" in source
    assert "radial-gradient" in source


def test_streamlit_metric_labels_stay_recruiter_oriented() -> None:
    source = _streamlit_source()
    metric_labels = re.findall(r'\.metric\(\s*"([^"]+)"', source)
    for expected_label in (
        "Focus asset",
        "Latest cycle",
        "Proxy RUL",
        "Report heuristic",
        "Capacity-retention proxy",
        "Heuristic alert state",
        "Replay cycle",
        "Anomaly score",
        "Flagged cycles",
        "Observed incidents",
        "Latest score vs threshold",
        "Retrieved cases",
        "Closest distance",
        "Best type overlap",
        "Proxy RUL MAE",
        "Alert precision",
        "Alert recall",
        "Evidence coverage",
        "Proxy RUL",
        "Alert marker",
    ):
        assert (expected_label in metric_labels) or (expected_label in source)
