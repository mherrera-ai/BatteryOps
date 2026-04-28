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
        "Fleet Cockpit",
        "Asset Replay",
        "Incident Evidence",
        "Similar Cases",
        "Model Evaluation",
        "Data & Provenance",
    ]
    assert re.search(r'subheader\("Fleet Cockpit"\)', source) is not None
    assert re.search(r'subheader\("Asset Replay"\)', source) is not None
    assert re.search(r'subheader\("Incident Evidence"\)', source) is not None
    assert re.search(r'subheader\("Similar Cases"\)', source) is not None
    assert re.search(r'subheader\("Model Evaluation"\)', source) is not None
    assert re.search(r'subheader\("Data & Provenance"\)', source) is not None


def test_streamlit_presentation_language_remains_local_first_and_safe() -> None:
    source = re.sub(r"\s+", " ", _streamlit_source().lower())
    assert "local-first" in source
    assert "runtime provenance" in source
    assert "public nasa data" in source
    assert "local models" in source
    assert "zero paid services" in source
    assert "validated bundle" in source
    assert "portfolio-grade ml engineering demo" in source
    assert "recruiter" not in source
    assert "not production ev safety software" in source
    assert "calibration benchmark" in source
    assert "validated risk model" in source


def test_streamlit_sidebar_copy_and_controls_stay_review_ready() -> None:
    source = _streamlit_source()
    for label in (
        "Runtime Context",
        "Runtime provenance",
        "Zero-cost contract",
        "Saved artifact scope",
        "Tab guide",
        "Report summary",
        "Triage handoff",
        "Public readiness checks",
        "Latest telemetry rows",
        "Cockpit decision ledger",
        "Selected-asset risk driver table",
        "Retrieved case table",
        "Data quality checks",
        "Model card summary",
    ):
        assert label in source
    for label in (
        "Step size",
        "Replay cycle cursor",
        "Alert threshold",
        "Flagged cycle queue",
        "Recommended next diagnostic tests",
        "Focus asset",
        "Proxy RUL MAE",
        "Proxy RUL",
        "Health index",
        "Runtime cost",
        "Bundle status",
        "Health scale",
    ):
        assert label in source
    assert "build_asset_risk_driver_frame" in source
    assert "build_cockpit_decision_frame" in source
    assert "build_fleet_risk_concentration_figure" in source
    assert "overview_asset_risk_drivers" in source
    assert "overview_fleet_risk_concentration" in source
    assert "REPLAY_STEP_SIZES" in source
    assert "FOCUS_ASSET_STATE_KEY" in source
    assert "_requested_focus_asset()" in source
    assert "_render_focus_asset_picker(data)" in source
    assert "Back " in source
    assert "Forward " in source
    assert "Jump to latest" in source


def test_streamlit_theme_tokens_preserve_presentation_direction() -> None:
    source = _streamlit_source()
    assert '"IBM Plex Sans"' in source
    assert '"IBM Plex Mono"' in source
    assert ".hero-card" in source
    assert ".hero-kicker" in source
    assert ".proof-strip" in source
    assert ".proof-item" in source
    assert "border-left: 6px solid" in source
    assert "border-radius: 8px" in source


def test_streamlit_metric_labels_stay_review_oriented() -> None:
    source = _streamlit_source()
    metric_labels = re.findall(r'\.metric\(\s*"([^"]+)"', source)
    for expected_label in (
        "Focus asset",
        "Latest cycle",
        "Proxy RUL",
        "Report heuristic",
        "Runtime cost",
        "Health index",
        "Selected-asset state",
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
