from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_dashboard_renders_tabs_and_controls() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(app_path)
    app.run(timeout=30)

    assert not list(app.exception)
    assert [tab.label for tab in app.tabs] == [
        "Fleet Cockpit",
        "Asset Replay",
        "Incident Evidence",
        "Similar Cases",
        "Model Evaluation",
        "Data & Provenance",
    ]
    assert "Focus asset" in [selectbox.label for selectbox in app.selectbox]
    assert "Step size" in [radio.label for radio in app.radio]
    assert {"Replay cycle cursor", "Alert threshold"}.issubset(
        {slider.label for slider in app.slider}
    )
    assert {
        "Proxy RUL",
        "Report heuristic",
        "Health index",
        "Selected-asset state",
        "Retrieved cases",
        "Closest distance",
        "Best type overlap",
        "Proxy RUL MAE",
        "Evidence coverage",
        "Runtime cost",
    }.issubset({metric.label for metric in app.metric})
    assert {
        "**Report summary**",
        "**Triage handoff**",
        "**Cockpit decision ledger**",
        "**Retrieved case table**",
        "**Recommended next diagnostic tests**",
        "**Data quality checks**",
    }.issubset({markdown.value for markdown in app.markdown if isinstance(markdown.value, str)})
    selected_asset_state = next(
        metric.value for metric in app.metric if metric.label == "Selected-asset state"
    )
    assert selected_asset_state in {
        "Inspect soon",
        "Monitor",
    }
    assert next(metric.value for metric in app.metric if metric.label == "Alert marker") in {
        "Threshold breach",
        "Monitor",
    }


def test_replay_controls_update_without_streamlit_state_errors() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(app_path)
    app.run(timeout=30)

    latest_cycle = int(app.slider[0].value)

    app.button[0].click().run(timeout=30)
    assert not list(app.exception)
    assert int(app.slider[0].value) == 1

    app.button[2].click().run(timeout=30)
    assert not list(app.exception)
    assert int(app.slider[0].value) == 2

    app.button[3].click().run(timeout=30)
    assert not list(app.exception)
    assert int(app.slider[0].value) == latest_cycle


def test_focus_asset_selector_updates_visible_state() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    app = AppTest.from_file(app_path)
    app.run(timeout=30)

    app.selectbox[0].set_value("battery36")
    app.run(timeout=30)

    assert not list(app.exception)
    assert app.selectbox[0].value == "battery36"
    focus_asset = next(metric.value for metric in app.metric if metric.label == "Focus asset")
    assert focus_asset == "battery36"
    assert any(
        isinstance(markdown.value, str) and "asset `battery36`." in markdown.value
        for markdown in app.markdown
    )
