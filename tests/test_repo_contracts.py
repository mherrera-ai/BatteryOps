from __future__ import annotations

import json
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import batteryops
from batteryops.cli import launch_demo
from batteryops.reports.demo import DEMO_ARTIFACT_FILENAMES, inspect_demo_bundle


def test_console_script_entrypoint_points_to_launch_demo() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["batteryops"] == "batteryops.cli:main"
    assert project["scripts"]["batteryops-demo"] == "batteryops.cli:launch_demo"


def test_launch_demo_honors_explicit_streamlit_app_override(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    override_app = tmp_path / "custom_streamlit_app.py"
    override_app.write_text("print('override')\n", encoding="utf-8")
    original_cwd = Path.cwd()

    def fake_run(cmd: list[str], check: bool) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = Path.cwd()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("batteryops.cli.subprocess.run", fake_run)
    monkeypatch.setattr("batteryops.cli._resolve_local_demo_app_path", lambda: None)
    monkeypatch.setenv("BATTERYOPS_STREAMLIT_APP", str(override_app))

    result = launch_demo(["--server.headless=true"])

    assert result == 0
    assert captured["check"] is False
    assert captured["cmd"][:4] == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
    ]
    assert captured["cmd"][4] == str(override_app.resolve())
    assert captured["cmd"][5:] == ["--server.headless=true"]
    assert captured["cwd"] == original_cwd


def test_readme_and_screenshot_docs_expose_canonical_local_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    screenshot_guide = (repo_root / "docs" / "screenshots" / "README.md").read_text(
        encoding="utf-8"
    )
    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")

    for expected in (
        'python3 -m pip install -e ".[dev]"',
        "batteryops-demo",
        (
            "`batteryops-demo` is the saved-bundle launch path documented here; "
            "`batteryops` is the shorter alias."
        ),
        "Run `batteryops-demo` from the repository root",
        "make check",
        "make screenshots",
        "Do not read `data/processed/` directly at app startup",
        "## 5-Minute Review",
        "## What Recruiters See",
        "## Publishing Notes",
        "[Contributing](CONTRIBUTING.md)",
        "[Security](SECURITY.md)",
    ):
        assert expected in readme

    for expected in (
        "make screenshots",
        "node docs/screenshots/capture.mjs",
        "npm ci",
        "BATTERYOPS_APP_COMMAND",
        "BATTERYOPS_SCREENSHOT_DIR",
    ):
        assert expected in screenshot_guide

    for expected in ("install:", "check:", "demo:", "demo-headless:", "screenshots:"):
        assert expected in makefile


def test_public_docs_keep_processed_parquet_out_of_the_launch_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    architecture = (repo_root / "docs" / "architecture.md").read_text(encoding="utf-8")

    assert "Local `data/processed/` parquet is optional" in readme
    assert "The checked-in public demo does not need those local caches to launch" in readme
    assert "does not read `data/processed/` directly at startup" in architecture
    assert "not a runtime dependency for the public demo path" in architecture


def test_project_urls_point_to_public_repo_owner() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    urls = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["urls"]

    assert urls["Homepage"] == "https://github.com/mherrera-ai/Batteryops"
    assert urls["Repository"] == "https://github.com/mherrera-ai/Batteryops"
    assert urls["Issues"] == "https://github.com/mherrera-ai/Batteryops/issues"
    assert urls["Documentation"] == "https://github.com/mherrera-ai/Batteryops#readme"


def test_readme_bundle_claims_match_checked_in_demo_bundle() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    architecture = (repo_root / "docs" / "architecture.md").read_text(encoding="utf-8")
    metrics = json.loads((repo_root / "artifacts" / "demo" / "demo_metrics.json").read_text())

    metrics_block = readme.split("## Current Demo Bundle Metrics", maxsplit=1)[1].split(
        "## ", maxsplit=1
    )[0]

    expected_rows = (
        f"| Data source | `{metrics['data_source']}` |",
        f"| Evaluation mode | `{metrics['evaluation_mode']}` |",
        f"| Assets in artifact bundle | `{metrics['asset_count']}` |",
        f"| Cycle count | `{metrics['cycle_count']}` |",
        f"| Incident case count | `{metrics['incident_case_count']}` |",
        f"| Alert lead time | `{metrics['alert_lead_time']:.4f}` cycles |",
        f"| Alert precision | `{metrics['alert_precision'] * 100:.2f}%` |",
        f"| Alert recall | `{metrics['alert_recall'] * 100:.2f}%` |",
        f"| False positive rate | `{metrics['false_positive_rate'] * 100:.1f}%` |",
        f"| RUL proxy MAE | `{metrics['rul_proxy_mae']:.3f}` cycles |",
        f"| Evidence source coverage | `{metrics['evidence_source_coverage'] * 100:.0f}%` |",
    )

    for row in expected_rows:
        assert row in metrics_block

    assert (
        f"current `{metrics['asset_count']}`-asset, `{metrics['cycle_count']}`-cycle, "
        f"`{metrics['incident_case_count']}`-incident bundle"
    ) in readme
    assert (
        f"The saved bundle is a {metrics['evaluation_mode']} over {metrics['asset_count']} assets "
        f"and {metrics['cycle_count']} cycles."
    ) in architecture


def test_readme_start_here_links_resolve_to_existing_sections() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    start_here_block = readme.split("## Start Here", maxsplit=1)[1].split("## ", maxsplit=1)[0]
    anchors = [
        anchor
        for anchor in (
            line.partition("(#")[2].rstrip(")")
            for line in start_here_block.splitlines()
            if "(#" in line
        )
        if anchor
    ]
    headings = {
        _github_anchor(line.removeprefix("## ").strip())
        for line in readme.splitlines()
        if line.startswith("## ")
    }

    assert anchors
    assert set(anchors).issubset(headings)


def test_screenshot_capture_script_targets_the_six_documented_tabs() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "docs" / "screenshots" / "capture.mjs"
    ).read_text(encoding="utf-8")

    assert [shot for shot in _extract_js_array_values(source, "name")] == [
        "overview.png",
        "live-telemetry-replay.png",
        "anomaly-timeline.png",
        "incident-report.png",
        "similar-cases.png",
        "evaluation-dashboard.png",
    ]
    assert [shot for shot in _extract_js_array_values(source, "tab")] == [
        "Overview",
        "Live Telemetry Replay",
        "Anomaly Timeline",
        "Incident Report",
        "Similar Cases",
        "Evaluation Dashboard",
    ]
    assert source.count("focusAsset: 'battery36'") == 3
    assert source.count("focusAsset: 'battery50'") == 3


def test_installed_distribution_metadata_matches_project_version_and_layout() -> None:
    dist = metadata.distribution("batteryops")

    assert dist.metadata["Name"] == "batteryops"
    assert dist.version == batteryops.__version__
    assert dist.metadata["Version"] == batteryops.__version__
    assert dist.metadata["Requires-Python"] == ">=3.11"
    assert dist.read_text("top_level.txt") == "batteryops\n"


def test_checked_in_demo_bundle_manifest_exposes_fingerprint_inventory() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "artifacts" / "demo" / "training_manifest.json"
    demo_dir = repo_root / "artifacts" / "demo"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = inspect_demo_bundle()

    assert status.healthy is True
    assert status.bundle_fingerprint
    assert len(status.bundle_fingerprint) == 64
    assert len(status.artifact_inventory) == len(DEMO_ARTIFACT_FILENAMES)
    assert manifest["provenance"]["bundle_fingerprint"]["value"] == status.bundle_fingerprint
    assert manifest["artifacts"]["training_manifest"] == str(manifest_path.relative_to(repo_root))
    assert {item.name for item in demo_dir.iterdir()} == {
        ".gitkeep",
        "training_manifest.json",
        "rul_model.joblib",
        "anomaly_model.joblib",
        "incident_retrieval.joblib",
        "demo_cycle_predictions.parquet",
        "demo_incident_cases.parquet",
        "demo_metrics.json",
        "demo_report.json",
    }


def test_gitignore_covers_local_only_data_and_generated_smoke_outputs() -> None:
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")

    for expected in (
        ".venv/",
        "AGENTS.md",
        ".pkgtest-wheel/",
        ".pkgtest/",
        ".pkgtest-*/",
        ".release-*/",
        "node_modules/",
        "build/",
        "dist/",
        "*.egg-info/",
        "data/raw/*",
        "!data/raw/.gitkeep",
        "data/processed/*",
        "!data/processed/.gitkeep",
        "artifacts/*",
        "!artifacts/demo/",
        "!artifacts/demo/**",
    ):
        assert expected in gitignore
    assert "package-lock.json" not in gitignore


def test_public_repo_community_files_exist_and_match_repo_scope() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contributing = (repo_root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    security = (repo_root / "SECURITY.md").read_text(encoding="utf-8")
    bug_report = (
        repo_root / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    ).read_text(encoding="utf-8")
    feature_request = (
        repo_root / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml"
    ).read_text(encoding="utf-8")
    issue_config = (
        repo_root / ".github" / "ISSUE_TEMPLATE" / "config.yml"
    ).read_text(encoding="utf-8")
    pr_template = (repo_root / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )
    dependabot = (repo_root / ".github" / "dependabot.yml").read_text(encoding="utf-8")

    for expected in (
        "local-first Streamlit demo",
        "make check",
        "make screenshots",
        "Do not add raw NASA ZIPs",
        "Do not introduce React, FastAPI, hosted services, paid APIs, auth, cloud infra",
    ):
        assert expected in contributing

    for expected in (
        "GitHub's security advisory flow",
        "does not operate a hosted service",
        "production EV safety software",
    ):
        assert expected in security

    assert 'labels: ["bug"]' in bug_report
    assert 'labels: ["enhancement"]' in feature_request
    assert "blank_issues_enabled: false" in issue_config
    assert "README and demo overview" in issue_config
    assert "## Summary" in pr_template
    assert "- [ ] `make check`" in pr_template
    assert 'package-ecosystem: "pip"' in dependabot
    assert 'package-ecosystem: "npm"' in dependabot
    assert 'package-ecosystem: "github-actions"' in dependabot


def test_screenshot_tooling_is_pinned_for_reproducible_refreshes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package_json = json.loads((repo_root / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads((repo_root / "package-lock.json").read_text(encoding="utf-8"))

    assert package_json["private"] is True
    assert package_json["engines"]["node"] == "22.x"
    assert package_json["devDependencies"]["playwright"] == "1.58.2"
    assert package_lock["lockfileVersion"] >= 3
    assert package_lock["packages"][""]["devDependencies"]["playwright"] == "1.58.2"


def test_ruff_excludes_local_smoke_environments_from_repo_checks() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    ruff_exclude = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["ruff"]["exclude"]

    assert ".venv" in ruff_exclude
    assert ".pkgtest*" in ruff_exclude
    assert "build" in ruff_exclude
    assert "dist" in ruff_exclude
    assert "node_modules" in ruff_exclude


def _extract_js_array_values(source: str, key: str) -> list[str]:
    lines: list[str] = []
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(f"{key}: '") and stripped.endswith("',"):
            lines.append(stripped.split("'", 2)[1])
    return lines


def _github_anchor(heading: str) -> str:
    collapsed = " ".join(heading.lower().split())
    sanitized = "".join(
        character for character in collapsed if character.isalnum() or character in " -"
    )
    return sanitized.replace(" ", "-")
