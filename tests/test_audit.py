from __future__ import annotations

import json
from pathlib import Path

from batteryops.audit import build_public_readiness_audit, format_public_readiness_audit, main


def test_public_readiness_audit_passes_current_repo_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    audit = build_public_readiness_audit(repo_root)

    assert audit["schema_version"] == 1
    assert audit["overall_status"] == "pass"
    assert audit["warning_count"] == 0
    assert audit["zero_cost_contract"]["runtime_cost_usd"] == 0
    assert {check["area"] for check in audit["checks"]} == {
        "Demo artifact bundle",
        "Zero-cost artifact flags",
        "Free Streamlit deploy path",
        "Reviewer documentation",
        "Screenshot gallery",
        "Repository secret scan",
        "Public repo hygiene",
        "Dependency cost guard",
    }
    assert all(check["status"] == "pass" for check in audit["checks"])


def test_public_readiness_audit_formats_recruiter_readable_markdown() -> None:
    audit = build_public_readiness_audit(Path(__file__).resolve().parents[1])

    formatted = format_public_readiness_audit(audit)

    assert "# BatteryOps Public Readiness Audit" in formatted
    assert "Overall: PASS" in formatted
    assert "[PASS] Demo artifact bundle" in formatted
    assert "Runtime cost: $0" in formatted
    assert "External APIs: none" in formatted


def test_public_readiness_audit_cli_supports_json(capsys) -> None:  # type: ignore[no-untyped-def]
    repo_root = Path(__file__).resolve().parents[1]

    exit_code = main(["--repo-root", str(repo_root), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["overall_status"] == "pass"
    assert payload["zero_cost_contract"]["external_apis"] == "none"
