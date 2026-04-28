"""Public-readiness audit for the BatteryOps portfolio repository."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from batteryops.reports.demo import DEMO_ARTIFACT_FILENAMES, inspect_demo_bundle

AUDIT_SCHEMA_VERSION = 1
EXPECTED_SCREENSHOTS = (
    "fleet-cockpit.png",
    "asset-replay.png",
    "incident-evidence.png",
    "similar-cases.png",
    "model-evaluation.png",
    "data-provenance.png",
)
FORBIDDEN_DEPENDENCY_TOKENS = (
    "openai",
    "anthropic",
    "pinecone",
    "weaviate",
    "supabase",
    "firebase",
    "stripe",
)
ZERO_COST_PROFILE_FLAGS = (
    "uses_external_apis",
    "requires_api_keys",
    "requires_paid_services",
)
SECRET_SCAN_EXCLUDED_DIRS = {
    ".deploy-check",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
SECRET_SCAN_TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_SCAN_TEXT_FILENAMES = {
    ".gitignore",
    "LICENSE",
    "Makefile",
    "requirements.txt",
}
SECRET_SCAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{30,}\b")),
    ("OpenAI-style secret key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
)


def build_public_readiness_audit(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Return a machine-readable audit of the public repo review contract."""
    root = _resolve_repo_root(repo_root)
    checks = [
        _audit_demo_bundle(),
        _audit_zero_cost_artifacts(root),
        _audit_free_deploy_path(root),
        _audit_reviewer_docs(root),
        _audit_screenshots(root),
        _audit_public_secret_scan(root),
        _audit_community_files(root),
        _audit_dependency_guard(root),
    ]
    warning_count = sum(1 for check in checks if check["status"] != "pass")
    pass_count = len(checks) - warning_count
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "overall_status": "pass" if warning_count == 0 else "warn",
        "pass_count": pass_count,
        "warning_count": warning_count,
        "repo_root": str(root),
        "checks": checks,
        "zero_cost_contract": {
            "runtime_cost_usd": 0,
            "external_apis": "none",
            "api_keys_or_secrets": "none",
            "hosted_database": "none",
            "paid_model_inference": "none",
            "metered_cloud_service": "none",
        },
    }


def format_public_readiness_audit(audit: dict[str, Any]) -> str:
    """Format an audit payload as concise terminal-friendly Markdown."""
    lines = [
        "# BatteryOps Public Readiness Audit",
        "",
        f"Overall: {str(audit.get('overall_status', 'warn')).upper()}",
        (
            f"Checks: {int(audit.get('pass_count', 0))} passed, "
            f"{int(audit.get('warning_count', 0))} warnings"
        ),
        "",
        "## Checks",
    ]
    for check in audit.get("checks", []):
        if not isinstance(check, dict):
            continue
        status = str(check.get("status", "warn")).upper()
        area = str(check.get("area", "Unknown"))
        evidence = str(check.get("evidence", "No evidence recorded."))
        lines.append(f"- [{status}] {area}: {evidence}")
    lines.extend(
        [
            "",
            "## Zero-Cost Contract",
            "- Runtime cost: $0",
            "- External APIs: none",
            "- API keys or secrets: none",
            "- Hosted database: none",
            "- Paid model inference: none",
            "- Metered cloud service: none",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the public-readiness audit from the command line."""
    parser = argparse.ArgumentParser(
        description="Validate BatteryOps public repo readiness without paid services."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root to audit. Defaults to the current BatteryOps checkout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the audit payload as JSON instead of Markdown.",
    )
    args = parser.parse_args(argv)
    audit = build_public_readiness_audit(args.repo_root)
    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print(format_public_readiness_audit(audit), end="")
    return 0 if audit["overall_status"] == "pass" else 1


def _resolve_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _audit_demo_bundle() -> dict[str, str]:
    status = inspect_demo_bundle()
    if status.healthy and status.bundle_fingerprint:
        return _check(
            "Demo artifact bundle",
            True,
            f"Validated bundle fingerprint {status.bundle_fingerprint[:12]}",
            status.reason,
        )
    return _check(
        "Demo artifact bundle",
        False,
        status.reason,
        "The dashboard will fall back if the bundle cannot be validated.",
    )


def _audit_zero_cost_artifacts(root: Path) -> dict[str, str]:
    demo_dir = root / "artifacts" / "demo"
    required_keys = ("model_card", "data_quality_report", "evaluation_report")
    violations: list[str] = []
    checked = 0
    for key in required_keys:
        filename = DEMO_ARTIFACT_FILENAMES[key]
        payload = _load_json(demo_dir / filename)
        if payload is None:
            violations.append(f"{filename} missing or invalid")
            continue
        checked += 1
        cost_profile = payload.get("cost_profile", {})
        if not isinstance(cost_profile, dict):
            violations.append(f"{filename} missing cost_profile")
            continue
        if float(cost_profile.get("runtime_cost_usd", -1)) != 0:
            violations.append(f"{filename} runtime_cost_usd is not 0")
        for flag in ZERO_COST_PROFILE_FLAGS:
            if bool(cost_profile.get(flag)):
                violations.append(f"{filename} sets {flag}=true")
    return _check(
        "Zero-cost artifact flags",
        not violations and checked == len(required_keys),
        f"{checked} reports declare $0 runtime",
        "; ".join(violations) if violations else "model/data/evaluation reports agree",
    )


def _audit_free_deploy_path(root: Path) -> dict[str, str]:
    requirements = _read_text(root / "requirements.txt").strip()
    streamlit_config = _read_text(root / ".streamlit" / "config.toml")
    app_wrapper = root / "app" / "streamlit_app.py"
    ok = (
        requirements == "-e ."
        and app_wrapper.exists()
        and 'toolbarMode = "minimal"' in streamlit_config
        and "gatherUsageStats = false" in streamlit_config
    )
    return _check(
        "Free Streamlit deploy path",
        ok,
        "requirements.txt and Streamlit config validated",
        "Expected requirements.txt to be '-e .' with a minimal no-telemetry Streamlit config.",
    )


def _audit_reviewer_docs(root: Path) -> dict[str, str]:
    doc_paths = (
        root / "README.md",
        root / "docs" / "recruiter-review.md",
        root / "docs" / "model-card.md",
        root / "docs" / "data-card.md",
        root / "docs" / "architecture.md",
    )
    missing = [path.name for path in doc_paths if not path.exists()]
    readme = _read_text(root / "README.md")
    required_readme_phrases = (
        "## 5-Minute Review",
        "## Zero-Cost Contract",
        "## Current Demo Bundle Metrics",
        "batteryops-audit",
    )
    missing_phrases = [phrase for phrase in required_readme_phrases if phrase not in readme]
    ok = not missing and not missing_phrases
    detail = []
    if missing:
        detail.append("Missing docs: " + ", ".join(missing))
    if missing_phrases:
        detail.append("README missing: " + ", ".join(missing_phrases))
    return _check(
        "Reviewer documentation",
        ok,
        f"{len(doc_paths) - len(missing)} reviewer docs with fast-path guidance",
        "; ".join(detail) if detail else "README, recruiter guide, model card, and data card align",
    )


def _audit_screenshots(root: Path) -> dict[str, str]:
    screenshot_dir = root / "docs" / "screenshots"
    missing = [
        filename
        for filename in EXPECTED_SCREENSHOTS
        if not _file_has_content(screenshot_dir / filename)
    ]
    return _check(
        "Screenshot gallery",
        not missing,
        f"{len(EXPECTED_SCREENSHOTS) - len(missing)} checked-in dashboard screenshots",
        "Missing or empty screenshots: " + ", ".join(missing)
        if missing
        else "six-tab gallery is ready for GitHub review",
    )


def _audit_public_secret_scan(root: Path) -> dict[str, str]:
    findings: list[str] = []
    scanned_count = 0
    for path in _iter_secret_scan_files(root):
        relative_path = path.relative_to(root)
        if path.name.startswith(".env") and path.name != ".env.example":
            findings.append(f"{relative_path} (local environment file)")
            continue
        text = _read_text(path)
        if not text:
            continue
        scanned_count += 1
        for label, pattern in SECRET_SCAN_PATTERNS:
            if pattern.search(text):
                findings.append(f"{relative_path} ({label})")
                break

    return _check(
        "Repository secret scan",
        not findings,
        f"{scanned_count} public text files scanned",
        "Potential secrets found: " + ", ".join(findings)
        if findings
        else "no secret-like tokens, local env files, or private key blocks found",
    )


def _audit_community_files(root: Path) -> dict[str, str]:
    required_paths = (
        root / "CONTRIBUTING.md",
        root / "SECURITY.md",
        root / ".github" / "workflows" / "ci.yml",
        root / ".github" / "workflows" / "release.yml",
        root / ".github" / "pull_request_template.md",
        root / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml",
        root / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml",
        root / ".github" / "dependabot.yml",
    )
    missing = [str(path.relative_to(root)) for path in required_paths if not path.exists()]
    return _check(
        "Public repo hygiene",
        not missing,
        f"{len(required_paths) - len(missing)} community and CI files present",
        "Missing files: " + ", ".join(missing)
        if missing
        else "community files, CI, release smoke test, and Dependabot are present",
    )


def _audit_dependency_guard(root: Path) -> dict[str, str]:
    dependency_text = "\n".join(
        _read_text(path).lower()
        for path in (root / "pyproject.toml", root / "requirements.txt", root / "package.json")
    )
    forbidden = [token for token in FORBIDDEN_DEPENDENCY_TOKENS if token in dependency_text]
    return _check(
        "Dependency cost guard",
        not forbidden,
        "No paid-service SDKs in manifests",
        "Forbidden dependency tokens found: " + ", ".join(forbidden)
        if forbidden
        else "Python and Node manifests stay local-first",
    )


def _check(area: str, passed: bool, evidence: str, detail: str) -> dict[str, str]:
    return {
        "area": area,
        "status": "pass" if passed else "warn",
        "evidence": evidence,
        "detail": detail,
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _file_has_content(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _iter_secret_scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in SECRET_SCAN_EXCLUDED_DIRS for part in relative_parts):
            continue
        if not path.is_file():
            continue
        if (
            path.name in SECRET_SCAN_TEXT_FILENAMES
            or path.suffix.lower() in SECRET_SCAN_TEXT_SUFFIXES
            or path.name.startswith(".env")
        ):
            files.append(path)
    return files


if __name__ == "__main__":
    raise SystemExit(main())
