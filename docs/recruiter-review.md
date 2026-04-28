# Recruiter Review Guide

BatteryOps is designed to be reviewed quickly from GitHub. It is a local-first ML engineering demo, not a paid-service integration or a mock dashboard.

## Fast Path

1. Read the first screen of `README.md` for the project contract and zero-cost boundary.
1. In the app, scan the proof strip and Fleet Cockpit `Triage handoff` table before opening the charts.
1. Open the screenshot gallery in `docs/screenshots/` to see the six-tab Streamlit workflow.
1. Run `batteryops-demo` to inspect the validated dashboard locally.
1. Open `artifacts/demo/training_manifest.json` to verify artifact hashes, bundle fingerprinting, and schema versioning.
1. Run `batteryops-audit` to validate the public-readiness contract: artifact bundle, docs, screenshots, zero-cost flags, and dependency guard.
1. Run `make check` to validate linting, typing, tests, the audit, and artifact contracts.
1. Run `make deploy-check` to validate the free Streamlit Community Cloud dependency path.

## What This Project Demonstrates

| Signal | Where to Look |
| --- | --- |
| ML workflow ownership | `src/batteryops/data/`, `src/batteryops/features/`, `src/batteryops/models/` |
| Local artifact validation | `src/batteryops/reports/demo.py`, `artifacts/demo/training_manifest.json` |
| Product-quality dashboarding | First-screen proof strip, Fleet Cockpit risk map, cockpit decision ledger, risk concentration chart, `Triage handoff`, selected-asset risk drivers, `src/batteryops/streamlit_app.py`, `src/batteryops/dashboard.py` |
| Evaluation discipline | `artifacts/demo/evaluation_report.json`, `docs/model-card.md` |
| Data-quality framing | `artifacts/demo/data_quality_report.json`, `docs/data-card.md` |
| Retrieval and evidence | Similar Cases and Incident Evidence tabs |
| Public readiness audit | `batteryops-audit`, `src/batteryops/audit.py`, Data & Provenance tab |
| Public GitHub safety | Audit secret-pattern scan, `.gitignore`, zero-cost manifests, no checked-in raw/private data |
| Public repo hygiene | `.github/`, `SECURITY.md`, `CONTRIBUTING.md`, `Makefile` |

## Zero-Cost Boundary

The public runtime uses checked-in artifacts and local Python packages only.

- Runtime cost: `$0`
- External APIs: `none`
- API keys or secrets: `none`
- Hosted databases: `none`
- Paid model inference: `none`
- Auth providers: `none`

## Responsible Scope

The RUL and alert outputs are proxy evidence for the checked-in demo bundle. They are useful for demonstrating ML engineering, evaluation, dashboarding, and artifact provenance; they are not production EV safety claims or calibrated risk predictions.
