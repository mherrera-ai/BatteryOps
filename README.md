# BatteryOps

[![CI](https://github.com/mherrera-ai/Batteryops/actions/workflows/ci.yml/badge.svg)](https://github.com/mherrera-ai/Batteryops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg)](pyproject.toml)

BatteryOps is a local-first Streamlit demo for telemetry triage and predictive maintenance on public NASA battery data. It turns a public dataset into a reviewable analyst workflow: preprocess local parquet, train simple baselines, and serve a validated demo bundle without backend infrastructure, auth, or paid services.

The fast path is the checked-in demo bundle in `artifacts/demo/`. If that bundle cannot be validated, the app still launches with deterministic synthetic telemetry so the repository remains runnable, but the reviewer-facing evidence comes from the saved bundle.

![BatteryOps hero](docs/assets/batteryops-hero.svg)

## 5-Minute Review

If you only skim the repo, start here:

1. `README.md` for the public story and launch path.
1. `docs/architecture.md` for the data flow, bundle validation rules, and runtime boundaries.
1. `docs/screenshots/` for the six-tab Streamlit walkthrough captured from the checked-in demo bundle.
1. `artifacts/demo/training_manifest.json` for artifact provenance, fingerprinting, and bundle metadata.
1. `make check` and `batteryops-demo` for the main verification path.

## Start Here

- [Quick start](#quick-start)
- [First launch expectations](#first-launch-expectations)
- [Demo flow](#demo-flow)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Current demo bundle metrics](#current-demo-bundle-metrics)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Data prep](#data-prep)
- [Model training](#model-training)
- [Limitations](#limitations)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## What It Is

An intentionally scoped portfolio project: a credible, reviewable workflow from raw public data to a local analyst dashboard.

The public repo is optimized for clone-and-run review, with the checked-in demo bundle, screenshots, and docs forming the visible evidence trail.

It is not production EV safety software, and it does not claim calibrated safety performance.

## At a Glance

| Item | Status |
| --- | --- |
| Stack | `Streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `plotly` |
| Packaging | Code lives under `src/batteryops`; `app/streamlit_app.py` is the thin local wrapper |
| Repo payload | Source, tests, docs, screenshots, artifact manifest, and the saved demo bundle |
| Local data | `data/raw/` and `data/processed/` stay out of git |
| Demo path | Prefers a validated bundle in `artifacts/demo/`; otherwise falls back to deterministic synthetic telemetry |
| Bundle provenance | `artifacts/demo/training_manifest.json` records artifact paths, SHA-256 hashes, bundle fingerprint, and runtime/training provenance for the current `26`-asset, `848`-cycle, `537`-incident bundle |
| Verification | `make check` is the main repo-level signal; `make screenshots` refreshes the gallery |

## Quick Start

From a fresh checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
make check
batteryops-demo
```

If you already have the virtual environment active, launch directly with:

```bash
batteryops-demo
```

`batteryops-demo` is the saved-bundle launch path documented here; `batteryops` is the shorter alias.

Run `batteryops-demo` from the repository root, or from another workspace that contains local
`artifacts/demo/`. Local `data/processed/` parquet is optional and only used for preprocessing
or retraining workflows that regenerate the demo bundle.

## First Launch Expectations

On startup, the app resolves the first healthy local bundle it can validate. In practice, that means:

1. Prefer the checked-in `artifacts/demo/` bundle when its manifest, metrics, report, timeline, and incident artifacts all agree.
1. Fall back to deterministic synthetic telemetry when the validated bundle is missing or incomplete.
1. Surface the active runtime source in the sidebar so reviewers can tell which path they are seeing.
1. Do not read `data/processed/` directly at app startup; local parquet only matters when you are regenerating the bundle.

Charts come from persisted cycle summaries. The capacity and RUL traces are proxy metrics for transparent triage storytelling, not calibrated physical-grade estimates.

## What Recruiters See

The public payload is intentionally small and honest. A recruiter should be able to verify three things quickly:

1. The project is real code, not a mock UI: the Streamlit app runs locally and the tests exercise the documented entry points.
1. The results are reproducible: the checked-in demo bundle is validated from hashes and manifest metadata before the app uses it.
1. The scope is responsible: the repo stays local-first, avoids hosted dependencies, and states its limits clearly.

## Demo Flow

The app opens on six tabs, in the same order used by the screenshot flow:

1. `Overview`: asset snapshot, health trend, and triage note
1. `Live Telemetry Replay`: cycle-by-cycle replay with `1x`, `4x`, and `10x` speed
1. `Anomaly Timeline`: threshold control and triage queue over replay windows
1. `Incident Report`: deterministic report assembled from saved evidence
1. `Similar Cases`: nearest-neighbor retrieval for comparable incidents
1. `Evaluation Dashboard`: saved proxy metrics and evaluation framing

## Screenshots

The captures below are refreshed from the checked-in demo bundle and should match the current six-tab app copy and layout.

![BatteryOps overview screenshot](docs/screenshots/overview.png)
![BatteryOps replay screenshot](docs/screenshots/live-telemetry-replay.png)
![BatteryOps anomaly timeline screenshot](docs/screenshots/anomaly-timeline.png)
![BatteryOps incident report screenshot](docs/screenshots/incident-report.png)
![BatteryOps similar cases screenshot](docs/screenshots/similar-cases.png)
![BatteryOps evaluation dashboard screenshot](docs/screenshots/evaluation-dashboard.png)

Open full-size captures directly if preview thumbnails are hard to read:

- [Overview](docs/screenshots/overview.png)
- [Live Telemetry Replay](docs/screenshots/live-telemetry-replay.png)
- [Anomaly Timeline](docs/screenshots/anomaly-timeline.png)
- [Incident Report](docs/screenshots/incident-report.png)
- [Similar Cases](docs/screenshots/similar-cases.png)
- [Evaluation Dashboard](docs/screenshots/evaluation-dashboard.png)

Refresh the gallery with `make screenshots`. That target auto-starts the local headless app, captures all six tabs into `docs/screenshots/`, and shuts the app down when capture completes.

On a fresh clone, run `npm ci` once before the first screenshot refresh so the pinned Playwright tooling from `package-lock.json` is available locally.

If the bundle, runtime source label, or tab copy changes, refresh the gallery and the README wording together so the public story stays aligned.

## Architecture

![BatteryOps workflow](docs/assets/batteryops-pipeline.svg)

Current flow:

1. Manually drop a supported NASA archive into `data/raw/`.
1. Run preprocessing into local parquet at `data/processed/`.
1. Train feature and model baselines into `artifacts/demo/`.
1. Validate the saved demo bundle before the app consumes it.
1. Load deterministic incident retrieval and report artifacts at startup.
1. Render the dashboard from the persisted bundle when validation succeeds.
1. Fall back to synthetic telemetry if the validated bundle is unavailable.

For module-level detail and bundle validation rules, see [docs/architecture.md](docs/architecture.md).

## Current Demo Bundle Metrics

Metrics below are from this workspace and are used as evidence only.

| Metric | Value |
| --- | ---: |
| Data source | `processed` |
| Evaluation mode | `leave-one-asset-out degradation proxy` |
| Assets in artifact bundle | `26` |
| Cycle count | `848` |
| Incident case count | `537` |
| Alert lead time | `0.4615` cycles |
| Alert precision | `67.31%` |
| Alert recall | `19.55%` |
| False positive rate | `16.4%` |
| RUL proxy MAE | `5.483` cycles |
| Evidence source coverage | `100%` |

These are proxy metrics for the current bundle, not deployment-ready performance claims. When bundle metrics change, regenerate the demo artifact bundle, refresh screenshots, and update this table together.

## Verification

The main repo-level signal is:

```bash
make check
```

Run `make screenshots` when you change screenshot capture, the validated bundle, or any wording that affects the gallery story.

For a quick local provenance check on the checked-in bundle:

```bash
python3 - <<'PY'
from batteryops.reports.demo import inspect_demo_bundle

status = inspect_demo_bundle()
print(status.healthy)
print(status.bundle_fingerprint)
for artifact in status.artifact_inventory:
    print(artifact["name"], artifact["sha256"])
PY
```

When bundle provenance, screenshot copy, or launch behavior changes, update `README.md`, `docs/architecture.md`, and `docs/screenshots/README.md` in the same change to keep the docs from drifting apart.

## Troubleshooting

- If `batteryops-demo` opens but the sidebar says `synthetic demo fallback`, the validated bundle was not found or did not pass validation. Check `artifacts/demo/training_manifest.json`, `demo_metrics.json`, `demo_report.json`, `demo_cycle_predictions.parquet`, `demo_incident_cases.parquet`, and the model files for completeness and consistency. If you launched from another directory, rerun from the repo root so the command can pick up the local bundle.
- If you expected the checked-in bundle but see fallback telemetry, regenerate the demo bundle from processed parquet with `python3 -m batteryops.models.train`, then rerun `make screenshots` if the gallery story changed.
- If the console command is missing after a fresh checkout, rerun the editable install with `python3 -m pip install -e ".[dev]"` inside the project virtual environment.
- If screenshots no longer match the app, recapture them after the bundle or tab copy change rather than treating stale PNGs as current evidence.

## Data Prep

Supported local source archives:

- `data/raw/nasa_rr_battery.zip`
- `data/raw/nasa_rw1_battery.zip`

Official sources:

- [Randomized and Recommissioned Battery Dataset](https://data.nasa.gov/dataset/randomized-and-recommissioned-battery-dataset/resource/cdea0a46-2c8d-4192-bbe7-8fbebfcfdb31)
- [Randomized Battery Usage 1: Random Walk](https://data.nasa.gov/dataset/randomized-battery-usage-1-random-walk/resource/ddffaa98-4c93-464f-aef1-b68eae64096f)

```bash
python3 -m batteryops.data.preprocess
```

The downloaded zip and local parquet are cache artifacts only and are intentionally out of the public payload.

The checked-in public demo does not need those local caches to launch; they are only inputs for rebuilding the saved bundle.

## Model Training

```bash
python3 -m batteryops.models.train
```

If parsed parquet exists, training consumes it; otherwise it uses deterministic synthetic inputs to regenerate a complete bundle so the demo remains runnable.

`training_manifest.json` records the emitted artifact inventory, bundle fingerprint, runtime/package versions, and the input/training recipe used to build the bundle. That makes it easier to verify a checked-in bundle quickly without retraining.

The training step writes:

- `artifacts/demo/rul_model.joblib`
- `artifacts/demo/anomaly_model.joblib`
- `artifacts/demo/incident_retrieval.joblib`
- `artifacts/demo/demo_cycle_predictions.parquet`
- `artifacts/demo/demo_incident_cases.parquet`
- `artifacts/demo/demo_metrics.json`
- `artifacts/demo/demo_report.json`
- `artifacts/demo/training_manifest.json`

## Limitations

- This repo is a demo with reproducibility and interview-ready clarity as its target, not a production benchmark.
- The public payload ships review artifacts, not the full raw/processed dataset cache.
- Evaluation values are local evidence for this bundle, not deployment-ready safety or regulatory claims.
- The report confidence score is deterministic and narrative; it is not a calibrated probability.
- Capacity and RUL traces are proxy metrics for triage storytelling, not physical-grade estimates.

## Publishing Notes

This repository is meant to be public. The files checked into `artifacts/demo/` are the intentionally saved reviewer bundle, while raw archives and regenerated parquet stay local-only. If you refresh the bundle or screenshots, update the README and docs together so the public story stays aligned with the app.

For public repo housekeeping, see [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
