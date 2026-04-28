# Architecture

## Purpose

BatteryOps is a zero-cost ML engineering showcase for local battery telemetry triage. It is intentionally scoped around public NASA data, reproducible local artifacts, and a Streamlit dashboard that recruiters can inspect without secrets, paid APIs, hosted databases, or backend infrastructure.

The saved bundle is a leave-one-asset-out degradation proxy over 26 assets and 848 cycles. Those numbers are demo provenance, not a benchmark, calibration result, or safety claim.

## Data Flow

1. Public NASA battery archives are downloaded manually into `data/raw/`.
1. `batteryops.data.preprocess` converts raw telemetry into normalized parquet outputs in local `data/processed/` storage.
1. `batteryops.features` derives cycle-level telemetry features and incident windows.
1. `batteryops.models.train` fits local scikit-learn RUL, anomaly, and retrieval baselines.
1. Training writes the demo bundle: models, cycle predictions, incident cases, metrics, incident report, model card, data-quality report, evaluation report, and manifest provenance.
1. `batteryops.reports.demo` validates the manifest, hashes, schema, report, timeline, metrics, and zero-cost metadata before the app consumes the bundle.
1. If no healthy bundle is found, the app uses deterministic synthetic telemetry so local launch still works.
1. `batteryops.streamlit_app` renders the Fleet Cockpit, Asset Replay, Incident Evidence, Similar Cases, Model Evaluation, and Data & Provenance tabs.
1. The Fleet Cockpit summarizes the full bundle through a risk map, ranked priority queue, cockpit decision ledger, risk concentration chart, and selected-asset risk-driver breakdown before drilling into a selected asset timeline.
1. `batteryops.audit` validates the public-readiness contract across artifacts, zero-cost flags, docs, screenshots, dependency manifests, and repo hygiene.
1. `app/streamlit_app.py` is the thin Streamlit Cloud/local wrapper.

## Bundle Contract

- A healthy bundle requires every file in `DEMO_ARTIFACT_FILENAMES`, including `model_card.json`, `data_quality_report.json`, and `evaluation_report.json`.
- The timeline must include `health_index_pct`, a bounded 0..100 reviewer-facing signal derived from source capacity-like values.
- The loader is all-or-nothing. If validation fails, the app does not mix partial bundle artifacts with fallback data.
- `training_manifest.json` records artifact paths, sizes, SHA-256 hashes, bundle fingerprint, metrics, report ID, runtime provenance, and schema version.
- Cost-profile fields in model/data/evaluation artifacts must not require external APIs, API keys, or paid services.

## Runtime Boundaries

- The app does not read `data/processed/` directly at startup.
- Local processed parquet is a regeneration input, not a runtime dependency for the public demo path.
- Raw NASA ZIPs and regenerated parquet stay out of git.
- The checked-in bundle under `artifacts/demo/` is the review payload for fast clone-and-run and free Streamlit hosting.
- `.streamlit/config.toml` only controls presentation and Streamlit toolbar behavior; it does not introduce any service dependency.

## Module Map

- `src/batteryops/data/`: NASA archive parsing, normalization, parquet persistence
- `src/batteryops/features/`: cycle features, health inputs, incident extraction
- `src/batteryops/models/`: baseline training, evaluation artifacts, model/data cards
- `src/batteryops/retrieval/`: nearest-neighbor similar-incident search
- `src/batteryops/reports/`: incident report assembly and demo bundle validation
- `src/batteryops/audit.py`: local public-readiness audit for recruiters, CI, and pre-publish checks
- `src/batteryops/dashboard.py`: data shaping, fleet prioritization, risk-driver breakdowns, artifact inventory formatting, and Plotly figure construction
- `src/batteryops/streamlit_app.py`: recruiter-facing dashboard
- `app/streamlit_app.py`: deployment/local wrapper

## Operating Notes

- Keep the public repo zero-cost: no paid APIs, API keys, hosted vector stores, hosted databases, auth providers, or metered services.
- Keep claims honest: proxy RUL and alert metrics are engineering evidence, not production safety validation.
- When the bundle changes, update `README.md`, this architecture page, `docs/model-card.md`, `docs/data-card.md`, and screenshots together.
