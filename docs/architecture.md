# Architecture

## Purpose

BatteryOps is a local-first analytics repository, not a hosted product stack. The public checkout is intentionally minimal: source, tests, docs, screenshots, and a validated demo bundle under `artifacts/demo/`. Raw ZIPs and generated parquet stay local and out of git, and the demo app does not read `data/processed/` directly at startup.

The saved bundle is a leave-one-asset-out degradation proxy over 26 assets and 848 cycles. Those numbers are demo provenance, not a benchmark, calibration result, or safety claim.

## Data Flow

1. Public NASA battery archives are downloaded manually into `data/raw/`.
1. `batteryops.data.preprocess` converts raw telemetry into normalized parquet outputs in local `data/processed/` storage.
1. `batteryops.features` derives cycle-level degradation and incident-window features.
1. `batteryops.models.train` fits baseline RUL and anomaly models and writes the review artifacts into `artifacts/demo/`.
1. `batteryops.reports.demo` validates the saved demo bundle by checking the manifest, metrics, report, timeline, and incident artifacts for consistency.
1. `batteryops.reports.demo` returns the first healthy local bundle it finds, preferring the current working directory bundle over the checkout copy.
1. If no healthy bundle is found, the app uses deterministic synthetic telemetry so the dashboard still launches.
1. `batteryops.streamlit_app` powers the dashboard from the validated bundle or the synthetic fallback. Local processed parquet is a regeneration input, not a runtime dependency for the public demo path.
1. `app/streamlit_app.py` provides the thin local wrapper used by the repo and screenshots.

## Bundle Validation

- A healthy demo bundle requires the manifest, metrics, report, timeline, incident cases, and model artifacts to all exist and agree.
- The manifest can record per-artifact SHA-256 hashes, sizes, and a bundle fingerprint so maintainers can verify the checked-in payload quickly.
- The loader is all-or-nothing. If validation fails, the app does not partially mix bundle artifacts with fresh fallback data.
- The sidebar surfaces the runtime source label so reviewers can tell whether the current session is using the checked-in demo bundle or the synthetic fallback.
- `artifacts/demo/training_manifest.json` is the primary provenance record for the saved bundle.
- If the bundle changes, refresh the README metrics block, the screenshot gallery, and any mirrored docs in the same change.

## Repo Signals

- Raw NASA ZIPs stay out of git.
- Processed parquet lives under local `data/processed/` storage, stays out of git, and is only used for preprocessing/training workflows.
- Demo artifacts live under `artifacts/demo/` for fast startup from a local checkout.
- Startup prefers a validated bundle from `artifacts/demo/`; if no validated bundle is found it uses deterministic synthetic telemetry so the app can still launch.
- `make screenshots` auto-starts the local app and refreshes the six-tab gallery against the current bundle.

## Module Map

- `src/batteryops/data/`: ingestion, validation, parquet persistence
- `src/batteryops/features/`: cycle features and incident extraction
- `src/batteryops/models/`: baseline training workflow
- `src/batteryops/retrieval/`: similar-case retrieval
- `src/batteryops/reports/`: report assembly and demo artifact loading
- `src/batteryops/dashboard.py`: UI-facing data shaping and Plotly figure construction
- `src/batteryops/eval/`: offline evaluation metrics
- `app/streamlit_app.py`: Streamlit wrapper used for local launches and screenshots

## Operating Notes

- Keep raw data out of git.
- Prefer simple baselines over complex stacks.
- Save processed outputs as parquet.
- Keep demo artifacts small so the app opens quickly.
- Treat the repository as an engineering demo, not production EV safety software.
- When you change bundle provenance, screenshot content, or fallback wording, update the public docs together so the story stays coherent.
