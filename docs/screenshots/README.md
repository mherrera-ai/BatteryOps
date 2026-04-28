# Screenshot Guide

This folder holds the public-facing Streamlit screenshots used in the README. The gallery should always be captured from the validated checked-in bundle, not from synthetic fallback telemetry.

Current screenshots, in app order:

- `fleet-cockpit.png`: fleet risk map, priority queue, cockpit decision ledger, risk concentration, selected-asset risk drivers, health trend, RUL proxy, and triage note
- `asset-replay.png`: replay controls, cursor state, and telemetry chart
- `incident-evidence.png`: incident report, threshold context, source evidence, and diagnostics
- `similar-cases.png`: nearest-neighbor retrieval distance chart and case table
- `model-evaluation.png`: metrics, confusion matrix, RUL scatter, threshold tradeoff, and feature signals
- `data-provenance.png`: data quality, artifact inventory, model-card summary, limits, and zero-cost contract

Preferred refresh path:

```bash
npm ci
make screenshots
```

That target auto-starts the local headless app through `batteryops.cli`, waits for the dashboard to be ready, captures all six screenshots into `docs/screenshots/`, and shuts the app down afterward. It uses the same launch path as `make demo` and `make demo-headless`.

If the app is showing synthetic fallback telemetry, treat the capture as invalid for the public gallery and fix the bundle first.

The underlying script supports environment overrides:

```bash
BATTERYOPS_APP_COMMAND="python3 -m batteryops.cli --server.headless=true --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats=false --server.fileWatcherType=none --server.runOnSave=false" \
BATTERYOPS_APP_URL=http://127.0.0.1:8501 \
BATTERYOPS_SCREENSHOT_DIR=docs/screenshots \
node docs/screenshots/capture.mjs
```

The screenshot toolchain is pinned in `package.json` and `package-lock.json`. It is not required for normal app use.

Related local commands:

- `make demo` opens the app with the checked-in demo bundle when available.
- `make demo-headless` launches the same app path without a browser window.
- `make check` validates the code and test suite.
- `make deploy-check` validates the zero-cost deployment dependency path.

If the bundle, app copy, or tab order changes, recapture this directory and update the README at the same time.
