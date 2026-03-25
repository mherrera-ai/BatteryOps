# Screenshot Guide

This folder holds the public-facing Streamlit screenshots used in the README and related docs. The images are captured from the local demo app and should reflect the same validated runtime source used by the walkthrough when the saved bundle is available.

Current screenshots, in app order:

- `overview.png`: dashboard landing view with the asset snapshot and health trend
- `live-telemetry-replay.png`: replay controls, cursor state, and telemetry chart
- `anomaly-timeline.png`: threshold tuning and triage queue view
- `incident-report.png`: saved incident evidence and heuristic summary
- `similar-cases.png`: nearest-neighbor retrieval scatter and case table
- `evaluation-dashboard.png`: saved proxy metrics and evaluation charts

The capture script keeps the story consistent by focusing on fixed, valid assets per tab. Time-series tabs use `battery36`, and report/retrieval/evaluation tabs use `battery50` in the checked-in bundle. If those IDs are not available in your local bundle, update the script to match your current asset set.

Preferred refresh path:

```bash
npm ci
make screenshots
```

That target auto-starts the local headless app through `batteryops.cli`, waits for the dashboard to be ready, captures all six screenshots into `docs/screenshots/`, and shuts the app down afterward. It uses the same launch path as `make demo` and `make demo-headless`, so the screenshots stay aligned with the local app entrypoint.

If the app is showing synthetic fallback telemetry, treat the capture as invalid for the public gallery and fix the bundle first. The gallery is only representative when the validated demo bundle is the active source.

If you need manual control over the app process or output location, the underlying script still supports environment overrides:

```bash
BATTERYOPS_APP_COMMAND="python3 -m batteryops.cli --server.headless=true --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats=false --server.fileWatcherType=none --server.runOnSave=false" \
BATTERYOPS_APP_URL=http://127.0.0.1:8501 \
BATTERYOPS_SCREENSHOT_DIR=docs/screenshots \
node docs/screenshots/capture.mjs
```

The screenshot toolchain is pinned in `package.json` and `package-lock.json`. Install it only when refreshing screenshots; it is not required for normal repo use.

```bash
npm ci
```

After capture, remove the local `node_modules` directory if you do not need it for other Node-based tooling:

```bash
rm -rf node_modules
```

The current SVG source assets live in `docs/assets/`:

- `batteryops-hero.svg`
- `batteryops-pipeline.svg`
- `batteryops-proof-card.svg`

Recommended root-relative paths:

- `docs/screenshots/overview.png`
- `docs/screenshots/live-telemetry-replay.png`
- `docs/screenshots/anomaly-timeline.png`
- `docs/screenshots/incident-report.png`
- `docs/screenshots/similar-cases.png`
- `docs/screenshots/evaluation-dashboard.png`

Related local commands:

- `make demo` opens the app with the checked-in demo bundle when available.
- `make demo-headless` launches the same app path without a browser window, which is useful for local automation.
- `make check` validates the code and test suite before you refresh screenshots.

If the bundle or app copy changes, recapture this directory and update the README at the same time so the public docs do not drift.
