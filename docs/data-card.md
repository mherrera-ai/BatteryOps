# BatteryOps Data Card

## Source

BatteryOps uses public NASA battery telemetry archives:

- Randomized and Recommissioned Battery Dataset
- Randomized Battery Usage 1: Random Walk

Raw ZIPs and regenerated parquet are intentionally excluded from git. The public repo ships a compact validated demo bundle under `artifacts/demo/`.

## Current Public Bundle

| Item | Value |
| --- | ---: |
| Data source | `processed` |
| Assets | `26` |
| Cycle snapshots | `848` |
| Incident windows | `537` |
| Health index range | `0.0-100.0%` |
| Evidence source coverage | `100%` |

The bundle is validated from `training_manifest.json` before the app renders it. If validation fails, the app uses deterministic fallback telemetry and clearly labels the runtime source.

## Health Index

NASA source files expose capacity-like values with dataset-specific scaling. BatteryOps keeps raw capacity proxy values in the artifacts for provenance, but public charts use `health_index_pct`, a bounded per-asset normalization.

This avoids presenting source-dependent proxy values as physical battery capacity.

## Quality Checks

The saved data-quality report checks:

- Health index bounds
- Anomaly score bounds
- Nonnegative proxy RUL
- Incident rows available for report and retrieval context

See `artifacts/demo/data_quality_report.json` for the machine-readable version.

## Exclusions

The repo does not check in:

- Raw NASA ZIP archives
- Local processed parquet cache
- Virtual environments
- `node_modules`
- Hosted service config
- API keys or secrets

## Cost Profile

- Runtime cost: `$0`
- External APIs: `none`
- API keys: `none`
- Paid services: `none`
- Hosted database: `none`
