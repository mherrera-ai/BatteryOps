# BatteryOps Model Card

## Intended Use

BatteryOps is a portfolio-grade ML engineering demo for battery telemetry triage. It helps reviewers inspect a complete local workflow: preprocessing, feature extraction, baseline modeling, anomaly scoring, incident retrieval, report generation, evaluation, and artifact validation.

It is not production EV safety software and does not claim calibrated safety performance.

## Models

| Component | Estimator | Purpose |
| --- | --- | --- |
| RUL proxy | `HistGradientBoostingRegressor` | Predict a degradation-threshold horizon in cycles |
| Anomaly score | `IsolationForest` | Rank cycle snapshots for inspect-soon triage |
| Similar cases | `NearestNeighbors` | Retrieve comparable incident windows from saved history |

All components run locally through scikit-learn/joblib. No external API, API key, paid inference, hosted vector store, or database is used.

## Current Evaluation

| Metric | Value |
| --- | ---: |
| Evaluation mode | `leave-one-asset-out degradation proxy` |
| Assets | `26` |
| Cycles | `848` |
| Incident cases | `537` |
| Alert precision | `67.31%` |
| Alert recall | `19.55%` |
| False positive rate | `16.4%` |
| RUL proxy MAE | `5.483` cycles |

The dashboard exposes the confusion matrix, RUL scatter, threshold tradeoff, per-asset error, and model-input signal ranking from `artifacts/demo/evaluation_report.json` and `artifacts/demo/model_card.json`.

## Feature Set

The baseline models use cycle-level telemetry summaries such as voltage range, temperature maximum/mean, absolute current maximum, throughput, internal resistance proxy, incident-window counts, and reference-cycle flags.

The feature ranking in the public artifact is deterministic portfolio evidence, not a claim of causal battery physics.

## Limitations

- Metrics are proxy evidence for the checked-in bundle.
- RUL is a degradation-threshold horizon proxy, not a calibrated lifetime forecast.
- The anomaly threshold is heuristic and selected for inspect-queue storytelling.
- The incident report confidence score is deterministic and narrative, not a probability.
- Public charts use `health_index_pct` because source capacity-like values are not consistently physical across NASA datasets.

## Cost Profile

- Runtime cost: `$0`
- External APIs: `none`
- API keys: `none`
- Paid services: `none`
- Hosted database: `none`
