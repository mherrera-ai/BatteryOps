# Contributing

BatteryOps is a local-first Streamlit demo built around public NASA battery data. Keep changes honest, small enough to review, and aligned with the existing stack.

## Local Setup

Use the repo-local virtual environment if you already have one, or create a fresh one:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

If you plan to refresh screenshots, install the pinned Node tooling once:

```bash
npm ci
```

## What To Verify

Run the repo checks before opening a PR:

```bash
make check
```

If you changed any UI text, layout, screenshots, or screenshot capture logic, refresh the gallery too:

```bash
make screenshots
```

Useful local commands:

```bash
batteryops-demo
python3 -m batteryops.data.preprocess
python3 -m batteryops.models.train
```

## Scope Boundaries

Keep the public payload clean and professional.

- Do keep source under `src/batteryops/`, the Streamlit wrapper in `app/`, tests, docs, GitHub workflows, and the checked-in demo bundle in `artifacts/demo/` when it is still valid.
- Do not add raw NASA ZIPs, processed parquet caches, virtual environments, `node_modules/`, build artifacts, logs, or scratch files to git.
- Do not introduce React, FastAPI, hosted services, paid APIs, auth, cloud infra, or production safety claims.
- Prefer simple, reliable baselines over complex modeling changes unless there is a clear reason to do more.

## High-Signal Issues And PRs

Please keep issues and pull requests specific and actionable.

- Describe the user-visible behavior, the affected file or command, and the expected result.
- Include the exact local command you used, especially `make check` or `make screenshots`.
- If the change affects the demo bundle, screenshots, or README claims, update the docs in the same change.
- Avoid cosmetic churn that does not improve the reviewer experience.

## Repo Hygiene

When in doubt, favor excluding a file rather than publishing it. The repo should remain runnable from a fresh clone, but it should not depend on local-only data being checked in.
