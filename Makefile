.PHONY: install lint format typecheck test audit check deploy-check demo demo-headless screenshots

ROOT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
RUN_IN_ROOT := cd "$(ROOT_DIR)" &&

VENV_PYTHON := $(firstword $(wildcard .venv/bin/python) $(wildcard .venv/bin/python3))
PYTHON ?= $(if $(VENV_PYTHON),$(VENV_PYTHON),python3)
DEMO_PORT ?= 8501
DEMO_HOST ?= 127.0.0.1
DEMO_ARGS ?=
STREAMLIT_ARGS := --server.port $(DEMO_PORT) --server.address $(DEMO_HOST)
STREAMLIT_QUIET_ARGS := --browser.gatherUsageStats=false --server.fileWatcherType=none --server.runOnSave=false
DEMO_CMD := $(PYTHON) -m batteryops.cli $(STREAMLIT_ARGS) $(DEMO_ARGS)
SCREENSHOT_DIR ?= $(ROOT_DIR)/docs/screenshots
SCREENSHOT_APP_CMD ?= $(PYTHON) -m batteryops.cli $(STREAMLIT_ARGS) --server.headless=true $(STREAMLIT_QUIET_ARGS)

install:
	$(RUN_IN_ROOT) $(PYTHON) -m pip install -e ".[dev]"

lint:
	$(RUN_IN_ROOT) $(PYTHON) -m ruff check .

format:
	$(RUN_IN_ROOT) $(PYTHON) -m ruff format .

typecheck:
	$(RUN_IN_ROOT) $(PYTHON) -m mypy src

test:
	$(RUN_IN_ROOT) $(PYTHON) -m pytest

audit:
	$(RUN_IN_ROOT) $(PYTHON) -m batteryops.audit

check: lint typecheck test audit

deploy-check:
	$(RUN_IN_ROOT) rm -rf .deploy-check
	$(RUN_IN_ROOT) python3 -m venv .deploy-check
	$(RUN_IN_ROOT) .deploy-check/bin/python -m pip install --upgrade pip
	$(RUN_IN_ROOT) .deploy-check/bin/python -m pip install -r requirements.txt
	$(RUN_IN_ROOT) .deploy-check/bin/python -m pip check
	$(RUN_IN_ROOT) .deploy-check/bin/python -c "from batteryops.dashboard import load_dashboard_data; from batteryops.reports.demo import inspect_demo_bundle; status = inspect_demo_bundle(); assert status.healthy, status.reason; data = load_dashboard_data(); assert data.model_card.get('cost_profile', {}).get('uses_external_apis') is False; assert data.data_quality_report.get('cost_profile', {}).get('requires_api_keys') is False; assert data.evaluation_report.get('cost_profile', {}).get('runtime_cost_usd') == 0"

demo:
	$(RUN_IN_ROOT) $(DEMO_CMD)

demo-headless:
	$(RUN_IN_ROOT) $(DEMO_CMD) --server.headless=true

screenshots:
	$(RUN_IN_ROOT) BATTERYOPS_APP_COMMAND="$(SCREENSHOT_APP_CMD)" \
		BATTERYOPS_SCREENSHOT_DIR="$(SCREENSHOT_DIR)" \
		node docs/screenshots/capture.mjs
