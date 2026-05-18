# Flat makefile used as command shortener

ENV ?= venv
ENV_PYTHON_VERSION ?= $(shell printf '%s\n' "$(ENV)" | sed -n 's/.*venv\([0-9]\)\([0-9][0-9]\)$$/\1.\2/p')
VENV_PYTHON ?= $(shell \
	if [ -f "$(ENV)/pyvenv.cfg" ]; then \
		home=$$(sed -n 's/^home = //p' "$(ENV)/pyvenv.cfg" | sed 1q); \
		if [ -n "$(ENV_PYTHON_VERSION)" ] && [ -x "$$home/python$(ENV_PYTHON_VERSION)" ]; then \
			printf '%s\n' "$$home/python$(ENV_PYTHON_VERSION)"; \
		elif [ -x "$$home/python3" ]; then \
			printf '%s\n' "$$home/python3"; \
		else \
			command -v python3; \
		fi; \
	elif [ -n "$(ENV_PYTHON_VERSION)" ] && command -v "python$(ENV_PYTHON_VERSION)" >/dev/null 2>&1; then \
		command -v "python$(ENV_PYTHON_VERSION)"; \
	else \
		command -v python3; \
	fi)
PYTHON := ./$(ENV)/bin/python
PIP := ./$(ENV)/bin/pip

.PHONY: setup_venv
setup_venv:
	rm -rf $(ENV)/
	$(VENV_PYTHON) -m venv $(ENV)
	$(PIP) install -r requirements.txt

.PHONY: update_venv
update_venv:
	$(PIP) install -r requirements.txt

.PHONY: mht_tests
mht_tests:
	$(PYTHON) -m pytest mht/tests

.PHONY: smoke
smoke:
	MPLBACKEND=Agg TOMHT_NO_SHOW=1 $(PYTHON) mht/runners/run_tomht_crossing.py
	MPLBACKEND=Agg TOMHT_NO_SHOW=1 $(PYTHON) mht/runners/run_tomht_bearing_range.py

.PHONY: smoke_compare
smoke_compare:
	$(PYTHON) replay/smoke_output_regression.py compare

.PHONY: smoke_run
smoke_run:
	$(PYTHON) replay/smoke_output_regression.py run

.PHONY: smoke_expansion_frontier
smoke_expansion_frontier:
	$(PYTHON) replay/smoke_output_regression.py run --expansion-frontier

.PHONY: smoke_compare_timing
smoke_compare_timing:
	$(PYTHON) replay/smoke_output_regression.py compare --timing-report

.PHONY: timing_summaries_regenerate_baselines
timing_summaries_regenerate_baselines:
	$(PYTHON) replay/timing_summary_from_log.py --known-set baseline

.PHONY: timing_summaries_regenerate_latest
timing_summaries_regenerate_latest:
	$(PYTHON) replay/timing_summary_from_log.py --known-set latest --skip-missing

.PHONY: smoke_update_baseline
smoke_update_baseline:
	$(PYTHON) replay/smoke_output_regression.py update

.PHONY: replay_compare
replay_compare:
	$(PYTHON) replay/standard_replay_regression.py compare

.PHONY: replay_run
replay_run:
	$(PYTHON) replay/standard_replay_regression.py run

.PHONY: replay_expansion_frontier
replay_expansion_frontier:
	$(PYTHON) replay/standard_replay_regression.py run --expansion-frontier

.PHONY: replay_compare_timing
replay_compare_timing:
	$(PYTHON) replay/standard_replay_regression.py compare --timing-report

.PHONY: replay_update_baseline
replay_update_baseline:
	$(PYTHON) replay/standard_replay_regression.py update
