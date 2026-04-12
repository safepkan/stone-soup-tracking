# Flat makefile used as command shortener

ENV ?= venv
PYTHON := ./$(ENV)/bin/python
PIP := ./$(ENV)/bin/pip

.PHONY: setup_venv
setup_venv:
	rm -rf $(ENV)/
	python3 -m venv $(ENV)
	$(PIP) install -r requirements.txt

.PHONY: update_venv
update_venv:
	$(PIP) install -r requirements.txt

.PHONY: smoke
smoke:
	MPLBACKEND=Agg TOMHT_NO_SHOW=1 $(PYTHON) mht/runners/run_tomht_crossing.py
	MPLBACKEND=Agg TOMHT_NO_SHOW=1 $(PYTHON) mht/runners/run_tomht_bearing_range.py

.PHONY: smoke_compare
smoke_compare:
	$(PYTHON) replay/smoke_output_regression.py compare

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

.PHONY: replay_compare_timing
replay_compare_timing:
	$(PYTHON) replay/standard_replay_regression.py compare --timing-report

.PHONY: replay_update_baseline
replay_update_baseline:
	$(PYTHON) replay/standard_replay_regression.py update
