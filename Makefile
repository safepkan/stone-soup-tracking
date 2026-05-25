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
REPLAY_REPO ?= $(CURDIR)/../l2-sp
REPLAY_VENV ?= $(REPLAY_REPO)/venv
REPLAY_INPUT ?= $(CURDIR)/replay/inputs/cpi_replay_2025-12-10_173948.mcap
REPLAY_PROFILE_MAX_CPIS ?= 400
REPLAY_PROFILE_DIR ?= $(CURDIR)/replay/outputs/profiles
REPLAY_PROFILE_NAME ?= standard_replay_mcap_replay_$(REPLAY_PROFILE_MAX_CPIS)
REPLAY_PROFILE_PATH ?= $(REPLAY_PROFILE_DIR)/$(REPLAY_PROFILE_NAME).prof
REPLAY_PROFILE_LOG ?= $(REPLAY_PROFILE_DIR)/$(REPLAY_PROFILE_NAME).log
REPLAY_PROFILE_MCAP ?= $(REPLAY_PROFILE_DIR)/$(REPLAY_PROFILE_NAME).replayed.mcap
REPLAY_PROFILE_EXTRA_ARGS ?=
REPLAY_STANDARD_OVERRIDE ?= $(CURDIR)/replay/overrides/tracker_standard_replay.json
REPLAY_DIM3_OVERRIDE ?= $(CURDIR)/replay/overrides/tracker_dim_3.json
SNAKEVIZ_HOST ?= 127.0.0.1
SNAKEVIZ_PORT ?= 8090

.PHONY: setup_venv
setup_venv:
	rm -rf $(ENV)/
	$(VENV_PYTHON) -m venv $(ENV)
	$(PIP) install -r requirements.txt

.PHONY: update_venv
update_venv:
	$(PIP) install -r requirements.txt

.PHONY: pre_commit
pre_commit:
	PATH="$(CURDIR)/$(ENV)/bin:$$PATH" $(PYTHON) pre_commit.py --no-dirty

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

.PHONY: replay_run_dim3
replay_run_dim3:
	$(PYTHON) replay/standard_replay_regression.py run --tracker-param-override-file "$(REPLAY_DIM3_OVERRIDE)"

.PHONY: replay_expansion_frontier
replay_expansion_frontier:
	$(PYTHON) replay/standard_replay_regression.py run --expansion-frontier

.PHONY: replay_compare_timing
replay_compare_timing:
	$(PYTHON) replay/standard_replay_regression.py compare --timing-report

.PHONY: replay_profile
replay_profile:
	mkdir -p "$(REPLAY_PROFILE_DIR)"
	cd "$(REPLAY_REPO)" && . "$(REPLAY_VENV)/bin/activate" && XDG_CACHE_HOME=/tmp/.cache MPLCONFIGDIR=/tmp/mplconfig MPLBACKEND=Agg TOMHT_NO_SHOW=1 python -m cProfile -o "$(REPLAY_PROFILE_PATH)" -m python.pipeline.mcap_replay "$(REPLAY_INPUT)" -o "$(REPLAY_PROFILE_MCAP)" --force --include-tracker --tracker-type stonesoup-mht --max-cpis "$(REPLAY_PROFILE_MAX_CPIS)" --tracker-param-override-file "$(REPLAY_STANDARD_OVERRIDE)" $(REPLAY_PROFILE_EXTRA_ARGS) > "$(REPLAY_PROFILE_LOG)" 2>&1
	. "$(ENV)/bin/activate" && python replay/timing_summary_from_log.py "$(REPLAY_PROFILE_LOG)"
	@echo "[profile] profile: $(REPLAY_PROFILE_PATH)"
	@echo "[profile] log: $(REPLAY_PROFILE_LOG)"
	@echo "[profile] timing summary: $(REPLAY_PROFILE_LOG).timing_summary.log"
	@echo "[profile] replayed MCAP: $(REPLAY_PROFILE_MCAP)"

.PHONY: replay_profile_snakeviz
replay_profile_snakeviz:
	. "$(ENV)/bin/activate" && snakeviz -H "$(SNAKEVIZ_HOST)" -p "$(SNAKEVIZ_PORT)" "$(REPLAY_PROFILE_PATH)"

.PHONY: replay_update_baseline
replay_update_baseline:
	$(PYTHON) replay/standard_replay_regression.py update

.PHONY: tomht_release_export
tomht_release_export:
	$(PYTHON) tools/export_tomht_release.py

.PHONY: tomht_release_export_commit
tomht_release_export_commit:
	$(PYTHON) tools/export_tomht_release.py --commit
