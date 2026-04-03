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
