# Flat makefile used as command shortener

.PHONY: setup_venv
setup_venv:
	rm -rf venv/
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

.PHONY: update_venv
update_venv:
	./venv/bin/pip install -r requirements.txt

.PHONY: smoke
smoke:
	MPLBACKEND=Agg TOMHT_NO_SHOW=1 ./venv/bin/python mht/run_tomht_crossing.py
	MPLBACKEND=Agg TOMHT_NO_SHOW=1 ./venv/bin/python mht/run_tomht_bearing_range.py
