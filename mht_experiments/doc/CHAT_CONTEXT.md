# Chat context (paste into a new ChatGPT thread)

This file exists to quickly bootstrap a new ChatGPT thread with project context.
Design/algorithm rationale and decisions live in the other docs listed below.

## Goal
Implement a TO-MHT-style tracker in Python on top of Stone Soup. Use ChatGPT for high-level design and
Codex/VS Code for implementation. Keep design docs in sync with code.

## Read first (canonical docs)
- `TO_MHT_CURRENT_STATE.md` — what the tracker currently does
- `TO_MHT_NEXT_STEPS.md` — the ordered task list
- `TO_MHT_ROADMAP.md` — phases / milestones / sequencing
- `TO_MHT_REFERENCE.md` — references and notes

## Current focus (update occasionally)
- As of 2026-02-04: Scoring - moving toward a simple MHT log-likelihood

## Primary entry point(s)
- `tomht_tracker.py` — core tracker implementation

## Ways of working
- High-level planning happens in ChatGPT (web UI); coding + running happens in VS Code/Codex.
- When code changes are made, update the relevant Markdown in `mht_experiments/doc` so docs track the code.
- Workflow details (how to run scenarios/tests, coding standards) live in `AGENTS.md`.

## Quick sanity checks
- Smoke test (canonical headless): `make smoke` (uses `MPLBACKEND=Agg TOMHT_NO_SHOW=1`)
- Crossing only (headless): `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht_experiments/run_tomht_crossing.py`
- Bearing-range only (headless): `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht_experiments/run_tomht_bearing_range.py`

(See `AGENTS.md` for scenario commands and repo workflow details.)
