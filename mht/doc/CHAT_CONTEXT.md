# Chat context (paste into a new ChatGPT thread)

This file exists to quickly bootstrap a new ChatGPT thread with project context.
Design and algorithm rationale live in the Markdown docs listed below.

## Goal
Implement a clear, general TO-MHT-style tracker in Python on top of Stone Soup.
Use ChatGPT for high-level design and review, and Codex/VS Code for implementation.
Keep the design docs in sync with the code.

## Read first (canonical docs)
- `TO_MHT_CURRENT_STATE.md` — what the tracker currently does
- `TO_MHT_NEXT_STEPS.md` — the current implementation phase
- `TO_MHT_ROADMAP.md` — high-level direction and priorities
- `TO_MHT_REFERENCE.md` — references and notes

## Current baseline
As of 2026-03-12, the tracker already has:
- stable per-scan detection ordering,
- beta-ratio v1.5 scoring via `ScoringModel`,
- association history plus N-scan-lite deduplication,
- scan/run instrumentation (`ScanStats`, `BirthStats`, summary metrics).

## Current focus
- External initiation support and birth-handling cleanup.
- Immediate target: support a mode where TO-MHT replaces an existing **system tracker** while upstream code still handles track starts.

## Application note
A key target use case is an ISAC / multi-sensor bistatic-radar-style setup in 6G mobile networks:
- each TX-RX link acts like a sensor,
- angle measurements are ambiguous,
- upstream processing may keep per-sensor tracking and cross-sensor correlation / ambiguity resolution,
- TO-MHT may initially plug in as the downstream system tracker.

This means external track initiation is a first-class integration requirement.

## Primary entry point(s)
- `tomht_tracker.py` — core tracker implementation

## Ways of working
- High-level planning and design review happen in ChatGPT.
- Coding and execution happen in VS Code / Codex.
- When code changes are made, update the relevant docs in `mht/doc`.
- Use the project venv Python for commands; avoid plain `python`.
- Workflow details and repo conventions live in `AGENTS.md`.

## Quick sanity checks
- Smoke test (headless): `make smoke`
- Crossing only (headless): `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht/run_tomht_crossing.py`
- Bearing-range only (headless): `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht/run_tomht_bearing_range.py`
