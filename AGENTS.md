# Working Approach

- The goal is to implement a TO-MHT-style tracker in Python on top of Stone Soup.
- High-level planning happens with ChatGPT via the web UI; coding and execution happen here in the CLI.
- TO-MHT Markdown documents to keep synchronized:
  - `mht/TO_MHT_API.md` — public API and integration guide
  - `mht/doc/TO_MHT_CURRENT_STATE.md` — what the tracker currently does
  - `mht/doc/TO_MHT_NEXT_STEPS.md` — the ordered task list
  - `mht/doc/TO_MHT_ROADMAP.md` — phases / milestones / sequencing
  - `mht/doc/TO_MHT_REFERENCE.md` — references and notes
- When code changes are made, update the relevant Markdown so docs track the code.
- Prefer concise updates near the top of the appropriate doc rather than duplicating content.
- Keep references current; if a new paper or PDF is used, add it to `mht/doc/TO_MHT_REFERENCE.md` and the `mht/doc/papers/` folder if available.
- Mark completed items in `mht/doc/TO_MHT_NEXT_STEPS.md` as implemented (and tighten the wording to match what was actually done if the original listed options), but leave them in place; they can be batch-cleaned in separate commits later.
- Ask before running commands that modify environments outside the repo or require new dependencies.
- Default to README/AGENTS for workflow notes; keep design/algorithm rationale in the doc files.
- For starting a fresh ChatGPT thread, paste `mht/doc/CHAT_CONTEXT.md` into the first message.

## Code Guidelines

- Target Python >=3.10.
- Use type hints throughout; prefer builtin generics (`list[int]`, `dict[str, Any]`) over `List`/`Dict`.
- Formatting/lint/type checks: `black`, `flake8`, `mypy`; always run `venv/bin/python pre_commit.py` after modifying code (not just before pushing).
- Use the repo venv interpreter for project commands; avoid plain `python`.
- Work inside the repo `venv`; manage dependencies via `requirements.txt` only.

## Scenarios / Smoke Tests

- Two baseline scenarios: `run_tomht("crossing")` and `run_tomht("bearing_range")`; convenience scripts `mht/runners/run_tomht_crossing.py` and `mht/runners/run_tomht_bearing_range.py`.
- Runner auto-creates `/tmp/.cache` and `/tmp/mplconfig` (if not set) and sets `XDG_CACHE_HOME`/`MPLCONFIGDIR` to avoid font-cache warnings. Canonical single-scenario headless incantations:
  - `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht/runners/run_tomht_crossing.py`
  - `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht/runners/run_tomht_bearing_range.py`
  The runner detects non-interactive backends and skips `plt.show()`.
- Control animation display: set `TOMHT_SHOW=1` to force showing even with non-interactive backends; set `TOMHT_NO_SHOW=1` to suppress entirely.
- Expected behavior: scripts complete without exceptions; logs print global hypotheses over time and end-of-run `SUMMARY ...` aggregate ScanStats lines. Use output to spot regressions; at minimum ensure they don’t crash after code changes.
- Quick smoke check: `make smoke` runs both scenarios headless (`MPLBACKEND=Agg TOMHT_NO_SHOW=1`) and fails on any crash.
- Standard post-change output-regression validation: `make smoke_compare`.
  - Run this after code updates unless the task is clearly unrelated.
  - If it reports differences, treat that as a review point: assess whether the
    delta is expected from the change.
  - Do **not** update smoke baselines unless explicitly requested by the user.
- Capture-only smoke runs are available when differences are expected or when
  optional diagnostics are enabled:
  - `make smoke_run`
  - `make smoke_expansion_frontier`
  These write latest artifacts under `replay/outputs/smoke_regression_latest/`
  and skip baseline comparison.
- Optional heavyweight replay regression (not part of routine validation):
  `make replay_compare`.
  - Use when replay-level behavior checks are useful.
  - Treat differences as a review point against expected change impact.
  - Do **not** update replay baselines unless explicitly requested by the user.
- Capture-only replay runs are available for expected output changes or optional
  diagnostics:
  - `make replay_run`
  - `make replay_expansion_frontier`
  These write latest artifacts under
  `replay/outputs/standard_replay_regression_latest/` and skip baseline
  comparison.
- For performance-focused investigations, optional timing-summary comparisons are
  available from raw baseline vs latest raw run output:
  - `make smoke_compare_timing`
  - `make replay_compare_timing`
- The TO-MHT convenience scripts (`run_tomht_crossing.py`, `run_tomht_bearing_range.py`) accept CLI flags to flip scoring mode and to toggle initiator/births or scenario initial tracks; use them for A/B testing without editing runner code.

## Standard Replay Example

- Canonical replay input: `replay/inputs/cpi_replay_2025-12-10_173948.mcap`.
- Keep replay artifacts under `replay/` with this convention:
  - `inputs/` = versioned replay inputs
  - `overrides/` = versioned JSON override templates
  - `outputs/` = replay outputs/logs/profiles (local working artifacts)
- Standard command from the `l2-sp` clone root (or any equivalent clone that
  contains `python.pipeline.batch_mcap_replay`):
  - `source venv/bin/activate && python -m python.pipeline.batch_mcap_replay ../stone-soup-tracking/replay/inputs/cpi_replay_2025-12-10_173948.mcap --include-tracker --tracker-type stonesoup-mht --max-cpis 400 --tracker-param-override-file ../stone-soup-tracking/replay/overrides/tracker_standard_replay.json --output-path ../stone-soup-tracking/replay/outputs/standard_replay_default`
- For backend/config overrides, add them after the standard replay override:
  - `--tracker-param-override-file ../stone-soup-tracking/replay/overrides/<file>.json`
- See `replay/README.md` for examples.
