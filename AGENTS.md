# Working Approach

- The goal is to implement a TO-MHT-style tracker in Python on top of Stone Soup.
- High-level planning happens with ChatGPT via the web UI; coding and execution happen here in the CLI.
- Markdown documents in `mht_experiments/doc`:
  - `TO_MHT_CURRENT_STATE.md` — what the tracker currently does
  - `TO_MHT_NEXT_STEPS.md` — the ordered task list
  - `TO_MHT_ROADMAP.md` — phases / milestones / sequencing
  - `TO_MHT_REFERENCE.md` — references and notes
- When code changes are made, update the relevant Markdown so docs track the code.
- Prefer concise updates near the top of the appropriate doc rather than duplicating content.
- Keep references current; if a new paper or PDF is used, add it to `TO_MHT_REFERENCE.md` and the `papers/` folder if available.
- Mark completed items in `TO_MHT_NEXT_STEPS.md` as implemented (and tighten the wording to match what was actually done if the original listed options), but leave them in place; they can be batch-cleaned in separate commits later.
- Ask before running commands that modify environments outside the repo or require new dependencies.
- Default to README/AGENTS for workflow notes; keep design/algorithm rationale in the doc files.
- For starting a fresh ChatGPT thread, paste `mht_experiments/doc/CHAT_CONTEXT.md` into the first message.

## Code Guidelines

- Target Python >=3.12.
- Use type hints throughout; prefer builtin generics (`list[int]`, `dict[str, Any]`) over `List`/`Dict`.
- Formatting/lint/type checks: `black`, `flake8`, `mypy`; run `python pre_commit.py` before pushing.
- Work inside the repo `venv`; manage dependencies via `requirements.txt` only.

## Scenarios / Smoke Tests

- Two baseline scenarios: `run_tomht("crossing")` and `run_tomht("bearing_range")`; convenience scripts `mht_experiments/run_tomht_crossing.py` and `mht_experiments/run_tomht_bearing_range.py`.
- Runner auto-creates `/tmp/.cache` and `/tmp/mplconfig` (if not set) and sets `XDG_CACHE_HOME`/`MPLCONFIGDIR` to avoid font-cache warnings. You can still run explicitly headless: `MPLBACKEND=Agg venv/bin/python mht_experiments/run_tomht_crossing.py` (same for `bearing_range`). The runner detects non-interactive backends and skips `plt.show()`.
- Control animation display: set `TOMHT_SHOW=1` to force showing even with non-interactive backends; set `TOMHT_NO_SHOW=1` to suppress entirely.
- Expected behavior: scripts complete without exceptions; logs print global hypotheses over time. Use output to spot regressions; at minimum ensure they don’t crash after code changes.
- Quick smoke check: `make smoke` runs both scenarios headless (`TOMHT_NO_SHOW=1`) and fails on any crash.
- The TO-MHT convenience scripts (`run_tomht_crossing.py`, `run_tomht_bearing_range.py`) accept CLI flags to flip scoring mode and to toggle initiator/births or scenario initial tracks; use them for A/B testing without editing runner code.
