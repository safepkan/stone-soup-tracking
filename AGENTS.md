# Working Approach

- High-level planning happens with ChatGPT via the web UI; coding and execution happen here in the CLI.
- When code changes are made, update the relevant Markdown in `mht_experiments/doc` (state, roadmap, next steps, references) so docs track the code.
- Prefer concise updates near the top of the appropriate doc rather than duplicating content.
- Keep references current; if a new paper or PDF is used, add it to `TO_MHT_REFERENCE.md` and the `papers/` folder if available.
- Ask before running commands that modify environments outside the repo or require new dependencies.
- Default to README/AGENTS for workflow notes; keep design/algorithm rationale in the doc files.

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
