# Stone Soup Tracking Experiments

Experimental repository for exploring Stone Soup tracking capabilities.

## Working with the Coding Agent

See `AGENTS.md` for the collaboration workflow and expectations on keeping the TO-MHT docs synchronized with code changes.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Make Shortcuts

The `Makefile` provides a few convenience targets:

- `make setup_venv` creates a venv and installs `requirements.txt`.
- `make update_venv` updates dependencies in the existing venv.
- `make smoke` runs both TO-MHT scenario smoke scripts headless.
- `make smoke_compare` runs normalized smoke-output regression against versioned baselines.
- `make smoke_compare_timing` runs smoke comparison and also prints timing-summary diff from raw logs.
- `make timing_summaries_regenerate_baselines` regenerates baseline timing summaries from existing raw logs (no rerun).
- `make timing_summaries_regenerate_latest` regenerates latest-run timing summaries from existing raw logs (no rerun).
- `make smoke_update_baseline` refreshes the versioned smoke baselines (raw + normalized)
  (use only when baseline updates are intentionally approved).
- `make replay_compare` runs heavyweight standard-replay regression against versioned baselines.
- `make replay_compare_timing` runs replay comparison and also prints timing-summary diff from raw logs.
- `make replay_update_baseline` refreshes standard-replay baselines
  (use only when baseline updates are intentionally approved).
- `make tomht_release_export` replaces the sibling `../stonesoup-tomht/mht`
  snapshot and appends a `RELEASE_HISTORY.md` entry with this repo's current
  commit.

You can select a different environment by passing `ENV` (default is `venv`):

```bash
make setup_venv ENV=.venv310
make smoke ENV=.venv310
make update_venv ENV=.venv312
make smoke ENV=.venv312
```

This is useful for compatibility checks across supported Python versions while
keeping your main development environment separate. For names like `.venv310`,
`make setup_venv` uses `python3.10` when it is available.

## TO-MHT Release Export

Run this from `stone-soup-tracking` when you are ready to prepare the external
release snapshot:

```bash
make tomht_release_export
```

The helper refuses to run if either repo has uncommitted or untracked changes.
It copies `mht/` to the sibling `../stonesoup-tomht` checkout, excluding the
internal `mht/doc/` and `mht/docs/` documentation folders, and appends a dated
entry to `RELEASE_HISTORY.md`. To intentionally export a dirty source snapshot,
run `source venv/bin/activate && python tools/export_tomht_release.py --allow-dirty`;
the recorded source commit gets a `-dirty` suffix.

## TO-MHT Snapshot Install

Install a `stonesoup-tomht` GitHub zip or unpacked snapshot into a target folder
with:

```bash
python tools/install_tomht_snapshot.py ~/releases/stonesoup-tomht-main.zip
```

The default target is the current ISAC integration folder:
`/home/eatpadn/isac/radiant-isac-coeur/apps/radiant_isac/radiant_isac/sandbox/fusion/mht`.
Pass `--target <folder>` to install somewhere else. The helper also accepts an
already-unpacked snapshot folder instead of a zip, preserves target `venv` and
`.venv*` folders, and skips source-side local artifacts such as `.git`,
`__pycache__`, virtualenv folders, caches, and build outputs. Use `--dry-run`
to preview the replacement.

## Dependency Updates

Direct runtime and dev dependencies are pinned in `pyproject.toml` so the
repo states explicit tested versions.

Dependabot is configured in `.github/dependabot.yml` to open weekly PRs for:

- Python dependencies (`pip`)
- GitHub Actions workflow dependencies

## CI

GitHub Actions CI is configured in `.github/workflows/ci.yml`.

It runs on:

- every `push` to `main`
- every `pull_request`

Current CI job:

- Ubuntu 24.04 runner
- Python 3.10, 3.12, 3.13, and 3.14
- dependency install from `requirements.txt`
- `python pre_commit.py --no-dirty`
- `make smoke`

## Testing

Unit tests for the TO-MHT work live under `mht/tests`.

```bash
source venv/bin/activate
python -m unittest discover -s mht/tests -p 'test_*.py'
```
