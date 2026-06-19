# Stone Soup TO-MHT Tracker

Development repository for a track-oriented multiple-hypothesis tracker (TO-MHT)
built on Stone Soup.

The tracker itself lives in [`mht/`](mht/) — start at
[`mht/README.md`](mht/README.md). That folder is the unit exported to the
external release repo (`../stonesoup-tomht`); everything outside it is
development and workflow support (replay regression, smoke scenarios, release
export, CI) and is not part of the release.

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
- `make mht_tests` runs all TO-MHT unit tests under `mht/tests`.
- `make replay_tests` runs replay-regression helper tests under `replay/tests`.
- `make smoke` runs both TO-MHT scenario smoke scripts headless.
- `make smoke_compare` runs normalized smoke-output regression against versioned baselines.
- `make smoke_compare_timing` runs smoke comparison and also prints timing-summary diff from raw logs.
- `make timing_summaries_regenerate_baselines` regenerates baseline timing summaries from existing raw logs (no rerun).
- `make timing_summaries_regenerate_latest` regenerates latest-run timing summaries from existing raw logs (no rerun).
- `make smoke_update_baseline` refreshes the versioned smoke baselines (raw + normalized)
  (use only when baseline updates are intentionally approved).
- `make replay_compare` runs heavyweight standard-replay regression against versioned baselines.
- `make replay_compare_timing` runs replay comparison and also prints timing-summary diff from raw logs.
- `make replay_profile` runs the standard replay through `cProfile` and writes artifacts under `replay/outputs/profiles/`.
- `make replay_profile_snakeviz` opens the latest standard replay profile in SnakeViz.
- `make replay_update_baseline` refreshes standard-replay baselines
  (use only when baseline updates are intentionally approved).
- `make tomht_release_export` replaces the sibling `../stonesoup-tomht/mht`
  snapshot and appends a `RELEASE_HISTORY.md` entry with this repo's current
  commit.
- `make tomht_release_export_commit` does the same export and commits it in the
  release repo.

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

Use `make tomht_release_export_commit` to also create the matching release
commit in the external repo.

The helper refuses to run if either repo has uncommitted or untracked changes.
It copies `mht/` to the sibling `../stonesoup-tomht` checkout, excluding the
internal `mht/doc/` and `mht/docs/` documentation folders, and appends a dated
entry to `RELEASE_HISTORY.md`. To intentionally export a dirty source snapshot,
run `source venv/bin/activate && python tools/export_tomht_release.py --allow-dirty`;
the recorded source commit gets a `-dirty` suffix.

Pass `--commit` to also commit the exported snapshot in the release repo with a
message like:

```text
Release 2026-05-20

Snapshot from stone-soup-tracking b3a6370d3822c86fe40bb27cfe6c1cf9c0980fcd.
Release notes are in RELEASE_HISTORY.md.
```

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

Unit tests for the exported TO-MHT package live under `mht/tests`. Replay
regression helper tests live under `replay/tests` because top-level `replay/`
is development infrastructure and is not exported.

```bash
make mht_tests
make replay_tests
```
