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

You can select a different environment by passing `ENV` (default is `venv`):

```bash
make update_venv ENV=.venv312
make smoke ENV=.venv312
```

This is useful for periodic Python 3.12 compatibility checks while keeping your
main development environment separate.

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
- Python 3.12
- dependency install from `requirements.txt`
- `python pre_commit.py --no-dirty`
- `make smoke`

## Testing

Unit tests for the TO-MHT work live under `mht/tests`.

```bash
source venv/bin/activate
python -m unittest discover -s mht/tests -p 'test_*.py'
```
