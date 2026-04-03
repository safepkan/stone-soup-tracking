# Stone Soup Tracking Experiments

Experimental repository for exploring Stone Soup tracking capabilities.

## Working with the Coding Agent

See `AGENTS.md` for the collaboration workflow and expectations on keeping the docs in `mht/doc` synchronized with code changes.

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

## Testing

Unit tests for the TO-MHT work live under `mht/tests`.

```bash
source venv/bin/activate
python -m unittest discover -s mht/tests -p 'test_*.py'
```
