# Stone Soup Tracking Experiments

Experimental repository for exploring Stone Soup tracking capabilities.

## Working with the Coding Agent

See `AGENTS.md` for the collaboration workflow and expectations on keeping the docs in `mht_experiments/doc` synchronized with code changes.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Testing

Unit tests for the TO-MHT work live under `mht_experiments/tests`.

```bash
source venv/bin/activate
python -m unittest discover -s mht_experiments/tests -p 'test_*.py'
```
