# Replay Runs Convention

Use this folder as the standard location for TO-MHT replay work artifacts.

## Layout

- `inputs/`: versioned replay input files (including the standard example MCAP).
- `overrides/`: versioned JSON tracker override examples.
- `outputs/`: replay outputs/logs/profiles (kept local, not versioned by default).

## Standard Replay Example

Canonical input:

- `replay/inputs/cpi_replay_2025-12-10_173948.mcap`

Run from the `l2-sp` clone root (or any equivalent clone that contains
`python.pipeline.batch_mcap_replay`):

```bash
source venv/bin/activate
python -m python.pipeline.batch_mcap_replay \
  replay/inputs/cpi_replay_2025-12-10_173948.mcap \
  --include-tracker \
  --tracker-type stonesoup-mht \
  --max-cpis 400 \
  --output-path replay/outputs/standard_replay_default
```

## Using JSON Overrides

Pass a JSON file with TOMHT parameter overrides:

```bash
source venv/bin/activate
python -m python.pipeline.batch_mcap_replay \
  replay/inputs/cpi_replay_2025-12-10_173948.mcap \
  --include-tracker \
  --tracker-type stonesoup-mht \
  --max-cpis 400 \
  --tracker-param-override-file replay/overrides/tracker_backend_ortools.json \
  --output-path replay/outputs/standard_replay_ortools
```

Available backend override templates:

- `replay/overrides/tracker_backend_branch_and_bound.json`
- `replay/overrides/tracker_backend_exhaustive.json`
- `replay/overrides/tracker_backend_ortools.json`
