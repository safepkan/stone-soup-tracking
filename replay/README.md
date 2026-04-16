# Replay Runs Convention

Use this folder as the standard location for TO-MHT replay work artifacts.

## Layout

- `inputs/`: versioned replay input files (including the standard example MCAP).
- `overrides/`: versioned JSON tracker override examples.
- `outputs/`: replay outputs/logs/profiles (kept local, not versioned by default).
- `smoke_baselines/`: versioned normalized golden output for TO-MHT smoke scenarios.

## Standard Replay Example

Canonical input:

- `replay/inputs/cpi_replay_2025-12-10_173948.mcap`

The canonical replay input is versioned in this repo (`stone-soup-tracking`).
When running from the `l2-sp` clone root (or any equivalent clone that
contains `python.pipeline.batch_mcap_replay`), pass input/output paths that
point into this repo. For input, use either:

- a relative path into this repo (for sibling clones):
  `../stone-soup-tracking/replay/inputs/cpi_replay_2025-12-10_173948.mcap`
- or an absolute path:
  `/Users/patrik/Git/stone-soup-tracking/replay/inputs/cpi_replay_2025-12-10_173948.mcap`

Example from `l2-sp` root (relative-path form):

```bash
source venv/bin/activate
python -m python.pipeline.batch_mcap_replay \
  ../stone-soup-tracking/replay/inputs/cpi_replay_2025-12-10_173948.mcap \
  --include-tracker \
  --tracker-type stonesoup-mht \
  --max-cpis 400 \
  --output-path ../stone-soup-tracking/replay/outputs/standard_replay_default
```

## Using JSON Overrides

Pass a JSON file with TOMHT parameter overrides:

```bash
source venv/bin/activate
python -m python.pipeline.batch_mcap_replay \
  ../stone-soup-tracking/replay/inputs/cpi_replay_2025-12-10_173948.mcap \
  --include-tracker \
  --tracker-type stonesoup-mht \
  --max-cpis 400 \
  --tracker-param-override-file ../stone-soup-tracking/replay/overrides/tracker_backend_ortools.json \
  --output-path ../stone-soup-tracking/replay/outputs/standard_replay_ortools
```

Available backend override templates:

- `replay/overrides/tracker_backend_branch_and_bound.json`
- `replay/overrides/tracker_backend_exhaustive.json`
- `replay/overrides/tracker_backend_ortools.json`

Legacy `hypothesis_backend` override templates were removed as that parameter is
no longer part of `TOMHTParams`.

## Standard Replay Regression

Heavyweight optional regression check for the canonical replay command:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py compare
```

This command compares only normalized output and stores latest run artifacts for
inspection under:

- `replay/outputs/standard_replay_regression_latest/latest.raw.log`
- `replay/outputs/standard_replay_regression_latest/latest.normalized.log`
- `replay/outputs/standard_replay_regression_latest/latest.timing_summary.log`
- `replay/outputs/standard_replay_regression_latest/latest.replayed.mcap`

Versioned golden baseline artifacts are stored in:

- `replay/replay_baselines/standard_replay_default.raw.log`
- `replay/replay_baselines/standard_replay_default.normalized.log`
- `replay/replay_baselines/standard_replay_default.timing_summary.log`
- `replay/replay_baselines/standard_replay_default.replayed.mcap` (local
  inspectable artifact; intentionally not versioned in Git)

For performance-oriented checks, include timing-summary comparison from the raw
logs:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py compare --timing-report
```

If replay logs do not contain `SUMMARY timing ...` lines, the timing report
automatically derives aggregate summaries from `SCAN_TIMING*` and
`SCAN_MEMORY` lines.

Timing-summary generation is also available as a standalone post-processing
step:

```bash
source venv/bin/activate
python replay/timing_summary_from_log.py replay/replay_baselines/standard_replay_default.raw.log
```

Regenerate known sets without reruns:

```bash
source venv/bin/activate
make timing_summaries_regenerate_baselines
make timing_summaries_regenerate_latest
```

Refresh replay baseline intentionally:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py update
```

Do not run replay baseline updates as part of routine validation; update only
when replay-output changes are intentional and explicitly approved.

## Smoke Output Regression

Run normalized smoke-output regression against the versioned baselines:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py compare
```

The harness stores both raw and normalized outputs:

- versioned baselines: `replay/smoke_baselines/<scenario>.raw.log`,
  `replay/smoke_baselines/<scenario>.normalized.log`,
  `replay/smoke_baselines/<scenario>.timing_summary.log`
- latest inspectable run: `replay/outputs/smoke_regression_latest/<scenario>.raw.log`,
  `replay/outputs/smoke_regression_latest/<scenario>.normalized.log`,
  `replay/outputs/smoke_regression_latest/<scenario>.timing_summary.log`

Only normalized output is used for pass/fail comparison.

The harness pins scenario start times so `SCAN t=...` timestamps remain stable
and diffable.

For performance-oriented checks, include timing-summary comparison from the raw
logs:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py compare --timing-report
```

Refresh baselines intentionally:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py update
```

Do not run baseline updates as part of routine validation; update only when the
output change is intentional and explicitly approved.
