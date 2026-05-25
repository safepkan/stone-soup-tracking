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
  --tracker-param-override-file ../stone-soup-tracking/replay/overrides/tracker_standard_replay.json \
  --output-path ../stone-soup-tracking/replay/outputs/standard_replay_default
```

## Using JSON Overrides

Always pass the standard replay override first, then add any experiment-specific
override files after it. The replay CLI accepts repeated
`--tracker-param-override-file` arguments and deep-merges them in order, so
later files override earlier files when they touch the same key.

```bash
source venv/bin/activate
python -m python.pipeline.batch_mcap_replay \
  ../stone-soup-tracking/replay/inputs/cpi_replay_2025-12-10_173948.mcap \
  --include-tracker \
  --tracker-type stonesoup-mht \
  --max-cpis 400 \
  --tracker-param-override-file ../stone-soup-tracking/replay/overrides/tracker_standard_replay.json \
  --tracker-param-override-file ../stone-soup-tracking/replay/overrides/tracker_backend_ortools.json \
  --output-path ../stone-soup-tracking/replay/outputs/standard_replay_ortools
```

Standard replay override:

- `replay/overrides/tracker_standard_replay.json`: keeps replay diagnostics
  explicit (`debug_display_detections=false`,
  `debug_display_scan_stats=true`, `debug_display_hypotheses=true`,
  `debug_display_births=true`).

Available backend override templates:

- `replay/overrides/tracker_backend_branch_and_bound.json`
- `replay/overrides/tracker_backend_exhaustive.json`
- `replay/overrides/tracker_backend_ortools.json`

Experimental policy override templates:

- `replay/overrides/overload_split_supported_pruning_apply.json`

Legacy `hypothesis_backend` override templates were removed as that parameter is
no longer part of `TOMHTParams`.

## Standard Replay Regression

Heavyweight optional regression check for the canonical replay command:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py compare
```

The regression wrapper always passes
`replay/overrides/tracker_standard_replay.json` to the sibling replay CLI.
Any `--tracker-param-override-file` arguments supplied to the wrapper are passed
after the standard file, preserving the same later-files-win merge behavior as
manual replay commands.

This command compares only normalized output and stores latest run artifacts for
inspection under:

- `replay/outputs/standard_replay_regression_latest/latest.raw.log`
- `replay/outputs/standard_replay_regression_latest/latest.normalized.log`
- `replay/outputs/standard_replay_regression_latest/latest.timing_summary.log`
- `replay/outputs/standard_replay_regression_latest/latest.replayed.mcap`

To run the same replay and write latest artifacts without comparing against the
baseline, use `run` mode:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py run
```

This is useful when current code is expected to change core outputs, or when
capturing opt-in diagnostics. For expansion/frontier diagnostics:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py run --expansion-frontier
```

To run the standard replay with the l2-sp `StoneSoupMhtTracker` configured for
3-D position estimates, use the dim-3 override:

```bash
source venv/bin/activate
make replay_run_dim3
```

This uses `replay/overrides/tracker_standard_replay.json` followed by
`replay/overrides/tracker_dim_3.json` and writes the same latest artifact paths
as the other standard replay `run` commands.

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

## Standard Replay Profiling

Capture a `cProfile` profile for the standard replay with:

```bash
make replay_profile
```

The profiling target runs the single-file `python.pipeline.mcap_replay` command
directly, because profiling the batch wrapper would mostly measure subprocess
waiting. It includes `replay/overrides/tracker_standard_replay.json` before
`REPLAY_PROFILE_EXTRA_ARGS`, so additional profile overrides can be layered in
the same way as normal replay runs. It writes artifacts under
`replay/outputs/profiles/`:

- `standard_replay_mcap_replay_400.prof`
- `standard_replay_mcap_replay_400.log`
- `standard_replay_mcap_replay_400.log.timing_summary.log`
- `standard_replay_mcap_replay_400.replayed.mcap`

Open the profile in SnakeViz with:

```bash
make replay_profile_snakeviz
```

Useful overrides:

```bash
make replay_profile REPLAY_REPO=/path/to/l2-sp
make replay_profile REPLAY_PROFILE_MAX_CPIS=100
make replay_profile_snakeviz SNAKEVIZ_PORT=8091
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

To run smoke scenarios and write latest artifacts without comparing against the
baseline, use `run` mode:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py run
```

For expansion/frontier diagnostics:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py run --expansion-frontier
```

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
