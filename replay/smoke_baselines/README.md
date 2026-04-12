# Smoke Output Baselines

This directory stores versioned golden outputs for the two TO-MHT smoke
scenarios.

Per scenario:

- `<scenario>.raw.log`: full raw command output (stdout/stderr), kept for
  inspection/performance analysis
- `<scenario>.normalized.log`: filtered output used for regression comparison
- `<scenario>.timing_summary.log`: extracted timing summary used for easy
  inspection (derived generically from raw output; from `SUMMARY ...` lines when
  present, otherwise synthesized from `SCAN_TIMING*`/`SCAN_MEMORY`)

You can regenerate these summaries from existing raw logs without rerunning
scenarios:

```bash
source venv/bin/activate
python replay/timing_summary_from_log.py replay/smoke_baselines/crossing.raw.log replay/smoke_baselines/bearing_range.raw.log
```

Normalization currently removes line families that are expected to change
between runs:

- `SCAN_TIMING ...`
- `SCAN_TIMING_PHASES ...`
- `SCAN_MEMORY ...`
- `SUMMARY timing ...`
- `SUMMARY timing_phases ...`
- `SUMMARY memory ...`

Timestamps are intentionally kept. The harness pins scenario start times via:

- `--scenario-start-time 2026-01-01T00:00:00`

Use the regression harness:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py compare
```

`compare` matches only the normalized outputs and always writes latest run logs
to `replay/outputs/smoke_regression_latest/`.

For performance work, include timing-summary comparison from raw logs:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py compare --timing-report
```

Refresh baselines intentionally:

```bash
source venv/bin/activate
python replay/smoke_output_regression.py update
```
