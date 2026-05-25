# Standard Replay Baselines

This directory stores versioned baseline artifacts for the canonical standard
replay command. The canonical command includes
`replay/overrides/tracker_standard_replay.json` so replay diagnostics remain
explicit even if the sibling tracker integration stops setting those flags in
code.

Files:

- `standard_replay_default.raw.log`: raw command output (stdout/stderr)
- `standard_replay_default.normalized.log`: filtered output used for regression
  comparison
- `standard_replay_default.timing_summary.log`: extracted timing summary used
  for easy inspection (derived generically from raw output; from `SUMMARY ...`
  lines when present, otherwise synthesized from `SCAN_TIMING*`/`SCAN_MEMORY`)
- `standard_replay_default.replayed.mcap`: replay output MCAP copied from the
  run for easy inspection (kept local; intentionally ignored by Git)

You can regenerate this summary from existing raw output without rerunning
replay:

```bash
source venv/bin/activate
python replay/timing_summary_from_log.py replay/replay_baselines/standard_replay_default.raw.log
```

Normalization currently removes line families that are expected to vary between
runs:

- `SCAN_TIMING ...`
- `SCAN_TIMING_PHASES ...`
- `SCAN_MEMORY ...`
- `SUMMARY timing ...`
- `SUMMARY timing_phases ...`
- `SUMMARY memory ...`

It also normalizes:

- replay run-id path segments (`mcap_replay__<RUN_ID>`)
- local absolute path prefixes (repo/replay executable paths)
- site-packages warning path line numbers

Run compare (heavyweight, optional):

```bash
source venv/bin/activate
python replay/standard_replay_regression.py compare
```

`compare` matches only normalized output and always writes latest raw/normalized
logs plus latest replay MCAP to
`replay/outputs/standard_replay_regression_latest/`.

For performance work, include timing-summary comparison from raw logs:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py compare --timing-report
```

If `SUMMARY timing ...` lines are absent in replay output, the report falls back
to synthesized aggregate summaries from `SCAN_TIMING*` and `SCAN_MEMORY`.

Refresh baseline intentionally:

```bash
source venv/bin/activate
python replay/standard_replay_regression.py update
```
