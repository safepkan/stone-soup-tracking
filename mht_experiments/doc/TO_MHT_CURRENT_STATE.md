# TO-MHT Current State

This document is a cleaned-up snapshot of the current implementation after completing the external-initiation and birth-pipeline cleanup phase.

Note (2026-03-20): this document reflects pre-Phase-B architecture context and is now stale. For current implementation status, see `TO_MHT_NEXT_STEPS.md`.

## What is now solid

### MFA baseline isolation

The legacy MFA baseline code now lives under `mht_experiments/mfa/` so TO-MHT work stays separated from baseline/reference code.

### TOMHT module flattening

`tomht_runner.py` and `tomht_tracker.py` now live directly under `mht_experiments/` (the old `runners/` and `trackers/` subpackages were removed).

### Deterministic tracker baseline

The tracker has deterministic ordering and stable beam-management behavior, making it practical to inspect, compare, and iterate on.

### External confirmed-start support

The tracker now supports externally supplied confirmed track starts via `add_external_starts(starts, timestamp)`.

Current semantics:
- external starts are injected after a completed `step()` at the same timestamp,
- they are inserted structurally into each current global hypothesis,
- they are not routed through internal residual/birth discovery,
- they do not receive internal birth penalties,
- the tracker starts empty and no longer supports constructor-time `initial_tracks`.

### Scenario / runner support

The existing `crossing` and `bearing_range` runners can exercise:
- external-only operation,
- internal-only operation,
- both external starts and internal births,
- custom flag-driven configurations.

The runner layer now makes operating mode, external-start enablement, and external-start timing explicit enough for practical experimentation.

### Internal birth path readability

The internal birth path has been refactored into explicit helper stages, making the current logic easier to reason about without changing its intended behavior.

### Metadata / startup cleanup

The tracker’s startup model is now simpler:
- start empty,
- process a step,
- optionally inject confirmed external starts at that timestamp,
- continue normal maintenance.

This is a cleaner interface for upcoming integration work than the earlier constructor-time initial-track path.

## What still works, but is architecturally wrong

The biggest remaining limitation is structural.

### Flat globals with copied track objects

Global hypotheses still store copied per-track `Track` objects rather than explicit shared per-track hypothesis ancestry.

That means:
- shared history is not represented directly,
- ancestry is implicit rather than explicit,
- multiple globals duplicate track-history content instead of sharing nodes,
- pruning/commitment cannot be expressed cleanly in true TO-MHT terms.

### N-scan is still only an approximation

The current N-scan behavior is still based on recent association-history tail logic rather than explicit ancestor-based commitment.

This is good enough for the current prototype, but it is not yet a true TO-MHT representation.

## What this means

The implementation is now in a good place to stop polishing startup/birth-interface details and move to the next real structural step:

**introducing explicit track-hypothesis structure and true N-scan pruning.**

That is the current architectural bottleneck, and it should be addressed before deeper scoring work.

## Deferred topics intentionally left for later

These are known topics, but are not the current priority:

- richer external-start scheduling,
- pre-first-step external starts,
- deeper scoring cleanup,
- more principled birth/existence modelling,
- performance optimisation.
