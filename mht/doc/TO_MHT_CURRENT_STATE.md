# TO-MHT Current State

This document is a current snapshot of the implementation after completing Phase B: explicit track-hypothesis structure and true ancestor-based N-scan commitment.

It replaces the earlier pre-Phase-B description in which globals still stored copied `Track` objects and N-scan was only a history-tail approximation. That older description is no longer accurate. `TO_MHT_NEXT_STEPS.md` remains the main phase-planning document; this file is the implementation-status snapshot. 

## Recent updates

- 2026-03-23: extracted core passive data-structure dataclasses (`TrackHypothesisNode`, `GlobalHypothesis`, `ChildCandidate`, `MAPHypothesisSnapshot`, `NScanCommitmentSnapshot`) from `mht/tomht_tracker.py` into `mht/tomht_model.py`; `ScanContext` and `ScanStats` remain in `mht/tomht_tracker.py`.
- 2026-03-23: removed `TrackHypothesisNode.track_metadata`; opaque metadata bags are no longer propagated through node ancestry, and reconstructed `Track.metadata` now contains explicit TOMHT-owned keys only.
- 2026-03-23: removed legacy `TOMHTParams.assoc_history_len`; `ns_scan_window` is now the only N-scan window parameter.
- 2026-03-22: added explicit TOMHT-facing helper `get_tomht_track_id(track)` in `mht/tomht_tracker.py` for extracting stable logical IDs from `TOMHTTracker` output tracks.
- 2026-03-22: tracker-construction helpers `build_tomht_linear()` and `build_tomht_ukf()` were moved out of `mht/tomht_tracker.py` into `mht/helpers/tracker_builders.py`; `TOMHTTracker` now type-hints `initiator` as generic Stone Soup `Initiator` instead of `SimpleMeasurementInitiator`.
- 2026-03-22: legacy MFA baseline scaffolding was moved from `mht/mfa` to `archive/mfa` to keep inactive/reference code out of the active TO-MHT package path.
- 2026-03-21: `update_tracker()` was refactored so scan stats and debug output are handled by dedicated private helpers. The core scan pipeline path is now easier to read without instrumentation details inline.
- 2026-03-21: `TOMHTTracker.get_unused_detections()` now exposes the residual detections from the most recent completed `update_tracker()`. Residuals are considered consumed when internal births are enabled (`initiator is not None`), so in that mode the method returns an empty list.
- 2026-03-21: `TOMHTTracker.add_external_starts()` now follows the same argument style as `update_tracker()`: `add_external_starts(time, starts)` with `time: datetime.datetime`.
- 2026-03-21: public API usage notes were tightened to make item-2/item-3 integration work less ambiguous.

## What is now structurally correct

### Explicit track-oriented hypothesis structure

The tracker now uses an explicit per-track hypothesis-node representation.

In the current implementation:
- `TrackHypothesisNode` is the canonical internal unit of branching,
- each node carries a same-track parent pointer,
- each node has explicit `track_id`, `node_id`, `scan_index`, per-step state payload, association label / used detection identity, and cached maintenance fields,
- ancestry is represented structurally by shared node identity rather than by copied history content. 

This means the internal representation is now much closer to a real TO-MHT than before.

### Globals now reference leaf nodes, not copied tracks

`GlobalHypothesis` now stores `track_id -> leaf node` plus cumulative log weight.

That means:
- globals no longer use copied full-history `Track` objects as primary internal truth,
- shared ancestry is expressed through shared node references,
- branch identity is node-based rather than reconstructed from copied recent history.

### Per-track branching is node-native

Continuation now works structurally as parent-leaf to child-node creation.

In practice:
- hit and miss continuations create new child nodes,
- global expansion operates over leaf nodes,
- detection-usage checks read node fields,
- deduplication is based on structural leaf identity (`track_id -> node_id`) rather than recent association-history tails.

### Births and external starts now share the same structural system

External starts and internal births are now both represented as root-like nodes in the same hypothesis structure.

Their semantics remain distinct:
- external starts are introduced through the external-start path and do not inherit internal-birth scoring semantics,
- internal births are introduced through the birth path and remain birth-scored.

So the structure is shared while provenance and scoring remain separate.

### True N-scan commitment is now explicit

N-scan handling is no longer just a history-tail approximation.

The current implementation now computes commitment as follows:
- after expansion/scoring and beam pruning,
- before births,
- at boundary `b = k - N`,
- per logical `track_id`,
- using explicit ancestor node identity at that boundary,
- considering only surviving globals that still contain that `track_id`.

A track is considered committed at boundary `b` only when all participating surviving globals agree on the same exact-boundary ancestor node.

This is the intended Phase B semantic correction.

### Commitment bookkeeping is explicit

The tracker now keeps explicit internal commitment state and exposes a small read-only snapshot for debug/tests.

This makes the current N-scan state inspectable without treating node cleanup or committed-history materialisation as already solved.

### Small read-only MAP inspection helpers now exist

The tracker now exposes small read-only helpers for inspecting the current MAP hypothesis in the new structure:
- a node-native MAP snapshot,
- and a public helper for reconstructed MAP output tracks.

This reduces the need for runner/test code to reach into private reconstruction internals just to inspect the MAP view.

### Stone Soup tracker interface compliance is now explicit

`TOMHTTracker` now implements Stone Soup's tracker interface directly:
- it subclasses the tracker mixin/base interface,
- supports `update_tracker(time,detections)` returning `(time,tracks)`,
- supports iterator-driven progression when `detector` is set.

### Public API quick reference (integration-facing)

This is the intended public surface for current integration tasks:

- `update_tracker(time,detections) -> (time,tracks)`:
  - main per-scan entry point,
  - consumes one timestamp plus an iterable of `Detection`,
  - returns the same timestamp with current MAP-output tracks.
- `tracks` property:
  - current MAP-output tracks,
  - equivalent in content to `get_map_output_tracks()`.
- iterator mode (`for time, tracks in tracker`):
  - supported when `detector` is set,
  - each iteration delegates through the same `update_tracker(...)` path.
- `add_external_starts(time,starts)`:
  - for externally confirmed starts only,
  - requires a completed `update_tracker()` first,
  - `time` must match the most recent completed `update_tracker()` timestamp.
- `get_unused_detections()`:
  - returns residual detections from the most recent completed `update_tracker()`,
  - raises if called before the first completed update,
  - returns an empty list when internal births are enabled (`initiator is not None`), since residuals are treated as consumed by the birth path.
- `get_map_hypothesis_snapshot()`:
  - read-only node-native MAP leaf-node view for inspection/tests.
- `get_n_scan_commitment_snapshot()`:
  - read-only commitment-state snapshot for inspection/tests.
- `get_tomht_track_id(track)`:
  - TOMHT-specific helper for stable logical track identity extraction from `TOMHTTracker`-produced tracks,
  - reads `track.metadata["track_id"]`,
  - not intended as a generic Stone Soup `Track` helper.

## What is still transitional or awkward

The core structure is now much better, but the implementation is not “finished” in every respect.

### Stone Soup `Track` reconstruction is still an adapter boundary

The tracker still reconstructs temporary Stone Soup `Track` objects from leaf-node ancestry for:
- hypothesiser compatibility,
- updater compatibility,
- output,
- some debugging/display paths.

This is acceptable for now, but it is still a compatibility boundary rather than the ideal end-state.

### Reconstructed `Track.metadata` is now explicit

`TrackHypothesisNode` no longer carries an opaque metadata bag.
Reconstructed compatibility `Track` outputs now project explicit TOMHT-owned metadata keys only (for example `track_id`, node/counter/debug fields), instead of carrying through arbitrary input metadata.

### Physical node cleanup / GC is still deferred

Commitment semantics are now explicit, but physical node lifecycle cleanup is still deferred.

In particular, the current implementation does **not** yet:
- garbage-collect unreachable/orphaned ancestry aggressively,
- compact committed prefixes,
- or use commitment state to shrink memory structurally.

This was an intentional Phase B non-goal.

### Committed-history materialisation is still deferred

The tracker does not yet maintain a separate committed-history store, committed-track object model, or detached committed-prefix representation.

Committed branch decisions are now explicit, but committed output materialisation is still a later-phase design question.

### N-scan window configuration is now single-knob

`TOMHTParams.ns_scan_window` is now the sole N-scan window parameter.
Its default is `3`.

### Performance has not been the focus yet

Phase B prioritised structural correctness and semantic clarity.

The tracker is therefore in a much better architectural state, but it has **not** yet been pushed hard on:
- memory efficiency,
- ancestry compaction,
- avoiding reconstruction overhead,
- or broader scaling/performance cleanup.

## What is now solid operationally

### External confirmed-start support remains in place

The tracker still supports externally supplied confirmed starts via the external-start path.

Current semantics remain:
- external starts are injected via `add_external_starts(time, starts)` after a completed `update_tracker()` at the same timestamp,
- they are inserted structurally into the current global hypotheses,
- they are not routed through internal residual birth discovery,
- they do not receive internal birth penalties.

### Runner operating-mode support remains usable

The existing runner still supports:
- external-only operation,
- internal-only operation,
- both external starts and internal births,
- explicit operating-mode and external-start timing configuration.

So the integration-facing work from the previous phase remains intact while the tracker internals have become more structurally correct.

### Deterministic/stable experimentation remains practical

The tracker still has deterministic enough ordering / beam-management behavior to make inspection, debugging, and scenario comparison practical.

That remains important because the new structure is more correct, but it also needs to stay inspectable.

## What is still not ideal or still clearly “bad”

To be explicit about the remaining shortcomings:

- reconstructed `Track` views are still used at several compatibility boundaries,
- node lifecycle / GC is not yet designed or implemented,
- committed-history output/materialisation does not exist,
- performance/efficiency has not yet been revisited after the structural refactor.

None of these invalidate the Phase B result, but they are the main remaining sources of technical awkwardness.

## Bottom line

The tracker is now in a meaningfully better architectural state than before:
- explicit node-based ancestry exists,
- globals reference per-track leaf nodes,
- dedupe is structural,
- births and external starts live in the same hypothesis structure,
- and N-scan commitment is now explicit and ancestor-identity-based.

So the main Phase B architectural goal has been achieved.

The remaining issues are no longer “the tracker is structurally not a TO-MHT.”
They are now mostly compatibility, cleanup, lifecycle, and future-design questions.
