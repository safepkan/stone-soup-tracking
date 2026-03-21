# TO-MHT Current State

This document is a current snapshot of the implementation after completing Phase B: explicit track-hypothesis structure and true ancestor-based N-scan commitment.

It replaces the earlier pre-Phase-B description in which globals still stored copied `Track` objects and N-scan was only a history-tail approximation. That older description is no longer accurate. `TO_MHT_NEXT_STEPS.md` remains the main phase-planning document; this file is the implementation-status snapshot. 

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

## What is still transitional or awkward

The core structure is now much better, but the implementation is not “finished” in every respect.

### Stone Soup `Track` reconstruction is still an adapter boundary

The tracker still reconstructs temporary Stone Soup `Track` objects from leaf-node ancestry for:
- hypothesiser compatibility,
- updater compatibility,
- output,
- some debugging/display paths.

This is acceptable for now, but it is still a compatibility boundary rather than the ideal end-state.

### `track_metadata` is still carried on nodes

`TrackHypothesisNode` still carries a `track_metadata` dict, and reconstructed `Track.metadata` is still partly projected from it.

This is now mostly a compatibility residue rather than a core architectural concept.

It does not block the current structure from being a real node-based TO-MHT, but it is one of the clearest remaining pieces of “old world” flavor in the implementation.

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

### Some compatibility knobs remain

`assoc_history` metadata projection has been removed, but `assoc_history_len` still exists in `TOMHTParams` as a legacy compatibility/defaulting knob for `ns_scan_window` when `ns_scan_window <= 0`.

This is not harmful, but it is a leftover compatibility feature rather than a clean long-term concept.

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
- external starts are injected after a completed `step()` at the same timestamp,
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
- `track_metadata` is still propagated on nodes,
- node lifecycle / GC is not yet designed or implemented,
- committed-history output/materialisation does not exist,
- the `assoc_history_len` parameter is still a legacy compatibility relic,
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
