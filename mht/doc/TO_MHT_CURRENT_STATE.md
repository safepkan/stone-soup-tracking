# TO-MHT Current State

## Update (2026-03-31): Phase D architecture in place

The tracker has now been moved to the track-oriented Phase D architecture:

- persistent explicit track trees are now the primary scan-to-scan state
- globals are rebuilt per scan/per cluster from current active leaves
- previous scan global lists are no longer the persistent search frontier
- MAP-only N-scan pruning is applied directly on explicit trees
- measurement-exclusivity clustering is explicit and recomputed each scan
- internal births and external starts are represented as new trees under the simplified Phase D assumptions
- an explicit per-tree frontier cap (`max_leaves_per_track_tree`) is now used to keep the first exhaustive solver implementation tractable on longer runs

This document is the **current-state snapshot for the first ISAC handoff release**.

It describes the state of the code that was presented in the workshop and handed off into the ISAC sandbox area for initial integration work. It is intentionally a **snapshot document**, not a roadmap and not a full design history.

## Bottom line

The current implementation is:

- usable enough for first handoff and initial integration,
- structurally much improved compared with the earlier copied-`Track` / history-tail version,
- Stone Soup-facing at the public API boundary,
- explicit in its node-based track-hypothesis representation,
- but **not yet a true track-oriented TO-MHT** in the stricter sense described in the workshop.

The key remaining architectural gap is:

- the tracker still carries an explicit **current frontier of global hypotheses** from one update to the next,
- rather than treating track trees / track hypotheses as the main persistent state and rebuilding globals from the current track set at each scan.

That distinction is now understood clearly and is expected to drive the next architectural phase.

---

## What is solid in the current handoff release

### 1. Public API and integration boundary

The intended public tracker surface is now clear and usable for integration:

- `update_tracker(time, detections) -> (time, tracks)`
- `tracks`
- `add_external_starts(time, starts)`
- `get_unused_detections()`
- `get_map_hypothesis_snapshot()`
- `get_map_output_tracks()`
- `get_n_scan_commitment_snapshot()`
- `get_tomht_track_id(track)`

The external interface is Stone Soup-oriented:

- detections are Stone Soup `Detection`
- output tracks are Stone Soup `Track`
- hypothesiser and updater are Stone Soup-facing
- the tracker follows Stone Soup tracker usage patterns closely enough for initial integration work

### 2. Explicit node-based track hypotheses

The internal representation now has an explicit track-hypothesis node model:

- `TrackHypothesisNode` is the canonical internal branching unit
- each node has:
  - stable `track_id`
  - stable `node_id`
  - same-track `parent`
  - `scan_index` / `timestamp`
  - per-step state payload
  - association identity / used detection key
  - per-step score contribution
  - cached counters / provenance fields

This is a substantial structural improvement over the earlier design where global hypotheses carried copied `Track` objects more directly.

### 3. Globals reference leaf nodes

`GlobalHypothesis` now stores:

- `track_id -> leaf node`
- cumulative `log_weight`

So:

- ancestry is structural,
- branch identity is node-based,
- dedupe is structural,
- and current MAP output can be reconstructed from leaf-node lineage.

### 4. Explicit ancestor-based N-scan commitment

N-scan commitment is now explicit and per-track:

- commitment is evaluated after global pruning
- boundary is `b = k - N`
- commitment is based on exact ancestor-node identity at that boundary
- commitment bookkeeping is inspectable through a read-only snapshot helper

This is a real correction relative to the earlier history-tail approximation.

### 5. External starts are supported

The tracker supports externally confirmed starts:

- `add_external_starts(time, starts)` inserts starts after the corresponding `update_tracker(...)`
- starts enter the same node-based structural system as other track hypotheses
- this is the most relevant near-term birth/start path for the ISAC use case

### 6. Replay integration and basic validation exist

The tracker has already been exercised through:

- simple synthetic scenarios,
- local replay integration through a Stone Soup adapter,
- output comparison / repeatability checks,
- debug instrumentation,
- memory monitoring.

A conservative post-commit ancestry cleanup pass was added and materially improved retained-node growth and replay memory behavior, while preserving logical outputs.

---

## What is still transitional or conceptually awkward

### 1. Not yet a true track-oriented TO-MHT update

This is now the main architectural caveat.

The implementation has moved toward TO-MHT structurally, but the update mechanics still work by carrying an explicit current set of global hypotheses forward from scan to scan and expanding those.

So the current tracker is best described as:

- **node-based and much closer to a proper TO-MHT structurally**
- but still retaining some **global-hypothesis-oriented mechanics**

More specifically:

- globals are still part of the persistent scan-to-scan state,
- expansion is still phrased as expansion of current globals,
- structural dedupe is still needed because different parent globals can converge to the same successor leaf configuration.

This is no longer viewed as the desired long-term architecture.

### 2. Track trees are still implicit

There is currently no explicit `TrackTree` structure.

Instead:

- same-track ancestry is represented by `parent` links on nodes,
- the tracker implicitly treats those same-track chains as track trees / families,
- but there is no explicit root/leaf container for one logical track.

That was acceptable for the first handoff, but is expected to change in the next architectural phase.

### 3. Stone Soup `Track` reconstruction is still a compatibility boundary

Temporary or reconstructed Stone Soup `Track` objects are still used for:

- hypothesiser compatibility,
- updater compatibility,
- output generation,
- some inspection/debug paths.

This is acceptable for now, but it is still an adapter boundary rather than the clean end-state.

### 4. Scoring is still pragmatic

Current scoring is still based on pragmatic assumptions:

- extracting probability-like information from hypothesiser outputs,
- using a beta-ratio style scoring model,
- heuristic handling for births / unused detections.

This is considered acceptable for the handoff release, but not a settled final scoring design.

### 5. Internal births remain secondary and somewhat provisional

Internal births still exist and function, but:

- they are not the important near-term path for ISAC,
- they are not the main focus of the architecture,
- their treatment is more heuristic than principled.

The external-start path is the more important operational path for the current integration story.

### 6. Lifecycle / deletion is not yet properly designed

Lifecycle handling remains incomplete:

- `max_missed` currently acts as a per-hypothesis miss budget in expansion logic,
- but there is no clean long-term deletion/lifecycle model yet,
- and multi-sensor miss handling remains a future design topic.

### 7. Physical cleanup is conservative only

The current ancestry cleanup pass is narrow and conservative:

- it reclaims unreachable node-registry entries,
- but does not implement committed-prefix compaction,
- does not materialize committed history separately,
- and does not yet represent a broader node lifecycle policy.

---

## What this first handoff should be understood as

This release should be understood as:

- a **stable-enough integration starting point**
- with a **clear public API**
- and a **much cleaner internal representation than before**

But it should **not** be understood as:

- the final tracker architecture,
- the final TO-MHT update mechanics,
- the final scoring design,
- or the final lifecycle / performance story.

The purpose of this handoff is:

- let the ISAC side start wiring the tracker into their pipeline,
- validate assumptions and expose integration issues early,
- and give both sides a concrete implementation to work from while the architecture continues to evolve.

---

## Current design interpretation

The clearest current interpretation is:

- **conceptually**, proper TO-MHT should persist track trees / track hypotheses and rebuild globals from the current surviving track set at each scan,
- **currently**, the implementation has adopted explicit node-based track hypotheses and explicit ancestor-based N-scan commitment,
- but still evolves an explicit scan-to-scan frontier of global hypotheses.

That is now the main architectural distinction to keep in mind when discussing the current handoff release.

---

## Immediate implications for the next phase

The next architectural phase is therefore expected to focus on:

- explicit track trees,
- clearer separation between persistent and transient state,
- rebuilding globals from current track hypotheses instead of carrying old globals forward directly,
- cleaner pruning/cluster/solver structure,
- and preserving the current public API where practical.
