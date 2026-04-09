# TO-MHT Current State

## Snapshot date

This document describes the tracker as it exists after the Phase D track-oriented transition and the subsequent replay-hardening passes completed on **2026-04-03**.

It is a **current-state snapshot**, not a roadmap and not a full design history.

## Update (2026-04-09): transitional constructor boundary cleanup

Implemented in `mht/tomht_tracker.py`, `mht/runners/tomht_runner.py`, and `mht/tomht_scoring.py`:

- `TOMHTTracker` now takes `predictor` + `updater` as the primary constructor dependencies.
- local branching still uses a hypothesis-generator object internally; behavior is intentionally unchanged in this pass.
- constructor precedence is now:
  - explicit `hypothesis_generator`,
  - deprecated `hypothesiser` compatibility path,
  - internally built backend from `TOMHTParams.hypothesis_backend` (`"pda"` or `"robust_pda"`).
- default scoring setup no longer reads values from hypothesiser attributes; it now uses explicit tracker params (`prob_detect`, `prob_gate`, `clutter_density`).
- tracker construction for crossing/UKF is now inlined in `run_tomht(...)` (the thin `build_tomht_*` wrappers were removed); runner still sets backend defaults via params (`pda` for linear, `robust_pda` for UKF).

## Update (2026-04-03): Stone Soup 1.8 / Python 3.14 property-type compatibility

Implemented in `mht/tomht_tracker.py`:

- `TOMHTTracker` declares its Stone Soup `Property(...)` fields with a Python-version gate:
- for Python `>=3.14`, uses explicit `Property(Type,doc=...)` form,
- for Python `<3.14`, keeps annotation-driven `field: Type = Property(doc=...)` form,
- this preserves import compatibility across Stone Soup `1.7`/`1.8` and avoids the `Type was not specified for property ...` failure seen under Python 3.14.

## Update (2026-04-03): robust scalar conversion in custom hypothesiser

Implemented in `mht/helpers/hypothesiser.py`:

- added a helper that converts scalar-like ndarray/matrix outputs to a Python `float` via explicit shape/size handling,
- replaced direct `float(...)` conversion of quadratic forms in both gating and log-pdf paths,
- this avoids NumPy/Python-version dependent `TypeError: only 0-dimensional arrays can be converted to Python scalars` seen in bearing-range smoke runs under Python 3.12 CI.

## Update (2026-04-02): targeted structural cleanup pass

Implemented in `mht/tomht_tracker.py` without behavior changes:

- split MAP-only N-scan pruning into an explicit planning stage and a mutation/application stage,
- added a conservative internal cluster-solve boundary via private solve input/output structures and a `_solve_cluster(...)` wrapper,
- extracted optional historical-relaxation retry into a dedicated helper under that solver boundary,
- kept exhaustive enumeration as the underlying solve implementation.

## Update (2026-04-02): constructor-level TOMHTParams key overrides

Implemented in `mht/tomht_tracker.py`:

- `TOMHTTracker.__init__` now accepts `params_overrides` (`dict`/mapping),
- override keys are validated against `TOMHTParams` fields and applied via `dataclasses.replace(...)`,
- unknown keys fail fast with a clear `ValueError`,
- this enables JSON-driven tracker parameter overrides while preserving immutable `TOMHTParams`.

## Update (2026-04-02): runner CLI JSON override plumbing

Implemented in `mht/runners` and tracker builders:

- `run_tomht_crossing.py` and `run_tomht_bearing_range.py` now accept `--params-override <path>` (JSON file),
- runner helper `load_params_overrides_json(...)` loads and validates a top-level JSON object,
- loaded overrides are threaded through `run_tomht(...)` and builder helpers to `TOMHTTracker(...,params_overrides=...)`.

## Update (2026-04-02): Stone Soup import-time compatibility fix

Implemented in `mht/tomht_tracker.py`:

- removed duplicate type specification for tracker class properties,
- adopted Stone Soup's typed pattern (`field: Type = Property(doc=...)`) so annotations provide property type metadata,
- this fixes import-time `BaseMeta` errors triggered when both annotation and `Property` type are set,
- Python 3.14/Stone Soup 1.8 follow-up is covered by the 2026-04-03 update above.

## Update (2026-04-02): tracker readability-only cleanup

Implemented in `mht/tomht_tracker.py` without behavior changes:

- clearer internal section/subsection grouping aligned to the runtime pipeline,
- tighter role-focused docstrings on helper methods,
- small local naming/comment cleanups for readability,
- removal of stale wording in one safety-valve comment.

---

## Bottom line

The tracker is now a **true track-oriented TO-MHT implementation** in the practical sense used in this codebase:

- persistent state is explicit `TrackTree` objects and their active leaves,
- globals are rebuilt per cluster on every scan from current leaves,
- the previous scan's explicit global list is **not** the persistent search frontier,
- MAP-only N-scan pruning operates directly on explicit trees,
- and current output tracks are reconstructed from retained leaf-node lineage.

This is stated both in the tracker module header and in the public tracker class docstring. The code now treats track trees as the primary scan-to-scan state, with rebuilt globals retained only as last-scan inspection/debug artifacts.

The current implementation is therefore:

- structurally aligned with the intended track-oriented TO-MHT direction,
- usable for replay-based experimentation and continued integration work,
- reasonably robust on recorded replay through end-of-file,
- but still **performance-limited** in large merged-cluster situations and still reliant on a few explicit approximation/safety-net mechanisms.

---

## Public API and integration boundary

The intended operational public surface remains:

- `update_tracker(time, detections) -> (time, tracks)`
- `tracks`
- `add_external_starts(time, starts)`
- `get_unused_detections()`

The tracker also exposes read-only inspection helpers such as:

- `get_map_hypothesis_snapshot()`
- `get_map_output_tracks()`
- `get_n_scan_commitment_snapshot()`
- `get_last_cluster_snapshots()`
- `get_track_tree_snapshot()`
- `print_summary_stats()`

The external interface remains Stone Soup-oriented:

- detections are Stone Soup `Detection`,
- output tracks are Stone Soup `Track`,
- predictor and updater are now the primary tracker constructor boundary objects,
- local hypothesis generation is still PDA-style in this phase via a transitional backend selector (`hypothesis_backend`) or explicit injection,
- output `Track` metadata is now an explicit TOMHT-owned projection from the current leaf node rather than arbitrary propagated metadata.

---

## Core architecture

### Persistent state

Persistent scan-to-scan state now consists primarily of:

- explicit `TrackTree` objects keyed by logical `track_id`,
- persistent `TrackHypothesisNode` objects linked by same-track parent/child structure,
- each tree's current root and active leaf set,
- N-scan commitment bookkeeping,
- and minimal long-lived stats/counters.

`TrackHypothesisNode` is the canonical per-step hypothesis unit. Nodes are mutable in this phase to support direct child-link maintenance through `child_node_ids`. Each node carries:

- stable `track_id`
- stable `node_id`
- same-track `parent`
- `scan_index` / `timestamp`
- state payload
- used detection key / association label
- local and accumulated score
- cached age / hit / miss counters
- root provenance fields such as `root_source` and `birth_scan_index`.

`TrackTree` is now explicit and persistent, with:

- `track_id`
- `root_node_id`
- `active_leaf_node_ids`
- `root_source`.

### Per-scan transient / last-scan state

Per scan, the tracker rebuilds:

- clusters from current active-leaf history overlap,
- cluster-local rebuilt globals,
- cluster MAP selections,
- overload-split summaries when triggered,
- and scan statistics.

These rebuilt artifacts are retained only as **last-scan inspection/debug snapshots**, not as the persistent search frontier. The compatibility slot `self.global_hypotheses` now contains only the latest merged MAP global for older inspection paths, not a scan-to-scan beam frontier.

---

## Current per-scan pipeline

The tracker class docstring now describes the pipeline as:

1. sort detections deterministically,
2. expand active leaves in every persistent tree,
3. drop empty trees,
4. optionally create internal birth trees from detections unused by the union of surviving active leaves,
5. recompute clusters from current trees,
6. rebuild feasible globals per cluster by exhaustive enumeration, with optional overload splitting first,
7. post-solve prune each cluster tree frontier to leaves supported by retained rebuilt globals,
8. merge cluster MAP selections into full-scan MAP, apply MAP-only N-scan pruning, then apply whole-track miss lifecycle,
9. keep last-scan debug snapshots and return MAP output tracks.

This is the main runtime story the code now implements.

---

## Current rebuild / pruning / commitment behavior

### Local expansion

Local expansion is still Stone Soup-boundary-driven:

- for each active leaf, reconstruct a compatibility `Track`,
- call the hypothesiser,
- score local hypotheses through the scoring model,
- create child nodes for kept hypotheses,
- always keep a miss hypothesis if the hypothesiser returned one,
- then apply an optional per-tree local leaf cap.

### Local leaf cap

`max_leaves_per_track_tree` is now explicitly treated as a **pre-solve safety valve**, not the main pruning semantics. Its default has been raised to `500` to keep it in that “safety-valve only” role.

### Clustering

Clusters are built from **full active-leaf historical detection-key overlap**, not current-scan-only overlap. This was an important correctness fix: clustering and solver feasibility now use consistent full-history detection-key semantics. Detection keys use the format `(scan_index, det_index)`.

### Global rebuild

For each cluster, the tracker currently uses:

- exhaustive Cartesian enumeration across active leaf sets,
- full-history exclusivity checks,
- an explicit per-combination cluster-local unused-detection term,
- streaming top-K retention of rebuilt globals,
- and cluster-local MAP extraction.

`max_global_hypotheses` is now only a **retained rebuilt-global cap for debug/snapshot storage**, not a persistent beam width carried scan-to-scan.

### Post-solve supported-leaf pruning

After each cluster rebuild, each non-overload-split cluster tree keeps only leaves that appear in at least one retained rebuilt global for that cluster. This is now the main pruning mechanism that keeps active leaf frontiers globally informed.

### MAP-only N-scan pruning

N-scan pruning remains MAP-only:

- boundary is `b = scan_index - ns_scan_window`,
- the child of the current root on the MAP path is promoted to be the new root,
- siblings are removed structurally,
- and disagreement statistics are computed against alternative rebuilt globals.

The default N-scan window is now `6`.

### Whole-track miss lifecycle

Per-branch miss-based pruning during local expansion has been removed.

Miss handling now happens as **whole-track termination after N-scan pruning**, using:

- configurable `track_miss_termination_mode`
  - `all_active_leaves`
  - `map_leaf` (default)
  - `global_k_leaves`
- threshold `max(max_missed, ns_scan_window + 1)`

This was a significant cleanup relative to earlier branch-local miss handling.

---

## Current approximation / safety-net mechanisms

The tracker now contains several explicit approximation/safety-net mechanisms that are part of the current operational semantics.

### 1. Overload cluster splitting

When a cluster's projected Cartesian combinations exceed `overload_split_projected_combination_threshold` (default `500000`), the tracker can approximately decompose that cluster by:

- building the exact conflict graph first,
- iteratively severing the weakest conflict edge,
- recomputing connected components,
- and solving resulting subclusters independently.

Weakest-edge criterion is the pure count of shared **full-history** detection keys, with deterministic tie-break. This approximation is instrumented through `OVERLOAD_SPLIT ...` logging and per-scan split counters.

### 2. Historical-conflict relaxation

If a cluster is still exact-infeasible, the tracker may apply a **narrow historical-conflict relaxation**:

- only keys forced in every active leaf of a track,
- only keys also present in that track's root history,
- only keys at or older than the current N-scan boundary,
- and only when shared by more than one track in the cluster.

Feasibility is then retried while ignoring overlaps on those specific historical keys only. All other exclusivity remains strict. This path is instrumented through `HIST_RELAX ...` logging and per-scan counters.

### 3. Internal birth load guards

Internal births remain intentionally simple and secondary. They are still based on the constructor-supplied initiator and Step-2 residual detections, but births can be skipped once active tree or leaf counts exceed configured thresholds.

These mechanisms are pragmatic, not final. They should be understood as explicit robustness/tractability measures for current replay use, not a final principled solution to large-cluster or extended-target behavior.

---

## Scoring state

Scoring remains based on the default beta-ratio-style model in `tomht_scoring.py` unless an explicit alternative scoring model is supplied.

Current default behavior:

- scores local track hypotheses using PDA-style β-ratio approximations,
- scores unused detections through a clutter-density-derived per-unused term,
- applies a fixed birth penalty for births,
- logs scoring diagnostics at tracker construction time.

This scoring is still best understood as **pragmatic and serviceable**, not final.

---

## Output / observability

Output tracks are reconstructed from leaf-node lineage through `reconstruct_track_from_leaf_node(...)`.

Returned `Track` metadata is now an explicit projection from the current leaf node, including:

- stable logical `track_id`
- `node_id`
- `age`
- `hits`
- `missed_count`
- `last_det_key`
- `last_det_hit`
- `root_source`
- `birth_scan_index`.

Per-scan and summary instrumentation now reports:

- active trees / leaves
- cluster counts
- evaluated / feasible combinations
- rebuilt globals stored
- overload split counters
- historical relaxation counters
- N-scan commitment counts
- birth statistics
- MAP track usage
- scan wall time
- memory / node counts.

This instrumentation has been important for replay diagnosis and is now part of the tracker’s practical observability story.

---

## Replay/runtime status

The current code can now complete the full recorded replay to end-of-file, which was the immediate near-term robustness goal. The remaining issue is no longer correctness instability, but runtime concentration in high-combinatoric merged-cluster scans.

Current runtime snapshot from the existing `CURRENT_STATE` notes:

- median scan time: ~31 ms
- p90: ~714 ms
- p95: ~1191 ms
- max: ~4576 ms

The long scans are concentrated where clusters merge and still produce large `comb_eval` / `comb_feas` counts even after overload splitting and historical relaxation.

So the tracker is now in a state that is:

- robust enough for replay-based experimentation,
- producing reasonable output on the recorded dataset checkpoint,
- but still clearly in need of runtime optimization for heavy merged-cluster scans.

---

## What is solid now

The following now look solid enough to treat as the current base architecture:

- explicit `TrackTree` + `TrackHypothesisNode` persistent state,
- scan-to-scan persistence through trees rather than globals,
- full-history `(scan_index, det_index)` detection-key semantics,
- per-scan rebuilt cluster globals,
- post-solve supported-leaf pruning,
- MAP-only N-scan pruning directly on trees,
- whole-track post-N-scan miss lifecycle,
- Stone Soup boundary compatibility,
- replay integration and end-to-end recorded replay completion.

---

## What remains provisional / future-work territory

The following are still provisional or explicitly not the final word:

### 1. Runtime / solver story
The current rebuild step is still exhaustive enumeration. Large merged clusters remain expensive even with overload splitting. A more scalable K-best global solver remains a clear future optimization target.

### 2. Overload splitting semantics
Overload splitting is useful and explicit, but still an approximation. It can later induce committed historical overlap, which is why historical relaxation exists. This is acceptable for now, but not conceptually final.

### 3. Historical-relaxation safety net
Historical conflict relaxation is narrow and pragmatic. It is a good safety net, but not the final principled treatment of approximation-induced overlap.

### 4. Internal births
Internal births remain simple, heuristic, and secondary. The external-start path is still the more important operational start path for integration work.

### 5. Scoring design
The beta-ratio scoring model remains pragmatic rather than fully settled.

### 6. False starts / tracking quality tuning
Replay output is now usable enough that false starts and similar quality issues can be revisited later, but they are not the main current blocker.

---

## Recommended interpretation of the current checkpoint

This checkpoint should be understood as:

- a **real track-oriented TO-MHT implementation**
- with a coherent persistent-tree / rebuilt-global architecture
- that now survives the target recorded replay end-to-end
- and is suitable for continued replay-based evaluation and cleanup work

But it should also be understood as:

- still pragmatically tuned,
- still carrying a few explicit approximation/safety-net paths,
- and still performance-limited on large merged clusters.

That is a good place to begin a cleanup/consolidation phase before the next round of deeper work.
