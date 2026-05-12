# TO-MHT Current State

## Snapshot date

This document describes the tracker as it exists after the track-oriented TO-MHT transition and the subsequent replay-hardening, determinism fixes, output-history restoration, solver-seam extraction, exact-backend experiments, local-association ownership refactor, explicit NLL scoring cleanup, and local-association optimization work completed through **2026-04-16**.

It is a **current-state snapshot**, not a roadmap and not a full design history.

Update (2026-04-20): core TO-MHT modules now use package-relative intra-module imports (for example `.tomht_model`, `.tomht_cluster_solver`) so the tracker package can be relocated within the repo without hard-coding `mht` in internal dependencies.
Update (2026-04-20): shared tracker/scoring context typing now starts in `mht/tomht_types.py` (`ScanContext`), reducing cross-module coupling and making room for additional shared type definitions.
Update (2026-04-20): runtime utility helpers now live in `mht/utils.py` (`env_flag`, `env_float`, `ns_to_ms`, cross-platform `get_process_maxrss_mb`) and are reused by tracker and OR-Tools profiling paths.
Update (2026-04-20): TOMHT output `Track.id` is now stable by default (internal integer track ID instead of Stone Soup auto-UUID), and constructor injection (`output_track_id_mapper`) allows mapping internal integer IDs to integration-specific public ID objects while keeping TOMHT internals dependency-free.
Update (2026-04-20): post-N-scan whole-track lifecycle now has two lanes: default node-native miss-threshold policy (`max_missed`) and optional injected Stone Soup `Deleter` policy (`deleter=` in tracker constructor). `track_miss_termination_mode` remains the leaf-selection mode for either lane.
Update (2026-04-20): expansion timing instrumentation now splits `expand_ms` into explicit hypothesiser and updater components (`expand_hypothesise_ms`, `expand_update_ms`) plus call counts, so replay timing can attribute expansion cost between `hypothesise()` and `update()` directly.
Update (2026-05-11): `TOMHTParams` now lives in `mht/tomht_params.py` so extracted helper modules can depend on tracker configuration without importing `TOMHTTracker`; `tomht_tracker.py` still re-exports it for compatibility. Local expansion orchestration now lives in `mht/tomht_expansion.py`, internal-birth candidate helpers now live in `mht/tomht_births.py`, cluster work construction and overload cluster decomposition now live in `mht/tomht_clustering.py`, post-solve supported-leaf pruning now lives in `mht/tomht_pruning.py`, and TOMHT-specific scan/debug helpers now live in `mht/tomht_utils.py`. `TOMHTTracker` still orchestrates the same pipeline, but local expansion/candidate handling, internal-birth residual/candidate utilities, full-history cluster construction, overload-split transformation, retained-global leaf-support pruning, detection sorting, detection-key filtering, and detection-key debug formatting are now isolated behind narrow helper functions.
Update (2026-05-12): persistent tree/node bookkeeping now lives in `mht/tomht_tree_store.py`. Stable ID allocation, node/root creation, single-root tree insertion, active-leaf bookkeeping, active count helpers, empty-tree removal, and unreachable-node cleanup are now owned by `TrackTreeStore`, while `TOMHTTracker` keeps compatibility properties for direct tree/node table inspection.

---

## Bottom line

The tracker is now a **real track-oriented TO-MHT implementation** in the practical sense used in this codebase:

- persistent scan-to-scan state is explicit `TrackTree` objects and their active leaves,
- globals are rebuilt per cluster on every scan from current leaves,
- the previous scan’s explicit global list is **not** the persistent search frontier,
- MAP-only N-scan pruning operates directly on explicit trees,
- and public output tracks are reconstructed from **committed prefix history plus current unresolved lineage**.

The code treats track trees as the primary persistent state, with rebuilt globals retained only as last-scan inspection/debug artifacts.

The current implementation is therefore:

- structurally aligned with the intended track-oriented TO-MHT direction,
- usable for replay-based experimentation and continued integration work,
- reasonably robust on the main recorded replay through end-of-file,
- on a materially better exact-solver footing than the earlier exhaustive baseline,
- and now on a much clearer local-association/scoring baseline than the earlier PDA/beta-oriented transitional path.

The main remaining practical weakness is no longer the exact cluster solver itself. With the default branch-and-bound backend in place and the recent local-association math cleanup completed, the main replay bottleneck remains **local expansion / hypothesis generation**, with the next likely leverage point being reduction of **expansion volume** rather than further inner-kernel cleanup alone.

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
- constructor now accepts exactly one of:
  - `predictor` (+ tracker-owned default distance hypothesiser), or
  - `hypothesiser` (custom distance hypothesiser injection),
- `updater` remains required in both constructor modes,
- local expansion consumes distance hypotheses directly rather than PDA-normalized probabilities,
- output `Track` metadata is an explicit TOMHT-owned projection from the current active leaf node rather than arbitrary propagated metadata.

### Constructor and local-association boundary

The tracker now has a narrow **distance-hypothesiser seam**:

- input: `(track, detections, timestamp)`
- output: Stone Soup `MultipleHypothesis` containing exactly one missed
  `SingleDistanceHypothesis` plus zero or more gated detection
  `SingleDistanceHypothesis` entries
- each detection hypothesis carries `distance = NLL = -log p(z|x)` in measurement
  space, **without** detection-probability or clutter-density factors,
- missed-detection distance is a sentinel and is ignored by local score
  construction.

Default local association is tracker-owned
`TrackerOwnedNLLDistanceHypothesiser` and uses explicit non-squared
Mahalanobis-threshold gating semantics via `mahalanobis_gate_threshold`.

This split is now explicit:

- **hypothesiser owns local distances and gating**
- **scoring model owns local NLL-to-LLR conversion plus tracker-level extras**

`ScoringModel` currently owns:

- `score_track_hypotheses(...)`
- `score_unused_detections(...)`
- `score_birth(...)`

`used_det_keys` / `used_det_key` are local detection indices into
`ScanContext.detections` for the scoring call.

The current solver pre-baking path assumes
`score_unused_detections(...)` is affine in the number of used detections.
Broader/non-linear alternatives are deferred to a later scoring redesign.

---

## Core architecture

### Persistent state

Persistent scan-to-scan state now consists primarily of:

- explicit `TrackTree` objects keyed by logical `track_id`,
- persistent `TrackHypothesisNode` objects linked by same-track parent/child structure,
- each tree’s current unresolved root and active leaf set,
- each tree’s committed prefix history before the unresolved root,
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
- provenance fields such as `root_source` and `birth_scan_index`

`TrackTree` is explicit and persistent, with:

- `track_id`
- `root_node_id`
- `active_leaf_node_ids`
- `root_source`
- `committed_states`

### Per-scan transient / last-scan state

Per scan, the tracker rebuilds:

- clusters from current active-leaf history overlap,
- cluster-local rebuilt globals,
- cluster MAP selections,
- overload-split summaries when triggered,
- and scan statistics.

These rebuilt artifacts are retained only as **last-scan inspection/debug snapshots**, not as the persistent search frontier. The compatibility slot `self.global_hypotheses` now contains only the latest merged MAP global for older inspection paths, not a scan-to-scan beam frontier.

Cluster work construction is a helper-layer responsibility: it takes the
`TrackTreeStore` and the scan index, then returns
deterministically ordered cluster work items with full-history conflict links and
current-scan detection-key metadata. Overload decomposition is also handled in
that helper layer as an explicit transformation from one cluster work item plus
per-track active leaf counts into subcluster work items and instrumentation. This
preserves the existing full-history exclusivity and weak-edge split semantics
while keeping the tracker class focused on orchestration.

Local expansion is also a helper-layer responsibility: `mht/tomht_expansion.py`
owns distance-hypothesis validation, local candidate scoring/ranking, mandatory
miss preservation, updater calls, pre-solve leaf capping, and expansion timing
counters. Persistent node ID allocation and node registration are owned by
`mht/tomht_tree_store.py`; expansion receives the `TrackTreeStore` directly and
uses it for persistent child-node creation.

TOMHT-specific utility helpers are separated from generic runtime utilities:
`mht/tomht_utils.py` owns deterministic detection sorting, current-scan
detection-key filtering, and compact detection-key sample formatting, while
`mht/utils.py` remains for generic environment/runtime helpers.

Internal-birth handling is split out narrowly: `mht/tomht_births.py` owns birth
used-key extraction, sanity checks, support/age/miss summaries, covariance-trace
ranking, deterministic candidate sorting/capping, residual detection-index
calculation after expansion, guardrail reasoning, initiator invocation, birth
debug printing, birth scoring calls, and root-field construction. `TOMHTTracker`
calls the high-level post-expansion birth helper, assigns `_last_unused_detections`
from the helper result, and keeps scan-level orchestration, while the store owns
ID allocation, root-node creation, and tree insertion.

Persistent tree/node bookkeeping is centralized in `mht/tomht_tree_store.py`:
`TrackTreeStore` owns logical track IDs, node IDs, the node table, the track-tree
table, root/child node construction, single-root tree insertion, active leaf/tree
counts, empty-tree removal, and unreachable-node cleanup. Expansion, clustering,
and post-solve pruning now take the store as their persistent-state dependency.
The store also provides a narrow new-track root insertion helper for internal
births and external starts. `TOMHTTracker` keeps `track_trees_by_track_id` and
`_nodes_by_id` as compatibility properties that forward to the store; tracker
implementation code refers to `self._tree_store` directly.

---

## Current per-scan pipeline

The tracker’s current runtime pipeline is:

1. sort detections deterministically,
2. expand active leaves in every persistent tree,
3. drop empty trees,
4. optionally create internal birth trees from detections unused by the union of surviving active leaves,
5. recompute clusters from current trees,
6. rebuild feasible globals per cluster through the exact cluster-solver contract (default backend = branch-and-bound exact search), with optional overload splitting first and optional narrow historical-relaxation retry around the exact solve,
7. post-solve prune each cluster tree frontier to leaves supported by retained rebuilt globals,
8. merge cluster MAP selections into full-scan MAP, apply MAP-only N-scan pruning, then apply whole-track lifecycle (node-native miss-threshold by default, optional Stone Soup deleter when configured),
9. reclaim unreachable node storage, keep last-scan debug snapshots, and return MAP output tracks.

This is the main runtime story the code now implements.

---

## Solver architecture and current exact backend status

A dedicated solver seam now exists:

- `mht/tomht_cluster_solver.py` defines the solver-facing exact cluster problem/result contract and shared helpers,
- `mht/tomht_cluster_solver_exhaustive.py` contains the exhaustive reference backend,
- `mht/tomht_cluster_solver_branch_and_bound.py` contains the current default exact backend,
- `mht/tomht_cluster_solver_ortools.py` contains an experimental exact CP-SAT backend.

The exact cluster problem contract now explicitly carries:

- one leaf option per track choice,
- full-history conflict keys for feasibility,
- pre-scored leaf accumulated scores,
- and a ranking-inert `constant_score_offset`.

Tracker-side problem preparation folds the current linear current-scan clutter correction into leaf scores before solve, leaving only a cluster-constant offset outside the optimizer.

Approximation/policy placement is explicit:

- overload splitting remains a tracker-side pre-solve policy,
- historical-conflict relaxation remains a tracker-side around-solver retry policy.

### Current default exact backend: branch-and-bound

The tracker default backend is now `TOMHTParams.cluster_solver_backend="branch_and_bound"` (aliases `"branch-and-bound"` / `"bnb"`).

This backend:

- performs deterministic depth-first exact search over ordered tracks,
- uses an exact 1-track fast path,
- uses full-history conflict-key exclusivity exactly,
- uses deterministic ordering:
  - tracks ordered by fewer leaves first, then stronger conflict burden, then `track_id`,
  - leaves ordered per track by descending score,
- uses a simple optimistic suffix-score upper bound for pruning,
- retains exact K-best solutions through the shared deterministic `TopKSolutionHeap`.

During this phase, branch state was tightened from Python `set`-based conflict tracking to compact integer conflict masks for shared keys, and selected-leaf branch state was reduced from per-branch dict churn to depth-indexed arrays before materializing final solutions.

Branch-and-bound diagnostics now include counters such as:

- `search_nodes_visited`
- `branches_pruned_conflict`
- `branches_pruned_bound`
- `complete_feasible_solutions`

`ClusterSolverDiagnostics` is intentionally a union-style schema:

- required across backends: `combinations_evaluated`, `feasible_combinations`,
- common fields: backend/termination/result summary fields,
- backend-specific optional fields for solver-local instrumentation.

In particular, `solves_attempted` is treated as backend-specific and is primarily meaningful for repeated-solve backends (currently OR-Tools CP-SAT), rather than being forced to `1` by single-pass backends.

### Exhaustive backend status

The exhaustive backend remains available as:

- exact reference implementation,
- parity oracle for tests and solver experiments,
- fallback backend when needed.

It is no longer the default.

### OR-Tools backend status

An experimental exact CP-SAT backend also exists behind the same solver contract.

It uses:

- one Boolean variable per leaf,
- exactly-one selection per track,
- per-history-key exclusion constraints,
- scaled integer objective coefficients,
- repeated optimal solve calls plus no-good cuts for K-best extraction.

This backend is **exact under the current solver contract**, but in the current repeated-solve K-best form it is **not** a runtime win on the primary replay workload used during this phase. Profiling showed:

- solve time dominated by repeated CP-SAT solve calls rather than Python-side setup,
- small-cluster overhead exists but is not the main cost driver,
- large clusters plus repeated K-best solves dominate elapsed time.

Current positioning:

- keep OR-Tools as an **experimental exact backend**,
- useful for comparison, fallback, and future hybrid/K-best experiments,
- not recommended as the default runtime path in the current configuration.

---

## Current rebuild / pruning / commitment behavior

### Local expansion

Local expansion is now explicitly **distance-hypothesis driven**:

- for each active leaf, reconstruct a compatibility `Track`,
- call the configured distance hypothesiser,
- require one missed-detection hypothesis plus zero or more gated detection hypotheses,
- derive local score via `ScoringModel.score_track_hypotheses(...)`,
- create child nodes for kept hypotheses,
- always retain a miss hypothesis,
- then apply an optional per-tree local leaf cap.

### Current local-association math kernel

The tracker-owned default local hypothesiser currently includes several conservative optimizations:

- rectangular pre-gating before full Mahalanobis/NLL work,
- one-entry exact-equality covariance-preparation reuse per `hypothesise(...)` call,
- scan-time prediction reuse when `detection_timestamp == timestamp`,
- one-entry measurement-prediction reuse when both `prediction` and `measurement_model` match by object identity,
- Cholesky-based Mahalanobis/NLL evaluation:
  - prepared covariance payload includes SPD covariance, diagonal, Cholesky factor `L`, and `logdet`,
  - `logdet = 2 * sum(log(diag(L)))`,
  - full Mahalanobis distance uses triangular solve on `L`.

Regression status for this math path remained exact on the normalized smoke/replay compare helpers during the phase, and timing helpers showed reduced `expand_ms` in both smoke scenarios and the standard replay baseline.

### Local leaf cap

`max_leaves_per_track_tree` is explicitly treated as a **pre-solve safety valve**, not the main pruning semantics. Its default is intentionally high enough that it should act as tractability protection rather than the primary meaning of pruning.

### Clustering

Clusters are built from **full active-leaf historical detection-key overlap**, not current-scan-only overlap. This is an important correctness property: clustering and solver feasibility now use consistent full-history exclusivity semantics. Detection keys use the format `(scan_index, det_index)`.

### Global rebuild

For each cluster, the tracker now solves the exact cluster K-best problem through the solver interface described above. The solver contract itself assumes:

- one selected leaf per track,
- no overlapping full-history conflict keys,
- score = sum of selected leaf scores + cluster constant,
- retain up to `max_results` best feasible combinations.

### Post-solve supported-leaf pruning

After each cluster rebuild, each non-overload-split cluster tree keeps only leaves that appear in at least one retained rebuilt global for that cluster. This remains the main pruning mechanism that keeps active leaf frontiers globally informed.

### MAP-only N-scan pruning

N-scan pruning remains MAP-only:

- boundary is `b = scan_index - ns_scan_window`,
- the child of the current root on the MAP path is promoted to be the new root,
- siblings are removed structurally,
- disagreement statistics are computed against alternative rebuilt globals,
- and the newly committed pre-promotion root state is appended to the tree’s committed prefix history.

The default N-scan window is currently `6`.

### Whole-track miss lifecycle

Per-branch miss-based pruning during local expansion has been removed.

Miss handling now happens as **whole-track termination after N-scan pruning**, using:

- configurable `track_miss_termination_mode`
  - `all_active_leaves`
  - `map_leaf` (default)
  - `global_k_leaves`
- threshold `max(max_missed, ns_scan_window + 1)`

This is cleaner than the earlier branch-local miss handling, but it is also part of why low-quality trees can persist longer than before.

---

## Current approximation / safety-net mechanisms

The tracker currently contains several explicit approximation/safety-net mechanisms that are part of its operational semantics.

### 1. Overload cluster splitting

When a cluster’s projected Cartesian combinations exceed `overload_split_projected_combination_threshold`, the tracker can approximately decompose that cluster by:

- building the exact conflict graph first,
- iteratively severing the weakest conflict edge,
- recomputing connected components,
- and solving resulting subclusters independently.

Weakest-edge criterion is the pure count of shared **full-history** detection keys, with deterministic tie-break.

### 2. Historical-conflict relaxation

If a cluster is still exact-infeasible, the tracker may apply a **narrow historical-conflict relaxation**:

- only keys forced in every active leaf of a track,
- only keys also present in that track’s root history,
- only keys at or older than the current N-scan boundary,
- and only when shared by more than one track in the cluster.

Feasibility is then retried while ignoring overlaps on those specific historical keys only. All other exclusivity remains strict.

### 3. Internal birth load guards

Internal births remain intentionally simple and secondary. They are still based on the constructor-supplied initiator and Step-2 residual detections, but births can be skipped once active tree or leaf counts exceed configured thresholds.

These mechanisms are pragmatic, not final. They should be understood as explicit robustness/tractability measures for current replay use, not a final principled solution to large-cluster or extended-target behavior.

---

## Scoring state

Scoring now uses an explicit NLL-to-LLR additive model in `tomht_scoring.py` unless an explicit alternative scoring model is supplied.

Current default behavior:

- local hit score: `log(P_D) - log(lambda) - NLL`
- local miss score: `log(1 - P_D)`
- current-scan clutter correction remains a simple linear term, pre-baked into cluster leaf scores before exact solve,
- birth scoring remains a fixed penalty,
- scoring diagnostics are logged at tracker construction time.

### Unit / scale contract

The unit contract is now explicit:

- hypothesiser detection distance must be `NLL = -log p(z|x)` in a particular measurement space,
- `clutter_density` (`lambda`) must be expressed in the **same measurement-space units**,
- with that contract, linear coordinate rescaling cancels between `-log(lambda)` and the Gaussian normalization term inside `NLL`.

Miss-hypothesis distance from the hypothesiser is intentionally ignored by `NLLScoringModel`; the miss score is computed explicitly.

This scoring split should still be understood as **pragmatic and serviceable**, not final.

---

## Output / observability

### Output tracks

Public output tracks are now reconstructed from:

- the tree’s committed prefix history,
- plus the current unresolved leaf-node lineage.

This restores full logical output history across N-scan pruning while preserving the intended meaning of the explicit unresolved tree structure.

Returned `Track` metadata remains an explicit projection from the current leaf node, including:

- stable logical `track_id`
- `node_id`
- `age`
- `hits`
- `missed_count`
- `last_det_key`
- `last_det_hit`
- `root_source`
- `birth_scan_index`

### Instrumentation

Per-scan and summary instrumentation reports:

- active trees / leaves
- cluster counts
- evaluated / feasible combinations
- rebuilt globals stored
- overload split counters
- historical relaxation counters
- N-scan commitment counts
- birth statistics
- MAP track usage
- explicit scan index in `SCAN ...` lines
- scan wall time
- memory / node counts

Instrumentation also reports **per-scan timing-phase breakdowns**:

- prep/context
- pre-expand validation
- local expansion
- expansion hypothesiser/update split and call counts
- post-expand prune/validation
- births
- cluster build + solve
- post-solve prune
- map merge
- N-scan / lifecycle
- cleanup
- residual `other_ms`

This has been important for locating the new dominant replay bottleneck after the solver change.

### Determinism

The tracker now has deterministic behavior in the investigated birth-capping path:

- initiator outputs are converted from unordered set-like candidate pools into a deterministic heuristic ranking before `max_births_per_scan` is applied,
- the ranking restores the previous support/miss/age/covariance-oriented quality heuristic before deterministic tie-break fields,
- this removes process-level nondeterminism that had been introduced by slicing directly from initiator outputs.

The broader tracker is intended to be deterministic, and this remains an important operational expectation.

---

## Replay/runtime status

The current code can complete the main recorded replay to end-of-file, which was the immediate robustness goal after the transition.

A key outcome of this phase is that the previous exact-solver bottleneck has been reduced enough that the main replay bottleneck has shifted.

### Exact-solver outcome

On the 400-CPI replay window used during the solver phase:

- earlier exhaustive baseline timing was approximately:
  - median ~33 ms
  - p95 ~715 ms
  - max ~2019 ms
- branch-and-bound timing on the same window was approximately:
  - median ~27–29 ms
  - p95 ~407–423 ms
  - max ~929–965 ms

This was sufficient to justify switching the default exact backend from exhaustive to branch-and-bound.

### Local-association optimization outcome

On the standard replay timing helpers used during the current phase, recent local-association passes reduced `expand_ms` materially, including both median and upper-tail timings. The Cholesky-based math pass in particular produced a visible additional reduction after the earlier reuse / rectangular-gating pass.

### Current dominant bottleneck

With branch-and-bound enabled and the recent hypothesiser/scoring cleanup in place, the slow scans are still dominated primarily by **local expansion / hypothesis generation**, not cluster solving.

So the current runtime picture is:

- exact cluster solving is no longer the main immediate blocker on the primary replay,
- local-association inner-kernel cleanup has already yielded worthwhile wins,
- the next likely leverage point is **reducing how many leaves require full expansion**,
- while large merged clusters and approximation semantics still matter for tail behavior.

---

## Current output-quality note

The current baseline is operationally usable, but output quality is not yet where it should ultimately be.

Known current notes:

- smoke/scenario behavior remains somewhat noisy, with false track starts still higher than desired,
- recorded-data replay remains broadly reasonable after the latest local association and scoring changes,
- but some segments appear to show somewhat more target swapping / track jumping than before.

These are treated as **known baseline-quality concerns**, not as immediate blockers for the current runtime-focused next step. They likely deserve a dedicated follow-up quality pass rather than being mixed into the next expansion-volume phase.

---

## Internal births: current interpretation

Internal births should currently be understood as:

- simple,
- heuristic,
- secondary relative to the external-start integration path,
- and not yet the final word on birth/existence semantics.

### What is solid in the current birth path

The current residual policy uses detections unused by the **union of all active leaves** after local expansion. This is conservative, but it preserves a strong no-conflict invariant for internal births and for residual-based external starts.

Birth capping is deterministic and quality-ranked rather than arbitrary.

### What looks weak or provisional

A few observations now look important:

- once a birth candidate survives capping, it is inserted directly as a persistent `TrackTree`,
- after insertion there is no old-style “track absent” alternative,
- false starts therefore appear easier to preserve, not just easier to create,
- whole-track miss kill is slower and more permissive than the earlier branch-local miss-pruning path,
- and the default `track_miss_termination_mode="map_leaf"` is conservative in ambiguous periods.

Taken together, the current false-start behavior is most plausibly explained by the combination of:

- direct birth insertion,
- effectively mandatory post-birth existence,
- and slower lifecycle kill.

Scoring may also contribute, but currently looks like a secondary factor rather than the first suspect.

### Operational note

For the external-start-only ISAC integration path, this is not necessarily an immediate blocker. But for the general internal-birth tracker path, birth/existence semantics remain a fairly high-priority quality topic.

A practical short-term lever is **initiator conservatism**: a stricter initiator can reduce candidate births without changing tracker semantics and can help separate “too many births created” from “births survive too long.”

### Carry-over observations from the pre-transition implementation

A few pre-transition birth-handling ideas still look relevant enough to retain as notes:

- the current residual policy is more conservative than the older top-k/global-supported notion,
- current birth insertion is more direct and less uncertain than the older birth-alternative flow,
- birth candidate observability/debugging could use richer pre/post-cap reporting,
- and birth impact statistics likely need TO-MHT-native definitions if revisited later.

These observations should be treated as design notes, not as decisions already made.

---

## What is solid now

The following now look solid enough to treat as the current base architecture:

- explicit `TrackTree` + `TrackHypothesisNode` persistent state,
- scan-to-scan persistence through trees rather than globals,
- full-history `(scan_index, det_index)` detection-key semantics,
- per-scan rebuilt cluster globals,
- post-solve supported-leaf pruning,
- MAP-only N-scan pruning directly on trees,
- committed-prefix output reconstruction across pruning,
- whole-track post-N-scan miss lifecycle,
- predictor/updater public integration boundary,
- exact-one-of `predictor` or `hypothesiser` constructor semantics,
- explicit distance-hypothesiser seam for local association,
- explicit NLL-to-LLR scoring baseline,
- deterministic birth capping,
- explicit solver seam with solver-facing exact cluster problem contract,
- branch-and-bound as the default exact backend,
- exhaustive retained as exact reference/fallback,
- OR-Tools retained as experimental exact backend,
- replay integration with end-to-end recorded replay completion,
- smoke/replay golden-regression harnesses for output/timing comparison,
- and per-scan timing-phase instrumentation.

---

## What remains provisional / future-work territory

The following are still provisional or explicitly not the final word:

### 1. Local expansion volume
This now looks like the most immediate runtime bottleneck on the primary replay used during this phase. The next likely win is not more math-kernel cleanup alone, but reducing how many leaves need full expansion.

### 2. Approximation semantics
Overload splitting and historical-conflict relaxation are useful and explicit, but still not conceptually final.

### 3. Scoring design
The current NLL-based scoring split is much clearer than the old beta/PDA path, but still pragmatic rather than fully settled.

### 4. Parallelization / orchestration
Parallel local expansion is a plausible future axis, but should likely be opt-in and architecturally separated from the current tracker-owned sequential default. This is a later concern.

### 5. Internal birth / existence semantics
Internal births remain simple, heuristic, and secondary. Their current semantics appear more permissive with respect to false-start persistence than the pre-transition behavior.

### 6. Tracking quality / false-start tuning
Replay output is now usable enough that false starts and target-swapping issues can be revisited meaningfully, but they are not the single main blocker at the current checkpoint.

### 7. Internal organization around cluster build/rebuild and expansion paths
The exact cluster-solver seam is now cleaner, and the local-association seam is much clearer than before, but some tracker internals still carry a lot of closely related logic. Further organization work is likely warranted, but should preferably happen as a sub-goal of deeper work rather than as a standalone cleanup-only phase.

---

## Recommended interpretation of the current checkpoint

This checkpoint should be understood as:

- a **real track-oriented TO-MHT implementation**
- with a coherent persistent-tree / rebuilt-global architecture
- that survives the target recorded replay end-to-end
- and is suitable for continued replay-based evaluation, integration work, and next-phase planning

It should also be understood as:

- now on a better exact-solver footing than the earlier exhaustive baseline,
- with branch-and-bound validated enough to become the default exact backend,
- with OR-Tools retained as an experimental comparison path,
- with a much clearer local-association and scoring baseline than before,
- and with the main replay bottleneck now centered on **local expansion volume** rather than exact cluster solving.

This is a good place to refresh the docs, consolidate what was learned in this phase, and then deliberately choose the next deeper branch of work.
