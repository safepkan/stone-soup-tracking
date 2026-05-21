# TO-MHT Current State

## Snapshot date

This document describes the tracker as it exists after the track-oriented TO-MHT transition and the subsequent replay-hardening, solver-seam extraction, branch-and-bound default switch, local-association ownership work, NLL/DPM scoring cleanup, start/lifecycle/publication redesign, module-extraction cleanup, live conflict-key cleanup, overload-split mode work, overload-solver module split, small expansion/frontier API cleanup, and smoke-runner parameter reset completed through **2026-05-20**.

It is a **current-state snapshot**, not a roadmap and not a full design history.

The long dated update stack from the previous version has been consolidated into the main text below. The recent scoring/start/lifecycle/API work gave the tracker coherent score semantics and a usable integration boundary. The subsequent expansion/frontier pass removed the main known frontier-control blocker by making overload splitting sound, restoring uniform supported-leaf pruning, and cleaning up expansion-related defaults.

---

## Bottom line

The tracker is now a practical track-oriented TO-MHT implementation in the sense used in this codebase:

- persistent scan-to-scan state is explicit `TrackTree` objects and their active leaves,
- globals are rebuilt per cluster on every scan from current leaves,
- the previous scan's explicit global list is not the persistent search frontier,
- MAP-only N-scan pruning operates directly on explicit trees,
- output tracks are reconstructed from committed prefix history plus current unresolved lineage,
- output publication is a sticky tree-level boundary separate from internal confirmation,
- and scoring is now an additive NLL/LLR model with explicit detection-probability and clutter-density inputs.

The code treats track trees as the primary persistent state. Rebuilt globals, cluster snapshots, internal scan context, and scan stats are retained as last-scan inspection/debug artifacts rather than as long-lived frontier state. Passive model/data containers, including `ScanContext`, live in `mht/tomht_model.py`.

Scan stats now include expansion/frontier usefulness counters: active tree/leaf
counts at the main scan-pipeline boundaries, expanded leaves and local child
volume, retained top-K supported leaves, MAP-selected leaves, supported-leaf
pruning removals, and lifecycle-state expansion split. These counters are
collected by default in `ScanStats`; compact per-scan and summary lines are
opt-in via
`TOMHTParams.debug_display_expansion_frontier` or
`TOMHT_DEBUG_EXPANSION_FRONTIER=1`.

The implementation is currently:

- structurally aligned with the intended track-oriented design,
- usable for replay-based experimentation and integration work,
- on a stronger exact-solver footing than the earlier exhaustive baseline,
- on a clearer scoring/lifecycle/publication baseline than the earlier PDA/beta-oriented path,
- and substantially less monolithic after the module extraction work.

The main remaining profiling hotspot is no longer exact cluster solving. With branch-and-bound as the default backend, overload splitting on a sound footing, and the local-association math path cleaned up, standard replay time is still dominated by **local expansion / hypothesis generation**. Current replay/smoke workloads are no longer blocked by frontier growth, but broader scenario runs may still expose expansion-volume or object-reconstruction overheads.

---

## Public API and integration boundary

The intended operational public surface is:

- `update_tracker(time, detections, *, caller_scan_context=None) -> (time, tracks)`,
- `tracks`,
- `add_external_starts(time, starts)`,
- `get_unused_detections()`.

The tracker also exposes inspection helpers:

- `get_map_hypothesis_snapshot()`,
- `get_map_output_tracks(include_unpublished=False)`,
- `get_n_scan_commitment_snapshot()`,
- `get_last_cluster_snapshots()`,
- `get_track_tree_snapshot()`,
- `print_summary_stats()`.

The external interface remains Stone Soup-oriented:

- input detections are Stone Soup `Detection`,
- output tracks are Stone Soup `Track`,
- `updater` is required,
- constructor accepts exactly one of `predictor` or `hypothesiser`,
- local expansion consumes distance hypotheses directly,
- output `Track` metadata is a TOMHT-owned projection from the selected leaf/tree context.

A dedicated API/integration guide now exists in `TO_MHT_API.md`. It is the best reference for integration assumptions such as one sensor / one measurement space per update, DPM semantics, start priors, public vs internal IDs, publication behavior, and stable metadata.

### Constructor and local-association boundary

The tracker has a narrow distance-hypothesiser seam:

- input: `(track, detections, timestamp)`,
- output: Stone Soup `MultipleHypothesis`,
- hypotheses are expected to be `SingleDistanceHypothesis` entries,
- each expansion result must contain exactly one missed-detection hypothesis,
- detection hypotheses must reference original detections from the current scan,
- detection-hypothesis distance must be `NLL = -log p(z|x)` in measurement space, without detection-probability or clutter-density factors.

Default local association is tracker-owned `TrackerOwnedNLLDistanceHypothesiser`. It uses non-squared Mahalanobis-threshold gating via `mahalanobis_gate_threshold` and emits NLL distances. Custom hypothesisers are allowed but must satisfy the stricter contract above. The reconstructed `Track` passed to a custom hypothesiser carries TOMHT metadata including internal/public IDs, lifecycle state, publication state, age, hits, and miss count; confirmation-state-dependent gating is currently best handled there, not by adding policy to the default hypothesiser.

The split is explicit:

- the hypothesiser owns local distances and gating,
- the `NLLScoringModel` owns NLL-to-LLR conversion using the DPM.

---

## Core architecture

### Tracker as orchestrator

`TOMHTTracker` is now primarily an orchestrator. Major substeps have been extracted into dedicated modules with explicit dependencies, generally passing `TrackTreeStore`, `TOMHTParams`, scan context, and specific per-phase inputs rather than reaching through broad tracker state.

Important extracted modules include:

- `tomht_params.py` for `TOMHTParams` and params overrides,
- `tomht_tree_store.py` for persistent tree/node/ID bookkeeping,
- `tomht_expansion.py` for local expansion orchestration,
- `tomht_births.py` for internal-birth residual/candidate handling,
- `tomht_external_starts.py` for external-start validation/insertion,
- `tomht_clustering.py` for live-conflict cluster construction,
- `tomht_cluster_overload.py` for overload-aware cluster solve entry and `OVERLOAD_SPLIT` logging,
- `tomht_cluster_overload_common.py` for shared overload-solve data structures, exact-solver bridging, filtering, recombination, and feasibility helpers,
- `tomht_cluster_split_policy.py` for overload split triggering and weak-link binary split selection,
- `tomht_cluster_overload_greedy.py` for the default `greedy_partition` overload strategy,
- `tomht_cluster_overload_conditional.py` for the reference `conditional_exact` overload strategy,
- `tomht_cluster_rebuild.py` for cluster option materialization, snapshot assembly, and MAP merge,
- `tomht_pruning.py` for post-solve supported-leaf pruning and MAP-only N-scan pruning,
- `tomht_lifecycle.py` for confirmation, score deletion, configured deleter deletion, and live-MAP filtering,
- `tomht_output.py` for publication policy, public-ID assignment, and output reconstruction,
- `tomht_stats.py` for scan stats, timing breakdowns, and reporting,
- `tomht_utils.py` for TOMHT-specific deterministic utility helpers,
- `utils.py` for generic runtime/environment/timer helpers.

This organization is not final, but the broad shape is now healthy: the tracker class describes the per-scan pipeline while the phase-specific implementation lives close to its data and semantics.

### Persistent state

Persistent scan-to-scan state consists primarily of:

- explicit `TrackTree` objects keyed by internal logical `track_id`,
- persistent `TrackHypothesisNode` objects linked by same-track parent/child structure,
- each tree's current unresolved root and active leaf set,
- each tree's committed prefix states and committed detection keys from fixed pre-frontier branch decisions,
- tree-level lifecycle/publication state,
- public ID assignment state,
- N-scan commitment bookkeeping,
- and stats/debug state.

`TrackTreeStore` owns:

- internal track ID allocation,
- node ID allocation,
- node table,
- track-tree table,
- root and child node creation,
- single-root tree insertion,
- active leaf/tree counts,
- empty-tree removal,
- unreachable-node cleanup.

`TOMHTTracker` keeps compatibility views for `_nodes_by_id` and `track_trees_by_track_id`, but implementation code now uses `self._tree_store` directly.

### `TrackHypothesisNode`

`TrackHypothesisNode` is the canonical per-step hypothesis unit. It carries:

- internal `track_id`,
- stable `node_id`,
- same-track `parent`,
- `scan_index` / `timestamp`,
- Stone Soup state payload,
- used `DetectionKey(scan_index, det_index)` / association label,
- local and accumulated score,
- cached age / hit / miss counters,
- provenance such as `root_source` and `birth_scan_index`.

Nodes are mutable in this phase so child-link bookkeeping can be maintained directly.

### `TrackTree`

`TrackTree` is explicit and persistent, with at least:

- internal `track_id`,
- unresolved `root_node_id`,
- active leaf node IDs,
- `root_source`,
- committed prefix states,
- committed prefix detection keys,
- sticky `lifecycle_state` (`tentative` / `confirmed`),
- sticky `publication_state` (`unpublished` / `published`),
- optional `public_track_id` assigned at first publication.

Internal track IDs may have gaps. Public output IDs are assigned only at publication.

---

## Current per-scan pipeline

The current runtime pipeline is:

1. sort detections deterministically,
2. expand active leaves in every persistent tree,
3. remove empty trees,
4. optionally create internal birth trees from residual detections,
5. build live unresolved conflict clusters,
6. rebuild feasible globals per original cluster through the exact cluster-solver contract, using an internal overload split mode when needed,
7. post-solve prune each cluster tree frontier to leaves supported by retained rebuilt globals,
8. merge cluster MAP selections into a full-scan MAP global,
9. apply MAP-only N-scan pruning on explicit trees,
10. apply whole-track lifecycle: sticky confirmation, then score deletion plus the configured deleter,
11. update sticky output-publication state for MAP-selected live trees,
12. reclaim unreachable node storage,
13. build/store scan stats and return published MAP output tracks.

Timing instrumentation still groups N-scan, lifecycle, and publication into the broader `nscan_lifecycle_ms` phase. That is acceptable for now because timing is not currently the bottleneck in those steps, and avoiding log-format churn has been useful.

---

## Scoring, DPM, and score semantics

### Local scoring baseline

Local scoring is now explicit NLL-to-LLR scoring:

```text
hit  = log(P_D) - log(lambda) - NLL
miss = log(1 - P_D)
```

where:

- `NLL = -log p(z|x)` comes from the distance hypothesiser,
- `P_D` is detection probability,
- `lambda` is clutter density in the same measurement-space units as the NLL,
- unused detections carry no separate score term because they cancel against the all-clutter baseline.

Legacy PDA/beta scoring, unused-detection scoring, `birth_log_penalty`, and `ScoringModel.score_birth(...)` have been removed. Cluster solving now ranks combinations directly by accumulated leaf scores.

### DetectionProbabilityModel

`NLLScoringModel` now consumes a narrow `DetectionProbabilityModel` rather than scalar `P_D`/`lambda` directly. The default `ConstantDetectionProbabilityModel` wraps `TOMHTParams.prob_detect` and `TOMHTParams.clutter_density`, preserving the simple scalar path.

Custom DPMs can vary detection probability and clutter density by:

- public track ID when available,
- Stone Soup prediction,
- concrete detection for hit clutter-density calls,
- opaque `caller_scan_context` supplied to `update_tracker(...)`.

Important semantics:

- one `update_tracker(...)` call should contain detections from one sensor / one measurement space,
- `caller_scan_context` is distinct from TOMHT's internal `ScanContext`,
- empty scans can still carry caller context,
- DPM callbacks receive public track IDs only after publication,
- unpublished trees pass `track_id=None`,
- DPM is not currently used for initiator-root or external-start root scoring.

### User-facing probabilities vs internal log-odds

Human-facing params remain probabilities, e.g.:

- `external_start_initial_existence_probability`,
- `initiator_start_initial_existence_probability`,
- `track_confirmation_existence_probability`,
- `track_deletion_existence_probability`,
- `publish_min_existence_probability`.

They are converted internally to log-odds where needed. Metadata inputs for starts may provide either:

- `metadata["existence_log_odds"]`, or
- `metadata["existence_probability"]`.

Precedence is:

```text
valid existence_log_odds > valid existence_probability > configured default
```

Output metadata exposes both `existence_log_odds` and `existence_probability` for inspection/calibration.

---

## Starts and birth semantics

### External starts

External starts are the clean integration path for systems that already own initiation/confirmation.

`add_external_starts(time, starts)` inserts starts after an `update_tracker(...)` call at the same timestamp. The preceding update establishes scan timestamp/bookkeeping; it can be an empty update when appropriate.

External-start roots use
`TOMHTParams.external_start_initial_existence_probability` by default, with
optional per-track metadata override via `existence_log_odds` or
`existence_probability`. After inserting starts and updating the MAP view,
`add_external_starts(...)` runs the same score-based confirmation pass used by
the normal scan lifecycle before applying output publication. With defaults, a
`0.95` external-start prior crosses the `0.9` confirmation threshold and is
published immediately; lower per-track existence metadata can leave the tree
tentative/unpublished but still inspectable with
`get_map_output_tracks(include_unpublished=True)`.

`add_external_starts(...)` does not run full lifecycle deletion, N-scan
pruning, cluster rebuild, or scan stats updates.

### Internal initiator starts

Internal starts are controlled by constructor initiator presence:

- `initiator=None`: no internal starts; residual detections remain available via `get_unused_detections()` for external workflows,
- `initiator=<Initiator>`: residual detections are passed to the configured Stone Soup initiator and returned tracks are treated as candidate starts.

The tracker does not own generic single-detection state initialization. Even one-detection starts require domain-specific state/covariance assumptions, so they belong in a caller-supplied initiator.

Internal-start roots use `TOMHTParams.initiator_start_initial_existence_probability` by default, again with optional metadata override using log-odds or probability.

Internal-birth candidate validity is now initiator-owned. The tracker no longer applies state-layout-specific position/covariance sanity checks. Remaining internal-start controls are capping, deterministic ordering, and optional load guardrails. `max_births_per_scan` defaults to `10`; it is still a guardrail. `birth_skip_if_active_trees_above` and `birth_skip_if_active_leaves_above` default to disabled (`None`) and are available for scenario-specific emergency use. Deterministic ranking is a safety valve; if `max_births_per_scan` routinely fires, the better fix is usually initiator-side filtering or candidate confidence metadata rather than relying on TOMHT's fallback ordering.

### Residual detections

`get_unused_detections()` is mainly intended for `initiator=None` external-start workflows. If an internal initiator is configured, residual detections passed to that initiator should generally be considered consumed by the internal-start path, even if no retained starts result after capping/guardrails.

---

## Lifecycle, publication, and identity

### Confirmation

`TrackTree.lifecycle_state` starts as `tentative` and promotes stickily to `confirmed` when:

```text
max(active_leaf.accumulated_log_score) >= logit(track_confirmation_existence_probability)
```

Default confirmation probability is `0.9`.

Confirmation is an internal tree state. It does not by itself delete tracks.

### Whole-track deletion

Score-based deletion is enabled by default and runs after N-scan pruning:

```text
delete tree if max(active_leaf.accumulated_log_score) <= logit(track_deletion_existence_probability)
```

Default deletion probability is `0.01`.

Score deletion always runs. In addition, TOMHT runs one configured deleter:

- without a custom deleter: an internal miss-count deleter resolved from params,
- with a custom Stone Soup deleter: the custom deleter replaces that default.

The default miss-count threshold uses an N-scan-aware floor:

```text
effective_miss_threshold = max(max_missed, ns_scan_window + 1)
```

The default miss-count deleter is intentionally minimal: it reads reconstructed
track `metadata["missed_count"]`, applies the effective threshold, and has no
sensor/context awareness. Custom Stone Soup deleters remain the path for
field-of-view exit, lifetime limits, sensor/context-aware invalidity, or
application-specific deletion.

`TRACK_LIFECYCLE` diagnostics report deletion reason groups (`score`, `miss`, `deleter`).

### Publication

Publication is separate from confirmation and deletion. `TrackTree.publication_state` starts `unpublished` and promotes stickily to `published` when a MAP-selected live tree satisfies publication policy.

Default publication is confirmed-only:

```text
publish_lifecycle_states = ("confirmed",)
```

Additional gates include `publish_min_hits`, `publish_min_age`, and `publish_min_existence_probability`.

Standard `update_tracker(...)`, `tracks`, and `get_map_output_tracks()` return published MAP tracks only. `get_map_output_tracks(include_unpublished=True)` reconstructs live MAP-selected unpublished tracks for inspection.

### Public and internal IDs

`TrackTree.track_id` is the internal logical ID assigned at tree creation. `TrackTree.public_track_id` is assigned once, at first publication.

Default public IDs are dense integers in first-publication order. A custom `output_track_id_mapper` can map internal logical IDs into caller-specific public IDs, but custom mappers are responsible for uniqueness and non-reuse.

Stable output metadata should use:

- `internal_track_id`,
- `public_track_id`,
- `existence_log_odds`,
- `existence_probability`,
- `lifecycle_state`,
- `publication_state`,
- `age`, `hits`, `missed_count`.

Legacy `metadata["track_id"]` is a deprecated compatibility alias for the internal ID. The old `get_tomht_track_id(track)` helper has been removed.

---

## Solver architecture and exact backend status

A dedicated solver seam exists:

- `tomht_cluster_solver.py`: solver-facing problem/result/diagnostics contract,
- `tomht_cluster_solver_exhaustive.py`: exhaustive reference backend,
- `tomht_cluster_solver_branch_and_bound.py`: current default exact backend,
- `tomht_cluster_solver_ortools.py`: experimental exact CP-SAT backend,
- `tomht_cluster_solver_factory.py`: backend construction.

The exact cluster problem contract carries:

- one leaf option per track choice,
- live unresolved conflict keys for feasibility,
- pre-scored accumulated leaf scores,
- retain up to K best feasible combinations.

Tracker-side overload handling remains outside the concrete solver backend:

- overload splitting is internal to one original-cluster solve,
- both overload modes return feasible globals for the original live cluster,
- no split subcluster pseudo-globals are exposed downstream.

### Branch-and-bound default

`TOMHTParams.cluster_solver_backend` defaults to `"branch_and_bound"` with aliases `"branch-and-bound"` and `"bnb"`.

The backend:

- performs deterministic depth-first exact search over ordered tracks,
- uses a 1-track fast path,
- enforces supplied live conflict-key exclusivity exactly,
- orders tracks/leaves deterministically,
- uses suffix-score optimistic bounds for pruning,
- retains exact K-best solutions through the shared deterministic heap.

It replaced exhaustive as the default after parity and replay/timing work.

### Exhaustive backend

The exhaustive backend remains available as:

- reference implementation,
- parity oracle,
- fallback backend.

### OR-Tools backend

The OR-Tools CP-SAT backend remains experimental. It is exact under the current solver contract but was not a runtime win on the primary replay workload in its current repeated-solve K-best form. It is useful for comparison and future experiments, not recommended as the default runtime path.

---

## Rebuild, pruning, and approximation behavior

### Local expansion

Local expansion is distance-hypothesis driven:

- reconstruct a compatibility `Track` for each active leaf unless the
  tracker-owned default hypothesiser state fast path is enabled,
- call the configured hypothesiser,
- validate the returned hypotheses,
- score hit/miss alternatives through `NLLScoringModel`,
- create child nodes,
- always preserve a miss alternative,
- apply optional per-tree local leaf cap.

The local child cap is `max_children_per_leaf`, a per-active-leaf branching
cap rather than a whole-track-tree cap.

`max_leaves_per_track_tree` remains a pre-solve safety valve rather than the main pruning semantics.

### Local-association math kernel

The tracker-owned default hypothesiser includes conservative optimizations:

- rectangular pre-gating,
- covariance-preparation reuse per call,
- scan-time prediction reuse,
- measurement-prediction reuse by object identity,
- a default-enabled leaf-state entry point gated by
  `TOMHTParams.enable_default_hypothesiser_state_fast_path`,
- Cholesky-based Mahalanobis/NLL evaluation.

These changes improved expansion timing, but the next likely leverage is expansion volume rather than further inner-kernel cleanup alone.

### Clustering

Clusters are built from live unresolved active-leaf `DetectionKey` overlap,
not only current-scan overlap. `DetectionKey` is a tuple-compatible named
tuple with `scan_index` and `det_index` fields, so it remains usable as an
immutable set/dict key while call sites that inspect the components avoid
positional indexing.

Each node still caches its full lineage in `detection_history_keys`, while each
`TrackTree` stores detection keys from branch decisions fixed by N-scan
promotion in `committed_detection_keys`. Active conflict keys are computed as:

```text
leaf.detection_history_keys - tree.committed_detection_keys
```

This keeps clustering and solver feasibility aligned while avoiding constraints from committed pre-root history that the current unresolved frontier can no longer change.
Historical-conflict relaxation has been removed from this path.

### Global rebuild

For each cluster, TOMHT builds leaf options and solves the cluster K-best
problem through the exact solver interface. Scores are accumulated leaf scores;
there is no current-scan unused-detection affine offset.

If a cluster exceeds `overload_split_projected_combination_threshold`, rebuild
keeps the original cluster as the public unit of work and applies one of the
configured overload split solution modes internally.

Implementation layout: the public entry/logging surface is
`tomht_cluster_overload.py`; split trigger and split-link selection live in
`tomht_cluster_split_policy.py`; shared exact-solver bridge, filtering,
recombination, cache/accumulator structures, and feasibility helpers live in
`tomht_cluster_overload_common.py`; the default greedy strategy lives in
`tomht_cluster_overload_greedy.py`; and the conditional-exact reference strategy
lives in `tomht_cluster_overload_conditional.py`.

`TOMHTParams.overload_split_solution_mode="greedy_partition"` is the default
operational overload fallback. It is sound but approximate:

- choose a deterministic weak-link binary split,
- assign contested cut keys by best local claiming-leaf score,
- solve the side with more assigned cut keys first,
- release assigned keys that no retained first-side global actually uses,
- solve the second side forbidding only first-side assigned keys that remain claimed,
- recombine left/right solutions by summed score,
- reject any recombined global that violates the original live conflict keys,
- retain deterministic top-K feasible globals for the original cluster,
- fall back to `conditional_exact` for that branch if the greedy partition cannot produce feasible parent globals.

Greedy mode may return different feasible top-K globals because it does not
preserve strict K-best optimality across all cut assignments. It is the default
because the standard replay quality looked acceptable while the scan-174
conditional-exact recombination hotspot disappeared.

`"conditional_exact"` remains available as a reference / higher-compute mode:

- choose a deterministic weak-link binary split,
- enumerate cut-key forbiddance assignments when the cut interface is small,
- recursively solve left/right subclusters with inherited forbidden keys,
- recombine left/right solutions by summed score,
- reject any recombined global that violates the original live conflict keys,
- retain deterministic top-K feasible globals for the original cluster.

Recursive subproblem results are memoized within one original-cluster solve by
`(track_ids, inherited_forbidden_keys)`, preserving the same conditional solve
semantics while avoiding repeated exact solves and recombinations for identical
branches. Large cut interfaces use a conservative fallback assignment set and
are reported in `OVERLOAD_SPLIT ...
interface_assignment_cap_fallbacks=...` diagnostics. The same log line also
reports recursive cache hit/miss counts, max recursion depth, max cut-key count,
total interface assignments, max recombination product size,
`branch_recomb_retained`, and `final_recomb_retained`. In greedy mode it also
reports compact `greedy_*` assignment, release, split, and fallback counters.
Split subclusters are not exposed as ordinary `ClusterRebuildSnapshot` objects.

### Post-solve supported-leaf pruning

For each cluster, active leaves are pruned to those appearing in at least one
retained rebuilt top-K global for that original cluster. This now applies
uniformly to clusters solved through overload recursion because every retained
rebuilt global is checked against the original live conflict constraints before
it reaches downstream MAP merge, N-scan pruning, lifecycle/output, or
supported-leaf pruning.

### MAP-only N-scan pruning

N-scan pruning is MAP-only:

- boundary `b = scan_index - ns_scan_window`,
- promote the child of the current root on the MAP path,
- remove siblings structurally,
- append committed states to the tree's committed prefix,
- add the promoted child's full detection history to the tree's committed detection keys,
- update commitment snapshot and disagreement stats.

Default `ns_scan_window` remains `6`.

### Approximation / safety-net mechanisms

The normal frontier-control stack is now coherent:

1. local association gating in the configured hypothesiser,
2. per-active-leaf local branching through `max_children_per_leaf`, with the miss alternative always preserved,
3. K-best feasible cluster solve over current leaves,
4. post-solve supported-leaf pruning to retained rebuilt globals,
5. MAP-only N-scan pruning,
6. score deletion plus the configured deleter,
7. sticky publication gating.

The explicit approximation/safety mechanisms are:

1. overload cluster splitting,
2. optional pre-solve per-tree leaf cap through `max_leaves_per_track_tree`,
3. internal-birth cap through `max_births_per_scan`,
4. optional internal-birth load guards,
5. optional hard projected-cluster-combination cap.

These are pragmatic robustness/tractability mechanisms and remain conceptually provisional. Birth load guards default to disabled. The overload split projected-combination threshold remains conservative and is still a future review item; it is a useful current smoke/replay exercise path, not a final difficulty predictor.

---

## Output and observability

### Output tracks

Published output tracks are reconstructed from:

- committed prefix history,
- current unresolved MAP leaf lineage,
- tree context for lifecycle/publication/public ID metadata.

This preserves logical output history across N-scan pruning while keeping unresolved tree structure explicit.

### Instrumentation

Per-scan and summary instrumentation reports include:

- active trees/leaves,
- tentative/confirmed counts,
- MAP-published / MAP-unpublished counts,
- clusters and rebuild stats,
- evaluated/feasible combinations,
- solver diagnostics,
- overload split counters,
- N-scan commitment counts,
- birth statistics,
- lifecycle deletion reasons,
- scan wall time,
- memory/node counts,
- phase timing breakdown.

Expansion timing is split into hypothesiser, updater, and local-expansion
`Track` reconstruction call counts/times, plus default-hypothesiser state
fast-path call counts. This avoids treating `expand_other_ms` as the only proxy
for reconstruction overhead. Public output and debug/inspection still
legitimately need full reconstructed tracks, but internal paths such as default
local expansion and default miss-count deletion may eventually benefit from
lighter leaf-local views.

### Regression status

Smoke and replay regression harnesses are now in place for both output and timing comparisons. Baselines have been updated for the scoring/lifecycle/publication changes, overload-split default, expansion/frontier parameter cleanup, and smoke-runner reset to nominal scenario scoring parameters.
Focused MHT unit tests are available through `make mht_tests`, which runs `pytest mht/tests`.

---

## What is solid now

The following are solid enough to treat as the current base architecture:

- explicit `TrackTree` + `TrackHypothesisNode` persistence,
- scan-to-scan persistence through trees rather than globals,
- live unresolved detection-key exclusivity semantics with full-lineage node caches,
- per-scan rebuilt cluster globals,
- exact cluster-solver seam,
- branch-and-bound as default exact backend,
- exhaustive retained as reference/fallback,
- OR-Tools retained as experimental,
- distance-hypothesiser seam,
- NLL/DPM scoring baseline,
- external/internal start lanes with existence priors,
- metadata log-odds/probability input support for starts,
- sticky confirmation,
- score-based deletion,
- sticky output publication,
- public vs internal ID split,
- committed-prefix output reconstruction,
- module-extracted tracker orchestration,
- API guide for integration assumptions,
- replay/smoke regression harnesses.

---

## What remains provisional / future-work territory

### 1. Local expansion and object-boundary cost

Local expansion remains the main profiling hotspot, but it is no longer an active blocker on the standard replay/smoke workloads. Future work should distinguish true expansion volume from per-leaf overhead. Profiling shows hypothesiser calls dominate, but also shows non-negligible overhead from reconstructing full Stone Soup `Track` objects and accessing Stone Soup attributes. A future pass should profile reconstruction call sites separately and consider default-hypothesiser fast paths or lightweight internal track views while preserving full Stone Soup tracks for public output/debug and default compatibility for custom components.

### 2. Frontier and score-based pruning

Now that scores are coherent, broader score/frontier pruning can be reconsidered. It should be approached carefully and preferably first as diagnostics: score-relative leaf gaps, tentative-vs-confirmed frontier volume, and whether low-score leaves ever appear in retained top-K/MAP hypotheses.

### 3. Internal birth ranking and capping

Internal birth candidate selection is now state-layout agnostic, but its ranking/capping remains heuristic. `max_births_per_scan=10` is the default cap and still a guardrail. The active-tree/active-leaf birth load guards default to disabled and should be used only for scenario-specific emergency control. Covariance trace is only a weak fallback quality proxy. If caps fire often, initiator-side filtering or candidate confidence metadata is likely more important than improving the fallback metric. This is not urgent for the external-start-only ISAC path, but it matters for general recorded-data runs.

### 4. Overload difficulty and approximation policy

Overload splitting is now sound and practical, with `greedy_partition` as the operational default and `conditional_exact` retained as a reference mode. The remaining open question is not soundness but policy: the projected Cartesian product threshold is a crude difficulty signal from the exhaustive-solver era. A future pass should compare exact branch-and-bound difficulty, conflict graph structure, score concentration, and split quality before changing the trigger.

### 5. Tracking quality / false-start / ID-switch review

The tracker is now stable enough to support a more meaningful output-quality review. Occasional ID switching / MAP-leaf switching remains a separate output-continuity topic. Future work may include output stitching, ID smoothing across MAP flips, or publication-side continuity policies, but this should not be mixed into the current frontier-control baseline.

### 6. Confirmation-state-dependent gating

The reconstructed `Track` passed to custom hypothesisers now carries TOMHT lifecycle metadata, so custom hypothesisers can implement tighter tentative-track gates or other confirmation-state-dependent policies. The tracker-owned default hypothesiser still uses one threshold; adding separate tentative/confirmed defaults should wait for scenario evidence.

### 7. Parallelization / orchestration

Parallel local expansion remains a plausible future axis, but should be opt-in and architecturally explicit. It should not be mixed into smaller profiling/object-boundary cleanup by default.

### 8. API feedback

`TO_MHT_API.md` is good enough to share for review. Integration feedback may drive small API or documentation adjustments.

---

## Recommended interpretation of the checkpoint

This checkpoint should be understood as:

- a real track-oriented TO-MHT implementation,
- with coherent scoring/start/lifecycle/publication semantics,
- with a clean enough API story for integration discussion,
- with a thin tracker orchestrator and extracted phase modules,
- with overload/frontier control no longer blocking the standard replay/smoke workloads,
- and with remaining performance attention focused on local expansion, Stone Soup object-boundary overhead, and broader-scenario validation.

The scoring/birth/confirmation/publication phase can be treated as complete, and the current expansion/frontier phase can be treated as mostly wrapped for the known scenarios. The next technical branch should be chosen from integration feedback, broader recorded-data runs, and profiling evidence rather than assuming another immediate pruning mechanism is needed.
