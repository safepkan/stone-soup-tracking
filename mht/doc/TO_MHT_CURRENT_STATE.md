# TO-MHT Current State

## Snapshot date

This document describes the tracker as it exists after the track-oriented TO-MHT transition and the subsequent replay-hardening, interface-cleanup, determinism, output-history, and solver-seam passes completed through **2026-04-10**.

It is a **current-state snapshot**, not a roadmap and not a full design history.

---

## Bottom line

The tracker is now a **real track-oriented TO-MHT implementation** in the practical sense used in this codebase:

- persistent scan-to-scan state is explicit `TrackTree` objects and their active leaves,
- globals are rebuilt per cluster on every scan from current leaves,
- the previous scan’s explicit global list is **not** the persistent search frontier,
- MAP-only N-scan pruning operates directly on explicit trees,
- and public output tracks are reconstructed from **committed prefix history plus current unresolved lineage**.

The code now treats track trees as the primary persistent state, with rebuilt globals retained only as last-scan inspection/debug artifacts.

The current implementation is therefore:

- structurally aligned with the intended track-oriented TO-MHT direction,
- usable for replay-based experimentation and continued integration work,
- reasonably robust on the main recorded replay through end-of-file,
- but still **performance-limited** in large merged-cluster situations and still reliant on a few explicit approximation/safety-net mechanisms.

## Update (2026-04-10): explicit cluster-solver contract seam

The exact cluster solve path now has a dedicated solver-facing module:

- `mht/tomht_cluster_solver.py` defines the solver-facing exact problem/result contract and the current exhaustive backend.
- The tracker prepares solver-facing cluster problems from node/tree state and maps solved leaf IDs back to node-native rebuilt globals.
- The exact problem contract now explicitly carries:
  - one leaf option per track choice,
  - full-history conflict keys for feasibility,
  - and pre-scored leaf accumulated scores that already include the current-scan
    per-hit clutter correction used by today’s tracker scoring.
- Approximation/policy placement is explicit:
  - overload splitting remains a tracker-side pre-solve policy,
  - historical-conflict relaxation remains a tracker-side around-solver retry policy.

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
- `predictor` and `updater` are now the primary constructor boundary objects,
- local hypothesis generation is still PDA-style in this phase via a transitional internal backend path,
- output `Track` metadata is an explicit TOMHT-owned projection from the current active leaf node rather than arbitrary propagated metadata.

### Transitional constructor boundary

The tracker constructor now centers on `predictor` + `updater`.

Local branching still uses a hypothesis-generator object internally, but this is currently treated as a **transitional implementation detail**, not the intended long-term public abstraction. Constructor precedence is:

1. explicit `hypothesis_generator`
2. deprecated `hypothesiser` compatibility path
3. internally constructed backend from `TOMHTParams.hypothesis_backend` (`"pda"` or `"robust_pda"`)

Default scoring setup no longer reads operational values from hypothesiser attributes. It now uses explicit tracker parameters such as `prob_detect`, `prob_gate`, and `clutter_density`.

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

---

## Current per-scan pipeline

The tracker’s current runtime pipeline is:

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

Local expansion is still Stone Soup-boundary-driven in this phase:

- for each active leaf, reconstruct a compatibility `Track`,
- call the local hypothesis generator,
- score local hypotheses through the scoring model,
- create child nodes for kept hypotheses,
- always keep a miss hypothesis if the generator returned one,
- then apply an optional per-tree local leaf cap.

### Local leaf cap

`max_leaves_per_track_tree` is explicitly treated as a **pre-solve safety valve**, not the main pruning semantics. Its default is intentionally high enough that it should act as tractability protection rather than the primary meaning of pruning.

### Clustering

Clusters are built from **full active-leaf historical detection-key overlap**, not current-scan-only overlap. This is an important correctness property: clustering and solver feasibility now use consistent full-history exclusivity semantics. Detection keys use the format `(scan_index, det_index)`.

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

This is cleaner than the earlier branch-local miss handling, but it is also part of why low-quality trees can now persist longer than before.

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

Scoring remains based on the default beta-ratio-style model in `tomht_scoring.py` unless an explicit alternative scoring model is supplied.

Current default behavior:

- scores local track hypotheses using PDA-style β-ratio approximations,
- scores unused detections through a clutter-density-derived per-unused term,
- applies a fixed birth penalty for births,
- logs scoring diagnostics at tracker construction time.

This scoring should still be understood as **pragmatic and serviceable**, not final.

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
- scan wall time
- memory / node counts

This instrumentation has been important for replay diagnosis and remains part of the tracker’s practical observability story.

### Determinism

The tracker now has deterministic behavior in the investigated birth-capping path:

- initiator outputs are converted from unordered set-like candidate pools into a deterministic heuristic ranking before `max_births_per_scan` is applied,
- the ranking restores the previous support/miss/age/covariance-oriented quality heuristic before deterministic tie-break fields,
- this removes process-level nondeterminism that had been introduced by slicing directly from initiator outputs.

The broader tracker is intended to be deterministic, and this remains an important operational expectation.

---

## Replay/runtime status

The current code can complete the main recorded replay to end-of-file, which was the immediate robustness goal after the transition.

The main remaining issue is no longer basic correctness instability, but runtime concentration in high-combinatoric merged-cluster scans.

Current runtime snapshot from the present replay checkpoint is roughly:

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

## Internal births: current interpretation

Internal births should currently be understood as:

- simple,
- heuristic,
- secondary relative to the external-start integration path,
- and not yet the final word on birth/existence semantics.

### What is solid in the current birth path

The current residual policy uses detections unused by the **union of all active leaves** after local expansion. This is conservative, but it preserves a strong no-conflict invariant for internal births and for residual-based external starts.

Birth capping is now deterministic and quality-ranked rather than arbitrary.

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
- deterministic birth capping,
- and replay integration with end-to-end recorded replay completion.

---

## What remains provisional / future-work territory

The following are still provisional or explicitly not the final word:

### 1. Runtime / solver story

The current rebuild step is still exhaustive enumeration. Large merged clusters remain expensive even with overload splitting. A more scalable K-best cluster solver remains a clear future target.

### 2. Overload splitting semantics

Overload splitting is useful and explicit, but still an approximation. It can later induce committed historical overlap, which is why historical relaxation exists. This is acceptable for now, but not conceptually final.

### 3. Historical-relaxation safety net

Historical conflict relaxation is narrow and pragmatic. It is a good safety net, but not the final principled treatment of approximation-induced overlap.

### 4. Local hypothesis-generation ownership

The public constructor boundary has shifted to predictor/updater, but local branching is still internally driven by a PDA-style generator path. This is an intentional transitional state, not necessarily the final local-association design.

### 5. Scoring design

The beta-ratio scoring model remains pragmatic rather than fully settled. Cleaner decomposition of local hypothesis score contributions, existence/survival interpretation, and reduced dependence on PDA-style packaging remain open topics.

### 6. Internal birth / existence semantics

Internal births remain simple, heuristic, and secondary. Their current semantics appear more permissive with respect to false-start persistence than the pre-transition behavior, and are not yet conceptually final.

### 7. Tracking quality / false-start tuning

Replay output is now usable enough that false starts and similar quality issues can be revisited meaningfully, but they are not the single main blocker at the current checkpoint.

### 8. Internal organization around cluster build/rebuild

There is now a clearer internal seam around cluster solving, but `_build_track_clusters(...)` and `_rebuild_cluster_globals(...)` still contain a lot of closely related logic. Further organization work is likely warranted, but should preferably happen as a sub-goal of deeper work rather than as a standalone cleanup-only phase.

---

## Recommended interpretation of the current checkpoint

This checkpoint should be understood as:

- a **real track-oriented TO-MHT implementation**
- with a coherent persistent-tree / rebuilt-global architecture
- that survives the target recorded replay end-to-end
- and is suitable for continued replay-based evaluation, integration work, and next-phase planning

But it should also be understood as:

- still pragmatically tuned,
- still carrying a few explicit approximation/safety-net paths,
- still performance-limited on large merged clusters,
- and still carrying some unresolved birth/existence and scoring questions.

This is a good place to begin a consolidation-and-priority-reset phase before the next deeper round of work.
