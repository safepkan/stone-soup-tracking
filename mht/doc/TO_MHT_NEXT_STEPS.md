# TO-MHT Next Steps

## Next architectural subphase

**Local expansion volume and frontier control**

The scoring/birth/lifecycle/API phase is now effectively complete. The tracker has coherent NLL/LLR scoring, explicit start priors, dynamic detection probability / clutter-density support through the DPM, sticky confirmation and publication, score-based whole-track deletion, and a much cleaner module structure.

The next phase returns to the earlier performance topic:

> reduce local expansion volume and frontier growth, using the now-coherent scoring/lifecycle foundation.

This phase should be exploratory. The goal is not to immediately add a large new pruning mechanism, but to understand where expansion volume comes from, which expanded hypotheses matter, and which conservative controls are safe.

---

## Current timing motivation

The latest standard replay summary still shows local expansion as the dominant runtime component:

```text
scan_wall_ms          median 12.297 ms   p95 94.690 ms   max 217.822 ms
expand_ms             median 11.118 ms   p95 86.764 ms   max 187.001 ms
expand_hypothesise_ms median  8.761 ms   p95 65.661 ms   max 148.591 ms
expand_update_ms      median  1.000 ms   p95  8.121 ms   max  27.273 ms
cluster_build_solve   median  0.550 ms   p95  7.500 ms   max  26.265 ms
```

This is substantially better than earlier phases, but the shape is unchanged: most replay time is still spent expanding local hypotheses, especially hypothesiser calls. The branch-and-bound cluster solver is no longer the main bottleneck on this workload.

The next improvement is therefore likely to come from reducing how much expansion work we ask the local association path to do, not only from making each individual association call faster.

---

## Current baseline to preserve

Update (2026-05-16): active cluster/solver conflicts now use live unresolved
detection keys, computed as each leaf's full-lineage `detection_history_keys`
minus `TrackTree.committed_detection_keys`. MAP-only N-scan pruning adds the
promoted child's full detection history to the tree-level committed key set,
while `committed_states` still records the output-state prefix before the
promoted root. Historical-conflict relaxation has been removed from the runtime
path; overload splitting remains active and is still a separate review item.

The following pieces should be treated as the current foundation, not reopened casually:

- track trees are the persistent frontier,
- globals are rebuilt per cluster from current leaves every scan,
- local association is distance-hypothesis based,
- local detection-hypothesis distance is NLL only,
- the NLL scorer applies `log(P_D) - log(lambda) - NLL` for hits and `log(1 - P_D)` for misses,
- unused-detection scoring remains removed,
- starts enter with explicit existence log-odds / probability priors,
- confirmation, deletion, and publication are separate tree-level concepts,
- standard output publishes confirmed tracks by default,
- score-based whole-tree deletion is active,
- DPM support is the caller-facing hook for dynamic `P_D` / clutter density,
- the tracker is now mostly an orchestrator delegating pipeline phases to modules.

Expansion-volume work should build on this foundation rather than reintroduce older PDA/beta/unused-detection semantics.

---

## Main questions for this phase

### 1. Where does expansion volume come from?

Characterize expansion pressure in replay and smoke scenarios:

- active tree count per scan,
- active leaf count per scan,
- leaves expanded per scan,
- hypothesiser calls per scan,
- detections per scan,
- children generated per expanded leaf,
- children retained after local ranking/capping,
- leaves later selected by cluster MAP or retained top-K globals,
- leaves later pruned as unsupported.

The useful question is not just “how many leaves exist?” but:

> which expansion work contributes to retained global hypotheses, and which work is systematically wasted?

### 2. Which controls are semantic and which are tractability safety valves?

Some controls affect tracking semantics directly. Others are safety valves to keep compute bounded.

This phase should make that distinction explicit. Candidate controls include:

- local association gate size,
- `max_children_per_track`,
- `max_leaves_per_track_tree`,
- post-solve supported-leaf pruning,
- N-scan pruning,
- score-based whole-tree deletion,
- publication gates,
- birth caps / guardrails,
- overload cluster splitting.

Avoid mixing semantic pruning and emergency tractability caps without documenting the distinction.

### 3. Can scores safely reduce expansion work?

Now that score semantics are cleaner, consider conservative score-aware expansion/frontier ideas:

- avoid expanding very low-score tentative trees that are near deletion,
- use tree lifecycle state to prioritize confirmed trees over tentative trees,
- cap tentative-tree frontier more aggressively than confirmed-tree frontier,
- prune or deprioritize leaves far below the best leaf in the same tree,
- use max active-leaf score / tree score as a pre-expansion priority signal,
- explore whether score deletion should happen earlier in the scan for hopeless trees.

Do not jump straight to aggressive pruning. Start by measuring how often such rules would fire and whether they would have changed retained MAP/top-K leaves.

### 4. How much of the cost is avoidable hypothesiser work?

The latest timing shows `expand_hypothesise_ms` dominates `expand_ms`. Investigate:

- repeated expansion of leaves that are later discarded,
- whether all active leaves need a full hypothesiser call each scan,
- whether some leaf groups share enough prediction/gating work to reuse more,
- whether confirmed/tentative state can drive expansion budgets,
- whether detection count or gate behavior explains high-tail scans.

Parallelization is a future axis, but not the first tool for this phase. First understand and reduce unnecessary work.

---

## Specific investigation items

### A. Expansion usefulness instrumentation

Implemented (2026-05-15): scan stats now carry aggregate expansion/frontier
usefulness counters by default. They connect local expansion work to retained
top-K supported leaves, MAP-selected leaves, supported-leaf pruning removals,
confirmed/tentative expansion split, and supported-pruning impact.
The compact `EXPANSION_FRONTIER ...` per-scan line and
`SUMMARY expansion_frontier ...` aggregate line remain opt-in via
`debug_display_expansion_frontier` or `TOMHT_DEBUG_EXPANSION_FRONTIER=1`, so the
default smoke/replay log shape is unchanged.

Add or extend diagnostics to connect expansion work to later retained hypotheses.

Possible metrics:

- expanded leaves per scan,
- generated children per scan,
- retained local children per scan,
- children that survive post-solve supported-leaf pruning,
- children that appear in cluster MAP,
- children that appear in any retained top-K cluster global,
- per-tree expanded/retained ratios,
- confirmed vs tentative expansion counts,
- score distribution of expanded leaves.

This can start as debug/stat output rather than permanent public API.

### B. Frontier growth analysis

Implemented (2026-05-15): `ExpansionFrontierStats` samples active tree/leaf
counts before expansion, after local expansion/capping, after empty-tree removal,
after births, after post-solve supported-leaf pruning, after MAP-only N-scan
pruning, and after lifecycle deletion.

Inspect how active-leaf counts grow and shrink through the scan pipeline:

1. before local expansion,
2. after local expansion,
3. after local leaf cap,
4. after cluster solve,
5. after post-solve supported-leaf pruning,
6. after N-scan pruning,
7. after lifecycle deletion.

This should clarify whether the main issue is local branching, weak pruning, overload-split behavior, births, or some combination.

### C. Overload-split soundness

Implemented (2026-05-17): overload splitting is now an internal recursive
cluster-solve strategy. Rebuild still starts from the original live cluster; if
the projected leaf product exceeds the overload threshold, the solver chooses a
binary weak-link split, recursively solves conditional subproblems with cut-key
forbiddance assignments, recombines left/right solutions, and rejects any
recombined global that is infeasible under the original live conflict keys.

The downstream snapshot shape is now one `ClusterRebuildSnapshot` per original
cluster. Split subclusters are diagnostic only and are no longer exposed as
ordinary downstream clusters. `snapshot.rebuilt_globals`, `snapshot.map_global`,
MAP merge, N-scan pruning, lifecycle/output, and supported-leaf pruning all see
feasible globals for the original cluster.

The old `overload_split_supported_pruning_policy` experiment and overload
supported-pruning skip counters have been removed. Supported-leaf pruning now
applies uniformly to every non-empty retained-global cluster snapshot, including
clusters solved through overload recursion.

Implemented (2026-05-17): recursive overload conditioning now memoizes identical
subproblems within one original-cluster solve, keyed by
`(track_ids, inherited_forbidden_keys)`. `OVERLOAD_SPLIT ...` diagnostics now
include recursive cache hit/miss counts, max recursion depth, max cut-key count,
total interface assignments, max recombination product size,
`branch_recomb_retained`, `final_recomb_retained`, and
`interface_assignment_cap_fallbacks`.

Remaining overload-solve review points:

- interface-assignment fallback behavior when a cut has many contested keys,
- remaining recombination candidate volume and timing on replay-heavy scans,
- whether a future K-best solver hint/warm-start can recover better quality
  without changing the downstream feasible-global invariant.

### D. Birth ranking and capping

Internal-birth capping is not the main expansion bottleneck, but it remains a heuristic control.

Keep on the review list:

- how often `max_births_per_scan` fires,
- whether initiator output quality/confidence metadata is available,
- whether `existence_probability` / `existence_log_odds` should influence candidate ordering,
- whether cap firing indicates the initiator should filter more aggressively upstream.

Do not let this distract from local expansion unless metrics show it matters.

### E. Candidate score/frontier pruning experiments

After instrumentation, try low-risk “what would happen if” analyses before changing behavior:

- mark leaves below relative score thresholds but do not prune,
- mark tentative trees that would be skipped or capped more aggressively,
- mark leaves that never survive to retained globals,
- compare these marks against MAP/top-K usage.

Use these experiments to decide which pruning rule is actually safe.

---

## Likely implementation direction

The first implementation branch should probably be instrumentation-heavy:

1. add expansion/frontier usefulness metrics,
2. run smoke and replay summaries,
3. identify the highest-volume scans and trees,
4. inspect whether wasted work is concentrated in tentative trees, low-score leaves, overload-split clusters, births, or broad gates,
5. only then add a conservative control.

The second branch can introduce one targeted control, for example:

- tentative-tree expansion cap,
- relative per-tree leaf score pruning,
- earlier score deletion for hopeless trees,
- overload recursive-solve quality/performance tuning,
- or a confirmed/tentative frontier budget.

Which one comes first should be data-driven.

---

## Non-goals for this phase

Do not yet:

- redesign local scoring,
- reintroduce unused-detection scoring,
- redesign start priors,
- remove the DPM API,
- make large publication/lifecycle changes,
- retune all scenarios by hand,
- implement broad parallel local expansion,
- rewrite cluster solving,
- or collapse overload-split approximation without understanding its impact.

Parallel expansion remains a plausible later path, but it should be opt-in and should come after we understand whether we can simply avoid a meaningful fraction of the work.

---

## Acceptance criteria

This phase is successful when we have:

- clear metrics showing where expansion volume comes from,
- clear metrics showing which expansion work is retained/useful,
- overload-split soundness maintained while supported-leaf pruning runs uniformly,
- at least one conservative expansion/frontier control identified or implemented,
- smoke/replay baselines updated when behavior changes intentionally,
- no regression in the coherent scoring/start/lifecycle/API model,
- and a clearer basis for deciding whether later parallelization is worth pursuing.

---

## Working notes

The current performance situation is not bad compared with earlier phases, but the distribution still has a long tail. The goal is to reduce that tail without making the tracker harder to reason about.

The most important principle for this phase:

> measure usefulness before pruning.

The tracker now has interpretable scores and lifecycle states. Use them to understand the frontier before using them to cut it.
