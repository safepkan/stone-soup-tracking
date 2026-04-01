# TO-MHT Next Steps

## Update (2026-04-01): overload cluster splitting mitigation implemented

Implemented (narrow pass):

- overload-aware approximate decomposition for oversized exact clusters
- trigger uses projected Cartesian combinations per cluster against
  `overload_split_projected_combination_threshold` (default `500000`)
- decomposition removes weakest conflict edges iteratively using pure
  full-history shared detection-key counts (`len(shared_history_keys)`)
- resulting connected components are solved exactly as independent subclusters
- clear runtime instrumentation now reports `OVERLOAD_SPLIT ...` events plus
  scan-level `split_clusters` / `split_ops` counters

Controls:

- `overload_split_enabled=True`
- `overload_split_projected_combination_threshold=500000`
- `overload_split_max_edge_removals_per_cluster=None`

## Update (2026-04-01): local cap relaxed to pure safety-valve range

Following the post-solve supported-leaf pruning refinement:

- default `max_leaves_per_track_tree` was relaxed to `100` (from `8`)
- local capping remains a pre-solve safety valve, not the primary pruning basis
- added opt-in pruning-feasibility validation (`TOMHT_DEBUG_VALIDATE_PRUNING_FEASIBILITY=1`) to identify the first pruning stage that leaves a cluster infeasible
- clustering now uses full active-leaf historical detection-key overlap so cluster decomposition matches the solver conflict criterion

## Update (2026-04-01): post-solve supported-leaf pruning refinement

A narrow pruning refinement was added on top of the committed Phase D rewrite:

- after per-cluster top-K rebuild, each tree in that cluster now keeps only leaf nodes that appear in at least one retained rebuilt global for that cluster
- local per-tree leaf capping (`max_leaves_per_track_tree`) remains in place only as a pre-solve safety valve
- pruning remains cluster-local; no cross-cluster merge step was added for pruning

## Implementation status (2026-03-31)

Phase D has now been implemented in code as the primary tracker architecture:

- explicit persistent `TrackTree` state with active leaf frontiers
- per-scan clustering and rebuilt cluster globals from current leaves
- exhaustive enumeration solver with explicit per-combination cluster-local unused-detection term
- `(scan_index, det_index)` detection keys for conflict checking
- MAP-only N-scan tree pruning with disagreement statistics from rebuilt alternatives
- simple `max_missed` leaf/tree lifecycle deletion
- simple internal births from Step-2 residual detections and external starts as new single-node trees
- practical per-tree active-leaf cap added (`max_leaves_per_track_tree`) as a pre-solve safety valve to keep exhaustive-enumeration runtime bounded in longer scenario runs

## Update (2026-04-01): narrow tractability controls tightened

Without changing the Phase D architecture:

- per-tree active-leaf default was tightened to `max_leaves_per_track_tree=8`
- internal births gained explicit load guards for active tree/leaf counts
- cluster-local unused-detection scoring now reuses one prebuilt cluster context
- optional hard projected-combination guardrail can now stop oversized cluster rebuilds explicitly

This planning document is therefore now mainly a reference/checklist for follow-up refinements, not a future architectural target.

## Next architectural phase

**Phase D: transition to a true track-oriented TO-MHT**

The first handoff release is now complete and available for initial ISAC integration work.

That release is good enough for:

- initial pipeline integration,
- workshop discussion,
- replay-based evaluation,
- and continued collaboration.

However, the workshop, presentation work, and architectural review made the main next step clear:

> move from the current node-based but still partly global-hypothesis-oriented implementation to a **true track-oriented TO-MHT** in which track trees / track hypotheses are the primary persistent state and global hypotheses are rebuilt from the current track set at each update.

This document now serves as the planning document for that transition.

---

## 1. Core architectural goal

### Desired end state

At the end of this phase, the tracker should be organized around the following principle:

- what persists from one update to the next is primarily the **track-tree / track-hypothesis structure**
- global hypotheses are **reconstructed at each scan** from the current surviving track hypotheses and their incompatibilities
- the tracker should no longer treat the previous scan's explicit list of global hypotheses as the main persistent search state

This is the key correction relative to the current handoff release.

### Public API stability

The public operational API should remain unchanged through this phase:

- `update_tracker(time, detections) -> (time, tracks)`
- `tracks`
- `add_external_starts(time, starts)`
- `get_unused_detections()`

Public debug / inspection helpers and instrumentation output may change as appropriate to fit the new architecture.

### Why this phase now

Reasons to prioritize this next:

- this is the clearest remaining architectural gap after the first handoff
- it aligns the implementation more directly with standard TO-MHT descriptions in textbooks/tutorials/papers
- it should simplify the conceptual model of the tracker
- it creates a better foundation for pruning, clustering, lifecycle work, and later optimization
- it should make the implementation easier to reason about and evolve after integration starts

---

## 2. Persistent vs transient state

This distinction should be made explicit in the redesign.

### Persistent state

The following should persist across scans:

- explicit track trees / track families
- all unpruned track-hypothesis nodes within those trees
- root / leaf structure per tree
- per-node state, score, association, counters, provenance
- stable logical track IDs
- N-scan / commitment-related persistent state as needed
- minimal long-lived tracker statistics / counters

Important clarification for this phase:

- when a local child hypothesis is materialized as a new `TrackHypothesisNode`, it becomes part of the persistent tree structure
- the primary persistent frontier is therefore the set of current leaf nodes across the track trees, not a persistent list of global hypotheses

### Per-scan transient state

The following should be rebuilt fresh on each update:

- temporary hypothesiser / gating results before node materialization
- conflict / incompatibility relations among current candidate track hypotheses
- per-scan track clusters
- rebuilt cluster-level global hypotheses
- solver outputs for best / K-best cluster hypotheses
- temporary score tables and scratch structures

### Conceptually transient but useful for debug / inspection

Some transient structures may be worth keeping available until the next update for inspection:

- clusters from the most recent update
- why tracks were clustered together
  - for example shared detections / incompatibility links
- rebuilt globals from the most recent update
- current cluster MAP selections
- current active leaves per track tree
- full current set of track trees
- statistics on disagreement between MAP-based pruning and alternative globals

This category should be treated deliberately:

- not persistent in the architecture,
- but intentionally exposed for debugging / visualization / validation.

---

## 3. Intended target data structures

### 3.1 TrackHypothesisNode

`TrackHypothesisNode` will likely remain the core per-step hypothesis object, but should evolve.

Expected changes:

- keep:
  - `track_id`
  - `node_id`
  - `parent`
  - `scan_index`
  - `timestamp`
  - `state`
  - association identity
  - score contribution / accumulated score inputs as needed
  - cached counters / provenance
- add:
  - child pointers or child IDs
- likely change:
  - node mutability, so child links can be maintained directly
- possibly refine:
  - score fields if clearer accumulated/local score separation is helpful
  - miss/lifecycle-related fields once the new architecture is in place

### 3.2 Explicit TrackTree / TrackFamily structure

Introduce an explicit track-tree structure.

Likely responsibilities:

- stable logical track identity
- root pointer / root node ID
- current leaf set
- maybe direct node registry for that tree
- metadata / provenance
- maybe a quick way to inspect active frontier and depth

Goal:

- make the “one logical track = one tree/family” idea explicit in code,
- rather than implicit in node parent chains only.

Note for this phase:

- a `TrackTree` does **not** need to carry a separate committed Stone Soup `Track` object yet
- output/history views can still be reconstructed from tree structure directly
- reconstructable history is therefore limited to the depth retained in the tree, which is acceptable for this phase

### 3.3 Current tree set

The tracker should hold an explicit collection of current track trees, for example something like:

- `track_trees_by_track_id`
or equivalent

This should become part of the primary persistent tracker state.

### 3.4 Cluster structures

Introduce explicit per-scan cluster concepts.

A cluster should represent a set of track trees that currently interact and therefore must be solved jointly.

Useful contents for a cluster:

- participating track IDs
- participating leaf hypotheses / candidate track hypotheses
- shared detections or conflict links that caused clustering
- rebuilt globals for that cluster
- cluster MAP selection
- optional debug/inspection explanation

These clusters do **not** need to persist across scans.

### 3.5 GlobalHypothesis

`GlobalHypothesis` probably still remains useful, but should change role.

Desired role:

- represent a current globally consistent set of selected track hypotheses
- mainly as a per-scan / per-cluster rebuilt object
- not as the main persistent scan-to-scan frontier

This is a role change more than necessarily a complete type removal.

---

## 4. Intended update pipeline

The desired scan update should look approximately like this.

### Step 1: Extend all track trees

For each active leaf in each track tree:

- call the hypothesiser at the new timestamp, as in the current implementation
- use the hypothesiser/updater boundary exactly as before
- translate returned local hypotheses into child nodes
- create matched and miss children
- use the **same local per-track scoring model as before** in this phase unless the rewrite forces a small mechanical cleanup

This phase is still hypothesiser-driven at the Stone Soup boundary; the architectural rewrite is about how the resulting hypotheses are stored and combined.

### Step 2: Minimal local pruning and simple lifecycle handling

In the first version, local pruning and lifecycle handling are intentionally simple.

At most:

- cap the number of children per leaf
- **always keep a miss branch if the hypothesiser returned one**, matching current behavior
- remove any leaf with `missed_count > max_missed` from the active leaf set
- remove an entire track tree if it has no surviving active leaves

This is the simple first-version replacement for the current architecture's per-global drop semantics.

Stronger local pruning and broader lifecycle design can be added later if needed.

### Step 3: Form independent track clusters

Build clusters from current track-tree interactions.

Planned approach:

- collect the set of detections currently used / competed for by each track tree frontier
- create a graph where track trees are connected if their current candidate sets intersect / conflict
- extract connected components
- treat each connected component as an independent cluster for global reconstruction

Important design choices:

- clusters are recomputed on each update
- they are **not** maintained as persistent tracker objects across scans
- in this phase, clustering models only **measurement-exclusivity conflicts** induced by shared detections among current candidate track hypotheses

This should simplify the architecture and provide a major performance win in many scenes.

### Step 4: Rebuild globals per cluster

For each cluster:

- construct the current incompatibility / conflict structure among candidate track hypotheses
- solve for one or more globally consistent leaf selections
- represent the result in a solver-independent way

The rebuilt globals should now be **derived from the current cluster track hypotheses**, not inherited directly from a persistent old global frontier.

### Step 5: N-scan pruning on explicit track trees

In this first rewrite, N-scan pruning is intentionally based on the **MAP global hypothesis only**.

At pruning depth `N`:

- for each tree old enough to prune, identify the child of the current root that contains the MAP-selected leaf
- keep that child and promote it to be the new root
- remove its siblings
- trees younger than the pruning depth are left unchanged

This is a deliberate simplification for the first version.

To assess whether this is too aggressive in practice, the implementation should collect disagreement statistics between:

- the MAP-selected pruning decision
- and the alternative rebuilt globals for that cluster

### Step 6: Extract MAP output

Keep output simple initially:

- output MAP only, as today
- reconstruct output tracks from the selected leaf hypotheses
- combine cluster outputs into one tracker output set

### Step 7: Keep debug / inspection artifacts from the last update

Retain useful transient structures from the last scan for inspection:

- current trees
- clusters
- rebuilt globals
- cluster explanations
- current MAP selection
- pruning disagreement statistics

Debug / instrumentation printing will likely need to change in this phase.

Goal:

- keep a similar level of usefulness for debugging and scenario comparison
- do not commit in advance to the exact output format
- make a best effort to provide a reasonable set of per-scan and summary outputs adapted to the new architecture

---

## 5. Solver plan

### 5.1 Solver abstraction first

The tracker should not depend directly on one specific global-hypothesis solver implementation.

Introduce a wrapper / abstraction layer so the main code can stay independent of:

- exhaustive enumeration in the first implementation
- pure Python Murty implementation later
- optimized external Murty implementation later
- future replacements

### 5.2 Exact first-version optimization problem

Per cluster, define the solver input as:

- a list of track trees in the cluster:
  - `[track_1, track_2, ..., track_T]`
- for each track, its current active leaves:
  - `track_i.leaves = [leaf_i_1, leaf_i_2, ..., leaf_i_Li]`
- for each leaf:
  - `leaf.score`
  - `leaf.detections`, the set of detection identifiers used within the unresolved window / current tree depth relevant to conflict checking

Detection identifier format in this phase:

- use `(scan_index, det_index)`-style keys
- not per-scan indices alone
- so cross-scan references inside the unresolved window cannot collide

Decision variable:

- choose exactly one active leaf from each surviving track tree in the cluster

Feasibility constraint:

- a combination of selected leaves is feasible iff for every pair of selected leaves from different tracks:
  - `leaf_i.detections ∩ leaf_j.detections = ∅`

Objective:

- maximize:
  - `sum(leaf.score for leaf in selected_leaves)`
  - **plus** the same style of unused-detection penalty as before, now treated as a per-combination term

Explicit rule for the unused-detection term in this phase:

- compute it **cluster-locally**
- define each cluster's conflict universe as the **union of current-scan detection keys that appear in any active leaf candidate in that cluster**
- use only those cluster-local current-scan detections for the unused-detection term
- for a feasible cluster combination, count which of those cluster-local current-scan detections are unused by that combination
- add the corresponding penalty to that cluster combination score
- full-scan score is then the sum of cluster scores

Important clarification:

- committed/shared history within a tree contributes the same additive constant to all leaves of that tree, so it does not affect which combination is optimal; it only shifts absolute scores
- the unused-detection term is not forced into per-leaf scores in this phase; it remains an explicit per-combination/global-style term

### 5.3 First implementation: exhaustive enumeration

For the first version, use exhaustive enumeration:

- generate the Cartesian product of the leaf lists in a cluster
- filter combinations for feasibility
- score each feasible combination
- return the best

To support pruning-disagreement statistics:

- exhaustive enumeration should also retain enough per-scan information about non-MAP combinations to compare their pruning choices against the MAP choice before those alternatives are discarded
- those alternatives do **not** need to persist as long-lived tracker state

This is acceptable for the first version because:

- clusters are expected to stay fairly small
- the main goal is correctness and architectural clarity
- clustering and minimal pruning should already reduce the search space substantially

### 5.4 Later optimization

After the architecture is working, solver replacement can be a separate step.

Later options may include:

- Murty-style K-best ranking
- optimized external implementations
- other assignment / relaxation formulations if profiling shows the need

The rest of the tracker should depend only on a small solver interface and not on solver internals.

---

## 6. Birth and external-start handling in this phase

Births are **not** the main architectural priority of this phase.

The ISAC path currently does not use internal births, so this part can stay intentionally modest.

### Internal births

Minimal acceptable approach:

- still use the `initiator` passed via the constructor when it is not `None`
- determine birth input detections **after Step 2**
- specifically, use the current-scan detections unused by the **union of all surviving active leaves after local pruning / simple lifecycle filtering**
- feed only those detections to the initiator
- create new birth trees from the resulting initiated tracks
- under that rule, births do not conflict with existing track trees
- assume the initiator does not generate mutually conflicting birth candidates unless proven otherwise
- add birth trees late in the update flow and let normal later survival determine whether they persist

### External track starts

External starts should remain supported.

Planned interpretation in this phase:

- each external start becomes a new single-node track tree
- as with births, assume external starts are created only from currently unused detections
- under that rule, they are effectively separate new clusters with no conflicts to existing trees at insertion time

### Explicit risk note

With minimal birth handling and incomplete lifecycle logic:

- internal births may be over-produced
- or published too early in some scenarios

This is acceptable for the first rewrite because ISAC does not use internal births, but should be monitored in synthetic/replay validation.

### Explicit non-goal for this phase

Do **not** let internal births derail the main TO-MHT transition.

If needed:

- keep births simple,
- mark them as provisional,
- revisit later with a cleaner lifecycle / candidate / tentative / confirmed model.

---

## 7. Pruning expectations in this phase

The first track-oriented rewrite does **not** need to solve all pruning questions.

Likely enough for the first version:

- minimal local branch limits
- simple `max_missed`-based leaf / tree deletion
- clustering
- rebuilt globals by exhaustive enumeration
- MAP-only N-scan pruning

That may already be sufficient to make many scenarios tractable.

Further pruning can be deferred unless the rewrite immediately shows a need.

---

## 8. Debug / visualization goals

This should be an explicit part of the target state.

The tracker should make it reasonably easy to inspect:

- the full current set of track trees
- active leaves in each tree
- current clusters
- why trees were clustered together
- rebuilt globals per cluster
- current MAP output
- disagreement between MAP-based pruning and alternative rebuilt globals

Suggested direction:

- add read-only debug/inspection properties or snapshot helpers
- keep the public operational API stable
- make internal structure easier to visualize during validation

This will be important both for development and for explaining the rewrite.

---

## 9. Expected impact on tests and parameters

### 9.1 Tests

Some current tests will no longer make sense because they are tied to the current architecture.

Expected actions:

- remove tests that assume persistent scan-to-scan global-hypothesis-frontier mechanics
- rewrite tests that should still hold conceptually but need new structural expectations
- add tests for:
  - explicit track trees
  - per-scan clustering
  - rebuilt globals
  - MAP-based tree pruning
  - simple leaf/tree deletion under `max_missed`
  - debug snapshots if exposed

### 9.2 TOMHTParams

`TOMHTParams` will likely need revision.

Reason:

- some parameters are tied to concepts from the current architecture

Expected approach:

- keep stable, still-meaningful knobs
- rename / remove parameters tied to old mechanics
- add new parameters only where clearly justified by the new architecture
- avoid overfitting the parameter block too early

---

## 10. Validation strategy after rewrite

Validation should be staged.

### Stage 1: synthetic scenarios

Use the existing scenario set first.

Goal:

- verify that the tracker is back in a sane working state
- check that outputs still make sense qualitatively
- catch obvious regressions in branching / scoring / pruning

### Stage 2: replay data

After synthetic scenarios are working:

- run the tracker on recorded data
- check that outputs remain sensible
- inspect clusters / trees / MAP outputs
- compare behavior and performance with the current handoff implementation

This should give a reasonable level of confidence that the core rewrite is sound.

---

## 11. Execution strategy

### Preferred implementation style

This is likely best treated as a **coherent architectural rewrite**, not a very long chain of tiny patches.

Reasons:

- the target architecture is different in a fundamental way
- many parts depend on each other
- intermediate half-converted states may be more confusing than helpful

### Practical note about using Codex

If Codex is used for implementation:

- first make the target architecture description clear enough that a human engineer would find it unambiguous
- then try for a strong single-shot or mostly-single-shot implementation attempt
- if execution stalls or drifts:
  - diagnose whether the issue is the plan, the scope, or the code generation
  - decide whether to steer incrementally or refresh the plan and retry

The plan should be treated as the most important dependency.

---

## 12. Proposed phase outcome

This phase should be considered successful if, at the end:

- explicit track trees exist
- globals are rebuilt from current surviving track hypotheses rather than propagated scan-to-scan as the main state
- per-scan clustering exists and works
- MAP-based pruning operates naturally on explicit trees
- simple leaf/tree deletion under `max_missed` works
- current public API remains usable for integration
- synthetic scenarios work again
- replay data gives sensible results
- the resulting code is conceptually closer to the TO-MHT model described in the workshop

---

## 13. Immediate next actions

1. Clean up `TO_MHT_CURRENT_STATE.md` so it accurately reflects the first handoff release.
2. Update this document into the true TO-MHT transition plan.
3. Review the target-state description carefully before implementation.
4. Then attempt the architectural rewrite.
