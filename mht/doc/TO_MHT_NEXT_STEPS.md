# TO-MHT Next Steps

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
- track-hypothesis nodes within those trees
- root / leaf structure per tree
- per-node state, score, association, counters, provenance
- stable logical track IDs
- N-scan / commitment-related persistent state as needed
- minimal long-lived tracker statistics / counters

### Per-scan transient state

The following should be rebuilt fresh on each update:

- local candidate child hypotheses for the current scan
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

- predict to the new timestamp
- gate detections
- generate feasible child hypotheses
- create matched and miss children
- keep existing scoring semantics initially unless the rewrite forces a small cleanup

This part should be close to what the tracker already does conceptually.

### Step 2: Local pruning of weak branches

After local expansion:

- cap children per leaf
- keep miss branch as needed
- apply miss-budget style local elimination if still appropriate
- optionally apply small local duplicate suppression if useful

The intent is to keep the candidate track-hypothesis set manageable before rebuilding globals.

### Step 3: Form independent track clusters

Build clusters from current track-tree interactions.

Planned approach:

- collect the set of detections currently used / competed for by each track tree frontier
- create a graph where track trees are connected if their current candidate sets intersect / conflict
- extract connected components
- treat each connected component as an independent cluster for global reconstruction

Important design choice:

- clusters are recomputed on each update
- they are **not** maintained as persistent tracker objects across scans

This should simplify the architecture and provide a major performance win in many scenes.

### Step 4: Rebuild globals per cluster

For each cluster:

- construct the current incompatibility / conflict structure among candidate track hypotheses
- solve for:
  - best global hypothesis
  - ideally also K-best globals
- represent the result in a solver-independent way

The rebuilt globals should now be **derived from the current cluster track hypotheses**, not inherited directly from a persistent old global frontier.

### Step 5: N-scan pruning on explicit track trees

Use the best global hypothesis (or the surviving set if needed) to perform N-scan pruning on the explicit trees.

In an explicit track-tree setting, this should become much cleaner than in the current architecture.

Current working expectation:

- for each tree, identify the branch selected by the best global hypothesis
- prune away root-level alternatives older than the N-scan cutoff
- retain the unresolved recent tail

The exact pruning semantics should be written carefully once the explicit tree structure exists.

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

This should be explicit and deliberate rather than accidental.

---

## 5. Solver plan

### 5.1 Solver abstraction first

The tracker should not depend directly on one specific global-hypothesis solver implementation.

Introduce a wrapper / abstraction layer so the main code can stay independent of:

- Hungarian-only fallback path
- pure Python Murty implementation
- optimized external Murty implementation
- future replacements

### 5.2 Initial solver requirements

We need at least:

- best global hypothesis solver
- ideally K-best support

Likely options to evaluate:

- `scipy.optimize.linear_sum_assignment` for best assignment / baseline cases
- simple Python Murty implementation as an understandable first step
- `motrom/fastmurty` or similar later if needed

The first implementation should prioritize:

- correctness
- inspectability
- clean integration into the tracker architecture

over ultimate speed.

### 5.3 Keep main code independent of solver details

The rest of the tracker should depend on a small interface such as:

- solve best cluster global
- solve K-best cluster globals

and not care how the solver works internally.

---

## 6. Birth handling in this phase

Births are **not** the main architectural priority of this phase.

The ISAC path currently does not use internal births, so this part can stay intentionally modest.

### Minimal acceptable approach

A pragmatic first-pass approach is acceptable, for example:

- create new birth trees from detections not explained by surviving associations
- assume those births do not conflict with existing track trees if generated only from unassociated detections
- assume the initiator does not generate mutually conflicting births unless proven otherwise
- add birth trees late in the update flow and let normal later survival determine whether they persist

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

- local branch limits
- clustering
- K-best per-cluster reconstruction
- N-scan pruning

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
  - tree-based N-scan pruning
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
- N-scan pruning operates naturally on explicit trees
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
