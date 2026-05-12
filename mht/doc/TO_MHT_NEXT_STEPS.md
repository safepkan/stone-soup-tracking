# TO-MHT Next Steps

## Next architectural subphase

**Local expansion volume reduction / pre-expansion control**

Update (2026-04-20): timing output now includes explicit expansion-call attribution (`expand_hypothesise_ms` / `expand_update_ms` and call counts), which closes part of the "characterize expansion volume/cost" prerequisite and makes it easier to separate hypothesis-generation vs state-update time on replay.
Update (2026-05-11): a conservative pre-phase refactor extracted `TOMHTParams`, local expansion orchestration, internal-birth handling, cluster work construction, overload cluster decomposition, post-solve supported-leaf pruning, and TOMHT-specific scan/debug utilities from `TOMHTTracker` without changing scoring, pruning semantics, clustering semantics, overload-splitting behavior, local expansion behavior, birth behavior, or public API. `_last_unused_detections` assignment remains in the tracker as scan orchestration.
Update (2026-05-12): persistent tree/node bookkeeping moved into `mht/tomht_tree_store.py`; expansion, clustering, and post-solve pruning now accept `TrackTreeStore` as their persistent-state dependency; and new-track root creation/insertion now uses a store helper. This did not change ID behavior, node/tree semantics, scoring, pruning, clustering, lifecycle, or public API.

The recent local-association phase is now complete enough that the tracker has:

- a tracker-owned default distance hypothesiser,
- explicit NLL-based local scoring,
- Mahalanobis-threshold gating semantics,
- and a modestly optimized local-association math kernel.

That work clarified the runtime story and delivered worthwhile `expand_ms` reductions, but it also made the next bottleneck clearer:

> the main remaining local-expansion cost is now likely driven less by the per-detection math itself and more by how many leaves still require full expansion.

So the next subphase is about investigating and reducing **expansion volume**.

---

## Why this subphase now

The current baseline now has:

- branch-and-bound as the default exact cluster backend,
- explicit timing-phase instrumentation,
- a tracker-owned local-association path,
- and enough regression/timing support to evaluate conservative runtime changes.

Recent local-association optimization passes improved `expand_ms`, including:
- rectangular pre-gating,
- limited reuse of predictions / measurement predictions / covariance prep,
- and Cholesky-based Mahalanobis/NLL evaluation.

Those passes were worthwhile, but the replay timing picture still suggests that the main remaining leverage is not only inside the local math kernel. The bigger question is now:

> which leaves actually need to be expanded, and where can that volume be reduced without damaging tracker behavior too much?

---

## Goal of this subphase

At the end of this subphase, the tracker should have:

1. a clearer understanding of which expanded leaves are actually useful,
2. at least one conservative mechanism for reducing how many leaves require full expansion,
3. a clearer picture of which knobs are semantic choices vs tractability controls,
4. preserved or acceptable replay/smoke behavior under those changes,
5. and updated docs/comments that describe the new runtime story honestly.

This phase is exploratory: it begins with analysis and characterization, then moves to targeted conservative implementation once the likely leverage points are clear.

---

## Core design intent

### 1. Reduce volume before trying to parallelize

Parallelization remains a plausible later axis, but it should not be the first answer here.

Before adding concurrency, we should first understand:
- how much expansion work is actually useful,
- how much of it is obviously low-value,
- and where conservative volume reductions are possible.

### 2. Preserve the clear runtime story

The runtime story should remain understandable:

- explicit trees are the persistent state,
- expansion produces local child leaves,
- globals are rebuilt later from surviving frontiers,
- and local runtime cost should be understandable in terms of both:
  - cost per expanded leaf
  - number of leaves expanded

This phase should improve that clarity, not obscure it.

### 3. Prefer conservative, inspectable controls

The next useful changes are likely to be things like:
- selective expansion,
- stronger pre-expansion filtering,
- better prioritization of which leaves are worth expanding,
- or better understanding of existing tractability guardrails.

These should be explicit and inspectable rather than hidden behind broad heuristics.

### 4. Keep quality concerns visible, but separate

The current baseline still has known quality concerns:
- false starts remain somewhat high,
- replay may show somewhat more target swapping / track jumping.

Those should remain visible, but the current subphase should stay focused on expansion-volume reduction rather than broad quality retuning.

---

## What should happen in this subphase

### 1. Characterize expansion volume

Before changing semantics, gather better evidence about:
- how many leaves are expanded per scan,
- how many children they produce,
- how many expanded leaves later survive supported-leaf pruning / MAP / N-scan,
- and whether many expanded leaves appear to be low-value or short-lived.

This should help identify whether the next best move is:
- fewer leaves entering expansion,
- fewer children retained per leaf,
- or some more selective expansion policy.

### 2. Identify plausible intervention points

Likely candidate areas include:
- per-tree leaf frontier entering expansion,
- local child retention count,
- selective expansion of only some leaves within a tree,
- pre-expansion score-based filtering / prioritization,
- or better use of existing tractability guardrails.

The goal is not to commit to all of these, but to identify which ones actually look promising from data.

### 3. Implement one conservative volume-reduction step

Once the likely leverage point is clearer, make one conservative targeted implementation pass.

Examples of the kind of thing that may be appropriate:
- modest selective expansion based on leaf score / support,
- stronger but explicit pre-expansion prioritization,
- or an adjustment that reduces clearly low-value expansion work without broadly changing the architecture.

The exact mechanism should be chosen after the analysis step, not assumed in advance.

### 4. Continue using the regression/timing harness

This subphase should lean on:
- smoke regression checks,
- replay compare helpers,
- timing summaries/logs,
- and output inspection where needed.

This is especially important because expansion-volume changes can be semantically meaningful even when they look like runtime-only work.

### 5. Keep the next possible later topics visible

This subphase does **not** mean parallelization is off the table.

But parallelization should stay a later, explicit architectural topic, likely with:
- opt-in behavior,
- a clean abstraction on top of the hypothesiser/orchestration boundary,
- and room for both tracker-owned and externally controlled parallel execution modes.

That is not the focus of this pass.

---

## What should **not** happen in this subphase

This phase should **not** yet:

- redesign the exact cluster solver again,
- do a full quality retuning pass,
- do a broad birth/existence redesign,
- do a full scoring-theory rewrite,
- or commit prematurely to a parallelization architecture.

The point here is to understand and reduce expansion volume first.

---

## Acceptance criteria

This subphase should be considered successful when:

- there is a clearer measured picture of where expansion volume is coming from,
- at least one conservative expansion-volume reduction has been implemented,
- replay/smoke behavior remains materially acceptable,
- timing data shows a worthwhile reduction in expansion-driven cost,
- and the docs/comments reflect the new baseline honestly.

A secondary success criterion is that the code and instrumentation make it easier to answer:

> how many leaves are we expanding, and which of those expansions are actually paying off?

---

## Recommended implementation style

This phase should follow the usual conservative style:

1. characterize expansion volume and usefulness,
2. choose a small, explicit intervention point,
3. implement one targeted volume-reduction pass,
4. validate on smoke/replay output and timing,
5. then decide whether another volume-control pass, a quality pass, or a later parallelization design step should follow.

That is the intended scope of this phase.
