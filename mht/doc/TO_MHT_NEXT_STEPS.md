# TO-MHT Next Steps

## Update (2026-04-11): experimental exact branch-and-bound backend implemented

Implemented a second experimental exact backend under the existing solver seam:

- added `mht/tomht_cluster_solver_branch_and_bound.py` with deterministic
  depth-first branch-and-bound solving,
- kept `mht/tomht_cluster_solver_exhaustive.py` as the reference backend,
- added exact 1-track score-sorted fast-path handling,
- used deterministic search ordering:
  - tracks ordered by fewer leaves first, then stronger conflict burden,
  - per-track leaves ordered by descending score,
- added a simple optimistic upper bound based on partial score plus best
  remaining-track score sums (ignoring conflicts),
- added branch-and-bound diagnostics (visited nodes, conflict-prune count,
  bound-prune count, complete feasible solutions found),
- set tracker default backend to
  `TOMHTParams.cluster_solver_backend="branch_and_bound"` (alias `"bnb"`),
- validated branch-and-bound against exhaustive on small exact problems in
  `mht/tests/test_tomht_cluster_solver_branch_and_bound.py`,
- documented tie-order caveat explicitly via set-based tie test:
  tied solution ordering can differ while selected sets and scores still match.

## Update (2026-04-11): added per-scan timing-phase breakdown

Implemented a granular timing instrumentation pass for diagnosis of large-CPU
scans:

- added `ScanTimingBreakdown` into `ScanStats`,
- timed major `update_tracker(...)` phases directly in tracker runtime code,
- kept `SCAN_TIMING` total wall-time output and added `SCAN_TIMING_PHASES`
  lines for per-scan phase attribution,
- added `SUMMARY timing_phases ...` med/max aggregates for run-level review.

## Update (2026-04-11): experimental exact OR-Tools backend implemented

Implemented this runtime/scalability follow-on step:

- added `mht/tomht_cluster_solver_ortools.py`, an experimental exact CP-SAT backend
  under the existing `ClusterSolver` contract,
- kept `mht/tomht_cluster_solver_exhaustive.py` as the reference backend,
- kept OR-Tools as opt-in experimental backend
  (`TOMHTParams.cluster_solver_backend="ortools"`),
- validated OR-Tools against exhaustive on small tractable exact problems in
  `mht/tests/test_tomht_cluster_solver_ortools.py`,
- routed OR-Tools returned candidates through exact float re-scoring plus the shared
  deterministic `TopKSolutionHeap` helper for backend-consistent ranking behavior,
- added `extra_k_best_iterations` (default `0`) as an optional OR-Tools constructor
  knob to run a few solves past K and reduce K-boundary rounding/tie risk,
- documented tie-order caveat explicitly: tied solutions can have different order
  than exhaustive even when selected sets/scores match.

## Update (2026-04-10): subphase implemented

Implemented this subphase as a conservative refactor:

- added a dedicated solver module (`mht/tomht_cluster_solver.py`) with an explicit exact cluster problem/result contract,
- moved the current exhaustive K-best cluster solve behind that contract in
  `mht/tomht_cluster_solver_exhaustive.py`,
- kept tracker-side problem preparation and result mapping in `tomht_tracker.py`,
- kept overload splitting as an explicit pre-solve tracker policy,
- kept historical-conflict relaxation as an explicit around-solver retry policy.

## Next architectural subphase

**Runtime / cluster-solver scalability, step 2: evaluate and harden alternative exact backends behind the existing solver seam.**

The remainder of this document currently includes the historical step-1 planning
notes; keep them as background context, but treat the concrete next work as
step-2 backend evaluation and scalability hardening.

This phase is intentionally **about backend experimentation and validation under the existing interface**, not a broader tracker/scoring redesign.

The tracker is now at a point where:

- the track-oriented persistent-state transition is complete enough to treat as the baseline,
- replay on the main recorded dataset reaches end-of-file,
- the main remaining technical weakness is runtime concentration in large merged clusters,
- and the current code already contains a conservative internal cluster-solver seam that can be made more explicit.

The immediate next step is therefore to define, in one place and in explicit terms, **what problem the cluster solver is actually solving today** and what the tracker expects back from it.

---

## Why this subphase now

This is the right first step for the runtime/scalability branch because it delivers several things at once:

- it makes the current solver assumptions explicit,
- it gives the current exhaustive implementation a clean home behind a formal interface,
- it reduces the amount of solver-related logic implicitly spread through tracker internals,
- and it prepares the codebase for later alternative backends without committing prematurely to any specific reformulation or algorithm.

This also fits the current state of the code well. The tracker already has:

- `_ClusterSolveInput`
- `_ClusterSolveOutcome`
- `_solve_cluster(...)`
- extracted historical-relaxation retry under that boundary
- and overload splitting wrapped around cluster solving

but the problem definition is still not yet expressed as a stable solver-facing contract.

---

## Goal of this subphase

At the end of this subphase, the tracker should have:

1. a **separate solver module** with a clean solver-facing interface,
2. a solver-facing problem definition that matches the tracker’s **actual current semantics**,
3. the current exhaustive K-best solver migrated to that interface with no intended behavior change,
4. tracker-side preparation and result-mapping logic that is thinner and easier to reason about,
5. and a clearer distinction between:
   - the **exact cluster K-best problem**
   - and the current **approximation/safety-net policies** around it.

This phase is successful even if runtime is not yet materially improved, provided the solver boundary becomes clear, explicit, and ready for follow-on work.

---

## Core design intent

### 1. Express the problem in its natural tracker-facing form

The solver interface should describe the cluster problem in the form the tracker naturally has available:

- a cluster consists of a set of tracks,
- each track contributes a set of active leaf options,
- each leaf has a local score,
- each leaf has a set of full-history conflict keys,
- each leaf has a set of current-scan used detections,
- and the solver must return the K best feasible selections of one leaf per track.

The contract should be solver-agnostic. It should **not** assume that the backend is exhaustive enumeration, Murty-style assignment, Lagrangian relaxation, or anything else.

### 2. Match current tracker semantics, not an idealized future problem

The first interface must cover the problem the tracker actually solves **today**, not a simplified problem invented for a future solver.

In particular, the current rebuild semantics include:

- one selected leaf per track,
- full-history exclusivity via overlapping detection keys,
- leaf-local accumulated scores,
- and an explicit per-combination cluster-local unused-detection score term for current-scan detections.

That means the interface should represent both:

- the exact conflict structure used for feasibility,
- and enough information for the current global score to be reconstructed faithfully.

### 3. Keep approximation paths explicit and separate

The current code also includes:

- overload cluster splitting for oversized exact clusters,
- and narrow historical-conflict relaxation when exact feasibility fails after approximation-induced overlap.

These should be made explicit as **policies around the solver**, not silently folded into the core exact-solver contract.

The exact solver contract should represent the exact cluster K-best problem. Approximate decomposition and relaxed-feasibility retry should remain explicit wrappers or policies around that contract.

This is one of the main reasons this phase is valuable: it forces the code to say clearly what is “the problem” and what is “a current tractability workaround.”

---

## Intended new structure

### Separate module

Create a dedicated solver-facing module, for example:

- `mht/tomht_cluster_solver.py`

or equivalent.

The exact file name is less important than the separation of responsibility.

This module should own:

- solver-facing dataclasses / protocol / abstract base,
- the current exhaustive backend implementation,
- and any solver-local helpers needed by that backend.

This module should **not** own:

- cluster construction from trees,
- N-scan pruning,
- birth handling,
- output reconstruction,
- or broader tracker lifecycle logic.

### Tracker responsibility after the split

The tracker should remain responsible for:

- building clusters from active tree frontiers,
- preparing solver input from current tree/leaf state,
- applying overload splitting before solve when enabled,
- optionally applying historical-relaxation retry around solve when enabled,
- mapping returned solver results back to current leaf nodes / rebuilt globals,
- post-solve supported-leaf pruning,
- and MAP/global snapshot construction.

In other words:

- tracker side prepares the problem,
- solver module solves the exact problem,
- tracker side interprets the result and applies current policies around it.

---

## Problem contract to make explicit

The new solver interface should make explicit, at minimum, the following assumptions:

### Feasibility

A feasible cluster solution:

- selects exactly one leaf per track,
- and no two selected leaves may overlap in full-history detection keys.

This must remain explicit because current clustering and rebuild feasibility are both based on full-history overlap, not current-scan-only overlap.

### Objective

The current exact cluster objective is not just the sum of per-leaf scores.

It also includes the explicit cluster-local unused-detection term derived from current-scan detection usage. The contract should therefore either:

- include enough information for the solver to compute this exactly via an
  explicit structured objective term.

The main requirement is that the current exhaustive backend can be migrated behind the new interface **without changing scoring semantics**.

### K-best semantics

The solver must:

- return up to `k` feasible solutions,
- in descending score order,
- and preserve deterministic tie behavior as far as the current tracker already intends to do so.

The existing exhaustive/top-K path already has explicit tie handling in parts of the code, so the new contract should not ignore this.

### Result shape

The result should remain decoupled from tree internals:

- return selected leaf IDs or equivalent stable solver-facing identifiers,
- not direct `TrackHypothesisNode` objects.

The tracker remains responsible for mapping those IDs back to nodes.

---

## What should happen in this subphase

### 1. Define the solver-facing datamodel

Create clear solver-facing structures that represent:

- one track’s candidate leaf options,
- one leaf’s score and conflict information,
- the full cluster solve problem,
- and one solved global hypothesis.

Naming does not need to be final, but the distinction should be clear.

Important modeling point:

- the interface should represent both **full conflict keys** and **current-scan used detections** if both are needed to match the current objective exactly.

### 2. Define the solver protocol / abstract interface

Define a solver interface that:

- takes the natural cluster problem representation,
- returns up to K solved global hypotheses,
- is backend-agnostic,
- and does not assume a specific internal reformulation.

This interface should be narrow enough to be stable through the next few backend experiments.

### 3. Move the current exhaustive solver behind that interface

Reimplement the current exhaustive enumeration backend as one solver implementation using the new contract.

This migration should aim for:

- no intended behavior change,
- no intended semantics change,
- and unchanged current replay behavior aside from any purely internal refactoring effects.

### 4. Adapt the tracker to use the new solver contract

Make `_rebuild_cluster_globals(...)` and related helpers call the new solver via the new interface.

The tracker should explicitly prepare:

- leaf options,
- conflict information,
- current-scan detection usage information,
- and any exact unused-detection scoring context needed by the current objective.

### 5. Make overload splitting / historical relaxation explicit around the interface

Refactor the current policy structure so that it is clear that:

- overload splitting is a current pre-solve approximation policy,
- the exact solver contract itself still represents the unsplit exact subproblem,
- historical-relaxation retry is a current around-solver fallback policy,
- and these are not silently baked into the core exact problem definition.

This is a key conceptual outcome of the subphase.

### 6. Update docs

Update the relevant docs so they describe:

- the new solver-facing interface,
- the fact that the current exhaustive solver now lives behind it,
- and the distinction between exact solve semantics and approximation wrappers.

---

## What should **not** happen in this subphase

This phase should **not** yet:

- replace exhaustive enumeration with a new backend,
- commit to a Murty/Hungarian-style internal reformulation,
- redesign scoring semantics,
- redesign birth semantics,
- change N-scan pruning policy,
- or broaden into a general cluster-code cleanup pass unrelated to the solver boundary.

The point here is to make the next runtime work possible and explicit, not to do it all at once.

---

## Acceptance criteria

This subphase should be considered complete when:

- there is a dedicated solver-facing module,
- there is a clear solver protocol / contract,
- the current exhaustive backend implements that contract,
- the tracker uses the new interface for cluster solving,
- overload splitting and historical-relaxation retry are clearly represented as policies around the exact solver problem,
- replay behavior remains materially consistent with the current baseline,
- and the docs clearly describe the new seam.

A strong secondary success criterion is that someone reading the code can now answer, in one place:

> what exact optimization problem does the tracker currently solve per cluster, and what current approximation paths sit around that problem?

---

## Follow-on work expected after this subphase

Once this seam is in place, the next likely steps become much clearer. Plausible follow-on directions include:

- a new exact or approximate K-best backend,
- profiling-guided reduction of cluster-growth pressure before solve,
- principled treatment of overload splitting and historical relaxation,
- and improved organization of the cluster build/rebuild code as part of that deeper work.

The current rebuild step remains exhaustive and is still the main runtime bottleneck on heavy merged clusters, which is why this seam is worth making explicit first.

---

## Recommended implementation style

This subphase should follow the usual conservative approach:

1. define the contract,
2. adapt the current exhaustive solver behind it,
3. validate behavior,
4. then use that seam for later solver experimentation.

That is the intended scope of this phase.
