# TO-MHT Roadmap

This roadmap is forward-looking, but it starts from the tracker's current real baseline after the track-oriented transition, exact-solver work, local-association ownership work, scoring/lifecycle/publication cleanup, DPM/API additions, and module-extraction cleanup.

The core question is no longer "how do we make this a true TO-MHT?" The tracker is now a practical track-oriented implementation. The more relevant question is:

> given the current persistent-tree architecture, exact solver seam, coherent scoring/lifecycle semantics, and replay bottlenecks, what should be improved next?

This roadmap is a **priority map and topic grouping document**, not a fixed execution order.

---

## 1. Current baseline

The tracker now has:

- explicit persistent `TrackTree` / `TrackHypothesisNode` state,
- `TrackTreeStore` owning persistent tree/node/ID bookkeeping,
- per-scan rebuilt globals from current leaf frontiers,
- full-history `(scan_index, det_index)` exclusivity semantics,
- post-solve supported-leaf pruning,
- MAP-only N-scan pruning directly on explicit trees,
- committed-prefix output history restoration,
- exact-one-of `predictor` or `hypothesiser` constructor semantics,
- a narrow distance-hypothesiser seam for local association,
- NLL/LLR-based scoring via `NLLScoringModel`,
- `DetectionProbabilityModel` for dynamic `P_D` and clutter density,
- explicit external/internal start lanes with existence priors,
- metadata support for `existence_log_odds` and `existence_probability`,
- sticky tree confirmation,
- score-based whole-track deletion,
- sticky output publication and dense publication-time public IDs,
- an explicit solver-facing exact cluster-solver contract,
- `branch_and_bound` as the default exact cluster backend,
- `exhaustive` retained as exact reference/fallback,
- `ortools` retained as experimental exact backend,
- per-scan timing-phase instrumentation,
- smoke/replay output and timing baselines,
- a dedicated API/integration guide.

The tracker is no longer blocked on:

- the structural track-oriented transition,
- the original exhaustive solver bottleneck,
- the PDA/beta-oriented local-association path,
- basic score/lifecycle semantics,
- or the monolithic tracker class shape.

The main runtime bottleneck is now best interpreted as **local expansion volume**.

---

## 2. Guiding principles

### 2.1 Preserve the clear runtime story

Future work should preserve the current understandable structure:

- trees/leaves are persistent state,
- globals are rebuilt per scan,
- exact solving lives behind a solver seam,
- scoring is additive and interpretable,
- lifecycle/publication are tree-level concepts,
- output tracks are public views of internal MAP-selected trees,
- the tracker class orchestrates modules rather than owning all mechanics.

### 2.2 Prefer conservative, evidence-driven changes

The preferred style remains:

1. characterize the issue,
2. add observability if needed,
3. make a targeted implementation pass,
4. compare smoke/replay output and timing,
5. decide the next step from evidence.

This matters especially for expansion-volume work, where pruning can easily change tracking quality.

### 2.3 Keep semantics separate

Avoid mixing concepts that are now cleanly separated:

- confirmation is internal lifecycle state,
- deletion removes a tree,
- publication controls output visibility,
- DPM calibrates local evidence,
- initiators own state initialization and candidate validity,
- solver exactness is per rebuilt cluster/subcluster, while overload splitting is an explicit guardrail.

### 2.4 Treat docs as part of the handoff

`CURRENT_STATE`, `NEXT_STEPS`, `ROADMAP`, and `TO_MHT_API.md` should continue to track the implementation. The API guide is especially important for ISAC-style integration discussions.

---

## 3. Main topic groups

### A. Local expansion volume reduction / pre-expansion control

This is now the clearest immediate performance topic.

The solver phase and local-association math cleanup moved the bottleneck. Timing instrumentation points to local expansion / hypothesis generation, especially the number of active leaves that require expansion.

This topic includes:

- profiling how many leaves are expanded per scan,
- identifying which expanded leaves later matter,
- measuring expansion cost by tree/lifecycle/publication state,
- reducing the number of leaves that require full hypothesiser work,
- revisiting local child generation/retention controls,
- considering score/frontier-aware pre-expansion filters,
- clarifying semantic pruning vs tractability caps,
- reviewing overload-split pruning behavior when clusters grow large.

This is the strongest candidate for the next major technical branch.

### B. Frontier / score-based pruning

Now that score semantics are coherent, broader pruning can be considered more safely.

This topic includes:

- score-based leaf/frontier pruning,
- lifecycle-aware expansion budgets,
- whether confirmed and tentative trees deserve different resource treatment,
- how deletion thresholds, publication state, and active frontier pruning should interact,
- avoiding premature loss of hypotheses that could reconnect through later measurements.

This belongs close to expansion-volume work, but should be introduced carefully.

### C. Internal birth / existence / candidate quality

Internal births are cleaner than before but still heuristic in their tractability controls.

This topic includes:

- candidate ranking/capping behavior,
- whether `max_births_per_scan` is firing routinely,
- whether candidate confidence metadata should have a stronger role,
- residual detection policy,
- false-start observability,
- interplay between start priors, confirmation, publication, and deletion,
- target swapping / track jumping review.

For external-start-only integrations this is less urgent, but it remains important for the general tracker path.

### D. Approximation semantics and guardrails

Current pragmatic mechanisms include:

- overload cluster splitting,
- local leaf caps,
- internal birth load guards.

These are explicit and useful but not conceptually final.

This topic includes:

- documenting/validating when each approximation is acceptable,
- whether overload-split clusters should participate in supported-leaf pruning differently,
- how guardrails should interact with future score/frontier pruning.

### E. Integration / API / operational hardening

This is ongoing supporting work:

- gather ISAC/API feedback,
- preserve Stone Soup compatibility,
- keep parameter override workflows practical,
- maintain smoke/replay baselines,
- keep backend parity tests healthy,
- improve docs as implementation stabilizes,
- ensure external-start and DPM integration remain ergonomic.

### F. Parallelization / orchestration

Parallel local expansion is a real future axis, but not the next default move.

When addressed, it should be:

- opt-in,
- deterministic,
- separated behind an orchestration abstraction,
- able to support sequential, tracker-owned parallel, and external/custom orchestration modes,
- compatible with ISAC needs without depending on ISAC internals.

---

## 4. Rough priority picture

### Highest-leverage near-term area

1. **Local expansion volume reduction / pre-expansion control**

This should be the next primary phase unless new integration feedback changes the priority.

### Important near-term companions

These should be considered during the expansion-volume phase, but do not necessarily define separate phases:

- score/frontier pruning,
- lifecycle-aware expansion policies,
- expansion observability,
- approximation semantics around overload and pruning,
- birth ranking/capping observations if they affect expansion pressure.

### Important but probably not first

- internal birth/existence quality review,
- target swapping / track-jumping audit,
- broader output-quality tuning,
- more detailed parameter tuning guidance.

### Later / lower priority

- local-expansion parallelization,
- richer external-start scheduling before first update,
- deeper solver backend experimentation,
- extensive packaging/handoff polish beyond near-term API docs,
- large-scale refactors not tied to a substantive technical branch.

---

## 5. Recommended next branch: expansion-volume-first

### Focus

Characterize and reduce expansion volume:

- how many leaves are expanded,
- which trees/leaves dominate expensive scans,
- which expanded leaves survive or influence retained globals,
- how expansion volume relates to lifecycle/publication state,
- which low-score or low-quality frontiers can be safely deprioritized or pruned,
- what controls are semantic vs emergency caps.

### Why this is attractive

- Exact cluster solving is no longer the main bottleneck.
- Local-association kernel optimizations have already delivered meaningful wins.
- Remaining replay tails appear driven by expansion volume.
- The scoring/lifecycle work now gives us interpretable scores to use for pruning or selective expansion.
- The tracker is now modular enough that expansion work can be targeted.

### What to avoid initially

Do not start with broad, aggressive pruning. First characterize:

- active leaf counts,
- expansion attempts,
- expansion result usefulness,
- retained-vs-pruned leaves,
- lifecycle/publication distribution of expensive expansions,
- output quality changes under candidate policies.

---

## 6. Secondary branch: quality / birth / existence review

This remains high-value, especially for general tracker behavior.

Focus:

- false starts,
- internal birth candidate cap behavior,
- start priors and metadata confidence,
- score deletion and publication thresholds,
- target swapping / track jumping,
- smoke/replay quality inspection.

Why it may wait:

- current outputs are usable enough for continued runtime work,
- the external-start-only ISAC path is less dependent on internal birth behavior,
- the most visible runtime bottleneck is still expansion volume.

This branch may follow expansion-volume work or be interleaved if expansion analysis shows births are a major volume driver.

---

## 7. What belongs together

### Expansion-volume work should include

- per-scan and per-tree expansion volume characterization,
- leaf usefulness analysis,
- lifecycle/publication-aware expansion statistics,
- pre-expansion pruning or prioritization experiments,
- review of local child-retention controls,
- validation against smoke/replay baselines.

### Score/frontier pruning work should include

- clear interpretation of score thresholds,
- relationship to confirmation/deletion/publication,
- impact on N-scan pruning and post-solve supported-leaf pruning,
- safeguards against losing reconnectable hypotheses.

### Birth/existence work should include

- initiator candidate observability,
- cap firing frequency,
- metadata confidence use,
- residual policy review,
- false-start and target-jump inspection.

### Approximation work should include

- overload split semantics,
- supported-leaf pruning behavior for split clusters,
- interaction with future score/frontier pruning.

### Parallelization work should include

- opt-in orchestration design,
- deterministic result merging,
- clear ownership boundary between tracker and custom/external parallel execution,
- timing and reproducibility validation.

---

## 8. Practical notes from the current checkpoint

### 8.1 Solver phase succeeded

The solver work produced:

- a clean solver seam,
- exhaustive reference extraction,
- exploratory OR-Tools backend,
- branch-and-bound backend strong enough to become default,
- diagnostics for branch-and-bound search behavior.

Exact cluster solving is not the urgent blocker it was earlier.

### 8.2 Local-association/scoring phase succeeded

The local-association and scoring work produced:

- tracker-owned default NLL distance hypothesiser,
- explicit NLL/LLR scoring,
- removed PDA/beta/unused-detection scoring path,
- DPM abstraction for dynamic `P_D` and clutter density,
- documented scoring unit contract.

The current baseline is good enough to support expansion-volume work.

### 8.3 Lifecycle/publication phase succeeded

The tracker now has:

- sticky confirmation,
- score-based deletion,
- sticky publication,
- confirmed-only default output,
- public vs internal ID split,
- existence probability/log-odds metadata.

This gives future score/frontier controls a meaningful interpretation.

### 8.4 API guide is ready for review

`TO_MHT_API.md` is good enough to share with integrators. Feedback may lead to small API/doc adjustments, but it should not block the next internal technical phase unless it reveals a real integration issue.

---

## 9. Near-term execution style

The immediate workflow should be:

1. replace `NEXT_STEPS` with a focused expansion-volume phase plan,
2. start with analysis/instrumentation rather than pruning changes,
3. identify candidate volume controls,
4. implement one conservative control at a time,
5. compare smoke/replay output and timing,
6. decide whether to continue volume work or switch to quality/birth review.

---

## 10. Near-term priority summary

At this checkpoint:

1. The track-oriented architecture is stable enough to treat as baseline.
2. Branch-and-bound solved the immediate exact-solver bottleneck.
3. NLL/DPM scoring and score-based lifecycle are coherent enough for pruning work.
4. API and integration assumptions are now documented.
5. The tracker is modular enough for targeted subsystem work.
6. The next main technical phase should return to **local expansion volume / frontier control**.
