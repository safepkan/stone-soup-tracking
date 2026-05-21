# TO-MHT Roadmap

This roadmap is forward-looking, but it starts from the current tracker baseline
after the track-oriented transition, exact-solver work, local-association
ownership, NLL/DPM scoring, lifecycle/publication cleanup, overload-split
soundness work, expansion/frontier API cleanup, object-boundary cleanup,
smoke-runner reset, and module extraction.

The core question is no longer "how do we make this a true TO-MHT?" The tracker
is now a practical track-oriented implementation. The more relevant question is:

> given the current persistent-tree architecture, coherent scoring/lifecycle
> semantics, sound overload handling, and integration API, which practical
> runtime, quality, and integration topics should be improved next?

This roadmap is a **priority map and topic grouping document**, not a fixed
execution order.

---

## 1. Current baseline

The tracker now has:

- explicit persistent `TrackTree` / `TrackHypothesisNode` state,
- `TrackTreeStore` owning persistent tree/node/ID bookkeeping,
- per-scan rebuilt globals from current leaf frontiers,
- live unresolved `(scan_index, det_index)` exclusivity semantics, with
  committed history removed from active conflict checks,
- post-solve supported-leaf pruning for every retained feasible original-cluster
  snapshot,
- MAP-only N-scan pruning directly on explicit trees,
- committed-prefix output history restoration,
- exact-one-of `predictor` or `hypothesiser` constructor semantics,
- a narrow distance-hypothesiser seam for local association,
- default internal fast paths that avoid Stone Soup `Track` reconstruction when
  history is not needed, gated by `TOMHTParams` flags,
- NLL/LLR-based scoring via `NLLScoringModel`,
- `DetectionProbabilityModel` for dynamic `P_D` and clutter density,
- explicit external/internal start lanes with existence priors,
- metadata support for `existence_log_odds` and `existence_probability`,
- sticky tree confirmation,
- score-based whole-tree deletion,
- one configured non-score deleter path, with an internal miss-count deleter as
  the default and custom Stone Soup deleters replacing that default,
- sticky output publication and dense publication-time public IDs,
- an explicit solver-facing exact cluster-solver contract,
- `branch_and_bound` as the default exact cluster backend,
- `exhaustive` retained as exact reference/fallback,
- `ortools` retained as experimental exact backend,
- sound overload splitting internal to one original-cluster solve, with
  `greedy_partition` as the default operational mode and `conditional_exact` as
  a reference / higher-compute mode,
- per-scan timing-phase instrumentation, explicit reconstruction counters, and
  expansion/frontier usefulness counters,
- smoke/replay output and timing baselines,
- a dedicated API/integration guide.

The tracker is no longer blocked on:

- the structural track-oriented transition,
- the original exhaustive solver bottleneck,
- the PDA/beta-oriented local-association path,
- basic score/lifecycle semantics,
- overload-split soundness,
- or the monolithic tracker class shape.

The accepted object-boundary cleanup removed avoidable default-path Stone Soup
`Track` reconstruction: standard replay now reports zero expansion/deleter
reconstruction calls on the default paths. The known remaining performance
attention is best interpreted as **local expansion and scenario-quality work**.
Standard replay still spends most time in expansion/hypothesiser work, while the
current replay/smoke workloads are no longer blocked by object-boundary
reconstruction, frontier growth, or exact cluster solving.

---

## 2. Guiding principles

### 2.1 Preserve the clear runtime story

Future work should preserve the current understandable structure:

- trees/leaves are persistent state,
- globals are rebuilt per scan,
- exact solving lives behind a solver seam,
- overload splitting is an explicit sound approximation/fallback inside one
  original cluster solve,
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

This matters especially for pruning/frontier work, where apparently safe local
rules can change tracking quality.

### 2.3 Keep normal operation distinct from guardrails

The normal frontier-control stack is now coherent:

1. local association gating,
2. per-active-leaf local branching through `max_children_per_leaf`, with the
   miss alternative preserved,
3. K-best feasible cluster solve,
4. post-solve supported-leaf pruning,
5. MAP-only N-scan pruning,
6. score deletion plus the configured deleter,
7. sticky publication gating.

Guardrails and approximation/safety mechanisms are separate:

- optional pre-solve per-tree leaf cap,
- internal-birth cap and optional birth load guards,
- overload split fallback,
- optional hard projected-cluster-combination cap.

Future work should avoid disguising emergency caps as semantic pruning.

### 2.4 Keep semantics separate

Avoid mixing concepts that are now cleanly separated:

- confirmation is internal lifecycle state,
- deletion removes a tree,
- publication controls output visibility,
- DPM calibrates local evidence,
- initiators own state initialization and candidate validity,
- custom hypothesisers own sensor-specific local association and gating,
- solver exactness is per rebuilt cluster, while overload splitting is an
  explicit sound approximation/fallback.

### 2.5 Treat docs as part of the handoff

`CURRENT_STATE`, `NEXT_STEPS`, `ROADMAP`, and `TO_MHT_API.md` should continue to
track the implementation. The API guide is especially important for ISAC-style
integration discussions.

---

## 3. Main topic groups

### A. Local expansion and default-hypothesiser profiling

This remains the clearest performance topic, but the near-term framing has
changed: current scenarios are not blocked by frontier growth, while profiling
still shows local expansion/hypothesiser work as the dominant runtime
component. The narrow object-boundary cleanup is complete for the current
default internal paths.

This topic includes:

- profiling how many leaves are expanded per scan,
- identifying which expanded leaves later matter,
- measuring expansion cost by tree/lifecycle/publication state,
- reducing the number of leaves that require full hypothesiser work when there
  is evidence this is safe,
- deeper default-hypothesiser math/kernel profiling,
- validating whether expansion volume rather than inner-kernel cost is the next
  practical limiter,
- preserving full Stone Soup `Track` reconstruction for public output and
  debug/inspection,
- preserving full Stone Soup `Track` reconstruction for custom hypothesisers
  and custom deleters.

### B. Frontier / score-based pruning

Now that score semantics are coherent, broader score/frontier pruning can be
considered more safely, but should still start as diagnostics.

This topic includes:

- score-relative leaf/frontier diagnostics,
- lifecycle-aware expansion budgets,
- whether tentative and confirmed trees deserve different resource treatment,
- whether low-score leaves far below the best leaf in a tree ever survive into
  retained top-K/MAP hypotheses,
- how deletion thresholds, publication state, and active frontier pruning should
  interact,
- safeguards against losing hypotheses that could reconnect through later
  measurements.

### C. Overload split policy and difficulty signals

The soundness problem is addressed. Both overload modes now return feasible
original-cluster globals, and supported-leaf pruning applies uniformly.

Future overload work is about policy and quality rather than restoring
correctness:

- improve the decision of when a cluster solve is actually too hard,
- replace or supplement projected Cartesian product with better difficulty
  signals,
- consider conflict graph density, cut-key structure, score concentration, and
  branch-and-bound search behavior,
- compare `greedy_partition` and `conditional_exact` on quality-sensitive or
  ID-switch-heavy scans,
- refine greedy ownership if broader scenarios expose quality issues.

### D. Internal birth / existence / candidate quality

Internal births are cleaner than before but still heuristic in their
tractability controls.

This topic includes:

- candidate ranking/capping behavior,
- whether `max_births_per_scan` is firing routinely,
- whether scenario-specific birth load guards are needed, since they default to
  disabled,
- whether candidate confidence metadata should have a stronger role,
- residual detection policy,
- false-start observability,
- interplay between start priors, confirmation, publication, and deletion,
- target swapping / track jumping review.

For external-start-only integrations this is less urgent, but it remains
important for the general tracker path and for running MHT on recorded data with
internal initiators.

### E. Output continuity and tracking quality

The tracker is now stable enough to support a more meaningful quality review.
Observed output issues such as occasional ID switching or MAP-leaf switching
should be treated as a separate output-continuity topic rather than mixed into
frontier-control work.

This topic includes:

- ID switching / target-jump analysis,
- publication-side continuity policies,
- possible stitching across track-tree fragments,
- smoothing or hysteresis for MAP-leaf changes within a published tree,
- false-start and false-publication review,
- broader scenario visualization and metrics.

### F. Integration / API / operational hardening

This is ongoing supporting work:

- gather ISAC/API feedback,
- preserve Stone Soup compatibility,
- keep Python 3.10 compatibility healthy,
- keep parameter override workflows practical,
- maintain smoke/replay baselines,
- keep backend parity tests healthy,
- improve docs as implementation stabilizes,
- ensure external-start and DPM integration remain ergonomic.

### G. Parallelization / orchestration

Parallel local expansion is a real future axis, but not the next default move.

When addressed, it should be:

- opt-in,
- deterministic,
- separated behind an orchestration abstraction,
- able to support sequential, tracker-owned parallel, and external/custom
  orchestration modes,
- compatible with ISAC needs without depending on ISAC internals.

---

## 4. Rough priority picture

### Highest-leverage near-term candidates

The next deeper branch should be chosen from integration feedback and broader
scenario runs. The strongest current candidates are:

1. **Deeper default-hypothesiser math/kernel profiling**
2. **Expansion volume / which leaves matter**
3. **Broader scenario quality validation**
4. **Output continuity / ID-switching review**
5. **Internal birth / existence quality review** for non-ISAC/general tracker use

### Important supporting areas

These should travel with the deeper work rather than define standalone phases:

- score/frontier diagnostics,
- overload split difficulty-signal review,
- additional scenario validation,
- smoke/replay baseline maintenance,
- small API/doc clarifications from integrator feedback.

### Later / lower priority

These still matter, but do not currently define the main next move:

- local-expansion parallelization,
- richer external-start scheduling before first update,
- deeper solver backend experimentation,
- broad packaging/handoff polish beyond near-term API docs,
- large-scale refactors not tied to a substantive technical branch.

---

## 5. Recommended next branch candidates

### Option 1: Profiling-guided local expansion cleanup

Focus:

- preserve current frontier semantics,
- add/inspect finer profiling inside the tracker-owned default hypothesiser,
- distinguish inner math/kernel cost from expansion volume,
- inspect which expanded leaves later matter,
- re-profile standard replay and at least one additional scenario.

Why this is attractive:

- hypothesiser calls remain the dominant cost,
- object-boundary reconstruction is no longer the primary default-path concern,
- this can improve runtime without introducing new pruning semantics.

What to avoid initially:

- changing custom component behavior by default,
- exposing lightweight-track API knobs before internal profiling proves the
  benefit,
- replacing public output reconstruction with anything other than full tracks.

### Option 2: Broader scenario validation and quality review

Focus:

- run more recorded/synthetic scenarios,
- compare smoke/replay/ISAC-style behavior,
- inspect ID switching, target jumping, false starts, and false publications,
- decide whether output continuity or birth quality is the next real pain point.

Why this is attractive:

- current standard replay is no longer sufficient to infer the next major
  blocker,
- smoke scenarios now run well with nominal parameters,
- ISAC feedback may change priorities.

### Option 3: Overload difficulty-signal review

Focus:

- leave `greedy_partition` as the default sound operational fallback,
- evaluate when exact branch-and-bound can solve large projected clusters
  cheaply,
- design a better split trigger than projected Cartesian combinations alone,
- preserve the current soundness invariant that downstream sees feasible
  original-cluster globals only.

Why it may wait:

- the current overload path is practical on standard replay,
- the low threshold still exercises the code path,
- no current scenario shows overload as an active blocker.

### Option 4: Birth/existence-focused branch

Focus:

- internal initiator quality,
- candidate confidence metadata,
- birth cap firing frequency,
- false-start and publication interaction,
- recorded-data runs that require internal starts.

Why it may wait:

- ISAC integration is external-start-only,
- recent defaults are now less restrictive,
- broader scenario evidence should drive the next birth-specific changes.

---

## 6. What belongs together

### Local expansion profiling work should include

- profiling and characterization of expensive expansion scans,
- deeper default-hypothesiser math/kernel timing,
- expansion-volume analysis and which leaves later matter,
- validation that custom hypothesiser behavior remains unchanged,
- smoke/replay timing and output comparison.

### Score/frontier pruning work should include

- clear interpretation of score thresholds,
- relationship to confirmation/deletion/publication,
- impact on N-scan pruning and post-solve supported-leaf pruning,
- diagnostics before pruning,
- safeguards against losing reconnectable hypotheses.

### Birth/existence work should include

- initiator candidate observability,
- cap firing frequency,
- metadata confidence use,
- residual policy review,
- false-start and target-jump inspection.

### Overload/approximation work should include

- split trigger and weak-link policy review,
- comparison of exact B&B vs greedy split cost,
- comparison of `greedy_partition` and `conditional_exact` quality,
- preservation of the feasible-original-cluster invariant.

### Output-continuity work should include

- ID switch metrics and visualization,
- MAP-leaf switch analysis,
- publication-side smoothing/hysteresis options,
- clear separation from internal scoring and pruning semantics.

### Parallelization work should include

- opt-in runtime behavior only,
- a clean orchestration abstraction above hypothesiser execution,
- room for sequential, tracker-owned parallel, and external/custom parallel
  modes,
- strong determinism/validation expectations.

---

## 7. Practical notes from the current checkpoint

### 7.1 Solver phase succeeded

The solver work produced:

- a clean solver seam,
- exhaustive reference extraction,
- exploratory OR-Tools backend,
- branch-and-bound backend strong enough to become default,
- diagnostics for branch-and-bound search behavior.

Exact cluster solving is not the urgent blocker it was earlier.

### 7.2 Local-association/scoring phase succeeded

The local-association and scoring work produced:

- tracker-owned default NLL distance hypothesiser,
- explicit NLL/LLR scoring,
- removed PDA/beta/unused-detection scoring path,
- DPM abstraction for dynamic `P_D` and clutter density,
- documented scoring unit contract.

The current baseline is good enough to support future pruning and integration
work.

### 7.3 Lifecycle/publication phase succeeded

The tracker now has:

- sticky confirmation,
- score-based deletion,
- one configured non-score deleter path,
- sticky publication,
- confirmed-only default output,
- public vs internal ID split,
- existence probability/log-odds metadata.

This gives future score/frontier controls a meaningful interpretation.

### 7.4 Overload/frontier phase removed the current blocker

The overload/frontier work produced:

- live conflict keys that exclude committed history,
- removal of historical relaxation,
- sound overload solving with feasible original-cluster globals downstream,
- uniform supported-leaf pruning for overload-solved clusters,
- `greedy_partition` as a practical default,
- `conditional_exact` as a reference/higher-compute mode,
- focused overload modules and split policy seams.

This does not mean expansion is solved universally, but the standard replay and
smoke workloads no longer show the previous frontier-control blocker.

### 7.5 Smoke scenario reset improved the baseline demonstration

The smoke runners now use the scenario's nominal `prob_detect` and
`clutter_density` instead of old hand-tuned TOMHT overrides. Visual inspection
showed better output quality with the current scoring/lifecycle/overload stack,
so the smoke scenarios now better represent the current tracker rather than early
tuning artifacts.

### 7.6 Profiling refined the next performance question

Profiling confirms that hypothesiser calls dominate standard replay. It also
shows non-negligible overhead from creating full Stone Soup `Track` objects and
accessing Stone Soup attributes. Future work should determine how much of this
can be avoided in internal/default paths without changing the public Stone Soup
component boundary.

---

## 8. Near-term execution style

The immediate workflow should likely be:

1. use ISAC feedback and broader scenario runs to choose the next real branch,
2. keep smoke/replay baselines healthy,
3. use profiling before adding new pruning rules,
4. prefer internal/default-path optimizations before exposing new API knobs,
5. implement one conservative change at a time,
6. compare output quality and timing before continuing.

This avoids prematurely mixing expansion optimization, output continuity, birth
quality, and parallelization design into one branch.

---

## 9. Near-term priority summary

At this checkpoint:

1. The track-oriented architecture is stable enough to treat as baseline.
2. Branch-and-bound solved the immediate exact-solver bottleneck.
3. NLL/DPM scoring and score-based lifecycle are coherent enough for future
   score/frontier work.
4. Overload splitting is now sound and practical; the previous frontier-control
   blocker is not active on current replay/smoke workloads.
5. Local expansion remains the main profiling hotspot; default-path Stone Soup
   object-boundary reconstruction has been addressed.
6. Internal birth/existence quality and output continuity remain important but
   should be driven by broader scenario evidence.
7. Parallelization should stay a later, explicit, opt-in architectural topic.

In other words: the tracker is in a good place for integration feedback and
broader scenario validation. The next technical branch should be selected from
that evidence, with deeper default-hypothesiser profiling, expansion-volume
analysis, broader scenario validation, output continuity, overload difficulty
policy, and internal birth quality as the main candidates.
