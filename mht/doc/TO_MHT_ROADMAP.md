# TO-MHT Roadmap

This roadmap is intentionally forward-looking, but it starts from the tracker’s **current real baseline** rather than from the pre-transition plans.

The structural track-oriented transition is complete enough that the main question is no longer “how do we make this a true TO-MHT?” The more relevant question is now:

> given the current track-oriented baseline, exact-solver seam, and current replay bottlenecks, what should be improved next, what belongs together, and which topics are highest leverage?

This roadmap is therefore best read as a **priority map and topic grouping document**, not as a fixed execution order.

---

## 1. Current baseline

The tracker now has:

- explicit persistent `TrackTree` / `TrackHypothesisNode` state,
- per-scan rebuilt globals from current leaf frontiers,
- full-history `(scan_index, det_index)` exclusivity semantics,
- post-solve supported-leaf pruning,
- MAP-only N-scan pruning directly on explicit trees,
- committed-prefix output history restoration,
- predictor/updater as the primary constructor boundary,
- an explicit solver-facing exact cluster-solver contract,
- `branch_and_bound` as the default exact cluster backend,
- `exhaustive` retained as exact reference/fallback,
- `ortools` retained as experimental exact backend,
- and per-scan timing-phase instrumentation showing where replay time is now going.

This means the tracker is no longer blocked on the core architectural transition or on the original exact-solver bottleneck that motivated the recent scalability phase.

---

## 2. Guiding principles for the next phase

### 2.1 Preserve the clear runtime story

The strongest gain from the transition is that the runtime story is now understandable:

- trees/leaves are the persistent state,
- globals are rebuilt per scan,
- MAP-only N-scan pruning commits tree structure directly,
- output tracks are reconstructed from committed prefix plus unresolved lineage,
- and exact cluster solving now sits behind a clean dedicated seam.

Future work should preserve that clarity.

### 2.2 Prefer conservative, incremental deepening

The preferred style remains:

1. analyze / review what should change
2. make a targeted implementation pass

This is especially important now that the tracker is good enough to experiment with meaningfully.

### 2.3 Avoid cleanup-only phases when the cleanup belongs to a deeper topic

There is still internal organization work worth doing, especially around local expansion and parts of the cluster-build/rebuild flow. But that organization should preferably be treated as a **sub-goal of deeper work** such as local branching/runtime work, scoring redesign, or approximation cleanup, rather than as an isolated readability-only phase.

### 2.4 Keep docs aligned with actual code and reasoning

The docs are part of the handoff and planning workflow. As deeper work proceeds, `CURRENT_STATE`, `NEXT_STEPS`, and the roadmap should continue to evolve with the implementation.

---

## 3. Main topic groups from the current checkpoint

The next work now appears to fall into five main topic groups.

### A. Local expansion / hypothesis-generation runtime

This now looks like the clearest immediate performance bottleneck on the primary replay used during the recent phase.

The recent timing breakdown work indicates that, once branch-and-bound replaced exhaustive as the default exact backend, heavy scans became dominated primarily by local expansion / hypothesis generation rather than by exact cluster rebuild solve time.

This topic includes:

- reducing per-leaf local hypothesis generation cost,
- reducing the number of leaves that need full local expansion work,
- revisiting how local hypotheses are produced and filtered,
- potential caching/reuse opportunities,
- and profiling where Stone Soup-boundary work is dominating.

This is currently the strongest candidate for the **next major technical branch**.

### B. Local branching / scoring design

The public boundary has shifted to predictor/updater, but local branching is still internally PDA-style. The default scoring remains beta-ratio-based and should still be understood as pragmatic rather than final.

This topic includes:

- clearer decomposition of local score contributions,
- eventual tracker-owned local branching/scoring logic rather than dependence on PDA-style packaging,
- reduced coupling to hypothesiser-oriented concepts,
- better alignment between local branch scoring and rebuilt-global semantics,
- and possible prediction/gating/likelihood reuse or batching opportunities.

This topic overlaps strongly with local-expansion runtime work and may belong in the same broader next branch.

### C. Internal birth / existence / quality semantics

Internal births now work, but they remain intentionally simple and are clearly not the final design.

Recent review notes suggest that false starts likely became worse after the transition for structural reasons:

- birth candidates become real trees immediately,
- post-birth existence is effectively mandatory,
- and whole-track lifecycle kill is slower/more permissive than before.

This topic includes:

- whether internal births should remain direct tree insertion or gain a more uncertain/probationary stage,
- whether the current residual policy is too conservative or appropriately protective,
- whether miss-lifecycle policy is too permissive for low-quality birth trees,
- candidate observability and TO-MHT-native birth impact statistics,
- and general false-start tuning.

For the **external-start-only ISAC path**, this is not necessarily the immediate blocker. But for the general tracker path, it remains a fairly high-priority quality topic.

### D. Approximation semantics and principling

The tracker currently uses explicit approximation/safety-net paths:

- overload cluster splitting,
- historical-conflict relaxation,
- and simple birth load guards.

These are useful and pragmatic, but not conceptually final.

This topic includes:

- deciding which approximations are acceptable operationally,
- clarifying the meaning and limits of overload splitting,
- deciding whether approximation-induced overlap should be treated differently,
- and improving the conceptual story around when and why these mechanisms are engaged.

This work still belongs more naturally with runtime/scoring/branching work than as an isolated phase.

### E. Integration / validation / operational hardening

Some earlier roadmap items remain relevant, but they now belong in a different context.

This topic includes:

- continued local replay validation,
- keeping the tracker easy to drop into Stone Soup-style workflows,
- preserving Python / Stone Soup compatibility,
- practical runner/parameter override support,
- backend parity/regression coverage,
- and scenario/replay validation sufficient to trust deeper changes.

This is important ongoing support work, but no longer the defining next phase.

---

## 4. Rough priority picture

The current rough priority picture looks like this.

### Highest-leverage near-term areas

These currently look the most likely to shape the next deeper phase:

1. **Local expansion / hypothesis-generation runtime**
2. **Scoring / local-branching ownership**
3. **Internal birth / existence semantics**

That does **not** mean they must be tackled in that exact order. But it does mean they currently look like the most consequential topics.

### Important supporting areas

These should travel with the deeper work rather than define standalone phases:

- cluster/build/rebuild and expansion-path code organization,
- approximation semantics cleanup,
- observability improvements,
- additional regression/validation coverage,
- and small integration/hardening fixes.

### Lower-priority / later areas

These still matter, but do not currently define the main next move:

- richer external-start scheduling beyond the current insertion model,
- optional pre-first-step external starts,
- broader lifecycle/materialisation refinements beyond the current committed-prefix output fix,
- node GC / ancestry cleanup beyond the current reachable-node cleanup,
- and more extensive packaging/handoff polish.

---

## 5. Recommended interpretation of likely next branches

At the current checkpoint, several next branches are plausible.

### Option 1: Local-expansion/runtime-first branch

Focus:
- local hypothesis generation cost,
- leaf-frontier growth pressure before solve,
- caching/reuse opportunities,
- and timing-guided reduction of expensive expansion work.

Why this is attractive:
- the recent branch-and-bound + timing work indicates that exact cluster solving is no longer the dominant bottleneck on the primary replay,
- replay now works well enough that profiling is meaningful,
- and the timing instrumentation provides a concrete starting point.

This currently looks like the strongest candidate.

### Option 2: Scoring/local-association-first branch

Focus:
- tracker-owned local branching math,
- less PDA-style dependency internally,
- cleaner score decomposition,
- and possibly better runtime through more direct local score generation.

Why this is attractive:
- the public interface already moved in this direction,
- scoring remains one of the most provisional parts of the design,
- and a cleaner local-association story may improve both runtime and quality.

This is compelling, and may in practice merge with Option 1.

### Option 3: Birth/existence-first branch

Focus:
- false starts,
- birth insertion semantics,
- existence/absence alternatives,
- lifecycle kill behavior,
- and TO-MHT-native birth observability/statistics.

Why this is attractive:
- recent reviews strongly suggest that internal-birth quality got worse structurally,
- and the issue is conceptually clearer than before.

Why it may not be first:
- ISAC integration is external-start-only,
- and the issue is entangled with scoring and lifecycle policy.

This should stay high on the list even if not chosen first.

---

## 6. What belongs together

To avoid fragmented work, some topics should be grouped deliberately.

### Local-expansion/runtime work should probably include

- profiling and characterization of expensive expansion scans,
- revisiting the number of local hypotheses retained/generated,
- any local caching/reuse opportunities,
- scoring/local-branching simplification where it materially reduces expansion work,
- and organization of the expansion-related tracker code as part of that work.

### Scoring work should probably include

- local branching ownership,
- raw local-score semantics,
- hypothesis-generator de-emphasis,
- and any resulting changes to clutter / birth / miss interpretation.

### Birth/existence work should probably include

- birth insertion semantics,
- residual policy review,
- lifecycle kill policy review,
- initiator interaction,
- candidate observability,
- and TO-MHT-native birth impact metrics.

### Approximation work should probably include

- overload split / historical relaxation review,
- whether current exact/approximate boundaries are still the right ones,
- and how those mechanisms should interact with future runtime or scoring changes.

---

## 7. Practical notes from the current checkpoint

A few concrete observations should shape later choices.

### 7.1 The solver phase was successful enough to change the default

The recent runtime/scalability phase produced:

- a clean solver seam,
- parity-preserving exhaustive extraction,
- a rejected Murty direction,
- an exploratory OR-Tools backend,
- and a branch-and-bound backend that was strong enough to become the default.

That means the exact cluster-solver topic is no longer the same kind of urgent blocker it was at the start of the phase.

### 7.2 OR-Tools should be kept, but not over-weighted

The experimental OR-Tools backend was useful and should remain available for comparison and future hybrid/K-best experiments, but the current repeated-solve CP-SAT path is not the main next runtime answer.

### 7.3 Timing instrumentation changed the picture

The new timing breakdown was important because it revealed that, once the exact solver improved, the dominant replay bottleneck moved to local expansion. That should directly shape what comes next.

### 7.4 Code organization is real, but should still be attached to real work

There is still tracker-internal organization work worth doing, but most of it should probably travel with the next substantive branch rather than define a standalone cleanup phase.

---

## 8. Near-term execution style

The immediate post-phase workflow should likely be:

1. refresh `CURRENT_STATE` and roadmap to match the new baseline,
2. leave `NEXT_STEPS` as historical context for the just-completed solver phase,
3. decide deliberately which next branch to take,
4. only then draft a fresh `NEXT_STEPS` for that branch,
5. then do a targeted implementation pass.

That avoids prematurely committing to the wrong next branch before the recent timing data is absorbed properly.

---

## 9. Near-term priority summary

At the current checkpoint, the best summary is:

1. The track-oriented architectural transition is complete enough to treat as the stable baseline.
2. The recent runtime/scalability phase succeeded in improving the exact cluster solver enough to make branch-and-bound the default backend.
3. The main replay bottleneck is now local expansion / hypothesis generation rather than exact cluster solving.
4. Scoring/local-branching design and internal birth/existence semantics remain the two other major unresolved design areas.
5. Approximation semantics, validation, observability, and code organization should continue to advance together with those deeper topics rather than as isolated phases.
6. The next concrete implementation phase should be chosen after a deliberate look at the refreshed roadmap and the new timing evidence.

In other words: the tracker is now in a good place to pause, reset the docs, and choose the next deeper branch from a stronger baseline than before.
