# TO-MHT Roadmap

This roadmap is intentionally forward-looking, but it starts from the tracker’s **current real baseline** rather than from the pre-transition plans.

The structural track-oriented transition is complete enough that the main question is no longer “how do we make this a true TO-MHT?” The more relevant question is now:

> given the current track-oriented baseline, exact-solver seam, local-association/sequential runtime story, and current replay bottlenecks, what should be improved next, what belongs together, and which topics are highest leverage?

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
- exact-one-of `predictor` or `hypothesiser` constructor semantics,
- a narrow distance-hypothesiser seam for local association,
- explicit NLL-based local scoring,
- an explicit solver-facing exact cluster-solver contract,
- `branch_and_bound` as the default exact cluster backend,
- `exhaustive` retained as exact reference/fallback,
- `ortools` retained as experimental exact backend,
- and per-scan timing-phase instrumentation showing where replay time is now going.

This means the tracker is no longer blocked on the core architectural transition, is no longer blocked on the original exact-solver bottleneck, and has now also landed the main local-association ownership cleanup that was the focus of the most recent phase.

---

## 2. Guiding principles for the next phase

### 2.1 Preserve the clear runtime story

The strongest gain from the transition is that the runtime story is now understandable:

- trees/leaves are the persistent state,
- globals are rebuilt per scan,
- MAP-only N-scan pruning commits tree structure directly,
- output tracks are reconstructed from committed prefix plus unresolved lineage,
- exact cluster solving sits behind a clean dedicated seam,
- and local association now sits behind a narrow distance-hypothesiser seam with explicit NLL scoring.

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

### A. Local expansion volume reduction / pre-expansion control

This now looks like the clearest immediate performance bottleneck on the primary replay used during the recent phase.

The recent timing and local-association work suggests that the next main leverage point is no longer only the inner local-association math. It is likely **how many leaves require full expansion**, and where explicit, conservative controls can reduce that volume.

This topic includes:

- profiling and characterizing how many leaves are expanded,
- determining which expanded leaves are later useful,
- reducing how many leaves require full local expansion work,
- revisiting how many children are retained/generated per leaf,
- and making explicit which controls are semantic vs tractability-oriented.

This is currently the strongest candidate for the **next major technical branch**.

### B. Local branching / scoring design

The old PDA/beta-oriented path has now been replaced by a much cleaner NLL-based local scoring story. That is a big improvement, but it should still be treated as pragmatic rather than final.

This topic includes:

- refining the distance-hypothesiser contract,
- clarifying which local score contributions should live in the hypothesiser vs scoring layer,
- possible future simplification of the current unused-detection correction,
- and any deeper local-association/scoring redesign beyond the current baseline.

This topic overlaps strongly with local-expansion runtime work and may travel with it, but it is no longer “own the local association path from scratch.” That part is now largely done.

### C. Internal birth / existence / quality semantics

Internal births now work, but they remain intentionally simple and are clearly not the final design.

Recent review notes still suggest that false starts likely became worse after the transition for structural reasons:

- birth candidates become real trees immediately,
- post-birth existence is effectively mandatory,
- and whole-track lifecycle kill is slower/more permissive than before.

This topic includes:

- whether internal births should remain direct tree insertion or gain a more uncertain/probationary stage,
- whether the current residual policy is too conservative or appropriately protective,
- whether miss-lifecycle policy is too permissive for low-quality birth trees,
- candidate observability and TO-MHT-native birth impact statistics,
- general false-start tuning,
- and follow-up review of target swapping / track jumping in replay output.

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
- smoke/replay baseline maintenance,
- and scenario/replay validation sufficient to trust deeper changes.

This is important ongoing support work, but no longer the defining next phase.

---

## 4. Rough priority picture

The current rough priority picture looks like this.

### Highest-leverage near-term areas

These currently look the most likely to shape the next deeper phase:

1. **Local expansion volume reduction / pre-expansion control**
2. **Internal birth / existence / quality semantics**
3. **Further local branching / scoring refinement**

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
- more extensive packaging/handoff polish,
- and local-expansion parallelization/orchestration architecture.

Parallelization is a real future axis, but it should likely remain later, explicit, and opt-in rather than being mixed into the current expansion-volume phase.

---

## 5. Recommended interpretation of likely next branches

At the current checkpoint, several next branches are plausible.

### Option 1: Expansion-volume-first branch

Focus:
- characterize how many leaves are expanded,
- identify which expansions are useful,
- reduce the number of leaves requiring full expansion,
- and improve runtime by pre-expansion control rather than only inner-kernel optimization.

Why this is attractive:
- the recent local-association math passes already delivered worthwhile wins,
- the remaining replay bottleneck still sits in local expansion,
- and the next likely leverage is now expansion volume rather than another comparable kernel-only win.

This currently looks like the strongest candidate.

### Option 2: Quality/birth/existence-first branch

Focus:
- false starts,
- birth insertion semantics,
- existence/absence alternatives,
- lifecycle kill behavior,
- and target-swapping / track-jumping review.

Why this is attractive:
- current output-quality concerns are now easier to see against a more stable runtime baseline,
- and some of these issues are conceptually clearer than before.

Why it may not be first:
- the current runtime bottleneck is still more clearly on the expansion side,
- and ISAC integration is external-start-only.

This should stay high on the list even if not chosen first.

### Option 3: Further local-association/scoring refinement branch

Focus:
- simplify or replace the current unused-detection correction,
- refine distance-hypothesiser/scoring boundaries,
- consider future tracker-owned or custom orchestration hooks,
- and possibly revisit local score decomposition more deeply.

Why this is attractive:
- the architecture is now much cleaner than before,
- and the current NLL baseline makes deeper reasoning easier.

Why it may not be first:
- the current local-association baseline is good enough to support a runtime-focused expansion-volume phase now.

---

## 6. What belongs together

To avoid fragmented work, some topics should be grouped deliberately.

### Expansion-volume work should probably include

- profiling and characterization of expensive expansion scans,
- revisiting the number of local hypotheses retained/generated,
- selective/pre-prioritized expansion where justified,
- any local caching/reuse opportunities that still materially reduce cost,
- and organization of the expansion-related tracker code as part of that work.

### Scoring work should probably include

- local branching ownership,
- raw local-score semantics,
- any future simplification or removal of the current unused-detection correction,
- and resulting changes to clutter / birth / miss interpretation.

### Birth/existence work should probably include

- birth insertion semantics,
- residual policy review,
- lifecycle kill policy review,
- initiator interaction,
- candidate observability,
- TO-MHT-native birth impact metrics,
- and explicit replay-quality inspection around false starts and track jumping.

### Approximation work should probably include

- overload split / historical relaxation review,
- whether current exact/approximate boundaries are still the right ones,
- and how those mechanisms should interact with future runtime or scoring changes.

### Parallelization work should probably include

- opt-in runtime behavior only,
- a clean orchestration abstraction above hypothesiser execution,
- room for sequential, tracker-owned parallel, and external/custom parallel modes,
- and strong determinism/validation expectations.

This is deliberately a later topic, not the main current branch.

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

### 7.2 The local-association ownership phase also achieved its main goal

The recent local-association phase produced:

- a tracker-owned default distance hypothesiser,
- explicit NLL-based local scoring,
- removal of the old PDA/beta-oriented main-path semantics,
- and measurable `expand_ms` wins from conservative math/runtime cleanup.

That means the “own the local association path” topic is no longer the same kind of open-ended restructuring task it was at the start of that phase.

### 7.3 Timing instrumentation changed the picture again

The timing work first revealed that exact cluster solving stopped being the main bottleneck, and then the local-association optimization work revealed that the next likely leverage point is **expansion volume** rather than only more inner-kernel math optimization.

That should directly shape what comes next.

### 7.4 Code organization is real, but should still be attached to real work

There is still tracker-internal organization work worth doing, but most of it should probably travel with the next substantive branch rather than define a standalone cleanup phase.

---

## 8. Near-term execution style

The immediate post-phase workflow should likely be:

1. refresh `CURRENT_STATE`, `NEXT_STEPS`, and roadmap to match the new baseline,
2. treat the just-completed local-association phase as done,
3. start the next branch with an analysis/review pass on expansion volume,
4. then do a targeted implementation pass,
5. then re-evaluate whether the following step should be another volume-control pass, a quality pass, or a later architectural topic.

That avoids prematurely mixing expansion-volume control, quality tuning, and parallelization design into one branch.

---

## 9. Near-term priority summary

At the current checkpoint, the best summary is:

1. The track-oriented architectural transition is complete enough to treat as the stable baseline.
2. The recent solver phase succeeded in improving the exact cluster solver enough to make branch-and-bound the default backend.
3. The recent local-association phase succeeded in establishing a tracker-owned distance-hypothesiser baseline with explicit NLL scoring and meaningful `expand_ms` wins.
4. The main replay bottleneck is now best interpreted as **local expansion volume**, not exact cluster solving and not only local-association math.
5. Internal birth/existence semantics and output-quality review remain high-priority follow-up topics, but do not have to block the next runtime-focused step.
6. Parallelization should stay a later, explicit, opt-in architectural topic rather than being mixed into the current next phase.

In other words: the tracker is now in a good place to consolidate the docs, acknowledge what the recent phase achieved, and move on to a targeted expansion-volume reduction phase from a stronger baseline than before.
