# TO-MHT Roadmap

This roadmap is intentionally forward-looking, but it starts from the tracker’s **current real baseline** rather than from the pre-transition plans.

The structural track-oriented transition is now complete enough that the central question is no longer “how do we make this a true TO-MHT?” The central question is now:

> given the current track-oriented baseline, what should be improved next, what belongs together, and which topics are highest leverage?

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
- and replay completion on the main recorded dataset.

This means the tracker is no longer blocked on the core architectural transition. It is now in a position where the next work should be chosen based on leverage and coherence, not by continuing to chase an already-completed architecture phase.

---

## 2. Guiding principles for the next phase

A few principles should shape the next decisions.

### 2.1 Preserve the clear runtime story

The strongest gain from the transition is that the runtime story is now understandable:

- trees/leaves are the persistent state,
- globals are rebuilt per scan,
- MAP-only N-scan pruning commits tree structure directly,
- and output tracks are reconstructed from committed prefix plus unresolved lineage.

Future work should preserve that clarity.

### 2.2 Prefer conservative, incremental deepening

The preferred style remains:

1. analyze / review what should change
2. make a targeted implementation pass

This is especially important now that the tracker is good enough to experiment with meaningfully.

### 2.3 Avoid cleanup-only phases when the cleanup belongs to a deeper topic

There is still internal organization work worth doing, especially around `_build_track_clusters(...)` and `_rebuild_cluster_globals(...)`. But that organization should preferably be treated as a **sub-goal of deeper work** such as solver/runtime work, approximation cleanup, or scoring redesign, rather than as an isolated readability-only phase.

### 2.4 Keep docs aligned with the actual code and reasoning

The docs are an important part of the handoff and planning workflow. As deeper work proceeds, `CURRENT_STATE`, `NEXT_STEPS`, and the roadmap should continue to evolve with the implementation.

---

## 3. Main topic groups from the current checkpoint

The next work appears to fall into five main topic groups.

### A. Runtime / cluster-solver scalability

This is the clearest current technical bottleneck.

The tracker is now robust enough to run through the main replay, but large merged clusters still create heavy-tailed scan times because the current rebuild path is still exhaustive enumeration. Overload splitting and historical relaxation help, but they are mitigations rather than a final scalability story.

This topic includes:

- replacing exhaustive enumeration with a more scalable K-best cluster solver,
- profiling where cluster combinatorics actually come from,
- reducing the number of expensive merged clusters that reach the rebuild stage,
- and improving the structure/readability of the cluster build/rebuild code as part of that work.

This is currently the strongest candidate for the **next major technical branch**.

### B. Local branching / scoring design

The public boundary has shifted to predictor/updater, but local branching is still internally PDA-style. The default scoring remains beta-ratio-based and should still be understood as pragmatic rather than final.

This topic includes:

- clearer decomposition of local score contributions,
- eventual tracker-owned local branching/scoring logic rather than dependence on PDA-style packaging,
- reduced coupling to hypothesiser-oriented concepts,
- better alignment between local branch scoring and rebuilt-global semantics,
- and possible prediction/gating/likelihood reuse or batching opportunities.

This topic is important, but it is not yet obvious whether it should come **before** runtime/scalability work or after it.

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

This work likely belongs together with runtime/solver work more than as an isolated phase.

### E. Integration / validation / operational hardening

Some earlier roadmap items remain relevant, but they now belong in a different context.

This topic includes:

- continued local replay validation,
- keeping the tracker easy to drop into Stone Soup-style workflows,
- preserving Python / Stone Soup compatibility,
- practical runner/parameter override support,
- and scenario/replay validation sufficient to trust the next deeper changes.

This is important ongoing support work, but no longer the defining next phase.

---

## 4. Rough priority picture

The current rough priority picture looks like this.

### Highest-leverage near-term areas

These currently look the most likely to shape the next deeper phase:

1. **Runtime / cluster-solver scalability**
2. **Scoring / local-branching ownership**
3. **Internal birth / existence semantics**

That does **not** mean they must be tackled in that exact order. But it does mean they currently look like the most consequential topics.

### Important supporting areas

These should travel with the deeper work rather than define standalone phases:

- cluster/build/rebuild code organization,
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

### Option 1: Runtime-first branch

Focus:
- cluster-growth pressure,
- solver replacement,
- overload-split/relaxation semantics,
- and organization of the cluster build/rebuild code as part of that work.

Why this is attractive:
- runtime is the clearest current bottleneck,
- replay now works well enough that performance profiling is meaningful,
- and the tracker already has a conservative internal cluster-solver seam.

This currently looks like the strongest candidate.

### Option 2: Scoring/local-association-first branch

Focus:
- tracker-owned local branching math,
- less PDA-style dependency internally,
- cleaner score decomposition,
- and possible batching/caching opportunities.

Why this is attractive:
- the public interface already moved in this direction,
- scoring remains one of the most obviously provisional parts of the design,
- and better local scores may improve both quality and runtime indirectly.

This is compelling, but may still be slightly less urgent than the runtime story.

### Option 3: Birth/existence-first branch

Focus:
- false starts,
- birth insertion semantics,
- existence/absence alternatives,
- lifecycle kill behavior,
- and TO-MHT-native birth observability/statistics.

Why this is attractive:
- recent reviews strongly suggest that internal-birth quality got worse structurally,
- and the issue is now conceptually clearer than before.

Why it may not be first:
- ISAC integration is external-start-only,
- and the issue is entangled with scoring and lifecycle policy.

This should stay high on the list even if not chosen first.

---

## 6. What belongs together

To avoid fragmented work, some topics should be grouped deliberately.

### Runtime / solver work should probably include

- profiling and characterization of merged clusters,
- solver replacement or solver-path refinement,
- overload split / historical relaxation review,
- and reorganization of `_build_track_clusters(...)` + `_rebuild_cluster_globals(...)` as part of that work.

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

This grouping should help avoid “half-fixes” that touch only one visible symptom.

---

## 7. Practical notes from the current checkpoint

A few concrete observations should shape later choices.

### 7.1 Determinism matters early

The recent birth-cap nondeterminism issue was small but highly worthwhile to fix. Deterministic behavior makes replay and scenario debugging much easier. Future deeper work should preserve that standard.

### 7.2 Output-history restoration completed an important usability gap

Committed-prefix output reconstruction was a small but valuable completion step. Similar improvements are worth doing when they clarify the runtime story without distorting the core architecture.

### 7.3 Internal births should not dominate the whole roadmap

Internal births are important, but the roadmap should not let them overshadow the broader runtime/scoring architecture questions unless operational use or replay evidence says they must.

### 7.4 Code organization is real, but should be attached to real work

The cluster build/rebuild section now carries a lot of closely related logic. This absolutely should improve, but most likely as part of whichever deeper branch is chosen next.

---

## 8. Near-term execution style

The immediate post-reset workflow should likely be:

1. keep `CURRENT_STATE` accurate and clean,
2. use this roadmap as a topic map rather than a strict sequence,
3. choose the next focused branch deliberately,
4. draft a fresh `NEXT_STEPS` only once that branch is chosen,
5. then do a targeted implementation pass.

This avoids committing too early to a precise phase order that may not survive contact with replay evidence.

---

## 9. Near-term priority summary

At the current checkpoint, the best summary is:

1. The track-oriented architectural transition is complete enough to treat as the new baseline.
2. The main open technical weakness is runtime on large merged clusters.
3. Scoring/local-branching design and internal birth/existence semantics are the two other major unresolved design areas.
4. Approximation semantics, validation, observability, and code organization should be advanced together with those deeper topics rather than as isolated phases.
5. The next concrete implementation phase should be chosen only after a focused decision between the main branches above.
