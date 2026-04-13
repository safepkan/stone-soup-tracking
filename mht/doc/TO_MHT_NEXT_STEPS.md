# TO-MHT Next Steps

## Next architectural subphase

**Local expansion / hypothesis-generation ownership and performance**

The previous runtime/scalability phase is now complete enough that the exact cluster
solver is no longer the dominant replay bottleneck. With `branch_and_bound` as the
default exact backend, the main cost driver on the current replay is now local
expansion / hypothesis generation.

This phase is therefore about taking ownership of the local association path,
removing the remaining internal dependence on the hypothesiser abstraction, and
streamlining expansion toward the exact shape the tracker actually needs.

---

## Why this subphase now

Recent timing work shows that `expand_ms` is now the dominant scan-time component on
the main replay path.

A direct comparison of the current internal hypothesiser backends also showed that
the built-in Stone Soup `PDAHypothesiser` is substantially faster than the current
`RobustPDAHypothesiser` path on the same replay configuration. That strongly suggests
that the current internal hypothesiser situation is both:

- a performance problem, and
- an abstraction mismatch now that the public tracker boundary has already shifted to
  `predictor + updater`.

The next step is therefore not another solver phase. It is to simplify and own the
local branching path.

---

## Goal of this subphase

At the end of this subphase, the tracker should have:

1. a tracker-owned local association / local branching path built around
   `predictor + updater`,
2. no remaining need for `RobustPDAHypothesiser`,
3. a clear baseline derived from Stone Soup PDA behavior rather than from the old
   custom robust wrapper,
4. a cleaner internal runtime story for local expansion,
5. and at least one obvious cheap performance improvement applied to that path,
   such as rectangular pre-gating before more expensive gating / likelihood work.

This phase is successful even if the scoring model is not yet fundamentally redesigned,
provided the tracker gains a clearer and faster local expansion path.

---

## Core design intent

### 1. Keep `predictor + updater` as the real public boundary

The public constructor/API has already shifted away from “user provides a
hypothesiser” toward “user provides predictor and updater”. The internal structure
should now move in the same direction.

The tracker should stop being internally organized around a generic hypothesiser
interface and instead implement the specific prediction / gating / scoring /
miss-handling flow it actually needs.

### 2. Use built-in Stone Soup PDA as the baseline reference, not `RobustPDAHypothesiser`

`RobustPDAHypothesiser` was introduced early as a practical workaround for numerical
issues, especially around the UKF bearing/range synthetic case. It should no longer be
treated as the conceptual baseline.

The intended starting point for this phase is:

- drop `RobustPDAHypothesiser`,
- copy the relevant behavior/structure from Stone Soup `PDAHypothesiser`,
- then refactor that copied logic into a tracker-owned local association path.

### 3. Keep scope narrower than a full scoring redesign

This phase is about ownership and performance of the local expansion path first.

It may expose scoring issues and may require small scoring-related adjustments, but it
is not yet intended to be the full “redo the scoring model” phase.

### 4. Prefer cheap obvious filtering before expensive work

One of the clearest expected early improvements is adding a cheap pre-gating stage
before full Mahalanobis gating / likelihood evaluation.

Rectangular gating is the most obvious candidate:
- use it to discard many detections cheaply,
- then run more expensive proper gating / measurement-likelihood work only on the
  survivors.

This is the kind of conservative optimization that should fit well in this phase.

---

## Intended implementation direction

### 1. Remove `RobustPDAHypothesiser` from the main path

The tracker should no longer depend on the custom robust wrapper as one of the normal
internal backends.

It is acceptable to keep old compatibility code temporarily if needed for transition,
but the direction should be clearly toward removal rather than preservation.

### 2. Build a tracker-owned local association helper/module

Create a tracker-owned local association path that explicitly performs:

- prediction,
- cheap pre-gating,
- proper gating,
- measurement likelihood evaluation,
- missed-detection handling,
- local score contribution construction,
- and handoff to updater for selected measurement hypotheses.

This should be organized around what the tracker actually consumes, not around
reproducing the full generic hypothesiser abstraction.

### 3. Keep behavior conservative at first

The first pass should aim for:

- similar candidate semantics,
- similar miss handling,
- similar score interpretation,
- and similar replay/scenario outputs as the current baseline,

while improving structure and performance.

### 4. Use the new regression/timing harness continuously

This phase should lean on the newly available:
- smoke-test regression checks,
- replay summaries,
- and full timing logs.

The goal is to make changes incrementally and observe whether `expand_ms` improves
without silently damaging tracker behavior.

---

## What should happen in this subphase

### 1. Replace the internal hypothesiser-shaped path

Refactor local expansion so that the tracker no longer fundamentally depends on:

- `hypothesis_generator.hypothesise(...)`,
- `params.hypothesis_backend`,
- or `RobustPDAHypothesiser`

as the core internal structure.

A transitional compatibility layer can remain briefly if needed, but the intended
runtime path should become tracker-owned.

### 2. Establish a clean copied baseline from Stone Soup PDA

Before deeper optimizations, create a tracker-owned baseline derived from the built-in
Stone Soup `PDAHypothesiser` logic.

This gives:
- a clearer starting point,
- simpler semantics than the old custom robust wrapper,
- and a faster baseline than the current default robust path.

### 3. Add rectangular pre-gating

Implement a cheap rectangular gate ahead of proper Mahalanobis gating / likelihood
work.

This should be the first obvious optimization to try once the tracker owns the local
association path directly.

### 4. Reassess where time goes inside local expansion

Once the tracker owns the path, profile local expansion more specifically:

- prediction cost
- measurement-prediction cost
- rectangular pre-gating
- Mahalanobis gating
- likelihood evaluation
- updater/update cost
- candidate sorting / retention

This should guide later optimizations inside the same broad branch.

### 5. Keep docs/comments aligned

As the hypothesiser abstraction is removed internally, update comments/docs so they no
longer describe PDA-style hypothesiser ownership as the intended runtime story.

---

## What should **not** happen in this subphase

This phase should **not** yet:

- redesign the exact cluster solver again,
- remove the solver seam,
- do a broad birth/existence redesign,
- do a full scoring-theory rewrite,
- or broaden into a general cleanup-only pass.

The point here is to simplify and speed up the local expansion path first.

---

## Acceptance criteria

This subphase should be considered complete when:

- the main internal local expansion path is tracker-owned and no longer depends on
  `RobustPDAHypothesiser`,
- Stone Soup PDA has been used as the baseline reference for the new owned path,
- at least one cheap prefiltering optimization such as rectangular gating is in place,
- replay/scenario behavior remains materially acceptable,
- and timing data shows that the expansion path is better understood and at least
  somewhat improved.

A strong secondary success criterion is that someone reading the tracker can now answer,
in one place:

> what exact steps are performed during local expansion, and where is the time going?

---

## Recommended implementation style

This phase should follow the usual conservative style:

1. copy / establish the built-in PDA baseline internally,
2. remove the custom robust wrapper from the main path,
3. add cheap pre-gating and other obvious local optimizations,
4. validate against smoke tests and replay timing,
5. then decide whether a deeper scoring/local-association redesign should follow.

That is the intended scope of this phase.