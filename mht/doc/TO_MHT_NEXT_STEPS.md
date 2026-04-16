# TO-MHT Next Steps

## Update (2026-04-16): Cholesky local-association math pass implemented

- Implemented in `TrackerOwnedNLLDistanceHypothesiser`:
  - conservative rectangular pre-gating from raw covariance diagonal before
    full Mahalanobis/NLL work,
  - one-entry exact-equality covariance-prep reuse (`np.array_equal(...)`) with
    prepared SPD covariance reuse in the NLL path,
  - prepared covariance payload now includes Cholesky factor `L`, with
    `logdet = 2 * sum(log(diag(L)))`,
  - full Mahalanobis/NLL path now uses triangular solve (`L y = x`,
    `d2 = y^T y`) rather than direct solve on full covariance,
  - scan-time prediction reuse when `detection_timestamp == timestamp`,
  - one-entry measurement-prediction reuse when both `prediction` and
    `measurement_model` match by object identity (`is`).
- Kept unchanged in this pass:
  - full Mahalanobis-threshold gate remains authoritative,
  - rectangular pre-gating semantics remain unchanged,
  - miss handling and scoring semantics remain unchanged.
- Validation:
  - `pytest mht/tests/test_tomht_hypothesiser.py` passed,
  - `make smoke_compare` exact normalized match,
  - `make replay_compare` exact normalized match,
  - `python pre_commit.py --no-dirty` passed.
- Timing snapshots (`make smoke_compare_timing`, `make replay_compare_timing`)
  showed lower `expand_ms` in both smoke scenarios and standard replay.

## Update (2026-04-16): baseline-quality concerns logged

- Logged for dedicated follow-up quality pass:
  - false starts are still somewhat high in smoke/scenario output,
  - replay can show somewhat more target swapping / track jumping.
- Planned near-term sequencing:
  1. refresh smoke/replay baselines to current intended behavior,
  2. proceed with expansion-path optimization work (including rectangular
     pre-gating),
  3. run a focused output-quality review/tuning pass.

## Update (2026-04-16): baseline cleanup around scoring/replay overrides

- Implemented: stale replay override templates for removed `hypothesis_backend`
  parameter were removed from `replay/overrides/`, and replay README references
  were updated accordingly.
- Clarified: scoring contract now explicitly documents that
  `used_det_keys`/`used_det_key` are local indices into the current
  `ScanContext.detections`.
- Clarified: current solver-preparation path assumes
  `score_unused_detections(...)` is affine in number of used detections.
  Revisit/removal/replacement of this concept is deferred to a later scoring
  redesign pass.

## Update (2026-04-16): explicit NLL scoring model restored

- Implemented: `ScoringModel.score_track_hypotheses(...)` is restored and now
  owns explicit local LLR scoring from distance hypotheses.
- Implemented: default `NLLScoringModel` now scores:
  - hit: `log(P_D) - log(lambda) - NLL`
  - miss: `log(1 - P_D)`
- Implemented: tracker local expansion now obtains local `log_delta` values
  from `score_track_hypotheses(...)` rather than implicitly using
  `-hypothesis.distance`.
- Implemented: tracker-owned default distance hypothesiser miss distance is a
  sentinel (ignored by scoring), while detection distance remains `NLL`.

## Update (2026-04-15): local-expansion seam reframed as distance hypothesiser

- Implemented: local expansion now consumes a narrow distance-hypothesiser
  contract via Stone Soup `MultipleHypothesis` of
  `SingleDistanceHypothesis` objects (one miss + gated detections, each with
  `distance`).
- Updated in 2026-04-16 follow-up: tracker local delta is now explicitly scored
  by `ScoringModel` from `NLL` distance plus `P_D`/`lambda` terms.
- Implemented: `TOMHTTracker` constructor now enforces exactly one of
  `predictor` or `hypothesiser` (with required `updater` in both cases).
- Implemented: tracker-owned default local hypothesiser is now
  `TrackerOwnedNLLDistanceHypothesiser` with Mahalanobis-threshold-first gating
  semantics.
- Updated in 2026-04-16 follow-up: scoring contract again includes local track
  scoring (`score_track_hypotheses`) with explicit NLL semantics.
- Implemented: transitional compatibility shims for
  `hypothesis_backend`/`prob_gate`/beta-ratio local semantics were removed from
  the main tracker path.

## Update (2026-04-13): first conservative ownership step implemented

- Implemented: main-path local expansion now runs through tracker-owned PDA-based
  logic (`mht/tomht_hypothesiser.py`) instead of relying on generic
  `hypothesis_generator.hypothesise(...)` in the default `"local_pda"` path.
- Implemented: Stone Soup `PDAHypothesiser` was used as the behavior/structure
  baseline for the owned path.
- Implemented: `RobustPDAHypothesiser` was moved to explicit compatibility-only
  backend usage (`"robust_pda"`), no longer the preferred runtime path.
- Implemented: distinct backend naming now separates tracker-owned
  `"local_pda"` from legacy Stone Soup `"stonesoup_pda"` usage.
- Remaining in this subphase: cheap pre-gating (for example rectangular gate),
  deeper expansion profiling, and optimization passes.

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
