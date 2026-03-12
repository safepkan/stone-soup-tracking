# TO-MHT next steps (current phase): External initiation + birth handling cleanup

This document lists the upcoming tasks for the current phase.
It intentionally focuses on what is next from the current baseline, rather than repeating already-completed work.

## 1. Why this is the next phase

The current tracker already has:
- stable detection ordering,
- beta-ratio v1.5 scoring,
- association history with N-scan-lite deduplication,
- and useful instrumentation.

The main immediate gap is **integration-facing initiation / birth handling**:
- the current internal-birth path is still heuristic,
- the first realistic integration target needs **external track initiation**,
- and the first integration step is likely to replace only an existing **system tracker**, not the whole upstream start pipeline.

For the near term, we want the tracker to work cleanly in two modes:

1. **External-initiation mode (priority)**
   - upstream code provides **confirmed new system tracks** after each scan,
   - TO-MHT takes over from there.

2. **Standalone internal-birth mode**
   - TO-MHT can still run self-contained experiments using an internal initiator.

## 2. Design goals for this phase

### 2.1 External initiation should be first-class

The tracker should be able to consume externally created tracks during a run, not only at construction time.

This matters for the ISAC system-tracker replacement path, where:
- sensor trackers and cross-sensor correlation stay upstream,
- ambiguity resolution for starts may also stay upstream initially,
- and TO-MHT begins from already-initiated global-coordinate tracks.

### 2.2 Start with confirmed external starts, not generic external birth candidates

For the first integration step, external starts should be treated as **confirmed upstream starts** rather than as soft birth candidates.

Rationale:
- this matches the current ISAC flow,
- it allows TO-MHT to replace the system tracker without changing the upstream start semantics,
- and it avoids immediate ambiguity about whether TO-MHT is allowed to reject an upstream-created start.

A softer “external candidate” mode may still be useful later, but it is deferred.

### 2.3 Use the existing scenario runners as the primary validation harness for this phase

Going into this phase, the practical regression harness is the existing scenario workflow rather than a formal unit-test suite.

In particular, the current `crossing` and `bearing_range` scenarios already provide:
- repeatable runs,
- visualisation,
- log output,
- and aggregate summary statistics.

For this phase, those runners should remain the main acceptance path.
Small direct tests are still useful, but only for narrow API-contract checks such as timestamp ordering and rejected invalid calls.

### 2.4 Birth logic should become easier to reason about

The current code is useful, but still mixes:
- residual-based birth discovery,
- birth ranking,
- compatibility checks,
- and branching policy.

This phase should make those pieces more explicit.

### 2.5 Keep this phase narrow and integration-facing

This phase should add the minimum surface needed for clean external initiation and modest internal-birth cleanup.
It should **not** expand into a broader redesign of scoring, tree structure, or existence modelling.

## 3. Proposed immediate implementation direction

### 3.1 Add an explicit external-start interface

Add a separate tracker call for injecting externally created starts after a scan update, for example:

- `step(detections, timestamp)`
- `add_external_starts(starts, timestamp)`

Initial scope:
- support **confirmed** external starts only,
- require the timestamp to match the most recent `step()`,
- and fail fast if the call is made before any `step()` or for a mismatched timestamp.

For this phase, avoid exposing a broader public mode like `mode="confirmed"` unless a real second mode exists.
The public API should match the actual supported semantics.

### 3.2 Make the timestamp invariant explicit in tracker state

Add explicit tracker state for the most recent scan timestamp so the external-start insertion path can enforce:
- “external starts must correspond to the most recent completed scan”,
- no stale or future-timestamp insertion,
- and no ambiguous ordering of `step()` versus external-start injection.

This should be treated as a hard interface invariant, not just a documentation note.

### 3.3 Treat external starts as already-initialised system tracks

For the first version, externally supplied starts should be assumed to:
- already be in the system-track state space,
- already be initialised at the given timestamp,
- already reflect any upstream ambiguity resolution or correlation logic,
- and already be intended as confirmed new system tracks.

TO-MHT should therefore insert them as new tracks into the current global hypotheses,
rather than trying to re-derive them from current detections.

### 3.4 Keep the internal birth path separate for now

Do **not** immediately force internal births and external starts through a shared public abstraction.

Instead:
- add a clean external-start path first,
- keep internal births as the existing standalone initiator-driven path,
- and revisit a shared internal/external birth abstraction later if it still looks useful.

### 3.5 Use one shared helper for inserted-track metadata initialisation

Newly inserted tracks currently need internal metadata such as track ID, age/hit counters, missed count,
and association-history fields.

This phase should define one shared helper for initialising newly inserted tracks so that:
- constructor-time initial tracks,
- internal births,
- and external starts

all use a consistent metadata initialisation path where appropriate.

This is mainly a maintainability and consistency improvement, but it should be done now to avoid three slightly different insertion conventions.

### 3.6 Make external-initiation-only mode easy

Add a clear configuration path so the tracker can run in:
- external-initiation-only mode,
- internal-birth-only mode,
- or both.

For the first ISAC integration step, external-initiation-only mode is likely the default.

Important nuance:
- the conceptual mode already exists at the design level,
- but it is only operationally complete once runtime external-start injection exists.

#### Assumptions for the first ISAC integration

For the first integration step, externally supplied starts are assumed to:
- already be initialised in global/system coordinates,
- already have any start-time ambiguity resolution applied upstream,
- already correspond to the current system-tracker timestamp,
- and be intended as confirmed new system tracks.

### 3.7 Make the existing scenarios runnable in delayed external-start mode

The existing `crossing` and `bearing_range` scenario runners should be extended so they can run in a mode with:
- no initial tracks,
- optional internal births disabled,
- and externally injected confirmed starts after a configurable delay or start scan.

This should be treated as part of the current phase rather than a later extra,
because it provides the most relevant integration-style validation of the new external-start workflow.

For the first version, this runner mode should stay simple:
- derive the externally injected starts from known scenario truth/start information,
- inject them at the configured scan/time,
- and avoid introducing extra “soft upstream candidate” semantics.

## 4. Birth handling cleanup in this phase

This phase does **not** need a fully principled birth/existence model.
But it should make the external-start path explicit and the remaining internal-birth behaviour more controlled and understandable.

### 4.1 Separate the birth pipeline conceptually

Make the following stages explicit in code/comments/docs:
- candidate generation,
- candidate filtering / sanity checks,
- candidate scoring / ranking,
- compatibility against existing global hypotheses,
- branching policy.

### 4.2 Keep external starts out of the internal birth-discovery path

Confirmed external starts should be inserted as structural additions to the current global hypotheses.
They should **not** be routed through:
- residual detection logic,
- internal initiator ranking,
- or support-detection rediscovery.

For this phase, they should also avoid any extra semantics beyond the existing “insert a confirmed new track” interpretation.
Any later attempt to score external starts more explicitly can be deferred until the scoring/existence-model phase.

### 4.3 Keep birth control simple for now

If birth pressure still looks problematic after the refactor, add only lightweight control measures, such as:
- limiting births to top-ranked parent globals,
- or making the maximum number of compatible birth branches more explicit.

Do **not** turn this phase into a large tuning exercise.

## 5. Acceptance criteria for this phase

### External-initiation path

- The tracker can accept externally created starts after `step()`.
- The external-start API clearly represents **confirmed starts only**.
- The tracker explicitly rejects calls made before any `step()` or with a mismatched timestamp.
- Confirmed external starts can be used with internal births disabled.
- Externally supplied starts are inserted with the correct timestamp and internal metadata initialised via the shared insertion path.
- Existing standalone scenarios still run unchanged when the external-start path is unused.
- The `crossing` and `bearing_range` scenario workflows can run in a delayed external-start mode.

### Validation harness

- Small direct tests cover narrow API-contract checks such as call ordering and timestamp mismatch rejection.
- The primary practical validation for this phase is still through the scenario runners and their existing summary/log/visualisation workflow.
- A headless scenario path exists for exercising the new external-start mode in repeatable smoke/regression runs.

### Internal-birth path

- Existing standalone scenarios still run.
- Internal birth behaviour is at least as understandable as before.
- Birth instrumentation still reports meaningful scan/run summaries.
- The code structure makes the internal birth stages easier to identify and review.

### Documentation

- Roadmap and chat-context docs reflect the new current phase.
- The current-state doc explains the external-initiation capability once implemented.
- The tracker API documentation/comments make the timestamp invariant and confirmed-start semantics explicit.
- Scenario-runner documentation/comments explain how delayed external-start mode is exercised.

## 6. Deferred for later phases

Not part of this phase:
- soft external birth candidates,
- public multi-mode external-start APIs that imply unsupported semantics,
- per-start support-detection handling,
- common internal/external birth-candidate abstraction,
- explicit shared hypothesis trees,
- true ancestor-based N-scan pruning,
- replacing beta-ratio v1.5 scoring,
- principled existence-probability modelling,
- large-scale performance optimisation,
- full evaluation / benchmarking framework.

## 7. Suggested Codex task sequence for this phase

The goal here is to give Codex tasks that are small enough to implement and validate cleanly,
while still leaving each step meaningful.

### Task 1 — Add external-start API skeleton and timestamp invariant

Scope:
- add tracker state for the most recent `step()` timestamp,
- add `add_external_starts(starts, timestamp)`,
- validate call ordering and timestamp matching,
- add focused direct tests for accepted/rejected call sequences.

Deliberately out of scope:
- actual insertion into globals,
- metadata-helper refactor,
- internal birth cleanup,
- scenario-runner changes.

Review focus:
- API shape,
- invariant enforcement,
- error clarity,
- no behavioural change when unused.

### Task 2 — Implemented: external confirmed-start insertion into globals

Status (2026-03-12):
- `add_external_starts(starts,timestamp)` now inserts the supplied confirmed starts into every current global hypothesis.
- Inserted tracks get tracker-owned `track_id` values and baseline maintenance metadata (`age`, `hits`, `missed_count`, `last_det_key`, `last_det_hit`, `assoc_history`).
- External starts are kept separate from residual/internal birth discovery and do **not** receive `birth_log_penalty`.
- Duplicate-like inputs are handled in the simplest way: each supplied start is treated as a distinct confirmed track and gets a fresh tracker-owned `track_id`.
- Focused tracker tests cover insertion, empty-input no-op, metadata initialisation, and repeated insertion behaviour.

Scope:
- insert externally supplied confirmed starts into each current global hypothesis,
- allocate stable track IDs,
- keep semantics simple and deterministic,
- treat them as structural additions rather than internal births,
- add narrow direct tests for insertion behaviour, empty input, and obvious invalid/edge cases.

Deliberately out of scope:
- scenario-runner external-start mode,
- internal-birth refactor,
- broader operating-mode cleanup beyond what is necessary for correctness.

Review focus:
- per-global insertion semantics,
- compatibility with beam/global bookkeeping,
- deterministic behaviour,
- whether the semantics match “confirmed upstream starts”.

### Task 3 — Add delayed external-start mode to the scenario runners

Scope:
- extend the `crossing` and `bearing_range` runner workflow so scenarios can run with delayed external confirmed starts,
- support configurations with no initial tracks,
- support configurations with internal births disabled,
- provide a simple configurable delay or start-scan mechanism,
- preserve existing default behaviour when the mode is unused,
- keep the mode runnable in headless smoke/regression form.

Deliberately out of scope:
- sophisticated upstream-start modelling,
- new scoring semantics for delayed starts,
- a broad evaluation framework.

Review focus:
- whether the new runner mode actually exercises the intended integration path,
- minimal disruption to existing workflows,
- clarity of the configuration/CLI surface,
- usefulness of the resulting logs and summaries.

### Task 4 — Extract shared inserted-track metadata helper

Scope:
- factor constructor-time initial-track setup, internal-birth insertion, and external-start insertion
  through a shared helper where appropriate,
- preserve current behaviour,
- add/update checks that verify consistent metadata initialisation.

Review focus:
- consistency of `track_id`, counters, `last_det_key`, `last_det_hit`, and `assoc_history`,
- minimal behavioural churn,
- readability.

### Task 5 — Make operating modes explicit and testable

Scope:
- make external-only / internal-only / both configuration paths explicit,
- ensure runner/config surface is clear,
- align the new scenario-runner mode with the intended tracker operating modes,
- add smoke-style coverage for each mode where practical.

Review focus:
- configuration clarity,
- no hidden coupling,
- accurate docs/comments.

### Task 6 — Refactor internal birth pipeline into explicit stages

Scope:
- split the current internal birth path into clearer helper stages,
- preserve behaviour as closely as practical,
- keep instrumentation intact,
- add/update checks around ranking/filtering/branching behaviour where feasible.

Review focus:
- readability,
- stable semantics,
- easier future modification,
- no accidental redesign disguised as refactor.

### Task 7 — Documentation sync after implementation lands

Scope:
- update current-state doc to describe the implemented external-start capability,
- tighten any roadmap / context wording if needed,
- ensure examples and assumptions match the actual public API,
- ensure runner usage notes match the delayed external-start workflow.

Review focus:
- docs match code,
- no overclaiming,
- no stale API wording.
