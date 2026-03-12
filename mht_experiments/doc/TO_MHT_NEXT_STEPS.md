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

### 2.3 Birth logic should become easier to reason about

The current code is useful, but still mixes:
- residual-based birth discovery,
- birth ranking,
- compatibility checks,
- and branching policy.

This phase should make those pieces more explicit.

## 3. Proposed immediate implementation direction

### 3.1 Add an explicit external-start interface

Add a separate tracker call for injecting externally created starts after a scan update, for example along the lines of:

- `step(detections, timestamp)`
- `add_external_starts(starts, timestamp, mode="confirmed")`

Initial scope:
- support **confirmed** external starts only,
- require the timestamp to match the most recent `step()`.

### 3.2 Treat external starts as already-initialised system tracks

For the first version, externally supplied starts should be assumed to:
- already be in the system-track state space,
- already be initialised at the given timestamp,
- already reflect any upstream ambiguity resolution or correlation logic.

TO-MHT should therefore insert them as new tracks into the current global hypotheses,
rather than trying to re-derive them from current detections.

### 3.3 Keep the internal birth path separate for now

Do **not** immediately force internal births and external starts through a shared public abstraction.

Instead:
- add a clean external-start path first,
- keep internal births as the existing standalone initiator-driven path,
- and revisit a shared internal/external birth abstraction later if it still looks useful.

### 3.4 Make external-initiation-only mode easy

Add a clear configuration path so the tracker can run in:
- external-initiation-only mode,
- internal-birth-only mode,
- or both.

For the first ISAC integration step, external-initiation-only mode is likely the default.

#### Assumptions for the first ISAC integration

For the first integration step, externally supplied starts are assumed to:
- already be initialised in global/system coordinates,
- already have any start-time ambiguity resolution applied upstream,
- already correspond to the current system-tracker timestamp,
- and be intended as confirmed new system tracks.

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

### 4.2 Keep birth control simple for now

If birth pressure still looks problematic after the refactor, add only lightweight control measures, such as:
- limiting births to top-ranked parent globals,
- or making the maximum number of compatible birth branches more explicit.

Do **not** turn this phase into a large tuning exercise.

## 5. Acceptance criteria for this phase

### External-initiation path

- The tracker can accept externally created starts after `step()`.
- Confirmed external starts can be used with internal births disabled.
- Externally supplied starts are inserted with the correct timestamp and internal metadata initialised.
- Existing standalone scenarios still run unchanged when the external-start path is unused.

### Internal-birth path

- Existing standalone scenarios still run.
- Internal birth behaviour is at least as understandable as before.
- Birth instrumentation still reports meaningful scan/run summaries.

### Documentation

- Roadmap and chat-context docs reflect the new current phase.
- The current-state doc explains the external-initiation capability once implemented.

## 6. Deferred for later phases

Not part of this phase:
- soft external birth candidates,
- per-start support-detection handling,
- common internal/external birth-candidate abstraction,
- explicit shared hypothesis trees,
- true ancestor-based N-scan pruning,
- replacing beta-ratio v1.5 scoring,
- large-scale performance optimisation,
- full evaluation / benchmarking framework.
