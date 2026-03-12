# TO-MHT next steps (current phase): External initiation + birth handling cleanup

This document lists the upcoming tasks for the current phase.
It intentionally focuses on what is next from the current baseline, rather than repeating already-completed work.

## 1. Why this is the next phase

The current tracker already has:
- stable detection ordering,
- beta-ratio v1.5 scoring,
- association history with N-scan-lite deduplication,
- and useful instrumentation.

The main immediate gap is **initiation / birth handling**:
- the current internal-birth path is still heuristic,
- and the first realistic integration target needs **external track initiation**.

For the near term, we want the tracker to work cleanly in two modes:

1. **External-initiation mode (priority)**
   - upstream code provides new track starts,
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

### 2.2 Internal and external births should share a common conceptual interface

Even if they come from different sources, both should look like “birth candidates” to the global-hypothesis logic.
That keeps the tracker generic and avoids application-specific branching in the core algorithm.

### 2.3 Birth logic should become easier to reason about

The current code is useful, but still mixes:
- residual-based birth discovery,
- birth ranking,
- compatibility checks,
- and branching policy.

This phase should make those pieces more explicit.

## 3. Proposed immediate implementation direction

### 3.1 Add an explicit external-birth candidate interface

Introduce a small wrapper object, e.g. `ExternalBirthCandidate`, with fields along the lines of:

- `track: Track`
- `support_detections: Iterable[Detection] | None = None`
- `log_delta: float | None = None`
- optional metadata / source label if useful for debugging

Rationale:
- if upstream code knows which current-scan detections support the birth, the tracker can map them to scan-local detection keys and enforce compatibility,
- if no support detections are available, the tracker can still accept the birth as an exogenous start.

### 3.2 Extend the tracker interface to accept external births per scan

Add an explicit way to pass external birth candidates into the tracker during `step()`.

Suggested direction:
- extend `step(...)` with an optional `external_births` argument,
- keep this optional so existing callers still work,
- document clearly when the external births are considered relative to normal scan processing.

Initial behaviour is allowed to be simple:
- external births enter at the birth-branching stage of the current scan,
- they are treated similarly to internal births,
- and they are scored either via an explicit `log_delta` override or the existing birth score fallback.

### 3.3 Refactor birth handling around a common birth-candidate path

Refactor the current internal-birth code so that:
- internal initiator output becomes one source of birth candidates,
- external births become another source,
- and the downstream branching code works on a shared candidate representation.

This should make it easy to:
- run internal-only,
- external-only,
- or mixed experiments if needed.

### 3.4 Make internal births easy to disable

Add a clear configuration path so the tracker can run in:
- external-initiation-only mode,
- internal-birth-only mode,
- or both.

For the first ISAC integration step, external-initiation-only mode is likely the default.

## 4. Birth handling cleanup in this phase

This phase does **not** need a fully principled birth/existence model.
But it should make the current behaviour more controlled and understandable.

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

- The tracker can accept externally created tracks during `step()`.
- External births can be run with internal births disabled.
- If support detections are provided, they are used for compatibility / exclusivity checks.
- If support detections are not provided, the tracker still behaves sensibly.

### Internal-birth path

- Existing standalone scenarios still run.
- Internal birth behaviour is at least as understandable as before.
- Birth instrumentation still reports meaningful scan/run summaries.

### Documentation

- Roadmap and chat-context docs reflect the new current phase.
- The current-state doc explains the external-initiation capability once implemented.

## 6. Deferred for later phases

Not part of this phase:
- explicit shared hypothesis trees,
- true ancestor-based N-scan pruning,
- replacing beta-ratio v1.5 scoring,
- large-scale performance optimisation,
- full evaluation / benchmarking framework.
