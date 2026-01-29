# TO-MHT Roadmap

This document gives the big-picture view of where the project is going, without getting too deep into implementation details. It’s meant to be readable by someone who knows Stone Soup and tracking, but not necessarily this codebase.

## 1. Context and goals

- **Context:** Experiment with Track-Oriented Multi-Hypothesis Tracking (TO-MHT) using Stone Soup as the “plumbing” for motion/measurement models, prediction, and updating.
- **Primary goals:**
  - Understand TO-MHT end-to-end, from data association to pruning.
  - Build a simple, readable, flexible implementation suitable for experimentation.
  - Use Stone Soup components wherever possible (predictors, updaters, hypothesizers, initiators) so we can focus on MHT-specific logic.
- **Non-goals (for now):**
  - Production-grade performance or memory use.
  - Exhaustive algorithmic optimisations.
  - Supporting all Stone Soup models and scenarios.

## 2. Phases and milestones

### Phase 0 — Scaffolding and MFA baseline ✅

- [x] Identify relevant TO-MHT literature and notation.
- [x] Refactor Stone Soup MFA examples (crossing targets, bearing-range) into:
  - Shared scenario builders.
  - Reusable plotting/runner code.
- [x] Confirm refactored MFA produces identical behaviour to original examples.

### Phase 1 — TO-MHT v0.5: “Working but hacky” ✅

- [x] Implement a first global-hypothesis structure:
  - `GlobalHypothesis = {tracks_by_id, log_weight}`.
- [x] Implement per-scan expansion:
  - Per-track child hypotheses from a Stone Soup `Hypothesiser`.
  - Backtracking to ensure per-scan detection exclusivity.
  - Beam pruning: keep top-K global hypotheses.
- [x] Integrate births via a Stone Soup multi-measurement initiator:
  - Use unassigned detections (from top hypotheses) as birth candidates.
  - Rank births heuristically and limit to max per scan.
  - Penalise births in the global log-weight.
- [x] Add basic heuristics and instrumentation:
  - Unused-detection penalty.
  - Max-missed per track.
  - Debug flags, deterministic ordering, sanity checks for numeric blow-ups.
  - Simple deduplication of “presently identical” global hypotheses.

Result: a working multi-hypothesis tracker that produces sensible tracks on the test scenarios, but with a lot of shortcuts and ad-hoc scoring.

### Phase 2 — Scoring v2 and N-scan-lite (next) ⚙️

**Goal:** Move from heuristic scores toward a simple, explicit MHT log-likelihood model, and introduce N-scan-like commitment to stabilise track identities.

Planned tasks:

- Define a per-scan log-likelihood increment model that includes:
  - Hit likelihood: measurement vs predicted measurement under clutter + detection probability.
  - Miss likelihood: missed detection probability.
  - Clutter term for unused detections.
  - Simple existence/birth terms.
- Map this model onto Stone Soup’s available quantities (likelihoods from hypothesisers, clutter density, etc.).
- Implement a short per-track association history (e.g. last N detection indices).
- Implement N-scan-lite:
  - Commit association decisions older than N scans when surviving globals agree.
  - Optionally merge track trees when committed histories are identical.

### Phase 3 — Track initiation and existence modelling 🎯

**Goal:** Treat initiation and existence more cleanly inside the TO-MHT framework.

Possible directions:

- Option A: keep using the Stone Soup multi-measurement initiator, but:
  - Define a clear probabilistic interpretation of its output.
  - Integrate its “evidence” into the global log-likelihood more explicitly.
- Option B: replace external initiator with internal tentative tracks:
  - Maintain “tentative tracks” inside TO-MHT with their own hit/miss thresholds.
  - Promote to “confirmed” when sufficient evidence accumulates.

Also consider:

- Simple birth intensity model (e.g. uniform over volume with rate λ).
- More principled death/termination logic than just `max_missed`.

### Phase 4 — Refinement, experiments, and documentation 📈

- Evaluate behaviour on more challenging scenarios (crossing, coasting, low SNR, high clutter).
- Compare TO-MHT vs MFA baselines in scenarios where they differ.
- Improve visualisation:
  - Show multiple global hypotheses and their relative weights.
  - Show committed vs tentative parts of each track.
- Polish documentation:
  - Architecture doc stays up to date.
  - Reference doc links to papers and notes.

## 3. Current status (high level)

- MFA scaffolding: **done**.
- TO-MHT v0.5 prototype: **implemented and running**.
- Known limitations:
  - Scoring is heuristic and somewhat hand-tuned.
  - N-scan/commitment only approximated via present-state deduplication.
  - Initiation relies on an external Stone Soup initiator with heuristic integration.

Next up: **Phase 2 — Scoring v2 + N-scan-lite.**
