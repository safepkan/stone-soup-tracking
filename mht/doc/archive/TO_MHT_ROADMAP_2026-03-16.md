# TO-MHT Roadmap

This document gives the big-picture view of where the project is now and where it is going next.
It is intentionally forward-looking: the goal is to make the current baseline and the next major design choices easy to understand.

## 1. Goal and design principles

### Goal

Build a **clear, general Track-Oriented Multi-Hypothesis Tracker (TO-MHT)** on top of Stone Soup.

The end goal is a tracker that:
- behaves like a proper TO-MHT,
- is readable enough to support research and experimentation,
- can be plugged into different Stone Soup-compatible motion / measurement / prediction / update setups,
- and can be used to compare TO-MHT against existing trackers in application-specific pipelines.

### Design principles

- **Clarity over performance.** Readable, inspectable code matters more than raw runtime for now.
- **Stone Soup for model-specific plumbing.** Reuse Stone Soup predictors, updaters, hypothesiser objects, and related components wherever possible.
- **Generic tracker core.** Keep TO-MHT logic independent of any specific motion model, measurement model, or sensing modality.
- **Incremental path to “proper TO-MHT”.** It is acceptable to implement lightweight approximations first, as long as they do not paint us into a corner.

## 2. Application context that affects priorities

A key target use case is a research setup for **Integrated Sensing and Communication (ISAC)** in 6G mobile networks.

At a high level:
- each TX-RX link between two 6G base stations acts like a sensor in a multi-sensor bistatic radar setting,
- angle measurements are ambiguous due to antenna-array spacing,
- the current system does simple per-sensor tracking in ambiguous sensor coordinates,
- then correlates across sensors to resolve ambiguities and start system tracks in global coordinates,
- and finally updates those system tracks using sensor detections.

### What this means for TO-MHT priorities

The first realistic integration target is **not** a full end-to-end replacement of the whole pipeline.
Instead, the first target is likely:
- keep the existing sensor trackers,
- keep the current cross-sensor correlation / ambiguity-resolution logic for track starts,
- replace the current **system tracker** with TO-MHT.

This means the tracker should support two operating modes:

1. **External-initiation / system-tracker mode**
   - new system tracks are supplied by upstream code,
   - TO-MHT focuses on multi-hypothesis maintenance, ambiguity handling, clutter robustness, and track continuity.

2. **Standalone / internal-birth mode**
   - TO-MHT uses its own birth logic / initiator for self-contained experiments.

The external-initiation path is the more immediate priority.

## 3. Current baseline

The current codebase already includes a usable experimental baseline:

- stable per-scan detection ordering and scan-local detection keys,
- `ScoringModel` abstraction with **beta-ratio v1.5** scoring,
- per-track association history plus **history-tail deduplication**,
- an **N-scan-lite** approximation based on recent association history rather than explicit tree ancestors,
- structured scan and run instrumentation (`ScanStats`, `BirthStats`, summary metrics),
- internal birth handling via a Stone Soup initiator plus heuristic filtering / ranking / branching.

This is **not yet a proper TO-MHT** in the full structural sense, but it is a good experimental platform:
- deterministic,
- inspectable,
- easy to run on multiple Stone Soup-compatible scenarios,
- and good enough to reveal where the next design effort should go.

## 4. What is still missing for a “proper TO-MHT”

The main remaining conceptual gaps are:

1. **Explicit shared hypothesis-tree structure**
   - current globals contain copied `Track` objects rather than explicit shared tree nodes.

2. **True ancestor-based N-scan pruning**
   - current N-scan-lite works by deduplicating on recent association history,
   - not by tracking common ancestors and committing branches older than N scans.

3. **Cleaner initiation / existence modelling**
   - current internal birth handling is useful, but still heuristic and somewhat opaque,
   - especially when compared to a clean TO-MHT existence / birth model.

4. **More explicit scoring model**
   - beta-ratio v1.5 is a practical bridge,
   - not yet a final, fully explicit TO-MHT likelihood model.

## 5. Near-term and medium-term phases

### Phase A — External system-track initiation (next)

**Goal:** make the tracker easy to integrate into an external pipeline where new system tracks are created upstream.

Key tasks:
- add a clean interface for **confirmed external starts**,
- support running the tracker with **internal births disabled**,
- document the assumptions for externally supplied starts (already initialised, current timestamp, system-track state space),
- keep compatibility with standalone internal-birth experiments.

Notes:
- For the first ISAC integration step, external starts are treated as confirmed upstream decisions, not as soft birth candidates.
- A more general external-candidate interface and any common internal/external birth abstraction are deferred until the integration semantics are clearer.

### Phase B — Proper track-oriented structure and true N-scan pruning

**Goal:** move from the current “copied tracks in flat globals” representation toward a proper TO-MHT representation with explicit ancestry.

Key tasks:
- introduce explicit shared track-hypothesis nodes / trees (or DAG-like sharing where appropriate),
- implement ancestor-based N-scan pruning,
- make commitment / branch-merging semantics explicit rather than implicit via history-tail dedupe.

This is the main structural step toward a proper TO-MHT.

### Phase C — Scoring and existence-model refinement

**Goal:** improve the probabilistic interpretation once the tracker structure and integration path are clearer.

Possible work:
- move beyond beta-ratio v1.5 toward a more explicit likelihood model,
- refine clutter, birth, and existence terms,
- make external-initiation evidence easier to score in a principled way.

This is important, but it does not have to block the next integration-facing work.

### Phase D — Scaling and experiments

**Goal:** make the tracker more useful on harder scenarios and larger problem sizes.

Possible work:
- clustering / partitioning before global expansion,
- k-best assignment style approximations instead of exhaustive backtracking,
- richer scenarios and evaluation metrics,
- systematic comparisons against existing baselines.

## 6. Current recommendation

Treat the tracker today as a **good experimental TO-MHT platform**, not yet the final TO-MHT.

The next step should be:
- **clean external initiation support and birth handling**,
- then move toward **explicit trees and proper N-scan pruning**,
- while keeping the code readable and generic enough for cross-application use.
