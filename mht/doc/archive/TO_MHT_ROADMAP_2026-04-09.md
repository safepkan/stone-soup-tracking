# TO-MHT Roadmap

This roadmap is intentionally forward-looking. It reflects the revised priorities after completing the explicit track-oriented structure phase.

## Guiding principle

The tracker is now in a much better architectural state than when this roadmap was first drafted: global hypotheses reference explicit per-track leaf nodes, shared ancestry is structural, and N-scan commitment is based on ancestor identity rather than association-history tails.

The near-term priority is therefore no longer “make the tracker structurally correct as a TO-MHT.” That Phase B work has been completed. The next priority is to make the implementation easier to integrate, easier to explain, and easier to hand off into external Stone Soup-based workflows before choosing the next larger algorithmic phase.

## Phase B — Explicit track-hypothesis structure and true N-scan pruning

### Status

**Completed.**

### Outcome

Phase B established the structural core that was previously missing:
- each logical track is represented by explicit hypothesis ancestry rather than copied whole-track objects in each global,
- global hypotheses point to current leaf nodes for each active logical track,
- shared ancestry is explicit,
- dedupe is structural rather than history-tail-based,
- N-scan commitment is based on explicit ancestor identity after beam pruning,
- external starts and internal births share the same node-based structural system while retaining distinct semantics.

Physical node cleanup / GC, committed-history materialisation, and broader performance work remain deferred and are not considered part of Phase B completion.

## Phase C — Integration-readiness and interface cleanup

### Goal

Make the tracker easier to integrate into external Stone Soup-based environments, easier to review and present, and easier to hand off for practical use.

### Why this comes next

- There is an ISAC workshop in the near term where the architecture and selected code will be presented.
- The core tracker is now structurally sound enough that near-term leverage comes more from interface polish and integration validation than from further internal architectural rewrites.
- Practical integration work is likely to reveal which later algorithmic topics actually matter most.

### Likely focus areas

- make `TOMHTTracker` more Stone Soup compliant (for example via `Tracker` / `_TrackerMixInUpdate` style integration points),
- validate integration against local radar replay data before relying on external environments,
- check whether the current scoring path relies too strongly on PDA-style hypothesiser assumptions,
- improve code-level clarity, comments, and presentation readiness,
- establish a practical export / packaging / handoff flow for sharing snapshots with ISAC.

### Intended outcome

- the tracker can be dropped more cleanly into Stone Soup-style workflows,
- local replay integration is working well enough to give confidence before external integration,
- obvious interface/readability issues are cleaned up,
- current hypothesiser/scoring assumptions are understood well enough to know whether they are blockers,
- the code is in a better state for workshop presentation and external review.

## Phase D / Phase E — Scoring refinement and birth/existence cleanup

The ordering between these later phases remains intentionally flexible.

Two clearly important topics still remain after the structural and interface-focused phases:

### Scoring refinement

Potential directions include:
- cleaner decomposition of hypothesis score contributions,
- more explicit existence / survival interpretation,
- better alignment with TO-MHT-style scoring semantics,
- reduced reliance on pragmatic heuristics such as the current beta-ratio scoring choices,
- reduced coupling between tracker behavior and PDA-style hypothesiser assumptions if integration reveals that to be a real constraint.

### Birth / existence cleanup

Potential directions include:
- more principled internal birth treatment,
- cleaner distinction between confirmed track starts and tentative/candidate starts,
- better existence-state handling for internally born tracks,
- revisiting whether internal births should remain part of the core tracker or be deemphasized for the ISAC-facing workflow.

The likely priority between these depends on what the integration phase reveals:
- if the ISAC integration is primarily external-start driven, scoring cleanup may come first,
- if internal-birth usage remains central, birth/existence cleanup may deserve priority,
- if integration exposes hypothesiser/scoring compatibility issues, scoring cleanup may move forward sooner.

## Longer-term cleanup / enhancement topics

These are real topics, but are not currently phase-defining:

- richer external-start scheduling beyond a single resolved injection scan,
- optional support for pre-first-step external starts via an implicit empty initial update,
- more explicit committed-track materialisation once true N-scan exists,
- node lifecycle / ancestry cleanup and garbage collection built on the explicit commitment machinery,
- performance / memory optimisation once the explicit node structure is in place,
- expanded scenario coverage and more principled regression harnesses,
- track lifecycle / deletion policy cleanup, including possible Stone Soup Deleter integration.

## Near-term priority summary

1. **Phase B:** completed structural TO-MHT correction.
2. **Phase C:** integration-readiness, interface cleanup, local validation, and practical handoff support.
3. **Then:** choose between scoring refinement and birth/existence cleanup based on what integration actually reveals.
4. **Later:** broader lifecycle, performance, and output-materialisation improvements.
