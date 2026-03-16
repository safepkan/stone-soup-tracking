# TO-MHT Roadmap

This roadmap is intentionally forward-looking. It reflects the revised priorities after completing the external-initiation and birth-pipeline cleanup phase.

## Guiding principle

The current tracker is a solid experimental and integration baseline: deterministic, inspectable, capable of maintaining externally initiated tracks, and usable in the existing scenario runners. The main remaining gap is structural: it still represents global hypotheses as copied per-track `Track` objects, and its current N-scan behavior is only an approximation of proper TO-MHT commitment.

The next priority is therefore to make the tracker structurally correct as a track-oriented MHT before spending effort on more refined scoring or existence modeling.

## Phase B — Explicit track-hypothesis structure and true N-scan pruning

### Goal

Replace copied tracks in flat global hypotheses with explicit shared per-track hypothesis ancestry, and replace the current history-tail N-scan approximation with true ancestor-based N-scan commitment/pruning.

### Why this comes next

- It is the biggest remaining gap between the current prototype and a proper TO-MHT.
- It should make both the implementation and the ISAC-facing discussion much cleaner.
- It gives later scoring/existence work a correct structural foundation.

### Intended outcome

- Each logical track is represented by a chain/tree of hypothesis nodes rather than copied whole-track objects in each global.
- A global hypothesis points to current leaf nodes for each active logical track.
- Shared ancestry is explicit.
- N-scan pruning operates on explicit ancestor identity after beam pruning, not on recent association-history heuristics.
- External starts and internal births remain semantically distinct, but enter the structure through the same node-based representation.

## Phase C / Phase D — Scoring refinement and birth/existence cleanup

The ordering between these later phases is intentionally left flexible.

Two clearly important topics remain after Phase B:

### Scoring refinement

Potential directions include:
- cleaner decomposition of hypothesis score contributions,
- more explicit existence / survival interpretation,
- better alignment with TO-MHT-style scoring semantics,
- reduced reliance on pragmatic heuristics such as the current beta-ratio scoring choices.

### Birth / existence cleanup

Potential directions include:
- more principled internal birth treatment,
- cleaner distinction between confirmed track starts and tentative/candidate starts,
- better existence-state handling for internally born tracks,
- revisiting whether internal births should remain part of the core tracker or be deemphasized for the ISAC-facing workflow.

The likely priority between these depends on the near-term usage pattern:
- if the ISAC integration is primarily external-start driven, scoring cleanup may come first,
- if internal-birth usage remains central, birth/existence cleanup may deserve priority.

## Longer-term cleanup / enhancement topics

These are real topics, but are not currently phase-defining:

- richer external-start scheduling beyond a single resolved injection scan,
- optional support for pre-first-step external starts via an implicit empty initial update,
- more explicit committed-track materialisation once true N-scan exists,
- performance / memory optimisation once the explicit node structure is in place,
- expanded scenario coverage and more principled regression harnesses.

## Near-term priority summary

1. **Phase B:** explicit track-oriented structure + true N-scan pruning.
2. **Then:** choose between scoring refinement and birth/existence cleanup based on actual integration needs.
3. **Later:** broader interface and performance improvements once the structure is correct.
