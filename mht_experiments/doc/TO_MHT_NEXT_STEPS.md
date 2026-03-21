# TO-MHT Next Steps

## Current near-term phase

**Phase C: integration-readiness and interface cleanup**

Phase B's main structural goals are now in code:
- globals store `track_id -> leaf node`,
- ancestry is explicit,
- dedupe is structural,
- N-scan commitment is ancestor-based and per-track,
- external starts and internal births share the same node-based structure.

The current priority is therefore practical integration and code/interface cleanup rather than another large architectural rewrite.

## Working checklist

This document is now intended as a working checklist for the integration-readiness phase.
It should be allowed to accumulate practical notes, follow-up findings, and small decisions as the work progresses.

### 1. Stone Soup compliance / tracker interface

Primary goal:
- make `TOMHTTracker` easier to use as a Stone Soup-style tracker without changing its current external behavior more than necessary.

Status (2026-03-21):
- this checklist item is now functionally complete in code.

Implemented:
- `(_TrackerMixInUpdate, Tracker)` inheritance is in place,
- `update_tracker(time, detections)` delegates to the scan pipeline and returns `(time, tracks)`,
- `tracks` property returns current MAP output tracks,
- small public inspection helpers now exist (`get_map_hypothesis_snapshot()`, `get_map_output_tracks()`, `get_n_scan_commitment_snapshot()`, `get_unused_detections()`).

Remaining scope for item 1:
- keep this public API stable while doing items 2 and 3,
- defer broader wording/style cleanup to item 4 (readability/documentation pass).

### 2. Hypothesiser / scoring dependency check

Primary goal:
- understand whether the current tracker/scoring path depends too strongly on `PDAHypothesiser`-style assumptions.

Questions to answer:
- what assumptions does the current default scoring path make about the hypothesiser,
- are those assumptions acceptable for local replay integration,
- are those assumptions acceptable for ISAC integration,
- if not, is the likely remedy a custom scoring model, a wrapper, or a broader scoring cleanup task later.

This item is primarily about identifying risk, not immediately redesigning scoring unless integration forces it.

### 3. Local replay-data integration and validation

Primary goal:
- get TOMHT working with local radar replay data using a Stone Soup-style hypothesiser/updater path that you can test independently.

Intended benefits:
- validate real integration assumptions before relying on external environments,
- surface timestamp / detection / updater / hypothesiser mismatches early,
- build confidence in the tracker on data you control.

This is expected to be driven mainly outside this document, with notes added here only when they affect design or priorities.

### 4. Code-level readability / cleanup

Primary goal:
- make the implementation easier to review, explain, and present.

Likely items:
- tighten docstrings and comments around the node model, MAP views, and N-scan commitment timing,
- clarify compatibility boundaries (reconstructed `Track` views, `track_metadata`, legacy parameter/defaulting behavior),
- simplify or rename helpers where that materially improves readability,
- keep cleanup disciplined and presentation-driven rather than open-ended.

### 5. Export / packaging / handoff flow

Primary goal:
- establish a controlled way to deliver TOMHT snapshots into the ISAC-facing environment.

Current working idea:
- create a dedicated export repo,
- add an export script that writes a clean snapshot into that repo,
- optionally include revision info / what-changed notes alongside each export,
- define a simple mirroring or handoff flow into the ISAC environment.

This checklist item is as much about process clarity as code.

### 6. Workshop preparation support

Primary goal:
- keep the code and architecture in a state that will be easy to present in the ISAC workshop.

Likely needs:
- identify which parts of the architecture should be presented,
- identify which code sections are readable enough to show directly,
- note any rough edges that should be polished before the workshop,
- defer actual presentation-material work until closer to the end of the coming week.

## Things to keep in mind during this phase

- avoid reopening broad algorithmic work unless integration exposes a concrete blocker,
- keep Phase B structural gains stable,
- prefer small practical improvements over another large refactor,
- use local replay integration to learn what actually matters before reprioritising later phases,
- update `CURRENT_STATE` and `ROADMAP` again if the integration phase reveals something materially new.

## Still-deferred topics

These remain real topics, but are not the primary focus of the current phase unless integration forces them forward:
- broader scoring redesign,
- principled existence modelling,
- deeper internal-birth cleanup,
- committed-history materialisation,
- node garbage collection / ancestry compaction,
- performance optimisation.
