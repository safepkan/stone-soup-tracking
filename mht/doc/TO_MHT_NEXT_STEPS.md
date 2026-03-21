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

Current implementation note (2026-03-21):
- `PDAHypothesiser` conformance is sufficient but not strictly necessary at runtime; the tracker currently relies on a narrower duck-typed contract at the hypothesiser boundary.
Runtime hypothesiser contract used by `TOMHTTracker` today:
- `hypothesiser.hypothesise(track,detections,timestamp)` must return an iterable of per-track hypotheses.
- Each hypothesis must support hit/miss truthiness (`bool(hyp)`): miss is handled on the `False` path, hit on the `True` path.
- Miss hypotheses must expose `.prediction` (used directly as the propagated state).
- Hit hypotheses must expose `.measurement`, and that measurement must be the same object identity as one of the input detections for this scan (the tracker maps detections by `id(det)`).
- Hit hypotheses must be compatible with `updater.update(hypothesis)`.
- Practical stability expectation: each active track should produce at least one candidate (normally including a miss hypothesis), or expansion can collapse.
Additional assumptions in current default scoring mode (`scoring_mode="beta_ratio"`):
- Per-hypothesis `.probability` is consumed as a numeric-like association weight.
- Miss probability mass (`beta0`) is derived from miss hypotheses (identified by miss truthiness and/or `MissedDetection` measurement).
- Hypothesiser-level parameters are read opportunistically: `prob_detect`, `prob_gate`, and `clutter_density` or `clutter_spatial_density` (with fallbacks if missing).
Implication for ISAC integration:
- If ISAC hypothesiser matches the runtime contract and provides PDA-like probability semantics, it should work with current default scoring.
- If it matches only the runtime association contract (but not PDA-like probability semantics), integration is still possible via a custom `scoring_model` adapter on our side.
Typing note for later cleanup:
- Current type hints use `PDAHypothesiser`, but the actual runtime contract is broader; deciding whether to loosen type annotations to a Stone Soup base `Hypothesiser`/interface can be deferred.

Working assessment for integration planning (2026-03-21):
- Local replay integration is treated as unblocked by hypothesiser shape; we control that path and can select/adapt a hypothesiser to satisfy the current contract.
- ISAC integration hypothesiser compatibility is still to be confirmed with the ISAC side.
- Current planning assumption: hypothesiser mismatch is unlikely to block initial ISAC integration, because short-term adaptation on either side is likely feasible if needed.
- Therefore, item 2 is currently treated as "risk understood, monitor during integration" rather than "hard blocker before item 3."
- Scoring redesign/cleanup is still expected in the near future, so current compatibility work should prioritise pragmatic initial integration over overfitting to the current beta-ratio scoring details.

### 3. Local replay-data integration and validation

Primary goal:
- get TOMHT working with local radar replay data using a Stone Soup-style hypothesiser/updater path that you can test independently.

Intended benefits:
- validate real integration assumptions before relying on external environments,
- surface timestamp / detection / updater / hypothesiser mismatches early,
- build confidence in the tracker on data you control.

Status (2026-03-21):
- this checklist item is now functionally complete for the current phase,
- keep adding concise operational notes here when new insights affect design, validation method, or priorities.

Status notes (2026-03-21):
- First end-to-end local replay with `python.pipeline.batch_mcap_replay` and `--tracker-type stonesoup-mht` is producing usable output on a ~1 minute gate-approach aircraft scenario (multiple discrete aircraft-part point targets).
- Equality check method for replay runs:
  - run the same input file twice,
  - extract `/radar_matlab_replay/tracks-legacy` using `mcap cat --json --topics ...`,
  - compare JSON streams.
- Current replay equality result:
  - raw `tracks-legacy` JSON differs because `data.processing_time_ms` is runtime-dependent,
  - after ignoring `data.processing_time_ms`, extracted `tracks-legacy` JSON is identical across repeated runs.
- Practical regression method (preferred for now):
  - compare stdout instrumentation lines (`[Scoring]`, `Birth candidates`, `Births kept`, `Global hypotheses`, `logW=`, `SCAN`, `SCAN_NSCAN_COMMITTED`) between two runs.
  - In the current test, this instrumentation stream matched exactly across two repeated replays.
- Performance/stability observations:
  - intermittent "hang" reports are currently treated as likely long per-scan compute spikes rather than confirmed deadlock (combinatorial expansion can become large on some scans),
  - memory growth during run is expected with current implementation because node/history state is retained (node GC/compaction is still deferred); observed growth of a few hundred MB is consistent with this.
- Near-term validation follow-up:
  - when stalls are observed, collect per-scan wall-clock timing and correlate with `SCAN ... exp=...`, detection count, and birth activity to distinguish true hangs from expansion spikes.
- Instrumentation update and first timing check (2026-03-21):
  - added `SCAN_TIMING t=... wall_ms=...` per scan in TO-MHT debug instrumentation (kept existing `SCAN ...` line format unchanged for deterministic diffing),
  - one replay run showed clear timing outliers late in the scenario (example max around scan 254, where expansion was also high),
  - two-run comparison showed outlier *positions* are largely repeatable (top-10 overlap 8/10), while exact wall-ms values vary between runs as expected.
- Lightweight memory instrumentation and first run (2026-03-21):
  - added `SCAN_MEMORY t=... nodes=... leaf_inst=... maxrss_mb=...` per scan, plus `SUMMARY memory ...` aggregates,
  - first full replay (`--max-cpis 400`) showed monotonic growth from about `202 MB` to `364 MB` max RSS and from `1` to about `56k` retained nodes,
  - this supports treating node/history retention as a known scalability limit to address after initial integration.
- Test command reference (kept for this phase while input file remains available):
  - `cd ~/Git/l2-sp && source ./venv/bin/activate && python -m python.pipeline.batch_mcap_replay ~/Documents/MATLAB/cpi_replay_2025-12-10_173948.mcap --include-tracker --output-path /tmp --tracker-type stonesoup-mht --max-cpis 400`
  - for repeatability checks, fix output folder/log names with `--folder-name` and redirect stdout/stderr to a log file.

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
