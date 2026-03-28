---
marp: true
paginate: true
theme: default
title: TOMHT for ISAC Workshop
---

# TOMHT for ISAC - workshop 2026-03-30

Patrik Andersson
SafeRadar


---

# Outline of this session

- TO-MHT background
- TO-MHT implementation in this project
  - Public API
  - Implementation
    - Core data structures and algorithms
    - Current state vs where we're heading
- ISAC integration

---

# Why MHT at all?

- Trackers need to explain the world in terms of associations:
  - Which detections originate from which target?
  - Which detections are clutter?
- Simple trackers like GNN keep one current explanation of the world
- MHT keeps multiple plausible explanations alive over time
- This is useful when ambiguity cannot be resolved immediately
  - Dense scenarios, high clutter, ...
- The price is more complexity
  - More state, more branching, more need for pruning/commitment, ...

---

# What this implementation is

- TODO: Rewrite
- A Stone Soup-facing tracker
- Internally structured as a track-oriented MHT
- Global hypotheses reference per-track leaf nodes
- N-scan commitment is explicit and ancestor-based
- MAP tracks are exposed at the API boundary
- Supports externally confirmed track starts

---

# What it is not trying to solve yet

- TODO: Rewrite
- Final multi-sensor miss/termination policy
- Full lifecycle/deletion redesign
- Committed-history materialisation/output store
- Deep performance optimization
- Final ISAC-specific scoring policy

---

# Public usage pattern for a "simple" Stone Soup tracker

```python
hypothesiser = PDAHypothesiser(predictor, updater, ...)
data_associator = GlobalNearestNeighbour(hypothesiser)

tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
)

for (time, detections) in source:
    time, tracks = tracker.update_tracker(time, detections)
```

or pass in a detector iterator to the constructor and say

```python
    for (time, detections) in tracker:
        ...
```

---

# Stone Soup MultiTargetTracker implementation

```python
class MultiTargetTracker(_TrackerMixInNext, Tracker):
    ...

    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        time, detections = next(self.detector_iter)

        associations = self.data_associator.associate(
            self.tracks, detections, time)
        associated_detections = set()
        for track, hypothesis in associations.items():
            if hypothesis:
                state_post = self.updater.update(hypothesis)
                track.append(state_post)
                associated_detections.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)

        self._tracks -= self.deleter.delete_tracks(self.tracks)
        self._tracks |= self.initiator.initiate(
            detections - associated_detections, time)

        return time, self.tracks
```

---

# Public usage pattern for TOMHTTracker

```python
tracker = TOMHTTracker(
    hypothesiser=...,
    updater=...,
    initiator=...,
    params=...,
)

for (time, detections) in source:
    time, tracks = tracker.update_tracker(time, detections)
```

Note: No `data_associator` since we're not making a single set of associations at each time step.

---

# Alternate usage pattern, with external track starts

```python
tracker = TOMHTTracker(
    hypothesiser=...,
    updater=...,
    initiator=None,
    params=...,
)

for (time, detections) in source:
    time, tracks = tracker.update_tracker(time, detections)

    unused_det = tracker.get_unused_detections()
    ... # Somehow generate track_starts from unused_det
    tracker.add_external_starts(time, track_starts)
```

---

# External starts

- Intended for externally confirmed starts
- Fits the ISAC workflow where other components do:
  - sensor tracks
  - correlation
  - ambiguity resolution
  - confirmed system track starts
- Those starts are injected after processing that same timestamp
- They enter the same node-based structure as internal births

---

# TODO: A few slides explaining TO-MHT conceptually

- Figures from tutorials
- Tree structures
- Track trees
- N-scan pruning
- Pruning/merging

---

# TODO: Typical data structures

---

# TODO: Typical update flow

---

# TODO: Current data structures and update flow

- TODO: Review and update the slides below

---

# Core internal data structures

## `TrackHypothesisNode`

One logical-track hypothesis at one scan step.

Groups of fields:
- core structural identity / ancestry / per-step payload
- cached operational fields
- provenance / instrumentation
- small compatibility/boundary support where still needed

## `GlobalHypothesis`

- one active leaf per logical track
- cumulative log weight

---

# `TrackHypothesisNode` at a glance

## Core structural
- `node_id`, `track_id`, `parent`, `scan_index`
- `state`, `state_kind`
- `used_det_key`, `assoc_label`
- `log_delta`

## Cached operational
- `age`, `hits`, `missed_count`
- `last_det_key`, `last_det_hit`

## Provenance / debug
- `root_source`, `birth_scan_index`

---

# Why this structure matters

- Shared ancestry is structural, not copied
- Branch identity is node identity + parent links
- Globals no longer carry copied full `Track` history
- N-scan commitment can be based on explicit ancestor agreement
- Stone Soup `Track` objects remain the external boundary, not the internal truth

---

# Main update pipeline

1. Sort detections deterministically
2. Expand each current global
3. Apply unused-detection score term
4. Collapse exact structural duplicates
5. Beam prune
6. Update N-scan commitment
7. Apply internal births
8. Produce MAP output

---

# Continuation / expansion

## Local per-track continuation
- reconstruct temporary Stone Soup `Track`
- hypothesise and score local alternatives
- keep bounded number of child candidates
- always keep a miss if present

## Joint global assembly
- combine local candidates across tracks
- enforce detection exclusivity
- drop tracks whose miss budget is exceeded
- accumulate resulting global log weights

---

# Dedupe and beam

- Expanded globals are adjusted for detections left unused
- Exact structural duplicates are identified by:
  - `track_id -> leaf_node_id`
- Highest-weight global survives per signature
- Then normal beam pruning keeps the top globals

---

# N-scan commitment in this implementation

For scan `k` and window `N`:

- boundary is `b = k - N`
- commitment is computed after beam pruning
- commitment is per-track
- only globals that still contain a track participate for that track
- ancestor node identity at boundary `b` is the criterion

---

# Important N-scan clarification

Current implementation has:

- explicit logical commitment
- conservative post-commit ancestry cleanup

Current implementation does **not** have:

- committed-history materialisation
- broad lifecycle redesign
- full deletion policy cleanup

---

# Internal births

Birth phase is a separate bounded stage after the main continuation pipeline.

Sequence:
- identify residual detections
- generate initiator birth proposals
- sanity-filter them
- rank and limit them
- prepare root templates
- branch globals with those templates
- compute post-birth beam stats

---

# External starts vs internal births

## Shared structure
- both become root-like node entries in the same hypothesis system

## Different semantics
- external starts are already confirmed
- internal births are still tracker-side initiation logic
- scoring/policy expectations are not the same

---

# Current status

## What is strong now
- explicit node-based TO-MHT structure
- explicit ancestor-based N-scan commitment
- structural dedupe
- cleaner API story
- cleaner code organization/readability
- external confirmed starts supported

## What is still provisional
- multi-sensor miss handling / termination policy
- scoring assumptions for broader integration cases
- deeper performance optimization

---

# Integration 

TODO: Shift gears and talk about integration

---

# Test scenarios

- A few simple synthetic scenarios, inherited from Stone Soup MFA example
- Still useful
  - Different underlying measurement models
  - Close/crossing targets
  - High clutter
  - Also used as a quick smoke test
- Could be extended and tweaked

---

# Integration with SafeRadar replay pipeline

- SafeRadar detection data can be used with this TO-MHT implementation
- Done via a Stone Soup adapter
- Allows evaluation using existing tools
  - Foxglove
  - Matlab/Python
- Demo

---

# ISAC-specific integration questions

Likely discussion points:
- detections and hypothesiser assumptions
- updater/state compatibility
- external-start interface details
- multi-sensor misses and lifecycle policy
- what should live on our side vs their side

TODO: Update

---

# Handoff flow

Current plan:
- maintain working development repo on our side
- export snapshots to a dedicated handoff/export repo
- mirror that into the ISAC environment
- first code handoff planned after the workshop

---

# TODO: Expectations for the current version

- good enough for initial integration/handoff
- strong enough to explain and collaborate around
- not yet the final lifecycle/scoring/performance story
- likely to evolve once integration reveals real constraints

---

# TODO: Likely next technical themes

- multi-sensor miss / termination policy
- scoring cleanup once ISAC assumptions are clearer
- broader performance/scaling work if needed
- deeper integration-facing cleanup only where it buys something

---

# Discussion

- Which part of the workflow should own what?
- What are the main integration constraints on the ISAC side?
- What should the first handoff contain?
- What should we explicitly defer?
