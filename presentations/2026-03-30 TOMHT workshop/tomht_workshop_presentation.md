---
marp: true
paginate: true
theme: default
title: TOMHT for ISAC Workshop
---

# TOMHT for ISAC - workshop 2026-03-30

Patrik Andersson
SafeRadar Research AB


---

# Outline of this session

- TO-MHT vs "simple" trackers (e.g., GNN) 
- TO-MHT data structures and update flow in general
- Current state of our implementation
- Integration
- Next steps

---

# Why MHT at all?

- Trackers need to explain the world in terms of associations:
  - Which detections originate from which target?
  - Which detections are clutter / false alarms?
- Simple trackers like GNN keep one current explanation of the world
- MHT keeps multiple plausible explanations alive over time
- This is useful when ambiguity cannot be resolved immediately
  - Dense scenarios, high clutter, large measurement errors, ...
- The price is more complexity and computational load
  - More state, more branching, more need for pruning/commitment, ...

---

# Design goals for our implementation

- Proper TO-MHT implementation
- Stone Soup based
  - Use standard Stone Soup APIs, without special hacks or assumptions
  - Should allow integration in any Stone Soup compliant environment
  - Abstracts away all dependencies on measurement and transition models, etc
- Support ISAC use case
- Code should be easy to read and work with
- Prioritize clean architecture and code over raw performance
  - Algorithmic optimizations to keep branching under control still essential
  - Some code-level optimization to remove obvious inefficiencies is fine

---

# Public usage pattern for a simple Stone Soup tracker

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
for (time, tracks) in tracker:
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

# Public usage pattern for our TO-MHT tracker

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
- They enter the same data structures as internal births

---

# Conceptual core of TO-MHT

### TO-MHT as track trees + global hypotheses

- A TO-MHT keeps multiple plausible association histories alive over time
- Conceptually, each logical track is a **track tree** (aka family)
- Each node in that tree corresponds to one local association decision at one scan
- A **global hypothesis** selects one active leaf from each track tree, in a way that is globally consistent between tracks, including history:
  - no detection is used twice
  - clutter / births are handled consistently

---

# Track trees (or families) + global hypotheses

![Track trees](./track_trees.jpg)

---

# Core TO-MHT pattern

This “local branching + global consistency” split is the core TO-MHT pattern.

- Ambiguity is stored **inside each track** as branching history
- Joint consistency is handled **across tracks** by global hypotheses
- Shared history is represented by parent links

---

# Conceptual core data structures

### Measurement / scan
- one batch of detections at one timestamp

### Track-hypothesis node
- one state estimate after one local association decision
- linked to its parent in the same logical track

### Track tree
- all competing histories for one logical target

### Global hypothesis
- one consistent joint choice of active leaves across trees

---

# Conceptual track-hypothesis node

Typical contents of a node:

- logical track identity
- parent pointer
- child pointers
- scan index / timestamp
- state estimate
- association type
  - detection, missed detection, birth / start
- measurement reference (if any)
- local score contribution

---

# Conceptual track tree

Typical contents:

- logical track identity
- root pointer
- leaf pointers
- metadata

Interpretation: 
- one tree per logical target
- leaves are the active competing hypotheses
- parent/child structure captures association history

---

# Conceptual global hypothesis

Core contents:

- selected leaf for each track
- log weight

---

# Conceptual update step

At a high level, one TO-MHT scan update looks like:

1. predict active leaves to the new timestamp
2. generate local child hypotheses for feasible associations + miss
3. combine local choices into globally consistent hypotheses
4. score and prune
5. apply delayed decision logic (for example N-scan)
6. apply confirmation/deletion logic
7. produce output tracks

---

# Generate child hypotheses inside each track tree

After prediction and gating, each active track leaf branches into alternatives:
- matched to detection 1
- matched to detection 4
- missed
- ...

For each alternative, update state and create a child node.

Result: every track tree gets a new frontier of child leaves at the current time step.

Note: At this point, we have not yet enforced that the same detection cannot be used by two different tracks globally.

---

# Create birth hypotheses for unexplained detections

Detections that are not assigned to an existing track in a given global hypothesis may represent:

- clutter / false alarm
- new target birth

So for relevant detections, create tentative new-track roots or first nodes.

In practice, these choices are usually folded into the later global association step.

---

# Build global hypotheses from consistent combinations

Form **globally consistent combinations** such that:

- each existing track selects at most one child branch
- each detection is used at most once
- unassigned detections are explained as clutter or births
- optional extra logic

Result: a pool of global hypotheses

This is often posed as a ranked assignment / multidimensional consistency problem.

---

# Score the new global hypotheses

Each new global hypothesis gets a score or log weight from:

- parent global score
- track update likelihoods
- missed-detection penalties
- clutter model
- birth model
- optional existence probabilities / priors

---

# Prune aggressively

Without pruning, MHT explodes.

Common pruning steps:

- keep only top `K` global hypotheses
- prune low-scoring global hypotheses below threshold
- prune dominated local branches not used by any surviving global
- merge nearly identical states
- apply N-scan pruning / deferred decision logic

This step is central to making the tracker practical.

---

# Apply N-scan logic and resolve old ambiguity

If using N-scan pruning:

- look back `N` scans
- find branch decisions that are now effectively common across the surviving best globals
- collapse older ambiguity
- promote stable prefixes to confirmed track history

So the tracker keeps ambiguity only in a moving recent window.

---

# Update track status

Based on the surviving hypotheses, update lifecycle state:

- tentative -> confirmed
- confirmed -> coasted
- coasted too long -> deleted

One may also maintain per-track stats like:

- hit count
- consecutive misses
- track existence probability
- age
- confirmation score

---

# Produce output tracks

Finally, derive the user-facing track set, usually from:

- the single best global hypothesis, or
- a weighted combination of several globals

Output each selected track’s current state and maybe its smoothed history.

---

# Summary

One can think of one update as:

- **predict** old leaves forward
- **branch** each track on all plausible explanations
- **pack** those branches into a small set of globally consistent worlds
- **prune** almost everything
- **extract** user-facing track set

---

# What the current implementation stores

### `TrackHypothesisNode`
One logical-track hypothesis at one scan step.

Carries:
- same-track parent link
- state payload
- association choice / used detection identity
- local score contribution
- cached counters such as hits / misses / age
- provenance such as internal birth vs external start

---

# What the current implementation stores, cont.

### `GlobalHypothesis`
One global hypothesis, i.e., one consistent joint choice of active leaves across trees.

Carries:
- one active leaf per logical track
- cumulative log weight

---

# Our current update pipeline

1. sort detections deterministically
2. expand each current global
3. apply unused-detection score term
4. collapse exact structural duplicates
5. beam prune
6. update N-scan commitment
7. apply internal births
8. produce MAP output

This is the current implementation-level version of the conceptual TO-MHT update.

---

# Continuation / expansion in our implementation

### Local per-track continuation
- reconstruct temporary Stone Soup `Track`
- hypothesise and score local alternatives
- keep bounded number of child candidates
- always keep a miss if present

### Joint global assembly
- combine local candidates across tracks
- enforce detection exclusivity
- drop track hypotheses whose miss budget is exceeded
- accumulate resulting global log weights

---

# Dedupe, beam, and commitment

After expansion:
- globals are adjusted for detections left unused
- exact structural duplicates are collapsed
- top globals survive beam pruning

Then:
- N-scan commitment is computed from the surviving globals
- commitment is per-track
- ancestor identity at boundary `b = k - N` is the criterion

---

# Internal births in the current implementation

Internal births are handled as a separate phase after the main continuation pipeline.

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

### Shared structure
- both enter the same node-based hypothesis system

### Different semantics
- external starts are already confirmed
- internal births are tracker-side initiation logic
- they should not be thought of as the same thing operationally

For ISAC, only external starts will be used, at least for the time being.

---

# Summary of the current implementation

- explicit node-based TO-MHT structure
- globals reference leaf nodes
- structural dedupe
- explicit ancestor-based N-scan commitment
- conservative post-commit ancestry cleanup
- external confirmed starts supported
- API is intended be stable

---

# What needs to be improved

### Intentionally simple / currently pragmatic
- Scoring details
  - currently simplified and/or based on heuristics
- Internal birth handling
- Track trees are currently implicit, no track tree data structure
- N-scan pruning implemented, other aspects of pruning largely missing

### Still future work
- Proper lifecycle / deletion
- Multi-sensor miss policy
- Broader scaling / performance optimization

---

# What is out of scope

In a real system, one might do things like:

- store nodes in arrays/arenas, not pointer-heavy objects, for cache efficiency
- use integer IDs instead of raw pointers
- represent measurement usage with bitsets
- ...

These kinds of optimizations seem out of scope for the current Python implementation.

---

# What to expect from the initial release

- a conceptually clean TO-MHT core
- a stable Stone Soup-facing API
- external-start support for ISAC integration
- enough maturity to start integration and validation/evaluation of usefulness
- a good foundation to iterate from
- still much work left to do, on multiple fronts

---

# Test scenarios

- A few simple synthetic scenarios included, inherited from Stone Soup MFA example
  - Was very useful to get started
- Still useful
  - Based on different variants of underlying Stone Soup components
  - Small and simple, so analyzing details is tractable
  - Each has some challenging aspects
    - targets starting close to each other, crossing targets, high clutter
  - Also used as a quick smoke test
- Could be extended and tweaked

---

# Integration with SafeRadar replay pipeline

- SafeRadar detection data has been integrated with this TO-MHT implementation
- Done via a Stone Soup adapter
  - translates between SafeRadar and Stone Soup representations of detections and tracks
- Allows evaluation of TO-MHT tracking using existing tools
  - Foxglove
  - Matlab/Python

---

# Quick tour of the code in VS Code + demo

- VS Code
  - Folder structure
  - Main files and classes
  - Test scenarios
- Demo
  - Test scenarios
  - SafeRadar integration

---

# ISAC integration: Assumptions

- The TO-MHT tracker will be used only for system tracks, at least initially
- No internal births / track starts
- External track starts are generated from unused detections
  - via existing sensor trackers + deghoster, outside the scope of the TO-MHT
- All ambiguities (u/v + doppler) are handled transparently by the hypothesiser
  - given a track prediction, resolution is trivial
- IMM handled inside hypothesiser + updater, transparent to the TO-MHT tracker
- Detections carry everything the hypothesiser + updater need for multi-sensor support
- Multiple sensors (TX-RX links) at the same timestamp are handled via successive `tracker_update` calls, each containing detections from one sensor

---

# ISAC handoff flow

Current plan:
- maintain working development repo on our side
- export snapshots to a dedicated handoff/export repo
- mirror export repo into the Ericsson environment
- likely mirror code into the sandbox folder of the ISAC repo
- first code handoff planned after the workshop

---

# Next steps after first code handoff

### Ericsson

- Start working on being able to swap in `TOMHTTracker` as system tracker
- Report any issues that come up in integration
- Try running it on a few recorded scenarios

### SafeRadar

- Fix integration issues as they come up
- Keep iterating on the implementation
- Likely near-term themes:
  - explicit track trees, refined scoring and miss models, improved lifecycle / pruning / deletion behavior, ...
