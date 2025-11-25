# TO‑MHT Roadmap (Stone Soup)

This repo currently has a refactored **MFA (multi‑frame assignment)** implementation that’s useful as:
- a working harness (scenario generation + plotting + “run in script OR interactive window”), and
- a reference for integrating Stone Soup models (predictor/updater/measurement models/etc).

Next step: implement a **Track‑Oriented Multiple Hypothesis Tracker (TO‑MHT)** that reuses Stone Soup building blocks but replaces MFA’s internal optimiser/pruning with explicit **track trees + global hypotheses**.

---

## Goals / non‑goals

### Goals
- **Readable, simple, flexible** implementation suitable for experimentation.
- Reuse Stone Soup for:
  - transition / measurement models
  - predictors / updaters (KF/UKF/etc)
  - detection types and simulators
- Provide **two runnable setups** (crossing + bearing‑range) just like MFA, so TO‑MHT can be validated in both.

### Non‑goals (for first versions)
- High performance, scalability to 100s of tracks/detections.
- Fancy global solvers and aggressive pruning heuristics.
- Full track management (initiation/termination) sophistication.

---

## Proposed folder structure (mirrors MFA refactor)

Add:

```
mht_experiments/
  trackers/
    tomht_tracker.py
  runners/
    tomht_runner.py
  run_tomht_crossing.py
  run_tomht_bearing_range.py
```

Keep the **scenario modules** as-is:
- `scenarios/crossing_targets.py`
- `scenarios/bearing_range_mht_example.py`

Keep plotting helpers; add TO‑MHT-specific plotting only if needed.

---

## Conceptual architecture

TO‑MHT state is maintained as:
- A set of **track trees** (one tree per “track identity”).
- A set of **global hypotheses**, each a consistent selection of one leaf from each active tree.

In the “simple version”, we can represent a *track hypothesis leaf* as:
- a `stonesoup.types.track.Track` instance (a single path), plus metadata:
  - `track.metadata["parent"]`  (pointer or ID)
  - `track.metadata["log_weight"]` (cumulative score)
  - `track.metadata["track_id"]` (stable ID for plotting)
  - `track.metadata["last_meas_id"]` (optional)

Global hypothesis:
- a list/set of leaf tracks, one per track-tree/identity
- plus cumulative log weight
- plus bookkeeping (which detections are already used this scan)

---

## First-cut TO‑MHT algorithm (per scan)

Given detections Z_k and timestamp t_k, and a set of global hypotheses G_{k-1}:

1. **For each global hypothesis g in G_{k-1}:**
   - For each track leaf (one per track identity):
     - Use a Stone Soup **hypothesiser** to generate a `MultipleHypothesis`:
       - missed detection hypothesis
       - measurement-associated hypotheses (gated)
     - For each single hypothesis, create a **child leaf track**:
       - If miss: append prediction state
       - If hit: append update state
       - Update the leaf’s log score with an incremental term

2. **Form new global hypotheses (data association constraint):**
   - Choose one child leaf per track identity
   - Ensure no detection is assigned to more than one track (per scan)
   - (Optional in v1) births: start a new track leaf from an unassigned detection

3. **Prune**
   - Keep top-K global hypotheses by log weight
   - (Optional v1.5) limit children per track leaf (top-M per track)
   - (Optional v2) N-scan pruning / back-prune

4. **Output**
   - Return the MAP global hypothesis’s tracks as the current estimate set (or expose all global hyps if desired)

---

## Scoring (keep it simple first)

For v1, keep scores “consistent enough” rather than perfectly statistically calibrated.

Suggested approach (v1):
- Use the hypothesis’ probability/weight if Stone Soup provides it (often PDA-style hypotheses do).
- Convert to log-space and accumulate:
  - `log_w_new = log_w_old + log(max(prob, eps))`

Later upgrades (v2):
- Switch to classic MHT log-likelihood increments:
  - detection likelihood under measurement model
  - missed detection term log(1 - P_D P_G)
  - clutter spatial density term
  - track birth prior

---

## Pruning roadmap

### v1 (minimum)
- `max_global_hypotheses = K` (e.g., 20)
- `max_children_per_track = M` (e.g., 5) before global combination

### v2
- Relative score threshold from best (“beam search”)
- Track deletion if too many consecutive missed detections
- Simple N-scan commitment:
  - if all surviving global hyps share the same association at time k-N, commit and prune older branches

### v3
- Replace brute-force global combination with a solver:
  - ILP with OR-Tools *or*
  - maximum weight matching / MWIS formulation (depending on representation)

---

## Concrete incremental milestones

### Milestone 0 — plumbing & interfaces
- Create `TOMHTTracker.step(detections, timestamp) -> set[Track]`
- Create `tomht_runner.run_tomht(setup=..., ...)` mirroring `mfa_runner.run_mfa`
- Add `run_tomht_crossing.py` and `run_tomht_bearing_range.py`

**Success criteria:** scripts run, produce tracks (even if bad), animation works.

### Milestone 1 — single global hypothesis only (sanity)
- Keep only one global hypothesis at all times (essentially greedy MHT)
- Still branch per track, but pick best non-conflicting assignment greedily

**Success criteria:** plausible tracking in crossing scenario; no explosions.

### Milestone 2 — K-best global hypotheses
- Maintain `K` global hypotheses (beam)
- Implement global combination by recursion/backtracking (OK for small problems)

**Success criteria:** ambiguity is maintained through the crossing region.

### Milestone 3 — births (simple)
- Start a new track from any unassigned detection if its “new track” score exceeds threshold
- Give each new track a unique track_id in metadata

**Success criteria:** bearing-range scenario maintains multiple targets without requiring hardcoded initial tracks.

### Milestone 4 — pruning / N-scan-lite
- Add max track age without update, miss limits
- Add basic N-scan commitment

**Success criteria:** hypothesis growth stays bounded; tracks don’t flicker too much.

---

## Implementation notes (Stone Soup integration)

### Reuse
- Predictor/updater: KF for linear scenario, UKF for bearing-range scenario.
- Hypothesiser: start with `PDAHypothesiser` (already used in MFA setup), optionally wrap with a helper that exposes:
  - the “miss” hypothesis
  - per-detection hypotheses and their weights

### Data structures
- Keep “track tree nodes” as lightweight metadata on Track objects:
  - avoids creating a parallel state history structure
  - easy to plot with existing Plotter functions

### Deterministic plotting
- Never rely on set iteration ordering.
- Plot tracks in stable order (by track_id or an assigned stable ID).

---

## Suggested parameters (v1 defaults)
- `K = 20` max global hypotheses
- `M = 5` max children per track per scan (after sorting)
- `gate_level` same as scenarios
- `max_missed = 5` scans before track deletion (later)

---

## Next file to implement
Start with:

`mht_experiments/trackers/tomht_tracker.py`

Define:
- `GlobalHypothesis`
- `TOMHTTracker.step(...)`

Then wire it into:

`mht_experiments/runners/tomht_runner.py`
to reuse scenario + plotting harness.
