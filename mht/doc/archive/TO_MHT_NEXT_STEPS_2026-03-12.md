# TO-MHT next steps (current phase): Association history + N-scan-lite

This document lists the upcoming implementation tasks for the current phase.
Older completed “next steps” snapshots are archived separately (date-stamped).

## Context: where this fits in the roadmap

We currently have:
- Stable per-scan detection ordering and stable detection keys.
- A scoring abstraction (`ScoringModel`) and a working Beta-ratio v1.5 scoring model.
- Beam pruning and global deduplication.

Next, we want to move closer to TO-MHT behavior without a major refactor:
- Track short association history per track.
- Use association history for global deduplication (avoid collapsing distinct recent histories).
- Implement an “N-scan-lite” commitment effect by keeping only the best global per last-N association signature.

Note: This is an approximation of classic N-scan pruning (which relies on explicit common ancestors in a hypothesis tree).
We are not introducing explicit trees yet; this phase should remain compatible with a later refactor to proper TO-MHT trees.

---

## 1. Association history (per track)

### 1.1 Design decisions

Store a fixed-length history of association outcomes per scan in track metadata as integers:

- DET key: `>= 0` (per-scan detection index)
- MISS: `-2` (track existed this scan, but missed)
- PAD: `-1` (track did not exist yet within the window)

Keep it as `deque[int](maxlen=assoc_history_len)` for constant memory.

Configuration:
- `assoc_history_len` (H): how much history we store (debuggable)
- `ns_scan_window` (N): how much history we *use* in signatures (N ≤ H)

Suggested defaults for now:
- H = 3, N = 3

### 1.2 Implementation tasks

1) Add tracker parameters **(implemented 2026-02-06)**:
   - `assoc_history_len` (default 3)
   - `ns_scan_window` (default = assoc_history_len)

2) Ensure each track has a stable `track_id` that survives copying **(implemented 2026-02-06)**:
   - standardize on metadata key (e.g. `track.metadata["track_id"]`)
   - assign at creation (birth/init) from a counter.

3) Update child-track creation to append one history element per scan **(implemented 2026-02-06)**:
   - On miss: append `MISS`
   - On hit: append `det_key` (stable per-scan detection index)

4) Birth initialization **(implemented 2026-02-06)**:
   - For a new track born on detection `det_key`:
     `assoc_hist = [PAD] * (H-1) + [det_key]`

5) Consistency rule:
   - Every scan, every surviving track hypothesis should advance history by exactly one element.

### 1.3 Acceptance criteria

- History is deterministic across runs.
- Every surviving track has an `assoc_hist` and it advances by 1 per scan.
- Birth tracks show PADs for scans before the track existed.

---

## 2. Global deduplication using association history

### 2.1 Rationale

Current deduplication based only on “last association” can merge globals that differ in recent history.
That collapses ambiguity too early and destabilizes track identities.

### 2.2 Implementation **(implemented 2026-02-06)**

Replace global signature with a history-aware signature:

- For each track in a global:
  - extract `track_id`
  - extract last `ns_scan_window` entries of `assoc_hist`
- Sort per-track entries by `track_id`
- Global signature is:
  `tuple((track_id, assoc_hist_tail_tuple) ...)`

When multiple globals share a signature:
- keep only the highest-weight global.

### 2.3 Acceptance criteria

- In the crossing-target scenario, globals with different recent histories are not prematurely merged.
- Hypothesis counts remain controlled by beam pruning + history dedupe.

---

## 3. N-scan-lite pruning (commitment effect)

### 3.1 Definition (for this codebase)

We approximate classic N-scan pruning by collapsing hypotheses that differ only in decisions older than N scans:
- signatures only include the last N association outcomes per track
- thus ambiguity older than N does not continue to branch.

Classic N-scan pruning relies on explicit common ancestors in a hypothesis tree; we are deferring explicit trees.

### 3.2 Implementation

If H == N, then section 2 already provides N-scan-lite behavior.
If H > N (kept for debugging), signatures should use only the last N entries.

### 3.3 Acceptance criteria

- Hypothesis diversity does not grow unbounded with time in long runs.
- Track identities are visibly more stable in the crossing scenario.

---

## 4. Minimal instrumentation (small, but required)

Implemented (2026-02-12): `tomht_tracker.step()` now emits a structured `ScanStats` per scan (`self.last_scan_stats`) and one compact `SCAN ...` line when `debug_display_scan_stats=True`. With `collect_stats=True` it also appends each scan to `self._stats`; `reset_stats()` clears history and `print_summary_stats()` prints aggregate run metrics (including beam full pre/post births vs `max_global_hypotheses`, `birth_tracks_created` vs `birth_tracks_kept`, and aggregated MAP miss histogram).

Included counters:
- detections per scan
- globals at step boundaries (in, expanded, after-unused, after-dedupe, after-beam, after-births)
- MAP summary (`map_tracks`, `map_used`, `map_unused`)
- birth flow counters from structured `BirthStats` (`birth_candidates`, `birth_tracks_created`, `birth_track_instances_in_beam`, `globals_with_birth`)
- birth flow counters now split into raw proposals (`birth_tracks_created`) and actual branch pressure (`birth_tracks_kept`)
- optional compact MAP quality summary (`map_miss_hist`, `map_mean_hit_rate`)

This remains lightweight (not a full evaluation framework) but is enough to spot expansion blow-up, weak dedupe, excessive births, and rising MAP unused detections.

---

## 5. Deferred work (explicitly not in this phase)

- Full track-oriented hypothesis trees / shared nodes.
- Efficient assignment (Murty/k-best) / clustering.
- Scoring beyond beta-ratio v1.5 (raw likelihood, calibrated clutter/birth priors).
- Formal evaluation metrics beyond basic counters/plots.

These remain on the roadmap.
