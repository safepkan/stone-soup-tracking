# TO-MHT next steps (current phase): Association history + N-scan-lite

This document lists the upcoming implementation tasks for the current phase.
Older completed “next steps” snapshots are archived separately (date-stamped).

## Context: where this fits in the roadmap

We currently have:
- Stable per-scan detection ordering and stable detection keys.
- A scoring abstraction (`ScoringModel`) and a working Beta-ratio v1.5 scoring model.
- Beam pruning and global deduplication (currently based on a short per-track signature).

Next, we want to move closer to TO-MHT behavior without a major refactor:
- Introduce short association history per track.
- Use that history for global deduplication.
- Implement an “N-scan-lite” commitment effect by keeping only the best global per last-N association signature.

N-scan-lite approximates classic N-scan pruning by collapsing hypotheses that only differ in decisions older than N scans,
using last-N association signatures rather than explicit common ancestors.

This improves track identity stability and keeps combinatorics under control, while we remain in the current
“copy Track objects, no explicit track-tree” architecture.

---

## 1. Association history (per track)

### 1.1 Design decisions

Store a fixed-length history of association outcomes per scan in track metadata:

- DET key: `>= 0` (per-scan detection index)
- MISS: `-2` (track existed this scan, but missed)
- PAD: `-1` (track did not exist yet within the window)

Keep it as `deque[int](maxlen=N)` for constant memory.

Suggested default:
- `N = 3` scans

### 1.2 Implementation tasks

1) Add tracker parameter(s):
   - `assoc_history_len` (default 3)

2) Ensure each track has a stable `track_id` that survives copying:
   - If already present: standardize on one metadata key (e.g. `track.metadata["track_id"]`)
   - If missing: assign on creation (birth/init) from a counter.

3) Update child-track creation to append to history:
   - On miss: append `MISS`
   - On hit: append `det_key` (stable per-scan detection index)

4) Birth initialization:
   - New track history should be padded then appended for the current scan:
     `assoc_hist = [PAD] * (N-1) + [det_key]`

5) Decide what happens when a track survives but is not expanded (should not happen):
   - In general, every scan should append exactly one history element per surviving track.

### 1.3 Acceptance criteria

- History is deterministic across runs.
- Every surviving track has `len(assoc_hist) <= N` and advances by 1 per scan.
- Birth tracks have PADs in their history for the scans before they existed.

---

## 2. Global deduplication using association history

### 2.1 Rationale

Current deduplication based only on “last association” can merge globals that differ in recent history.
That collapses ambiguity too early, destabilizes identities, and makes N-scan style commitment impossible.

### 2.2 Implementation

Replace the global signature with a history-aware signature:

- For each track in a global:
  - Extract `track_id`
  - Extract `assoc_hist` as a tuple of length up to N
- Sort per-track entries by `track_id`
- Global signature is `tuple((track_id, assoc_hist_tuple) ...)`

When multiple globals share a signature:
- Keep only the highest-weight global (drop the rest)

### 2.3 Acceptance criteria

- In scenarios where history differs, globals no longer collapse into one prematurely.
- Total number of globals remains controlled by beam pruning + history dedupe.

---

## 3. N-scan-lite pruning (commitment effect)

### 3.1 Definition (for this codebase)

We implement an “N-scan-lite” effect by:
- keeping only the best global hypothesis for each “last N associations” signature.

This is equivalent to committing ambiguity older than N scans, without explicit hypothesis trees.

### 3.2 Implementation

If association history length == N, then section 2 already provides N-scan-lite behavior.
If you later store longer history for debugging, then:
- build the signature using only the last N entries per track.

### 3.3 Acceptance criteria

- As the scenario runs, hypothesis diversity does not grow unbounded with time.
- Track identities are visibly more stable (less swapping / churn).
- No regression: true tracks remain stable.

---

## 4. Diagnostics and regression checks

Add lightweight counters/logging (or debug prints behind a flag):
- num globals before/after history dedupe
- num tracks in MAP global
- births per scan

Run at least:
- smoke scenarios (with and without initiator)
- any “two target” scenario where identity stability matters

---

## 5. Deferred work (do not implement in this phase)

- “Full” track-oriented hypothesis trees / shared nodes for memory efficiency.
- Efficient assignment (Murty/k-best) / clustering.
- Scoring beyond beta-ratio v1.5 (raw likelihood, calibrated clutter/birth priors).

These remain on the roadmap.
