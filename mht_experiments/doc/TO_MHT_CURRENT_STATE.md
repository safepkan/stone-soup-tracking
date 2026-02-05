# TO-MHT Current State

This document describes the current architecture and logic of the TO-MHT prototype, as implemented in `tomht_tracker.py` and the associated runners/scenarios. It is meant as a reference for future work and for anyone trying to understand the code without digging through all of it.

- Known warning (bearing_range only): Stone Soup emits a single `LinAlgError('Matrix is not positive definite')` during `make smoke`. It arises when the UKF in the `NoHistoryMultiMeasurementInitiator` predicts a holding track whose covariance has a tiny negative eigenvalue; Stone Soup catches it and regularises with `cholesky_eps`, so runtime behaviour is unaffected. This is a scenario/initiator tuning issue (covariance nearly singular), not a TO-MHT logic bug.

## 1. High-level structure

The central class is:

- `TOMHTTracker`
  - Inputs per step: a collection of Stone Soup `Detection`s and a timestamp.
  - Outputs per step: a set of Stone Soup `Track`s (currently: the tracks from the MAP global hypothesis).

Supporting concepts:

- `GlobalHypothesis`
  - `tracks_by_id: dict[int, Track]`
  - `log_weight: float`

- `Track`
  - Standard Stone Soup `Track` (sequence of states).
  - `metadata` is used to store MHT-related information:
    - `track_id: int`
    - `age: int` (number of scans since track was created in this tree)
    - `hits: int` (number of detection updates)
    - `missed_count: int` (consecutive misses)
    - `last_det_key: int | None` (per-scan detection index)
    - `last_det_hit: bool` (hit vs miss at last step)

- Parameters in `TOMHTParams` (examples):
  - `max_global_hypotheses: int` (beam width)
  - `max_children_per_track: int`
  - `max_missed: int`
  - `birth_log_penalty: float`
  - `unused_det_log_penalty: float`
  - `max_births_per_scan: int`
  - `births_k: int` (number of top globals used to define “residual” for births)
  - Debug flags: `debug`, `debug_births`, etc.

The tracker is tightly coupled to Stone Soup predictor/updater/hypothesiser objects, which are created externally by scenario-specific builder functions.

## 2. Per-scan step logic

The `step()` method roughly does:

1. **Prepare detections**

   - Convert `detections` into a `det_list: list[Detection]`, sorted by a deterministic key:
     - timestamp (float seconds when available),
     - measurement vector length,
     - flattened measurement vector with type-tagged elements (`float` values ordered first; NaN/inf treated as `+inf`; non-numerics ordered by `str(x)`).
     - If two detections are identical on those fields, their relative order falls back to the input iterable’s order (Python sort is stable); unordered inputs like `set` will therefore give nondeterministic duplicate ordering.
   - Build a per-scan index:
     ```python
     det_index_by_obj = {id(det): i for i, det in enumerate(det_list)}
     ```

   This index is used as the canonical detection key for the current scan.
   Sorting removes nondeterminism when the incoming collection is unordered.

2. **Expand global hypotheses via existing tracks**

   For each global hypothesis `gh`:

   - For each track in `gh.tracks_by_id`:
     - Call `hypothesiser.hypothesise(track, det_list, timestamp)` (Stone Soup).
     - Collect the resulting `Hypothesis` objects (hit and miss).
   - Score each hypothesis via the `ScoringModel` (currently **beta-ratio v1.5** only):
       - Hit: `log(betai) - log(beta0) + log(1 - P_D * P_G)`.
       - Miss: same common term `log(1 - P_D * P_G)`.
     - Prune to at most `max_children_per_track`, ensuring that at least one miss is kept if present.
     - Build `ChildCandidate` objects for each kept hypothesis:
       - `child_track`: a copy of the parent track with:
         - either prediction appended (miss) or update appended (hit),
         - metadata updated: `age`, `hits`, `missed_count`, `last_det_key`, `last_det_hit`.
       - `used_det_key`: an int index if a measurement was used, else `None`.
       - `log_delta`: the log-score increment for that child.

   - Perform a backtracking search across tracks to produce all **per-scan consistent** combinations:
     - For each combination, ensure that `used_det_key`s are unique (no detection is used by two tracks in the same scan).
     - Drop tracks whose `missed_count` exceeds `max_missed`.
     - For each combination, create a new `GlobalHypothesis`:
       - `tracks_by_id`: mapping from track IDs to child tracks.
       - `log_weight`: parent log weight + sum of `log_delta`s.

3. **Apply unused-detection penalty**

   For each new global:

   - Compute the set of used detection indices from its tracks’ `last_det_key`.
   - Count `unused = len(det_list) - len(used)`.
   - Apply clutter score:
     - Beta mode: add `unused * log(clutter_density)` (falls back to `-unused_det_log_penalty * unused` if density ≤ 0).
     - Legacy mode: subtract `unused_det_log_penalty * unused` (previous behaviour).

   This is a heuristic way to reward hypotheses that explain more detections, and discourage always “ignoring” targets.

4. **Deduplicate globals by present association**

   - Define a signature per global:
     ```python
     sig = tuple(sorted((tid, last_det_key_of_track_tid) for tid in tracks_by_id))
     ```
   - Keep only the best log_weight per signature.

   This collapses globals that differ only in past history but are currently identical in which tracks they have and which detections they used this scan.

5. **Beam pruning**

   - Sort globals by `log_weight` descending.
   - Keep only the top `max_global_hypotheses`.

6. **Handle track births via initiator**

   - Compute “residual” detections:
     - Look at the top `births_k` global hypotheses.
     - Union their used detection indices.
     - Residuals = detections whose indices are not in that union.
   - If there is an initiator and residuals are non-empty:
     - Call `NoHistoryMultiMeasurementInitiator.initiate(OrderedSet(residuals))`.
     - This returns a set of newly confirmed birth tracks, each with a `holding_track` in metadata.
     - Filter out numerically insane births:
       - Non-finite positions or covariance, absurdly large values, etc.
     - For each birth, compute:
       - `used` = the detection index of the measurement used to create the final state (if available).
       - `support`, `age`, `misses` from the `holding_track`:
         - support = number of update states,
         - age = number of states in holding,
         - misses = age − support.
       - Covariance trace of the last state.
     - Rank births using a heuristic key, e.g.:
       ```python
       (-support, misses, age, cov_trace, used_index)
       ```
     - Keep up to `max_births_per_scan` births.

   - For each existing global hypothesis `gh`:
     - Always keep the original `gh` (no births) except for special start-up cases.
     - For each compatible birth (i.e., its `used` index is not already used by `gh` this scan):
       - Create a new global `gh'` with that single birth added:
         - Copy `gh.tracks_by_id`, insert a copy of the birth track with metadata initialised (`age=1`, `hits=1`, etc.).
         - Subtract `birth_log_penalty` from `log_weight`.
     - Optionally, if there are 2 compatible births and `max_births_per_scan>=2`, also create a “both births” variant and penalise accordingly.

   - After birth branching:
     - Sort by `log_weight`, keep top `max_global_hypotheses`.

7. **Output tracks**

   - The tracker currently outputs the tracks from the **best** (MAP) global hypothesis:
     ```python
     best = self.global_hypotheses[0]
     return set(best.tracks_by_id.values())
     ```

   Only the present state of the MAP global is visualised in the current runners.

## 3. Where this diverges from a “clean” textbook TO-MHT

Some important differences and simplifications:

1. **Track tree structure vs. flat `Track` copies**

   - Textbook TO-MHT often models tracks as nodes in a tree (each hypothesised path has a parent, depth, etc.).
   - Here, each `Track` is a standalone Stone Soup track with copies of all past states.
   - This simplifies implementation but makes it harder to:
     - do formal N-scan back-pruning,
     - reason about hypothesis trees as explicit trees.

2. **Scoring model (beta-ratio v1.5)**

   - Scoring is routed through a `ScoringModel`; only the **beta_ratio** implementation is active.
   - Beta-ratio v1.5:
     - Uses PDA β values (normalised per-track association probabilities) to approximate MHT-style log increments.
     - Hit: `log(betai) - log(beta0) + log(1 - P_D * P_G)` where `beta0` is the miss β.
     - Miss: same common term `log(1 - P_D * P_G)`.
     - Unused detections: `len(unused) * log(clutter_density)`; if clutter density ≤ 0, falls back to `-unused_det_log_penalty * len(unused)`.
     - Births: `-birth_log_penalty` per birth.

3. **Initiation is external and opaque**

   - The multi-measurement initiator has its own internal association and filtering logic (a mini tracker).
   - Its “confirmed births” are taken as new tracks, with only heuristic use of its internal history (support/age/misses).
   - In a “clean” TO-MHT, track existence and birth are part of the same global hypothesis machinery.

4. **N-scan-like pruning is approximated**

   - Only the current scan’s association signature is used for deduplication.
   - There is no notion of “commit all associations older than N scans” or merging trees based on agreed history.
   - As a result, some redundant hypothesis structure persists, and identity stability relies heavily on beam pruning and the unused-detection penalty.

5. **Track termination logic is very simple**

   - Tracks are deleted when `missed_count > max_missed`.
   - There is no explicit existence probability or death process.

## 4. Reproducibility (current status)

- Scenario generation is seeded (`crossing_targets.py` uses `np.random.seed(2001)`, `bearing_range.py` uses `np.random.seed(1908)`), so truths and detections are repeatable on the same Python/NumPy stack.
- Per-scan detection ordering is made explicit and deterministic (`_sorted_detections`), which fixes `last_det_key` and residual selection.
- Per-scan data is bundled into a `ScanContext` (timestamp, ordered detections, per-scan det index); it is passed to scoring and birth logic to keep inputs consistent across helper functions.
- Global hypothesis branching/pruning uses deterministic sort keys; the tracker itself does no additional sampling.
- `make smoke` (both scenarios, headless) produces identical logs across repeated runs except for wall-clock timestamps in the debug output.
- A/B convenience: `run_tomht_crossing.py` / `run_tomht_bearing_range.py` accept:
  - `--births` / `--no-births` (BooleanOptionalAction) to toggle initiator use,
  - `--initial-tracks` / `--no-initial-tracks` (BooleanOptionalAction) to toggle scenario-provided initial tracks (defaults match each scenario: crossing = births on, initial tracks off; bearing_range = births on, initial tracks off).

## 5. Summary

The current implementation is best thought of as a **TO-MHT-flavoured multi-hypothesis tracker**:

- It maintains multiple global hypotheses and explores per-track association alternatives.
- It uses Stone Soup’s hypothesis machinery for local association and updating.
- It glues this together with beam search, simple penalties, and a Stone Soup initiator.

It is already useful as a playground/experimental platform, but it **intentionally shortcuts** many of the details of a “proper” TO-MHT. The next major steps are the N-scan-lite groundwork (association history) and further scoring refinements (beyond the current beta-ratio v1.5 bridge).
- A more principled N-scan-lite commitment / merging mechanism.
- Cleaner integration (or replacement) of the external initiator.
