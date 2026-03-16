# TO-MHT Current State

This document describes the current architecture and logic of the TO-MHT prototype, as implemented in `tomht_tracker.py` and the associated runners/scenarios. It is meant as a reference for future work and for anyone trying to understand the code without digging through all of it.

- Known warning (bearing_range only): Stone Soup emits a single `LinAlgError('Matrix is not positive definite')` during `make smoke`. It arises when the UKF in the `NoHistoryMultiMeasurementInitiator` predicts a holding track whose covariance has a tiny negative eigenvalue; Stone Soup catches it and regularises with `cholesky_eps`, so runtime behaviour is unaffected. This is a scenario/initiator tuning issue (covariance nearly singular), not a TO-MHT logic bug.
- Scenario run command note (2026-02-12): the stable headless invocations are `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht_experiments/run_tomht_crossing.py` and `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht_experiments/run_tomht_bearing_range.py`; `make smoke` uses the same env settings for both scenarios.
- Scan instrumentation note (2026-02-12): `step()` now creates a structured `ScanStats` object (`self.last_scan_stats`) each scan and prints exactly one compact `SCAN ...` line when `debug_display_scan_stats=True`. `TOMHTParams.collect_stats` (default `True`) appends per-scan stats to `self._stats`, `reset_stats()` clears collected history, and `print_summary_stats()` prints run-level aggregates (globals expanded/dedup/beam, beam full pre/post births against `max_global_hypotheses`, birth activity, MAP tracks/unused). `run_tomht(...)` now calls `print_summary_stats()` automatically at end-of-run when `collect_stats=True`. Birth branching returns `BirthStats` with explicit beam semantics (`birth_track_instances_in_beam`, `globals_with_birth`) plus residual/initiator counters and globals before/after births; stats now include both `birth_tracks_created` (raw initiator output) and `birth_tracks_kept` (post-filter/post-limit). Per-scan `SCAN` output is compact by default (no `miss_hist` unless `debug_display_map_miss_hist=True`), while summary output includes aggregated MAP miss histogram.
- Runner compatibility note (2026-02-12): `run_tomht_crossing.py` and `run_tomht_bearing_range.py` now use `argparse.parse_known_args()` so they can run via VS Code/Jupyter Interactive Window, which injects kernel args such as `--f=...`.
- Runner debug-CLI note (2026-02-12): `run_tomht_crossing.py` and `run_tomht_bearing_range.py` now expose `--debug-detections`, `--debug-scan-stats`, `--debug-hypotheses`, and `--debug-births` (plus `--no-...` forms) and pass them through to `run_tomht(...)`; defaults remain tracker defaults unless explicitly overridden.
- Scoring consistency note (2026-02-12): beta-ratio clutter scoring now uses a shared `_per_unused_log_delta()` helper for both `score_unused_detections()` and the startup debug/sanity check, preventing drift between applied score and asserted/logged value.
- External-start API note (2026-03-12): `TOMHTTracker.add_external_starts(starts,timestamp)` now enforces the completed-`step()` timestamp invariant and inserts confirmed external starts into every current global hypothesis. Inserted tracks receive tracker-owned IDs and baseline maintenance metadata; they are not routed through residual/internal birth discovery and do not receive `birth_log_penalty`. Duplicate-like inputs are intentionally not deduplicated in this phase: each supplied start is treated as a distinct confirmed track.
- Delayed external-start runner note (2026-03-12): `run_tomht_crossing.py` and `run_tomht_bearing_range.py` now accept `--external-start-delay-scans N` or `--external-start-scan N` to inject confirmed external starts after `step()` at a chosen scan. Starts are derived from scenario truth state at the injection scan, inserted via `add_external_starts(...)`, logged as `EXTERNAL_STARTS ...`, and intended for runs without scenario initial tracks. Internal births remain independently toggleable via `--births` / `--no-births`.
- Metadata-initialisation consistency note (2026-03-16): constructor-time initial tracks now write tracker-owned maintenance metadata through the same shared write path used by inserted tracks (internal births and external starts). Constructor defaults remain intentionally unchanged (`age` from `len(track)`, `hits` default `0`), while inserted tracks keep their existing inserted-start conventions. Focused tracker tests now check constructor, birth, and external-start metadata field initialisation (`track_id`, `age`, `hits`, `missed_count`, `last_det_key`, `last_det_hit`, `assoc_history`).
- Operating-mode simplification note (2026-03-16): the runner layer now uses explicit modes `CUSTOM`, `EXTERNAL`, `INTERNAL`, and `BOTH`, logged via `OPERATING_MODE ...` each run. `EXTERNAL`/`INTERNAL`/`BOTH` fully determine births/initial-track/external-start enablement; `CUSTOM` keeps low-level flag control with no mode-specific combination checks.
- Internal-birth pipeline refactor note (2026-03-16): `_branch_globals_with_births(...)` is now a staged orchestration over explicit helper boundaries (residual/candidate generation, sanity filtering, ranking/limit, template prep, compatibility/branching, and birth-stat accounting). This was a structural readability refactor only; ranking key, branching semantics, beam truncation, `BirthStats`, and external-start separation are unchanged.

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
    - `assoc_history: deque[int]` (fixed length `assoc_history_len`; PAD=`-1`, MISS=`-2`, DET=`det_index`)
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
  - `assoc_history_len: int` (stored history length, default 3)
  - `ns_scan_window: int` (history tail length used for global dedupe; defaults to `assoc_history_len`)
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

4. **Deduplicate globals by recent association history**

   - Define a signature per global using the last `ns_scan_window` entries of each track’s `assoc_history` (tail padded with PADs as needed):
     ```python
     sig = tuple(sorted((tid, assoc_hist_tail_tuple) for tid, track in tracks_by_id.items()))
     ```
   - Keep only the best `log_weight` per signature.

   This prevents globals with distinct recent histories from collapsing while still merging hypotheses that only differ beyond the N-scan window.

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

3. **Initiation is still not part of one clean existence model**

   - Internal births currently come from a Stone Soup multi-measurement initiator with its own internal association and filtering logic (a mini tracker).
   - Its “confirmed births” are taken as new tracks, with only heuristic use of its internal history (support/age/misses).
   - **Confirmed externally supplied starts** can now be inserted after `step()` as structural additions to the current globals, without routing them through the internal initiator/birth path.
   - In a “clean” TO-MHT, track existence and birth would be part of the same global hypothesis machinery.

4. **N-scan-like pruning is approximated (history-tail dedupe)**

   - We do not maintain explicit hypothesis trees or “common ancestor” pointers.
   - Instead, each track stores a short `assoc_history`, and global hypotheses are deduplicated
     using the last `ns_scan_window` entries per track.
   - This acts as an **N-scan-lite** commitment mechanism: hypotheses that differ only in decisions
     older than N scans collapse to the best-scoring representative.
   - A “proper” N-scan pruning implementation based on explicit shared trees/ancestors is deferred.

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
  - `--operating-mode {CUSTOM,EXTERNAL,INTERNAL,BOTH}`:
    - `CUSTOM`: uses `--births` / `--initial-tracks` and optional delayed external-start config directly.
    - `EXTERNAL`: births off, initial tracks off, external-start injection on.
    - `INTERNAL`: births on, initial tracks off, external-start injection off.
    - `BOTH`: births on, initial tracks off, external-start injection on.
  - `--births` / `--no-births` and `--initial-tracks` / `--no-initial-tracks` are used in `CUSTOM` mode (defaults: births on, initial tracks off),
  - `--external-start-delay-scans N` or `--external-start-scan N` (mutually exclusive) to inject confirmed external starts after the specified 0-based scan; the delay form is equivalent to `start_scan=N` for the current scenarios because all truth tracks are pre-existing from scan 0.
    - In `EXTERNAL` and `BOTH`, if no external-start scan is specified, the runner uses `start_scan=0`.
    - In `INTERNAL`, external-start scan flags are ignored.
  - `--debug-detections`, `--debug-scan-stats`, `--debug-hypotheses`, `--debug-births` (and `--no-...` forms) to override per-run debug log output toggles exposed by `TOMHTParams` while preserving the existing defaults when omitted.
- External-start derivation in delayed mode:
  - `crossing`: inject two confirmed starts, one per truth path, using the scenario truth state vector at the configured scan and the same covariance used by the scenario’s TO-MHT initial tracks.
  - `bearing_range`: inject three confirmed starts, one per pre-existing truth path from the Stone Soup simulator, using the truth state vector at the configured scan and the same covariance used by the scenario’s TO-MHT initial tracks.
- Headless mode examples (`crossing`):
  - internal: `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht_experiments/run_tomht_crossing.py --operating-mode INTERNAL --no-debug-hypotheses --no-debug-births`
  - external: `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht_experiments/run_tomht_crossing.py --operating-mode EXTERNAL --external-start-scan 3 --no-debug-hypotheses --no-debug-births`
  - both: `MPLBACKEND=Agg TOMHT_NO_SHOW=1 venv/bin/python mht_experiments/run_tomht_crossing.py --operating-mode BOTH --external-start-scan 3 --no-debug-hypotheses --no-debug-births`
- Current limitation: delayed external-start mode assumes the scenario truths are already active from scan 0 and injects all confirmed starts together at one configured scan. Per-track staggered external-start schedules and broader operating-mode cleanup remain future work.

## 5. Summary

The current implementation is best thought of as a **TO-MHT-flavoured multi-hypothesis tracker**:

- It maintains multiple global hypotheses and explores per-track association alternatives.
- It uses Stone Soup’s hypothesis machinery for local association and updating.
- It glues this together with beam search, beta-ratio v1.5 scoring, N-scan-lite history-tail dedupe, instrumentation, and a Stone Soup initiator for internal births.

It is already useful as a playground / experimental platform, but it **intentionally shortcuts** many of the details of a “proper” TO-MHT.

The tracker now has a practical delayed external-start harness in the existing scenario runners,
so `crossing` and `bearing_range` can exercise the confirmed external-start path with headless-friendly commands.
Recent cleanup has made inserted-track metadata handling, runner operating modes, and the internal birth pipeline staging explicit,
without broadening external-start semantics beyond confirmed upstream starts.

The next major steps are later N-scan-lite / scoring refinements and broader TO-MHT model evolution tasks.
