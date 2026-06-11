# TO-MHT Next Steps

## Completed object-boundary cleanup phase

- Bounded detection-conflict retention: tracker-created node
  `detection_history_keys` now retain only keys in the N-scan conflict horizon,
  and tree-level `committed_detection_keys` is a bounded masking set rather than
  a complete committed detection audit log. Public output state-history
  reconstruction is unchanged.
- Added committed output-state history retention controls:
  `max_stored_history_age_s` and `max_stored_history_updates` cap the stored
  committed prefix at N-scan promotion time. Both default to `None`, and the
  current unresolved N-scan lineage remains included in reconstructed tracks.
- Implemented the first narrow object-boundary optimization for local expansion:
  the tracker-owned `TrackerOwnedNLLDistanceHypothesiser` now has a
  state-prior entry point, and TOMHT expansion uses it without reconstructing a
  full Stone Soup `Track` for the default hypothesiser path. Custom
  hypothesisers still receive normal reconstructed Stone Soup `Track` objects.
  The fast path is default-enabled and gated by
  `TOMHTParams.enable_default_hypothesiser_state_fast_path` for profiling/debug
  comparisons.
- Added explicit local-expansion reconstruction observability:
  `expand_track_reconstruct_calls`, `expand_track_reconstruct_ms`, and
  `expand_default_state_fast_path_calls` are now reported so reconstruction
  overhead no longer has to be inferred from `expand_other_ms`.
- Implemented the next narrow object-boundary cleanup for lifecycle deletion:
  the resolved default internal miss-count path now uses `FastMissCountDeleter`
  to check leaf `missed_count` directly instead of reconstructing a full Stone
  Soup `Track`; the fast-deleter interface also receives the owning `TrackTree`.
  Custom Stone Soup deleters still replace that default and still receive a full
  reconstructed `Track` built from the committed prefix plus unresolved lineage.
  Lifecycle deleter
  reconstruction/check counters are now surfaced in scan timing output.
- Added `TOMHTParams.enable_default_miss_deleter_fast_path` as a default-on
  profiling/debug gate for the internal miss-count fast path. Scan timing now
  splits the old broad N-scan/lifecycle/publication bucket into
  `nscan_prune_ms`, `lifecycle_ms`, and `publication_ms`; lifecycle deleter
  reconstruction/check timings are nested under `lifecycle_ms`.
- Accepted replay timing for this phase shows the default paths now avoid
  Stone Soup `Track` reconstruction where history is not needed:
  `expand_track_reconstruct_calls=0` and
  `lifecycle_deleter_track_reconstruct_calls=0` on standard replay, while
  `expand_default_state_fast_path_calls` and
  `lifecycle_default_miss_deleter_fast_path_calls` account for the default
  internal fast-path work. Public output/debug reconstruction remains
  full-history where appropriate.

## Candidate next work

The next subphase should be chosen from the current evidence rather than
assuming another object-boundary cleanup will dominate:

- deeper default-hypothesiser math/kernel profiling,
- expansion volume / which leaves matter,
- broader scenario quality validation,
- output continuity / ID switching,
- internal birth quality.
