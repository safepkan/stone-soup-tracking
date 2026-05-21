# TO-MHT Next Steps

## Implemented profiling-guided cleanup

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

## Next architectural subphase to be defined
