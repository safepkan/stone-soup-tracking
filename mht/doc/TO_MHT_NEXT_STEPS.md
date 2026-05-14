# TO-MHT Next Steps

## Next architectural subphase

**Scoring, birth semantics, and confirmation gates**

The previously planned phase was local expansion volume reduction / pre-expansion control. That remains an important downstream goal, but after returning to the project and reviewing score/probability-based pruning ideas, the immediate prerequisite has become clearer:

> before we use scores to prune, terminate, confirm, or selectively expand tracks, the score semantics themselves need to be coherent.

The current focus is therefore to complete the scoring and birth/initiation migration so that later expansion-volume and pruning work can be based on interpretable scores/probabilities rather than legacy heuristics.

---

## Recent baseline changes

The tracker is now in a better shape for this work because several pieces have already landed:

- major `TOMHTTracker` substeps have been extracted into dedicated modules,
- persistent tree/node bookkeeping now lives in `TrackTreeStore`,
- local expansion, internal-birth handling, clustering, overload splitting, cluster rebuild, post-solve pruning, MAP-only N-scan pruning, and TOMHT utilities are no longer monolithic tracker methods,
- external starts now use an existence-prior probability mapped internally to log-odds,
- external starts can optionally override that prior per track via `Track.metadata["existence_probability"]`,
- output tracks now expose score-implied existence metadata,
- legacy unused-detection scoring has been removed,
- cluster solving now uses accumulated leaf scores directly rather than affine unused-detection pre-baking,
- constructor `initiator=None` now cleanly disables internal starts while a configured initiator is the generic internal-start lane,
- initiator-created starts now use an explicit initiator-start existence prior, with optional per-track metadata override,
- the legacy fixed `birth_log_penalty` and `ScoringModel.score_birth(...)` hook have been removed,
- `TOMHTParams.internal_birth_mode` has been removed,
- the public `TOMHTTracker(scoring_model=...)` injection point and `make_default_scoring_model(...)` factory have been removed; the tracker directly constructs `NLLScoringModel`,
- there is no tracker-core `birth_density` parameter; birth-density reasoning is guidance for choosing an initiator-start existence prior,
- `TrackTree` now has sticky `tentative`/`confirmed` lifecycle state driven by max active-leaf score crossing `TOMHTParams.track_confirmation_existence_probability`,
- output publication is now a separate sticky tree-level state (`unpublished`/`published`) with configurable emit gating; the default publishes confirmed tracks only,
- sticky output-publication and MAP output reconstruction helpers now live in `mht/tomht_output.py`,
- whole-track score deletion now removes trees whose max active-leaf score falls below `TOMHTParams.track_deletion_existence_probability`,
- whole-track lifecycle implementation now lives in `mht/tomht_lifecycle.py`, with `TOMHTTracker` retaining thin gateway methods only.
- `NLLScoringModel` now consumes a narrow `DetectionProbabilityModel`; the default `ConstantDetectionProbabilityModel` wraps the scalar `TOMHTParams.prob_detect` and `TOMHTParams.clutter_density`, while custom DPMs can vary `P_D` and clutter density by prediction, detection, and opaque caller scan context.
- internal-birth candidate selection no longer applies tracker-owned state-layout sanity checks; sorting and debug output flatten arbitrary state vectors generically.

These changes make the next scoring/birth steps easier to reason about.

---

## Current scoring baseline

Existing-track local scoring is now the cleanest part of the scoring model.

The tracker-owned distance hypothesiser emits detection hypotheses whose distance is:

```text
NLL = -log p(z | x)
```

without detection-probability or clutter-density factors.

The tracker-owned `NLLScoringModel` then applies:

```text
hit  = log(P_D) - log(lambda) - NLL
miss = log(1 - P_D)
```

where:

- `P_D` is detection probability,
- `lambda` is clutter density,
- `lambda` must be expressed in the same measurement-space units as the hypothesiser NLL.
- default scalar `P_D`/`lambda` come from `ConstantDetectionProbabilityModel`,
- advanced callers can provide a DPM for finite-FOV, range-dependent, sensor-mode, or other scan/state-dependent scoring,
- the opaque `caller_scan_context` passed to `update_tracker(...)` is distinct from TOMHT's internal scan bookkeeping and is available even when a scan has no detections,
- one `update_tracker(...)` call should contain detections from one sensor / one measurement space; multi-sensor applications should call it separately for each sensor update,
- DPM callbacks receive public track IDs only after publication; unpublished trees pass `track_id=None`,
- hit clutter density callbacks receive both the prediction and concrete detection, and `P_D ~= 0` outside coverage makes misses nearly penalty-free.

Legacy unused-detection scoring has been removed because the clutter-density contrast is already represented in the hit score. This was expected to change smoke/replay outputs because old scenarios were tuned against older heuristic scoring. That is acceptable: we are prioritizing a coherent scoring interpretation before retuning.

The remaining weak area is birth/initiation semantics.

---

## Direction: explicit start lanes

The tracker should keep two start lanes explicit.

### 1. External starts

External starts are already on the cleaner path.

They represent caller-supplied starts that should be treated as externally confirmed or externally meaningful. They are initialized from an existence probability mapped to log-odds. The global default can be overridden by `Track.metadata["existence_probability"]`.

This lane should remain the preferred integration path for systems that already have their own domain-specific initiation or confirmation process.

### 2. Internal initiator starts

Internal starts are generated by passing residual detections to a configured Stone Soup initiator and inserting the returned tracks. From the TO-MHT tracker’s perspective, a one-shot `SimpleMeasurementInitiator` and a more complex M/N or domain-aware initiator are the same abstraction:

```text
residual detections -> configured initiator -> candidate start tracks
```

Even a one-detection start needs a domain-specific state initializer and prior. The tracker therefore does not own generic single-detection birth initialization. It treats initiator output conceptually as:

```text
external-style starts generated inside the tracker for convenience
```

That means initiator-created roots should be scored with an explicit initial existence prior, analogous to external starts, but probably with a separate parameter and typically lower default confidence.

This keeps the scoring story honest:

- if an external caller produces a confirmed start, the tracker applies the external-start prior,
- if a configured initiator produces an internal start, the tracker applies the initiator-start prior,
- if the domain wants one-detection starts, it should configure an appropriate one-shot initiator.

---

## Immediate implementation plan

### Step 1: Simplify internal-start switching

Implemented (2026-05-13): `TOMHTParams.internal_birth_mode` was removed. The internal-start switch is now constructor initiator presence: `initiator=None` creates no internal starts and leaves residual detections available; `initiator=<Initiator>` passes residual detections to that initiator.

The earlier reserved `"single_detection"` stub was removed after the initialization-boundary decision: one-detection starts are represented by configuring an initiator, not by a tracker-owned mode.

Important behavior expectations:

- `initiator=None` means no internal starts; external starts can still be used.
- `initiator=<Initiator>` means pass residual detections to the configured initiator.

This keeps the public control surface smaller while preserving the same active internal-start behavior.

### Step 2: Replace fixed initiator birth penalty

Implemented (2026-05-13): initiator-created starts now use `TOMHTParams.initiator_start_initial_existence_probability` (default `0.8`) mapped to log-odds for the root `log_delta`. Valid initiated-track `metadata["existence_probability"]` overrides the default; missing or invalid metadata falls back to the parameter value.

The current fixed birth penalty is not a good scoring story for initiator output.

For initiator-created starts, introduce something like:

```text
initiator_start_initial_existence_probability
```

and map it to log-odds for the root `log_delta`, just as external starts do.

This should replace the current fixed penalty for the internal initiator lane.

Suggested semantics:

- valid per-track metadata may eventually override it,
- missing metadata uses the parameter default,
- invalid optional metadata should fall back rather than reject the start,
- this parameter should be separate from `external_start_initial_existence_probability`.

Potential default should probably be lower than external starts, because internally generated initiator starts may be tentative rather than externally confirmed.

### Step 3: Tune/configure internal initiators and priors

Do not add a tracker-owned single-detection birth mode. Even one-detection starts need a domain-specific initializer and prior, so they should be provided by a configured initiator.

For a `SimpleMeasurementInitiator`-style one-detection initializer, one principled way to choose the default prior is:

```text
logit(P_init) = log(P_D * beta_NT / lambda)
```

equivalently:

```text
P_init = sigmoid(log(P_D) + log(beta_NT) - log(lambda))
```

where:

- `beta_NT` is a new-target/birth density in measurement-space units,
- `lambda` is clutter density in the same measurement-space units,
- `P_D` is detection probability.

This remains guidance for choosing `initiator_start_initial_existence_probability`, not a new tracker-core `birth_density` parameter.

Do not introduce a tracker-owned two-point or M/N initiator in this step. If output noise becomes a problem, solve that with initiator configuration and output confirmation gates first.

### Step 4: Add first tree-level confirmation lifecycle state

Implemented (2026-05-14): `TrackTree.lifecycle_state` now starts as `"tentative"` for both internal initiator starts and external starts, and promotes stickily to `"confirmed"` when:

```text
max(active_leaf.accumulated_log_score) >= logit(track_confirmation_existence_probability)
```

The new user-facing parameter is:

```text
track_confirmation_existence_probability = 0.9
```

with validation `0.0 < p < 1.0` and internal conversion to log-odds.

Confirmation is applied after supported-leaf pruning and MAP-only N-scan pruning, before whole-track deletion lifecycle and output generation. It is intentionally conservative:

- no un-confirming,
- confirmation itself does not delete tracks or directly filter output,
- MAP output tracks include `metadata["lifecycle_state"]`,
- scan stats and summary output report tentative vs confirmed active-tree counts.

This gives later publication and termination work a stable tree-level state to build on.

### Step 5: Add output confirmation / emit gate

Implemented (2026-05-14): `TrackTree.publication_state` now starts as `"unpublished"` and promotes stickily to `"published"` when a MAP-selected live tree satisfies the configured output-publication policy. Publication is separate from internal confirmation and does not alter MAP hypotheses, active leaves, N-scan pruning, or whole-track deletion.

The user-facing publication parameters are:

```text
publish_lifecycle_states = ("confirmed",)
publish_min_hits = 0
publish_min_age = 0
publish_min_existence_probability = 0.0
```

The default keeps tentative MAP tracks internal and publishes only confirmed tracks. Stricter settings can add hit, age, or score-implied existence requirements; permissive settings can opt back into tentative publication for experiments.

Publication criteria are evaluated against the MAP-selected leaf for each live tree:

- tree `lifecycle_state` is allowed by `publish_lifecycle_states`,
- `leaf.hits >= publish_min_hits`,
- `leaf.age >= publish_min_age`,
- score-implied existence from `leaf.accumulated_log_score` meets `publish_min_existence_probability`.

Once published, a tree remains published until it is deleted/recreated. Standard `get_map_output_tracks()`, `tracks`, and `update_tracker(...)` return only published MAP tracks. `get_map_hypothesis_snapshot()` remains the internal MAP inspection API, and `get_map_output_tracks(include_unpublished=True)` can reconstruct tentative/unpublished MAP tracks for inspection with `metadata["publication_state"]`. Scan stats and summary output now report MAP-published and MAP-unpublished counts so publication suppression is visible without changing internal MAP logging.

Public output identity is now separate from internal tree identity. `TrackTree.track_id` remains the internal logical ID allocated when the tree is created, while `TrackTree.public_track_id` starts as `None` and is assigned exactly once when publication first flips to `"published"`. The default mapper assigns dense integer public IDs in first-publication order, so unpublished internal trees no longer force gaps in published `Track.id` values. Output metadata keeps `internal_track_id` as the explicit internal logical ID and adds `public_track_id`; the legacy `track_id` metadata field remains only as a deprecated compatibility alias. The old `get_tomht_track_id(track)` helper has been removed. Unpublished inspection tracks have `public_track_id=None` and do not consume a public ID.

When simple one-detection initiators are used, the tracker may internally carry more tentative tracks. That is expected; the publication gate now controls only returned/emitted tracks, not whether the internal MHT state keeps tentative hypotheses.

### Step 6: Add score-based whole-track deletion

Implemented (2026-05-14): `TOMHTParams.track_deletion_existence_probability` now defaults to `0.01` and is validated as `0.0 < p < 1.0`. The tracker converts it to log-odds internally and applies it after sticky confirmation and MAP-only N-scan pruning:

```text
delete tree if max(active_leaf.accumulated_log_score) <= logit(track_deletion_existence_probability)
```

This deletes the whole `TrackTree` and filters the current MAP global to surviving live trees. Confirmation and deletion now form hysteresis-style score gates: confirmation is sticky at the high score threshold, while deletion is a low score threshold that removes the tree.

Score deletion is the primary principled mechanism for killing low-score spurious starts. Existing miss-count deletion and optional Stone Soup deleters remain lifecycle backstops/domain hooks. Score deletion runs in both existing lifecycle lanes; with no custom deleter it OR-composes with node-native miss deletion, and with a custom deleter it still runs alongside deleter deletion. `TRACK_LIFECYCLE` logs now report terminated IDs by deterministic reason groups (`score`, `miss`, `deleter`).

Published-tree deletion needs no separate unpublish step. Public IDs are not reused after deletion, and dense publication-time ID assignment remains unchanged.

### Step 7: Revisit pruning and expansion-volume controls

After scoring, births, confirmation, publication, and whole-track score deletion are coherent, return to:

- broader score-based leaf pruning,
- selective expansion,
- expansion-volume characterization,
- overload-split pruning behavior,
- and frontier growth controls.

This is the earlier expansion-volume phase, but with a stronger foundation.

---

## Non-goals for this phase

Do not yet:

- implement broad score-based pruning,
- make aggressive expansion-volume changes,
- retune all smoke scenarios,
- redesign cluster solving,
- introduce parallel local expansion,
- remove black-box initiator support entirely,
- add tracker-owned two-point or M/N initiation,
- or treat output confirmation gating as internal hypothesis deletion.

The goal is to get the scoring and initiation semantics clean enough that later pruning/volume work has a meaningful score basis.

---

## Important design notes

### Scores and probabilities

The tracker is moving toward score/probability-based decisions. Therefore, score offsets matter more than they used to.

External starts and internal initiator starts now use existence probabilities mapped to log-odds rather than arbitrary penalties.

### One-detection initiator starts may be noisy internally

A one-detection measurement initiator can create tentative starts from clutter detections. This is not necessarily wrong.

The right question is not whether every internal tentative track is real. The right question is whether:

- scores evolve sensibly,
- low-quality tracks die or are pruned,
- and output consumers only see confirmed-enough tracks.

### Output gating is not the same as internal pruning

A confirmed-output gate should reduce visible clutter tracks without prematurely deleting internal hypotheses.

Internal pruning can come later, once score behavior is better understood.

### Initiators are convenience integration points

A configured initiator can be simple, stateful, or domain-specific. But unless it provides a likelihood/existence model, the tracker should not pretend to know how to score it from first principles.

Treat it as an internal convenience wrapper around external-style starts.

---

## Acceptance criteria for this subphase

This subphase is successful when:

- unused-detection scoring remains removed,
- external starts continue to have configurable/per-track existence priors,
- initiator-created starts no longer rely on a fixed arbitrary birth penalty,
- one-detection starts are represented by configured initiators rather than a tracker-owned mode,
- guidance exists for mapping `P_D`, new-target density, and clutter density into an initiator-start existence prior,
- output confirmation/emit gating is implemented or clearly staged,
- smoke/replay outputs remain operationally usable,
- and the code/docs make the external-start and internal-initiator lanes clear.

At that point, it should be reasonable to return to score/frontier/expansion-volume pruning work.
