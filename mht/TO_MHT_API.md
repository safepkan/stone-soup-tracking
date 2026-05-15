# TO-MHT API and Integration Guide

This document describes the public integration surface and modeling assumptions for
the track-oriented TOMHT tracker.

It is intended for users integrating TOMHT into an application-specific tracking
system, including multi-sensor systems where detection probability, clutter
density, track initiation, and publication policy are domain-specific.

The goal is to explain not only *what* the API does, but also the model
assumptions behind it.

---

## 1. Mental model

TOMHT is a track-oriented multiple-hypothesis tracking layer built around Stone
Soup boundary objects.

The tracker owns:

- persistent track-tree bookkeeping,
- local hypothesis expansion,
- additive score accumulation,
- measurement-exclusivity clustering,
- exact solving within rebuilt clusters/subclusters,
- optional overload/relaxation guardrails around cluster rebuilds,
- N-scan pruning,
- whole-track confirmation and deletion,
- sticky output publication,
- and public output track reconstruction.

The caller owns the application-specific sensing model:

- predictor,
- updater,
- measurement model,
- optional custom hypothesiser,
- optional initiator,
- optional deleter,
- optional detection-probability / clutter-density model,
- and any sensor context needed to evaluate those models.

This separation is intentional. TOMHT should not know whether detections come
from radar, bearing/range sensors, ISAC processing, or a synthetic scenario. It
should only know how to manage hypotheses and apply scores once the caller has
provided the Stone Soup components and probabilistic assumptions.

---

## 2. Core tracker usage

A typical setup looks like:

```python
tracker = TOMHTTracker(
    updater=updater,
    predictor=predictor,          # or hypothesiser=custom_hypothesiser
    initiator=initiator,          # optional
    deleter=deleter,              # optional
    params=params,
    detection_probability_model=dpm,  # optional
)

time, published_tracks = tracker.update_tracker(
    timestamp,
    detections,
    caller_scan_context=sensor_context,
)
```

The most important public methods are:

```python
update_tracker(time, detections, *, caller_scan_context=None)
tracks
get_map_output_tracks(include_unpublished=False)
get_map_hypothesis_snapshot()
get_track_tree_snapshot()
get_unused_detections()
add_external_starts(time, starts)
```

`update_tracker(...)` processes one scan/update and returns the current
published MAP output tracks.

`tracks` is a convenience property returning the same published MAP output view
for the most recently processed scan. In normal use, the set returned by
`update_tracker(...)` and the value of `tracker.tracks` immediately afterwards
represent the same output boundary.

`get_map_output_tracks(include_unpublished=True)` can be used for inspection of
unpublished MAP-selected tracks.

`get_map_hypothesis_snapshot()` exposes the internal MAP-selected leaf nodes for
debugging and evaluation. It is not the public output boundary.

`get_unused_detections()` returns residual detections from the most recent scan
primarily for external-start workflows where no internal initiator is configured.

`add_external_starts(...)` injects caller-vetted external starts after an
`update_tracker(...)` call at the same timestamp.

---

## 3. One sensor / one measurement space per update

One call to `update_tracker(...)` is expected to represent one sensor update in
one measurement space.

Do not mix heterogeneous sensors or measurement spaces in a single update call.
The current TOMHT model assumes that each track can be associated with at most
one detection per update. If detections from multiple sensors are mixed into a
single call, that assumption becomes wrong.

For multi-sensor systems, call the tracker once per sensor update:

```python
tracker.update_tracker(t1, radar_a_detections, caller_scan_context=radar_a_context)
tracker.update_tracker(t2, radar_b_detections, caller_scan_context=radar_b_context)
```

The caller-provided context should identify the sensor and any scan-specific
state needed by the detection-probability model. This context must be meaningful
even when the detection list is empty, since empty scans still provide miss
evidence.

---

## 4. Constructor components

### `updater`

The Stone Soup updater is required. It is used to produce posterior states for
selected detection hypotheses.

### `predictor` or `hypothesiser`

Exactly one of `predictor` or `hypothesiser` must be supplied.

If `predictor` is supplied, TOMHT constructs its tracker-owned default
NLL-distance hypothesiser.

If `hypothesiser` is supplied, it must be a distance hypothesiser that returns
Stone Soup hypotheses whose detection-hypothesis distance is:

```text
NLL = -log p(z | x)
```

This distance must be the measurement likelihood NLL only. It must not include
detection-probability factors, clutter-density factors, birth terms, or
additional score offsets. TOMHT applies those factors separately.

The provided hypothesiser is expected to expose a predictor for tracker wiring.

#### Custom hypothesiser contract

Custom hypothesiser integrations must satisfy the following constraints for each
track leaf and scan:

- return a Stone Soup `MultipleHypothesis`,
- include only `SingleDistanceHypothesis` entries,
- use finite distances,
- include exactly one missed-detection hypothesis,
- return detection-hypothesis distances as NLL only,
- and reference the original detection objects from the current scan, not copies
  or reconstructed detections.

The last point is important because TOMHT recovers scan-local detection identity
from object identity.

### `initiator`

The initiator is optional.

If `initiator=None`, TOMHT does not create internal starts. Residual detections
remain available through `get_unused_detections()`, allowing an external process
to decide whether and when to start tracks.

If an initiator is provided, TOMHT passes residual detections to it and treats
the returned Stone Soup tracks as candidate internal starts. Candidate starts may
be capped or skipped by current internal-start guardrails before insertion. The
tracker does not apply state-layout-specific candidate validity checks; those
belong in the configured initiator. The exact guardrail details are
implementation controls and should not be treated as a stable integration
contract.

Conceptually, the tracker does not distinguish between simple one-detection
initializers, M/N initiators, or domain-specific initiators. They are all
caller-owned start generators.

### `deleter`

The deleter is optional.

If provided, it is used as a Stone Soup whole-track deletion hook after N-scan
pruning. Score-based deletion still runs even when a deleter is configured. The
deleter is a domain-specific hook for rules such as field-of-view exit, lifetime
limits, or application-specific invalidity checks.

### `detection_probability_model`

The detection-probability model is optional.

If omitted, TOMHT wraps scalar `TOMHTParams.prob_detect` and
`TOMHTParams.clutter_density` in a `ConstantDetectionProbabilityModel`.

These scalar parameters are caller-provided model assumptions. They are not
tracker mechanics in the same sense as frontier caps, N-scan window, or overload
guards. Use them when a constant detection probability and constant clutter
density are a reasonable approximation for the deployment. Use a custom
`DetectionProbabilityModel` when those quantities depend on sensor, state,
geometry, or scan context.

If supplied, the DPM provides dynamic per-hypothesis detection probability and
clutter density.

### `output_track_id_mapper`

The output ID mapper is optional.

If omitted, TOMHT assigns dense public output IDs in first-publication order.

If supplied, it maps the internal TOMHT logical track ID to the public
`Track.id` assigned when a tree first becomes published. A custom mapper is
responsible for returning unique, non-`None`, non-reused public IDs.

---

## 5. Local scoring model

For each local track hypothesis, TOMHT uses the log-likelihood-ratio style
increments:

```text
hit  = log(P_D) - log(lambda) - NLL
miss = log(1 - P_D)
```

where:

- `P_D` is detection probability for the predicted target state in the current
  scan,
- `lambda` is clutter density in the relevant measurement space,
- `NLL = -log p(z | x)` is the measurement likelihood negative log likelihood
  returned by the hypothesiser.

`P_D` and `lambda` are supplied by the caller's sensing model, either through
the scalar defaults or through a custom DPM. Treat them as calibration/model
inputs, not as generic tuning knobs. They are special because every downstream
score threshold depends on them: confirmation, deletion, and publication all
operate on accumulated scores built from these per-hypothesis increments. If
`P_D` or `lambda` are wrong, those thresholds will be operating on
miscalibrated evidence.

Unused detections do not receive a separate score term. In the all-clutter
baseline formulation, unused-detection terms cancel out; the clutter contrast is
already carried by the hit term through `-log(lambda)`.

### Unit contract

`lambda` must use the same measurement-space coordinates as the NLL computation.

For example, if the hypothesiser evaluates a Gaussian likelihood in
bearing/range coordinates, the clutter density must be in detections per
bearing/range measurement volume per scan. If the measurement coordinates are
rescaled, the Gaussian normalization term inside the NLL and the clutter density
term must transform consistently.

This unit contract is important for scale-invariant scores.

---

## 6. DetectionProbabilityModel

Advanced integrations can provide a dynamic detection-probability model.

The protocol is:

```python
class DetectionProbabilityModel(Protocol):
    def detection_probability(
        self,
        *,
        track_id: object | None,
        prediction: Prediction,
        caller_scan_context: object | None,
    ) -> float: ...

    def clutter_density(
        self,
        *,
        prediction: Prediction,
        detection: Detection | None,
        caller_scan_context: object | None,
    ) -> float: ...
```

The default constant implementation is:

```python
ConstantDetectionProbabilityModel(
    prob_detect=params.prob_detect,
    clutter_density=params.clutter_density,
)
```

### When DPM methods are called

During local expansion, TOMHT scores every hypothesis returned for each active
track-tree leaf.

For each hypothesis, TOMHT calls:

```python
detection_probability(
    track_id=public_track_id_or_none,
    prediction=hypothesis.prediction,
    caller_scan_context=caller_scan_context,
)
```

This happens for both detection hypotheses and missed-detection hypotheses.

For detection hypotheses only, TOMHT also calls:

```python
clutter_density(
    prediction=hypothesis.prediction,
    detection=hypothesis.measurement,
    caller_scan_context=caller_scan_context,
)
```

`clutter_density(...)` is not currently called for missed-detection hypotheses,
because the miss score is `log(1 - P_D)` and does not contain `lambda`.

In normal local scoring, `detection` is therefore a concrete detection whenever
`clutter_density(...)` is called. The `Detection | None` signature leaves room
for future or diagnostic uses where a clutter density may be queried around a
prediction without a specific detection.

DPM methods are not currently called for initiator-root scoring or external
starts. Those roots are initialized from existence-probability priors.

### Why dynamic `P_D` matters

A scalar `P_D` is often wrong in real systems.

Detection probability can depend on:

- sensor identity,
- sensor mode,
- field of view,
- range,
- aspect,
- SNR,
- target class,
- weather or environmental state,
- or scan geometry.

The most important case is finite coverage. If a predicted target is outside a
sensor's field of view, then `P_D` should be close to zero. A miss should then
contribute approximately:

```text
log(1 - P_D) ≈ 0
```

That avoids unfairly penalizing a track for failing to appear in a sensor update
where the sensor could not have detected it.

This is especially important now that score-based deletion is enabled.

### Why dynamic `lambda` matters

Clutter density can also depend on:

- sensor identity,
- measurement space,
- range,
- bearing,
- Doppler,
- weather,
- ground/sea clutter regions,
- or other scan context.

For hit hypotheses, the DPM receives both:

- the Stone Soup prediction, and
- the concrete detection.

The strict LLR formula evaluates clutter density at the detection location. In
practice, the detection and prediction are close because the association passed a
gate, but passing both lets the DPM choose the appropriate evaluation point.

### `caller_scan_context`

`caller_scan_context` is opaque caller-provided data passed to:

```python
update_tracker(..., caller_scan_context=...)
```

It is distinct from TOMHT's internal scan bookkeeping.

It can contain any application-specific information needed by the DPM, such as:

- sensor name,
- scan ID,
- sensor mode,
- platform pose,
- calibration state,
- field-of-view object,
- clutter map handle,
- weather/rain/sea-state estimate,
- or domain-specific processing state.

Do not rely on `detections[0]` to infer sensor context. Empty scans are valid and
important.

### Track ID passed to the DPM

The `track_id` passed to the DPM is the public output ID if the track tree has
already been published.

For unpublished trees, `track_id` is `None`.

TOMHT does not pass internal logical track IDs to the DPM. This preserves the
public/private boundary:

- published tracks have a caller-visible identity,
- unpublished tentative trees are internal,
- internal IDs remain available for debugging and evaluation through metadata
  and snapshots.

A track that is first published at the end of scan `N` will only have a public ID
available to the DPM from scan `N+1` onward.

---

## 7. Example DPMs

### Constant scalar model

Most simple applications can use the default scalar model:

```python
params = TOMHTParams(
    prob_detect=0.9,
    clutter_density=1e-4,
)

tracker = TOMHTTracker(
    updater=updater,
    predictor=predictor,
    initiator=initiator,
    params=params,
)
```

This is equivalent to using `ConstantDetectionProbabilityModel`.

### Finite-field-of-view model

A custom DPM can make misses free outside coverage:

```python
class FovDetectionProbabilityModel:
    def __init__(self, clutter_density_by_sensor):
        self.clutter_density_by_sensor = clutter_density_by_sensor

    def detection_probability(
        self,
        *,
        track_id,
        prediction,
        caller_scan_context,
    ) -> float:
        sensor = caller_scan_context.sensor

        if not sensor.fov_contains_prediction(prediction):
            return 0.0

        return sensor.nominal_detection_probability(prediction)

    def clutter_density(
        self,
        *,
        prediction,
        detection,
        caller_scan_context,
    ) -> float:
        sensor_name = caller_scan_context.sensor_name
        return self.clutter_density_by_sensor[sensor_name]
```

The exact implementation is application-specific. The important part is that the
DPM receives Stone Soup prediction/detection objects and caller scan context, so
it can apply the application's own sensor model.

---

## 8. Initiation and starts

TOMHT does not own a generic birth-state initializer.

This is deliberate. A single-detection track start requires assumptions about
the state space, observed dimensions, unobserved velocity/acceleration priors,
and covariance. Those assumptions are domain-specific and belong in the caller's
Stone Soup initiator.

### Internal starts through `initiator`

If an initiator is supplied, TOMHT:

1. expands existing track trees,
2. determines residual detections unused by surviving active leaves,
3. passes residual detections to the initiator,
4. treats returned tracks as candidate internal starts,
5. applies current internal-start capping/guardrails,
6. inserts retained candidates as new internal track trees,
7. scores their roots using an initial existence prior.

The tracker does not conceptually care whether the initiator is:

- a simple measurement initiator,
- a two-point initializer,
- an M/N initiator,
- or a domain-aware ISAC start generator.

However, the current implementation still has internal-start guardrails and a
per-scan cap. These are intended as tractability controls, not as part of the
mathematical initiation model, and they may be revised. Layout-specific state
validity checks should be implemented by the initiator.

### Initial score for initiator-created starts

Initiator-created roots use:

```python
params.initiator_start_initial_existence_probability
```

converted internally to log-odds.

If the returned Stone Soup track has valid metadata:

```python
track.metadata["existence_log_odds"]
track.metadata["existence_probability"]
```

that value overrides the configured default for that start. The precedence is:

```text
valid metadata["existence_log_odds"]
    > valid metadata["existence_probability"]
    > configured default probability converted to log-odds
```

Invalid metadata at one level falls through to the next level. Log-odds metadata
is accepted as any finite float and is not clamped. Probability metadata must
satisfy `0.0 < p < 1.0`.

`metadata["existence_probability"]` is the preferred way for an initiator to
communicate manually calibrated candidate confidence. `metadata["existence_log_odds"]`
is preferred for upstream systems that already compute additive LLR/evidence,
because TOMHT can use it directly without converting through probability.
TOMHT may also use valid probability metadata as a candidate-quality hint when
internal-start capping/ordering is needed. Exact internal-start ordering remains
an implementation detail.

### Residual detections and `get_unused_detections()`

`get_unused_detections()` is primarily intended for integrations that do not
configure an internal initiator and instead run an external start process. In
that mode, TOMHT returns the residual detections from the most recent scan.

If an internal initiator is configured, residual detections passed to that
initiator should generally be considered consumed by the internal-start path,
even if the initiator returns no retained starts after capping.
Guardrail-blocked birth processing may leave residuals available, but callers
should not rely on that as a stable external-start mechanism.

### Choosing a prior for simple one-detection starts

For a simple one-detection initiator, one principled way to choose the initial
existence probability is:

```text
logit(P_init) = log(P_D * beta_NT / lambda)
```

where:

- `beta_NT` is new-target density in the same measurement-space units as
  clutter density,
- `lambda` is clutter density,
- `P_D` is detection probability.

Equivalently:

```text
P_init = sigmoid(log(P_D) + log(beta_NT) - log(lambda))
```

TOMHT does not currently expose `beta_NT` as a core parameter. This formula is
guidance for choosing an existence-probability prior appropriate for the
initiator and environment.

### Internal initiator vs external starts

Use an internal initiator when TOMHT should own the residual-detection handoff:
residual detections are passed to the initiator, candidate starts may be subject
to internal capping/guardrails, and retained starts enter the normal tree
lifecycle.

Use external starts when another caller-side process has already decided which
starts should enter the tracker. External starts bypass the internal initiator
path and its current candidate capping/guardrails. They are still inserted as
TOMHT trees and go through TOMHT confirmation, deletion, and publication after
insertion.

It is valid to use `initiator=None` and manage all starts externally. That is
the cleanest integration pattern when the application already has a
domain-specific start/confirmation process.

### External starts

External starts are inserted with:

```python
tracker.add_external_starts(time, starts)
```

This must be called after an `update_tracker(...)` call at the same timestamp.

The preceding update establishes the scan timestamp and internal bookkeeping for
the external insertion point. It does not have to contain detections; an empty
single-sensor update is a valid way to advance the tracker to a timestamp before
adding external starts.

External starts are assumed to come from a caller-side process that has already
vetted the start. They use:

```python
params.external_start_initial_existence_probability
```

converted to log-odds, unless valid per-track metadata
`"existence_log_odds"` or `"existence_probability"` overrides it. The same
precedence is used as for initiator-created starts:

```text
valid metadata["existence_log_odds"]
    > valid metadata["existence_probability"]
    > configured default probability converted to log-odds
```

Use probabilities for convenient human-facing defaults. Use log-odds when an
external start generator already computes additive LLR/evidence; TOMHT accepts
any finite log-odds value directly and does not clamp it.

External starts may also provide `"age"` and `"hits"` metadata.

---

## 9. Confirmation, deletion, and publication

TOMHT separates three concepts:

1. **confirmation**: internal tree lifecycle state,
2. **deletion**: removal of a whole track tree,
3. **publication**: external output visibility.

These are intentionally distinct.

### Confirmation

Each track tree has a lifecycle state:

```text
tentative -> confirmed
```

Confirmation is sticky. A confirmed tree does not become tentative again.

The default confirmation threshold is controlled by:

```python
track_confirmation_existence_probability
```

Internally this is converted to log-odds and compared against the maximum
accumulated score over active leaves.

### Deletion

Score-based deletion is enabled by default and controlled by:

```python
track_deletion_existence_probability
```

This is also converted to log-odds and compared against the maximum accumulated
score over active leaves after N-scan pruning.

Deletion removes the whole track tree.

Score-based deletion always runs. In addition, TOMHT uses one of two
non-score deletion lanes:

- without a custom Stone Soup deleter, TOMHT applies the native miss-count lane;
- with a custom Stone Soup deleter, the deleter lane replaces the native
  miss-count lane.

The native miss-count threshold is not raw `max_missed`; it uses an N-scan-aware
floor:

```text
effective_miss_threshold = max(max_missed, ns_scan_window + 1)
```

This avoids deleting a whole tree before the N-scan machinery has had enough
history to commit safely.

### Publication

Publication is the output boundary. Each tree has a publication state:

```text
unpublished -> published
```

Publication is sticky. Once a tree is published, it remains published until
deleted.

Publication is controlled by:

```python
publish_lifecycle_states
publish_min_hits
publish_min_age
publish_min_existence_probability
```

By default, TOMHT publishes confirmed MAP-selected tracks only:

```python
publish_lifecycle_states=("confirmed",)
```

Tentative trees can still exist internally and can be inspected with:

```python
get_map_output_tracks(include_unpublished=True)
```

---

## 10. Public IDs, internal IDs, and metadata

TOMHT distinguishes internal and public track identity.

### Internal track ID

The internal track ID is assigned when a `TrackTree` is created. It is stable and
not reused. Internal IDs may have gaps because many tentative trees are never
published.

Internal IDs are useful for debugging, evaluation, and internal snapshots.

### Public track ID

The public `Track.id` is assigned when a tree first becomes published.

By default, public IDs are dense integers in first-publication order. Callers can
provide `output_track_id_mapper` to map internal logical IDs to their own public
ID namespace.

The default dense mapper does not reuse public IDs after deletion. If a custom
`output_track_id_mapper` is supplied, uniqueness and non-reuse are the caller's
responsibility.

### Output metadata

TOMHT-produced output tracks include an explicit metadata projection. Not every
field should be treated as equally stable.

The intended stable/public metadata contract is:

```python
metadata["internal_track_id"]      # internal TOMHT logical ID
metadata["public_track_id"]        # public Track.id, or None for unpublished inspection
metadata["existence_log_odds"]     # accumulated score as log-odds-style value
metadata["existence_probability"]  # sigmoid(existence_log_odds)
metadata["lifecycle_state"]        # "tentative" or "confirmed"
metadata["publication_state"]      # "unpublished" or "published"
metadata["age"]
metadata["hits"]
metadata["missed_count"]
```

For published tracks:

```python
track.id == track.metadata["public_track_id"]
```

For unpublished inspection tracks returned by
`get_map_output_tracks(include_unpublished=True)`, `Track.id` is an
inspection-only internal ID and:

```python
metadata["public_track_id"] is None
```

Additional metadata fields, such as node IDs, last-detection keys, root source,
or birth scan index, may be present for diagnostics and development. Treat those
as inspection aids rather than long-term integration contract unless they are
explicitly promoted in this document.

Do not use legacy `metadata["track_id"]` in new integration code. Use
`metadata["internal_track_id"]` for the internal TOMHT ID and `Track.id` /
`metadata["public_track_id"]` for the public output ID.

---

## 11. Common integration patterns

### Simple single-sensor tracker

Use scalar defaults:

```python
params = TOMHTParams(
    prob_detect=0.9,
    clutter_density=1e-4,
)

tracker = TOMHTTracker(
    updater=updater,
    predictor=predictor,
    initiator=initiator,
    params=params,
)
```

### Single sensor with finite coverage

Provide a DPM that returns `P_D≈0` outside coverage and a scan-appropriate clutter
density.

Call:

```python
tracker.update_tracker(
    timestamp,
    detections,
    caller_scan_context=sensor_context,
)
```

### Multi-sensor tracker

Call `update_tracker(...)` once per sensor update:

```python
tracker.update_tracker(t, detections_a, caller_scan_context=context_a)
tracker.update_tracker(t, detections_b, caller_scan_context=context_b)
```

Each call should contain one measurement space. Do not concatenate detections
from multiple sensors into one update.

### External-start-only integration

Disable internal starts by not passing an initiator:

```python
tracker = TOMHTTracker(
    updater=updater,
    predictor=predictor,
    initiator=None,
    params=params,
)

time, tracks = tracker.update_tracker(t, detections)
unused = tracker.get_unused_detections()

external_starts = external_start_logic(unused)
tracker.add_external_starts(t, external_starts)
```

### Internal initiator integration

Pass a Stone Soup initiator:

```python
tracker = TOMHTTracker(
    updater=updater,
    predictor=predictor,
    initiator=initiator,
    params=params,
)
```

Set `initiator_start_initial_existence_probability` according to how selective
the initiator is. A simple one-detection initiator should usually get a lower
prior than a domain-specific M/N or externally validated start generator.

If the initiator can estimate confidence per start, set:

```python
track.metadata["existence_log_odds"] = llr
track.metadata["existence_probability"] = p
```

on returned tracks. Valid log-odds take precedence; probability remains a
convenient fallback for human-calibrated priors.

---

## 12. Calibration and tuning checklist

The most important score-related settings are:

- `prob_detect`, or custom DPM detection probability,
- `clutter_density`, or custom DPM clutter density,
- `initiator_start_initial_existence_probability`,
- `external_start_initial_existence_probability`,
- `track_confirmation_existence_probability`,
- `track_deletion_existence_probability`,
- publication gates.

Practical signs of miscalibration:

- tracks die when leaving sensor coverage:
  - make `P_D` state/context-dependent and return near zero outside coverage,
- too many false tentative tracks:
  - tune initiator priors, confirmation threshold, birth guards, and publication
    gates,
- too few published tracks:
  - lower confirmation/publication thresholds or increase initial priors,
- real tracks get deleted too quickly:
  - lower effective `P_D` for missed scans where the target may not be visible,
    or lower deletion strictness,
- low-quality tracks never die:
  - raise deletion threshold, check miss policy, or inspect DPM values.

Non-score safety valves include:

- `max_children_per_track`,
- `max_leaves_per_track_tree`,
- `max_births_per_scan`,
- birth load guards,
- cluster overload splitting,
- and N-scan window.

These are tractability controls, not replacements for a coherent scoring model.

Calibration scale example:

```text
With P_D = 0.9, a miss contributes log(1 - 0.9) ≈ -2.3.
Confirmation at P = 0.9 is logit(0.9) ≈ +2.2.
Deletion at P = 0.01 is logit(0.01) ≈ -4.6.
A track at the confirmation threshold therefore needs to lose about 6.8
log-odds, or roughly three consecutive high-P_D misses, to reach deletion.
```

---

---

## 13. Parameter reference quick map

This is not a complete substitute for `TOMHTParams`, but it groups the most
important public parameters by purpose.

### Sensor/scoring calibration

- `prob_detect`: scalar default `P_D` used by `ConstantDetectionProbabilityModel`.
- `clutter_density`: scalar default `lambda` used by
  `ConstantDetectionProbabilityModel`.
- `log_epsilon`: numerical floor for safe logarithms.
- `mahalanobis_gate_threshold`: local association gate used by the tracker-owned
  default hypothesiser.

### Start priors

- `initiator_start_initial_existence_probability`: default prior for starts
  returned by the internal initiator.
- `external_start_initial_existence_probability`: default prior for external
  starts passed through `add_external_starts(...)`.

### Tree lifecycle and publication

- `track_confirmation_existence_probability`: sticky confirmation threshold.
- `track_deletion_existence_probability`: whole-tree score deletion threshold.
- `publish_lifecycle_states`: lifecycle states eligible for first publication.
- `publish_min_hits`: minimum hits for first publication.
- `publish_min_age`: minimum age for first publication.
- `publish_min_existence_probability`: optional existence gate for first
  publication.

### Tractability and guardrails

These controls are implementation safety valves and are subject to revision:

- `max_children_per_track`
- `max_leaves_per_track_tree`
- `max_births_per_scan`
- birth load guards
- `max_global_hypotheses`
- `max_projected_cluster_combinations`
- overload-splitting parameters
- historical conflict relaxation
- `ns_scan_window`

Tune the probabilistic model first. Use these controls to keep computation
bounded once the scoring model is roughly calibrated.

## 14. Current limitations and assumptions

Current important assumptions:

- one sensor / measurement space per `update_tracker(...)` call,
- each track can be associated with at most one detection per update,
- the hypothesiser distance for detection hypotheses must be NLL only,
- custom hypothesisers must return valid Stone Soup `MultipleHypothesis`
  objects with finite `SingleDistanceHypothesis` entries, exactly one miss, and
  detection hypotheses tied to the original scan detections,
- clutter density units must match the NLL measurement coordinates,
- the tracker does not own generic birth-state initialization,
- current internal-start candidates may be capped or skipped by implementation
  guardrails before insertion,
- internal-start candidate validity remains initiator-owned; TOMHT does not
  interpret state-vector components as specific coordinates in the birth path,
- no tracker-core `beta_NT` / birth-density parameter exists,
- the DPM sees public IDs only after publication,
- unpublished trees pass `track_id=None` to the DPM,
- existence probabilities are score-implied and only as calibrated as `P_D`,
  `lambda`, and initial priors,
- only the documented stable metadata fields should be treated as integration
  contract,
- thresholds should be tuned per deployment.

These assumptions are deliberate for now. They keep TOMHT focused on MHT
machinery while leaving sensor physics and domain-specific initiation to caller
components.
