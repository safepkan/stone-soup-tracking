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
- solver-backed K-best solving within rebuilt clusters,
- optional overload split solving inside one original cluster solve,
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

This separation is intentional. TOMHT should not know what sensor produced the detections,
what measurement coordinates they live in, or whether they come from real or simulated data.
It should only know how to manage hypotheses and apply scores once the caller has
provided the Stone Soup components and probabilistic assumptions.

---

## 2. Core tracker usage

### Typical setup

The basic setup is:

```python
tracker = TOMHTTracker(
    updater=updater,
    predictor=predictor,  # pass predictor= OR hypothesiser=, not both
    initiator=initiator,  # optional
    deleter=deleter,      # optional
    params=params,
    detection_probability_model=dpm,  # optional
)

time, published_tracks = tracker.update_tracker(
    timestamp,
    detections,
    caller_scan_context=sensor_context,
)
```
The `caller_scan_context` argument and the optional `detection_probability_model` are central to non-trivial integrations; see Section 6 for their detailed contracts.

For integrations where another caller-side process decides which tracks should enter the tracker, pass `initiator=None` at construction and use `add_external_starts(...)` after each `update_tracker(...)` call at the same timestamp to inject caller-vetted starts.

The tracker also supports the Stone Soup iterator-style usage pattern via `_TrackerMixInUpdate`, where detections are pulled from a `Detector` passed to the constructor. This mode is convenient for simple cases but cannot accommodate `caller_scan_context` or `add_external_starts(...)`. Use the push pattern above for any integration that needs scan-dependent detection-probability models or external start handoff.

See Section 11 for worked examples of each integration pattern.

### Public methods

#### Main methods

```python
update_tracker(time, detections, *, caller_scan_context=None)
tracks
get_unused_detections()
add_external_starts(time, starts)
```

`update_tracker(...)` processes one scan/update and returns the current
published output tracks.

`tracks` is a convenience property returning the same published output view
for the most recently processed scan. In normal use, the set returned by
`update_tracker(...)` and the value of `tracker.tracks` immediately afterwards
represent the same output boundary.

`get_unused_detections()` returns residual detections from the most recent scan
primarily for external-start workflows where no internal initiator is configured.

`add_external_starts(...)` injects caller-vetted external starts after an
`update_tracker(...)` call at the same timestamp.

#### Inspection and debug methods

These helpers are intended for inspection, evaluation, and debugging.
They are not part of the stable integration contract and may change without notice.

```python
get_map_output_tracks(include_unpublished=False)
get_map_hypothesis_snapshot()
get_n_scan_commitment_snapshot()
get_last_cluster_snapshots()
get_track_tree_snapshot()
print_summary_stats()
```

`get_map_output_tracks(include_unpublished=True)` can be used for inspection of
unpublished MAP-selected tracks.

`get_map_hypothesis_snapshot()` exposes the internal MAP-selected leaf nodes for
debugging and evaluation. It always returns a snapshot; before any scan, that
snapshot is the empty MAP. It is not the public output boundary.

`print_summary_stats()` prints aggregate instrumentation collected in
`ScanStats` when `collect_stats=True`.

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
NLL-distance hypothesiser, parameterized by `params.mahalanobis_gate_threshold`
(see §14). The default evaluates a Gaussian likelihood in the measurement space
implied by the supplied `Predictor` and `Updater` and gates detections by
Mahalanobis distance.

If `hypothesiser` is supplied, it replaces the default. TOMHT then never
constructs an internal hypothesiser and never applies its own gating on top of
the one provided.

#### When to provide a custom hypothesiser

The tracker-owned default is sufficient when the measurement-space innovation
is well-modeled as `z − h(x)` with a Gaussian likelihood `N(0, S)`. It uses
only standard Stone Soup interfaces (`Predictor`, `Updater`, and each
detection's `measurement_model`), so it has no way to learn about
sensor-specific structure that isn't expressible through those interfaces.

Provide a custom hypothesiser when that model breaks down for your sensor.
Typical reasons:

- **Folded or ambiguous measurement components.** One or more measurement
  components are observed only modulo some interval, so the raw `z − h(x)`
  is not the right innovation. Common radar examples are Doppler folding
  (velocity ambiguous modulo the unambiguous Doppler interval), PRF / range
  folding (range ambiguous modulo `c / (2·PRF)`), and angle ambiguities in
  receive arrays with element spacing larger than `λ/2`. Standard Stone Soup
  interfaces carry no notion of the wrap period, so the caller must inject
  it — typically by folding the innovation into a symmetric interval around
  zero before computing the Mahalanobis distance and NLL.
- **Discrete ambiguity sets with several admissible candidates.** A single
  detection is consistent with several candidate measurement values, and
  local association needs to evaluate the candidate closest to the
  prediction (or emit one detection-hypothesis per candidate) rather than
  the raw reported value.
- **Non-Gaussian likelihoods.** The measurement likelihood is heavy-tailed,
  a mixture, or otherwise not `N(z; h(x), S)`, and a closed-form Gaussian
  NLL is the wrong objective.

A custom hypothesiser is also the natural place to put any sensor-specific
gating logic, since the tracker applies no additional gating on top of the
hypothesiser's output.

#### Custom hypothesiser contract

A custom hypothesiser must implement the Stone Soup hypothesiser signature:

```python
def hypothesise(
    self,
    track: Track,
    detections: Iterable[Detection],
    timestamp,
    **kwargs,
) -> MultipleHypothesis: ...
```

and must satisfy the following constraints for each track leaf and scan:

- **Returns a Stone Soup `MultipleHypothesis`** whose entries are
  `SingleDistanceHypothesis` instances. Subclasses are permitted; TOMHT does
  not consume hypothesis fields other than those listed in this contract, so
  custom subclasses may carry additional fields for diagnostics without
  affecting tracker behavior.
- **Includes exactly one missed-detection hypothesis.** Its `distance` field
  is a sentinel and is ignored by tracker scoring; any finite value
  (the default uses `0.0`) is acceptable. The miss score is computed by the
  tracker as `log(1 − P_D)`.
- **Detection-hypothesis `distance` is the full measurement-likelihood NLL,
  and nothing else:**
  ```text
  distance = NLL = −log p(z | x)
  ```
  For a Gaussian innovation with covariance `S` in `d` dimensions, NLL must
  include the full normalization:
  ```text
  NLL = 0.5 · ( d · log(2π) + log|S| + (z − ẑ)ᵀ S⁻¹ (z − ẑ) )
  ```
  Not only the Mahalanobis term `½ · (z − ẑ)ᵀ S⁻¹ (z − ẑ)`, and not
  `½ · (log|S| + Mahalanobis)`. The §5 unit contract between `lambda` and
  NLL only holds when both are densities in the same measurement-space
  coordinates; dropping constants from NLL silently miscalibrates every
  score threshold downstream. For non-Gaussian likelihoods the same
  principle applies: `distance` must be the full negative log density,
  not a partial form. The distance must not include detection-probability
  factors, clutter-density factors, birth terms, or any other score
  offsets — TOMHT applies those separately.
- **All emitted distances are finite,** both for the miss and for detection
  hypotheses.
- **Detection hypotheses reference the original detection objects** from
  the current scan directly (no copies, no reconstructed detections). TOMHT
  recovers scan-local detection identity from object identity.
- **`measurement_prediction` is attached to each detection hypothesis,**
  set to the predicted measurement used to compute that hypothesis's NLL.
  Stone Soup updaters consume this field to avoid recomputing the predicted
  measurement during the update step; omitting it forces a slower or
  potentially inconsistent update path depending on the updater.
- **Gating is the hypothesiser's responsibility.** Detections that should
  not be associated with this track should simply be omitted from the
  returned `MultipleHypothesis`. TOMHT applies no additional gating on top
  of a custom hypothesiser's output. `params.mahalanobis_gate_threshold`
  is consumed only by the tracker-owned default and has no effect when a
  custom hypothesiser is supplied.
- **The input `track` carries TOMHT metadata.** During local expansion,
  TOMHT reconstructs a Stone Soup `Track` for the active leaf and populates
  metadata such as `internal_track_id`, `public_track_id`,
  `lifecycle_state`, `publication_state`, `age`, `hits`, and
  `missed_count`. Advanced custom hypothesisers may use this metadata,
  including lifecycle state, to implement confirmation-state-dependent gates
  or other sensor-specific policy. The tracker-owned default hypothesiser
  does not currently vary gates by confirmation state.
- **A `predictor` attribute is exposed for tracker wiring.** The
  hypothesiser must expose a `predictor` attribute — typically a Stone
  Soup `Property`, as in the default implementation — that the tracker
  uses for state prediction outside of local association, such as
  advancing track states across empty scans.

Timestamp handling within a scan is the hypothesiser's choice. The default
honors per-detection timestamps when `detection.timestamp` differs from the
scan `timestamp`, by predicting the track to each distinct detection
timestamp before computing the innovation. Custom hypothesisers may either
honor per-detection timestamps or treat the scan timestamp as authoritative;
either policy is acceptable as long as it is applied consistently within a
scan.

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

If provided, it is used as a Stone Soup deletion hook after N-scan pruning.
Score-based deletion still runs even when a deleter is configured. The deleter
is a domain-specific hook for rules such as field-of-view exit, lifetime limits,
or application-specific invalidity checks.

If no deleter is supplied, TOMHT resolves an internal successive-miss-count
deleter from `TOMHTParams`. The default miss-count deleter is intentionally
minimal and has no awareness of sensor identity or scan context, so a custom
deleter is the recommended path as soon as deletion logic needs to be sensor- or
context-aware.

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

### Python 3.14 note

If a custom Stone Soup component declares `Property` fields in a module
using `from __future__ import annotations`, prefer the explicit
`Property(Type,...)` form, or mirror the built-in version gate,
because Stone Soup may not recover annotation-only property types there.

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

To see why, substitute `NLL = −log p(z | x)` back into the hit increment to get
its log-likelihood-ratio form:

```text
hit = log( P_D · p(z | x) / lambda )
```

The dimensionless ratio `p(z | x) / lambda` is what gives the score its scale
invariance, so the two terms must be evaluated as densities in the same
measurement-space coordinates. Truncating normalization constants from NLL —
for example, keeping only the Mahalanobis term — leaves a Jacobian factor
uncancelled and breaks that invariance.

For example, if the hypothesiser evaluates a Gaussian likelihood in
bearing/range coordinates, the clutter density must be in detections per
bearing/range measurement volume per scan. If the measurement coordinates are
rescaled, the Gaussian normalization term inside the NLL and the clutter
density term must transform consistently.

### Why log-likelihood-ratio scoring?

Stone Soup hypothesisers can in principle return any "distance" that makes
sense to their consuming associator — for some simple associators only a
ranking is needed, and for a GNN associator only summability across an
assignment.

TOMHT needs more. Per-hypothesis increments are accumulated into running
track scores that are compared against confirmation, deletion, and publication
thresholds, and the same accumulated scores drive the assignment problem
solved at each scan. For those thresholds to be principled, and for scores
to remain comparable across scans and across sensors, the per-hypothesis
increment must be a log-likelihood ratio.

The hit increment is the log-ratio between the "target generated this
detection" and "clutter generated this detection" hypotheses for the same `z`.
The miss increment is its counterpart for the "target present, no detection
produced" alternative. Summed over a track's history, these increments form
the accumulated LLR — the standard MHT track score (Reid 1979; Blackman &
Popoli 1999) — and a threshold on the accumulated score corresponds via the
sigmoid to a threshold on existence probability, which is what gives the
calibration values in Section 12 their meaning.

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

A scalar `P_D` is often wrong in real systems. Detection probability depends on
the sensor and its mode, target-relative geometry (range, aspect, field-of-view
coverage), target properties (class, SNR), and environmental conditions.

The most important case is finite coverage. If a predicted target is outside a
sensor's field of view, then `P_D` should be close to zero. A miss should then
contribute approximately:

```text
log(1 - P_D) ≈ 0
```

That avoids unfairly penalizing a track for failing to appear in a sensor update
where the sensor could not have detected it.

This becomes especially important in conjunction with score-based deletion.

### Why dynamic `lambda` matters

Clutter density similarly depends on the sensor and measurement space, on the
location within that measurement space (range, bearing, Doppler), and on
environmental factors such as weather or ground/sea clutter regions.

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

The `track_id` passed to the DPM is the public output ID under which this track
tree's output most recently appeared to the caller, or `None` if it has not yet
appeared in any output scan.

Under TOMHT's current output policy this is equivalent to "the public ID
assigned to this tree when it was first published, retained until deletion."
If the output policy is later extended to support stitching, ID smoothing across
MAP flips, or other continuity strategies, `track_id` will reflect the public ID
the caller most recently saw — which is what caller-side state keyed by ID is
built around.

The snapshot used here reflects the previous scan's output, not the in-progress
one. During scan N's scoring, the DPM sees the mapping from scan N−1's output,
matching what the caller has actually received.

TOMHT does not pass internal logical track IDs to the DPM. This preserves the
public/private boundary:

- published output is the caller-visible identity surface,
- unpublished tentative trees remain internal,
- internal IDs remain available for debugging and evaluation through metadata
  and snapshots.

A track tree that is first published at the end of scan N will only have a
`track_id` available to the DPM from scan N+1 onward.

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
- or a domain-aware start generator.

However, the current implementation still has internal-start guardrails and a
per-scan cap. These are intended as tractability controls, not as part of the
mathematical initiation model, and they may be revised. Layout-specific state
validity checks should be implemented by the initiator.

### Initial score for initiator-created starts

Initiator-created roots use:

```python
params.initiator_start_initial_existence_probability
```

converted internally to log-odds, `log( p / (1 − p) )`.

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

The two fields are equivalent ways of expressing the same prior.
`metadata["existence_probability"]` is usually the more convenient form for
human-calibrated priors. Use `metadata["existence_log_odds"]` when an upstream
system already computes additive LLR/evidence, to avoid a probability roundtrip.

TOMHT may also consult valid probability metadata as a candidate-quality hint
for internal-start capping and ordering. The exact ordering strategy is an
implementation detail.

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

- `logit(p) = log( p / (1 − p) )` is the log-odds, and `sigmoid` is its inverse,
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
TOMHT trees. After insertion, TOMHT updates the MAP view, runs the same
score-based confirmation pass used in the normal scan lifecycle, and then
applies output publication. Full lifecycle deletion, N-scan pruning, cluster
rebuild, and scan stats are not run from `add_external_starts(...)`.
For immediate publication of an externally started track, ensure that its
existence probability exceeds the confirmation and publication thresholds.

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

With the defaults, `external_start_initial_existence_probability=0.95` crosses
the default `track_confirmation_existence_probability=0.9`, so an external
start is confirmed and published immediately unless publication gates are
tightened. A lower per-track `"existence_probability"` or `"existence_log_odds"`
can keep the inserted tree tentative and unpublished while still making it
available through `get_map_output_tracks(include_unpublished=True)`.

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

Score-based deletion always runs. In addition, TOMHT runs one configured
deleter:

- without a custom Stone Soup deleter, TOMHT resolves the default internal
  miss-count deleter from `TOMHTParams`;
- with a custom Stone Soup deleter, TOMHT uses that deleter instead of the
  default miss-count deleter.

The default miss-count threshold is not raw `max_missed`; it uses an
N-scan-aware floor:

```text
effective_miss_threshold = max(max_missed, ns_scan_window + 1)
```

This avoids deleting a whole tree before the N-scan machinery has had enough
history to commit safely.

The default miss-count deleter reads TOMHT's reconstructed track
`metadata["missed_count"]` and counts every miss equally, with no awareness of
sensor identity, geometry, or scan context. In particular, a track predicted to
be outside a sensor's field of view will accumulate misses that count toward the
threshold even though the sensor could not have detected the target. Any
sensor- or context-aware deletion logic should go in a custom Stone Soup
deleter, which is the recommended path for field-of-view exit, lifetime limits,
sensor/context-aware invalidity, or application-specific deletion.

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

By default, TOMHT publishes confirmed tracks only:

```python
publish_lifecycle_states=("confirmed",)
```

TOMHT currently selects output tracks by MAP selection over the track tree
hypotheses. The public contract is the resulting output tracks themselves —
their IDs, states, and metadata — not the underlying selection strategy.
Future versions may apply continuity-preserving policies (stitching across
track-tree fragments, ID smoothing across MAP flips between competing
hypotheses, and similar) without breaking this API.

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
  - tune initiator priors, confirmation threshold, birth caps/guards, and
    publication gates,
- too few published tracks:
  - lower confirmation/publication thresholds or increase initial priors,
- real tracks get deleted too quickly:
  - lower effective `P_D` for missed scans where the target may not be visible,
    or lower deletion strictness,
- low-quality tracks never die:
  - raise deletion threshold, check miss policy, or inspect DPM values.

Non-score safety valves include:

- `max_children_per_leaf`,
- `max_leaves_per_track_tree`,
- `max_births_per_scan`,
- birth load guards,
- cluster overload splitting,
- and N-scan window.

These are tractability controls, not replacements for a coherent scoring model.
`max_children_per_leaf` is a per-active-leaf local branching cap.
`max_births_per_scan` defaults to `10` and remains a guardrail. Birth load
guards default to disabled and should be set only for scenario-specific
emergency load control. Confirmation-state-dependent gating is best handled in
a custom hypothesiser using TOMHT track metadata, not by adding policy to the
tracker-owned default hypothesiser.

Calibration scale example:

```text
With P_D = 0.9, a miss contributes log(1 - 0.9) ≈ -2.3.
Confirmation at P = 0.9 is logit(0.9) ≈ +2.2.
Deletion at P = 0.01 is logit(0.01) ≈ -4.6.
A track at the confirmation threshold therefore needs to lose about 6.8
log-odds, or roughly three consecutive high-P_D misses, to reach deletion.
```

---

## 13. Expansion/frontier instrumentation

`ScanStats.expansion_frontier` carries aggregate counters that are cheap enough
to collect by default:

- active tree/leaf counts at expansion, birth, pruning, N-scan, and lifecycle
  boundaries,
- expanded leaves split by tentative vs confirmed trees,
- raw local child candidates and created/retained local children,
- miss-child vs detection-child creation,
- MAP-selected leaf count,
- unique leaves supported by retained top-K rebuilt globals,
- unsupported leaves removed by post-solve supported-leaf pruning.

Default `SCAN ...` and `SUMMARY ...` lines are kept stable. To emit compact
expansion/frontier diagnostics, set either:

```text
TOMHT_DEBUG_EXPANSION_FRONTIER=1
```

or:

```python
params = TOMHTParams(debug_display_expansion_frontier=True)
```

This adds `EXPANSION_FRONTIER ...` per-scan lines and a
`SUMMARY expansion_frontier ...` line when summary stats are printed.

---

## 14. Parameter reference quick map

This is not a complete substitute for `TOMHTParams`, but it groups the most
important public parameters by purpose.

### Sensor/scoring calibration

- `prob_detect`: scalar default `P_D` used by `ConstantDetectionProbabilityModel`.
- `clutter_density`: scalar default `lambda` used by
  `ConstantDetectionProbabilityModel`.
- `log_epsilon`: numerical floor for safe logarithms.
- `mahalanobis_gate_threshold`: local association gate used by the tracker-owned
  default hypothesiser, not used for custom hypothesisers.

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

### Debug and stats

- `collect_stats`: retain per-scan `ScanStats` for later summaries.
- `debug_display_scan_stats`: print the standard compact per-scan diagnostics.
- `debug_display_expansion_frontier`: print opt-in expansion/frontier usefulness
  diagnostics.

### Tractability and guardrails

These controls are implementation safety valves and are subject to revision:

- `max_children_per_leaf`: per-active-leaf local branching cap. It limits
  retained hit/miss continuation candidates from each expanded leaf, not from
  the whole track tree.
- `max_leaves_per_track_tree`: optional pre-solve per-tree leaf cap.
- `max_births_per_scan`: default `10`; a deterministic internal-birth cap and
  still a guardrail, not a substitute for initiator-side quality control.
- `birth_skip_if_active_trees_above` and
  `birth_skip_if_active_leaves_above`: disabled by default (`None`) and
  available as scenario-specific emergency load guards.
- `max_global_hypotheses`
- `max_projected_cluster_combinations`
- `overload_split_enabled`
- `overload_split_projected_combination_threshold`: still conservative and a
  future review item; this cleanup does not change its default.
- `overload_split_max_edge_removals_per_cluster`
- `overload_split_solution_mode`: `"greedy_partition"` by default for the
  operational sound-but-approximate overload fallback, or
  `"conditional_exact"` for reference / higher-compute K-best-oriented
  conditioning
- `overload_split_greedy_ownership_metric`: currently only
  `"best_leaf_score"`
- `ns_scan_window`

Tune the probabilistic model first. Use these controls to keep computation
bounded once the scoring model is roughly calibrated.

## 15. Current limitations and assumptions

Current important assumptions:

- one sensor / measurement space per `update_tracker(...)` call,
- each track can be associated with at most one detection per update,
- tracker scoring assumes detection-hypothesis distance is NLL only;
  `P_D`, `lambda`, birth terms, and other factors are applied separately by the tracker,
- custom hypothesisers must satisfy the additional constraints listed in Section 4,
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
