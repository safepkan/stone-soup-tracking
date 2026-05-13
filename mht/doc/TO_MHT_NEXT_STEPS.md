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
- local expansion, internal-birth handling, clustering, overload splitting, cluster rebuild, post-solve pruning, and TOMHT utilities are no longer monolithic tracker methods,
- external starts now use an existence-prior probability mapped internally to log-odds,
- external starts can optionally override that prior per track via `Track.metadata["existence_probability"]`,
- output tracks now expose score-implied existence metadata,
- legacy unused-detection scoring has been removed,
- cluster solving now uses accumulated leaf scores directly rather than affine unused-detection pre-baking.

These changes make the next scoring/birth steps easier to reason about.

---

## Current scoring baseline

Existing-track local scoring is now the cleanest part of the scoring model.

The tracker-owned distance hypothesiser emits detection hypotheses whose distance is:

```text
NLL = -log p(z | x)
```

without detection-probability or clutter-density factors.

The scoring model then applies:

```text
hit  = log(P_D) - log(lambda) - NLL
miss = log(1 - P_D)
```

where:

- `P_D` is detection probability,
- `lambda` is clutter density,
- `lambda` must be expressed in the same measurement-space units as the hypothesiser NLL.

Legacy unused-detection scoring has been removed because the clutter-density contrast is already represented in the hit score. This was expected to change smoke/replay outputs because old scenarios were tuned against older heuristic scoring. That is acceptable: we are prioritizing a coherent scoring interpretation before retuning.

The remaining weak area is birth/initiation semantics.

---

## Direction: three initiation lanes

The migration should make three lanes explicit.

### 1. Tracker-owned single-detection births

This should become the preferred model-native internal birth path.

Conceptually:

```text
one residual detection -> one tentative track root
```

These tracks are tentative. They should not necessarily all be emitted/published. The MHT machinery, score accumulation, lifecycle, pruning, and output confirmation logic should decide which survive and which are visible externally.

The intended model-native single-detection birth score is a birth-vs-clutter log-likelihood-ratio-like term, probably of the form:

```text
birth_log_delta = log(P_D) + log(beta_NT) - log(lambda)
```

where:

- `beta_NT` is a new-target/birth density in measurement-space units,
- `lambda` is clutter density in the same measurement-space units,
- `P_D` is detection probability.

The exact naming and parameterization of `beta_NT` still needs a small design choice. The direct form is mathematically clean but can be harder for users. A later convenience parameterization might express expected new targets per scan over a surveillance/birth volume.

### 2. External starts

External starts are already on the cleaner path.

They represent caller-supplied starts that should be treated as externally confirmed or externally meaningful. They are initialized from an existence probability mapped to log-odds. The global default can be overridden by `Track.metadata["existence_probability"]`.

This lane should remain the preferred integration path for systems that already have their own domain-specific initiation or confirmation process.

### 3. Black-box initiator starts

A black-box Stone Soup initiator does not expose a clean scoring model to the tracker. It may hide multiple scans, confirmation logic, motion gates, or other domain-specific choices.

Therefore, black-box initiator output should be treated conceptually as:

```text
external-style starts generated inside the tracker for convenience
```

rather than as model-native single-detection births.

That means initiator-created roots should be scored with an explicit initial existence prior, analogous to external starts, but probably with a separate parameter and typically lower default confidence.

This keeps the scoring story honest:

- if the tracker owns the birth model, it can score it model-natively,
- if a black box produces a start, the tracker assigns an explicit existence prior.

---

## Immediate implementation plan

### Step 1: Make birth modes explicit

Add an explicit internal birth mode, likely something like:

```text
internal_birth_mode = "initiator" | "single_detection" | "disabled"
```

or equivalent names.

The exact default should be chosen conservatively. It is fine to preserve current behavior initially while making the mode visible.

Important behavior expectations:

- `"disabled"` means no internal births; external starts can still be used.
- `"initiator"` means use the configured black-box initiator, if present.
- `"single_detection"` means use tracker-owned residual-detection births.

This step is mostly about making the architecture explicit.

### Step 2: Replace fixed initiator birth penalty

The current fixed birth penalty is not a good scoring story for black-box initiator output.

For initiator-created starts, introduce something like:

```text
initiator_start_initial_existence_probability
```

and map it to log-odds for the root `log_delta`, just as external starts do.

This should replace the current fixed penalty for the black-box initiator lane.

Suggested semantics:

- valid per-track metadata may eventually override it,
- missing metadata uses the parameter default,
- invalid optional metadata should fall back rather than reject the start,
- this parameter should be separate from `external_start_initial_existence_probability`.

Potential default should probably be lower than external starts, because internally generated initiator starts may be tentative rather than externally confirmed.

### Step 3: Add tracker-owned single-detection birth mode

Implement model-native single-detection births as an opt-in mode first.

Behavior sketch:

- compute residual detections as now,
- create one candidate birth per residual detection,
- apply existing deterministic ordering/capping/load guards or close equivalents,
- create one root tree per retained residual detection,
- score each root with the model-native birth-vs-clutter term,
- do not use the black-box initiator for this mode.

The first implementation should be conservative and inspectable.

Likely new parameter:

```text
birth_density
```

or a similarly clear name, with the same measurement-space unit contract as `clutter_density`.

Do not introduce a two-point or M/N initiator in this step. If output noise becomes a problem, solve that with output confirmation gates first.

### Step 4: Add output confirmation / emit gate

Once single-detection births exist, the tracker may internally carry more tentative tracks. That is expected.

Visible output should be controlled separately from internal hypothesis maintenance.

Add an output gate based on some combination of:

- score-implied existence probability,
- minimum hits,
- minimum age.

The first version should probably preserve current output behavior by default, then allow stricter settings for cleaner consumer-facing output.

This gate should apply to returned/emitted tracks, not to whether the internal MHT state keeps tentative hypotheses.

### Step 5: Revisit pruning and expansion-volume controls

After scoring, births, and output confirmation are coherent, return to:

- score-based track/leaf pruning,
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

External starts now use an existence probability mapped to log-odds. Births and initiator starts should likewise get explicit interpretations rather than arbitrary penalties.

### Single-detection births are expected to be noisy internally

A tracker-owned single-detection birth model will create tentative tracks from clutter detections. This is not necessarily wrong.

The right question is not whether every internal tentative track is real. The right question is whether:

- scores evolve sensibly,
- low-quality tracks die or are pruned,
- and output consumers only see confirmed-enough tracks.

### Output gating is not the same as internal pruning

A confirmed-output gate should reduce visible clutter tracks without prematurely deleting internal hypotheses.

Internal pruning can come later, once score behavior is better understood.

### Black-box initiators are convenience integration points

A black-box initiator can still be useful, especially for existing runners and domain-specific setups. But unless it provides a likelihood/existence model, the tracker should not pretend to know how to score it from first principles.

Treat it as an internal convenience wrapper around external-style starts.

---

## Acceptance criteria for this subphase

This subphase is successful when:

- unused-detection scoring remains removed,
- external starts continue to have configurable/per-track existence priors,
- black-box initiator starts no longer rely on a fixed arbitrary birth penalty,
- tracker-owned single-detection births exist as an opt-in model-native lane,
- the birth score has a clear relationship to `P_D`, birth density, and clutter density,
- output confirmation/emit gating is implemented or clearly staged,
- smoke/replay outputs remain operationally usable,
- and the code/docs make the three initiation lanes clear.

At that point, it should be reasonable to return to score/frontier/expansion-volume pruning work.
