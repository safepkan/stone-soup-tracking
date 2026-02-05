# TO-MHT Next Steps

This document is for planning upcoming work in more detail than the high-level roadmap. It will evolve as we implement each step.

## 1. Scoring v2: toward a simple MHT log-likelihood

**Update (implemented in code):** Added a pluggable `ScoringModel` with a default **beta-ratio** mode that converts PDA β values into per-association log deltas and replaces the fixed unused-detection penalty with a clutter-density term. Misses now use the same common term `log(1 - P_D * P_G)` as the hit baseline (previously 0). A legacy mode preserves the previous scoring. See `tomht_tracker.py` for the new `scoring_mode` parameter. (Details mirrored into `TO_MHT_CURRENT_STATE.md` for longer-term reference.)

### 1.1. Desired model (conceptual)

We want a per-scan log-likelihood increment for each global hypothesis that roughly follows the standard MHT model:

- For each track:
  - **Hit** (track was detected and correctly associated):
    - Likelihood term based on measurement vs predicted measurement under the track’s filter.
    - Includes detection probability `P_D`.
  - **Miss** (track was not detected):
    - Likelihood term based on missed detection probability `1 - P_D`.
- For each unused detection:
  - Likelihood term based on the clutter process intensity (e.g. Poisson with rate λ over the measurement volume).
- For each birth:
  - Prior term for starting a new track (birth intensity).
- For track continuation:
  - Optional existence/termination prior (e.g. `P_S` survival probability).

At this stage, the aim is not a perfect derivation, but a **consistent and explainable scoring scheme** that uses Stone Soup’s existing quantities where possible.

### 1.2. Mapping to existing Stone Soup objects

#### 1.2.1 PDA β probabilities and a practical "v1.5" bridge

Stone Soup's `PDAHypothesiser` typically returns *normalised* association probabilities β per track
(including a missed-detection hypothesis β₀). For global MHT scoring we usually want unnormalised
likelihood-ratio-style increments. As a practical bridge that stays within Stone Soup's existing outputs:

- Let `beta0` be the miss hypothesis probability, and `betai` a hit probability.
- Define an approximate per-association likelihood ratio term:

  `logL_i ≈ log(betai) - log(beta0) + log(1 - P_D * P_G)`

This is not a perfect derivation, but it is consistent and easy to reason about as an initial scoring model.

**Numerical note:** clamp `betai` and `beta0` by an `epsilon` before taking logs to avoid `-inf` / NaNs.

Questions to resolve:

- What does the current `Hypothesis` probability/weight represent?
  - Is it directly a (normalised) association probability?
  - Can we extract or approximate a measurement likelihood `p(z | track)`?
- How to incorporate detection probability `P_D`:
  - From scenario config? From Stone Soup measurement models?
- How to approximate clutter likelihood:
  - Use existing clutter density or area from the scenarios.
  - Relate `unused_det_log_penalty` to `log(lambda)` instead of a hand-tuned constant.

### 1.3. Concrete changes planned in code

- Introduce a dedicated `ScoringModel` or helper methods in `TOMHTTracker` that:
  - Compute log-deltas for hit and miss hypotheses based on:
    - Measurement likelihood from the hypothesiser (or directly from innovations).
    - `P_D` and `1-P_D`.
  - Compute a log penalty for each unused detection based on a clutter intensity parameter.
  - Optionally, include simple existence terms for tracks and births.
- **Done (beta-ratio v1):** `ScoringModel` abstraction added with `BetaRatioScoringModel` (default) and `LegacyScoringModel` switchable via `TOMHTParams.scoring_mode`. Beta mode uses `log(betai) - log(beta0) + log(1 - P_D * P_G)` for hits, zero for misses, and `len(unused) * log(clutter_density)` for clutter; births still use `birth_log_penalty`.
- **A/B hooks:** `run_tomht_crossing.py` and `run_tomht_bearing_range.py` accept `--scoring-mode` to flip between `beta_ratio` and `legacy`, plus `--births/--no-births` and `--initial-tracks/--no-initial-tracks` to toggle initiator and initial-track usage without editing code.

- Replace the current mixture of:
  - `birth_log_penalty`
  - `unused_det_log_penalty`
  - implicit hypothesis probabilities
  with a more explicit combination, while trying to remain backward compatible enough to compare behaviours.

#### 1.3.1 Proposed ScoringModel API (implementation guide)

Introduce a `ScoringModel` abstraction so scoring can evolve independently from hypothesis generation,
pruning, and N-scan-lite logic.

```python
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

@dataclass(frozen=True)
class ScanContext:
    timestamp: Any
    detections: list  # already stably ordered
    det_index_by_obj: dict[int, int]  # id(det) -> index in detections

class ScoringModel(Protocol):
    def score_track_hypotheses(
        self,
        *,
        track: Any,
        hypotheses: Iterable,   # iterable of SingleHypothesis objects
        ctx: ScanContext,
    ) -> Mapping[object, float]:
        """Return {hypothesis_object: log_delta} for each hypothesis."""

    def score_unused_detections(
        self,
        *,
        used_det_keys: set[int],
        ctx: ScanContext,
    ) -> float:
        """Return a log_delta for clutter / unused detections for the global hypothesis."""

    def score_birth(
        self,
        *,
        birth_track: Any,
        used_det_key: int | None,
        ctx: ScanContext,
    ) -> float:
        """Return a log_delta for adding a birth (prior / evidence)."""
```

Implementation notes:
- The interface is *hypothesis-driven* (we pass hypothesis objects), so we can later compute raw likelihoods
  from `measurement_prediction` without changing the tracker loop.
- If using the β-ratio bridge, use log-space with an `epsilon` clamp for numerical stability.

### 1.4. Open design questions

- How much do we want to rely on the Stone Soup `Hypothesis` probabilities vs rolling our own likelihoods from innovations?
- Should the `ScoringModel` operate directly on `Hypothesis` objects (for access to `measurement_prediction` / raw likelihood scoring later)?
- Do we want to start with a **very simple** likelihood (e.g. Mahalanobis distance with a scaled constant) or go straight to a more accurate derivation?
- Should we integrate initiation evidence (holding track stats) into the global score explicitly, or keep it as a separate heuristic for birth ranking for now?

*(More sections will be added here for N-scan-lite, initiation rework, etc.)*
