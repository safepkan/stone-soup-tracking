# TO-MHT Next Steps

This document is for planning upcoming work in more detail than the high-level roadmap. It will evolve as we implement each step.

## 0. Step 0: Stable detection ordering and per-scan detection keys (prerequisite)

### 0.1 Why this matters

The tracker currently assigns a per-scan "detection key" by:

- converting the incoming detection collection to `det_list = list(detections)`
- using the list index as `last_det_key` / `used_det_key`

This is only deterministic if the incoming `detections` already has a stable iteration order. If `detections` is a `set` (common in Stone Soup-style measurement sets), the ordering may vary, which will silently affect:

- `last_det_key` stored in track metadata
- global deduplication signatures
- residual detection selection for births
- (later) per-track association history for N-scan-lite

So: before we touch scoring or history, we must make detection ordering explicit and stable.

### 0.2 Proposed approach

Add a scan-local stable ordering step before we assign indices. For example:

- Define a deterministic sort key derived from detection content, e.g.
  - `(timestamp, measurement_vector_components...)`
- Sort `det_list` using that key.
- Then assign indices from the sorted list.

### 0.3 Acceptance criteria

- Running the same scenario twice yields identical:
  - detection ordering per scan,
  - `last_det_key` sequences per track,
  - residual sets for birth initiation,
  - MAP result (modulo any remaining RNG elsewhere).

## 1. Scoring v2: toward a simple MHT log-likelihood

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

Questions to resolve:

- What does the current `Hypothesis` probability/weight represent?
  - Is it directly a (normalised) association probability?
  - Can we extract or approximate a measurement likelihood `p(z | track)`?
- How to incorporate detection probability `P_D`:
  - From scenario config? From Stone Soup measurement models?
- How to approximate clutter likelihood:
  - Use existing clutter density or area from the scenarios.
  - Relate `unused_det_log_penalty` to `log(lambda)` instead of a hand-tuned constant.

### 1.2.1 What Stone Soup's PDAHypothesiser gives us (important)

Stone Soup's `PDAHypothesiser` computes *unnormalised* weights roughly like:

- Miss: `w0 = 1 - P_D * P_G`
- Hit i: `wi = (pdf_i * P_D) / lambda`

and then returns `SingleProbabilityHypothesis` objects inside a `MultipleHypothesis(..., normalise=True)`.
This means each returned `hyp.probability` is a *normalised association probability* (β), i.e. the per-track
hypotheses sum to 1 (including the miss hypothesis).

For global MHT scoring we generally want something closer to the unnormalised likelihood / likelihood ratio per association.
We can "undo" the normalisation up to a constant using the ratio trick:

- Let `beta0` be the miss hypothesis probability, and `betai` a hit probability.
- Then (up to constants): `Li ≈ (betai / beta0) * (1 - P_D * P_G)`
- So: `logLi ≈ log(betai) - log(beta0) + log(1 - P_D * P_G)`

This gives a practical "v1.5 scoring" path:
- Use `log(1 - P_D * P_G)` for miss.
- Use `logLi` for hit hypotheses.
- Defer more exact modelling (Poisson clutter constants, existence priors) until after behaviour matches expectations.

### 1.3. Concrete changes planned in code

- Introduce a dedicated `ScoringModel` or helper methods in `TOMHTTracker` that:
  - Compute log-deltas for hit and miss hypotheses based on:
    - Measurement likelihood from the hypothesiser (or directly from innovations).
    - `P_D` and `1-P_D`.
  - Compute a log penalty for each unused detection based on a clutter intensity parameter.
  - Optionally, include simple existence terms for tracks and births.

- Replace the current mixture of:
  - `birth_log_penalty`
  - `unused_det_log_penalty`
  - implicit hypothesis probabilities
  with a more explicit combination, while trying to remain backward compatible enough to compare behaviours.

### 1.3.1 Proposed ScoringModel API (implementation guide)

Goal: keep scoring logic out of `TOMHTTracker` so we can iterate on scoring without touching hypothesis-generation.

Suggested minimal interface:

```python
@dataclass(frozen=True)
class ScanContext:
    timestamp: Any
    detections: list[Detection]           # already stably ordered
    det_index_by_obj: dict[int, int]      # id(det) -> index in detections

class ScoringModel(Protocol):
    def score_track_hypotheses(
        self,
        *,
        track: Track,
        multihyp: Any,       # Stone Soup MultipleHypothesis
        ctx: ScanContext,
    ) -> dict[Any, float]:
        """Return {hypothesis_object: log_delta} for each hypothesis in multihyp."""

    def score_unused_detections(
        self,
        *,
        used_det_keys: set[int],
        ctx: ScanContext,
    ) -> float:
        """Return log_delta for clutter / unused detections for the global hypothesis."""

    def score_birth(
        self,
        *,
        birth_track: Track,
        used_det_key: int | None,
        ctx: ScanContext,
    ) -> float:
        """Return log_delta for adding a birth (prior / evidence)."""

```

### 1.4. Open design questions

- How much do we want to rely on the Stone Soup `Hypothesis` probabilities vs rolling our own likelihoods from innovations?
- Do we want to start with a **very simple** likelihood (e.g. Mahalanobis distance with a scaled constant) or go straight to a more accurate derivation?
- Should we integrate initiation evidence (holding track stats) into the global score explicitly, or keep it as a separate heuristic for birth ranking for now?
- Scoring work order:
  - First: make detection ordering stable (Step 0).
  - Then: implement ScoringModel + v1.5 scoring (β-ratio) as a bridge.
  - Only after scoring is consistent: start N-scan-lite and association history.
  
*(More sections will be added here for N-scan-lite, initiation rework, etc.)*
