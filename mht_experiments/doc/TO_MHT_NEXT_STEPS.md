# TO-MHT Next Steps

This document is for planning upcoming work in more detail than the high-level roadmap. It will evolve as we implement each step.

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

### 1.4. Open design questions

- How much do we want to rely on the Stone Soup `Hypothesis` probabilities vs rolling our own likelihoods from innovations?
- Do we want to start with a **very simple** likelihood (e.g. Mahalanobis distance with a scaled constant) or go straight to a more accurate derivation?
- Should we integrate initiation evidence (holding track stats) into the global score explicitly, or keep it as a separate heuristic for birth ranking for now?

*(More sections will be added here for N-scan-lite, initiation rework, etc.)*
