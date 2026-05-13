"""Tracker scoring contract for TO-MHT local/global additive terms."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, log1p
from typing import Protocol, Sequence

from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis

from .tomht_types import ScanContext


def _existence_probability_to_log_odds(
    probability: float,
    *,
    parameter_name: str = "existence_probability",
) -> float:
    """Map a public existence probability to an internal log-odds score.

    External configuration uses an intuitive probability. TOMHT root scores are
    additive log-deltas, so ``0.5`` maps to the old neutral ``0.0`` score.
    """
    invalid_message = (
        f"{parameter_name} must satisfy 0.0 < p < 1.0; got {probability!r}."
    )
    try:
        p = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError(invalid_message) from exc
    if not isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError(invalid_message)
    return log(p) - log1p(-p)


class ScoringModel(Protocol):
    def score_track_hypotheses(
        self,
        *,
        hypotheses: Sequence[SingleDistanceHypothesis],
        ctx: ScanContext,
    ) -> list[float]:
        """Return one local log-delta per hypothesis (same order as input)."""


@dataclass(frozen=True)
class NLLScoringModel:
    """NLL-based local scoring with explicit miss/hit LLR terms.

    Local score contributions:
    - hit: ``log(P_D) - log(lambda) - NLL``
    - miss: ``log(1 - P_D)``

    Unit/scale contract:
    - ``NLL`` must be computed from the same measurement coordinates used by the
      association hypothesiser (i.e. from ``p(z|x)`` in that measurement space),
    - ``clutter_density`` (``lambda``) must be in the same measurement-space
      units (detections per measurement-volume per scan).
    With that contract, linear coordinate rescaling cancels between
    ``-log(lambda)`` and the Gaussian normalisation term inside ``NLL``.

    Miss-hypothesis ``distance`` from the hypothesiser is intentionally ignored.

    Note:
    - Unused-detection scoring has been removed from the default scoring
      contract. The clutter-density contrast is already carried by the local
      hit term through ``-log(lambda)``.
    - Whether this API remains the right abstraction is deferred to a later
      scoring redesign pass.
    """

    prob_detect: float
    clutter_density: float
    log_epsilon: float

    def _clamped_prob_detect(self) -> float:
        return min(1.0, max(0.0, float(self.prob_detect)))

    def _safe_log_clutter_density(self) -> float:
        return log(max(float(self.clutter_density), self.log_epsilon))

    def _log_hit_base(self) -> float:
        prob_detect = self._clamped_prob_detect()
        return (
            log(max(prob_detect, self.log_epsilon)) - self._safe_log_clutter_density()
        )

    def _log_miss(self) -> float:
        prob_detect = self._clamped_prob_detect()
        return log(max(1.0 - prob_detect, self.log_epsilon))

    def score_track_hypotheses(
        self,
        *,
        hypotheses: Sequence[SingleDistanceHypothesis],
        ctx: ScanContext,
    ) -> list[float]:
        del ctx
        log_hit_base = self._log_hit_base()
        log_miss = self._log_miss()

        out: list[float] = []
        for hypothesis in hypotheses:
            if isinstance(hypothesis.measurement, MissedDetection):
                out.append(log_miss)
            else:
                out.append(log_hit_base - float(hypothesis.distance))
        return out


def make_default_scoring_model(
    *,
    scoring_mode: str,
    prob_detect: float,
    log_epsilon: float,
    clutter_density: float,
) -> ScoringModel:
    """Build the tracker's default scoring model from tracker-owned params."""
    mode = str(scoring_mode).strip().lower()
    if mode == "nll":
        return NLLScoringModel(
            prob_detect=float(prob_detect),
            clutter_density=float(clutter_density),
            log_epsilon=log_epsilon,
        )
    raise ValueError(
        f"Unsupported scoring_mode='{scoring_mode}'. " "Supported values: 'nll'."
    )


def maybe_log_scoring_diagnostics(scoring_model: ScoringModel) -> None:
    """Emit optional scoring diagnostics for known scoring-model implementations."""
    if isinstance(scoring_model, NLLScoringModel):
        clutter = scoring_model.clutter_density
        print(
            f"[Scoring] nll: prob_detect={scoring_model.prob_detect}, "
            f"clutter_density={clutter}, "
            f"log_hit_base={scoring_model._log_hit_base():+.3f}, "
            f"log_miss={scoring_model._log_miss():+.3f}"
        )
