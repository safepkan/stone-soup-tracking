"""Internal scoring helpers for TO-MHT local additive terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log, log1p
from typing import Any, Mapping, Protocol, Sequence

from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.prediction import Prediction

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


def existence_metadata_to_log_odds(
    metadata: Mapping[str, object],
    *,
    default_log_odds: float,
    source_name: str,
) -> float:
    """Resolve optional start-confidence metadata to an initial log-odds score."""
    metadata_log_odds: Any = metadata.get("existence_log_odds")
    if metadata_log_odds is not None:
        try:
            log_odds = float(metadata_log_odds)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if isfinite(log_odds):
                return log_odds

    metadata_probability: Any = metadata.get("existence_probability")
    if metadata_probability is None:
        return default_log_odds
    try:
        return _existence_probability_to_log_odds(
            metadata_probability,
            parameter_name=f"{source_name} metadata['existence_probability']",
        )
    except ValueError:
        return default_log_odds


class ScoringModel(Protocol):
    """Narrow internal protocol consumed by expansion helpers."""

    def score_track_hypotheses(
        self,
        *,
        hypotheses: Sequence[SingleDistanceHypothesis],
        ctx: ScanContext,
        track_id: object | None = None,
    ) -> list[float]:
        """Return one local log-delta per hypothesis (same order as input)."""


class DetectionProbabilityModel(Protocol):
    """Caller-facing dynamic detection/clutter model used by NLL scoring.

    ``caller_scan_context`` is opaque caller data supplied to
    ``TOMHTTracker.update_tracker(...)``. It is distinct from TOMHT's internal
    ``ScanContext`` bookkeeping and may contain sensor identity, scan geometry,
    operating mode, weather/calibration data, or any other domain-specific
    information needed to evaluate detection probability and clutter density.
    """

    def detection_probability(
        self,
        *,
        track_id: object | None,
        prediction: Prediction,
        caller_scan_context: object | None,
    ) -> float:
        """Return ``P_D`` for one predicted target state in this scan."""

    def clutter_density(
        self,
        *,
        prediction: Prediction,
        detection: Detection | None,
        caller_scan_context: object | None,
    ) -> float:
        """Return clutter density in the same measurement-space units as NLL."""


@dataclass(frozen=True, init=False)
class ConstantDetectionProbabilityModel:
    """Scalar fallback DPM preserving the historical NLL scoring behavior."""

    prob_detect: float
    _clutter_density: float = field(repr=False)

    def __init__(self, prob_detect: float, clutter_density: float) -> None:
        object.__setattr__(self, "prob_detect", prob_detect)
        object.__setattr__(self, "_clutter_density", clutter_density)

    @property
    def constant_clutter_density(self) -> float:
        """Scalar clutter density backing the constant protocol method."""
        return self._clutter_density

    def detection_probability(
        self,
        *,
        track_id: object | None,
        prediction: Prediction,
        caller_scan_context: object | None,
    ) -> float:
        del track_id, prediction, caller_scan_context
        return float(self.prob_detect)

    def clutter_density(
        self,
        *,
        prediction: Prediction,
        detection: Detection | None,
        caller_scan_context: object | None,
    ) -> float:
        del prediction, detection, caller_scan_context
        return float(self._clutter_density)


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
    - Initiator/external-start root scores are existence-prior log-odds and are
      handled outside this local NLL scorer.
    - ``detection_probability_model`` may vary ``P_D`` and ``lambda`` per
      prediction, detection, or caller-provided scan context. A scalar
      ``ConstantDetectionProbabilityModel`` preserves the default behavior.
    - ``clutter_density`` units must match the measurement-space NLL; hit
      clutter density callbacks receive both the hypothesis prediction and
      concrete detection. If a DPM returns ``P_D`` near zero outside sensor
      coverage, miss scores become near zero and avoid unfair miss penalties.
    """

    detection_probability_model: DetectionProbabilityModel
    log_epsilon: float

    def _clamped_prob_detect(self, prob_detect: float) -> float:
        return min(1.0, max(0.0, float(prob_detect)))

    def _safe_log_clutter_density(self, clutter_density: float) -> float:
        return log(max(float(clutter_density), self.log_epsilon))

    def _log_hit_base(self, *, prob_detect: float, clutter_density: float) -> float:
        prob_detect = self._clamped_prob_detect(prob_detect)
        return log(max(prob_detect, self.log_epsilon)) - self._safe_log_clutter_density(
            clutter_density
        )

    def _log_miss(self, *, prob_detect: float) -> float:
        prob_detect = self._clamped_prob_detect(prob_detect)
        return log(max(1.0 - prob_detect, self.log_epsilon))

    def score_track_hypotheses(
        self,
        *,
        hypotheses: Sequence[SingleDistanceHypothesis],
        ctx: ScanContext,
        track_id: object | None = None,
    ) -> list[float]:
        out: list[float] = []
        for hypothesis in hypotheses:
            prediction = hypothesis.prediction
            measurement = hypothesis.measurement
            prob_detect = self.detection_probability_model.detection_probability(
                track_id=track_id,
                prediction=prediction,
                caller_scan_context=ctx.caller_scan_context,
            )
            if isinstance(measurement, MissedDetection):
                out.append(self._log_miss(prob_detect=prob_detect))
            else:
                clutter_density = self.detection_probability_model.clutter_density(
                    prediction=prediction,
                    detection=measurement,
                    caller_scan_context=ctx.caller_scan_context,
                )
                out.append(
                    self._log_hit_base(
                        prob_detect=prob_detect,
                        clutter_density=clutter_density,
                    )
                    - float(hypothesis.distance)
                )
        return out


def maybe_log_scoring_diagnostics(scoring_model: NLLScoringModel) -> None:
    """Emit optional diagnostics for the tracker-owned NLL scorer."""
    dpm = scoring_model.detection_probability_model
    if isinstance(dpm, ConstantDetectionProbabilityModel):
        log_hit_base = scoring_model._log_hit_base(
            prob_detect=dpm.prob_detect,
            clutter_density=dpm.constant_clutter_density,
        )
        log_miss = scoring_model._log_miss(prob_detect=dpm.prob_detect)
        print(
            f"[Scoring] nll: prob_detect={dpm.prob_detect}, "
            f"clutter_density={dpm.constant_clutter_density}, "
            f"log_hit_base={log_hit_base:+.3f}, "
            f"log_miss={log_miss:+.3f}"
        )
        return

    print(
        "[Scoring] nll: dynamic DetectionProbabilityModel "
        f"log_epsilon={scoring_model.log_epsilon}"
    )
