"""Tracker scoring contract for TO-MHT local/global additive terms."""

from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING, Protocol, Sequence

from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.track import Track

if TYPE_CHECKING:
    from mht.tomht_tracker import ScanContext

LocalDetectionIndex = int


class ScoringModel(Protocol):
    def score_track_hypotheses(
        self,
        *,
        hypotheses: Sequence[SingleDistanceHypothesis],
        ctx: "ScanContext",
    ) -> list[float]:
        """Return one local log-delta per hypothesis (same order as input)."""

    def score_unused_detections(
        self,
        *,
        used_det_keys: set[LocalDetectionIndex],
        ctx: "ScanContext",
    ) -> float:
        """Return a cluster-level log-delta for clutter / unused detections.

        ``used_det_keys`` are local indices into ``ctx.detections`` for the
        current scan context (which may be a cluster-local subset).

        Current tracker solver-preparation logic assumes this score is affine in
        ``len(used_det_keys)``. Non-linear implementations are not supported in
        the current baseline and may raise at runtime.
        """

    def score_birth(
        self,
        *,
        birth_track: Track,
        used_det_key: LocalDetectionIndex | None,
        ctx: "ScanContext",
    ) -> float:
        """Return a log-delta to add for a birth (usually negative).

        ``used_det_key`` uses the same local-index convention as
        ``score_unused_detections``.
        """


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
    - ``score_unused_detections`` remains intentionally simple/linear in this
      baseline to match current solver pre-baking.
    - Whether this API remains the right abstraction is deferred to a later
      scoring redesign pass.
    """

    prob_detect: float
    clutter_density: float
    log_epsilon: float
    fallback_unused_det_log_penalty: float
    birth_log_penalty: float

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

    def _per_unused_log_delta(self) -> float:
        """Return the per-unused log increment used by clutter scoring."""
        lam = max(float(self.clutter_density), 0.0)
        if lam <= 0.0:
            return -self.fallback_unused_det_log_penalty
        return log(max(lam, self.log_epsilon))

    def score_track_hypotheses(
        self,
        *,
        hypotheses: Sequence[SingleDistanceHypothesis],
        ctx: "ScanContext",
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

    def score_unused_detections(
        self, *, used_det_keys: set[LocalDetectionIndex], ctx: "ScanContext"
    ) -> float:
        unused = len(ctx.detections) - len(used_det_keys)
        if unused <= 0:
            return 0.0
        return float(unused) * self._per_unused_log_delta()

    def score_birth(
        self,
        *,
        birth_track: Track,
        used_det_key: LocalDetectionIndex | None,
        ctx: "ScanContext",
    ) -> float:
        del birth_track, used_det_key, ctx
        return -self.birth_log_penalty


def make_default_scoring_model(
    *,
    scoring_mode: str,
    prob_detect: float,
    log_epsilon: float,
    clutter_density: float,
    unused_det_log_penalty: float,
    birth_log_penalty: float,
) -> ScoringModel:
    """Build the tracker's default scoring model from tracker-owned params."""
    mode = str(scoring_mode).strip().lower()
    if mode == "nll":
        return NLLScoringModel(
            prob_detect=float(prob_detect),
            clutter_density=float(clutter_density),
            log_epsilon=log_epsilon,
            fallback_unused_det_log_penalty=unused_det_log_penalty,
            birth_log_penalty=birth_log_penalty,
        )
    raise ValueError(
        f"Unsupported scoring_mode='{scoring_mode}'. " "Supported values: 'nll'."
    )


def maybe_log_scoring_diagnostics(scoring_model: ScoringModel) -> None:
    """Emit optional scoring diagnostics for known scoring-model implementations."""
    if isinstance(scoring_model, NLLScoringModel):
        clutter = scoring_model.clutter_density
        per_unused = scoring_model._per_unused_log_delta()
        if per_unused > 0.0:
            print(
                "[WARN] per_unused_delta is positive; unused detections are rewarded. "
                "Check clutter_density units/config."
            )
        print(
            f"[Scoring] nll: prob_detect={scoring_model.prob_detect}, "
            f"clutter_density={clutter}, "
            f"log_hit_base={scoring_model._log_hit_base():+.3f}, "
            f"log_miss={scoring_model._log_miss():+.3f}, "
            f"per_unused_delta={per_unused:+.3f}"
        )
