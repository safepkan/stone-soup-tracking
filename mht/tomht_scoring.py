"""Scoring model contract and default beta-ratio implementation for TOMHT."""

from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING, Iterable, Mapping, Protocol

from stonesoup.types.detection import MissedDetection
from stonesoup.types.track import Track

if TYPE_CHECKING:
    from mht.tomht_tracker import ScanContext


class ScoringModel(Protocol):
    def score_track_hypotheses(
        self,
        *,
        track: Track,
        hypotheses: Iterable,
        ctx: "ScanContext",
    ) -> Mapping[int, float]:
        """Return {id(hypothesis_object): log_delta} for each hypothesis."""

    def score_unused_detections(
        self,
        *,
        used_det_keys: set[int],
        ctx: "ScanContext",
    ) -> float:
        """Return a log_delta to add for clutter / unused detections."""

    def score_birth(
        self,
        *,
        birth_track: Track,
        used_det_key: int | None,
        ctx: "ScanContext",
    ) -> float:
        """Return a log_delta to add for a birth (usually negative)."""


@dataclass(frozen=True)
class BetaRatioScoringModel:
    """Approximate MHT-like scoring using PDA β ratios."""

    prob_detect: float
    prob_gate: float
    clutter_density: float
    log_epsilon: float
    fallback_unused_det_log_penalty: float
    birth_log_penalty: float

    def _per_unused_log_delta(self) -> float:
        """Return the per-unused log increment used by clutter scoring."""
        lam = max(self.clutter_density, 0.0)
        if lam <= 0.0:
            return -self.fallback_unused_det_log_penalty
        return log(max(lam, self.log_epsilon))

    def _beta_values(
        self, hypotheses: Iterable
    ) -> tuple[float, Mapping[int, tuple[float, bool]]]:
        beta0 = None
        scores: dict[int, tuple[float, bool]] = {}
        for hyp in hypotheses:
            p = getattr(hyp, "probability", None)
            try:
                p_float = float(p) if p is not None else 0.0
            except Exception:
                p_float = 0.0

            is_miss = isinstance(
                getattr(hyp, "measurement", None), MissedDetection
            ) or (not hyp)
            if is_miss:
                beta0 = p_float if beta0 is None else beta0 + p_float
            scores[id(hyp)] = (p_float, is_miss)

        beta0_val = beta0 if beta0 is not None else self.log_epsilon
        beta0_val = max(beta0_val, self.log_epsilon)
        return beta0_val, scores

    def score_track_hypotheses(
        self, *, track: Track, hypotheses: Iterable, ctx: "ScanContext"
    ) -> Mapping[int, float]:
        beta0, prob_map = self._beta_values(hypotheses)
        pd_pg = min(1.0, max(0.0, self.prob_detect * self.prob_gate))
        common = log(max(1.0 - pd_pg, self.log_epsilon))

        out: dict[int, float] = {}
        for hyp_id, (beta, is_miss) in prob_map.items():
            beta_clamped = max(beta, self.log_epsilon)
            beta0_clamped = max(beta0, self.log_epsilon)
            if is_miss:
                out[hyp_id] = common
            else:
                out[hyp_id] = log(beta_clamped) - log(beta0_clamped) + common
        return out

    def score_unused_detections(
        self, *, used_det_keys: set[int], ctx: "ScanContext"
    ) -> float:
        unused = len(ctx.detections) - len(used_det_keys)
        if unused <= 0:
            return 0.0
        return float(unused) * self._per_unused_log_delta()

    def score_birth(
        self, *, birth_track: Track, used_det_key: int | None, ctx: "ScanContext"
    ) -> float:
        return -self.birth_log_penalty
