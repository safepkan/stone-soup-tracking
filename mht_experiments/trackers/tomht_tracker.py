from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, TypeAlias

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.models.transition.base import TransitionModel
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater


PredictorT: TypeAlias = KalmanPredictor | UnscentedKalmanPredictor
UpdaterT: TypeAlias = KalmanUpdater | UnscentedKalmanUpdater


@dataclass(frozen=True)
class TOMHTParams:
    max_global_hypotheses: int = 20
    max_children_per_track: int = 5
    max_missed: int = 5
    log_epsilon: float = 1e-12


@dataclass(frozen=True)
class ChildCandidate:
    track_id: int
    child_track: Track
    used_det_key: int | None
    log_delta: float


@dataclass(frozen=True)
class GlobalHypothesis:
    """One global hypothesis = one leaf per track_id + cumulative log weight."""

    tracks_by_id: dict[int, Track]
    log_weight: float


class TOMHTTracker:
    """
    Track-Oriented MHT with K-best global hypotheses (beam search).

    - Maintains a list of GlobalHypothesis objects of size <= K.
    - Each scan: branch each track (per global hyp), then form consistent globals
      (one child per track_id, no shared detections).
    """

    def __init__(
        self,
        hypothesiser: PDAHypothesiser,
        updater: UpdaterT,
        tracks: Iterable[Track],
        *,
        params: TOMHTParams = TOMHTParams(),
    ) -> None:
        self.hypothesiser = hypothesiser
        self.updater = updater
        self.params = params

        init_tracks_by_id: dict[int, Track] = {}
        for i, tr in enumerate(list(tracks)):
            tr.metadata.setdefault("track_id", i)
            tr.metadata.setdefault("log_weight", 0.0)
            tr.metadata.setdefault("missed_count", 0)
            init_tracks_by_id[int(tr.metadata["track_id"])] = tr

        self.global_hypotheses: list[GlobalHypothesis] = [
            GlobalHypothesis(tracks_by_id=init_tracks_by_id, log_weight=0.0)
        ]

    @staticmethod
    def _det_key(det: Detection) -> int:
        # Per-scan identity; OK for exclusivity constraint
        return id(det)

    def _hyp_probability(self, hyp) -> float:
        p = getattr(hyp, "probability", None)
        if p is None:
            w = getattr(hyp, "weight", 0.0)
            try:
                return float(w)
            except Exception:
                return 0.0
        try:
            return float(p)
        except Exception:
            return 0.0

    def _hyp_log_delta(self, hyp) -> float:
        return log(max(self._hyp_probability(hyp), self.params.log_epsilon))

    @staticmethod
    def _copy_track(track: Track) -> Track:
        # Track is list-like; copying states is enough for our simple bookkeeping
        return Track(list(track.states))

    def _candidates_for_track(
        self,
        track_id: int,
        track: Track,
        detections: list[Detection],
        timestamp,
    ) -> list[ChildCandidate]:
        multi = self.hypothesiser.hypothesise(track, detections, timestamp)
        singles = list(multi)

        # Sort best-first and cap. Always keep a "miss" if present.
        singles_sorted = sorted(singles, key=self._hyp_probability, reverse=True)
        kept = singles_sorted[: self.params.max_children_per_track]
        miss = next((h for h in singles_sorted if not h), None)
        if miss is not None and miss not in kept:
            kept.append(miss)

        candidates: list[ChildCandidate] = []
        for hyp in kept:
            child = self._copy_track(track)

            if not hyp:
                # missed detection -> append prediction
                child.append(hyp.prediction)
                child.metadata.update(track.metadata)
                child.metadata["missed_count"] = (
                    int(track.metadata.get("missed_count", 0)) + 1
                )
                used = None
            else:
                upd = self.updater.update(hyp)
                child.append(upd)
                child.metadata.update(track.metadata)
                child.metadata["missed_count"] = 0
                used = self._det_key(hyp.measurement)

            candidates.append(
                ChildCandidate(
                    track_id=track_id,
                    child_track=child,
                    used_det_key=used,
                    log_delta=self._hyp_log_delta(hyp),
                )
            )

        # Best candidates first helps pruning in backtracking
        candidates.sort(key=lambda c: c.log_delta, reverse=True)
        return candidates

    def _expand_global_hypothesis(
        self,
        gh: GlobalHypothesis,
        detections: list[Detection],
        timestamp,
    ) -> list[GlobalHypothesis]:
        track_ids = sorted(gh.tracks_by_id.keys())
        per_track_candidates: list[list[ChildCandidate]] = []
        for tid in track_ids:
            per_track_candidates.append(
                self._candidates_for_track(
                    tid, gh.tracks_by_id[tid], detections, timestamp
                )
            )

        new_globals: list[GlobalHypothesis] = []

        def backtrack(
            i: int, used: set[int], acc_tracks: dict[int, Track], acc_log: float
        ) -> None:
            if i == len(track_ids):
                new_globals.append(
                    GlobalHypothesis(tracks_by_id=dict(acc_tracks), log_weight=acc_log)
                )
                return

            tid = track_ids[i]
            for cand in per_track_candidates[i]:
                if cand.used_det_key is not None and cand.used_det_key in used:
                    continue

                # Track deletion (simple): if too many misses, drop this track from the global
                child = cand.child_track
                missed = int(child.metadata.get("missed_count", 0))
                if missed > self.params.max_missed:
                    # do not include this track in the global hypothesis
                    acc_tracks.pop(tid, None)
                    backtrack(i + 1, used, acc_tracks, acc_log + cand.log_delta)
                    acc_tracks[tid] = child  # restore for other branches
                    continue

                if cand.used_det_key is not None:
                    used.add(cand.used_det_key)
                acc_tracks[tid] = child
                backtrack(i + 1, used, acc_tracks, acc_log + cand.log_delta)
                if cand.used_det_key is not None:
                    used.remove(cand.used_det_key)

        backtrack(
            i=0,
            used=set(),
            acc_tracks=dict(),
            acc_log=gh.log_weight,
        )
        return new_globals

    def step(self, detections: Iterable[Detection], timestamp) -> set[Track]:
        det_list = list(detections)

        expanded: list[GlobalHypothesis] = []
        for gh in self.global_hypotheses:
            expanded.extend(self._expand_global_hypothesis(gh, det_list, timestamp))

        # Keep top-K globals (beam)
        expanded.sort(key=lambda g: g.log_weight, reverse=True)
        self.global_hypotheses = expanded[: self.params.max_global_hypotheses]

        # Output MAP global hypothesis
        if not self.global_hypotheses:
            return set()

        best = self.global_hypotheses[0]
        return set(best.tracks_by_id.values())


def build_tomht_linear(
    transition_model: TransitionModel,
    measurement_model: MeasurementModel,
    *,
    prob_detect: float,
    prob_gate: float,
    clutter_density: float,
    tracks: Iterable[Track],
    params: TOMHTParams = TOMHTParams(),
) -> TOMHTTracker:
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = PDAHypothesiser(
        predictor,
        updater,
        clutter_density,
        prob_gate=prob_gate,
        prob_detect=prob_detect,
    )
    return TOMHTTracker(hypothesiser, updater, tracks, params=params)


def build_tomht_ukf(
    transition_model: TransitionModel,
    measurement_model: MeasurementModel,
    *,
    prob_detect: float,
    prob_gate: float,
    clutter_density: float,
    tracks: Iterable[Track],
    params: TOMHTParams = TOMHTParams(),
) -> TOMHTTracker:
    predictor = UnscentedKalmanPredictor(transition_model)
    updater = UnscentedKalmanUpdater(measurement_model)
    hypothesiser = PDAHypothesiser(
        predictor,
        updater,
        clutter_density,
        prob_gate=prob_gate,
        prob_detect=prob_detect,
    )
    return TOMHTTracker(hypothesiser, updater, tracks, params=params)
