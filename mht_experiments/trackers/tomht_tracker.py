from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, TypeAlias

import numpy as np

from stonesoup.dataassociator.base import Associator  # only for typing readability
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
    """Parameters for a simple greedy TO-MHT."""
    max_children_per_track: int = 5
    max_missed: int = 5
    log_epsilon: float = 1e-12


class TOMHTTracker:
    """
    First-cut Track-Oriented MHT:
    - maintains ONE global hypothesis (greedy assignment per scan)
    - branches per-track using a PDAHypothesiser
    - chooses one hypothesis per track with measurement exclusivity
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

        # Keep as list for stable ordering; we also assign stable track IDs.
        self.tracks: list[Track] = list(tracks)
        for i, track in enumerate(self.tracks):
            track.metadata.setdefault("track_id", i)
            track.metadata.setdefault("log_weight", 0.0)
            track.metadata.setdefault("missed_count", 0)

    @staticmethod
    def _hypothesis_probability(hyp) -> float:
        # PDAHypothesiser yields SingleProbabilityHypothesis with `.probability`.
        p = getattr(hyp, "probability", None)
        if p is None:
            # Fallback: some hypotheses provide `.weight` (e.g. distance-based)
            w = getattr(hyp, "weight", 0.0)
            try:
                return float(w)
            except Exception:
                return 0.0
        try:
            return float(p)
        except Exception:
            return 0.0

    def _hypothesis_log_score(self, hyp) -> float:
        p = self._hypothesis_probability(hyp)
        return log(max(p, self.params.log_epsilon))

    @staticmethod
    def _detection_key(det: Detection) -> int:
        # Use object identity for this scan; safe and fast.
        return id(det)

    def step(self, detections: Iterable[Detection], timestamp) -> set[Track]:
        """
        Advance one scan. Returns the current set of tracks (MAP global hypothesis).
        """
        det_list = list(detections)
        used_dets: set[int] = set()

        # Track ordering: highest-weight tracks first tends to be less “greedy-bad”.
        tracks_ordered = sorted(
            self.tracks,
            key=lambda tr: float(tr.metadata.get("log_weight", 0.0)),
            reverse=True,
        )

        for track in tracks_ordered:
            multi_hyp = self.hypothesiser.hypothesise(track, det_list, timestamp)

            # Sort single hypotheses best-first and cap branching factor.
            # Ensure "miss" hypothesis stays available even if it sorts low.
            single = list(multi_hyp)
            single_sorted = sorted(single, key=self._hypothesis_probability, reverse=True)

            # Keep top-N plus always keep a missed-detection hypothesis if present.
            kept = single_sorted[: self.params.max_children_per_track]
            miss_hyps = [h for h in single_sorted if not h]  # Stone Soup convention
            if miss_hyps and miss_hyps[0] not in kept:
                kept.append(miss_hyps[0])

            chosen = None
            chosen_det_key: int | None = None

            for hyp in kept:
                if not hyp:
                    # Missed detection is always admissible
                    chosen = hyp
                    chosen_det_key = None
                    break

                det = hyp.measurement
                dk = self._detection_key(det)
                if dk not in used_dets:
                    chosen = hyp
                    chosen_det_key = dk
                    break

            # As a final fallback, take a miss if we somehow didn't select anything
            if chosen is None:
                chosen = miss_hyps[0] if miss_hyps else kept[-1]

            # Apply chosen hypothesis: append prediction or update state
            if not chosen:
                # Missed detection -> append prediction
                track.append(chosen.prediction)
                track.metadata["missed_count"] = int(track.metadata.get("missed_count", 0)) + 1
            else:
                update = self.updater.update(chosen)
                track.append(update)
                track.metadata["missed_count"] = 0
                if chosen_det_key is not None:
                    used_dets.add(chosen_det_key)

            # Update track score
            track.metadata["log_weight"] = float(track.metadata.get("log_weight", 0.0)) + self._hypothesis_log_score(chosen)

        # Optionally delete “dead” tracks
        self.tracks = [
            tr for tr in self.tracks
            if int(tr.metadata.get("missed_count", 0)) <= self.params.max_missed
        ]

        return set(self.tracks)


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
