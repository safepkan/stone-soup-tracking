from __future__ import annotations

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.base import Initiator
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.models.transition.base import TransitionModel
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater

from mht.helpers.hypothesiser import RobustPDAHypothesiser
from mht.tomht_tracker import TOMHTParams, TOMHTTracker


def build_tomht_linear(
    transition_model: TransitionModel,
    measurement_model: MeasurementModel,
    *,
    prob_detect: float,
    clutter_density: float,
    initiator: Initiator | None = None,
    params: TOMHTParams = TOMHTParams(),
) -> TOMHTTracker:
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = PDAHypothesiser(
        predictor,
        updater,
        clutter_density,
        prob_gate=params.prob_gate,
        prob_detect=prob_detect,
    )
    return TOMHTTracker(hypothesiser, updater, initiator=initiator, params=params)


def build_tomht_ukf(
    transition_model: TransitionModel,
    measurement_model: MeasurementModel,
    *,
    prob_detect: float,
    clutter_density: float,
    initiator: Initiator | None = None,
    params: TOMHTParams = TOMHTParams(),
) -> TOMHTTracker:
    predictor = UnscentedKalmanPredictor(transition_model)
    updater = UnscentedKalmanUpdater(measurement_model)
    hypothesiser = RobustPDAHypothesiser(
        predictor,
        updater,
        clutter_density,
        prob_gate=params.prob_gate,
        prob_detect=prob_detect,
    )
    return TOMHTTracker(hypothesiser, updater, initiator=initiator, params=params)
