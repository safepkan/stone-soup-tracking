from __future__ import annotations

from dataclasses import dataclass

from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from stonesoup.dataassociator.mfa import MFADataAssociator
from stonesoup.predictor.base import Predictor
from stonesoup.updater.base import Updater

from mht_experiments.scenarios.crossing_targets import ScenarioConfig


@dataclass
class MFAComponents:
    predictor: Predictor
    updater: Updater
    data_associator: MFADataAssociator


def _build_mfa_data_associator(
    predictor: Predictor, updater: Updater, config: ScenarioConfig
) -> MFADataAssociator:
    base = PDAHypothesiser(
        predictor,
        updater,
        config.clutter_density,
        prob_gate=config.prob_gate,
        prob_detect=config.prob_detect,
    )
    return MFADataAssociator(
        MFAHypothesiser(base),
        slide_window=config.slide_window,
    )


def build_mfa_components_linear(
    transition_model, measurement_model, config: ScenarioConfig
) -> MFAComponents:
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    return MFAComponents(
        predictor=predictor,
        updater=updater,
        data_associator=_build_mfa_data_associator(predictor, updater, config),
    )


def build_mfa_components_ukf(
    transition_model, measurement_model, config: ScenarioConfig
) -> MFAComponents:
    predictor = UnscentedKalmanPredictor(transition_model)
    updater = UnscentedKalmanUpdater(measurement_model)
    return MFAComponents(
        predictor=predictor,
        updater=updater,
        data_associator=_build_mfa_data_associator(predictor, updater, config),
    )
