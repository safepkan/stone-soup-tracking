from __future__ import annotations

from dataclasses import dataclass

from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.dataassociator.mfa import MFADataAssociator

from mht_experiments.scenarios.crossing_targets import ScenarioConfig


@dataclass
class MFAComponents:
    """Convenience bundle of MFA-related components."""
    predictor: KalmanPredictor
    updater: KalmanUpdater
    data_associator: MFADataAssociator


def build_mfa_components(
    transition_model: CombinedLinearGaussianTransitionModel,
    measurement_model: LinearGaussian,
    config: ScenarioConfig,
) -> MFAComponents:
    """Build the predictor, updater and MFA data associator."""

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    base_hypothesiser = PDAHypothesiser(
        predictor,
        updater,
        config.clutter_density,
        prob_gate=config.prob_gate,
        prob_detect=config.prob_detect,
    )

    mfa_hypothesiser = MFAHypothesiser(base_hypothesiser)
    data_associator = MFADataAssociator(
        mfa_hypothesiser,
        slide_window=config.slide_window,
    )

    return MFAComponents(
        predictor=predictor,
        updater=updater,
        data_associator=data_associator,
    )
