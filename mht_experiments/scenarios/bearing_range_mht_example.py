from __future__ import annotations

import datetime
from typing import List, Tuple

import numpy as np
from ordered_set import OrderedSet

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)

from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import (
    SimpleMeasurementInitiator,
    NoHistoryMultiMeasurementInitiator,
)

from stonesoup.simulator.simple import (
    MultiTargetGroundTruthSimulator,
    SimpleDetectionSimulator,
)

from stonesoup.measures import Mahalanobis
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.state import GaussianState
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track


from mht_experiments.scenarios.common import ScenarioConfig


def _det_sort_key(det: Detection) -> tuple[float, float]:
    # bearing, range
    z = det.state_vector
    return (float(z[0, 0]), float(z[1, 0]))


def create_bearing_range_mht_example() -> Tuple[
    OrderedSet[GroundTruthPath],
    List[OrderedSet[Detection]],
    List[datetime.datetime],
    CombinedLinearGaussianTransitionModel,
    CartesianToBearingRange,
    ScenarioConfig,
]:
    """
    Refactor of Stone Soup v1.4 "General Multi Hypotheses tracking implementation example".
    Uses a CartesianToBearingRange measurement model and is intended for UKF.
    """
    np.random.seed(1908)  # as per MFT example

    start_time = datetime.datetime.now().replace(microsecond=0)
    simulation_steps = 50
    timestep_size = datetime.timedelta(seconds=2)

    prob_detection = 0.99
    clutter_area = np.array([[-1, 1], [-1, 1]]) * 150
    clutter_rate = 9
    surveillance_area = (clutter_area[0, 1] - clutter_area[0, 0]) * (
        clutter_area[1, 1] - clutter_area[1, 0]
    )
    clutter_spatial_density = clutter_rate / surveillance_area

    config = ScenarioConfig(
        prob_detect=prob_detection,
        prob_gate=0.9999,  # as per MFT example
        clutter_density=clutter_spatial_density,
        v_bounds=clutter_area,
        slide_window=3,  # as per MFT example
    )

    initial_state = GaussianState(
        state_vector=StateVector([[10], [0], [10], [0]]),
        covar=CovarianceMatrix(np.diag([30, 1, 40, 1])),
        timestamp=start_time,
    )

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )

    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.diag([np.radians(1), 5]),
    )

    ground_truth_simulator = MultiTargetGroundTruthSimulator(
        transition_model=transition_model,
        initial_state=initial_state,
        timestep=timestep_size,
        number_steps=simulation_steps,
        birth_rate=0,
        death_probability=0,
        preexisting_states=[
            [10, 1, 10, 1],
            [-10, -1, -10, -1],
            [-10, -1, 10, 1],
        ],
    )

    detection_sim = SimpleDetectionSimulator(
        groundtruth=ground_truth_simulator,
        measurement_model=measurement_model,
        detection_probability=prob_detection,
        meas_range=clutter_area,
        clutter_rate=clutter_rate,
    )

    truths: OrderedSet[GroundTruthPath] = OrderedSet()
    scans: List[OrderedSet[Detection]] = []
    timestamps: List[datetime.datetime] = []

    for time, dets in detection_sim:
        truths |= OrderedSet(ground_truth_simulator.groundtruth_paths)
        dets_sorted = sorted(dets, key=_det_sort_key)
        scans.append(OrderedSet(dets_sorted))
        timestamps.append(time)

    return truths, scans, timestamps, transition_model, measurement_model, config


def initial_mfa_tracks_for_bearing_range(
    start_time: datetime.datetime,
) -> OrderedSet[Track]:
    """Create the 3 priors used in the Stone Soup MFT example."""
    cov = np.diag([10, 1, 10, 1])

    priors = [
        StateVector([10, 1, 10, 1]),
        StateVector([-10, -1, -10, -1]),
        StateVector([-10, -1, 10, 1]),
    ]

    tracks: OrderedSet[Track] = OrderedSet()
    for sv in priors:
        gm = GaussianMixture(
            [
                TaggedWeightedGaussianState(
                    sv,
                    cov,
                    timestamp=start_time,
                    weight=Probability(1),
                    tag=[],
                )
            ]
        )
        tracks.add(Track([gm]))

    return tracks


def initial_tomht_tracks_for_bearing_range(start_time) -> list[Track]:
    """Initial tracks for TO-MHT (single Gaussian, not a mixture)."""
    cov = CovarianceMatrix(np.diag([10.0, 1.0, 10.0, 1.0]))
    priors = [
        StateVector([[10.0], [1.0], [10.0], [1.0]]),
        StateVector([[-10.0], [-1.0], [-10.0], [-1.0]]),
        StateVector([[-10.0], [-1.0], [10.0], [1.0]]),
    ]
    return [Track([GaussianState(p, covar=cov, timestamp=start_time)]) for p in priors]


def tomht_initiator_for_bearing_range_simple(
    start_time, measurement_model
) -> SimpleMeasurementInitiator:
    """
    Simple initiator based on SimpleMeasurementInitiator.
    """
    prior = GaussianState(
        state_vector=StateVector([[0.0], [0.0], [0.0], [0.0]]),
        covar=CovarianceMatrix(np.diag([200.0, 20.0, 200.0, 20.0])),
        timestamp=start_time,
    )
    return SimpleMeasurementInitiator(
        prior_state=prior, measurement_model=measurement_model
    )


def tomht_initiator_for_bearing_range(start_time, transition_model, measurement_model):
    """
    Confirmation-style initiator to avoid single-detection clutter tracks.

    Uses NoHistoryMultiMeasurementInitiator, which is like MultiMeasurementInitiator
    but releases confirmed tracks with only one state (holding history moved to metadata).
    """
    predictor = UnscentedKalmanPredictor(transition_model)
    updater = UnscentedKalmanUpdater(measurement_model)

    # Association inside the initiator’s “holding” logic (small internal tracker)
    hypothesiser = DistanceHypothesiser(
        predictor,
        updater,
        measure=Mahalanobis(),
        missed_distance=2.0,  # tune
    )
    data_associator = GNNWith2DAssignment(hypothesiser)

    # Delete uncertain holding tracks (tune threshold)
    deleter = CovarianceBasedDeleter(covar_trace_thresh=1e4)

    # Prior for unmapped dims (vel); mapped dims replaced via reversible model inverse
    prior = GaussianState(
        state_vector=StateVector([[0.0], [0.0], [0.0], [0.0]]),
        covar=CovarianceMatrix(np.diag([1.0, 50.0, 1.0, 50.0])),
        timestamp=start_time,
    )

    return NoHistoryMultiMeasurementInitiator(
        prior_state=prior,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=3,
        updates_only=True,
    )
