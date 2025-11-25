from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
from ordered_set import OrderedSet

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator, SimpleDetectionSimulator
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.state import GaussianState
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track

from mht_experiments.scenarios.crossing_targets import ScenarioConfig


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
    np.random.seed(1908)  # as per example :contentReference[oaicite:1]{index=1}

    start_time = datetime.datetime.now().replace(microsecond=0)
    simulation_steps = 50
    timestep_size = datetime.timedelta(seconds=2)

    prob_detection = 0.99
    clutter_area = np.array([[-1, 1], [-1, 1]]) * 150
    clutter_rate = 9
    surveillance_area = ((clutter_area[0, 1] - clutter_area[0, 0]) *
                         (clutter_area[1, 1] - clutter_area[1, 0]))
    clutter_spatial_density = clutter_rate / surveillance_area

    config = ScenarioConfig(
        prob_detect=prob_detection,
        prob_gate=0.9999,               # as per example :contentReference[oaicite:2]{index=2}
        clutter_density=clutter_spatial_density,
        v_bounds=clutter_area,
        slide_window=3,                 # as per example :contentReference[oaicite:3]{index=3}
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
        scans.append(OrderedSet(dets))
        timestamps.append(time)

    return truths, scans, timestamps, transition_model, measurement_model, config


def initial_mfa_tracks_for_bearing_range(start_time: datetime.datetime) -> Set[Track]:
    """Create the 3 priors used in the Stone Soup example. :contentReference[oaicite:4]{index=4}"""
    cov = np.diag([10, 1, 10, 1])

    priors = [
        StateVector([10, 1, 10, 1]),
        StateVector([-10, -1, -10, -1]),
        StateVector([-10, -1, 10, 1]),
    ]

    tracks: Set[Track] = set()
    for sv in priors:
        gm = GaussianMixture([
            TaggedWeightedGaussianState(
                sv,
                cov,
                timestamp=start_time,
                weight=Probability(1),
                tag=[],
            )
        ])
        tracks.add(Track([gm]))

    return tracks
