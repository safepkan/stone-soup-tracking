from __future__ import annotations

from typing import List, Tuple

import datetime
import numpy as np
from ordered_set import OrderedSet
from scipy.stats import chi2

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.initiator.simple import SimpleMeasurementInitiator

from stonesoup.types.detection import Clutter, Detection, TrueDetection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track


from mht_experiments.scenarios.common import ScenarioConfig


# ----- Global-ish parameters for this scenario -----

# Reproduce the original example behaviour
np.random.seed(2001)

_PROB_DETECT = 0.9
_GATE_LEVEL = 8.0  # chi-square gate level (2D)
_V_BOUNDS = np.array([[-5.0, 30.0], [-5.0, 30.0]])  # surveillance area bounds
_LAMBDA_V = 5.0  # mean number of clutter points over full area

# Derived quantities
_PROB_GATE = chi2.cdf(_GATE_LEVEL, 2)  # gating probability
# Clutter spatial density (lambda_) – note v_bounds[:, 0] - v_bounds[:, 1] is negative, but
# the product is still positive, so this matches the example's implementation.
_CLUTTER_DENSITY = _LAMBDA_V / np.prod(_V_BOUNDS[:, 0] - _V_BOUNDS[:, 1])

_SLIDE_WINDOW = 3  # MFA slide window length (in scans)


_CONFIG = ScenarioConfig(
    prob_detect=_PROB_DETECT,
    prob_gate=_PROB_GATE,
    clutter_density=_CLUTTER_DENSITY,
    v_bounds=_V_BOUNDS,
    slide_window=_SLIDE_WINDOW,
)


def create_crossing_scenario() -> Tuple[
    OrderedSet[GroundTruthPath],
    List[OrderedSet[Detection]],
    datetime.datetime,
    CombinedLinearGaussianTransitionModel,
    LinearGaussian,
    ScenarioConfig,
]:
    """Create the 'two crossing targets + clutter' MFA scenario.

    Returns
    -------
    truths:
        Ordered set of ground truth paths.
    scans:
        List of detection sets (one per time step).
    start_time:
        Start timestamp.
    transition_model:
        Near-constant-velocity 2D motion model.
    measurement_model:
        Position-only linear Gaussian measurement model.
    config:
        ScenarioConfig with detection/clutter parameters.
    """
    # Models: 4D state [x, vx, y, vy], position-only measurements [x, y]
    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.005), ConstantVelocity(0.005)]
    )
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.7**2, 0.0], [0.0, 0.7**2]]),
    )

    # Ground truth: two crossing targets
    truths: OrderedSet[GroundTruthPath] = OrderedSet()
    start_time = datetime.datetime.now()

    # Target 1: starts at (0, 0), heading roughly up-right
    truth1 = GroundTruthPath(
        [GroundTruthState([0.0, 1.0, 0.0, 1.0], timestamp=start_time)]
    )
    for k in range(1, 21):
        prev_state = truth1[k - 1]
        new_state_vec = transition_model.function(
            prev_state, noise=True, time_interval=datetime.timedelta(seconds=1)
        )
        truth1.append(
            GroundTruthState(
                new_state_vec,
                timestamp=start_time + datetime.timedelta(seconds=k),
            )
        )
    truths.add(truth1)

    # Target 2: starts at y=20, heading down-right
    truth2 = GroundTruthPath(
        [GroundTruthState([0.0, 1.0, 20.0, -1.0], timestamp=start_time)]
    )
    for k in range(1, 21):
        prev_state = truth2[k - 1]
        new_state_vec = transition_model.function(
            prev_state, noise=True, time_interval=datetime.timedelta(seconds=1)
        )
        truth2.append(
            GroundTruthState(
                new_state_vec,
                timestamp=start_time + datetime.timedelta(seconds=k),
            )
        )
    truths.add(truth2)

    # Measurements + clutter
    scans: List[OrderedSet[Detection]] = []
    num_steps = 20

    for k in range(num_steps):
        detections: OrderedSet[Detection] = OrderedSet()
        for truth in truths:
            gt_state = truth[k]

            # True detection with probability P_D
            if np.random.rand() <= _PROB_DETECT:
                measurement_vec = measurement_model.function(gt_state, noise=True)
                detections.add(
                    TrueDetection(
                        state_vector=measurement_vec,
                        groundtruth_path=truth,
                        timestamp=gt_state.timestamp,
                    )
                )

            # Clutter
            for _ in range(np.random.poisson(_LAMBDA_V)):
                x = np.random.uniform(*_V_BOUNDS[0])
                y = np.random.uniform(*_V_BOUNDS[1])
                detections.add(
                    Clutter(
                        [[x], [y]],
                        timestamp=gt_state.timestamp,
                    )
                )

        scans.append(detections)

    return truths, scans, start_time, transition_model, measurement_model, _CONFIG


def initial_mfa_tracks_for_crossing(start_time: datetime.datetime) -> OrderedSet[Track]:
    prior1 = GaussianMixture(
        [
            TaggedWeightedGaussianState(
                [[0.0], [1.0], [0.0], [1.0]],
                np.diag([1.5, 0.5, 1.5, 0.5]),
                timestamp=start_time,
                weight=Probability(1.0),
                tag=[],
            )
        ]
    )

    prior2 = GaussianMixture(
        [
            TaggedWeightedGaussianState(
                [[0.0], [1.0], [20.0], [-1.0]],
                np.diag([1.5, 0.5, 1.5, 0.5]),
                timestamp=start_time,
                weight=Probability(1.0),
                tag=[],
            )
        ]
    )

    return OrderedSet((Track([prior1]), Track([prior2])))


def initial_tomht_tracks_for_crossing(start_time) -> list[Track]:
    """Initial tracks for TO-MHT (single Gaussian, not a mixture)."""
    cov = CovarianceMatrix(np.diag([1.5, 0.5, 1.5, 0.5]))
    s1 = GaussianState(
        StateVector([[0.0], [1.0], [0.0], [1.0]]), covar=cov, timestamp=start_time
    )
    s2 = GaussianState(
        StateVector([[0.0], [1.0], [20.0], [-1.0]]), covar=cov, timestamp=start_time
    )
    return [Track([s1]), Track([s2])]


def tomht_initiator_for_crossing_simple(
    start_time, measurement_model
) -> SimpleMeasurementInitiator:
    # Broad-ish prior; tune later if needed
    prior = GaussianState(
        state_vector=StateVector([[0.0], [0.0], [0.0], [0.0]]),
        covar=CovarianceMatrix(np.diag([50.0, 5.0, 50.0, 5.0])),
        timestamp=start_time,
    )
    return SimpleMeasurementInitiator(
        prior_state=prior, measurement_model=measurement_model
    )
