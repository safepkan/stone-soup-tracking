from __future__ import annotations

import datetime

import numpy as np
from ordered_set import OrderedSet

from stonesoup.types.array import StateVector
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track


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


def initial_mfa_tracks_for_bearing_range(
    start_time: datetime.datetime,
) -> OrderedSet[Track]:
    """Create the three priors used in the Stone Soup MFT example."""
    cov = np.diag([10, 1, 10, 1])
    priors = [
        StateVector([10, 1, 10, 1]),
        StateVector([-10, -1, -10, -1]),
        StateVector([-10, -1, 10, 1]),
    ]

    tracks: OrderedSet[Track] = OrderedSet()
    for state_vector in priors:
        gm = GaussianMixture(
            [
                TaggedWeightedGaussianState(
                    state_vector,
                    cov,
                    timestamp=start_time,
                    weight=Probability(1),
                    tag=[],
                )
            ]
        )
        tracks.add(Track([gm]))

    return tracks
