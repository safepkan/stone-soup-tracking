from __future__ import annotations

import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.hypothesiser.base import Hypothesiser
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.base import Updater
from stonesoup.updater.kalman import KalmanUpdater

from mht.tomht_hypothesiser import TrackerOwnedNLLDistanceHypothesiser
from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class ProfileStampedHypothesis(SingleDistanceHypothesis):
    """Distance hypothesis carrying the profile key that produced it."""

    profile = Property(str, doc="Profile key that produced this hypothesis.")


class ProfileDispatchingUpdater(Updater):
    """Updater delegating update() to the profile that emitted the hypothesis."""

    profiles = Property(
        dict, doc="Mapping from profile key to (hypothesiser, updater)."
    )
    measurement_model = Property(
        MeasurementModel,
        default=None,
        doc="Unused; the per-profile updaters own their measurement models.",
    )

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):
        # Top-level TOMHT updater use only: components that need measurement
        # predictions (e.g. the tracker-owned default hypothesiser) cannot
        # use this dispatcher.
        raise NotImplementedError(
            "ProfileDispatchingUpdater only supports update(hypothesis)."
        )

    def update(self, hypothesis, **kwargs):
        profile_key = getattr(hypothesis, "profile", None)
        if profile_key is None:
            raise TypeError(
                "ProfileDispatchingUpdater requires ProfileStampedHypothesis input."
            )
        _, profile_updater = self.profiles[profile_key]
        return profile_updater.update(hypothesis, **kwargs)


class ProfileSwitchingHypothesiser(Hypothesiser):
    """Per-track (hypothesiser, updater) dispatch on caller metadata.

    Each profile is a self-contained pair satisfying the §4
    custom-hypothesiser contract, and all profiles emit distances under the
    same NLL convention.
    """

    profiles = Property(
        dict, doc="Mapping from profile key to (hypothesiser, updater)."
    )
    default_profile = Property(
        str, doc="Profile key used when a track carries no profile metadata."
    )
    metadata_key = Property(
        str,
        default="imm_profile",
        doc=(
            "Caller metadata key holding the requested profile. "
            "Stored values must be keys of profiles."
        ),
    )

    @property
    def predictor(self):
        # §4 wiring requirement: expose a predictor attribute.
        profile_hypothesiser, _ = self.profiles[self.default_profile]
        return profile_hypothesiser.predictor

    @property
    def updater(self):
        # One paired dispatching updater per adapter instance.
        if not hasattr(self, "_dispatching_updater"):
            self._dispatching_updater = ProfileDispatchingUpdater(
                profiles=self.profiles
            )
        return self._dispatching_updater

    def hypothesise(self, track, detections, timestamp, **kwargs):
        profile_key = track.metadata.get(self.metadata_key, self.default_profile)
        profile_hypothesiser, _ = self.profiles[profile_key]
        hypotheses = profile_hypothesiser.hypothesise(
            track, detections, timestamp, **kwargs
        )
        return MultipleHypothesis(
            [
                ProfileStampedHypothesis(
                    prediction=hyp.prediction,
                    measurement=hyp.measurement,
                    measurement_prediction=hyp.measurement_prediction,
                    distance=hyp.distance,
                    profile=profile_key,
                )
                for hyp in hypotheses
            ]
        )


class CountingKalmanUpdater(KalmanUpdater):
    profile_label: str
    call_counts: dict[str, int]

    def update(self, hypothesis, **kwargs):
        self.call_counts[self.profile_label] += 1
        return super().update(hypothesis, **kwargs)


def _profile_pair(
    *,
    noise_diff_coeff: float,
    profile_label: str,
    call_counts: dict[str, int],
    measurement_model: LinearGaussian,
) -> tuple[TrackerOwnedNLLDistanceHypothesiser, CountingKalmanUpdater]:
    transition_model = CombinedLinearGaussianTransitionModel(
        [
            ConstantVelocity(noise_diff_coeff),
            ConstantVelocity(noise_diff_coeff),
        ]
    )
    predictor = KalmanPredictor(transition_model)
    updater = CountingKalmanUpdater(measurement_model)
    updater.profile_label = profile_label
    updater.call_counts = call_counts
    hypothesiser = TrackerOwnedNLLDistanceHypothesiser(
        predictor=predictor,
        updater=updater,
        mahalanobis_gate_threshold=5.0,
    )
    return hypothesiser, updater


def _run_target_scans(
    tracker: TOMHTTracker,
    measurement_model: LinearGaussian,
    start_time: datetime.datetime,
    scan_numbers: range,
) -> None:
    for scan_number in scan_numbers:
        timestamp = start_time + datetime.timedelta(seconds=scan_number)
        detection = Detection(
            [float(scan_number), float(scan_number)],
            timestamp=timestamp,
            measurement_model=measurement_model,
        )
        tracker.update_tracker(time=timestamp, detections=[detection])


def test_runtime_per_track_behavior_switching_example() -> None:
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=0.5 * np.eye(2),
    )
    call_counts = {"A": 0, "B": 0}
    switcher = ProfileSwitchingHypothesiser(
        profiles={
            "A": _profile_pair(
                noise_diff_coeff=0.05,
                profile_label="A",
                call_counts=call_counts,
                measurement_model=measurement_model,
            ),
            "B": _profile_pair(
                noise_diff_coeff=2.0,
                profile_label="B",
                call_counts=call_counts,
                measurement_model=measurement_model,
            ),
        },
        default_profile="A",
    )
    tracker = TOMHTTracker(
        updater=switcher.updater,
        hypothesiser=switcher,
        params=TOMHTParams(
            prob_detect=0.9,
            clutter_density=1e-4,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        ),
    )
    start_time = datetime.datetime(2026, 8, 27, 12, 0, 0)
    tracker.update_tracker(time=start_time, detections=[])
    tracker.add_external_starts(
        start_time,
        [
            Track(
                [
                    GaussianState(
                        [0.0, 1.0, 0.0, 1.0],
                        covar=np.eye(4),
                        timestamp=start_time,
                    )
                ]
            )
        ],
    )

    _run_target_scans(tracker, measurement_model, start_time, range(1, 5))

    count_a_before_switch = call_counts["A"]
    count_b_before_switch = call_counts["B"]
    assert count_a_before_switch > 0
    assert count_b_before_switch == 0

    output_tracks = tracker.get_map_output_tracks(include_unpublished=True)
    assert len(output_tracks) == 1
    output_track = next(iter(output_tracks))
    internal_track_id = output_track.metadata["internal_track_id"]
    tracker.update_track_metadata(
        internal_track_id=internal_track_id,
        updates={"imm_profile": "B"},
    )

    _run_target_scans(tracker, measurement_model, start_time, range(5, 9))

    assert call_counts["A"] == count_a_before_switch
    assert call_counts["B"] > count_b_before_switch
    switched_output_tracks = tracker.get_map_output_tracks(include_unpublished=True)
    assert len(switched_output_tracks) == 1
    switched_output_track = next(iter(switched_output_tracks))
    assert switched_output_track.metadata["imm_profile"] == "B"
