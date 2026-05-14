from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import unittest
from typing import Any, Callable, Iterable

import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from mht.tomht_hypothesiser import TrackerOwnedNLLDistanceHypothesiser
from mht.tomht_scoring import NLLScoringModel
from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class _NoopPredictor:
    def predict(self, prior, timestamp=None, **kwargs):
        del prior, timestamp, kwargs
        raise RuntimeError("No prediction expected in this test helper")


class _NoopHypothesiser:
    def __init__(self) -> None:
        self.predictor = _NoopPredictor()

    def hypothesise(
        self,
        track: Track,
        detections: Iterable[Detection],
        timestamp,
        **kwargs,
    ) -> MultipleHypothesis:
        del detections, kwargs
        prediction = track.states[-1]
        return MultipleHypothesis(
            [
                SingleDistanceHypothesis(
                    prediction=prediction,
                    measurement=MissedDetection(timestamp=timestamp),
                    distance=0.0,
                )
            ],
            normalise=False,
        )


class _NoopUpdater:
    def update(self, hypothesis):
        del hypothesis
        raise RuntimeError("No update expected in this test helper")


class _LegacyPositionalNoopHypothesiser:
    def __init__(self, predictor: _NoopPredictor, updater: _NoopUpdater) -> None:
        self.predictor = predictor
        self.updater = updater

    def hypothesise(
        self,
        track: Track,
        detections: Iterable[Detection],
        timestamp,
        **kwargs,
    ) -> MultipleHypothesis:
        del kwargs
        prediction = track.states[-1]
        del detections
        return MultipleHypothesis(
            [
                SingleDistanceHypothesis(
                    prediction=prediction,
                    measurement=MissedDetection(timestamp=timestamp),
                    distance=0.0,
                )
            ],
            normalise=False,
        )


@dataclass(frozen=True, order=True, unsafe_hash=True, slots=True)
class _SystemTrackId:
    id: int
    metadata: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __iter__(self):
        return iter((self.id,))

    def __int__(self):
        return self.id

    def __str__(self):
        return f"sys:{self.id}"


def _build_tracker(
    *,
    params: TOMHTParams | None = None,
    output_track_id_mapper: Callable[[int], object] | None = None,
) -> TOMHTTracker:
    if params is None:
        params = TOMHTParams(
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )
    return TOMHTTracker(
        hypothesiser=_NoopHypothesiser(),
        updater=_NoopUpdater(),
        params=params,
        output_track_id_mapper=output_track_id_mapper,
    )


def _build_tracker_with_overrides(params_overrides: dict[str, object]) -> TOMHTTracker:
    return TOMHTTracker(
        hypothesiser=_NoopHypothesiser(),
        updater=_NoopUpdater(),
        params=TOMHTParams(
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        ),
        params_overrides=params_overrides,
    )


def _external_start(timestamp: datetime.datetime) -> Track:
    state = GaussianState(
        [0.0, 0.0, 0.0, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )
    return Track([state])


class TOMHTTrackerBasicsTest(unittest.TestCase):
    def test_removed_parameters_are_absent(self) -> None:
        param_fields = TOMHTParams.__dataclass_fields__
        self.assertNotIn("scoring_mode", param_fields)
        self.assertNotIn("internal_birth_mode", param_fields)
        self.assertNotIn("birth_log_penalty", param_fields)
        self.assertNotIn("birth_density", param_fields)

    def test_constructor_builds_nll_scoring_model_from_params(self) -> None:
        params = TOMHTParams(
            prob_detect=0.7,
            clutter_density=0.25,
            log_epsilon=1e-9,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )

        tracker = TOMHTTracker(
            hypothesiser=_NoopHypothesiser(),
            updater=_NoopUpdater(),
            params=params,
        )

        self.assertIsInstance(tracker.scoring_model, NLLScoringModel)
        self.assertEqual(0.7, tracker.scoring_model.prob_detect)
        self.assertEqual(0.25, tracker.scoring_model.clutter_density)
        self.assertEqual(1e-9, tracker.scoring_model.log_epsilon)

    def test_initiator_start_initial_existence_probability_rejects_boundaries(
        self,
    ) -> None:
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "initiator_start_initial_existence_probability",
                ):
                    TOMHTParams(
                        initiator_start_initial_existence_probability=probability,
                    )

    def test_track_confirmation_existence_probability_rejects_boundaries(
        self,
    ) -> None:
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "track_confirmation_existence_probability",
                ):
                    TOMHTParams(
                        track_confirmation_existence_probability=probability,
                    )

    def test_publication_params_validate_domains(self) -> None:
        self.assertEqual(("confirmed",), TOMHTParams().publish_lifecycle_states)
        TOMHTParams(publish_lifecycle_states=())

        invalid_cases: list[tuple[dict[str, Any], str]] = [
            ({"publish_lifecycle_states": ("tentative", "invalid")}, "lifecycle"),
            ({"publish_lifecycle_states": "confirmed"}, "not a string"),
            ({"publish_min_hits": -1}, "publish_min_hits"),
            ({"publish_min_age": -1}, "publish_min_age"),
            ({"publish_min_existence_probability": -0.1}, "existence"),
            ({"publish_min_existence_probability": 1.0}, "existence"),
            ({"publish_min_existence_probability": float("nan")}, "existence"),
            ({"publish_min_existence_probability": float("inf")}, "existence"),
        ]
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    TOMHTParams(**overrides)

    def test_constructor_applies_params_overrides(self) -> None:
        tracker = _build_tracker_with_overrides(
            {
                "max_children_per_track": 3,
                "debug_display_births": True,
            }
        )

        self.assertEqual(3, tracker.params.max_children_per_track)
        self.assertTrue(tracker.params.debug_display_births)
        self.assertFalse(tracker.params.debug_display_scan_stats)

    def test_constructor_rejects_unknown_params_override_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown TOMHTParams override key"):
            _build_tracker_with_overrides({"not_a_param": 1})

    def test_constructor_rejects_non_string_params_override_keys(self) -> None:
        with self.assertRaisesRegex(TypeError, "params_overrides keys must be strings"):
            _build_tracker_with_overrides({1: 1})  # type: ignore[dict-item]

    def test_constructor_rejects_both_predictor_and_hypothesiser(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "exactly one of predictor or hypothesiser"
        ):
            TOMHTTracker(
                predictor=_NoopPredictor(),
                updater=_NoopUpdater(),
                hypothesiser=_NoopHypothesiser(),
            )

    def test_constructor_rejects_neither_predictor_nor_hypothesiser(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "exactly one of predictor or hypothesiser"
        ):
            TOMHTTracker(
                updater=_NoopUpdater(),
            )

    def test_constructor_with_predictor_builds_default_distance_hypothesiser(
        self,
    ) -> None:
        tracker = TOMHTTracker(
            predictor=_NoopPredictor(),
            updater=_NoopUpdater(),
            params=TOMHTParams(
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )
        self.assertIsInstance(
            tracker._hypothesiser, TrackerOwnedNLLDistanceHypothesiser
        )

    def test_constructor_with_hypothesiser_keeps_instance(self) -> None:
        custom_hypothesiser = _LegacyPositionalNoopHypothesiser(
            _NoopPredictor(), _NoopUpdater()
        )
        tracker = TOMHTTracker(
            hypothesiser=custom_hypothesiser,
            updater=_NoopUpdater(),
            params=TOMHTParams(
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        self.assertIs(custom_hypothesiser, tracker._hypothesiser)

    def test_tracker_starts_with_empty_map_and_no_trees(self) -> None:
        tracker = _build_tracker()

        self.assertEqual({}, tracker.track_trees_by_track_id)
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual({}, dict(map_snapshot.leaf_nodes_by_track_id))
        self.assertEqual(set(), tracker.get_map_output_tracks())

    def test_update_tracker_returns_timestamp_and_tracks(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        result_timestamp, tracks = tracker.update_tracker(timestamp, set())

        self.assertEqual(timestamp, result_timestamp)
        self.assertEqual(set(), tracks)
        self.assertEqual(set(), tracker.tracks)

    def test_get_unused_detections_rejects_call_before_update_tracker(self) -> None:
        tracker = _build_tracker()

        with self.assertRaisesRegex(RuntimeError, "completed update_tracker"):
            tracker.get_unused_detections()

    def test_get_unused_detections_returns_residual_when_initiator_disabled(
        self,
    ) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        detection = Detection(np.array([[1.0], [2.0]]), timestamp=timestamp)

        tracker.update_tracker(timestamp, [detection])

        unused = tracker.get_unused_detections()
        self.assertEqual([detection], unused)
        unused.clear()
        self.assertEqual([detection], tracker.get_unused_detections())

    def test_map_output_tracks_default_track_id_is_stable_integer(self) -> None:
        tracker = _build_tracker(
            params=TOMHTParams(
                publish_lifecycle_states=("tentative", "confirmed"),
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            )
        )
        t0 = datetime.datetime(2026, 3, 12, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_external_start(t0)])

        output_track_t0 = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(0, output_track_t0.id)
        self.assertIsInstance(output_track_t0.id, int)

        tracker.update_tracker(t1, [])
        output_track_t1 = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(0, output_track_t1.id)
        self.assertIsInstance(output_track_t1.id, int)

    def test_map_output_tracks_support_custom_track_id_mapping(self) -> None:
        tracker = _build_tracker(
            params=TOMHTParams(
                publish_lifecycle_states=("tentative", "confirmed"),
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            output_track_id_mapper=lambda track_id: _SystemTrackId(
                id=track_id,
                metadata={"origin": "test"},
            ),
        )
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_external_start(timestamp)])

        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertIsInstance(output_track.id, _SystemTrackId)
        self.assertEqual(0, int(output_track.id))
        self.assertEqual("sys:0", str(output_track.id))
        self.assertEqual({"origin": "test"}, output_track.id.metadata)


if __name__ == "__main__":
    unittest.main()
