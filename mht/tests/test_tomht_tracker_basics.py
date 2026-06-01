from __future__ import annotations

from dataclasses import dataclass, field, replace
import datetime
from pathlib import Path
import unittest
from typing import Callable, Iterable

import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from mht.tomht_hypothesiser import TrackerOwnedNLLDistanceHypothesiser
from mht.tomht_scoring import ConstantDetectionProbabilityModel, NLLScoringModel
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


class _EndpointDetectionProbabilityModel:
    def detection_probability(
        self,
        *,
        track_id: object | None,
        prediction,
        caller_scan_context: object | None,
    ) -> float:
        del track_id, prediction, caller_scan_context
        return 0.0

    def clutter_density(
        self,
        *,
        prediction,
        detection: Detection | None,
        caller_scan_context: object | None,
    ) -> float:
        del prediction, detection, caller_scan_context
        return 1.0


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
            clutter_density=1.0,
        )
    elif params.clutter_density <= 0.0:
        params = replace(params, clutter_density=1.0)
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
            clutter_density=1.0,
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
    def test_active_conflict_code_uses_live_conflict_helpers(self) -> None:
        mht_dir = Path(__file__).resolve().parents[1]
        active_conflict_modules = [
            "tomht_clustering.py",
            "tomht_cluster_rebuild.py",
            "tomht_births.py",
        ]

        for module_name in active_conflict_modules:
            with self.subTest(module_name=module_name):
                source = (mht_dir / module_name).read_text(encoding="utf-8")
                self.assertNotIn("leaf.detection_history_keys", source)

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
        self.assertIsInstance(
            tracker.scoring_model.detection_probability_model,
            ConstantDetectionProbabilityModel,
        )
        constant_dpm = tracker.scoring_model.detection_probability_model
        assert isinstance(constant_dpm, ConstantDetectionProbabilityModel)
        self.assertEqual(0.7, constant_dpm.prob_detect)
        self.assertEqual(0.25, constant_dpm.constant_clutter_density)
        self.assertEqual(1e-9, tracker.scoring_model.log_epsilon)

    def test_constructor_rejects_default_clutter_density_without_custom_dpm(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "clutter_density must be a positive, finite density",
        ):
            TOMHTTracker(
                hypothesiser=_NoopHypothesiser(),
                updater=_NoopUpdater(),
                params=TOMHTParams(
                    debug_display_scan_stats=False,
                    debug_display_hypotheses=False,
                    debug_display_births=False,
                    collect_stats=False,
                ),
            )

    def test_constructor_ignores_scalar_scoring_params_with_custom_dpm(self) -> None:
        dpm = _EndpointDetectionProbabilityModel()

        tracker = TOMHTTracker(
            hypothesiser=_NoopHypothesiser(),
            updater=_NoopUpdater(),
            params=TOMHTParams(
                prob_detect=0.0,
                clutter_density=0.0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            detection_probability_model=dpm,
        )

        self.assertIs(dpm, tracker.scoring_model.detection_probability_model)

    def test_constructor_applies_params_overrides(self) -> None:
        tracker = _build_tracker_with_overrides(
            {
                "max_children_per_leaf": 3,
                "debug_display_births": True,
            }
        )

        self.assertEqual(3, tracker.params.max_children_per_leaf)
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
                clutter_density=1.0,
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
                clutter_density=1.0,
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
        self.assertEqual(0, output_track_t0.metadata["track_id"])
        self.assertEqual(0, output_track_t0.metadata["internal_track_id"])
        self.assertEqual(0, output_track_t0.metadata["public_track_id"])

        tracker.update_tracker(t1, [])
        output_track_t1 = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(0, output_track_t1.id)
        self.assertIsInstance(output_track_t1.id, int)
        self.assertEqual(0, output_track_t1.metadata["public_track_id"])

    def test_map_output_tracks_support_custom_track_id_mapping(self) -> None:
        mapper_calls: list[int] = []

        def mapper(track_id: int) -> _SystemTrackId:
            mapper_calls.append(track_id)
            return _SystemTrackId(
                id=track_id,
                metadata={"origin": "test"},
            )

        tracker = _build_tracker(
            params=TOMHTParams(
                publish_lifecycle_states=("tentative", "confirmed"),
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            output_track_id_mapper=mapper,
        )
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_external_start(timestamp)])

        self.assertEqual([0], mapper_calls)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertIsInstance(output_track.id, _SystemTrackId)
        self.assertEqual(0, int(output_track.id))
        self.assertEqual("sys:0", str(output_track.id))
        self.assertEqual({"origin": "test"}, output_track.id.metadata)
        self.assertEqual(output_track.id, output_track.metadata["public_track_id"])
        self.assertEqual(0, output_track.metadata["track_id"])
        self.assertEqual(0, output_track.metadata["internal_track_id"])

        tracker.get_map_output_tracks()
        self.assertEqual([0], mapper_calls)


if __name__ == "__main__":
    unittest.main()
