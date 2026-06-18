from __future__ import annotations

import datetime
from math import exp, log, log1p
import unittest
from typing import Any, Iterable, cast

import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from mht.tomht_model import TrackHypothesisNode
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


def _quiet_params(**overrides: Any) -> TOMHTParams:
    defaults: dict[str, Any] = {
        "debug_display_scan_stats": False,
        "debug_display_hypotheses": False,
        "debug_display_births": False,
        "collect_stats": False,
        "clutter_density": 1.0,
    }
    defaults.update(overrides)
    return TOMHTParams(**defaults)


def _build_tracker(params: TOMHTParams | None = None) -> TOMHTTracker:
    if params is None:
        params = _quiet_params()
    return TOMHTTracker(
        hypothesiser=_NoopHypothesiser(),
        updater=_NoopUpdater(),
        params=params,
    )


def _external_start(timestamp: datetime.datetime) -> Track:
    state = GaussianState(
        [0.0, 0.0, 0.0, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )
    return Track([state])


def _logit(probability: float) -> float:
    return log(probability) - log1p(-probability)


def _sigmoid(log_odds: float) -> float:
    return 1.0 / (1.0 + exp(-log_odds))


def _single_map_leaf(tracker: TOMHTTracker) -> TrackHypothesisNode:
    map_snapshot = tracker.get_map_hypothesis_snapshot()
    assert map_snapshot is not None
    assert len(map_snapshot.leaf_nodes_by_track_id) == 1
    return next(iter(map_snapshot.leaf_nodes_by_track_id.values()))


def _external_start_leaf_for_metadata(
    metadata: dict[str, object],
    *,
    default_probability: float = 0.8,
) -> TrackHypothesisNode:
    tracker = _build_tracker(
        params=_quiet_params(
            external_start_initial_existence_probability=default_probability,
        )
    )
    timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
    tracker.update_tracker(timestamp, [])
    start = _external_start(timestamp)
    start.metadata.update(metadata)

    tracker.add_external_starts(timestamp, [start])

    return _single_map_leaf(tracker)


class TOMHTTrackerExternalStartsTest(unittest.TestCase):
    def test_add_external_starts_rejects_call_before_update_tracker(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        with self.assertRaisesRegex(RuntimeError, "completed update_tracker"):
            tracker.add_external_starts(timestamp, [_external_start(timestamp)])

    def test_add_external_starts_rejects_non_datetime_time(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])

        with self.assertRaisesRegex(TypeError, r"time must be a datetime.datetime"):
            tracker.add_external_starts(cast(datetime.datetime, object()), [])

    def test_add_external_starts_rejects_mismatched_timestamp_after_update_tracker(
        self,
    ) -> None:
        tracker = _build_tracker()
        update_timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        external_timestamp = update_timestamp + datetime.timedelta(seconds=1)

        tracker.update_tracker(update_timestamp, [])

        with self.assertRaisesRegex(
            ValueError,
            "must match the most recent completed update_tracker",
        ):
            tracker.add_external_starts(
                external_timestamp,
                [_external_start(external_timestamp)],
            )

    def test_default_external_start_confirms_and_publishes_immediately(
        self,
    ) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])

        start = _external_start(timestamp)
        start.metadata["opaque_source_tag"] = "upstream"
        tracker.add_external_starts(timestamp, [start])

        self.assertEqual(1, len(tracker.track_trees_by_track_id))
        tree_snapshot = tracker.get_track_tree_snapshot()
        self.assertEqual(1, len(tree_snapshot))
        tree = next(iter(tracker.track_trees_by_track_id.values()))
        self.assertEqual("confirmed", tree.lifecycle_state)
        self.assertEqual("published", tree.publication_state)
        self.assertEqual(0, tree.public_track_id)
        self.assertEqual(
            "published",
            next(iter(tree_snapshot.values()))["publication_state"],
        )
        self.assertEqual(0, next(iter(tree_snapshot.values()))["public_track_id"])

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(1, len(map_snapshot.leaf_nodes_by_track_id))
        leaf = next(iter(map_snapshot.leaf_nodes_by_track_id.values()))
        self.assertEqual("external_start", leaf.root_source)
        expected_log_delta = _logit(0.95)
        self.assertAlmostEqual(expected_log_delta, leaf.log_delta)
        self.assertAlmostEqual(expected_log_delta, leaf.accumulated_log_score)
        self.assertAlmostEqual(expected_log_delta, map_snapshot.log_weight)

        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual(1, len(output_tracks))
        output_track = next(iter(output_tracks))
        self.assertEqual(0, output_track.id)
        self.assertEqual(leaf.track_id, output_track.metadata["track_id"])
        self.assertEqual(leaf.track_id, output_track.metadata["internal_track_id"])
        self.assertEqual(0, output_track.metadata["public_track_id"])
        self.assertEqual(leaf.node_id, output_track.metadata["node_id"])
        self.assertEqual("external_start", output_track.metadata["root_source"])
        self.assertEqual("confirmed", output_track.metadata["lifecycle_state"])
        self.assertEqual("published", output_track.metadata["publication_state"])
        self.assertAlmostEqual(
            expected_log_delta,
            output_track.metadata["existence_log_odds"],
        )
        self.assertAlmostEqual(
            _sigmoid(expected_log_delta),
            output_track.metadata["existence_probability"],
        )
        self.assertNotIn("opaque_source_tag", output_track.metadata)

    def test_external_start_caller_metadata_whitelist_copies_requested_keys(
        self,
    ) -> None:
        tracker = _build_tracker(
            params=_quiet_params(
                external_start_caller_metadata_keys=("sensor_id", "imm_profile"),
            )
        )
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])
        start = _external_start(timestamp)
        start.metadata.update(
            {
                "sensor_id": "radar-a",
                "imm_profile": "cv_ca",
                "opaque_source_tag": "not-whitelisted",
                "age": 3,
                "hits": 2,
            }
        )

        tracker.add_external_starts(timestamp, [start])

        tree = next(iter(tracker.track_trees_by_track_id.values()))
        self.assertEqual(
            {"sensor_id": "radar-a", "imm_profile": "cv_ca"},
            tree.caller_metadata,
        )
        tree_snapshot = next(iter(tracker.get_track_tree_snapshot().values()))
        self.assertEqual(
            {"sensor_id": "radar-a", "imm_profile": "cv_ca"},
            tree_snapshot["caller_metadata"],
        )
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual("radar-a", output_track.metadata["sensor_id"])
        self.assertEqual("cv_ca", output_track.metadata["imm_profile"])
        self.assertNotIn("opaque_source_tag", output_track.metadata)
        self.assertEqual(3, output_track.metadata["age"])
        self.assertEqual(2, output_track.metadata["hits"])

    def test_external_start_caller_metadata_whitelist_rejects_tomht_owned_keys(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "TOMHT-owned.*age"):
            _quiet_params(external_start_caller_metadata_keys=("sensor_id", "age"))

    def test_update_track_metadata_modifies_caller_metadata_only(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_external_start(timestamp)])

        tracker.update_track_metadata(
            internal_track_id=0,
            updates={"sensor_id": "radar-a", "track_class": "fo"},
        )
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual("radar-a", output_track.metadata["sensor_id"])
        self.assertEqual("fo", output_track.metadata["track_class"])

        tracker.update_track_metadata(
            public_track_id=output_track.id,
            updates={"imm_profile": "cv_ca"},
            remove_keys=("sensor_id",),
        )
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertNotIn("sensor_id", output_track.metadata)
        self.assertEqual("fo", output_track.metadata["track_class"])
        self.assertEqual("cv_ca", output_track.metadata["imm_profile"])

    def test_update_track_metadata_rejects_tomht_owned_keys(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_external_start(timestamp)])

        with self.assertRaisesRegex(ValueError, "TOMHT-owned.*existence_probability"):
            tracker.update_track_metadata(
                internal_track_id=0,
                updates={"existence_probability": 0.8},
            )
        with self.assertRaisesRegex(ValueError, "TOMHT-owned.*age"):
            tracker.update_track_metadata(
                internal_track_id=0,
                remove_keys=("age",),
            )

    def test_metadata_low_probability_external_start_remains_unpublished(
        self,
    ) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])
        start = _external_start(timestamp)
        start.metadata["existence_probability"] = 0.6

        tracker.add_external_starts(timestamp, [start])

        tree = next(iter(tracker.track_trees_by_track_id.values()))
        self.assertEqual("tentative", tree.lifecycle_state)
        self.assertEqual("unpublished", tree.publication_state)
        self.assertIsNone(tree.public_track_id)
        self.assertEqual(set(), tracker.get_map_output_tracks())

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(0, inspection_track.id)
        self.assertEqual(0, inspection_track.metadata["internal_track_id"])
        self.assertIsNone(inspection_track.metadata["public_track_id"])
        self.assertEqual("tentative", inspection_track.metadata["lifecycle_state"])
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )
        self.assertAlmostEqual(
            _logit(0.6),
            inspection_track.metadata["existence_log_odds"],
        )

    def test_add_external_starts_does_not_run_deletion_lifecycle(self) -> None:
        tracker = _build_tracker(
            params=_quiet_params(
                track_deletion_existence_probability=0.4,
                collect_stats=True,
            )
        )
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])
        stats_before = tracker.last_scan_stats
        start = _external_start(timestamp)
        start.metadata["existence_probability"] = 0.1

        tracker.add_external_starts(timestamp, [start])

        self.assertIs(stats_before, tracker.last_scan_stats)
        self.assertEqual(1, len(tracker.track_trees_by_track_id))
        tree = next(iter(tracker.track_trees_by_track_id.values()))
        self.assertEqual("tentative", tree.lifecycle_state)
        self.assertEqual("unpublished", tree.publication_state)
        self.assertEqual(
            1,
            len(tracker.get_map_output_tracks(include_unpublished=True)),
        )

    def test_external_start_tree_can_remain_unpublished_by_policy(self) -> None:
        tracker = _build_tracker(params=_quiet_params(publish_lifecycle_states=()))
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])

        tracker.add_external_starts(timestamp, [_external_start(timestamp)])

        self.assertEqual(1, len(tracker.track_trees_by_track_id))
        tree = next(iter(tracker.track_trees_by_track_id.values()))
        self.assertEqual("confirmed", tree.lifecycle_state)
        self.assertEqual("unpublished", tree.publication_state)
        self.assertEqual(set(), tracker.get_map_output_tracks())

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )

    def test_add_external_starts_repeated_insertion_creates_distinct_track_ids(
        self,
    ) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])

        tracker.add_external_starts(timestamp, [_external_start(timestamp)])
        tracker.add_external_starts(timestamp, [_external_start(timestamp)])

        self.assertEqual(2, len(tracker.track_trees_by_track_id))
        self.assertEqual([0, 1], sorted(tracker.track_trees_by_track_id.keys()))
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual([0, 1], sorted(map_snapshot.leaf_nodes_by_track_id))
        self.assertAlmostEqual(2.0 * _logit(0.95), map_snapshot.log_weight)

    def test_add_external_starts_missing_existence_probability_uses_params_default(
        self,
    ) -> None:
        tracker = _build_tracker(
            params=_quiet_params(
                external_start_initial_existence_probability=0.8,
            )
        )
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])

        tracker.add_external_starts(timestamp, [_external_start(timestamp)])

        leaf = _single_map_leaf(tracker)
        expected_log_delta = _logit(0.8)
        self.assertAlmostEqual(expected_log_delta, leaf.log_delta)
        self.assertAlmostEqual(expected_log_delta, leaf.accumulated_log_score)
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        assert map_snapshot is not None
        self.assertAlmostEqual(expected_log_delta, map_snapshot.log_weight)

    def test_add_external_starts_uses_metadata_existence_probability_override(
        self,
    ) -> None:
        tracker = _build_tracker(
            params=_quiet_params(
                external_start_initial_existence_probability=0.8,
            )
        )
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])
        start = _external_start(timestamp)
        start.metadata["existence_probability"] = 0.6

        tracker.add_external_starts(timestamp, [start])

        leaf = _single_map_leaf(tracker)
        expected_log_delta = _logit(0.6)
        self.assertAlmostEqual(expected_log_delta, leaf.log_delta)
        self.assertAlmostEqual(expected_log_delta, leaf.accumulated_log_score)

    def test_add_external_starts_uses_metadata_existence_log_odds_directly(
        self,
    ) -> None:
        for value in (-2.5, 1000.0):
            with self.subTest(value=value):
                leaf = _external_start_leaf_for_metadata(
                    {"existence_log_odds": value},
                )

                self.assertAlmostEqual(value, leaf.log_delta)
                self.assertAlmostEqual(value, leaf.accumulated_log_score)

    def test_add_external_starts_log_odds_metadata_precedes_probability(
        self,
    ) -> None:
        leaf = _external_start_leaf_for_metadata(
            {
                "existence_log_odds": 1.25,
                "existence_probability": 0.6,
            },
        )

        self.assertAlmostEqual(1.25, leaf.log_delta)
        self.assertAlmostEqual(1.25, leaf.accumulated_log_score)

    def test_add_external_starts_invalid_log_odds_metadata_falls_back_to_probability(
        self,
    ) -> None:
        invalid_values: list[Any] = [
            "not-a-number",
            float("nan"),
            float("inf"),
            float("-inf"),
            None,
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                leaf = _external_start_leaf_for_metadata(
                    {
                        "existence_log_odds": value,
                        "existence_probability": 0.6,
                    },
                )

                expected_log_delta = _logit(0.6)
                self.assertAlmostEqual(expected_log_delta, leaf.log_delta)
                self.assertAlmostEqual(expected_log_delta, leaf.accumulated_log_score)

    def test_add_external_starts_invalid_existence_metadata_falls_back(
        self,
    ) -> None:
        invalid_metadata: list[dict[str, object]] = [
            {
                "existence_log_odds": "not-a-number",
                "existence_probability": "also-not-a-number",
            },
            {
                "existence_log_odds": float("nan"),
                "existence_probability": 0.0,
            },
            {
                "existence_log_odds": float("inf"),
                "existence_probability": 1.0,
            },
        ]
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                leaf = _external_start_leaf_for_metadata(metadata)

                expected_log_delta = _logit(0.8)
                self.assertAlmostEqual(expected_log_delta, leaf.log_delta)
                self.assertAlmostEqual(expected_log_delta, leaf.accumulated_log_score)

    def test_add_external_starts_invalid_metadata_existence_probability_falls_back(
        self,
    ) -> None:
        invalid_values: list[Any] = [
            "not-a-number",
            float("nan"),
            float("inf"),
            -0.1,
            1.1,
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                tracker = _build_tracker(
                    params=_quiet_params(
                        external_start_initial_existence_probability=0.8,
                    )
                )
                timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
                tracker.update_tracker(timestamp, [])
                start = _external_start(timestamp)
                start.metadata["existence_probability"] = value

                tracker.add_external_starts(timestamp, [start])

                leaf = _single_map_leaf(tracker)
                expected_log_delta = _logit(0.8)
                self.assertAlmostEqual(expected_log_delta, leaf.log_delta)
                self.assertAlmostEqual(expected_log_delta, leaf.accumulated_log_score)

    def test_add_external_starts_boundary_metadata_existence_probability_falls_back(
        self,
    ) -> None:
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                tracker = _build_tracker(
                    params=_quiet_params(
                        external_start_initial_existence_probability=0.8,
                    )
                )
                timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
                tracker.update_tracker(timestamp, [])
                start = _external_start(timestamp)
                start.metadata["existence_probability"] = value

                tracker.add_external_starts(timestamp, [start])

                leaf = _single_map_leaf(tracker)
                expected_log_delta = _logit(0.8)
                self.assertAlmostEqual(expected_log_delta, leaf.log_delta)
                self.assertAlmostEqual(expected_log_delta, leaf.accumulated_log_score)

    def test_external_start_initial_existence_probability_rejects_boundaries(
        self,
    ) -> None:
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "external_start_initial_existence_probability",
                ):
                    _quiet_params(
                        external_start_initial_existence_probability=probability,
                    )


if __name__ == "__main__":
    unittest.main()
