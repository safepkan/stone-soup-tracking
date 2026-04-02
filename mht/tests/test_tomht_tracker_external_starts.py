from __future__ import annotations

import datetime
import unittest
from typing import cast

import numpy as np
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.base import Updater

from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class _NoopHypothesiser:
    def hypothesise(self, track: Track, detections, timestamp):
        del track, detections, timestamp
        return []


class _NoopUpdater:
    def update(self, hypothesis):
        del hypothesis
        raise RuntimeError("No update expected in this test helper")


class _ZeroScoringModel:
    def score_track_hypotheses(self, *, track, hypotheses, ctx) -> dict[int, float]:
        del track, hypotheses, ctx
        return {}

    def score_unused_detections(self, *, used_det_keys: set[int], ctx) -> float:
        del used_det_keys, ctx
        return 0.0

    def score_birth(
        self, *, birth_track: Track, used_det_key: int | None, ctx
    ) -> float:
        del birth_track, used_det_key, ctx
        return 0.0


def _build_tracker() -> TOMHTTracker:
    params = TOMHTParams(
        debug_display_scan_stats=False,
        debug_display_hypotheses=False,
        debug_display_births=False,
        collect_stats=False,
    )
    return TOMHTTracker(
        hypothesiser=cast(PDAHypothesiser, _NoopHypothesiser()),
        updater=cast(Updater, _NoopUpdater()),
        params=params,
        scoring_model=_ZeroScoringModel(),
    )


def _build_tracker_with_overrides(params_overrides: dict[str, object]) -> TOMHTTracker:
    return TOMHTTracker(
        hypothesiser=cast(PDAHypothesiser, _NoopHypothesiser()),
        updater=cast(Updater, _NoopUpdater()),
        params=TOMHTParams(
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        ),
        params_overrides=params_overrides,
        scoring_model=_ZeroScoringModel(),
    )


def _external_start(timestamp: datetime.datetime) -> Track:
    state = GaussianState(
        [0.0, 0.0, 0.0, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )
    return Track([state])


class TOMHTTrackerExternalStartsTest(unittest.TestCase):
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
            _build_tracker_with_overrides(cast(dict[str, object], {1: 1}))

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

    def test_add_external_starts_inserts_new_tree_and_map_output(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.update_tracker(timestamp, [])

        start = _external_start(timestamp)
        start.metadata["opaque_source_tag"] = "upstream"
        tracker.add_external_starts(timestamp, [start])

        self.assertEqual(1, len(tracker.track_trees_by_track_id))
        tree_snapshot = tracker.get_track_tree_snapshot()
        self.assertEqual(1, len(tree_snapshot))

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(1, len(map_snapshot.leaf_nodes_by_track_id))
        leaf = next(iter(map_snapshot.leaf_nodes_by_track_id.values()))
        self.assertEqual("external_start", leaf.root_source)

        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual(1, len(output_tracks))
        output_track = next(iter(output_tracks))
        self.assertEqual(leaf.track_id, output_track.metadata["track_id"])
        self.assertEqual(leaf.node_id, output_track.metadata["node_id"])
        self.assertEqual("external_start", output_track.metadata["root_source"])
        self.assertNotIn("opaque_source_tag", output_track.metadata)

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


if __name__ == "__main__":
    unittest.main()
