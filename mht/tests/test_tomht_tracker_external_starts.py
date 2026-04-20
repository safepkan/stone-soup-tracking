from __future__ import annotations

import datetime
import unittest
from typing import Iterable, cast

import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

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


class _ZeroScoringModel:
    def score_track_hypotheses(self, *, hypotheses, ctx) -> list[float]:
        del ctx
        return [0.0 for _ in hypotheses]

    def score_unused_detections(self, *, used_det_keys: set[int], ctx) -> float:
        del used_det_keys, ctx
        return 0.0

    def score_birth(
        self, *, birth_track: Track, used_det_key: int | None, ctx
    ) -> float:
        del birth_track, used_det_key, ctx
        return 0.0


def _build_tracker() -> TOMHTTracker:
    return TOMHTTracker(
        hypothesiser=_NoopHypothesiser(),
        updater=_NoopUpdater(),
        params=TOMHTParams(
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        ),
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
