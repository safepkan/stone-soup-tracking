from __future__ import annotations

import datetime
import unittest
from typing import cast

import numpy as np
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.updater.base import Updater

from mht_experiments.tomht_tracker import (
    GlobalHypothesis,
    ScanContext,
    TOMHTParams,
    TrackHypothesisNode,
    TOMHTTracker,
)


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


class _SingleBirthInitiator:
    def __init__(self, birth_track: Track) -> None:
        self._birth_track = birth_track

    def initiate(self, detections, timestamp):
        del detections, timestamp
        return [self._birth_track]


def _build_tracker(
    *,
    initiator: SimpleMeasurementInitiator | None = None,
) -> TOMHTTracker:
    params = TOMHTParams(
        debug_display_scan_stats=False,
        debug_display_hypotheses=False,
        debug_display_births=False,
        collect_stats=False,
    )
    return TOMHTTracker(
        hypothesiser=cast(PDAHypothesiser, object()),
        updater=cast(Updater, object()),
        initiator=initiator,
        params=params,
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
    def _assert_leaf_maintenance_fields(
        self,
        node: TrackHypothesisNode,
        *,
        age: int,
        hits: int,
        missed_count: int,
        last_det_key: int | None,
        last_det_hit: bool,
        root_source: str,
    ) -> None:
        self.assertIsInstance(node.track_id, int)
        self.assertEqual(age, node.age)
        self.assertEqual(hits, node.hits)
        self.assertEqual(missed_count, node.missed_count)
        self.assertEqual(last_det_key, node.last_det_key)
        self.assertEqual(last_det_hit, node.last_det_hit)
        self.assertEqual(root_source, node.root_source)

    def test_tracker_starts_with_empty_initial_global_hypothesis(self) -> None:
        tracker = _build_tracker()

        self.assertEqual(1, len(tracker.global_hypotheses))
        self.assertEqual({}, tracker.global_hypotheses[0].leaf_nodes_by_track_id)
        self.assertEqual(0.0, tracker.global_hypotheses[0].log_weight)
        self.assertEqual(0, tracker._next_track_id)
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(0.0, map_snapshot.log_weight)
        self.assertEqual({}, dict(map_snapshot.leaf_nodes_by_track_id))
        self.assertEqual(set(), tracker.get_map_output_tracks())

    def test_update_tracker_returns_timestamp_and_tracks(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        result_timestamp, tracks = tracker.update_tracker(timestamp, set())

        self.assertEqual(timestamp, result_timestamp)
        self.assertEqual(set(), tracks)
        self.assertEqual(set(), tracker.tracks)

    def test_tracker_iter_uses_detector_and_returns_update_output(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        tracker.detector = [(timestamp, set())]

        result_timestamp, tracks = next(iter(tracker))

        self.assertEqual(timestamp, result_timestamp)
        self.assertEqual(set(), tracks)

    def test_map_snapshot_leaf_mapping_is_read_only(self) -> None:
        tracker = _build_tracker()
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        mutable_view = cast(
            dict[int, TrackHypothesisNode],
            map_snapshot.leaf_nodes_by_track_id,
        )
        with self.assertRaises(TypeError):
            mutable_view[0] = cast(TrackHypothesisNode, object())

    def test_internal_birth_inserted_track_metadata_uses_shared_conventions(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        birth_track = _external_start(timestamp)
        birth_track.metadata["source"] = "initiator"
        tracker = _build_tracker(
            initiator=cast(
                SimpleMeasurementInitiator,
                _SingleBirthInitiator(birth_track),
            )
        )
        detection = Detection(np.array([[1.0], [2.0]]), timestamp=timestamp)
        ctx = ScanContext(
            scan_index=0,
            timestamp=timestamp,
            detections=[detection],
            det_index_by_obj={id(detection): 0},
        )

        tracker._branch_globals_with_births(ctx)
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        inserted_node = next(iter(map_snapshot.leaf_nodes_by_track_id.values()))

        self._assert_leaf_maintenance_fields(
            inserted_node,
            age=1,
            hits=0,
            missed_count=0,
            last_det_key=None,
            last_det_hit=False,
            root_source="internal_birth",
        )
        self.assertEqual("initiator", inserted_node.track_metadata["source"])

    def test_add_external_starts_rejects_call_before_update_tracker(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        with self.assertRaisesRegex(
            RuntimeError, "completed update_tracker\\(\\) first"
        ):
            tracker.add_external_starts([_external_start(timestamp)], timestamp)

    def test_add_external_starts_accepts_matching_timestamp_after_update_tracker(
        self,
    ) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts([], timestamp)

        self.assertEqual(1, len(tracker.global_hypotheses))
        self.assertEqual({}, tracker.global_hypotheses[0].leaf_nodes_by_track_id)

    def test_add_external_starts_rejects_mismatched_timestamp_after_update_tracker(
        self,
    ) -> None:
        tracker = _build_tracker()
        update_timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        external_timestamp = update_timestamp + datetime.timedelta(seconds=1)

        tracker.update_tracker(update_timestamp, [])

        with self.assertRaisesRegex(
            ValueError,
            "must match the most recent completed update_tracker\\(\\) timestamp",
        ):
            tracker.add_external_starts(
                [_external_start(external_timestamp)],
                external_timestamp,
            )

    def test_add_external_starts_uses_most_recent_update_tracker_timestamp(
        self,
    ) -> None:
        tracker = _build_tracker()
        first_timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        second_timestamp = first_timestamp + datetime.timedelta(seconds=1)

        tracker.update_tracker(first_timestamp, [])
        tracker.update_tracker(second_timestamp, [])

        with self.assertRaises(ValueError):
            tracker.add_external_starts(
                [_external_start(first_timestamp)],
                first_timestamp,
            )

        tracker.add_external_starts(
            [_external_start(second_timestamp)],
            second_timestamp,
        )

    def test_add_external_starts_inserts_into_each_active_global(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        tracker.update_tracker(timestamp, [])
        tracker.global_hypotheses = [
            GlobalHypothesis(leaf_nodes_by_track_id={}, log_weight=-1.5),
            GlobalHypothesis(leaf_nodes_by_track_id={}, log_weight=-2.5),
        ]

        tracker.add_external_starts([_external_start(timestamp)], timestamp)

        self.assertEqual(2, len(tracker.global_hypotheses))
        first_id, first_node = next(
            iter(tracker.global_hypotheses[0].leaf_nodes_by_track_id.items())
        )
        second_id, second_node = next(
            iter(tracker.global_hypotheses[1].leaf_nodes_by_track_id.items())
        )

        self.assertEqual(first_id, second_id)
        self.assertEqual(-1.5, tracker.global_hypotheses[0].log_weight)
        self.assertEqual(-2.5, tracker.global_hypotheses[1].log_weight)
        self.assertIs(first_node, second_node)

    def test_add_external_starts_empty_list_is_noop(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)

        tracker.update_tracker(timestamp, [])
        before_next_track_id = tracker._next_track_id
        before_globals = [
            (gh.log_weight, dict(gh.leaf_nodes_by_track_id))
            for gh in tracker.global_hypotheses
        ]

        tracker.add_external_starts([], timestamp)

        after_globals = [
            (gh.log_weight, dict(gh.leaf_nodes_by_track_id))
            for gh in tracker.global_hypotheses
        ]
        self.assertEqual(before_next_track_id, tracker._next_track_id)
        self.assertEqual(before_globals, after_globals)

    def test_inserted_external_start_metadata_is_initialised(self) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        external_start = _external_start(timestamp)
        external_start.metadata["source"] = "upstream"

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts([external_start], timestamp)

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        inserted_node = next(iter(map_snapshot.leaf_nodes_by_track_id.values()))

        self._assert_leaf_maintenance_fields(
            inserted_node,
            age=1,
            hits=1,
            missed_count=0,
            last_det_key=None,
            last_det_hit=False,
            root_source="external_start",
        )
        self.assertEqual("upstream", inserted_node.track_metadata["source"])

    def test_add_external_starts_repeated_insertion_creates_distinct_tracks(
        self,
    ) -> None:
        tracker = _build_tracker()
        timestamp = datetime.datetime(2026, 3, 12, 10, 0, 0)
        external_start = _external_start(timestamp)

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts([external_start], timestamp)
        tracker.add_external_starts([external_start], timestamp)

        track_ids = sorted(tracker.global_hypotheses[0].leaf_nodes_by_track_id)
        self.assertEqual(2, len(track_ids))
        self.assertNotEqual(track_ids[0], track_ids[1])


if __name__ == "__main__":
    unittest.main()
