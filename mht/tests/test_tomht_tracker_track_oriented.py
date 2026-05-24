from __future__ import annotations

import datetime
import unittest
from unittest import mock

from stonesoup.types.track import Track

from mht.tests.tomht_tracker_test_support import (
    _build_tracker,
    _detection,
    _RecordingMetadataHypothesiser,
    _RecordingPriorPredictor,
    _ScriptedHypothesiser,
    _ScriptedUpdater,
    _sigmoid,
    _track_start,
)
from mht.tomht_hypothesiser import TrackerOwnedNLLDistanceHypothesiser
from mht.tomht_output import reconstruct_track_from_leaf_node
from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class TOMHTTrackOrientedArchitectureTest(unittest.TestCase):
    def test_explicit_trees_clustering_and_rebuilt_globals(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0), _track_start(10.0, t0)])

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 5.0), (None, 0.0)],
        )
        hypothesiser.set_options(
            timestamp=t1,
            track_id=1,
            options=[(0, 4.0), (1, 3.0), (None, 0.0)],
        )

        det0 = _detection(1.0, 1.0, t1)
        det1 = _detection(2.0, 2.0, t1)
        tracker.update_tracker(t1, [det0, det1])

        self.assertEqual({0, 1}, set(tracker.track_trees_by_track_id.keys()))
        for tree in tracker.track_trees_by_track_id.values():
            root = tracker._nodes_by_id[tree.root_node_id]
            self.assertGreater(len(root.child_node_ids), 0)
            self.assertGreater(len(tree.active_leaf_node_ids), 0)

        clusters = tracker.get_last_cluster_snapshots()
        self.assertEqual(1, len(clusters))
        cluster = clusters[0]
        self.assertEqual((0, 1), cluster.track_ids)
        self.assertGreaterEqual(len(cluster.conflict_links), 1)
        self.assertGreaterEqual(len(cluster.rebuilt_globals), 2)

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(2, len(map_snapshot.leaf_nodes_by_track_id))
        used_det_indices = {
            int(leaf.used_det_key.det_index)
            for leaf in map_snapshot.leaf_nodes_by_track_id.values()
            if leaf.used_det_key is not None
        }
        self.assertEqual({0, 1}, used_det_indices)

        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual(2, len(output_tracks))
        for output_track in output_tracks:
            track_id = int(output_track.metadata["track_id"])
            leaf = map_snapshot.leaf_nodes_by_track_id[track_id]
            existence_log_odds = float(leaf.accumulated_log_score)
            self.assertAlmostEqual(
                existence_log_odds,
                output_track.metadata["existence_log_odds"],
            )
            self.assertAlmostEqual(
                _sigmoid(existence_log_odds),
                output_track.metadata["existence_probability"],
            )

        tree_snapshot = tracker.get_track_tree_snapshot()
        self.assertEqual({0, 1}, set(tree_snapshot.keys()))

    def test_custom_hypothesiser_receives_tomht_track_metadata(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _RecordingMetadataHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        tracker.update_tracker(t1, [])

        self.assertEqual(1, len(hypothesiser.track_metadata))
        metadata = hypothesiser.track_metadata[0]
        self.assertEqual(1, len(hypothesiser.track_objects))
        self.assertIsInstance(hypothesiser.track_objects[0], Track)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        timing = tracker.last_scan_stats.timing_breakdown
        self.assertEqual(1, timing.expand_track_reconstruct_calls)
        self.assertGreaterEqual(timing.expand_track_reconstruct_ms, 0.0)
        self.assertEqual(0, timing.expand_default_state_fast_path_calls)
        self.assertEqual(0, metadata["internal_track_id"])
        self.assertEqual(0, metadata["public_track_id"])
        self.assertEqual("confirmed", metadata["lifecycle_state"])
        self.assertEqual("published", metadata["publication_state"])

    def test_default_hypothesiser_expansion_skips_track_reconstruction(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        predictor = _RecordingPriorPredictor()
        tracker = TOMHTTracker(
            predictor=predictor,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                max_missed=999,
                enable_default_hypothesiser_state_fast_path=True,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        tree = tracker.track_trees_by_track_id[0]
        leaf = tracker._nodes_by_id[next(iter(tree.active_leaf_node_ids))]

        with mock.patch(
            "mht.tomht_expansion.reconstruct_track_from_leaf_node",
            side_effect=AssertionError(
                "default hypothesiser should use state fast path"
            ),
        ):
            tracker.update_tracker(t1, [])

        self.assertEqual(1, len(predictor.priors))
        self.assertIs(predictor.priors[0], leaf.state)
        self.assertNotIsInstance(predictor.priors[0], Track)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        timing = tracker.last_scan_stats.timing_breakdown
        self.assertEqual(0, timing.expand_track_reconstruct_calls)
        self.assertEqual(0.0, timing.expand_track_reconstruct_ms)
        self.assertEqual(1, timing.expand_default_state_fast_path_calls)

    def test_default_hypothesiser_fast_path_flag_false_uses_track_api(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        predictor = _RecordingPriorPredictor()
        tracker = TOMHTTracker(
            predictor=predictor,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                max_missed=999,
                enable_default_hypothesiser_state_fast_path=False,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        tree = tracker.track_trees_by_track_id[0]
        leaf = tracker._nodes_by_id[next(iter(tree.active_leaf_node_ids))]

        with (
            mock.patch(
                "mht.tomht_expansion.reconstruct_track_from_leaf_node",
                wraps=reconstruct_track_from_leaf_node,
            ) as reconstruct_mock,
            mock.patch.object(
                TrackerOwnedNLLDistanceHypothesiser,
                "hypothesise",
                autospec=True,
                side_effect=TrackerOwnedNLLDistanceHypothesiser.hypothesise,
            ) as hypothesise_mock,
        ):
            tracker.update_tracker(t1, [])

        self.assertEqual(1, reconstruct_mock.call_count)
        self.assertEqual(1, hypothesise_mock.call_count)
        hypothesise_args = hypothesise_mock.call_args.args
        self.assertIsInstance(hypothesise_args[1], Track)
        self.assertEqual(1, len(predictor.priors))
        self.assertIs(predictor.priors[0], leaf.state)
        self.assertNotIsInstance(predictor.priors[0], Track)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        timing = tracker.last_scan_stats.timing_breakdown
        self.assertEqual(1, timing.expand_track_reconstruct_calls)
        self.assertGreaterEqual(timing.expand_track_reconstruct_ms, 0.0)
        self.assertEqual(0, timing.expand_default_state_fast_path_calls)

    def test_track_tree_committed_detection_keys_start_empty(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual(frozenset(), tree.committed_detection_keys)


if __name__ == "__main__":
    unittest.main()
