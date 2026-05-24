from __future__ import annotations

from dataclasses import fields
import datetime
import unittest

from mht.tests.tomht_tracker_test_support import (
    _build_tracker,
    _detection,
    _ScriptedHypothesiser,
    _ScriptedUpdater,
    _single_track_rebuild_snapshot,
    _tracker_with_manual_frontier,
    _track_start,
)
from mht.tomht_tracker import TOMHTParams


class TOMHTPruningIntegrationTest(unittest.TestCase):
    def test_map_only_n_scan_pruning_promotes_map_branch_and_collects_disagreement(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                ns_scan_window=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 5.0), (1, 1.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
        )

        scan1_map = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(scan1_map)
        assert scan1_map is not None
        scan1_leaf = scan1_map.leaf_nodes_by_track_id[0]
        self.assertEqual((1, 0), scan1_leaf.used_det_key)

        hypothesiser.set_options(timestamp=t2, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t2, [])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual(scan1_leaf.node_id, tree.root_node_id)

        nscan = tracker.get_n_scan_commitment_snapshot()
        self.assertEqual(1, nscan.boundary_scan_index)
        self.assertIn(0, nscan.latest_committed_ancestor_by_track_id)
        self.assertEqual(
            scan1_leaf.node_id,
            nscan.latest_committed_ancestor_by_track_id[0].node_id,
        )

        clusters = tracker.get_last_cluster_snapshots()
        self.assertEqual(1, len(clusters))
        self.assertGreaterEqual(clusters[0].disagreement_count, 1)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        self.assertGreaterEqual(tracker.last_scan_stats.nscan_disagreement_total, 1)
        timings = tracker.last_scan_stats.timing_breakdown
        phase_sum = (
            timings.prep_ctx_ms
            + timings.pre_expand_validate_ms
            + timings.expand_ms
            + timings.post_expand_prune_validate_ms
            + timings.births_ms
            + timings.cluster_build_and_solve_ms
            + timings.post_solve_prune_ms
            + timings.map_merge_ms
            + timings.nscan_prune_ms
            + timings.lifecycle_ms
            + timings.publication_ms
            + timings.cleanup_ms
        )
        self.assertGreaterEqual(timings.cluster_build_and_solve_ms, 0.0)
        self.assertGreaterEqual(timings.expand_ms, 0.0)
        self.assertGreaterEqual(timings.expand_hypothesise_calls, 0)
        self.assertGreaterEqual(timings.expand_track_reconstruct_calls, 0)
        self.assertGreaterEqual(timings.expand_track_reconstruct_ms, 0.0)
        self.assertGreaterEqual(timings.expand_default_state_fast_path_calls, 0)
        self.assertGreaterEqual(timings.expand_update_calls, 0)
        self.assertGreaterEqual(timings.nscan_prune_ms, 0.0)
        self.assertGreaterEqual(timings.lifecycle_ms, 0.0)
        self.assertGreaterEqual(timings.lifecycle_deleter_track_reconstruct_calls, 0)
        self.assertGreaterEqual(timings.lifecycle_deleter_track_reconstruct_ms, 0.0)
        self.assertGreaterEqual(
            timings.lifecycle_default_miss_deleter_fast_path_calls, 0
        )
        self.assertGreaterEqual(timings.lifecycle_deleter_check_ms, 0.0)
        self.assertGreaterEqual(timings.publication_ms, 0.0)
        self.assertLessEqual(
            timings.expand_hypothesise_ms
            + timings.expand_track_reconstruct_ms
            + timings.expand_update_ms,
            timings.expand_ms + 1.0,
        )
        self.assertLessEqual(
            timings.lifecycle_deleter_track_reconstruct_ms
            + timings.lifecycle_deleter_check_ms,
            timings.lifecycle_ms + 1.0,
        )
        self.assertLessEqual(
            phase_sum, tracker.last_scan_stats.scan_wall_ms + 1.0
        )  # keep tolerance for measurement overhead/noise

    def test_supported_leaf_pruning_applies_to_all_snapshots(self) -> None:
        tracker, leaves = _tracker_with_manual_frontier()
        snapshot = _single_track_rebuild_snapshot(
            track_id=0,
            supported_leaves=[leaves[0]],
            overload_split_origin_cluster_id=10,
        )

        stats = tracker._apply_post_solve_supported_leaf_pruning([snapshot])

        active_leaf_ids = tracker.track_trees_by_track_id[0].active_leaf_node_ids
        self.assertEqual({leaves[0].node_id}, active_leaf_ids)
        self.assertEqual(2, stats.unsupported_leaf_count_pruned)
        for stat_field in fields(stats):
            self.assertGreaterEqual(getattr(stats, stat_field.name), 0)

    def test_supported_leaf_pruning_is_deterministic(self) -> None:
        tracker, leaves = _tracker_with_manual_frontier()
        snapshot = _single_track_rebuild_snapshot(
            track_id=0,
            supported_leaves=[leaves[0], leaves[2]],
            overload_split_origin_cluster_id=10,
        )

        first_stats = tracker._apply_post_solve_supported_leaf_pruning([snapshot])
        active_leaf_ids = tracker.track_trees_by_track_id[0].active_leaf_node_ids

        self.assertEqual({leaves[0].node_id, leaves[2].node_id}, active_leaf_ids)
        self.assertEqual(1, first_stats.unsupported_leaf_count_pruned)
        for stat_field in fields(first_stats):
            self.assertGreaterEqual(getattr(first_stats, stat_field.name), 0)

        tracker_again, leaves_again = _tracker_with_manual_frontier()
        snapshot_again = _single_track_rebuild_snapshot(
            track_id=0,
            supported_leaves=[leaves_again[0], leaves_again[2]],
            overload_split_origin_cluster_id=10,
        )
        second_stats = tracker_again._apply_post_solve_supported_leaf_pruning(
            [snapshot_again]
        )
        self.assertEqual(first_stats, second_stats)


if __name__ == "__main__":
    unittest.main()
