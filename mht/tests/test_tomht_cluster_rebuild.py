from __future__ import annotations

import contextlib
import datetime
import io
import unittest

from mht.tests.tomht_tracker_test_support import (
    _add_manual_tree_with_committed_prefix,
    _add_manual_tree_with_live_options,
    _detection,
    _manual_cluster_for_track_ids,
)
from mht.tomht_clustering import build_track_clusters
from mht.tomht_cluster_rebuild import (
    is_global_feasible_under_live_conflicts,
    rebuild_cluster_globals,
)
from mht.tomht_cluster_solver_factory import make_cluster_solver
from mht.tomht_model import DetectionKey, ScanContext
from mht.tomht_params import TOMHTParams
from mht.tomht_pruning import apply_post_solve_supported_leaf_pruning
from mht.tomht_tree_store import TrackTreeStore


class TOMHTClusterRebuildIntegrationTest(unittest.TestCase):
    def test_clustering_ignores_committed_prefix_only_conflicts(self) -> None:
        store = TrackTreeStore()
        _add_manual_tree_with_committed_prefix(
            store,
            root_x=0.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=0),
            live_hit_score=5.0,
        )
        _add_manual_tree_with_committed_prefix(
            store,
            root_x=10.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=1),
            live_hit_score=5.0,
        )

        clusters = build_track_clusters(tree_store=store, scan_index=2)

        self.assertEqual([(0,), (1,)], [cluster.track_ids for cluster in clusters])
        self.assertTrue(all(not cluster.conflict_links for cluster in clusters))

    def test_live_unresolved_conflicts_are_enforced_by_cluster_solver(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        track0_id, track0_hit, track0_miss = _add_manual_tree_with_committed_prefix(
            store,
            root_x=0.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=0),
            live_hit_score=10.0,
            live_miss_score=9.0,
        )
        track1_id, track1_hit, _ = _add_manual_tree_with_committed_prefix(
            store,
            root_x=10.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=0),
            live_hit_score=10.0,
        )
        self.assertIsNotNone(track0_hit)
        self.assertIsNotNone(track0_miss)
        self.assertIsNotNone(track1_hit)

        clusters = build_track_clusters(tree_store=store, scan_index=2)
        self.assertEqual(1, len(clusters))
        self.assertEqual((track0_id, track1_id), clusters[0].track_ids)
        self.assertEqual(
            ((track0_id, track1_id, ((2, 0),)),), clusters[0].conflict_links
        )

        ctx = ScanContext(
            scan_index=2,
            timestamp=timestamp,
            detections=[_detection(1.0, 1.0, timestamp)],
            det_index_by_obj={},
        )
        snapshots, stats = rebuild_cluster_globals(
            clusters=clusters,
            ctx=ctx,
            tree_store=store,
            params=TOMHTParams(
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        self.assertEqual(1, len(snapshots))
        self.assertEqual(1, stats.feasible_combinations)
        map_global = snapshots[0].map_global
        self.assertIsNotNone(map_global)
        assert map_global is not None
        self.assertIs(map_global.leaf_nodes_by_track_id[track0_id], track0_miss)
        self.assertIs(map_global.leaf_nodes_by_track_id[track1_id], track1_hit)

    def test_overload_solve_returns_one_original_cluster_snapshot(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        track0_id, _ = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        track1_id, _ = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (None, 0.0)],
        )
        clusters = build_track_clusters(tree_store=store, scan_index=2)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            snapshots, stats = rebuild_cluster_globals(
                clusters=clusters,
                ctx=ScanContext(
                    scan_index=2,
                    timestamp=timestamp,
                    detections=[_detection(1.0, 1.0, timestamp)],
                    det_index_by_obj={},
                ),
                tree_store=store,
                params=TOMHTParams(
                    max_global_hypotheses=3,
                    overload_split_projected_combination_threshold=1,
                    overload_split_solution_mode="conditional_exact",
                    debug_display_scan_stats=False,
                    debug_display_hypotheses=False,
                    debug_display_births=False,
                    collect_stats=False,
                ),
                cluster_solver=make_cluster_solver("branch_and_bound"),
            )
        overload_log = stdout.getvalue()

        self.assertEqual(1, len(snapshots))
        snapshot = snapshots[0]
        self.assertEqual((track0_id, track1_id), snapshot.track_ids)
        self.assertIsNone(snapshot.overload_split_origin_cluster_id)
        self.assertGreaterEqual(stats.overload_split_clusters, 1)
        self.assertGreaterEqual(stats.overload_split_operations, 1)
        self.assertTrue(snapshot.rebuilt_globals)
        for rebuilt_global in snapshot.rebuilt_globals:
            self.assertTrue(
                is_global_feasible_under_live_conflicts(
                    global_hypothesis=rebuilt_global,
                    tree_store=store,
                )
            )
        self.assertRegex(overload_log, r"recursive_cache_hits=[1-9][0-9]*")
        self.assertIn("recursive_cache_misses=", overload_log)
        self.assertIn("max_recursion_depth=1", overload_log)
        self.assertIn("max_cut_key_count=1", overload_log)
        self.assertIn("total_interface_assignments=3", overload_log)
        self.assertIn("max_recombination_product_size=", overload_log)
        self.assertIn("branch_recomb_retained=", overload_log)
        self.assertIn(
            f"final_recomb_retained={len(snapshot.rebuilt_globals)}",
            overload_log,
        )
        self.assertIn("interface_assignment_cap_fallbacks=0", overload_log)

    def test_overload_recombination_recovers_beyond_naive_subcluster_top1(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        track0_id, track0_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        track1_id, track1_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (None, 0.0)],
        )
        clusters = build_track_clusters(tree_store=store, scan_index=2)

        snapshots, _ = rebuild_cluster_globals(
            clusters=clusters,
            ctx=ScanContext(
                scan_index=2,
                timestamp=timestamp,
                detections=[_detection(1.0, 1.0, timestamp)],
                det_index_by_obj={},
            ),
            tree_store=store,
            params=TOMHTParams(
                max_global_hypotheses=1,
                overload_split_projected_combination_threshold=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        self.assertEqual(1, len(snapshots))
        map_global = snapshots[0].map_global
        self.assertIsNotNone(map_global)
        assert map_global is not None
        self.assertIs(map_global.leaf_nodes_by_track_id[track0_id], track0_leaves[0])
        self.assertIs(map_global.leaf_nodes_by_track_id[track1_id], track1_leaves[1])
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=map_global,
                tree_store=store,
            )
        )

    def test_overload_supported_leaf_pruning_uses_original_feasible_globals(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        track0_id, track0_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        track1_id, track1_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (None, 0.0)],
        )

        snapshots, _ = rebuild_cluster_globals(
            clusters=build_track_clusters(tree_store=store, scan_index=2),
            ctx=ScanContext(
                scan_index=2,
                timestamp=timestamp,
                detections=[_detection(1.0, 1.0, timestamp)],
                det_index_by_obj={},
            ),
            tree_store=store,
            params=TOMHTParams(
                max_global_hypotheses=1,
                overload_split_projected_combination_threshold=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        stats = apply_post_solve_supported_leaf_pruning(
            cluster_snapshots=snapshots,
            tree_store=store,
        )

        self.assertEqual(2, stats.unsupported_leaf_count_pruned)
        self.assertEqual(
            {track0_leaves[0].node_id},
            store.track_trees_by_track_id[track0_id].active_leaf_node_ids,
        )
        self.assertEqual(
            {track1_leaves[1].node_id},
            store.track_trees_by_track_id[track1_id].active_leaf_node_ids,
        )

    def test_overload_recombined_globals_are_deterministically_ordered(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        track0_id, track0_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(None, 1.0), (None, 1.0)],
        )
        track1_id, track1_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(None, 0.0)],
        )
        cluster = _manual_cluster_for_track_ids(
            store=store,
            track_ids=(track0_id, track1_id),
            scan_index=2,
        )

        snapshots, _ = rebuild_cluster_globals(
            clusters=[cluster],
            ctx=ScanContext(
                scan_index=2,
                timestamp=timestamp,
                detections=[],
                det_index_by_obj={},
            ),
            tree_store=store,
            params=TOMHTParams(
                max_global_hypotheses=2,
                overload_split_projected_combination_threshold=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        selections = [
            (
                global_hypothesis.leaf_nodes_by_track_id[track0_id].node_id,
                global_hypothesis.leaf_nodes_by_track_id[track1_id].node_id,
            )
            for global_hypothesis in snapshots[0].rebuilt_globals
        ]
        self.assertEqual(
            [
                (track0_leaves[0].node_id, track1_leaves[0].node_id),
                (track0_leaves[1].node_id, track1_leaves[0].node_id),
            ],
            selections,
        )

    def test_overload_no_feasible_conditional_branch_reports_clear_error(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (shared_key, 8.0)],
        )
        _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (shared_key, 7.0)],
        )

        with self.assertRaisesRegex(RuntimeError, "no feasible combination"):
            rebuild_cluster_globals(
                clusters=build_track_clusters(tree_store=store, scan_index=2),
                ctx=ScanContext(
                    scan_index=2,
                    timestamp=timestamp,
                    detections=[_detection(1.0, 1.0, timestamp)],
                    det_index_by_obj={},
                ),
                tree_store=store,
                params=TOMHTParams(
                    max_global_hypotheses=1,
                    overload_split_projected_combination_threshold=1,
                    debug_display_scan_stats=False,
                    debug_display_hypotheses=False,
                    debug_display_births=False,
                    collect_stats=False,
                ),
                cluster_solver=make_cluster_solver("branch_and_bound"),
            )


if __name__ == "__main__":
    unittest.main()
