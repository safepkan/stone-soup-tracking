from __future__ import annotations

import contextlib
import datetime
import io
import unittest

import numpy as np
from stonesoup.types.state import GaussianState

from mht.tomht_clustering import build_track_clusters
from mht.tomht_cluster_rebuild import (
    is_global_feasible_under_live_conflicts,
    rebuild_cluster_globals,
)
from mht.tomht_cluster_solver_factory import make_cluster_solver
from mht.tomht_model import (
    ClusterRebuildSnapshot,
    DetectionKey,
    GlobalHypothesis,
    ScanContext,
    TrackHypothesisNode,
)
from mht.tomht_params import TOMHTParams
from mht.tomht_pruning import apply_post_solve_supported_leaf_pruning
from mht.tomht_tree_store import TrackTreeStore

_ASSOC_MISS = -2


def _state(x: float, timestamp: datetime.datetime) -> GaussianState:
    return GaussianState(
        np.array([[float(x)], [0.0], [0.0], [0.0]]),
        covar=np.eye(4),
        timestamp=timestamp,
    )


def _add_manual_tree_with_live_options(
    store: TrackTreeStore,
    *,
    root_x: float,
    live_options: list[tuple[DetectionKey | None, float]],
) -> tuple[int, list[TrackHypothesisNode]]:
    t1 = datetime.datetime(2026, 3, 28, 10, 0, 1)
    t2 = t1 + datetime.timedelta(seconds=1)
    track_id = store.allocate_track_id()
    root_key = DetectionKey(scan_index=1, det_index=track_id)
    root = store.create_root_node(
        track_id=track_id,
        scan_index=1,
        timestamp=t1,
        state=_state(root_x, t1),
        state_kind="manual_root",
        used_det_key=root_key,
        assoc_label=int(root_key.det_index),
        log_delta=0.0,
        age=1,
        hits=1,
        root_source="manual",
    )
    tree = store.add_track_tree_for_root(root, root_source="manual")
    tree.committed_detection_keys = frozenset({root_key})

    leaves: list[TrackHypothesisNode] = []
    for option_index, (live_det_key, score) in enumerate(live_options):
        if live_det_key is None:
            assoc_label = _ASSOC_MISS
            state = _state(root_x, t2)
            hits = 1
            missed_count = 1
            last_det_key = root.last_det_key
            last_det_hit = False
        else:
            assoc_label = int(live_det_key.det_index)
            state = _state(root_x + float(option_index + 1), t2)
            hits = 2
            missed_count = 0
            last_det_key = live_det_key
            last_det_hit = True

        leaf = store.create_track_hypothesis_node(
            track_id=track_id,
            parent=root,
            scan_index=2,
            timestamp=t2,
            state=state,
            state_kind="manual_live",
            used_det_key=live_det_key,
            assoc_label=assoc_label,
            log_delta=float(score),
            age=2,
            hits=hits,
            missed_count=missed_count,
            last_det_key=last_det_key,
            last_det_hit=last_det_hit,
            root_source=root.root_source,
            birth_scan_index=root.birth_scan_index,
        )
        leaves.append(leaf)

    tree.active_leaf_node_ids = {leaf.node_id for leaf in leaves}
    return track_id, leaves


def _run_rebuild(
    store: TrackTreeStore,
    *,
    mode: str,
    max_global_hypotheses: int = 1,
) -> tuple[ClusterRebuildSnapshot, str]:
    timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        snapshots, _ = rebuild_cluster_globals(
            clusters=build_track_clusters(tree_store=store, scan_index=2),
            ctx=ScanContext(
                scan_index=2,
                timestamp=timestamp,
                detections=[],
                det_index_by_obj={},
            ),
            tree_store=store,
            params=TOMHTParams(
                max_global_hypotheses=max_global_hypotheses,
                overload_split_projected_combination_threshold=1,
                overload_split_solution_mode=mode,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )
    if len(snapshots) != 1:
        raise AssertionError(f"Expected one snapshot, got {len(snapshots)}.")
    return snapshots[0], stdout.getvalue()


def _selected_leaf(
    snapshot: ClusterRebuildSnapshot,
    *,
    track_id: int,
) -> TrackHypothesisNode:
    return _map_global(snapshot).leaf_nodes_by_track_id[track_id]


def _map_global(snapshot: ClusterRebuildSnapshot) -> GlobalHypothesis:
    if snapshot.map_global is None:
        raise AssertionError("Expected a MAP global.")
    return snapshot.map_global


class TOMHTOverloadGreedyPartitionTest(unittest.TestCase):
    def test_params_validate_greedy_mode_controls(self) -> None:
        self.assertEqual(
            "greedy_partition",
            TOMHTParams().overload_split_solution_mode,
        )
        TOMHTParams(overload_split_solution_mode="greedy_partition")
        TOMHTParams(overload_split_solution_mode="conditional_exact")

        with self.assertRaisesRegex(ValueError, "overload_split_solution_mode"):
            TOMHTParams(overload_split_solution_mode="not-a-mode")

        with self.assertRaisesRegex(
            ValueError,
            "overload_split_greedy_ownership_metric",
        ):
            TOMHTParams(overload_split_greedy_ownership_metric="not-a-metric")

    def test_greedy_ownership_uses_higher_best_claiming_leaf_score(self) -> None:
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        left_id, left_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 4.0), (None, 0.0)],
        )
        right_id, right_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 5.0), (None, 0.0)],
        )

        snapshot, overload_log = _run_rebuild(store, mode="greedy_partition")

        self.assertIs(_selected_leaf(snapshot, track_id=left_id), left_leaves[1])
        self.assertIs(_selected_leaf(snapshot, track_id=right_id), right_leaves[0])
        self.assertIn("stop=greedy_partition", overload_log)
        self.assertIn("greedy_assign_l=0", overload_log)
        self.assertIn("greedy_assign_r=1", overload_log)
        self.assertIn("greedy_fallbacks=0", overload_log)
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=_map_global(snapshot),
                tree_store=store,
            )
        )

    def test_greedy_ownership_tie_breaks_to_left_track_tuple(self) -> None:
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        left_id, left_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 5.0), (None, 0.0)],
        )
        right_id, right_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 5.0), (None, 0.0)],
        )

        snapshot, overload_log = _run_rebuild(store, mode="greedy_partition")

        self.assertIs(_selected_leaf(snapshot, track_id=left_id), left_leaves[0])
        self.assertIs(_selected_leaf(snapshot, track_id=right_id), right_leaves[1])
        self.assertIn("greedy_assign_l=1", overload_log)
        self.assertIn("greedy_assign_r=0", overload_log)

    def test_greedy_releases_first_side_assigned_but_unused_cut_keys(self) -> None:
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        left_id, left_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 5.0), (None, 10.0)],
        )
        right_id, right_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 4.0), (None, 0.0)],
        )

        snapshot, overload_log = _run_rebuild(store, mode="greedy_partition")

        self.assertIs(_selected_leaf(snapshot, track_id=left_id), left_leaves[1])
        self.assertIs(_selected_leaf(snapshot, track_id=right_id), right_leaves[0])
        self.assertIn("greedy_released=1", overload_log)
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=_map_global(snapshot),
                tree_store=store,
            )
        )

    def test_greedy_claimed_first_side_keys_remain_forbidden(self) -> None:
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        left_id, left_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 9.0)],
        )
        right_id, right_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.5), (None, 0.0)],
        )

        snapshot, overload_log = _run_rebuild(store, mode="greedy_partition")

        self.assertIs(_selected_leaf(snapshot, track_id=left_id), left_leaves[0])
        self.assertIs(_selected_leaf(snapshot, track_id=right_id), right_leaves[1])
        self.assertIn("greedy_released=0", overload_log)
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=_map_global(snapshot),
                tree_store=store,
            )
        )

    def test_greedy_falls_back_when_partition_has_no_second_solution(self) -> None:
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        left_id, left_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        right_id, right_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0)],
        )

        snapshot, overload_log = _run_rebuild(store, mode="greedy_partition")

        self.assertIs(_selected_leaf(snapshot, track_id=left_id), left_leaves[1])
        self.assertIs(_selected_leaf(snapshot, track_id=right_id), right_leaves[0])
        self.assertIn("greedy_fallbacks=1", overload_log)
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=_map_global(snapshot),
                tree_store=store,
            )
        )

    def test_conditional_exact_mode_does_not_emit_greedy_diagnostics(self) -> None:
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        left_id, left_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        right_id, right_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (None, 0.0)],
        )

        snapshot, overload_log = _run_rebuild(store, mode="conditional_exact")

        self.assertIs(_selected_leaf(snapshot, track_id=left_id), left_leaves[0])
        self.assertIs(_selected_leaf(snapshot, track_id=right_id), right_leaves[1])
        self.assertIn("stop=recursive_conditioning", overload_log)
        self.assertNotIn("greedy_", overload_log)

    def test_supported_leaf_pruning_applies_after_greedy_overload_solve(self) -> None:
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        left_id, left_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 9.0)],
        )
        right_id, right_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.5), (None, 0.0)],
        )

        snapshot, _ = _run_rebuild(store, mode="greedy_partition")
        stats = apply_post_solve_supported_leaf_pruning(
            cluster_snapshots=[snapshot],
            tree_store=store,
        )

        self.assertEqual(2, stats.unsupported_leaf_count_pruned)
        self.assertEqual(
            {left_leaves[0].node_id},
            store.track_trees_by_track_id[left_id].active_leaf_node_ids,
        )
        self.assertEqual(
            {right_leaves[1].node_id},
            store.track_trees_by_track_id[right_id].active_leaf_node_ids,
        )

    def test_greedy_can_drop_kbest_optimality_while_remaining_feasible(self) -> None:
        greedy_store = TrackTreeStore()
        exact_store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        greedy_left_id, greedy_left_leaves = _add_manual_tree_with_live_options(
            greedy_store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 9.0)],
        )
        greedy_right_id, greedy_right_leaves = _add_manual_tree_with_live_options(
            greedy_store,
            root_x=10.0,
            live_options=[(shared_key, 9.5), (None, 0.0)],
        )
        exact_left_id, exact_left_leaves = _add_manual_tree_with_live_options(
            exact_store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 9.0)],
        )
        exact_right_id, exact_right_leaves = _add_manual_tree_with_live_options(
            exact_store,
            root_x=10.0,
            live_options=[(shared_key, 9.5), (None, 0.0)],
        )

        greedy_snapshot, _ = _run_rebuild(greedy_store, mode="greedy_partition")
        exact_snapshot, _ = _run_rebuild(exact_store, mode="conditional_exact")

        self.assertIs(
            _selected_leaf(greedy_snapshot, track_id=greedy_left_id),
            greedy_left_leaves[0],
        )
        self.assertIs(
            _selected_leaf(greedy_snapshot, track_id=greedy_right_id),
            greedy_right_leaves[1],
        )
        self.assertIs(
            _selected_leaf(exact_snapshot, track_id=exact_left_id),
            exact_left_leaves[1],
        )
        self.assertIs(
            _selected_leaf(exact_snapshot, track_id=exact_right_id),
            exact_right_leaves[0],
        )
        self.assertLess(
            float(_map_global(greedy_snapshot).log_weight),
            float(_map_global(exact_snapshot).log_weight),
        )
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=_map_global(greedy_snapshot),
                tree_store=greedy_store,
            )
        )
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=_map_global(exact_snapshot),
                tree_store=exact_store,
            )
        )


if __name__ == "__main__":
    unittest.main()
