from __future__ import annotations

import datetime
import unittest
from typing import cast

import numpy as np
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.types.state import GaussianState
from stonesoup.updater.base import Updater

from mht.tomht_tracker import (
    GlobalHypothesis,
    TOMHTParams,
    TOMHTTracker,
    TrackHypothesisNode,
)


class _ZeroScoringModel:
    def score_track_hypotheses(self, *, track, hypotheses, ctx) -> dict[int, float]:
        del track, hypotheses, ctx
        return {}

    def score_unused_detections(self, *, used_det_keys: set[int], ctx) -> float:
        del used_det_keys, ctx
        return 0.0

    def score_birth(self, *, birth_track, used_det_key: int | None, ctx) -> float:
        del birth_track, used_det_key, ctx
        return 0.0


class TOMHTTrackerNScanCommitmentTest(unittest.TestCase):
    def _build_tracker(self, *, ns_scan_window: int) -> TOMHTTracker:
        params = TOMHTParams(
            ns_scan_window=ns_scan_window,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )
        return TOMHTTracker(
            hypothesiser=cast(PDAHypothesiser, object()),
            updater=cast(Updater, object()),
            params=params,
            scoring_model=_ZeroScoringModel(),
        )

    @staticmethod
    def _state(scan_index: int) -> GaussianState:
        timestamp = datetime.datetime(2026, 3, 19, 12, 0, 0) + datetime.timedelta(
            seconds=scan_index
        )
        return GaussianState(
            [float(scan_index), 0.0, 0.0, 0.0],
            covar=np.eye(4),
            timestamp=timestamp,
        )

    def _make_node(
        self,
        tracker: TOMHTTracker,
        *,
        track_id: int,
        scan_index: int,
        parent: TrackHypothesisNode | None,
    ) -> TrackHypothesisNode:
        if parent is None:
            age = 1
            hits = 1
            birth_scan_index = scan_index
        else:
            age = int(parent.age) + 1
            hits = int(parent.hits)
            birth_scan_index = int(parent.birth_scan_index)
        return tracker._create_track_hypothesis_node(
            track_id=track_id,
            parent=parent,
            scan_index=scan_index,
            timestamp=cast(datetime.datetime, self._state(scan_index).timestamp),
            state=self._state(scan_index),
            state_kind="test",
            used_det_key=None,
            assoc_label=TOMHTTracker.ASSOC_PAD,
            log_delta=0.0,
            age=age,
            hits=hits,
            missed_count=0,
            last_det_key=None,
            last_det_hit=False,
            root_source="test",
            birth_scan_index=birth_scan_index,
        )

    def test_commit_when_surviving_globals_agree_on_boundary_ancestor(self) -> None:
        tracker = self._build_tracker(ns_scan_window=1)
        track_id = 7
        root = self._make_node(tracker, track_id=track_id, scan_index=0, parent=None)
        shared = self._make_node(tracker, track_id=track_id, scan_index=1, parent=root)
        leaf_a = self._make_node(
            tracker, track_id=track_id, scan_index=2, parent=shared
        )
        leaf_b = self._make_node(
            tracker, track_id=track_id, scan_index=2, parent=shared
        )

        boundary, in_scope, committed_count = tracker._update_n_scan_commitment(
            scan_index=2,
            post_beam_globals=[
                GlobalHypothesis({track_id: leaf_a}, log_weight=0.0),
                GlobalHypothesis({track_id: leaf_b}, log_weight=-1.0),
            ],
        )

        self.assertEqual(1, boundary)
        self.assertEqual(1, in_scope)
        self.assertEqual(1, committed_count)
        snapshot = tracker.get_n_scan_commitment_snapshot()
        self.assertIs(shared, snapshot.latest_committed_ancestor_by_track_id[track_id])
        self.assertIs(shared, snapshot.committed_ancestor_by_track_id[track_id])
        self.assertEqual(1, snapshot.committed_boundary_by_track_id[track_id])

    def test_no_commit_when_surviving_globals_disagree_on_boundary_ancestor(
        self,
    ) -> None:
        tracker = self._build_tracker(ns_scan_window=1)
        track_id = 11
        root = self._make_node(tracker, track_id=track_id, scan_index=0, parent=None)
        left = self._make_node(tracker, track_id=track_id, scan_index=1, parent=root)
        right = self._make_node(tracker, track_id=track_id, scan_index=1, parent=root)
        leaf_left = self._make_node(
            tracker, track_id=track_id, scan_index=2, parent=left
        )
        leaf_right = self._make_node(
            tracker, track_id=track_id, scan_index=2, parent=right
        )

        _, in_scope, committed_count = tracker._update_n_scan_commitment(
            scan_index=2,
            post_beam_globals=[
                GlobalHypothesis({track_id: leaf_left}, log_weight=0.0),
                GlobalHypothesis({track_id: leaf_right}, log_weight=-1.0),
            ],
        )

        self.assertEqual(1, in_scope)
        self.assertEqual(0, committed_count)
        snapshot = tracker.get_n_scan_commitment_snapshot()
        self.assertNotIn(track_id, snapshot.latest_committed_ancestor_by_track_id)
        self.assertNotIn(track_id, snapshot.committed_boundary_by_track_id)

    def test_track_absence_in_other_globals_is_not_disagreement(self) -> None:
        tracker = self._build_tracker(ns_scan_window=1)
        track_id = 23
        other_track_id = 99
        root = self._make_node(tracker, track_id=track_id, scan_index=0, parent=None)
        shared = self._make_node(tracker, track_id=track_id, scan_index=1, parent=root)
        leaf = self._make_node(tracker, track_id=track_id, scan_index=2, parent=shared)

        other_root = self._make_node(
            tracker, track_id=other_track_id, scan_index=0, parent=None
        )
        other_mid = self._make_node(
            tracker, track_id=other_track_id, scan_index=1, parent=other_root
        )
        other_leaf = self._make_node(
            tracker, track_id=other_track_id, scan_index=2, parent=other_mid
        )

        _, _, committed_count = tracker._update_n_scan_commitment(
            scan_index=2,
            post_beam_globals=[
                GlobalHypothesis({track_id: leaf}, log_weight=0.0),
                GlobalHypothesis({other_track_id: other_leaf}, log_weight=-1.0),
            ],
        )

        self.assertEqual(2, committed_count)
        snapshot = tracker.get_n_scan_commitment_snapshot()
        self.assertIs(shared, snapshot.latest_committed_ancestor_by_track_id[track_id])

    def test_no_commit_when_boundary_is_before_available_ancestry(self) -> None:
        tracker = self._build_tracker(ns_scan_window=3)
        track_id = 5
        root = self._make_node(tracker, track_id=track_id, scan_index=1, parent=None)

        boundary, in_scope, committed_count = tracker._update_n_scan_commitment(
            scan_index=1,
            post_beam_globals=[GlobalHypothesis({track_id: root}, log_weight=0.0)],
        )

        self.assertEqual(-2, boundary)
        self.assertEqual(0, in_scope)
        self.assertEqual(0, committed_count)

    def test_no_commit_when_track_born_after_boundary(self) -> None:
        tracker = self._build_tracker(ns_scan_window=2)
        track_id = 41
        root = self._make_node(tracker, track_id=track_id, scan_index=2, parent=None)
        leaf = self._make_node(tracker, track_id=track_id, scan_index=3, parent=root)

        boundary, in_scope, committed_count = tracker._update_n_scan_commitment(
            scan_index=3,
            post_beam_globals=[GlobalHypothesis({track_id: leaf}, log_weight=0.0)],
        )

        self.assertEqual(1, boundary)
        self.assertEqual(1, in_scope)
        self.assertEqual(0, committed_count)
        snapshot = tracker.get_n_scan_commitment_snapshot()
        self.assertNotIn(track_id, snapshot.latest_committed_ancestor_by_track_id)

    def test_cleanup_reclaims_nodes_not_reachable_from_active_or_committed_refs(
        self,
    ) -> None:
        tracker = self._build_tracker(ns_scan_window=2)

        active_track_id = 7
        active_root = self._make_node(
            tracker, track_id=active_track_id, scan_index=0, parent=None
        )
        active_mid = self._make_node(
            tracker, track_id=active_track_id, scan_index=1, parent=active_root
        )
        active_leaf = self._make_node(
            tracker, track_id=active_track_id, scan_index=2, parent=active_mid
        )

        orphan_track_id = 99
        orphan_root = self._make_node(
            tracker, track_id=orphan_track_id, scan_index=0, parent=None
        )
        orphan_leaf = self._make_node(
            tracker, track_id=orphan_track_id, scan_index=1, parent=orphan_root
        )

        tracker.global_hypotheses = [
            GlobalHypothesis({active_track_id: active_leaf}, log_weight=0.0)
        ]
        tracker._cleanup_committed_ancestry()

        retained_ids = set(tracker._nodes_by_id.keys())
        self.assertIn(active_root.node_id, retained_ids)
        self.assertIn(active_mid.node_id, retained_ids)
        self.assertIn(active_leaf.node_id, retained_ids)
        self.assertNotIn(orphan_root.node_id, retained_ids)
        self.assertNotIn(orphan_leaf.node_id, retained_ids)

    def test_cleanup_retains_nodes_referenced_by_commitment_bookkeeping(self) -> None:
        tracker = self._build_tracker(ns_scan_window=2)

        active_track_id = 3
        active_root = self._make_node(
            tracker, track_id=active_track_id, scan_index=0, parent=None
        )
        active_leaf = self._make_node(
            tracker, track_id=active_track_id, scan_index=1, parent=active_root
        )
        tracker.global_hypotheses = [
            GlobalHypothesis({active_track_id: active_leaf}, log_weight=0.0)
        ]

        committed_track_id = 44
        committed_root = self._make_node(
            tracker, track_id=committed_track_id, scan_index=0, parent=None
        )
        committed_node = self._make_node(
            tracker, track_id=committed_track_id, scan_index=1, parent=committed_root
        )
        tracker._committed_boundary_by_track_id[committed_track_id] = 1
        tracker._committed_ancestor_by_track_id[committed_track_id] = committed_node

        tracker._cleanup_committed_ancestry()

        retained_ids = set(tracker._nodes_by_id.keys())
        self.assertIn(committed_node.node_id, retained_ids)
        self.assertIn(committed_root.node_id, retained_ids)


if __name__ == "__main__":
    unittest.main()
