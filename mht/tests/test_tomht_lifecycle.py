from __future__ import annotations

import contextlib
import datetime
import io
import unittest
from unittest import mock

from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeDeleter, UpdateTimeStepsDeleter
from stonesoup.types.track import Track

from mht.tests.tomht_tracker_test_support import (
    _build_tracker,
    _detection,
    _logit,
    _MetadataMissCountDeleter,
    _RecordingMetadataMissCountDeleter,
    _replace_active_leaves_with_scores,
    _run_post_n_scan_lifecycle,
    _ScriptedHypothesiser,
    _ScriptedPredictingHypothesiser,
    _ScriptedUpdater,
    _ScriptedUpdaterWithUpdateStates,
    _set_track_active_leaf_scores,
    _state,
    _tracker_with_two_miss_candidate_leaves,
    _track_start,
)
from mht.tomht_lifecycle import (
    effective_track_miss_threshold,
    FastMissCountDeleter,
    LifecycleDeleterStats,
    resolve_deleter_with_metadata,
    TOMHTMissCountDeleter,
)
from mht.tomht_model import GlobalHypothesis, TrackHypothesisNode, TrackTree
from mht.tomht_output import reconstruct_track_from_committed_prefix_and_leaf_node
from mht.tomht_tracker import TOMHTParams


class TOMHTLifecycleIntegrationTest(unittest.TestCase):
    def test_score_deletion_removes_tentative_tree_below_threshold(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        self.assertEqual(
            "tentative", tracker.track_trees_by_track_id[0].lifecycle_state
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual({}, filtered.leaf_nodes_by_track_id)
        self.assertEqual(set(), tracker.get_map_output_tracks(include_unpublished=True))

    def test_score_deletion_removes_confirmed_tree_below_threshold(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        tracker.track_trees_by_track_id[0].lifecycle_state = "confirmed"
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.get_map_output_tracks(include_unpublished=True))

    def test_score_deletion_keeps_tree_above_threshold(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.02)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertIn(0, tracker.track_trees_by_track_id)
        self.assertEqual({0}, set(filtered.leaf_nodes_by_track_id))

    def test_score_deletion_uses_max_active_leaf_score(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        low_leaf, high_leaf = _replace_active_leaves_with_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005), _logit(0.02)],
            timestamp=timestamp,
        )
        map_global = GlobalHypothesis(
            leaf_nodes_by_track_id={0: low_leaf},
            log_weight=float(low_leaf.accumulated_log_score),
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(
                tracker,
                timestamp=timestamp,
                map_global=map_global,
            )

        self.assertIn(0, tracker.track_trees_by_track_id)
        self.assertEqual(low_leaf.node_id, filtered.leaf_nodes_by_track_id[0].node_id)
        tree_score = tracker._tree_store.active_tree_max_accumulated_log_score(
            tracker.track_trees_by_track_id[0]
        )
        self.assertIsNotNone(tree_score)
        assert tree_score is not None
        self.assertAlmostEqual(
            float(high_leaf.accumulated_log_score),
            float(tree_score),
        )

    def test_score_deletion_filters_current_map_global_to_live_trees(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(
            timestamp,
            [_track_start(0.0, timestamp), _track_start(10.0, timestamp)],
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=1,
            scores=[_logit(0.02)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({1}, set(tracker.track_trees_by_track_id))
        self.assertEqual({1}, set(filtered.leaf_nodes_by_track_id))
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual({1}, set(map_snapshot.leaf_nodes_by_track_id))

    def test_score_deletion_composes_with_miss_deletion_diagnostics(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                max_missed=0,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(
            timestamp,
            [_track_start(0.0, timestamp), _track_start(10.0, timestamp)],
        )
        score_leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )[0]
        miss_leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=1,
            scores=[_logit(0.02)],
        )[0]
        self.assertEqual(0, score_leaf.missed_count)
        miss_leaf.missed_count = 1

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual({}, filtered.leaf_nodes_by_track_id)
        log_output = log_stream.getvalue()
        self.assertIn("TRACK_LIFECYCLE", log_output)
        self.assertIn("terminated=[0, 1]", log_output)
        self.assertIn("score:[0]", log_output)
        self.assertIn("miss:[1]", log_output)

    def test_score_deletion_runs_with_configured_deleter(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            deleter=_MetadataMissCountDeleter(threshold=99),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )[0]
        leaf.missed_count = 0

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertIn("deleter=_MetadataMissCountDeleter", log_stream.getvalue())
        self.assertIn("score:[0]", log_stream.getvalue())

    def test_default_deleter_uses_effective_miss_threshold(self) -> None:
        params = TOMHTParams(max_missed=1, ns_scan_window=3)

        self.assertEqual(4, effective_track_miss_threshold(params=params))
        resolved = resolve_deleter_with_metadata(params=params, deleter=None)

        self.assertIsInstance(resolved.deleter, TOMHTMissCountDeleter)
        self.assertIsInstance(resolved.fast_deleter, FastMissCountDeleter)
        self.assertEqual("miss", resolved.reason)
        self.assertEqual(4, resolved.miss_threshold)
        self.assertTrue(
            resolved.deleter.check_for_deletion(
                Track(init_metadata={"missed_count": 4})
            )
        )
        self.assertFalse(
            resolved.deleter.check_for_deletion(
                Track(init_metadata={"missed_count": 3})
            )
        )
        leaf = TrackHypothesisNode(
            node_id=0,
            track_id=0,
            parent=None,
            scan_index=0,
            timestamp=datetime.datetime(2026, 3, 28, 10, 0, 0),
            state=_state(0.0, datetime.datetime(2026, 3, 28, 10, 0, 0)),
            state_kind="root",
            used_det_key=None,
            assoc_label=-2,
            log_delta=0.0,
            accumulated_log_score=0.0,
            detection_history_keys=frozenset(),
            age=1,
            hits=0,
            missed_count=4,
            last_det_key=None,
            last_det_hit=False,
            root_source="external",
            birth_scan_index=0,
        )
        tree = TrackTree(
            track_id=0,
            root_node_id=0,
            active_leaf_node_ids={0},
            root_source="external",
            lifecycle_state="confirmed",
            publication_state="published",
        )
        assert resolved.fast_deleter is not None
        self.assertTrue(
            resolved.fast_deleter.check_for_deletion(
                leaf_node=leaf,
                track_tree=tree,
            )
        )
        leaf.missed_count = 3
        self.assertFalse(
            resolved.fast_deleter.check_for_deletion(
                leaf_node=leaf,
                track_tree=tree,
            )
        )

    def test_default_miss_deleter_reports_miss_reason(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                enable_default_miss_deleter_fast_path=True,
                max_missed=1,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[10.0],
        )[0]
        leaf.missed_count = 1

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        log_output = log_stream.getvalue()
        self.assertIn("TRACK_LIFECYCLE", log_output)
        self.assertIn("miss_threshold=1", log_output)
        self.assertIn("reasons=miss:[0]", log_output)
        self.assertNotIn("deleter=", log_output)

    def test_default_miss_deleter_fast_path_deletes_without_reconstruction(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                enable_default_miss_deleter_fast_path=True,
                max_missed=1,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[10.0],
        )[0]
        leaf.missed_count = 1

        lifecycle_stats = LifecycleDeleterStats()
        with (
            mock.patch(
                "mht.tomht_lifecycle."
                "reconstruct_track_from_committed_prefix_and_leaf_node",
                side_effect=AssertionError(
                    "default miss deleter should not reconstruct Track"
                ),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            _run_post_n_scan_lifecycle(
                tracker,
                timestamp=timestamp,
                lifecycle_deleter_stats=lifecycle_stats,
            )

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(0, lifecycle_stats.track_reconstruct_calls)
        self.assertEqual(0, lifecycle_stats.track_reconstruct_wall_ns)
        self.assertEqual(1, lifecycle_stats.default_miss_fast_path_calls)
        self.assertGreaterEqual(lifecycle_stats.check_wall_ns, 0)

    def test_default_miss_deleter_flag_false_uses_reconstructed_track(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                enable_default_miss_deleter_fast_path=False,
                max_missed=1,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[10.0],
        )[0]
        leaf.missed_count = 1

        lifecycle_stats = LifecycleDeleterStats()
        with (
            mock.patch(
                "mht.tomht_lifecycle."
                "reconstruct_track_from_committed_prefix_and_leaf_node",
                wraps=reconstruct_track_from_committed_prefix_and_leaf_node,
            ) as reconstruct_mock,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            _run_post_n_scan_lifecycle(
                tracker,
                timestamp=timestamp,
                lifecycle_deleter_stats=lifecycle_stats,
            )

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(1, reconstruct_mock.call_count)
        self.assertEqual(1, lifecycle_stats.track_reconstruct_calls)
        self.assertGreaterEqual(lifecycle_stats.track_reconstruct_wall_ns, 0)
        self.assertEqual(0, lifecycle_stats.default_miss_fast_path_calls)
        self.assertGreaterEqual(lifecycle_stats.check_wall_ns, 0)

    def test_score_deletion_still_runs_with_default_miss_fast_path(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                enable_default_miss_deleter_fast_path=True,
                track_deletion_existence_probability=0.01,
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )[0]
        leaf.missed_count = 0

        lifecycle_stats = LifecycleDeleterStats()
        log_stream = io.StringIO()
        with (
            mock.patch(
                "mht.tomht_lifecycle."
                "reconstruct_track_from_committed_prefix_and_leaf_node",
                side_effect=AssertionError(
                    "default miss deleter should not reconstruct Track"
                ),
            ),
            contextlib.redirect_stdout(log_stream),
        ):
            _run_post_n_scan_lifecycle(
                tracker,
                timestamp=timestamp,
                lifecycle_deleter_stats=lifecycle_stats,
            )

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertIn("score:[0]", log_stream.getvalue())
        self.assertEqual(0, lifecycle_stats.track_reconstruct_calls)
        self.assertEqual(1, lifecycle_stats.default_miss_fast_path_calls)

    def test_custom_deleter_reports_deleter_reason(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            deleter=_MetadataMissCountDeleter(threshold=1),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[10.0],
        )[0]
        leaf.missed_count = 1

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        log_output = log_stream.getvalue()
        self.assertIn("TRACK_LIFECYCLE", log_output)
        self.assertIn("deleter=_MetadataMissCountDeleter", log_output)
        self.assertIn("reasons=deleter:[0]", log_output)
        self.assertNotIn("miss_threshold=", log_output)

    def test_custom_deleter_gets_full_track_and_reconstruction_stats(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        for fast_path_flag in (True, False):
            with self.subTest(fast_path_flag=fast_path_flag):
                deleter = _RecordingMetadataMissCountDeleter(threshold=99)
                tracker = _build_tracker(
                    hypothesiser=_ScriptedHypothesiser(),
                    updater=_ScriptedUpdater(),
                    deleter=deleter,
                    params=TOMHTParams(
                        enable_default_miss_deleter_fast_path=fast_path_flag,
                        max_missed=1,
                        ns_scan_window=0,
                        debug_display_scan_stats=False,
                        debug_display_hypotheses=False,
                        debug_display_births=False,
                        collect_stats=False,
                    ),
                )

                tracker.update_tracker(timestamp, [])
                tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
                tree = tracker.track_trees_by_track_id[0]
                tree.committed_states = [
                    _state(-1.0, timestamp - datetime.timedelta(seconds=1))
                ]
                leaf = _set_track_active_leaf_scores(
                    tracker,
                    track_id=0,
                    scores=[10.0],
                )[0]
                leaf.missed_count = 1

                lifecycle_stats = LifecycleDeleterStats()
                with (
                    mock.patch(
                        "mht.tomht_lifecycle."
                        "reconstruct_track_from_committed_prefix_and_leaf_node",
                        wraps=reconstruct_track_from_committed_prefix_and_leaf_node,
                    ) as reconstruct_mock,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    _run_post_n_scan_lifecycle(
                        tracker,
                        timestamp=timestamp,
                        lifecycle_deleter_stats=lifecycle_stats,
                    )

                self.assertIn(0, tracker.track_trees_by_track_id)
                self.assertEqual(1, reconstruct_mock.call_count)
                self.assertEqual([0], deleter.track_ids)
                self.assertEqual([1], deleter.missed_counts)
                self.assertEqual([2], deleter.track_state_counts)
                self.assertEqual([True], deleter.track_is_track)
                self.assertEqual(1, lifecycle_stats.track_reconstruct_calls)
                self.assertGreaterEqual(lifecycle_stats.track_reconstruct_wall_ns, 0)
                self.assertEqual(0, lifecycle_stats.default_miss_fast_path_calls)
                self.assertGreaterEqual(lifecycle_stats.check_wall_ns, 0)

    def test_track_miss_mode_controls_default_miss_deleter_candidates(self) -> None:
        cases = [
            ("map_leaf", True),
            ("global_k_leaves", True),
            ("all_active_leaves", False),
        ]

        for mode, expected_deleted in cases:
            with self.subTest(mode=mode):
                (
                    tracker,
                    timestamp,
                    _,
                    map_global,
                    cluster_snapshots,
                ) = _tracker_with_two_miss_candidate_leaves(mode=mode)

                with contextlib.redirect_stdout(io.StringIO()):
                    _run_post_n_scan_lifecycle(
                        tracker,
                        timestamp=timestamp,
                        map_global=map_global,
                        cluster_snapshots=cluster_snapshots,
                    )

                if expected_deleted:
                    self.assertEqual({}, tracker.track_trees_by_track_id)
                else:
                    self.assertIn(0, tracker.track_trees_by_track_id)

    def test_track_miss_mode_controls_custom_deleter_candidates(self) -> None:
        cases = [
            ("map_leaf", True, [1]),
            ("global_k_leaves", True, [1]),
            ("all_active_leaves", False, [1, 0]),
        ]

        for mode, expected_deleted, expected_misses in cases:
            with self.subTest(mode=mode):
                deleter = _RecordingMetadataMissCountDeleter(threshold=1)
                (
                    tracker,
                    timestamp,
                    _,
                    map_global,
                    cluster_snapshots,
                ) = _tracker_with_two_miss_candidate_leaves(
                    mode=mode,
                    deleter=deleter,
                )

                with contextlib.redirect_stdout(io.StringIO()):
                    _run_post_n_scan_lifecycle(
                        tracker,
                        timestamp=timestamp,
                        map_global=map_global,
                        cluster_snapshots=cluster_snapshots,
                    )

                self.assertEqual(expected_misses, deleter.missed_counts)
                self.assertEqual([0] * len(expected_misses), deleter.track_ids)
                if expected_deleted:
                    self.assertEqual({}, tracker.track_trees_by_track_id)
                else:
                    self.assertIn(0, tracker.track_trees_by_track_id)

    def test_score_deletion_does_not_reuse_public_ids(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                publish_lifecycle_states=("tentative", "confirmed"),
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        first_tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", first_tree.publication_state)
        self.assertEqual(0, first_tree.public_track_id)

        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)
        self.assertEqual({}, tracker.track_trees_by_track_id)

        tracker.add_external_starts(timestamp, [_track_start(10.0, timestamp)])

        self.assertEqual({1}, set(tracker.track_trees_by_track_id))
        second_tree = tracker.track_trees_by_track_id[1]
        self.assertEqual("published", second_tree.publication_state)
        self.assertEqual(1, second_tree.public_track_id)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(1, output_track.id)
        self.assertEqual(1, output_track.metadata["public_track_id"])

    def test_score_deletion_removes_published_and_unpublished_trees(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(
            timestamp,
            [_track_start(0.0, timestamp), _track_start(10.0, timestamp)],
        )
        published_tree = tracker.track_trees_by_track_id[0]
        tracker._ensure_public_track_id(published_tree)
        published_tree.publication_state = "published"
        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[1].publication_state
        )

        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=1,
            scores=[_logit(0.005)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual({}, filtered.leaf_nodes_by_track_id)
        self.assertEqual(set(), tracker.get_map_output_tracks(include_unpublished=True))

    def test_max_missed_drops_leaf_and_removes_tree(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                max_missed=0,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])

        tracker.update_tracker(t1, [])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_configured_deleter_replaces_default_miss_policy(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            deleter=_MetadataMissCountDeleter(threshold=99),
            params=TOMHTParams(
                max_missed=0,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])

        tracker.update_tracker(t1, [])

        self.assertIn(0, tracker.track_trees_by_track_id)
        leaf_id = next(iter(tracker.track_trees_by_track_id[0].active_leaf_node_ids))
        self.assertEqual(1, tracker.nodes_by_id[leaf_id].missed_count)

    def test_configured_deleter_can_delete_tracks(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            deleter=_MetadataMissCountDeleter(threshold=1),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])

        tracker.update_tracker(t1, [])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_builtin_covariance_based_deleter_deletes_track(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            deleter=CovarianceBasedDeleter(covar_trace_thresh=0.1),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t1, [])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_builtin_update_time_steps_deleter_behavior(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedPredictingHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdaterWithUpdateStates(),
            deleter=UpdateTimeStepsDeleter(time_steps_since_update=2),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t2, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t2, [])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t3, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t3, [])
        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_builtin_update_time_deleter_behavior(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedPredictingHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdaterWithUpdateStates(),
            deleter=UpdateTimeDeleter(time_since_update=datetime.timedelta(seconds=1)),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t2, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t2, [])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t3, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t3, [])
        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_track_confirmation_uses_max_active_leaf_score(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=True,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        self.assertEqual(
            "tentative",
            tracker.track_trees_by_track_id[0].lifecycle_state,
        )

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, -1.0), (1, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
        )

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("confirmed", tree.lifecycle_state)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        self.assertEqual(0, tracker.last_scan_stats.active_tentative_trees)
        self.assertEqual(1, tracker.last_scan_stats.active_confirmed_trees)
        self.assertEqual(1, tracker.last_scan_stats.map_tracks)
        self.assertEqual(1, tracker.last_scan_stats.map_published_tracks)
        self.assertEqual(0, tracker.last_scan_stats.map_unpublished_tracks)

        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual("confirmed", output_track.metadata["lifecycle_state"])

    def test_track_confirmation_is_sticky_if_score_later_drops(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
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
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        self.assertEqual(
            "confirmed",
            tracker.track_trees_by_track_id[0].lifecycle_state,
        )

        tree = tracker.track_trees_by_track_id[0]
        for leaf_id in tree.active_leaf_node_ids:
            tracker.nodes_by_id[leaf_id].accumulated_log_score = _logit(0.1)
        tracker._apply_score_based_track_confirmation()

        self.assertEqual("confirmed", tree.lifecycle_state)
        tree_score = tracker._tree_store.active_tree_max_accumulated_log_score(tree)
        self.assertIsNotNone(tree_score)
        assert tree_score is not None
        self.assertLess(tree_score, _logit(0.8))


if __name__ == "__main__":
    unittest.main()
