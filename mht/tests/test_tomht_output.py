from __future__ import annotations

import contextlib
import datetime
import io
import unittest

import numpy as np

from mht.tests.tomht_tracker_test_support import (
    _build_tracker,
    _detection,
    _logit,
    _ScriptedHypothesiser,
    _ScriptedUpdater,
    _track_start,
)
from mht.tomht_tracker import TOMHTParams
from mht.tomht_tree_utils import live_conflict_keys_for_leaf


class TOMHTOutputIntegrationTest(unittest.TestCase):
    def test_default_publication_keeps_tentative_map_track_internal(self) -> None:
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

        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t1, [])

        self.assertIn(0, tracker.track_trees_by_track_id)
        self.assertEqual(
            "tentative",
            tracker.track_trees_by_track_id[0].lifecycle_state,
        )
        self.assertEqual(
            "unpublished",
            tracker.track_trees_by_track_id[0].publication_state,
        )
        self.assertIsNone(tracker.track_trees_by_track_id[0].public_track_id)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        self.assertEqual(1, tracker.last_scan_stats.active_tentative_trees)
        self.assertEqual(0, tracker.last_scan_stats.active_confirmed_trees)
        self.assertEqual(1, tracker.last_scan_stats.map_tracks)
        self.assertEqual(0, tracker.last_scan_stats.map_published_tracks)
        self.assertEqual(1, tracker.last_scan_stats.map_unpublished_tracks)

        self.assertEqual(set(), tracker.get_map_output_tracks())
        self.assertEqual(set(), tracker.tracks)
        self.assertEqual((), tracker.get_map_association_history().histories)

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(1, len(map_snapshot.leaf_nodes_by_track_id))

        association_history = tracker.get_map_association_history(
            include_unpublished=True
        )
        self.assertEqual("map", association_history.selection)
        self.assertEqual(t1, association_history.timestamp)
        self.assertEqual(1, len(association_history.histories))
        track_history = association_history.histories[0]
        self.assertEqual(0, track_history.internal_track_id)
        self.assertIsNone(track_history.public_track_id)
        self.assertEqual("unpublished", track_history.publication_state)
        self.assertEqual(
            ["tentative", "tentative"],
            [step.association_status for step in track_history.steps],
        )
        self.assertEqual(
            [None, None],
            [step.input_detection_index for step in track_history.steps],
        )

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(0, inspection_track.id)
        self.assertEqual(0, inspection_track.metadata["track_id"])
        self.assertEqual(0, inspection_track.metadata["internal_track_id"])
        self.assertIsNone(inspection_track.metadata["public_track_id"])
        self.assertEqual("tentative", inspection_track.metadata["lifecycle_state"])
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )

    def test_default_publication_publishes_confirmed_map_output(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
                debug_display_scan_stats=True,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("tentative", tree.lifecycle_state)
        self.assertEqual("unpublished", tree.publication_state)
        self.assertEqual(set(), tracker.get_map_output_tracks())
        self.assertEqual(set(), tracker.tracks)

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(1, len(map_snapshot.leaf_nodes_by_track_id))

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        with contextlib.redirect_stdout(log_stream):
            _, output_tracks = tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("confirmed", tree.lifecycle_state)
        self.assertEqual("published", tree.publication_state)
        self.assertEqual(0, tree.public_track_id)
        self.assertIn(
            "MAP tracks=1 published=1 unpublished=0",
            log_stream.getvalue(),
        )
        self.assertEqual(1, len(output_tracks))
        self.assertEqual(1, len(tracker.tracks))
        output_track = next(iter(output_tracks))
        self.assertEqual(0, output_track.id)
        self.assertEqual(0, output_track.metadata["track_id"])
        self.assertEqual(0, output_track.metadata["internal_track_id"])
        self.assertEqual(0, output_track.metadata["public_track_id"])
        self.assertEqual("confirmed", output_track.metadata["lifecycle_state"])
        self.assertEqual("published", output_track.metadata["publication_state"])

    def test_map_association_history_reports_caller_input_detection_index(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        first_input_detection = _detection(2.0, 2.0, t1)
        second_input_detection = _detection(1.0, 1.0, t1)
        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 5.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [first_input_detection, second_input_detection])

        association_history = tracker.get_map_association_history(
            include_unpublished=True
        )
        self.assertEqual(1, len(association_history.histories))
        hit_step = association_history.histories[0].steps[-1]
        self.assertEqual("tentative", hit_step.association_status)
        self.assertEqual(1, hit_step.input_detection_index)
        self.assertEqual(0, hit_step.internal_detection_index)
        self.assertIsNotNone(hit_step.detection_key)
        assert hit_step.detection_key is not None
        self.assertEqual(1, hit_step.detection_key.scan_index)
        self.assertEqual(0, hit_step.detection_key.det_index)

    def test_default_public_track_ids_are_dense_in_publication_order(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

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
        tracker.add_external_starts(t0, [_track_start(0.0, t0), _track_start(10.0, t0)])

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual({0, 1}, {int(track.id) for track in inspection_tracks})
        self.assertEqual(
            {None},
            {track.metadata["public_track_id"] for track in inspection_tracks},
        )

        hypothesiser.set_options(
            timestamp=t1,
            track_id=1,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(10.0, 10.0, t1)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertIsNone(tracker.track_trees_by_track_id[0].public_track_id)
        self.assertEqual(
            "published", tracker.track_trees_by_track_id[1].publication_state
        )
        self.assertEqual(0, tracker.track_trees_by_track_id[1].public_track_id)

        tracker.add_external_starts(t1, [_track_start(20.0, t1)])
        self.assertIn(2, tracker.track_trees_by_track_id)
        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[2].publication_state
        )
        self.assertIsNone(tracker.track_trees_by_track_id[2].public_track_id)

        hypothesiser.set_options(
            timestamp=t2,
            track_id=2,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t2, [_detection(20.0, 20.0, t2)])

        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual({0, 1}, {int(track.id) for track in output_tracks})
        self.assertEqual(
            {1, 2},
            {int(track.metadata["internal_track_id"]) for track in output_tracks},
        )
        self.assertEqual(
            {1, 2},
            {int(track.metadata["track_id"]) for track in output_tracks},
        )
        self.assertEqual(
            {0, 1},
            {int(track.metadata["public_track_id"]) for track in output_tracks},
        )
        for track in output_tracks:
            self.assertEqual(track.id, track.metadata["public_track_id"])

        tracker.update_tracker(t3, [])
        self.assertEqual(0, tracker.track_trees_by_track_id[1].public_track_id)
        self.assertEqual(1, tracker.track_trees_by_track_id[2].public_track_id)

    def test_publish_min_hits_gates_initial_publication(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                publish_min_hits=2,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertEqual(set(), tracker.get_map_output_tracks())

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 1.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(2, output_track.metadata["hits"])

    def test_publish_min_age_gates_initial_publication(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                publish_min_age=2,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertEqual(set(), tracker.get_map_output_tracks())

        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t1, [])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(2, output_track.metadata["age"])

    def test_publish_min_existence_probability_gates_initial_publication(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                publish_min_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertEqual(set(), tracker.get_map_output_tracks())

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertGreaterEqual(
            float(output_track.metadata["existence_probability"]),
            0.8,
        )

    def test_publication_is_sticky_if_score_later_drops(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                publish_min_existence_probability=0.8,
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

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        for leaf_id in tree.active_leaf_node_ids:
            tracker.nodes_by_id[leaf_id].accumulated_log_score = _logit(0.1)

        tracker._apply_output_publication(tracker._last_map_global)

        self.assertEqual("published", tree.publication_state)
        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual(1, len(output_tracks))

    def test_map_output_reconstruction_keeps_committed_prefix_across_pruning(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

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
            timestamp=t1, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        scan1_map = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(scan1_map)
        assert scan1_map is not None
        scan1_leaf = scan1_map.leaf_nodes_by_track_id[0]
        tree_before_detection_commit = tracker.track_trees_by_track_id[0]
        self.assertEqual(
            frozenset(),
            tree_before_detection_commit.committed_detection_keys,
        )
        self.assertEqual(
            scan1_leaf.detection_history_keys,
            live_conflict_keys_for_leaf(
                leaf=scan1_leaf,
                tree=tree_before_detection_commit,
            ),
        )

        hypothesiser.set_options(
            timestamp=t2, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t2, [_detection(2.0, 2.0, t2)])

        tree_after_first_prune = tracker.track_trees_by_track_id[0]
        self.assertEqual(1, len(tree_after_first_prune.committed_states))
        self.assertEqual(
            frozenset({(1, 0)}),
            tree_after_first_prune.committed_detection_keys,
        )

        hypothesiser.set_options(
            timestamp=t3, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t3, [_detection(3.0, 3.0, t3)])

        tree_after_second_prune = tracker.track_trees_by_track_id[0]
        self.assertEqual(2, len(tree_after_second_prune.committed_states))
        self.assertEqual(
            frozenset({(2, 0)}),
            tree_after_second_prune.committed_detection_keys,
        )
        scan3_map = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(scan3_map)
        assert scan3_map is not None
        scan3_leaf = scan3_map.leaf_nodes_by_track_id[0]
        self.assertEqual(
            frozenset({(3, 0)}),
            live_conflict_keys_for_leaf(
                leaf=scan3_leaf,
                tree=tree_after_second_prune,
            ),
        )
        self.assertEqual(
            frozenset({(2, 0), (3, 0)}),
            scan3_leaf.detection_history_keys,
        )
        association_history = tracker.get_map_association_history()
        self.assertEqual(1, len(association_history.histories))
        track_history = association_history.histories[0]
        self.assertEqual(0, track_history.internal_track_id)
        self.assertEqual(2, track_history.committed_boundary_scan_index)
        self.assertEqual(
            ["committed", "tentative"],
            [step.association_status for step in track_history.steps],
        )
        self.assertEqual(
            [2, 3],
            [step.scan_index for step in track_history.steps],
        )
        self.assertEqual(
            [0, 0],
            [step.input_detection_index for step in track_history.steps],
        )
        self.assertEqual(
            [0, 0],
            [step.internal_detection_index for step in track_history.steps],
        )

        committed_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in tree_after_second_prune.committed_states
        ]
        self.assertEqual([0.0, 1.0], committed_x)

        output_track = next(iter(tracker.get_map_output_tracks()))
        output_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in output_track.states
        ]
        self.assertEqual([0.0, 1.0, 2.0, 3.0], output_x)
        self.assertEqual(1, output_x.count(1.0))
        self.assertEqual(4, len(output_track))

    def test_detection_conflict_history_is_bounded_by_n_scan_horizon(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)

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
        for scan_index in range(1, 6):
            timestamp = t0 + datetime.timedelta(seconds=scan_index)
            hypothesiser.set_options(
                timestamp=timestamp,
                track_id=0,
                options=[(0, 5.0), (None, 0.0)],
            )
            tracker.update_tracker(
                timestamp,
                [_detection(float(scan_index), float(scan_index), timestamp)],
            )

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual(
            frozenset({(4, 0)}),
            tree.committed_detection_keys,
        )

        for node in tracker.nodes_by_id.values():
            with self.subTest(node_id=node.node_id):
                min_scan_index = int(node.scan_index) - 1
                self.assertTrue(
                    all(
                        int(key.scan_index) >= min_scan_index
                        for key in node.detection_history_keys
                    )
                )
                self.assertLessEqual(
                    len({key.scan_index for key in node.detection_history_keys}),
                    2,
                )

    def test_n_scan_promotion_trims_retained_intermediate_node_histories(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                ns_scan_window=3,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        for scan_index in range(1, 8):
            timestamp = t0 + datetime.timedelta(seconds=scan_index)
            hypothesiser.set_options(
                timestamp=timestamp,
                track_id=0,
                options=[(0, 5.0), (None, 0.0)],
            )
            tracker.update_tracker(
                timestamp,
                [_detection(float(scan_index), float(scan_index), timestamp)],
            )

        nscan_snapshot = tracker.get_n_scan_commitment_snapshot()
        self.assertEqual(4, nscan_snapshot.boundary_scan_index)
        self.assertIsNotNone(nscan_snapshot.boundary_scan_index)
        assert nscan_snapshot.boundary_scan_index is not None
        min_scan_index = int(nscan_snapshot.boundary_scan_index)

        tree = tracker.track_trees_by_track_id[0]
        retained_node_ids = tracker._tree_store.reachable_node_ids_from_seeds(
            tracker.nodes_by_id[leaf_id] for leaf_id in tree.active_leaf_node_ids
        )
        self.assertGreater(len(retained_node_ids), len(tree.active_leaf_node_ids))
        for node_id in retained_node_ids:
            node = tracker.nodes_by_id[node_id]
            with self.subTest(node_id=node_id, scan_index=node.scan_index):
                self.assertTrue(
                    all(
                        int(key.scan_index) >= min_scan_index
                        for key in node.detection_history_keys
                    )
                )

    def test_committed_output_history_can_be_limited_by_update_count(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                ns_scan_window=1,
                max_stored_history_updates=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        for scan_index in range(1, 4):
            timestamp = t0 + datetime.timedelta(seconds=scan_index)
            hypothesiser.set_options(
                timestamp=timestamp,
                track_id=0,
                options=[(0, 5.0), (None, 0.0)],
            )
            tracker.update_tracker(
                timestamp,
                [_detection(float(scan_index), float(scan_index), timestamp)],
            )

        tree = tracker.track_trees_by_track_id[0]
        committed_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in tree.committed_states
        ]
        self.assertEqual([1.0], committed_x)

        output_track = next(iter(tracker.get_map_output_tracks()))
        output_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in output_track.states
        ]
        self.assertEqual([1.0, 2.0, 3.0], output_x)

    def test_committed_output_history_can_be_limited_by_age(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                ns_scan_window=1,
                max_stored_history_age_s=1.0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        for scan_index in range(1, 5):
            timestamp = t0 + datetime.timedelta(seconds=scan_index)
            hypothesiser.set_options(
                timestamp=timestamp,
                track_id=0,
                options=[(0, 5.0), (None, 0.0)],
            )
            tracker.update_tracker(
                timestamp,
                [_detection(float(scan_index), float(scan_index), timestamp)],
            )

        tree = tracker.track_trees_by_track_id[0]
        committed_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in tree.committed_states
        ]
        self.assertEqual([2.0], committed_x)

        output_track = next(iter(tracker.get_map_output_tracks()))
        output_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in output_track.states
        ]
        self.assertEqual([2.0, 3.0, 4.0], output_x)


if __name__ == "__main__":
    unittest.main()
