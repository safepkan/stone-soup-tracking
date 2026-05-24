from __future__ import annotations

import contextlib
from dataclasses import fields
import datetime
import io
import unittest

from mht.tests.tomht_tracker_test_support import (
    _build_tracker,
    _detection,
    _ScriptedHypothesiser,
    _ScriptedUpdater,
    _track_start,
)
from mht.tomht_stats import RebuildStats, ScanStats
from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class TOMHTStatsIntegrationTest(unittest.TestCase):
    def test_historical_relaxation_runtime_stats_are_absent(self) -> None:
        rebuild_fields = RebuildStats.__dataclass_fields__
        scan_fields = ScanStats.__dataclass_fields__

        for field_name in (
            "historical_relaxation_attempts",
            "historical_relaxation_successes",
            "historical_relaxed_keys_total",
        ):
            with self.subTest(field_name=field_name):
                self.assertNotIn(field_name, rebuild_fields)
                self.assertNotIn(field_name, scan_fields)

    def test_expansion_frontier_stats_are_populated_on_simple_scan(self) -> None:
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

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (1, 1.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
        )

        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        frontier = tracker.last_scan_stats.expansion_frontier
        for stat_field in fields(frontier):
            self.assertGreaterEqual(getattr(frontier, stat_field.name), 0)

        self.assertEqual(1, frontier.leaves_before_expansion)
        self.assertEqual(3, frontier.leaves_after_expansion)
        self.assertEqual(3, frontier.leaves_after_empty_tree_removal)
        self.assertEqual(3, frontier.leaves_after_births)
        self.assertEqual(3, frontier.leaves_after_post_solve_supported_pruning)
        self.assertEqual(3, frontier.leaves_after_n_scan_pruning)
        self.assertEqual(3, frontier.leaves_after_lifecycle)
        self.assertEqual(1, frontier.expanded_leaf_count)
        self.assertEqual(0, frontier.expanded_leaves_tentative)
        self.assertEqual(1, frontier.expanded_leaves_confirmed)
        self.assertEqual(3, frontier.local_child_candidates_total)
        self.assertEqual(3, frontier.local_children_created_total)
        self.assertEqual(3, frontier.local_children_retained_total)
        self.assertEqual(1, frontier.local_miss_children_created)
        self.assertEqual(2, frontier.local_detection_children_created)
        self.assertEqual(1, frontier.map_selected_leaf_count)
        self.assertEqual(3, frontier.retained_topk_supported_leaf_count)
        self.assertEqual(0, frontier.unsupported_leaf_count_pruned)

    def test_expansion_frontier_debug_flag_does_not_change_behavior(self) -> None:
        def run_case(
            *,
            debug_display_expansion_frontier: bool,
        ) -> tuple[TOMHTTracker, str]:
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
                    debug_display_expansion_frontier=(debug_display_expansion_frontier),
                    collect_stats=False,
                ),
            )
            log_stream = io.StringIO()
            with contextlib.redirect_stdout(log_stream):
                tracker.update_tracker(t0, [])
            tracker.add_external_starts(t0, [_track_start(0.0, t0)])
            hypothesiser.set_options(
                timestamp=t1,
                track_id=0,
                options=[(0, 2.0), (1, 1.0), (None, 0.0)],
            )
            with contextlib.redirect_stdout(log_stream):
                tracker.update_tracker(
                    t1,
                    [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
                )
            return tracker, log_stream.getvalue()

        plain_tracker, plain_log = run_case(debug_display_expansion_frontier=False)
        debug_tracker, debug_log = run_case(debug_display_expansion_frontier=True)

        plain_map = plain_tracker.get_map_hypothesis_snapshot()
        debug_map = debug_tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(plain_map)
        self.assertIsNotNone(debug_map)
        assert plain_map is not None
        assert debug_map is not None
        plain_keys = {
            track_id: leaf.used_det_key
            for track_id, leaf in plain_map.leaf_nodes_by_track_id.items()
        }
        debug_keys = {
            track_id: leaf.used_det_key
            for track_id, leaf in debug_map.leaf_nodes_by_track_id.items()
        }
        self.assertEqual(plain_keys, debug_keys)
        self.assertIsNotNone(plain_tracker.last_scan_stats)
        self.assertIsNotNone(debug_tracker.last_scan_stats)
        assert plain_tracker.last_scan_stats is not None
        assert debug_tracker.last_scan_stats is not None
        self.assertEqual(
            plain_tracker.last_scan_stats.expansion_frontier,
            debug_tracker.last_scan_stats.expansion_frontier,
        )
        del plain_log

        debug_lines = [
            line
            for line in debug_log.splitlines()
            if line.startswith("EXPANSION_FRONTIER ")
        ]
        self.assertEqual(2, len(debug_lines))
        self.assertEqual(
            "EXPANSION_FRONTIER scan=1 t=2026-03-28 10:00:01 "
            "leaves_before=1 leaves_after_expansion=3 leaves_after_empty=3 "
            "leaves_after_births=3 leaves_after_supported_prune=3 "
            "leaves_after_nscan=3 leaves_after_lifecycle=3 trees_before=1 "
            "trees_after_lifecycle=1 expanded=1 expanded_tentative=0 "
            "expanded_confirmed=1 child_candidates=3 children_created=3 "
            "children_retained=3 miss_children=1 detection_children=2 "
            "track_reconstruct_calls=1 default_state_fast_path_calls=0 "
            "topk_supported=3 map_selected=1 unsupported_pruned=0",
            debug_lines[-1],
        )

    def test_expansion_frontier_summary_output_is_deterministic(self) -> None:
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
                debug_display_expansion_frontier=True,
                collect_stats=True,
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (1, 1.0), (None, 0.0)],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tracker.update_tracker(
                t1,
                [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
            )

        first = io.StringIO()
        second = io.StringIO()
        with contextlib.redirect_stdout(first):
            tracker.print_summary_stats()
        with contextlib.redirect_stdout(second):
            tracker.print_summary_stats()

        self.assertEqual(first.getvalue(), second.getvalue())
        self.assertIn("SUMMARY expansion_frontier ", first.getvalue())


if __name__ == "__main__":
    unittest.main()
