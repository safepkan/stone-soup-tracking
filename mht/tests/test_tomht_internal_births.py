from __future__ import annotations

import datetime
import unittest
from typing import cast

from stonesoup.initiator.simple import SimpleMeasurementInitiator

from mht.tests.tomht_tracker_test_support import (
    _build_tracker,
    _CaptureInitiator,
    _detection,
    _initiator_birth_root_for_metadata,
    _logit,
    _ScriptedHypothesiser,
    _ScriptedUpdater,
    _track_start,
)
from mht.tomht_tracker import TOMHTParams


class TOMHTInternalBirthsIntegrationTest(unittest.TestCase):
    def test_internal_births_use_step2_residual_detections(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        capture_initiator = _CaptureInitiator([_track_start(99.0, t1)])
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            initiator=cast(SimpleMeasurementInitiator, capture_initiator),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1, track_id=0, options=[(0, 4.0), (None, 0.0)]
        )

        det0 = _detection(1.0, 1.0, t1)
        det1 = _detection(2.0, 2.0, t1)
        tracker.update_tracker(t1, [det0, det1])

        self.assertEqual([det1], capture_initiator.last_received)
        self.assertIn(1, tracker.track_trees_by_track_id)
        self.assertEqual([], tracker.get_unused_detections())

    def test_no_initiator_leaves_residuals_available(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        detection = _detection(1.0, 1.0, timestamp)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
        )

        tracker.update_tracker(timestamp, [detection])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual([detection], tracker.get_unused_detections())

    def test_initiator_birth_score_uses_initiator_start_prior(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        capture_initiator = _CaptureInitiator([_track_start(99.0, timestamp)])
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            initiator=cast(SimpleMeasurementInitiator, capture_initiator),
            params=TOMHTParams(
                initiator_start_initial_existence_probability=0.7,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        _, output_tracks = tracker.update_tracker(
            timestamp,
            [_detection(1.0, 1.0, timestamp)],
        )

        tree = next(iter(tracker.track_trees_by_track_id.values()))
        root = tracker.nodes_by_id[tree.root_node_id]
        expected_log_delta = _logit(0.7)
        self.assertEqual("internal_birth", root.root_source)
        self.assertEqual("tentative", tree.lifecycle_state)
        self.assertEqual("unpublished", tree.publication_state)
        self.assertAlmostEqual(expected_log_delta, root.log_delta)
        self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)
        self.assertEqual(set(), output_tracks)

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )

    def test_initiator_birth_metadata_existence_probability_overrides_default(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        birth = _track_start(99.0, timestamp)
        birth.metadata["existence_probability"] = 0.6
        capture_initiator = _CaptureInitiator([birth])
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            initiator=cast(SimpleMeasurementInitiator, capture_initiator),
            params=TOMHTParams(
                initiator_start_initial_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [_detection(1.0, 1.0, timestamp)])

        tree = next(iter(tracker.track_trees_by_track_id.values()))
        root = tracker.nodes_by_id[tree.root_node_id]
        expected_log_delta = _logit(0.6)
        self.assertAlmostEqual(expected_log_delta, root.log_delta)
        self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)

    def test_initiator_birth_metadata_existence_log_odds_overrides_default(
        self,
    ) -> None:
        for value in (-2.5, 1000.0):
            with self.subTest(value=value):
                root = _initiator_birth_root_for_metadata(
                    {"existence_log_odds": value},
                )

                self.assertAlmostEqual(value, root.log_delta)
                self.assertAlmostEqual(value, root.accumulated_log_score)

    def test_initiator_birth_log_odds_metadata_precedes_probability(self) -> None:
        root = _initiator_birth_root_for_metadata(
            {
                "existence_log_odds": 1.25,
                "existence_probability": 0.6,
            },
        )

        self.assertAlmostEqual(1.25, root.log_delta)
        self.assertAlmostEqual(1.25, root.accumulated_log_score)

    def test_initiator_birth_invalid_log_odds_metadata_falls_back_to_probability(
        self,
    ) -> None:
        invalid_values: list[object] = [
            "not-a-number",
            float("nan"),
            float("inf"),
            float("-inf"),
            None,
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                root = _initiator_birth_root_for_metadata(
                    {
                        "existence_log_odds": value,
                        "existence_probability": 0.6,
                    },
                )

                expected_log_delta = _logit(0.6)
                self.assertAlmostEqual(expected_log_delta, root.log_delta)
                self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)

    def test_initiator_birth_invalid_existence_metadata_falls_back(self) -> None:
        invalid_metadata: list[dict[str, object]] = [
            {
                "existence_log_odds": "not-a-number",
                "existence_probability": "also-not-a-number",
            },
            {
                "existence_log_odds": float("nan"),
                "existence_probability": 0.0,
            },
            {
                "existence_log_odds": float("inf"),
                "existence_probability": 1.0,
            },
        ]
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                root = _initiator_birth_root_for_metadata(metadata)

                expected_log_delta = _logit(0.8)
                self.assertAlmostEqual(expected_log_delta, root.log_delta)
                self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)

    def test_initiator_birth_invalid_metadata_existence_probability_falls_back(
        self,
    ) -> None:
        invalid_values = ["not-a-number", float("nan"), float("inf"), -0.1, 1.1]
        for value in invalid_values:
            with self.subTest(value=value):
                timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
                birth = _track_start(99.0, timestamp)
                birth.metadata["existence_probability"] = value
                capture_initiator = _CaptureInitiator([birth])
                tracker = _build_tracker(
                    hypothesiser=_ScriptedHypothesiser(),
                    updater=_ScriptedUpdater(),
                    initiator=cast(SimpleMeasurementInitiator, capture_initiator),
                    params=TOMHTParams(
                        initiator_start_initial_existence_probability=0.8,
                        debug_display_scan_stats=False,
                        debug_display_hypotheses=False,
                        debug_display_births=False,
                        collect_stats=False,
                    ),
                )

                tracker.update_tracker(timestamp, [_detection(1.0, 1.0, timestamp)])

                tree = next(iter(tracker.track_trees_by_track_id.values()))
                root = tracker.nodes_by_id[tree.root_node_id]
                expected_log_delta = _logit(0.8)
                self.assertAlmostEqual(expected_log_delta, root.log_delta)
                self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)


if __name__ == "__main__":
    unittest.main()
