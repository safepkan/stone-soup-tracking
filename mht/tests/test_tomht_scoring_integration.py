from __future__ import annotations

import datetime
import unittest

from mht.tests.tomht_tracker_test_support import (
    _build_tracker,
    _CaptureDetectionProbabilityModel,
    _detection,
    _ScriptedHypothesiser,
    _ScriptedUpdater,
    _track_start,
)
from mht.tomht_scoring import ConstantDetectionProbabilityModel
from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class TOMHTScoringIntegrationTest(unittest.TestCase):
    def test_dpm_receives_public_track_id_and_caller_scan_context(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        hit_context = {"sensor": "radar-a", "mode": "track"}
        empty_context = {"sensor": "radar-a", "mode": "coast"}

        hypothesiser = _ScriptedHypothesiser()
        params = TOMHTParams(
            external_start_initial_existence_probability=0.6,
            track_confirmation_existence_probability=0.8,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )
        dpm = _CaptureDetectionProbabilityModel(
            clutter_density=params.log_epsilon,
        )
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=params,
            detection_probability_model=dpm,
        )

        tracker.update_tracker(t0, [], caller_scan_context={"sensor": "bootstrap"})
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        detection = _detection(1.0, 1.0, t1)
        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [detection],
            caller_scan_context=hit_context,
        )

        calls_after_hit = len(dpm.detection_probability_calls)
        self.assertGreaterEqual(calls_after_hit, 2)
        for track_id, _, caller_context in dpm.detection_probability_calls:
            self.assertIsNone(track_id)
            self.assertIs(hit_context, caller_context)
        self.assertEqual(1, len(dpm.clutter_density_calls))
        self.assertIs(detection, dpm.clutter_density_calls[0][1])
        self.assertIs(hit_context, dpm.clutter_density_calls[0][2])
        self.assertEqual(0, tracker.track_trees_by_track_id[0].public_track_id)

        tracker.update_tracker(t2, [], caller_scan_context=empty_context)

        empty_scan_calls = dpm.detection_probability_calls[calls_after_hit:]
        self.assertGreaterEqual(len(empty_scan_calls), 1)
        for track_id, _, caller_context in empty_scan_calls:
            self.assertEqual(0, track_id)
            self.assertIs(empty_context, caller_context)
        self.assertEqual(
            1,
            len(dpm.clutter_density_calls),
            "empty-detection miss scans must not request hit clutter density",
        )

    def test_explicit_constant_dpm_matches_default_tracker_scoring(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        params = TOMHTParams(
            prob_detect=0.8,
            clutter_density=0.25,
            external_start_initial_existence_probability=0.6,
            track_confirmation_existence_probability=0.8,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )

        default_hypothesiser = _ScriptedHypothesiser()
        default_tracker = TOMHTTracker(
            hypothesiser=default_hypothesiser,
            updater=_ScriptedUpdater(),
            params=params,
        )
        explicit_hypothesiser = _ScriptedHypothesiser()
        explicit_tracker = TOMHTTracker(
            hypothesiser=explicit_hypothesiser,
            updater=_ScriptedUpdater(),
            params=params,
            detection_probability_model=ConstantDetectionProbabilityModel(
                prob_detect=params.prob_detect,
                clutter_density=params.clutter_density,
            ),
        )

        for tracker in (default_tracker, explicit_tracker):
            tracker.update_tracker(t0, [])
            tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        detection = _detection(1.0, 1.0, t1)
        for hypothesiser in (default_hypothesiser, explicit_hypothesiser):
            hypothesiser.set_options(
                timestamp=t1,
                track_id=0,
                options=[(0, 2.0), (None, 0.0)],
            )
        default_tracker.update_tracker(t1, [detection])
        explicit_tracker.update_tracker(t1, [detection])

        default_snapshot = default_tracker.get_map_hypothesis_snapshot()
        explicit_snapshot = explicit_tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(default_snapshot)
        self.assertIsNotNone(explicit_snapshot)
        assert default_snapshot is not None
        assert explicit_snapshot is not None
        self.assertAlmostEqual(
            default_snapshot.log_weight, explicit_snapshot.log_weight
        )
        default_leaf = next(iter(default_snapshot.leaf_nodes_by_track_id.values()))
        explicit_leaf = next(iter(explicit_snapshot.leaf_nodes_by_track_id.values()))
        self.assertAlmostEqual(
            default_leaf.accumulated_log_score,
            explicit_leaf.accumulated_log_score,
        )


if __name__ == "__main__":
    unittest.main()
