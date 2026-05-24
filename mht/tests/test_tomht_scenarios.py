from __future__ import annotations

import unittest

import numpy as np

from mht.scenarios.bearing_range import (
    create_bearing_range_mht_example,
    external_tomht_tracks_for_bearing_range,
)
from mht.scenarios.crossing_targets import (
    create_crossing_scenario,
    external_tomht_tracks_for_crossing,
)


class ScenarioExternalStartsTest(unittest.TestCase):
    def test_crossing_external_starts_follow_truth_state_at_scan(self) -> None:
        truths, scans, start_time, _transition_model, _measurement_model, _config = (
            create_crossing_scenario()
        )
        del scans, start_time
        scan_index = 4
        truth_list = list(truths)
        timestamp = truth_list[0][scan_index].timestamp

        starts = external_tomht_tracks_for_crossing(truths, scan_index, timestamp)

        self.assertEqual(2, len(starts))
        for truth_index, (truth, start) in enumerate(zip(truth_list, starts)):
            np.testing.assert_allclose(
                start[0].state_vector,
                truth[scan_index].state_vector,
            )
            self.assertEqual(timestamp, start[0].timestamp)
            self.assertEqual("crossing", start.metadata["scenario"])
            self.assertEqual(truth_index, start.metadata["scenario_truth_index"])

    def test_bearing_range_external_starts_follow_truth_state_at_scan(self) -> None:
        truths, scans, timestamps, _transition_model, _measurement_model, _config = (
            create_bearing_range_mht_example()
        )
        del scans
        scan_index = 5
        truth_list = list(truths)
        timestamp = timestamps[scan_index]

        starts = external_tomht_tracks_for_bearing_range(truths, scan_index, timestamp)

        self.assertEqual(3, len(starts))
        for truth_index, (truth, start) in enumerate(zip(truth_list, starts)):
            np.testing.assert_allclose(
                start[0].state_vector,
                truth[scan_index].state_vector,
            )
            self.assertEqual(timestamp, start[0].timestamp)
            self.assertEqual("bearing_range", start.metadata["scenario"])
            self.assertEqual(truth_index, start.metadata["scenario_truth_index"])


if __name__ == "__main__":
    unittest.main()
