from __future__ import annotations

import unittest

import numpy as np

from mht.tomht_runner import (
    ExternalStartTimingConfiguration,
    ModeConfiguration,
    TOMHTOperatingMode,
    _normalize_operating_mode,
    _resolve_external_start_enablement,
    _resolve_external_start_timing,
    _resolve_mode_configuration,
)
from mht.scenarios.bearing_range import (
    create_bearing_range_mht_example,
    external_tomht_tracks_for_bearing_range,
)
from mht.scenarios.crossing_targets import (
    create_crossing_scenario,
    external_tomht_tracks_for_crossing,
)


class ExternalStartResolutionTest(unittest.TestCase):
    def test_enablement_custom_defaults_off(self) -> None:
        self.assertFalse(
            _resolve_external_start_enablement(
                mode=TOMHTOperatingMode.CUSTOM,
                use_external_starts=False,
            )
        )

    def test_enablement_custom_follows_explicit_flag(self) -> None:
        self.assertTrue(
            _resolve_external_start_enablement(
                mode=TOMHTOperatingMode.CUSTOM,
                use_external_starts=True,
            )
        )

    def test_enablement_external_mode_forces_on(self) -> None:
        self.assertTrue(
            _resolve_external_start_enablement(
                mode=TOMHTOperatingMode.EXTERNAL,
                use_external_starts=False,
            )
        )

    def test_enablement_internal_mode_forces_off(self) -> None:
        self.assertFalse(
            _resolve_external_start_enablement(
                mode=TOMHTOperatingMode.INTERNAL,
                use_external_starts=True,
            )
        )

    def test_timing_uses_explicit_delay_value(self) -> None:
        self.assertEqual(
            ExternalStartTimingConfiguration(start_scan=3, source="explicit_delay"),
            _resolve_external_start_timing(
                mode=TOMHTOperatingMode.CUSTOM,
                num_scans=10,
                external_start_scan=None,
                external_start_delay_scans=3,
            ),
        )

    def test_timing_defaults_to_scan_zero_when_mode_enables_external_starts(
        self,
    ) -> None:
        self.assertEqual(
            ExternalStartTimingConfiguration(start_scan=0, source="mode_default_scan0"),
            _resolve_external_start_timing(
                mode=TOMHTOperatingMode.EXTERNAL,
                num_scans=10,
                external_start_scan=None,
                external_start_delay_scans=None,
            ),
        )

    def test_timing_rejects_custom_enabled_without_timing(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUSTOM mode"):
            _resolve_external_start_timing(
                mode=TOMHTOperatingMode.CUSTOM,
                num_scans=10,
                external_start_scan=None,
                external_start_delay_scans=None,
            )

    def test_timing_rejects_conflicting_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one"):
            _resolve_external_start_timing(
                mode=TOMHTOperatingMode.CUSTOM,
                num_scans=10,
                external_start_scan=2,
                external_start_delay_scans=3,
            )

    def test_timing_rejects_out_of_range_scan(self) -> None:
        with self.assertRaisesRegex(ValueError, "must fall within"):
            _resolve_external_start_timing(
                mode=TOMHTOperatingMode.EXTERNAL,
                num_scans=10,
                external_start_scan=10,
                external_start_delay_scans=None,
            )


class OperatingModeResolutionTest(unittest.TestCase):
    def test_normalize_mode_defaults_to_custom(self) -> None:
        self.assertEqual(TOMHTOperatingMode.CUSTOM, _normalize_operating_mode(None))

    def test_normalize_mode_accepts_case_insensitive_values(self) -> None:
        self.assertEqual(
            TOMHTOperatingMode.EXTERNAL,
            _normalize_operating_mode("external"),
        )
        self.assertEqual(
            TOMHTOperatingMode.INTERNAL,
            _normalize_operating_mode("INTERNAL"),
        )
        self.assertEqual(TOMHTOperatingMode.BOTH, _normalize_operating_mode("Both"))

    def test_normalize_mode_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown operating mode"):
            _normalize_operating_mode("legacy")

    def test_custom_mode_preserves_manual_configuration(self) -> None:
        self.assertEqual(
            ModeConfiguration(
                use_initiator=False,
                external_starts_enabled=True,
            ),
            _resolve_mode_configuration(
                requested_mode=TOMHTOperatingMode.CUSTOM,
                configuration=ModeConfiguration(
                    use_initiator=False,
                    external_starts_enabled=True,
                ),
            ),
        )

    def test_external_mode_forces_external_only(self) -> None:
        self.assertEqual(
            ModeConfiguration(
                use_initiator=False,
                external_starts_enabled=True,
            ),
            _resolve_mode_configuration(
                requested_mode=TOMHTOperatingMode.EXTERNAL,
                configuration=ModeConfiguration(
                    use_initiator=True,
                    external_starts_enabled=False,
                ),
            ),
        )

    def test_internal_mode_forces_births_on_and_external_starts_off(self) -> None:
        self.assertEqual(
            ModeConfiguration(
                use_initiator=True,
                external_starts_enabled=False,
            ),
            _resolve_mode_configuration(
                requested_mode=TOMHTOperatingMode.INTERNAL,
                configuration=ModeConfiguration(
                    use_initiator=False,
                    external_starts_enabled=True,
                ),
            ),
        )

    def test_both_mode_forces_births_and_external_starts(self) -> None:
        self.assertEqual(
            ModeConfiguration(
                use_initiator=True,
                external_starts_enabled=True,
            ),
            _resolve_mode_configuration(
                requested_mode=TOMHTOperatingMode.BOTH,
                configuration=ModeConfiguration(
                    use_initiator=False,
                    external_starts_enabled=False,
                ),
            ),
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
