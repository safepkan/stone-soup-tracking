from __future__ import annotations

import datetime
import json
import tempfile
import unittest
from pathlib import Path

from mht.runners.tomht_runner import (
    ExternalStartTimingConfiguration,
    ModeConfiguration,
    TOMHTOperatingMode,
    _normalize_operating_mode,
    _resolve_external_start_enablement,
    _resolve_external_start_timing,
    _resolve_mode_configuration,
    load_params_overrides_json,
    parse_scenario_start_time,
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


class ParamsOverrideJsonLoadTest(unittest.TestCase):
    def test_load_params_overrides_json_accepts_top_level_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "params_overrides.json"
            json_path.write_text(
                json.dumps({"max_children_per_leaf": 2, "debug_display_births": True}),
                encoding="utf-8",
            )

            loaded = load_params_overrides_json(json_path)

            self.assertEqual(2, loaded["max_children_per_leaf"])
            self.assertTrue(loaded["debug_display_births"])

    def test_load_params_overrides_json_rejects_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "params_overrides.json"
            json_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "top-level object"):
                load_params_overrides_json(json_path)


class ScenarioStartTimeParseTest(unittest.TestCase):
    def test_parse_scenario_start_time_accepts_iso_without_timezone(self) -> None:
        parsed = parse_scenario_start_time("2026-01-01T12:34:56")

        self.assertEqual(datetime.datetime(2026, 1, 1, 12, 34, 56), parsed)

    def test_parse_scenario_start_time_accepts_z_suffix(self) -> None:
        parsed = parse_scenario_start_time("2026-01-01T12:34:56Z")

        self.assertEqual(
            datetime.datetime(2026, 1, 1, 12, 34, 56, tzinfo=datetime.timezone.utc),
            parsed,
        )

    def test_parse_scenario_start_time_rejects_invalid_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid scenario start time"):
            parse_scenario_start_time("not-a-timestamp")


if __name__ == "__main__":
    unittest.main()
