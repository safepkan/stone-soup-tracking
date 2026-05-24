from __future__ import annotations

import unittest
from typing import Any

from mht.tomht_params import TOMHTParams


class TOMHTParamsTest(unittest.TestCase):
    def test_removed_parameters_are_absent(self) -> None:
        param_fields = TOMHTParams.__dataclass_fields__
        self.assertNotIn("scoring_mode", param_fields)
        self.assertNotIn("internal_birth_mode", param_fields)
        self.assertNotIn("birth_log_penalty", param_fields)
        self.assertNotIn("birth_density", param_fields)
        self.assertNotIn("birth_max_abs_pos", param_fields)
        self.assertNotIn("birth_max_covar_trace", param_fields)
        self.assertNotIn("historical_conflict_relaxation_enabled", param_fields)

    def test_initiator_start_initial_existence_probability_rejects_boundaries(
        self,
    ) -> None:
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "initiator_start_initial_existence_probability",
                ):
                    TOMHTParams(
                        initiator_start_initial_existence_probability=probability,
                    )

    def test_track_confirmation_existence_probability_rejects_boundaries(
        self,
    ) -> None:
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "track_confirmation_existence_probability",
                ):
                    TOMHTParams(
                        track_confirmation_existence_probability=probability,
                    )

    def test_track_deletion_existence_probability_rejects_boundaries(
        self,
    ) -> None:
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "track_deletion_existence_probability",
                ):
                    TOMHTParams(
                        track_deletion_existence_probability=probability,
                    )

    def test_publication_params_validate_domains(self) -> None:
        self.assertEqual(("confirmed",), TOMHTParams().publish_lifecycle_states)
        TOMHTParams(publish_lifecycle_states=())

        invalid_cases: list[tuple[dict[str, Any], str]] = [
            ({"publish_lifecycle_states": ("tentative", "invalid")}, "lifecycle"),
            ({"publish_lifecycle_states": "confirmed"}, "not a string"),
            ({"publish_min_hits": -1}, "publish_min_hits"),
            ({"publish_min_age": -1}, "publish_min_age"),
            ({"publish_min_existence_probability": -0.1}, "existence"),
            ({"publish_min_existence_probability": 1.0}, "existence"),
            ({"publish_min_existence_probability": float("nan")}, "existence"),
            ({"publish_min_existence_probability": float("inf")}, "existence"),
        ]
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    TOMHTParams(**overrides)

    def test_expansion_and_birth_guardrail_defaults(self) -> None:
        params = TOMHTParams()

        self.assertEqual(5, params.max_children_per_leaf)
        self.assertEqual(10, params.max_births_per_scan)
        self.assertIsNone(params.birth_skip_if_active_trees_above)
        self.assertIsNone(params.birth_skip_if_active_leaves_above)


if __name__ == "__main__":
    unittest.main()
