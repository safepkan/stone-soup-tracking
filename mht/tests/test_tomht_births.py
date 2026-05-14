from __future__ import annotations

import datetime
from pathlib import Path
import unittest

import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from mht.tomht_births import (
    birth_existence_probability_sort_value,
    birth_track_sort_key,
    format_birth_state_vector,
    select_internal_birth_candidates,
)
from mht.tomht_params import TOMHTParams
from mht.tomht_types import ScanContext


def _track_with_state(
    values: list[float],
    *,
    timestamp: datetime.datetime,
    covar_scale: float = 1.0,
) -> Track:
    dim = len(values)
    covar = np.eye(dim) * covar_scale
    state_vector = np.asarray(values, dtype=float).reshape((dim, 1))
    return Track([GaussianState(state_vector, covar=covar, timestamp=timestamp)])


def _scan_context(timestamp: datetime.datetime) -> ScanContext:
    detections: list[Detection] = []
    return ScanContext(
        scan_index=3,
        timestamp=timestamp,
        detections=detections,
        det_index_by_obj={},
    )


class TOMHTBirthHelpersTest(unittest.TestCase):
    def test_candidate_selection_keeps_layout_extreme_states(self) -> None:
        timestamp = datetime.datetime(2026, 5, 14, 12, 0, 0)
        layout_extreme = _track_with_state(
            [1e9, 0.0, -1e9, 0.0],
            timestamp=timestamp,
            covar_scale=1e13,
        )
        short_state = _track_with_state([5.0, 6.0], timestamp=timestamp)

        selected = select_internal_birth_candidates(
            initiated_tracks=[layout_extreme, short_state],
            ctx=_scan_context(timestamp),
            params=TOMHTParams(max_births_per_scan=10),
        )

        self.assertEqual(2, len(selected))
        self.assertEqual(
            {id(layout_extreme), id(short_state)},
            {id(track) for track in selected},
        )

    def test_sort_key_prefers_higher_existence_probability_before_covariance(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 5, 14, 12, 0, 0)
        high_confidence = _track_with_state(
            [0.0],
            timestamp=timestamp,
            covar_scale=100.0,
        )
        low_confidence = _track_with_state(
            [0.0],
            timestamp=timestamp,
            covar_scale=1.0,
        )
        high_confidence.metadata["existence_probability"] = 0.9
        low_confidence.metadata["existence_probability"] = 0.2

        ordered = sorted(
            [low_confidence, high_confidence],
            key=lambda track: birth_track_sort_key(
                track,
                scan_index=3,
                det_index_by_obj={},
            ),
        )

        self.assertIs(high_confidence, ordered[0])

    def test_valid_existence_probability_sorts_before_missing_or_invalid(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 5, 14, 12, 0, 0)
        valid_confidence = _track_with_state(
            [0.0],
            timestamp=timestamp,
            covar_scale=100.0,
        )
        missing_confidence = _track_with_state(
            [0.0],
            timestamp=timestamp,
            covar_scale=1.0,
        )
        invalid_confidence = _track_with_state(
            [0.0],
            timestamp=timestamp,
            covar_scale=1.0,
        )
        valid_confidence.metadata["existence_probability"] = 0.6
        invalid_confidence.metadata["existence_probability"] = "not-a-number"

        ordered = sorted(
            [missing_confidence, invalid_confidence, valid_confidence],
            key=lambda track: birth_track_sort_key(
                track,
                scan_index=3,
                det_index_by_obj={},
            ),
        )

        self.assertIs(valid_confidence, ordered[0])

    def test_invalid_existence_probability_sort_values_fall_back_to_inf(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 5, 14, 12, 0, 0)
        for value in ("not-a-number", float("nan"), 0.0, 1.0):
            with self.subTest(value=value):
                track = _track_with_state([0.0], timestamp=timestamp)
                track.metadata["existence_probability"] = value

                self.assertEqual(
                    float("inf"),
                    birth_existence_probability_sort_value(track),
                )

    def test_sort_key_uses_covariance_when_existence_probabilities_are_missing(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 5, 14, 12, 0, 0)
        low_covariance = _track_with_state(
            [10.0],
            timestamp=timestamp,
            covar_scale=1.0,
        )
        high_covariance = _track_with_state(
            [0.0],
            timestamp=timestamp,
            covar_scale=100.0,
        )

        ordered = sorted(
            [high_covariance, low_covariance],
            key=lambda track: birth_track_sort_key(
                track,
                scan_index=3,
                det_index_by_obj={},
            ),
        )

        self.assertIs(low_covariance, ordered[0])

    def test_sort_key_is_deterministic_for_arbitrary_state_dimensions(self) -> None:
        timestamp = datetime.datetime(2026, 5, 14, 12, 0, 0)
        tracks = [
            _track_with_state([], timestamp=timestamp),
            _track_with_state([1.0], timestamp=timestamp),
            _track_with_state([1.0, 2.0], timestamp=timestamp),
            _track_with_state([1.0, 2.0, 3.0, 4.0], timestamp=timestamp),
            _track_with_state([1.0, 2.0, 3.0, 4.0, 5.0], timestamp=timestamp),
            _track_with_state([float("nan")], timestamp=timestamp),
        ]

        def _sorted_ids() -> list[int]:
            ordered = sorted(
                tracks,
                key=lambda track: birth_track_sort_key(
                    track,
                    scan_index=3,
                    det_index_by_obj={},
                ),
            )
            return [id(track) for track in ordered]

        self.assertEqual(_sorted_ids(), _sorted_ids())

    def test_sort_key_uses_all_flattened_state_components(self) -> None:
        timestamp = datetime.datetime(2026, 5, 14, 12, 0, 0)
        high_late_component = _track_with_state(
            [0.0, 0.0, 0.0, 0.0, 2.0],
            timestamp=timestamp,
        )
        low_late_component = _track_with_state(
            [0.0, 0.0, 0.0, 0.0, 1.0],
            timestamp=timestamp,
        )

        ordered = sorted(
            [high_late_component, low_late_component],
            key=lambda track: birth_track_sort_key(
                track,
                scan_index=3,
                det_index_by_obj={},
            ),
        )

        self.assertIs(low_late_component, ordered[0])

    def test_format_birth_state_vector_is_generic(self) -> None:
        cases = [
            (np.array([[1.25]]), "(0=1.25)"),
            (np.array([[1.0], [2.0]]), "(0=1, 1=2)"),
            (np.array([[1.0], [2.5], [3.0], [4.0]]), "(0=1, 1=2.5, 2=3, 3=4)"),
            (
                np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [float("nan")]]),
                "(0=1, 1=2, 2=3, 3=4, 4=5, 5=inf)",
            ),
        ]

        for state_vector, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(expected, format_birth_state_vector(state_vector))

    def test_tomht_births_has_no_fixed_state_layout_terms(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "tomht_births.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("birth_is_sane", source)
        self.assertNotIn("xyvxvy", source)
        self.assertNotIn("vx", source)
        self.assertNotIn("state_vector[0]", source)
        self.assertNotIn("state_vector[2]", source)
        self.assertNotIn("# x", source)
        self.assertNotIn("# y", source)


if __name__ == "__main__":
    unittest.main()
