from __future__ import annotations

import unittest
from unittest.mock import patch

from mht.utils import elapsed_ms, elapsed_ns, ns_to_ms, start_timer


class TimerUtilsTest(unittest.TestCase):
    def test_ns_to_ms_converts_nanoseconds_to_milliseconds(self) -> None:
        self.assertEqual(1.5, ns_to_ms(1_500_000))

    def test_timer_helpers_use_elapsed_perf_counter_nanoseconds(self) -> None:
        with patch(
            "mht.utils.wall_clock.perf_counter_ns",
            side_effect=[10_000, 1_010_000, 2_010_000],
        ):
            t0 = start_timer()

            self.assertEqual(10_000, t0)
            self.assertEqual(1_000_000, elapsed_ns(t0))
            self.assertEqual(2.0, elapsed_ms(t0))
