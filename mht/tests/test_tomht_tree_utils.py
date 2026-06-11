from __future__ import annotations

import unittest

from mht.tomht_model import DetectionKey
from mht.tomht_tree_utils import trim_detection_history_keys_before_scan


class TOMHTTreeUtilsTest(unittest.TestCase):
    def test_trim_detection_history_keys_preserves_identity_when_no_keys_drop(
        self,
    ) -> None:
        history_keys = frozenset(
            {
                DetectionKey(scan_index=3, det_index=0),
                DetectionKey(scan_index=4, det_index=1),
            }
        )

        trimmed = trim_detection_history_keys_before_scan(
            history_keys=history_keys,
            min_scan_index=3,
        )

        self.assertIs(history_keys, trimmed)

    def test_trim_detection_history_keys_drops_keys_before_cutoff(self) -> None:
        history_keys = frozenset(
            {
                DetectionKey(scan_index=2, det_index=0),
                DetectionKey(scan_index=3, det_index=1),
                DetectionKey(scan_index=4, det_index=2),
            }
        )

        trimmed = trim_detection_history_keys_before_scan(
            history_keys=history_keys,
            min_scan_index=3,
        )

        self.assertEqual(
            frozenset(
                {
                    DetectionKey(scan_index=3, det_index=1),
                    DetectionKey(scan_index=4, det_index=2),
                }
            ),
            trimmed,
        )


if __name__ == "__main__":
    unittest.main()
