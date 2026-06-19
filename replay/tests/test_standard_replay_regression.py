from __future__ import annotations

from pathlib import Path
import unittest

from replay import standard_replay_regression as regression


class StandardReplayRegressionOverrideTest(unittest.TestCase):
    def test_standard_override_is_resolved_before_user_overrides(self) -> None:
        user_override = Path("replay/overrides/tracker_dim_3.json")

        resolved = regression._resolve_tracker_override_paths([user_override])

        self.assertEqual(
            [
                regression.STANDARD_TRACKER_OVERRIDE_PATH.resolve(),
                (regression.REPO_ROOT / user_override).resolve(),
            ],
            resolved,
        )

    def test_command_repeats_override_flag_in_order(self) -> None:
        standard_override = Path("/tmp/standard.json")
        user_override = Path("/tmp/dim3.json")

        cmd = regression._build_standard_replay_command(
            replay_python=Path("/tmp/replay-python"),
            replay_output_root=Path("/tmp/replay-output"),
            max_cpis=400,
            tracker_overrides=[standard_override, user_override],
        )

        override_values = [
            cmd[index + 1]
            for index, value in enumerate(cmd)
            if value == "--tracker-param-override-file"
        ]
        self.assertEqual(
            [str(standard_override), str(user_override)],
            override_values,
        )


if __name__ == "__main__":
    unittest.main()
