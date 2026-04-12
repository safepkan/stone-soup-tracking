#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replay.regression_common import write_timing_summary_from_raw_log

REPO_ROOT = Path(__file__).resolve().parents[1]

KNOWN_RAW_LOGS_BASELINE: tuple[Path, ...] = (
    REPO_ROOT / "replay" / "smoke_baselines" / "crossing.raw.log",
    REPO_ROOT / "replay" / "smoke_baselines" / "bearing_range.raw.log",
    REPO_ROOT / "replay" / "replay_baselines" / "standard_replay_default.raw.log",
)
KNOWN_RAW_LOGS_LATEST: tuple[Path, ...] = (
    REPO_ROOT / "replay" / "outputs" / "smoke_regression_latest" / "crossing.raw.log",
    REPO_ROOT
    / "replay"
    / "outputs"
    / "smoke_regression_latest"
    / "bearing_range.raw.log",
    REPO_ROOT
    / "replay"
    / "outputs"
    / "standard_replay_regression_latest"
    / "latest.raw.log",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate timing summary artifacts from existing raw log file(s) "
            "without rerunning scenarios/replay."
        )
    )
    parser.add_argument(
        "raw_logs",
        nargs="*",
        type=Path,
        help="Raw log file path(s).",
    )
    parser.add_argument(
        "--known-set",
        choices=("baseline", "latest", "all"),
        default=None,
        help=(
            "Regenerate a known set of log summaries: baseline, latest, or all. "
            "Can be combined with explicit raw log paths."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output summary path. Only valid when exactly one input raw log is "
            "selected. Defaults to replacing `.raw.log` with `.timing_summary.log`."
        ),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing raw-log paths instead of failing.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Also print summary lines after writing each file.",
    )
    return parser.parse_args()


def _resolve_input_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def _known_logs_for_set(known_set: str | None) -> tuple[Path, ...]:
    if known_set is None:
        return ()
    if known_set == "baseline":
        return KNOWN_RAW_LOGS_BASELINE
    if known_set == "latest":
        return KNOWN_RAW_LOGS_LATEST
    return KNOWN_RAW_LOGS_BASELINE + KNOWN_RAW_LOGS_LATEST


def _dedupe_preserve_order(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def main() -> int:
    args = _parse_args()
    selected_paths = [
        _resolve_input_path(path)
        for path in [*args.raw_logs, *_known_logs_for_set(args.known_set)]
    ]
    selected_paths = _dedupe_preserve_order(selected_paths)

    if not selected_paths:
        raise ValueError("No input raw logs selected.")
    if args.output is not None and len(selected_paths) != 1:
        raise ValueError("--output requires exactly one selected input raw log.")

    for raw_log_path in selected_paths:
        if not raw_log_path.exists():
            if args.skip_missing:
                print(f"[skip] missing raw log: {raw_log_path}")
                continue
            raise FileNotFoundError(f"Raw log not found: {raw_log_path}")

        output_path = None
        if args.output is not None:
            output_path = _resolve_input_path(args.output)

        summary_path, summary_lines = write_timing_summary_from_raw_log(
            raw_log_path,
            output_path=output_path,
        )
        print(
            f"[write] {summary_path.relative_to(REPO_ROOT)} "
            f"({len(summary_lines)} lines) from {raw_log_path.relative_to(REPO_ROOT)}"
        )
        if args.print_summary:
            for line in summary_lines:
                print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
