#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from replay.regression_common import (
    VOLATILE_LINE_PREFIXES,
    emit_timing_report,
    ensure_mpl_cache_env,
    format_diff,
    load_lines_or_raise,
    normalize_python_warning_paths,
    should_drop_matplotlib_cache_warning,
    write_timing_summary_from_raw_log,
    write_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_MCAP = REPO_ROOT / "replay" / "inputs" / "cpi_replay_2025-12-10_173948.mcap"

BASELINE_DIR = REPO_ROOT / "replay" / "replay_baselines"
BASELINE_RAW_PATH = BASELINE_DIR / "standard_replay_default.raw.log"
BASELINE_NORMALIZED_PATH = BASELINE_DIR / "standard_replay_default.normalized.log"
BASELINE_TIMING_SUMMARY_PATH = (
    BASELINE_DIR / "standard_replay_default.timing_summary.log"
)

LATEST_DIR = REPO_ROOT / "replay" / "outputs" / "standard_replay_regression_latest"
LATEST_RAW_PATH = LATEST_DIR / "latest.raw.log"
LATEST_NORMALIZED_PATH = LATEST_DIR / "latest.normalized.log"
LATEST_TIMING_SUMMARY_PATH = LATEST_DIR / "latest.timing_summary.log"

_BASELINE_MISSING_HELP = (
    "Run this first (only when baseline refresh is intended):\n"
    "  source venv/bin/activate && python replay/standard_replay_regression.py update"
)
_RUN_ID_RE = re.compile(r"mcap_replay__\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}Z")


def _detect_default_replay_repo() -> Path | None:
    env_override = os.environ.get("REPLAY_REGRESSION_REPO")
    if env_override:
        return Path(env_override).expanduser()
    candidate = REPO_ROOT.parent / "l2-sp"
    if candidate.exists():
        return candidate
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard replay command, store raw + normalized output, and "
            "compare normalized output against a versioned baseline."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("compare", "update"),
        default="compare",
        help="`compare` (default) or `update` baseline artifacts.",
    )
    parser.add_argument(
        "--replay-repo",
        type=Path,
        default=_detect_default_replay_repo(),
        help=(
            "Path to repo containing `python.pipeline.batch_mcap_replay` "
            "(default: $REPLAY_REGRESSION_REPO or sibling `../l2-sp` if present)."
        ),
    )
    parser.add_argument(
        "--replay-python",
        type=Path,
        default=None,
        help=(
            "Python executable used to run replay. "
            "Default: <replay-repo>/venv/bin/python."
        ),
    )
    parser.add_argument(
        "--max-cpis",
        type=int,
        default=400,
        help="Max CPIs for the standard replay command (default: 400).",
    )
    parser.add_argument(
        "--tracker-param-override-file",
        type=Path,
        default=None,
        help="Optional tracker override JSON path.",
    )
    parser.add_argument(
        "--keep-output-artifacts",
        action="store_true",
        help=(
            "Keep replay-generated output artifacts instead of deleting the "
            "temporary output directory after run."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Replay output directory root (used only with --keep-output-artifacts; "
            "default: replay/outputs/standard_replay_regression_last_run)."
        ),
    )
    parser.add_argument(
        "--max-diff-lines",
        type=int,
        default=240,
        help="Maximum number of unified-diff lines to print on mismatch.",
    )
    parser.add_argument(
        "--timing-report",
        action="store_true",
        help=(
            "Print raw-log timing summary comparison against baseline "
            "(compare mode only)."
        ),
    )
    return parser.parse_args()


def _require_existing_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _require_existing_executable(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"{label} not found: {expanded}")
    # Keep the venv symlink path instead of resolving to the system interpreter.
    return expanded


def _resolve_replay_python(replay_repo: Path, replay_python: Path | None) -> Path:
    if replay_python is not None:
        return _require_existing_executable(
            replay_python, label="Replay python executable"
        )
    return _require_existing_executable(
        replay_repo / "venv" / "bin" / "python",
        label="Replay python executable",
    )


def _resolve_override_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = path if path.is_absolute() else REPO_ROOT / path
    return _require_existing_path(candidate, label="Tracker override file")


def _normalize_output(
    raw_output: str,
    *,
    replay_repo: Path,
    replay_python: Path,
    replay_output_root: Path,
) -> list[str]:
    normalized_lines: list[str] = []
    repo_root_str = str(REPO_ROOT)
    replay_repo_str = str(replay_repo)
    replay_python_str = str(replay_python)
    replay_output_root_str = str(replay_output_root)

    for raw_line in raw_output.splitlines():
        line = raw_line.rstrip()
        if should_drop_matplotlib_cache_warning(line):
            continue
        if line.startswith(VOLATILE_LINE_PREFIXES):
            continue
        line = _RUN_ID_RE.sub("mcap_replay__<RUN_ID>", line)
        line = line.replace(repo_root_str, "<STONE_SOUP_TRACKING_ROOT>")
        line = line.replace(replay_repo_str, "<REPLAY_REPO>")
        line = line.replace(replay_python_str, "<REPLAY_PYTHON>")
        line = line.replace(replay_output_root_str, "<REPLAY_OUTPUT_ROOT>")
        normalized_lines.append(normalize_python_warning_paths(line))
    return normalized_lines


def _run_standard_replay(
    *,
    replay_repo: Path,
    replay_python: Path,
    max_cpis: int,
    tracker_override: Path | None,
    keep_output_artifacts: bool,
    output_root: Path | None,
) -> tuple[str, list[str], Path]:
    if keep_output_artifacts:
        replay_output_root = (
            output_root
            if output_root is not None
            else REPO_ROOT
            / "replay"
            / "outputs"
            / "standard_replay_regression_last_run"
        )
        replay_output_root.mkdir(parents=True, exist_ok=True)
    else:
        replay_output_root = Path(
            tempfile.mkdtemp(prefix="standard_replay_regression_", dir="/tmp")
        )

    cmd = [
        str(replay_python),
        "-m",
        "python.pipeline.batch_mcap_replay",
        str(INPUT_MCAP),
        "--include-tracker",
        "--tracker-type",
        "stonesoup-mht",
        "--max-cpis",
        str(max_cpis),
        "--output-path",
        str(replay_output_root),
    ]
    if tracker_override is not None:
        cmd.extend(
            [
                "--tracker-param-override-file",
                str(tracker_override),
            ]
        )

    env = ensure_mpl_cache_env(os.environ.copy())
    completed = subprocess.run(
        cmd,
        cwd=replay_repo,
        check=False,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            "Standard replay failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {completed.returncode}\n"
            f"Output tail:\n{tail}"
        )

    raw_output = completed.stdout
    normalized_lines = _normalize_output(
        raw_output,
        replay_repo=replay_repo,
        replay_python=replay_python,
        replay_output_root=replay_output_root,
    )
    return raw_output, normalized_lines, replay_output_root


def _save_latest(raw_output: str, normalized_lines: list[str]) -> list[str]:
    write_text(LATEST_RAW_PATH, raw_output)
    write_text(LATEST_NORMALIZED_PATH, "\n".join(normalized_lines) + "\n")
    _, summary_lines = write_timing_summary_from_raw_log(
        LATEST_RAW_PATH,
        output_path=LATEST_TIMING_SUMMARY_PATH,
    )
    return summary_lines


def _update_baseline(raw_output: str, normalized_lines: list[str]) -> None:
    write_text(BASELINE_RAW_PATH, raw_output)
    write_text(BASELINE_NORMALIZED_PATH, "\n".join(normalized_lines) + "\n")
    _, timing_summary_lines = write_timing_summary_from_raw_log(
        BASELINE_RAW_PATH,
        output_path=BASELINE_TIMING_SUMMARY_PATH,
    )
    print(
        f"[write] {BASELINE_RAW_PATH.relative_to(REPO_ROOT)} "
        f"({len(raw_output.splitlines())} lines)"
    )
    print(
        f"[write] {BASELINE_NORMALIZED_PATH.relative_to(REPO_ROOT)} "
        f"({len(normalized_lines)} lines)"
    )
    print(
        f"[write] {BASELINE_TIMING_SUMMARY_PATH.relative_to(REPO_ROOT)} "
        f"({len(timing_summary_lines)} lines)"
    )


def _compare(normalized_lines: list[str], *, max_diff_lines: int) -> int:
    expected_lines = load_lines_or_raise(
        BASELINE_NORMALIZED_PATH,
        missing_help=_BASELINE_MISSING_HELP,
    )
    if expected_lines == normalized_lines:
        print(f"[ok] standard replay matches baseline ({len(normalized_lines)} lines)")
        return 0
    print("[diff] standard replay differs from baseline")
    diff_lines = format_diff(
        expected_lines,
        normalized_lines,
        name=BASELINE_NORMALIZED_PATH.name,
    )
    for line in diff_lines[:max_diff_lines]:
        print(line)
    if len(diff_lines) > max_diff_lines:
        print(
            f"... diff truncated ({len(diff_lines)} total lines, "
            f"showing first {max_diff_lines})"
        )
    print(f"[hint] baseline raw: {BASELINE_RAW_PATH}")
    print(f"[hint] latest raw: {LATEST_RAW_PATH}")
    return 1


def main() -> int:
    args = _parse_args()
    if args.replay_repo is None:
        raise ValueError(
            "Replay repo was not provided and no default was detected. "
            "Pass --replay-repo or set REPLAY_REGRESSION_REPO."
        )
    replay_repo = _require_existing_path(args.replay_repo, label="Replay repo")
    replay_python = _resolve_replay_python(replay_repo, args.replay_python)
    _require_existing_path(INPUT_MCAP, label="Input MCAP")
    tracker_override = _resolve_override_path(args.tracker_param_override_file)

    print(f"[run] replay_repo={replay_repo}")
    print(f"[run] replay_python={replay_python}")
    raw_output, normalized_lines, replay_output_root = _run_standard_replay(
        replay_repo=replay_repo,
        replay_python=replay_python,
        max_cpis=args.max_cpis,
        tracker_override=tracker_override,
        keep_output_artifacts=bool(args.keep_output_artifacts),
        output_root=args.output_root,
    )
    latest_timing_summary_lines = _save_latest(raw_output, normalized_lines)
    print(
        f"[write] {LATEST_RAW_PATH.relative_to(REPO_ROOT)} "
        f"({len(raw_output.splitlines())} lines)"
    )
    print(
        f"[write] {LATEST_NORMALIZED_PATH.relative_to(REPO_ROOT)} "
        f"({len(normalized_lines)} lines)"
    )
    print(
        f"[write] {LATEST_TIMING_SUMMARY_PATH.relative_to(REPO_ROOT)} "
        f"({len(latest_timing_summary_lines)} lines)"
    )

    if args.mode == "update":
        _update_baseline(raw_output, normalized_lines)
        print("[done] replay baseline updated")
        status = 0
    else:
        status = _compare(normalized_lines, max_diff_lines=args.max_diff_lines)
        if args.timing_report:
            expected_raw_lines = load_lines_or_raise(
                BASELINE_RAW_PATH,
                missing_help=_BASELINE_MISSING_HELP,
            )
            emit_timing_report(
                name="standard replay",
                expected_raw_lines=expected_raw_lines,
                actual_raw_lines=raw_output.splitlines(),
                max_diff_lines=args.max_diff_lines,
                format_diff_name="standard_replay.timing-summary",
            )

    if args.keep_output_artifacts:
        print(f"[keep] replay output artifacts: {replay_output_root}")
    else:
        shutil.rmtree(replay_output_root, ignore_errors=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
