#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

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
BASELINE_DIR = REPO_ROOT / "replay" / "smoke_baselines"
LATEST_DIR = REPO_ROOT / "replay" / "outputs" / "smoke_regression_latest"
_BASELINE_MISSING_HELP = (
    "Run this first:\n"
    "  source venv/bin/activate && python replay/smoke_output_regression.py update"
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    script_relpath: str
    start_time_iso: str

    @property
    def script_path(self) -> Path:
        return REPO_ROOT / self.script_relpath

    @property
    def baseline_normalized_path(self) -> Path:
        return BASELINE_DIR / f"{self.name}.normalized.log"

    @property
    def baseline_raw_path(self) -> Path:
        return BASELINE_DIR / f"{self.name}.raw.log"

    @property
    def baseline_timing_summary_path(self) -> Path:
        return BASELINE_DIR / f"{self.name}.timing_summary.log"

    @property
    def latest_normalized_path(self) -> Path:
        return LATEST_DIR / f"{self.name}.normalized.log"

    @property
    def latest_raw_path(self) -> Path:
        return LATEST_DIR / f"{self.name}.raw.log"

    @property
    def latest_timing_summary_path(self) -> Path:
        return LATEST_DIR / f"{self.name}.timing_summary.log"


@dataclass(frozen=True)
class ScenarioOutput:
    raw_output: str
    normalized_lines: list[str]


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        name="crossing",
        script_relpath="mht/runners/run_tomht_crossing.py",
        start_time_iso="2026-01-01T00:00:00",
    ),
    ScenarioSpec(
        name="bearing_range",
        script_relpath="mht/runners/run_tomht_bearing_range.py",
        start_time_iso="2026-01-01T00:00:00",
    ),
)


def _normalize_output(raw_output: str) -> list[str]:
    normalized_lines: list[str] = []
    for raw_line in raw_output.splitlines():
        line = raw_line.rstrip()
        if should_drop_matplotlib_cache_warning(line):
            continue
        if line.startswith(VOLATILE_LINE_PREFIXES):
            continue
        normalized_lines.append(normalize_python_warning_paths(line))
    return normalized_lines


def _run_one_scenario(
    spec: ScenarioSpec,
    *,
    expansion_frontier: bool,
) -> ScenarioOutput:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["TOMHT_NO_SHOW"] = "1"
    if expansion_frontier:
        env["TOMHT_DEBUG_EXPANSION_FRONTIER"] = "1"
    env = ensure_mpl_cache_env(env)
    cmd = [
        sys.executable,
        str(spec.script_path),
        "--no-debug-detections",
        "--debug-scan-stats",
        "--no-debug-hypotheses",
        "--no-debug-births",
        "--scenario-start-time",
        spec.start_time_iso,
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-60:])
        raise RuntimeError(
            f"Scenario {spec.name!r} failed with exit code {completed.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output tail:\n{tail}"
        )
    raw_output = completed.stdout
    return ScenarioOutput(
        raw_output=raw_output,
        normalized_lines=_normalize_output(raw_output),
    )


def _run_selected_scenarios(
    selected_names: set[str],
    *,
    expansion_frontier: bool,
) -> dict[str, ScenarioOutput]:
    selected_specs = [
        scenario for scenario in SCENARIOS if scenario.name in selected_names
    ]
    outputs: dict[str, ScenarioOutput] = {}
    for spec in selected_specs:
        print(f"[run] {spec.name}")
        outputs[spec.name] = _run_one_scenario(
            spec,
            expansion_frontier=expansion_frontier,
        )
    return outputs


def _write_latest_outputs(outputs_by_name: dict[str, ScenarioOutput]) -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SCENARIOS:
        output = outputs_by_name.get(spec.name)
        if output is None:
            continue
        write_text(spec.latest_raw_path, output.raw_output)
        write_text(
            spec.latest_normalized_path,
            "\n".join(output.normalized_lines) + "\n",
        )
        _, timing_summary_lines = write_timing_summary_from_raw_log(
            spec.latest_raw_path,
            output_path=spec.latest_timing_summary_path,
        )
        print(
            f"[write] {spec.latest_raw_path.relative_to(REPO_ROOT)} "
            f"({len(output.raw_output.splitlines())} lines)"
        )
        print(
            f"[write] {spec.latest_normalized_path.relative_to(REPO_ROOT)} "
            f"({len(output.normalized_lines)} lines)"
        )
        print(
            f"[write] {spec.latest_timing_summary_path.relative_to(REPO_ROOT)} "
            f"({len(timing_summary_lines)} lines)"
        )


def _write_baselines(outputs_by_name: dict[str, ScenarioOutput]) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SCENARIOS:
        output = outputs_by_name.get(spec.name)
        if output is None:
            continue
        write_text(spec.baseline_raw_path, output.raw_output)
        write_text(
            spec.baseline_normalized_path,
            "\n".join(output.normalized_lines) + "\n",
        )
        _, timing_summary_lines = write_timing_summary_from_raw_log(
            spec.baseline_raw_path,
            output_path=spec.baseline_timing_summary_path,
        )
        print(
            f"[write] {spec.baseline_raw_path.relative_to(REPO_ROOT)} "
            f"({len(output.raw_output.splitlines())} lines)"
        )
        print(
            f"[write] {spec.baseline_normalized_path.relative_to(REPO_ROOT)} "
            f"({len(output.normalized_lines)} lines)"
        )
        print(
            f"[write] {spec.baseline_timing_summary_path.relative_to(REPO_ROOT)} "
            f"({len(timing_summary_lines)} lines)"
        )


def _compare(
    outputs_by_name: dict[str, ScenarioOutput],
    *,
    max_diff_lines: int,
    timing_report: bool,
) -> int:
    exit_code = 0
    for spec in SCENARIOS:
        output = outputs_by_name.get(spec.name)
        if output is None:
            continue
        expected_lines = load_lines_or_raise(
            spec.baseline_normalized_path,
            missing_help=_BASELINE_MISSING_HELP,
        )
        actual_lines = output.normalized_lines
        if expected_lines == actual_lines:
            print(f"[ok] {spec.name} matches baseline ({len(actual_lines)} lines)")
        else:
            exit_code = 1
            diff_lines = format_diff(
                expected_lines,
                actual_lines,
                name=spec.baseline_normalized_path.name,
            )
            print(
                f"[diff] {spec.name} differs from baseline: "
                f"{spec.baseline_normalized_path}"
            )
            if diff_lines:
                for line in diff_lines[:max_diff_lines]:
                    print(line)
                if len(diff_lines) > max_diff_lines:
                    print(
                        f"... diff truncated ({len(diff_lines)} total lines, "
                        f"showing first {max_diff_lines})"
                    )
            print(f"[hint] baseline raw: {spec.baseline_raw_path}")
            print(f"[hint] latest raw: {spec.latest_raw_path}")
        if timing_report:
            expected_raw_lines = load_lines_or_raise(
                spec.baseline_raw_path,
                missing_help=_BASELINE_MISSING_HELP,
            )
            actual_raw_lines = output.raw_output.splitlines()
            emit_timing_report(
                name=spec.name,
                expected_raw_lines=expected_raw_lines,
                actual_raw_lines=actual_raw_lines,
                max_diff_lines=max_diff_lines,
            )
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TO-MHT smoke scenarios, store raw + normalized output, and "
            "compare normalized output against versioned baselines."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("compare", "run", "update"),
        default="compare",
        help=(
            "`compare` (default): fail on baseline mismatch, "
            "`run`: write latest artifacts and skip comparison, "
            "`update`: rewrite baselines."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(spec.name for spec in SCENARIOS),
        default=None,
        help="Run only selected scenario(s); can be repeated.",
    )
    parser.add_argument(
        "--max-diff-lines",
        type=int,
        default=200,
        help="Maximum unified-diff lines printed per scenario mismatch.",
    )
    parser.add_argument(
        "--timing-report",
        action="store_true",
        help=(
            "Print raw-log timing summary comparison against baseline "
            "(compare mode only)."
        ),
    )
    parser.add_argument(
        "--expansion-frontier",
        action="store_true",
        help=(
            "Enable opt-in expansion/frontier diagnostics for the scenario run. "
            "Usually combined with `run` mode so diagnostic output does not "
            "fail baseline comparison."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected_names = (
        {spec.name for spec in SCENARIOS}
        if args.scenario is None
        else set(args.scenario)
    )
    outputs_by_name = _run_selected_scenarios(
        selected_names,
        expansion_frontier=bool(args.expansion_frontier),
    )
    _write_latest_outputs(outputs_by_name)
    if args.mode == "run":
        print("[done] latest outputs written (comparison skipped)")
        return 0
    if args.mode == "update":
        _write_baselines(outputs_by_name)
        print("[done] baselines updated")
        return 0
    return _compare(
        outputs_by_name,
        max_diff_lines=args.max_diff_lines,
        timing_report=bool(args.timing_report),
    )


if __name__ == "__main__":
    raise SystemExit(main())
