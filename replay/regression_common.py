from __future__ import annotations

import difflib
from pathlib import Path
import re
import statistics
from typing import Iterable

VOLATILE_LINE_PREFIXES = (
    "SCAN_TIMING ",
    "SCAN_TIMING_PHASES ",
    "SCAN_MEMORY ",
    "SUMMARY timing ",
    "SUMMARY timing_phases ",
    "SUMMARY memory ",
)
TIMING_SUMMARY_PREFIXES = (
    "SUMMARY timing ",
    "SUMMARY timing_phases ",
    "SUMMARY memory ",
)
MATPLOTLIB_CACHE_WARNING_PATTERNS = (
    ".matplotlib is not a writable directory",
    "Matplotlib created a temporary cache directory at ",
)
_SITE_PACKAGES_PREFIX_RE = re.compile(r"^.*site-packages/")
_PY_WARNING_LINE_RE = re.compile(r"^(site-packages/[^:]+):\d+:(\s+\w+Warning: .+)$")


def ensure_mpl_cache_env(env: dict[str, str] | None = None) -> dict[str, str]:
    env_out = dict(env or {})
    mpl_cache_dir = Path("/tmp/.cache")
    mpl_config_dir = Path("/tmp/mplconfig")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    env_out.setdefault("XDG_CACHE_HOME", str(mpl_cache_dir))
    env_out.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    return env_out


def should_drop_matplotlib_cache_warning(
    line: str,
    *,
    patterns: tuple[str, ...] = MATPLOTLIB_CACHE_WARNING_PATTERNS,
) -> bool:
    return any(pattern in line for pattern in patterns)


def normalize_python_warning_paths(line: str) -> str:
    normalized = _SITE_PACKAGES_PREFIX_RE.sub("site-packages/", line)
    return _PY_WARNING_LINE_RE.sub(r"\1:<LINE>:\2", normalized)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_lines_or_raise(path: Path, *, missing_help: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}\n{missing_help}")
    return path.read_text(encoding="utf-8").splitlines()


def format_diff(
    expected: Iterable[str], actual: Iterable[str], *, name: str
) -> list[str]:
    return list(
        difflib.unified_diff(
            list(expected),
            list(actual),
            fromfile=f"baseline/{name}",
            tofile=f"current/{name}",
            lineterm="",
        )
    )


def extract_timing_summary(
    lines: Iterable[str],
    *,
    timing_summary_prefixes: tuple[str, ...] = TIMING_SUMMARY_PREFIXES,
) -> list[str]:
    line_list = list(lines)
    summary_lines = [
        line for line in line_list if line.startswith(timing_summary_prefixes)
    ]
    if summary_lines:
        return summary_lines
    return _build_timing_summary_from_scan_lines(line_list)


def emit_timing_report(
    *,
    name: str,
    expected_raw_lines: list[str],
    actual_raw_lines: list[str],
    max_diff_lines: int,
    timing_summary_prefixes: tuple[str, ...] = TIMING_SUMMARY_PREFIXES,
    format_diff_name: str | None = None,
) -> None:
    expected_summary = extract_timing_summary(
        expected_raw_lines,
        timing_summary_prefixes=timing_summary_prefixes,
    )
    actual_summary = extract_timing_summary(
        actual_raw_lines,
        timing_summary_prefixes=timing_summary_prefixes,
    )
    print(f"[timing] {name}")
    if not expected_summary and not actual_summary:
        print("  no SUMMARY timing/timing_phases/memory lines found")
        return
    for line in expected_summary:
        print(f"  baseline: {line}")
    for line in actual_summary:
        print(f"  current : {line}")
    if expected_summary == actual_summary:
        print("  summary lines identical")
        return
    diff_lines = format_diff(
        expected_summary,
        actual_summary,
        name=format_diff_name or f"{name}.timing-summary",
    )
    for line in diff_lines[:max_diff_lines]:
        print(line)
    if len(diff_lines) > max_diff_lines:
        print(
            f"... timing diff truncated ({len(diff_lines)} total lines, "
            f"showing first {max_diff_lines})"
        )


def write_timing_summary(
    path: Path,
    *,
    raw_lines: Iterable[str],
    timing_summary_prefixes: tuple[str, ...] = TIMING_SUMMARY_PREFIXES,
) -> list[str]:
    summary_lines = extract_timing_summary(
        raw_lines,
        timing_summary_prefixes=timing_summary_prefixes,
    )
    write_text(path, "\n".join(summary_lines) + "\n")
    return summary_lines


def default_timing_summary_path_for_raw_log(raw_log_path: Path) -> Path:
    if raw_log_path.name.endswith(".raw.log"):
        return raw_log_path.with_name(
            raw_log_path.name[: -len(".raw.log")] + ".timing_summary.log"
        )
    return raw_log_path.with_name(raw_log_path.name + ".timing_summary.log")


def write_timing_summary_from_raw_log(
    raw_log_path: Path,
    *,
    output_path: Path | None = None,
    timing_summary_prefixes: tuple[str, ...] = TIMING_SUMMARY_PREFIXES,
) -> tuple[Path, list[str]]:
    raw_lines = raw_log_path.read_text(encoding="utf-8").splitlines()
    summary_path = (
        output_path
        if output_path is not None
        else default_timing_summary_path_for_raw_log(raw_log_path)
    )
    summary_lines = write_timing_summary(
        summary_path,
        raw_lines=raw_lines,
        timing_summary_prefixes=timing_summary_prefixes,
    )
    return summary_path, summary_lines


def _parse_numeric_fields(line: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        try:
            values[key] = float(raw_value)
        except ValueError:
            continue
    return values


def _build_timing_summary_from_scan_lines(lines: list[str]) -> list[str]:
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if quantile <= 0.0:
            return sorted_values[0]
        if quantile >= 1.0:
            return sorted_values[-1]
        rank = (len(sorted_values) - 1) * quantile
        lower_idx = int(rank)
        upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
        fraction = rank - float(lower_idx)
        return (
            sorted_values[lower_idx] * (1.0 - fraction)
            + sorted_values[upper_idx] * fraction
        )

    wall_ms_samples: list[float] = []
    phase_samples_by_key: dict[str, list[float]] = {}
    call_samples_by_key: dict[str, list[float]] = {}
    node_samples: list[float] = []
    maxrss_samples: list[float] = []

    for line in lines:
        if line.startswith("SCAN_TIMING "):
            values = _parse_numeric_fields(line)
            wall_ms = values.get("wall_ms")
            if wall_ms is not None:
                wall_ms_samples.append(wall_ms)
            continue
        if line.startswith("SCAN_TIMING_PHASES "):
            values = _parse_numeric_fields(line)
            for key, value in values.items():
                if key.endswith("_ms"):
                    phase_samples_by_key.setdefault(key, []).append(value)
                elif key.endswith("_calls"):
                    call_samples_by_key.setdefault(key, []).append(value)
            continue
        if line.startswith("SCAN_MEMORY "):
            values = _parse_numeric_fields(line)
            nodes = values.get("nodes")
            maxrss_mb = values.get("maxrss_mb")
            if nodes is not None:
                node_samples.append(nodes)
            if maxrss_mb is not None:
                maxrss_samples.append(maxrss_mb)

    if not wall_ms_samples and not phase_samples_by_key and not maxrss_samples:
        return []

    summary_lines: list[str] = []
    if wall_ms_samples:
        summary_lines.append(
            "SUMMARY_FROM_SCANS timing "
            f"scan_wall_ms count={len(wall_ms_samples)} "
            f"med={statistics.median(wall_ms_samples):.3f} "
            f"mean={statistics.mean(wall_ms_samples):.3f} "
            f"p65={_percentile(wall_ms_samples, 0.65):.3f} "
            f"p80={_percentile(wall_ms_samples, 0.80):.3f} "
            f"p90={_percentile(wall_ms_samples, 0.90):.3f} "
            f"p95={_percentile(wall_ms_samples, 0.95):.3f} "
            f"max={max(wall_ms_samples):.3f}"
        )
    for phase_key in phase_samples_by_key:
        samples = phase_samples_by_key[phase_key]
        summary_lines.append(
            "SUMMARY_FROM_SCANS timing_phase "
            f"{phase_key} count={len(samples)} "
            f"med={statistics.median(samples):.3f} "
            f"p65={_percentile(samples, 0.65):.3f} "
            f"p80={_percentile(samples, 0.80):.3f} "
            f"p90={_percentile(samples, 0.90):.3f} "
            f"p95={_percentile(samples, 0.95):.3f} "
            f"max={max(samples):.3f}"
        )
    for call_key in call_samples_by_key:
        samples = call_samples_by_key[call_key]
        summary_lines.append(
            "SUMMARY_FROM_SCANS timing_counter "
            f"{call_key} count={len(samples)} "
            f"med={statistics.median(samples):.1f} "
            f"mean={statistics.mean(samples):.2f} "
            f"max={max(samples):.1f} "
            f"sum={sum(samples):.0f}"
        )
    if node_samples or maxrss_samples:
        parts = ["SUMMARY_FROM_SCANS memory"]
        if node_samples:
            parts.extend(
                [
                    f"nodes_count={len(node_samples)}",
                    f"nodes_med={statistics.median(node_samples):.1f}",
                    f"nodes_max={max(node_samples):.0f}",
                ]
            )
        if maxrss_samples:
            parts.extend(
                [
                    f"maxrss_count={len(maxrss_samples)}",
                    f"maxrss_final={maxrss_samples[-1]:.1f}",
                    f"maxrss_peak={max(maxrss_samples):.1f}",
                ]
            )
        summary_lines.append(" ".join(parts))
    return summary_lines
