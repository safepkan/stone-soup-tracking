from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mht.runners.tomht_runner import (
    TOMHTOperatingMode,
    load_params_overrides_json,
    parse_scenario_start_time,
    run_tomht,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TO-MHT crossing scenario.")
    parser.add_argument(
        "--operating-mode",
        dest="operating_mode",
        type=str.upper,
        choices=TOMHTOperatingMode.choices(),
        default="CUSTOM",
        help=(
            "Operating mode: CUSTOM (use detailed flags), EXTERNAL, INTERNAL, "
            "or BOTH. Mode resolves births/external-start behavior; CUSTOM "
            "uses --births and --external-starts. Timing is configured "
            "separately via external-start scan options."
        ),
    )
    parser.add_argument(
        "--births",
        dest="births",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Enable/disable initiator/births. " "Used in CUSTOM mode (default: on)."),
    )
    parser.add_argument(
        "--external-starts",
        dest="external_starts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable/disable external starts. Used in CUSTOM mode " "(default: off)."),
    )
    external_start_group = parser.add_mutually_exclusive_group()
    external_start_group.add_argument(
        "--external-start-delay-scans",
        dest="external_start_delay_scans",
        type=int,
        default=None,
        help=(
            "Inject confirmed external starts after this many completed scans "
            "(0-based delay from scan 0). Timing only; enablement is resolved "
            "before timing (mode-controlled except CUSTOM --external-starts)."
        ),
    )
    external_start_group.add_argument(
        "--external-start-scan",
        dest="external_start_scan",
        type=int,
        default=None,
        help=(
            "Inject confirmed external starts after processing this 0-based scan. "
            "Timing only; enablement is resolved before timing "
            "(mode-controlled except CUSTOM --external-starts)."
        ),
    )
    parser.add_argument(
        "--debug-detections",
        dest="debug_detections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable detection debug output (default: tracker default).",
    )
    parser.add_argument(
        "--debug-scan-stats",
        dest="debug_scan_stats",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable per-scan summary logs (default: tracker default).",
    )
    parser.add_argument(
        "--debug-hypotheses",
        dest="debug_hypotheses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable hypothesis debug output (default: tracker default).",
    )
    parser.add_argument(
        "--debug-births",
        dest="debug_births",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable birth debug output (default: tracker default).",
    )
    parser.add_argument(
        "--params-override",
        dest="params_override_json",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file with TOMHTParams field overrides "
            "(top-level object: key->value)."
        ),
    )
    parser.add_argument(
        "--scenario-start-time",
        dest="scenario_start_time",
        type=parse_scenario_start_time,
        default=None,
        help=(
            "Pin scenario start time (ISO-8601) for reproducible timestamps, "
            "for example 2026-01-01T12:00:00."
        ),
    )
    # VS Code/Jupyter Interactive injects kernel args (for example --f=...).
    # Ignore unknown args so this script works both as CLI and in notebooks.
    args, _unknown = parser.parse_known_args()
    params_overrides = (
        load_params_overrides_json(args.params_override_json)
        if args.params_override_json is not None
        else None
    )

    run_tomht(
        "crossing",
        use_initiator=args.births,
        use_external_starts=args.external_starts,
        operating_mode=args.operating_mode,
        external_start_scan=args.external_start_scan,
        external_start_delay_scans=args.external_start_delay_scans,
        debug_display_detections=args.debug_detections,
        debug_display_scan_stats=args.debug_scan_stats,
        debug_display_hypotheses=args.debug_hypotheses,
        debug_display_births=args.debug_births,
        scenario_start_time=args.scenario_start_time,
        params_overrides=params_overrides,
    )
