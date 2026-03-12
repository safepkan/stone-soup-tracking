from __future__ import annotations

import argparse

from mht_experiments.runners.tomht_runner import run_tomht

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TO-MHT crossing scenario.")
    parser.add_argument(
        "--births",
        dest="births",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable initiator/births (default: on for crossing).",
    )
    parser.add_argument(
        "--initial-tracks",
        dest="initial_tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable scenario initial tracks (default: off for crossing).",
    )
    external_start_group = parser.add_mutually_exclusive_group()
    external_start_group.add_argument(
        "--external-start-delay-scans",
        dest="external_start_delay_scans",
        type=int,
        default=None,
        help=(
            "Inject confirmed external starts after this many completed scans "
            "(0-based delay from scan 0)."
        ),
    )
    external_start_group.add_argument(
        "--external-start-scan",
        dest="external_start_scan",
        type=int,
        default=None,
        help=("Inject confirmed external starts after processing this 0-based scan."),
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
    # VS Code/Jupyter Interactive injects kernel args (for example --f=...).
    # Ignore unknown args so this script works both as CLI and in notebooks.
    args, _unknown = parser.parse_known_args()
    run_tomht(
        "crossing",
        use_initiator=args.births,
        use_initial_tracks=args.initial_tracks,
        external_start_scan=args.external_start_scan,
        external_start_delay_scans=args.external_start_delay_scans,
        debug_display_detections=args.debug_detections,
        debug_display_scan_stats=args.debug_scan_stats,
        debug_display_hypotheses=args.debug_hypotheses,
        debug_display_births=args.debug_births,
    )
