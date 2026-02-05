from __future__ import annotations

import argparse

from mht_experiments.runners.tomht_runner import run_tomht

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TO-MHT bearing/range scenario.")
    parser.add_argument(
        "--scoring-mode",
        choices=["beta_ratio", "legacy"],
        default=None,
        help="Override scoring mode (default: tracker default).",
    )
    parser.add_argument(
        "--births",
        dest="births",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable initiator/births (default: on for bearing_range).",
    )
    parser.add_argument(
        "--initial-tracks",
        dest="initial_tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable scenario initial tracks (default: off for bearing_range).",
    )
    args = parser.parse_args()
    run_tomht(
        "bearing_range",
        scoring_mode=args.scoring_mode,
        use_initiator=args.births,
        use_initial_tracks=args.initial_tracks,
    )
