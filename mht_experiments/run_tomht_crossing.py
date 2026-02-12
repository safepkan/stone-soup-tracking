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
    # VS Code/Jupyter Interactive injects kernel args (for example --f=...).
    # Ignore unknown args so this script works both as CLI and in notebooks.
    args, _unknown = parser.parse_known_args()
    run_tomht(
        "crossing",
        use_initiator=args.births,
        use_initial_tracks=args.initial_tracks,
    )
