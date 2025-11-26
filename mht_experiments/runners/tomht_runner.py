from __future__ import annotations

import datetime
from typing import List, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

from stonesoup.plotter import Plotter

from mht_experiments.plotting import plot_tracks_stable_xy
from mht_experiments.scenarios.bearing_range_mht_example import (
    create_bearing_range_mht_example,
    initial_tomht_tracks_for_bearing_range,
    tomht_initiator_for_bearing_range,
)
from mht_experiments.scenarios.crossing_targets import (
    create_crossing_scenario,
    initial_tomht_tracks_for_crossing,
    tomht_initiator_for_crossing_simple,
)
from mht_experiments.trackers.tomht_tracker import (
    TOMHTParams,
    build_tomht_linear,
    build_tomht_ukf,
)


SetupName = Literal["crossing", "bearing_range"]


def _running_in_ipython_kernel() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _show_animation(ani) -> None:
    if _running_in_ipython_kernel():
        mpl.rcParams["animation.html"] = "jshtml"
        mpl.rcParams["animation.embed_limit"] = 100
        from IPython.display import HTML, display  # type: ignore

        display(HTML(ani.to_jshtml()))
        plt.close(ani._fig)
    else:
        plt.show()


def run_tomht(setup: SetupName) -> None:
    styles: tuple[str, ...]
    if setup == "crossing":
        truths, scans, start_time, transition_model, measurement_model, config = (
            create_crossing_scenario()
        )
        timestamps = [
            start_time + datetime.timedelta(seconds=i) for i in range(len(scans))
        ]
        tracks = initial_tomht_tracks_for_crossing(start_time)
        initiator = tomht_initiator_for_crossing_simple(start_time, measurement_model)
        initiator = None  # Skip births
        # tracks = []  # No initial tracks
        tracker = build_tomht_linear(
            transition_model,
            measurement_model,
            prob_detect=config.prob_detect,
            clutter_density=config.clutter_density,
            tracks=tracks,
            initiator=initiator,
            params=TOMHTParams(
                max_children_per_track=5,
                max_missed=5,
                prob_gate=0.9999,
                birth_log_penalty=15.0,
            ),
        )
        styles = ("r-", "b-")
    else:
        truths, scans, timestamps, transition_model, measurement_model, config = (
            create_bearing_range_mht_example()
        )
        tracks = initial_tomht_tracks_for_bearing_range(timestamps[0])
        initiator = tomht_initiator_for_bearing_range(
            timestamps[0], transition_model, measurement_model
        )
        # initiator = None  # Skip births
        tracks = []  # No initial tracks
        tracker = build_tomht_ukf(
            transition_model,
            measurement_model,
            prob_detect=config.prob_detect,
            clutter_density=config.clutter_density,
            tracks=tracks,
            initiator=initiator,
            params=TOMHTParams(
                max_global_hypotheses=10,
                max_children_per_track=3,
                max_missed=5,
                max_births_per_scan=2,
                birth_log_penalty=2.0,
                unused_det_log_penalty=4.0,
                prob_gate=0.99,
            ),
        )
        styles = ("g-",)

    plotter = Plotter()
    frames: List[list] = []

    for n, (timestamp, detections) in enumerate(zip(timestamps, scans)):
        artists: List = []

        tracks_out = tracker.step(detections, timestamp)

        ax = plotter.ax
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(*config.v_bounds[0])
        ax.set_ylim(*config.v_bounds[1])

        artists.extend(
            plotter.plot_ground_truths([t[: n + 1] for t in truths], mapping=[0, 2])
        )
        artists.extend(
            plotter.plot_measurements(
                detections, mapping=[0, 2], measurement_model=measurement_model
            )
        )
        artists.extend(plot_tracks_stable_xy(tracks_out, ax, styles=styles))

        frames.append(artists)

    ani = animation.ArtistAnimation(
        plotter.fig, frames, interval=400, blit=True, repeat_delay=1000
    )
    _show_animation(ani)
