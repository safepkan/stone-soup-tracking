from __future__ import annotations

import datetime
from typing import List, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from stonesoup.plotter import Plotter
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from mht_experiments.plotting import plot_tracks_stable_xy
from mht_experiments.scenarios.bearing_range_mht_example import create_bearing_range_mht_example
from mht_experiments.scenarios.crossing_targets import create_crossing_scenario
from mht_experiments.trackers.tomht_tracker import TOMHTParams, build_tomht_linear, build_tomht_ukf

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


def _initial_tracks_crossing(start_time: datetime.datetime) -> list[Track]:
    # Matches the MFA crossing priors, but as GaussianState (not mixture)
    cov = CovarianceMatrix(np.diag([1.5, 0.5, 1.5, 0.5]))
    s1 = GaussianState(StateVector([[0.0], [1.0], [0.0], [1.0]]), covar=cov, timestamp=start_time)
    s2 = GaussianState(StateVector([[0.0], [1.0], [20.0], [-1.0]]), covar=cov, timestamp=start_time)
    return [Track([s1]), Track([s2])]


def _initial_tracks_bearing_range(start_time: datetime.datetime) -> list[Track]:
    # Matches Stone Soup “general MHT” example priors, but as GaussianState
    cov = CovarianceMatrix(np.diag([10.0, 1.0, 10.0, 1.0]))
    priors = [
        StateVector([[10.0], [1.0], [10.0], [1.0]]),
        StateVector([[-10.0], [-1.0], [-10.0], [-1.0]]),
        StateVector([[-10.0], [-1.0], [10.0], [1.0]]),
    ]
    return [Track([GaussianState(p, covar=cov, timestamp=start_time)]) for p in priors]


def run_tomht(setup: SetupName) -> None:
    if setup == "crossing":
        truths, scans, start_time, transition_model, measurement_model, config = create_crossing_scenario()
        timestamps = [start_time + datetime.timedelta(seconds=i) for i in range(len(scans))]
        tracks = _initial_tracks_crossing(start_time)
        tracker = build_tomht_linear(
            transition_model,
            measurement_model,
            prob_detect=config.prob_detect,
            prob_gate=config.prob_gate,
            clutter_density=config.clutter_density,
            tracks=tracks,
            params=TOMHTParams(max_children_per_track=5, max_missed=5),
        )
        styles = ("r-", "b-")
    else:
        truths, scans, timestamps, transition_model, measurement_model, config = create_bearing_range_mht_example()
        tracks = _initial_tracks_bearing_range(timestamps[0])
        tracker = build_tomht_ukf(
            transition_model,
            measurement_model,
            prob_detect=config.prob_detect,
            prob_gate=config.prob_gate,
            clutter_density=config.clutter_density,
            tracks=tracks,
            params=TOMHTParams(max_children_per_track=5, max_missed=5),
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

        artists.extend(plotter.plot_ground_truths([t[: n + 1] for t in truths], mapping=[0, 2]))
        artists.extend(plotter.plot_measurements(detections, mapping=[0, 2], measurement_model=measurement_model))
        artists.extend(plot_tracks_stable_xy(tracks_out, ax, styles=styles))

        frames.append(artists)

    ani = animation.ArtistAnimation(plotter.fig, frames, interval=400, blit=True, repeat_delay=1000)
    _show_animation(ani)
