from __future__ import annotations

import datetime
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

from stonesoup.plotter import Plotter
from stonesoup.types.update import GaussianMixtureUpdate

from mht.mfa.mfa_plotting import plot_mfa_component_tracks
from mht.mfa.mfa_scenarios import (
    initial_mfa_tracks_for_bearing_range,
    initial_mfa_tracks_for_crossing,
)
from mht.mfa.mfa_tracker import (
    build_mfa_components_linear,
    build_mfa_components_ukf,
)
from mht.helpers.plotting import plot_tracks_stable_xy
from mht.scenarios.bearing_range import (
    create_bearing_range_mht_example,
)
from mht.scenarios.crossing_targets import create_crossing_scenario

SetupName = Literal["crossing", "bearing_range"]


def _running_in_ipython_kernel() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _show_animation(ani) -> None:
    """Show animation nicely in VS Code Interactive/Jupyter, otherwise use plt.show()."""
    if _running_in_ipython_kernel():
        mpl.rcParams["animation.html"] = "jshtml"
        mpl.rcParams["animation.embed_limit"] = 100
        from IPython.display import HTML, display  # type: ignore

        display(HTML(ani.to_jshtml()))
        plt.close(ani._fig)
    else:
        plt.show()


def run_mfa(setup: SetupName, *, show_components: bool = False) -> None:
    """
    Run MFA-based multi-hypothesis tracking in one of two setups.

    - "crossing": linear measurement model + KF, supports component-branch plotting
    - "bearing_range": nonlinear bearing-range + UKF, uses normal track plotting
    """
    if setup == "crossing":
        truths, scans, start_time, transition_model, measurement_model, config = (
            create_crossing_scenario()
        )
        timestamps = [
            start_time + datetime.timedelta(seconds=i) for i in range(len(scans))
        ]
        tracks = initial_mfa_tracks_for_crossing(start_time)
        components = build_mfa_components_linear(
            transition_model, measurement_model, config
        )
        can_show_components = True
    elif setup == "bearing_range":
        truths, scans, timestamps, transition_model, measurement_model, config = (
            create_bearing_range_mht_example()
        )
        tracks = initial_mfa_tracks_for_bearing_range(timestamps[0])
        components = build_mfa_components_ukf(
            transition_model, measurement_model, config
        )
        can_show_components = False
    else:
        raise ValueError(f"Unknown setup: {setup}")

    plotter = Plotter()
    frames: list[list] = []

    for n, (timestamp, detections) in enumerate(zip(timestamps, scans)):
        artists: list = []

        associations = components.data_associator.associate(
            tracks, detections, timestamp
        )

        for track, hypotheses in associations.items():
            mixture_components = []
            for hypothesis in hypotheses:
                if not hypothesis:
                    mixture_components.append(hypothesis.prediction)
                else:
                    mixture_components.append(components.updater.update(hypothesis))
            track.append(
                GaussianMixtureUpdate(
                    components=mixture_components, hypothesis=hypotheses
                )
            )

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

        if show_components and can_show_components:
            artists.extend(
                plot_mfa_component_tracks(
                    tracks,
                    ax,
                    measurement_model=measurement_model,
                    slide_window=config.slide_window,
                )
            )
        else:
            # Replicate Stone Soup originals:
            # - Crossing example: red/blue (and more if needed)
            # - Bearing-range example: all green
            if setup == "bearing_range":
                artists.extend(plot_tracks_stable_xy(tracks, ax, styles=("g-",)))
            else:
                artists.extend(plot_tracks_stable_xy(tracks, ax, styles=("r-", "b-")))

        frames.append(artists)

    ani = animation.ArtistAnimation(
        plotter.fig, frames, interval=400, blit=True, repeat_delay=1000
    )
    _show_animation(ani)
