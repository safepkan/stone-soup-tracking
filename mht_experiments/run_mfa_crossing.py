from __future__ import annotations

import datetime
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from ordered_set import OrderedSet

from stonesoup.plotter import Plotter
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianMixtureUpdate

from mht_experiments.plotting import plot_tracks
from mht_experiments.scenarios.crossing_targets import (
    ScenarioConfig,
    create_crossing_scenario,
)
from mht_experiments.trackers.mfa_tracker import build_mfa_components


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
        # Render with JS controls (play/pause/slider) inline
        mpl.rcParams["animation.html"] = "jshtml"
        # If your animation is large, you may need to bump this (MB)
        mpl.rcParams["animation.embed_limit"] = 100

        from IPython.display import HTML, display  # type: ignore

        display(HTML(ani.to_jshtml()))
        plt.close(ani._fig)  # avoid a duplicate static figure output
    else:
        plt.show()


def _initialise_tracks(
    start_time: datetime.datetime,
) -> OrderedSet[Track]:
    """Create the two initial Gaussian-mixture priors, as in the MFA example."""
    from ordered_set import OrderedSet as OS  # avoid shadowing

    prior1 = GaussianMixture(
        [
            TaggedWeightedGaussianState(
                [[0.0], [1.0], [0.0], [1.0]],
                np.diag([1.5, 0.5, 1.5, 0.5]),
                timestamp=start_time,
                weight=Probability(1.0),
                tag=[],
            )
        ]
    )

    prior2 = GaussianMixture(
        [
            TaggedWeightedGaussianState(
                [[0.0], [1.0], [20.0], [-1.0]],
                np.diag([1.5, 0.5, 1.5, 0.5]),
                timestamp=start_time,
                weight=Probability(1.0),
                tag=[],
            )
        ]
    )

    tracks: OrderedSet[Track] = OS((Track([prior1]), Track([prior2])))
    return tracks


def main() -> None:
    # --- Build scenario ---
    (
        truths,
        scans,
        start_time,
        transition_model,
        measurement_model,
        config,
    ) = create_crossing_scenario()

    # --- Build MFA components ---
    components = build_mfa_components(transition_model, measurement_model, config)

    # --- Initialise tracks ---
    tracks = _initialise_tracks(start_time)

    # --- Plotter & animation setup ---
    plotter = Plotter()
    frames: List[list] = []

    for n, detections in enumerate(scans):
        artists: List = []

        timestamp = start_time + datetime.timedelta(seconds=n)

        # Associate using MFA
        associations = components.data_associator.associate(
            tracks, detections, timestamp
        )

        # For each track, build a GaussianMixtureUpdate from the hypotheses
        for track, hypotheses in associations.items():
            mixture_components = []
            for hypothesis in hypotheses:
                if not hypothesis:
                    # Missed detection: use prediction directly
                    mixture_components.append(hypothesis.prediction)
                else:
                    update = components.updater.update(hypothesis)
                    mixture_components.append(update)

            track.append(
                GaussianMixtureUpdate(
                    components=mixture_components,
                    hypothesis=hypotheses,
                )
            )

        # Axes labels and bounds
        ax = plotter.ax
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(*config.v_bounds[0])
        ax.set_ylim(*config.v_bounds[1])

        # Plot ground truth up to current time
        artists.extend(
            plotter.plot_ground_truths(
                [truth[: n + 1] for truth in truths],
                mapping=[0, 2],
            )
        )

        # Plot detections
        artists.extend(
            plotter.plot_measurements(
                detections,
                mapping=[0, 2],
                measurement_model=measurement_model,
            )
        )

        # Plot tracks (MFA mixture components)
        artists.extend(
            plot_tracks(
                tracks,
                ax,
                measurement_model=measurement_model,
                slide_window=config.slide_window,
            )
        )

        frames.append(artists)

    # Build and display animation.
    ani = animation.ArtistAnimation(
        plotter.fig,
        frames,
        interval=400,
        blit=True,
        repeat_delay=1000,
    )

    _show_animation(ani)


if __name__ == "__main__":
    main()

# This file is based on the MFA example from the Stonesoup repository.
# https://stonesoup.readthedocs.io/en/v1.4/auto_examples/dataassociation/MFA_example.html
# 
# References
# ----------
# .. [#] Xia, Y., Granström, K., Svensson, L., García-Fernández, Á.F., and Williams, J.L.,
#        2019. Multiscan Implementation of the Trajectory Poisson Multi-Bernoulli Mixture Filter.
#        J. Adv. Information Fusion, 14(2), pp. 213–235.
