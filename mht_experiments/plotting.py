from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import State
from stonesoup.types.track import Track


def plot_covar(
    state: State,
    ax: Axes,
    measurement_model: LinearGaussian,
    color: Optional[str] = None,
) -> Ellipse:
    """Plot an error ellipse of the state's covariance in measurement space."""
    H = measurement_model.matrix()
    cov_m = H @ state.covar @ H.T
    w, v = np.linalg.eig(cov_m)
    max_ind = np.argmax(w)
    min_ind = np.argmin(w)
    orient = np.arctan2(v[1, max_ind], v[0, max_ind])

    ellipse = Ellipse(
        xy=state.state_vector[(0, 2), 0],
        width=2 * np.sqrt(w[max_ind]),
        height=2 * np.sqrt(w[min_ind]),
        angle=np.rad2deg(orient),
        alpha=0.2,
        color=color,
    )
    ax.add_artist(ellipse)
    return ellipse


def plot_tracks(
    tracks: Iterable[Track],
    ax: Axes,
    measurement_model: LinearGaussian,
    slide_window: Optional[int] = None,
) -> List:
    """Plot MFA tracks, including covariance ellipses for each Gaussian component.

    This assumes each track state is a GaussianMixture* with 'components' attribute
    whose elements have a 'tag' indicating the assignment history, as in the MFA example.
    """
    artists: List = []
    # Simple cycle of styles; the MFA example assumes exactly 2 tracks
    for plot_style, track in zip(("r-", "b-"), tracks):
        mini_tracks = []

        if slide_window is None or slide_window > len(track):
            hist_window = len(track)
        else:
            hist_window = slide_window

        # Each mixture component corresponds to a specific assignment history (tag)
        for component in track.state.components:
            child_tag = component.tag
            parents = []
            for j in range(1, hist_window):
                parent = next(
                    comp
                    for comp in track.states[-(j + 1)].components
                    if comp.tag == child_tag[:-j]
                )
                parents.append(parent)
            parents.reverse()
            parents.append(component)
            mini_tracks.append(parents)

        drawn_states = set()
        for mini_track in mini_tracks:
            # Avoid re-plotting drawn trajectory
            states_to_plot = [s for s in mini_track if s not in drawn_states]
            if len(states_to_plot) < len(mini_track):
                # Insert the state before so the line is continuous
                states_to_plot.insert(0, mini_track[-(len(states_to_plot) + 1)])

            artists.extend(
                ax.plot(
                    [s.state_vector[0, 0] for s in states_to_plot],
                    [s.state_vector[2, 0] for s in states_to_plot],
                    plot_style,
                )
            )

            # Avoid re-plotting drawn ellipses
            for state in set(states_to_plot) - drawn_states:
                artists.append(
                    plot_covar(state, ax, measurement_model, plot_style[0])
                )
                drawn_states.add(state)

    return artists
