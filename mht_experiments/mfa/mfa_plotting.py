from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import State
from stonesoup.types.track import Track


def _plot_covar(
    state: State,
    ax: Axes,
    measurement_model: LinearGaussian,
    color: str | None = None,
) -> Ellipse:
    """Plot an error ellipse of the state's covariance in measurement space."""
    h_matrix = measurement_model.matrix()
    cov_m = h_matrix @ state.covar @ h_matrix.T
    eigenvalues, eigenvectors = np.linalg.eig(cov_m)
    max_ind = int(np.argmax(eigenvalues))
    min_ind = int(np.argmin(eigenvalues))
    orient = np.arctan2(eigenvectors[1, max_ind], eigenvectors[0, max_ind])

    ellipse = Ellipse(
        xy=state.state_vector[(0, 2), 0],
        width=2 * np.sqrt(eigenvalues[max_ind]),
        height=2 * np.sqrt(eigenvalues[min_ind]),
        angle=np.rad2deg(orient),
        alpha=0.2,
        color=color,
    )
    ax.add_artist(ellipse)
    return ellipse


def plot_mfa_component_tracks(
    tracks: Iterable[Track],
    ax: Axes,
    measurement_model: LinearGaussian,
    slide_window: int | None = None,
) -> list:
    """Plot MFA tracks including per-component covariance ellipses."""
    artists: list = []
    for plot_style, track in zip(("r-", "b-"), tracks):
        mini_tracks: list[list] = []
        history_window = (
            len(track) if slide_window is None else min(slide_window, len(track))
        )

        for component in track.state.components:
            child_tag = component.tag
            parents = []
            for j in range(1, history_window):
                parent = next(
                    comp
                    for comp in track.states[-(j + 1)].components
                    if comp.tag == child_tag[:-j]
                )
                parents.append(parent)
            parents.reverse()
            parents.append(component)
            mini_tracks.append(parents)

        drawn_states: set[object] = set()
        for mini_track in mini_tracks:
            states_to_plot = [
                state for state in mini_track if state not in drawn_states
            ]
            if len(states_to_plot) < len(mini_track):
                states_to_plot.insert(0, mini_track[-(len(states_to_plot) + 1)])

            artists.extend(
                ax.plot(
                    [state.state_vector[0, 0] for state in states_to_plot],
                    [state.state_vector[2, 0] for state in states_to_plot],
                    plot_style,
                )
            )
            for state in set(states_to_plot) - drawn_states:
                artists.append(_plot_covar(state, ax, measurement_model, plot_style[0]))
                drawn_states.add(state)

    return artists
