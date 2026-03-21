from __future__ import annotations

from itertools import cycle
from matplotlib.axes import Axes
from typing import Iterable
from stonesoup.types.track import Track


def plot_tracks_stable_xy(
    tracks: Iterable[Track],
    ax: Axes,
    *,
    mapping: tuple[int, int] = (0, 2),
    styles: tuple[str, ...] = ("r-", "b-", "g-", "m-", "c-", "y-", "k-"),
):
    """Plot track means with stable colours/styles independent of container iteration order."""
    artists = []

    def sort_key(tr: Track) -> int:
        tid = tr.metadata.get("track_id", None)
        return int(tid) if tid is not None else id(tr)

    ordered_tracks = sorted(tracks, key=sort_key)

    style_iter = cycle(styles)
    for track in ordered_tracks:
        style = next(style_iter)
        xs = [state.state_vector[mapping[0], 0] for state in track]
        ys = [state.state_vector[mapping[1], 0] for state in track]
        artists.extend(ax.plot(xs, ys, style))

    return artists
