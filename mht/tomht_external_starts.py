"""External-start insertion helpers for the track-oriented TOMHT tracker."""

from __future__ import annotations

from dataclasses import dataclass
import datetime

from stonesoup.types.track import Track

from .tomht_model import GlobalHypothesis, TrackHypothesisNode
from .tomht_scoring import _existence_probability_to_log_odds
from .tomht_tree_store import TrackTreeStore


@dataclass(frozen=True)
class ExternalStartInsertionResult:
    """External-start side effects plus the updated MAP view."""

    map_global: GlobalHypothesis
    new_roots: list[TrackHypothesisNode]


def validate_external_starts_timestamp(
    *,
    time: datetime.datetime,
    last_update_timestamp: datetime.datetime | None,
    last_scan_index: int | None,
) -> None:
    """Validate add_external_starts(...) ordering and timestamp match."""
    if not isinstance(time, datetime.datetime):
        raise TypeError(
            "add_external_starts() time must be a datetime.datetime instance."
        )
    if last_update_timestamp is None or last_scan_index is None:
        raise RuntimeError(
            "add_external_starts() requires a completed update_tracker() first."
        )
    if time != last_update_timestamp:
        raise ValueError(
            "add_external_starts() time must match the most recent "
            f"completed update_tracker() timestamp. Expected {last_update_timestamp!r}, "
            f"got {time!r}."
        )


def external_start_initial_log_delta(
    start: Track,
    *,
    default_log_delta: float,
) -> float:
    """Return the initial log-odds score for one externally confirmed start."""
    metadata_existence_probability = start.metadata.get("existence_probability")
    if metadata_existence_probability is None:
        return default_log_delta
    try:
        return _existence_probability_to_log_odds(
            metadata_existence_probability,
            parameter_name="external start metadata['existence_probability']",
        )
    except ValueError:
        return default_log_delta


def make_external_start_root(
    *,
    start: Track,
    time: datetime.datetime,
    tree_store: TrackTreeStore,
    last_scan_index: int | None,
    external_start_default_log_delta: float,
    assoc_pad_label: int,
) -> TrackHypothesisNode:
    """Convert one confirmed external start Track into an inserted root node."""
    if len(start) == 0:
        raise ValueError(
            "External starts must contain at least one state at the current timestamp."
        )

    start_timestamp = getattr(start.states[-1], "timestamp", None)
    if start_timestamp != time:
        raise ValueError(
            "External starts must already be initialised at the supplied "
            f"timestamp. Expected {time!r}, got {start_timestamp!r}."
        )

    age = max(int(start.metadata.get("age", len(start))), 1)
    hits = int(start.metadata.get("hits", age))
    hits = min(max(hits, 1), age)
    state = start.states[-1]
    if last_scan_index is None:
        raise RuntimeError(
            "External starts require at least one completed update_tracker() call."
        )
    log_delta = external_start_initial_log_delta(
        start,
        default_log_delta=external_start_default_log_delta,
    )

    return tree_store.create_root_tree_for_new_track(
        scan_index=int(last_scan_index),
        timestamp=getattr(state, "timestamp", time),
        state=state,
        state_kind="external_start",
        used_det_key=None,
        assoc_label=assoc_pad_label,
        log_delta=log_delta,
        age=age,
        hits=hits,
        root_source="external_start",
    )


def insert_external_start_trees(
    *,
    time: datetime.datetime,
    starts: list[Track],
    tree_store: TrackTreeStore,
    last_scan_index: int | None,
    last_map_global: GlobalHypothesis,
    external_start_default_log_delta: float,
    assoc_pad_label: int,
) -> ExternalStartInsertionResult:
    """Insert external-start roots and return the updated full-scan MAP view."""
    new_roots = [
        make_external_start_root(
            start=start,
            time=time,
            tree_store=tree_store,
            last_scan_index=last_scan_index,
            external_start_default_log_delta=external_start_default_log_delta,
            assoc_pad_label=assoc_pad_label,
        )
        for start in starts
    ]

    # External starts are assumed to be from currently unused detections, so add
    # them directly to the last MAP view.
    merged = dict(last_map_global.leaf_nodes_by_track_id)
    for track_id, tree in tree_store.track_trees_by_track_id.items():
        if track_id in merged:
            continue
        if len(tree.active_leaf_node_ids) != 1:
            continue
        only_leaf_id = next(iter(tree.active_leaf_node_ids))
        merged[track_id] = tree_store.nodes_by_id[only_leaf_id]

    map_global = GlobalHypothesis(
        leaf_nodes_by_track_id=merged,
        log_weight=float(last_map_global.log_weight)
        + sum(float(root.log_delta) for root in new_roots),
    )
    return ExternalStartInsertionResult(
        map_global=map_global,
        new_roots=new_roots,
    )
