"""Policy seams for overload split triggering and link selection.

The current overload split policy is intentionally conservative and
experimental. The trigger compares the projected Cartesian leaf product against
``overload_split_projected_combination_threshold``. When a split is needed, the
link selector removes the weakest live-conflict edges first, where weakness is
the number of shared live conflict keys, with deterministic track-ID
tie-breaking. Both choices are expected policy seams for future work.
"""

from __future__ import annotations

from .tomht_clustering import (
    ClusterWorkItem,
    OverloadSplitRemovedEdge,
    canonical_edge_pair,
    connected_components_from_pairs,
)
from .tomht_cluster_overload_common import (
    _BinaryClusterSplit,
    _RecursiveSolveAccumulator,
)
from .tomht_model import DetectionKey
from .tomht_params import TOMHTParams


def _subcluster_from_track_ids(
    *,
    cluster: ClusterWorkItem,
    track_ids: tuple[int, ...],
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...],
) -> ClusterWorkItem:
    """Build an internal subcluster for recursive solving."""
    track_id_set = set(track_ids)
    return ClusterWorkItem(
        cluster_id=cluster.cluster_id,
        track_ids=track_ids,
        current_scan_det_keys_by_track_id={
            track_id: set(cluster.current_scan_det_keys_by_track_id[track_id])
            for track_id in track_ids
        },
        conflict_links=tuple(
            link
            for link in conflict_links
            if link[0] in track_id_set and link[1] in track_id_set
        ),
        overload_split_origin_cluster_id=None,
    )


def _partition_binary_subclusters(
    *,
    cluster: ClusterWorkItem,
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...],
    left_track_ids: tuple[int, ...],
    right_track_ids: tuple[int, ...],
    removed_edges: tuple[OverloadSplitRemovedEdge, ...],
) -> _BinaryClusterSplit:
    """Create left/right subclusters and collect all cross-cut live keys."""
    left_track_set = set(left_track_ids)
    right_track_set = set(right_track_ids)
    cut_keys: set[DetectionKey] = set()
    for left_track_id, right_track_id, shared_keys in conflict_links:
        if (left_track_id in left_track_set and right_track_id in right_track_set) or (
            left_track_id in right_track_set and right_track_id in left_track_set
        ):
            cut_keys |= set(shared_keys)

    return _BinaryClusterSplit(
        left_cluster=_subcluster_from_track_ids(
            cluster=cluster,
            track_ids=left_track_ids,
            conflict_links=conflict_links,
        ),
        right_cluster=_subcluster_from_track_ids(
            cluster=cluster,
            track_ids=right_track_ids,
            conflict_links=conflict_links,
        ),
        cut_keys=frozenset(cut_keys),
        removed_edges=removed_edges,
    )


def _choose_binary_overload_split(
    *,
    cluster: ClusterWorkItem,
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...],
    accumulator: _RecursiveSolveAccumulator,
    params: TOMHTParams,
) -> _BinaryClusterSplit | None:
    """Choose one deterministic binary split for an overloaded solve branch."""
    if len(cluster.track_ids) <= 1:
        return None

    remaining_edge_keys_by_pair: dict[tuple[int, int], tuple[DetectionKey, ...]] = {
        canonical_edge_pair(left_track_id, right_track_id): tuple(shared_keys)
        for left_track_id, right_track_id, shared_keys in conflict_links
    }
    components = connected_components_from_pairs(
        cluster.track_ids,
        remaining_edge_keys_by_pair.keys(),
    )
    if len(components) > 1:
        left_track_ids = components[0]
        right_track_ids = tuple(
            track_id for component in components[1:] for track_id in component
        )
        return _partition_binary_subclusters(
            cluster=cluster,
            conflict_links=conflict_links,
            left_track_ids=left_track_ids,
            right_track_ids=tuple(sorted(right_track_ids)),
            removed_edges=(),
        )

    if not remaining_edge_keys_by_pair:
        split_at = max(1, len(cluster.track_ids) // 2)
        return _partition_binary_subclusters(
            cluster=cluster,
            conflict_links=conflict_links,
            left_track_ids=tuple(cluster.track_ids[:split_at]),
            right_track_ids=tuple(cluster.track_ids[split_at:]),
            removed_edges=(),
        )

    max_removals = params.overload_split_max_edge_removals_per_cluster
    removed_edges: list[OverloadSplitRemovedEdge] = []
    while remaining_edge_keys_by_pair:
        if max_removals is not None and len(accumulator.removed_edges) + len(
            removed_edges
        ) >= int(max_removals):
            return None

        left_track_id, right_track_id = min(
            remaining_edge_keys_by_pair,
            key=lambda pair: (
                len(remaining_edge_keys_by_pair[pair]),
                int(pair[0]),
                int(pair[1]),
            ),
        )
        shared_keys = remaining_edge_keys_by_pair.pop((left_track_id, right_track_id))
        removed_edges.append(
            OverloadSplitRemovedEdge(
                left_track_id=left_track_id,
                right_track_id=right_track_id,
                shared_live_key_count=len(shared_keys),
            )
        )

        components = connected_components_from_pairs(
            cluster.track_ids,
            remaining_edge_keys_by_pair.keys(),
        )
        if len(components) <= 1:
            continue

        left_track_ids = components[0]
        right_track_ids = tuple(
            track_id for component in components[1:] for track_id in component
        )
        return _partition_binary_subclusters(
            cluster=cluster,
            conflict_links=conflict_links,
            left_track_ids=left_track_ids,
            right_track_ids=tuple(sorted(right_track_ids)),
            removed_edges=tuple(removed_edges),
        )

    return None


def _record_removed_edges_once(
    *,
    accumulator: _RecursiveSolveAccumulator,
    removed_edges: tuple[OverloadSplitRemovedEdge, ...],
) -> None:
    """Record unique removed-edge diagnostics across recursive branches."""
    for edge in removed_edges:
        edge_key = (
            int(edge.left_track_id),
            int(edge.right_track_id),
            int(edge.shared_live_key_count),
        )
        if edge_key in accumulator.removed_edge_keys:
            continue
        accumulator.removed_edge_keys.add(edge_key)
        accumulator.removed_edges.append(edge)


def _should_try_overload_split(
    *,
    params: TOMHTParams,
    projected_combinations: int,
) -> bool:
    """Return whether the projected solve size exceeds the overload threshold."""
    threshold = params.overload_split_projected_combination_threshold
    if not params.overload_split_enabled or threshold is None:
        return False
    threshold_int = int(threshold)
    if threshold_int <= 0:
        raise ValueError(
            "overload_split_projected_combination_threshold must be positive "
            "when overload splitting is enabled."
        )
    return int(projected_combinations) > threshold_int
