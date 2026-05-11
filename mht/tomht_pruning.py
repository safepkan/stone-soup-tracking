"""Pruning helpers for the track-oriented MHT tracker."""

from __future__ import annotations

from .tomht_model import ClusterRebuildSnapshot, TrackTree


def supported_leaf_ids_by_track_from_rebuilt_globals(
    snapshot: ClusterRebuildSnapshot,
) -> dict[int, set[int]]:
    """Collect cluster leaf IDs that appear in at least one retained rebuilt global."""
    supported_by_track_id: dict[int, set[int]] = {
        track_id: set() for track_id in snapshot.track_ids
    }
    for rebuilt_global in snapshot.rebuilt_globals:
        for track_id, leaf_node in rebuilt_global.leaf_nodes_by_track_id.items():
            if track_id in supported_by_track_id:
                supported_by_track_id[track_id].add(int(leaf_node.node_id))
    return supported_by_track_id


def apply_post_solve_supported_leaf_pruning(
    *,
    cluster_snapshots: list[ClusterRebuildSnapshot],
    track_trees_by_track_id: dict[int, TrackTree],
) -> None:
    """Prune each cluster tree to leaves supported by retained rebuilt globals."""
    for snapshot in cluster_snapshots:
        # Overload-decomposed clusters are approximate; keep their current
        # frontiers to avoid over-pruning branches that may be needed once
        # severed weak links reconnect in later scans.
        #
        # TODO: Revisit this policy when overload-split semantics are reviewed.
        if snapshot.overload_split_origin_cluster_id is not None:
            continue

        # Keep k=0 behavior non-destructive for compatibility/debug edge cases.
        if not snapshot.rebuilt_globals:
            continue

        supported_by_track_id = supported_leaf_ids_by_track_from_rebuilt_globals(
            snapshot
        )
        for track_id in snapshot.track_ids:
            tree = track_trees_by_track_id.get(track_id)
            if tree is None:
                continue
            supported_leaf_ids = supported_by_track_id.get(track_id, set())
            if not supported_leaf_ids:
                raise RuntimeError(
                    "Post-solve supported-leaf pruning found no retained leaves "
                    f"for cluster={snapshot.cluster_id} track_id={track_id}."
                )
            tree.active_leaf_node_ids = set(supported_leaf_ids)
