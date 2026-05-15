"""Pruning helpers for the track-oriented MHT tracker."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .tomht_model import (
    ClusterRebuildSnapshot,
    GlobalHypothesis,
    NScanCommitmentSnapshot,
    TrackHypothesisNode,
)
from .tomht_tree_store import TrackTreeStore
from .tomht_tree_utils import child_of_root_on_path, is_descendant_of


@dataclass(frozen=True)
class MapNScanPruningPlan:
    """Planned MAP-only N-scan choices and diagnostics before mutation."""

    boundary_scan_index: int
    root_before_by_track_id: dict[int, TrackHypothesisNode]
    map_choice_by_track_id: dict[int, int]
    disagreement_total: int
    updated_snapshots: list[ClusterRebuildSnapshot]


@dataclass(frozen=True)
class MapNScanPruningResult:
    """Result of applying MAP-only N-scan pruning to explicit trees."""

    boundary_scan_index: int
    tracks_in_scope: int
    committed_count: int
    disagreement_total: int
    updated_snapshots: list[ClusterRebuildSnapshot]
    nscan_commitment_snapshot: NScanCommitmentSnapshot


@dataclass(frozen=True)
class SupportedLeafPruningStats:
    """Counters from post-solve retained-global supported-leaf pruning."""

    unsupported_leaf_count_pruned: int = 0
    overload_split_unsupported_leaf_count_pruned: int = 0
    overload_split_clusters_skipped_supported_pruning: int = 0
    overload_split_trees_skipped_supported_pruning: int = 0
    overload_split_leaves_skipped_supported_pruning: int = 0


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
    tree_store: TrackTreeStore,
    overload_split_supported_pruning_policy: str = "skip",
) -> SupportedLeafPruningStats:
    """Prune each cluster tree to leaves supported by retained rebuilt globals."""
    if overload_split_supported_pruning_policy not in {"skip", "apply"}:
        raise ValueError(
            "overload_split_supported_pruning_policy must be 'skip' or 'apply'; "
            f"got {overload_split_supported_pruning_policy!r}."
        )

    track_trees_by_track_id = tree_store.track_trees_by_track_id
    unsupported_leaf_count_pruned = 0
    overload_split_unsupported_leaf_count_pruned = 0
    overload_split_clusters_skipped_supported_pruning = 0
    overload_split_trees_skipped_supported_pruning = 0
    overload_split_leaves_skipped_supported_pruning = 0

    for snapshot in cluster_snapshots:
        is_overload_split = snapshot.overload_split_origin_cluster_id is not None
        # Overload-decomposed clusters are approximate. The default policy keeps
        # their current frontiers; the experimental "apply" policy uses the same
        # retained-global support rule as normal clusters.
        if is_overload_split and overload_split_supported_pruning_policy == "skip":
            overload_split_clusters_skipped_supported_pruning += 1
            for track_id in snapshot.track_ids:
                tree = track_trees_by_track_id.get(track_id)
                if tree is None:
                    continue
                overload_split_trees_skipped_supported_pruning += 1
                overload_split_leaves_skipped_supported_pruning += len(
                    tree.active_leaf_node_ids
                )
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
            pruned_leaf_count = len(tree.active_leaf_node_ids - supported_leaf_ids)
            unsupported_leaf_count_pruned += pruned_leaf_count
            if is_overload_split:
                overload_split_unsupported_leaf_count_pruned += pruned_leaf_count
            tree.active_leaf_node_ids = set(supported_leaf_ids)

    return SupportedLeafPruningStats(
        unsupported_leaf_count_pruned=unsupported_leaf_count_pruned,
        overload_split_unsupported_leaf_count_pruned=(
            overload_split_unsupported_leaf_count_pruned
        ),
        overload_split_clusters_skipped_supported_pruning=(
            overload_split_clusters_skipped_supported_pruning
        ),
        overload_split_trees_skipped_supported_pruning=(
            overload_split_trees_skipped_supported_pruning
        ),
        overload_split_leaves_skipped_supported_pruning=(
            overload_split_leaves_skipped_supported_pruning
        ),
    )


def compute_cluster_pruning_disagreement(
    *,
    snapshot: ClusterRebuildSnapshot,
    root_before_by_track_id: dict[int, TrackHypothesisNode],
    map_choice_by_track_id: dict[int, int],
) -> tuple[dict[int, int], int]:
    """Compare MAP pruning child choices against alternative rebuilt globals."""
    tracks_in_cluster = list(snapshot.track_ids)
    map_choice_for_cluster = {
        track_id: map_choice_by_track_id[track_id]
        for track_id in tracks_in_cluster
        if track_id in map_choice_by_track_id
    }
    if not map_choice_for_cluster:
        return {}, 0

    disagreement_count = 0
    for alternative in snapshot.rebuilt_globals[1:]:
        disagrees = False
        for track_id, map_child_id in map_choice_for_cluster.items():
            leaf = alternative.leaf_nodes_by_track_id.get(track_id)
            if leaf is None:
                continue
            root_before = root_before_by_track_id[track_id]
            alt_child = child_of_root_on_path(root=root_before, leaf=leaf)
            alt_child_id = None if alt_child is None else alt_child.node_id
            if alt_child_id != map_child_id:
                disagrees = True
                break
        if disagrees:
            disagreement_count += 1

    return map_choice_for_cluster, disagreement_count


def annotate_cluster_snapshots_with_map_pruning_disagreement(
    *,
    cluster_snapshots: list[ClusterRebuildSnapshot],
    root_before_by_track_id: dict[int, TrackHypothesisNode],
    map_choice_by_track_id: dict[int, int],
) -> tuple[list[ClusterRebuildSnapshot], int]:
    """Attach per-cluster MAP pruning choices and disagreement diagnostics."""
    updated_snapshots: list[ClusterRebuildSnapshot] = []
    disagreement_total = 0
    for snapshot in cluster_snapshots:
        map_choice_for_cluster, disagreement_count = (
            compute_cluster_pruning_disagreement(
                snapshot=snapshot,
                root_before_by_track_id=root_before_by_track_id,
                map_choice_by_track_id=map_choice_by_track_id,
            )
        )
        disagreement_total += disagreement_count
        updated_snapshots.append(
            replace(
                snapshot,
                map_pruning_child_by_track_id=map_choice_for_cluster,
                disagreement_count=disagreement_count,
            )
        )
    return updated_snapshots, disagreement_total


def plan_map_n_scan_pruning(
    *,
    boundary_scan_index: int,
    map_global: GlobalHypothesis,
    cluster_snapshots: list[ClusterRebuildSnapshot],
    tree_store: TrackTreeStore,
) -> MapNScanPruningPlan:
    """Plan MAP child commits and disagreement diagnostics without mutation."""
    root_before_by_track_id: dict[int, TrackHypothesisNode] = {
        track_id: tree_store.nodes_by_id[tree.root_node_id]
        for track_id, tree in tree_store.track_trees_by_track_id.items()
    }

    map_choice_by_track_id: dict[int, int] = {}
    for track_id, tree in sorted(tree_store.track_trees_by_track_id.items()):
        root_before = root_before_by_track_id[track_id]
        if int(root_before.scan_index) >= boundary_scan_index:
            continue
        map_leaf = map_global.leaf_nodes_by_track_id.get(track_id)
        if map_leaf is None:
            continue
        child = child_of_root_on_path(root=root_before, leaf=map_leaf)
        if child is None:
            continue
        map_choice_by_track_id[track_id] = child.node_id

    updated_snapshots, disagreement_total = (
        annotate_cluster_snapshots_with_map_pruning_disagreement(
            cluster_snapshots=cluster_snapshots,
            root_before_by_track_id=root_before_by_track_id,
            map_choice_by_track_id=map_choice_by_track_id,
        )
    )
    return MapNScanPruningPlan(
        boundary_scan_index=boundary_scan_index,
        root_before_by_track_id=root_before_by_track_id,
        map_choice_by_track_id=map_choice_by_track_id,
        disagreement_total=disagreement_total,
        updated_snapshots=updated_snapshots,
    )


def apply_planned_map_n_scan_pruning(
    *,
    plan: MapNScanPruningPlan,
    tree_store: TrackTreeStore,
    nscan_commitment_snapshot: NScanCommitmentSnapshot,
) -> tuple[int, NScanCommitmentSnapshot]:
    """Apply one precomputed N-scan pruning plan to trees and bookkeeping."""
    committed_count = 0
    latest_committed_ancestor_by_track_id = dict(
        nscan_commitment_snapshot.latest_committed_ancestor_by_track_id
    )
    committed_boundary_by_track_id = dict(
        nscan_commitment_snapshot.committed_boundary_by_track_id
    )
    committed_ancestor_by_track_id = dict(
        nscan_commitment_snapshot.committed_ancestor_by_track_id
    )
    for track_id, chosen_child_id in plan.map_choice_by_track_id.items():
        current_tree = tree_store.track_trees_by_track_id.get(track_id)
        if current_tree is None:
            continue
        root_before = plan.root_before_by_track_id[track_id]
        chosen_child = tree_store.nodes_by_id[chosen_child_id]

        # Preserve committed output prefix strictly before the new unresolved root.
        current_tree.committed_states.append(root_before.state)

        current_tree.root_node_id = chosen_child.node_id
        chosen_child.parent = None

        retained_leaf_ids = {
            leaf_id
            for leaf_id in current_tree.active_leaf_node_ids
            if is_descendant_of(
                node=tree_store.nodes_by_id[leaf_id],
                ancestor=chosen_child,
            )
        }
        if not retained_leaf_ids:
            retained_leaf_ids = {chosen_child.node_id}
        current_tree.active_leaf_node_ids = retained_leaf_ids

        latest_committed_ancestor_by_track_id[track_id] = chosen_child
        prev_boundary = committed_boundary_by_track_id.get(track_id)
        if prev_boundary is None or plan.boundary_scan_index > prev_boundary:
            committed_boundary_by_track_id[track_id] = plan.boundary_scan_index
            committed_ancestor_by_track_id[track_id] = chosen_child
        committed_count += 1

        # Root promotion detaches the old root lineage from this tree.
        root_before.child_node_ids = {
            child_id
            for child_id in root_before.child_node_ids
            if child_id == chosen_child_id
        }

    updated_snapshot = replace(
        nscan_commitment_snapshot,
        latest_committed_ancestor_by_track_id=latest_committed_ancestor_by_track_id,
        committed_boundary_by_track_id=committed_boundary_by_track_id,
        committed_ancestor_by_track_id=committed_ancestor_by_track_id,
    )
    tree_store.remove_empty_trees()
    return committed_count, updated_snapshot


def apply_map_n_scan_pruning(
    *,
    scan_index: int,
    ns_scan_window: int,
    map_global: GlobalHypothesis,
    cluster_snapshots: list[ClusterRebuildSnapshot],
    tree_store: TrackTreeStore,
    nscan_commitment_snapshot: NScanCommitmentSnapshot,
) -> MapNScanPruningResult:
    """Apply MAP-only N-scan root-child promotion and disagreement bookkeeping."""
    boundary_scan_index = int(scan_index) - int(ns_scan_window)
    updated_snapshot = replace(
        nscan_commitment_snapshot,
        boundary_scan_index=boundary_scan_index,
        tracks_in_scope=0,
        latest_committed_ancestor_by_track_id={},
    )

    if boundary_scan_index < 0 or not tree_store.track_trees_by_track_id:
        return MapNScanPruningResult(
            boundary_scan_index=boundary_scan_index,
            tracks_in_scope=0,
            committed_count=0,
            disagreement_total=0,
            updated_snapshots=cluster_snapshots,
            nscan_commitment_snapshot=updated_snapshot,
        )

    plan = plan_map_n_scan_pruning(
        boundary_scan_index=boundary_scan_index,
        map_global=map_global,
        cluster_snapshots=cluster_snapshots,
        tree_store=tree_store,
    )
    updated_snapshot = replace(
        updated_snapshot,
        tracks_in_scope=len(plan.map_choice_by_track_id),
    )
    committed_count, updated_snapshot = apply_planned_map_n_scan_pruning(
        plan=plan,
        tree_store=tree_store,
        nscan_commitment_snapshot=updated_snapshot,
    )
    return MapNScanPruningResult(
        boundary_scan_index=boundary_scan_index,
        tracks_in_scope=len(plan.map_choice_by_track_id),
        committed_count=committed_count,
        disagreement_total=plan.disagreement_total,
        updated_snapshots=plan.updated_snapshots,
        nscan_commitment_snapshot=updated_snapshot,
    )
