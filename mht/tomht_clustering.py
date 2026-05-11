"""Cluster work construction for the track-oriented MHT tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .tomht_model import DetectionKey, TrackHypothesisNode, TrackTree
from .tomht_params import TOMHTParams


@dataclass(frozen=True)
class ClusterWorkItem:
    """Transient per-scan cluster build input."""

    cluster_id: int
    track_ids: tuple[int, ...]
    current_scan_det_keys_by_track_id: dict[int, set[DetectionKey]]
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...]
    overload_split_origin_cluster_id: int | None = None


@dataclass(frozen=True)
class OverloadSplitRemovedEdge:
    """One removed conflict-graph edge during overload decomposition."""

    left_track_id: int
    right_track_id: int
    shared_history_key_count: int


@dataclass(frozen=True)
class OverloadSplitSummary:
    """Compact instrumentation for one original cluster overload split pass."""

    original_cluster_id: int
    original_track_ids: tuple[int, ...]
    projected_before: int
    projected_threshold: int
    removed_edges: tuple[OverloadSplitRemovedEdge, ...]
    resulting_subclusters: tuple[tuple[int, ...], ...]
    projected_after_by_subcluster: tuple[int, ...]
    stopping_reason: str


def current_scan_candidate_keys_for_tree(
    *,
    tree: TrackTree,
    nodes_by_id: Mapping[int, TrackHypothesisNode],
    scan_index: int,
) -> set[DetectionKey]:
    """Return current-scan detection keys present in one tree frontier."""
    keys: set[DetectionKey] = set()
    for leaf_id in tree.active_leaf_node_ids:
        leaf = nodes_by_id[leaf_id]
        if leaf.used_det_key is not None and int(leaf.used_det_key[0]) == scan_index:
            keys.add(leaf.used_det_key)
    return keys


def history_conflict_keys_for_tree(
    *,
    tree: TrackTree,
    nodes_by_id: Mapping[int, TrackHypothesisNode],
) -> set[DetectionKey]:
    """Return all detection-history keys present in this tree's active leaves."""
    keys: set[DetectionKey] = set()
    for leaf_id in tree.active_leaf_node_ids:
        leaf = nodes_by_id[leaf_id]
        keys |= set(leaf.detection_history_keys)
    return keys


def build_track_clusters(
    *,
    track_trees_by_track_id: Mapping[int, TrackTree],
    nodes_by_id: Mapping[int, TrackHypothesisNode],
    scan_index: int,
) -> list[ClusterWorkItem]:
    """Build independent clusters from shared active-leaf history detections."""
    track_ids = sorted(track_trees_by_track_id.keys())
    if not track_ids:
        return []

    history_keys_by_track: dict[int, set[DetectionKey]] = {
        track_id: history_conflict_keys_for_tree(
            tree=track_trees_by_track_id[track_id],
            nodes_by_id=nodes_by_id,
        )
        for track_id in track_ids
    }
    current_scan_keys_by_track: dict[int, set[DetectionKey]] = {
        track_id: current_scan_candidate_keys_for_tree(
            tree=track_trees_by_track_id[track_id],
            nodes_by_id=nodes_by_id,
            scan_index=scan_index,
        )
        for track_id in track_ids
    }

    adjacency: dict[int, set[int]] = {track_id: set() for track_id in track_ids}
    conflict_links: list[tuple[int, int, tuple[DetectionKey, ...]]] = []
    for i, left_track_id in enumerate(track_ids):
        for right_track_id in track_ids[i + 1 :]:
            shared = (
                history_keys_by_track[left_track_id]
                & history_keys_by_track[right_track_id]
            )
            if not shared:
                continue
            adjacency[left_track_id].add(right_track_id)
            adjacency[right_track_id].add(left_track_id)
            conflict_links.append(
                (
                    left_track_id,
                    right_track_id,
                    tuple(sorted(shared)),
                )
            )

    components: list[list[int]] = []
    seen: set[int] = set()
    for seed in track_ids:
        if seed in seen:
            continue
        stack = [seed]
        component: list[int] = []
        seen.add(seed)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nbr in sorted(adjacency[cur]):
                if nbr in seen:
                    continue
                seen.add(nbr)
                stack.append(nbr)
        component.sort()
        components.append(component)

    out: list[ClusterWorkItem] = []
    for cluster_id, component in enumerate(sorted(components, key=lambda c: tuple(c))):
        comp_track_ids = tuple(component)
        comp_track_set = set(comp_track_ids)
        comp_links = tuple(
            link
            for link in conflict_links
            if link[0] in comp_track_set and link[1] in comp_track_set
        )
        out.append(
            ClusterWorkItem(
                cluster_id=cluster_id,
                track_ids=comp_track_ids,
                current_scan_det_keys_by_track_id={
                    track_id: set(current_scan_keys_by_track[track_id])
                    for track_id in comp_track_ids
                },
                conflict_links=comp_links,
            )
        )
    return out


def projected_combination_count_for_track_ids(
    *,
    track_ids: tuple[int, ...],
    leaf_count_by_track_id: Mapping[int, int],
) -> int:
    """Projected Cartesian leaf combinations for one track-id tuple."""
    projected = 1
    for track_id in track_ids:
        leaf_count = int(leaf_count_by_track_id[track_id])
        if leaf_count <= 0:
            raise RuntimeError(
                "Cluster rebuild encountered a tree with no active leaves. "
                "Lifecycle filtering should remove empty trees before clustering."
            )
        projected *= leaf_count
    return projected


def canonical_edge_pair(
    left_track_id: int,
    right_track_id: int,
) -> tuple[int, int]:
    """Return canonical undirected edge ordering for track-id pairs."""
    if left_track_id <= right_track_id:
        return (left_track_id, right_track_id)
    return (right_track_id, left_track_id)


def connected_components_from_pairs(
    track_ids: tuple[int, ...],
    edge_pairs: Iterable[tuple[int, int]],
) -> list[tuple[int, ...]]:
    """Return connected components for the supplied undirected edge set."""
    adjacency: dict[int, set[int]] = {track_id: set() for track_id in track_ids}
    for left_track_id, right_track_id in edge_pairs:
        adjacency[left_track_id].add(right_track_id)
        adjacency[right_track_id].add(left_track_id)

    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for seed in sorted(track_ids):
        if seed in seen:
            continue
        stack = [seed]
        component: list[int] = []
        seen.add(seed)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nbr in sorted(adjacency[cur]):
                if nbr in seen:
                    continue
                seen.add(nbr)
                stack.append(nbr)
        components.append(tuple(sorted(component)))
    components.sort()
    return components


def cluster_edge_strengths(
    cluster: ClusterWorkItem,
) -> dict[tuple[int, int], int]:
    """Return conflict-edge strengths = shared full-history key counts."""
    strengths: dict[tuple[int, int], int] = {}
    for left_track_id, right_track_id, shared_keys in cluster.conflict_links:
        strengths[canonical_edge_pair(left_track_id, right_track_id)] = len(shared_keys)
    return strengths


def split_overloaded_cluster(
    *,
    cluster: ClusterWorkItem,
    leaf_count_by_track_id: Mapping[int, int],
    projected_before: int,
    threshold: int,
    max_edge_removals_per_cluster: int | None,
) -> tuple[list[ClusterWorkItem], OverloadSplitSummary]:
    """Approximate one overloaded cluster by severing weakest conflict edges."""
    remaining_edge_keys_by_pair: dict[tuple[int, int], tuple[DetectionKey, ...]] = {
        canonical_edge_pair(left_track_id, right_track_id): tuple(shared_keys)
        for left_track_id, right_track_id, shared_keys in cluster.conflict_links
    }
    edge_strengths = cluster_edge_strengths(cluster)
    removed_edges: list[OverloadSplitRemovedEdge] = []

    stopping_reason = "all_components_under_threshold"
    max_removals_int = (
        None
        if max_edge_removals_per_cluster is None
        else int(max_edge_removals_per_cluster)
    )

    while True:
        components = connected_components_from_pairs(
            cluster.track_ids,
            remaining_edge_keys_by_pair.keys(),
        )
        overloaded_components: list[tuple[int, ...]] = []
        for component_track_ids in components:
            projected = projected_combination_count_for_track_ids(
                track_ids=component_track_ids,
                leaf_count_by_track_id=leaf_count_by_track_id,
            )
            if projected > threshold:
                overloaded_components.append(component_track_ids)
        if not overloaded_components:
            break

        if max_removals_int is not None and len(removed_edges) >= max_removals_int:
            stopping_reason = "max_edge_removals_reached"
            break

        weakest: tuple[int, int, int] | None = None
        for component_track_ids in overloaded_components:
            component_track_set = set(component_track_ids)
            for left_track_id, right_track_id in remaining_edge_keys_by_pair:
                if (
                    left_track_id not in component_track_set
                    or right_track_id not in component_track_set
                ):
                    continue
                strength = edge_strengths[(left_track_id, right_track_id)]
                candidate = (strength, left_track_id, right_track_id)
                if weakest is None or candidate < weakest:
                    weakest = candidate

        if weakest is None:
            stopping_reason = "no_edges_left_in_overloaded_component"
            break

        strength, left_track_id, right_track_id = weakest
        remaining_edge_keys_by_pair.pop((left_track_id, right_track_id), None)
        removed_edges.append(
            OverloadSplitRemovedEdge(
                left_track_id=left_track_id,
                right_track_id=right_track_id,
                shared_history_key_count=strength,
            )
        )

    final_components = connected_components_from_pairs(
        cluster.track_ids,
        remaining_edge_keys_by_pair.keys(),
    )
    subclusters: list[ClusterWorkItem] = []
    projected_after_by_subcluster: list[int] = []
    for component_track_ids in final_components:
        component_track_set = set(component_track_ids)
        component_links = tuple(
            (
                left_track_id,
                right_track_id,
                remaining_edge_keys_by_pair[(left_track_id, right_track_id)],
            )
            for left_track_id, right_track_id in sorted(remaining_edge_keys_by_pair)
            if left_track_id in component_track_set
            and right_track_id in component_track_set
        )
        subclusters.append(
            ClusterWorkItem(
                cluster_id=-1,
                track_ids=component_track_ids,
                current_scan_det_keys_by_track_id={
                    track_id: set(cluster.current_scan_det_keys_by_track_id[track_id])
                    for track_id in component_track_ids
                },
                conflict_links=component_links,
                overload_split_origin_cluster_id=cluster.cluster_id,
            )
        )
        projected_after_by_subcluster.append(
            projected_combination_count_for_track_ids(
                track_ids=component_track_ids,
                leaf_count_by_track_id=leaf_count_by_track_id,
            )
        )

    summary = OverloadSplitSummary(
        original_cluster_id=cluster.cluster_id,
        original_track_ids=cluster.track_ids,
        projected_before=projected_before,
        projected_threshold=threshold,
        removed_edges=tuple(removed_edges),
        resulting_subclusters=tuple(subcluster.track_ids for subcluster in subclusters),
        projected_after_by_subcluster=tuple(projected_after_by_subcluster),
        stopping_reason=stopping_reason,
    )
    return subclusters, summary


def maybe_split_cluster_under_overload(
    *,
    cluster: ClusterWorkItem,
    leaf_count_by_track_id: Mapping[int, int],
    overload_split_enabled: bool,
    projected_combination_threshold: int | None,
    max_edge_removals_per_cluster: int | None,
) -> tuple[list[ClusterWorkItem], OverloadSplitSummary | None]:
    """Split one cluster only when projected Cartesian size exceeds threshold."""
    if not overload_split_enabled:
        return [cluster], None

    if projected_combination_threshold is None:
        return [cluster], None
    threshold = int(projected_combination_threshold)
    if threshold <= 0:
        raise ValueError(
            "overload_split_projected_combination_threshold must be positive "
            "when overload splitting is enabled."
        )

    projected_before = projected_combination_count_for_track_ids(
        track_ids=cluster.track_ids,
        leaf_count_by_track_id=leaf_count_by_track_id,
    )
    if projected_before <= threshold:
        return [cluster], None

    return split_overloaded_cluster(
        cluster=cluster,
        leaf_count_by_track_id=leaf_count_by_track_id,
        projected_before=projected_before,
        threshold=threshold,
        max_edge_removals_per_cluster=max_edge_removals_per_cluster,
    )


def apply_overload_splitting_to_clusters(
    *,
    clusters: list[ClusterWorkItem],
    leaf_count_by_track_id: Mapping[int, int],
    params: TOMHTParams,
) -> tuple[list[ClusterWorkItem], list[OverloadSplitSummary]]:
    """Apply optional overload splitting to an ordered cluster-work list."""
    clusters_for_rebuild: list[ClusterWorkItem] = []
    split_summaries: list[OverloadSplitSummary] = []

    for cluster in clusters:
        subclusters, split_summary = maybe_split_cluster_under_overload(
            cluster=cluster,
            leaf_count_by_track_id=leaf_count_by_track_id,
            overload_split_enabled=params.overload_split_enabled,
            projected_combination_threshold=(
                params.overload_split_projected_combination_threshold
            ),
            max_edge_removals_per_cluster=(
                params.overload_split_max_edge_removals_per_cluster
            ),
        )
        clusters_for_rebuild.extend(subclusters)
        if split_summary is not None:
            split_summaries.append(split_summary)

    return clusters_for_rebuild, split_summaries
