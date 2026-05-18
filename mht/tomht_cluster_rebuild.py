"""Cluster rebuild orchestration for the track-oriented MHT tracker.

This module keeps the public cluster-rebuild flow small:

1. materialize active leaf options for each original live cluster,
2. delegate exact/overload solving to ``tomht_cluster_overload``,
3. wrap feasible original-cluster globals in snapshots,
4. merge cluster MAP globals for downstream scan processing.

Overload split subproblems are internal. MAP merge, N-scan, lifecycle, output,
and post-solve supported-leaf pruning all consume the same original-cluster
snapshot shape regardless of whether a cluster was solved directly,
``greedy_partition`` split, or ``conditional_exact`` split.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tomht_clustering import ClusterWorkItem, OverloadSplitSummary
from .tomht_cluster_overload import (
    ClusterSolveInput,
    has_any_feasible_cluster_combination,
    infeasible_cluster_debug_summary,
    is_global_feasible_under_live_conflicts,
    log_overload_split_summary,
    projected_combination_count,
    solve_cluster_globals,
)
from .tomht_cluster_solver import ClusterSolver
from .tomht_model import (
    ClusterRebuildSnapshot,
    DetectionKey,
    GlobalHypothesis,
    ScanContext,
    TrackHypothesisNode,
)
from .tomht_params import TOMHTParams
from .tomht_stats import RebuildStats
from .tomht_tree_store import TrackTreeStore

__all__ = [
    "cluster_leaf_options",
    "has_any_feasible_cluster_combination",
    "infeasible_cluster_debug_summary",
    "is_global_feasible_under_live_conflicts",
    "merge_cluster_map_globals",
    "rebuild_cluster_globals",
]


@dataclass(frozen=True)
class _ClusterRebuildResult:
    """Cluster rebuild result for one solved original cluster."""

    snapshot: ClusterRebuildSnapshot


def cluster_leaf_options(
    *,
    track_ids: tuple[int, ...],
    tree_store: TrackTreeStore,
) -> list[list[TrackHypothesisNode]]:
    """Materialize sorted active-leaf options for each track in a cluster."""
    out: list[list[TrackHypothesisNode]] = []
    for track_id in track_ids:
        tree = tree_store.track_trees_by_track_id[track_id]
        leaves = [
            tree_store.nodes_by_id[node_id]
            for node_id in sorted(tree.active_leaf_node_ids)
        ]
        if not leaves:
            raise RuntimeError(
                "Cluster rebuild encountered a tree with no active leaves. "
                "Lifecycle filtering should remove empty trees before clustering."
            )
        out.append(leaves)
    return out


def merge_cluster_map_globals(
    cluster_snapshots: list[ClusterRebuildSnapshot],
) -> GlobalHypothesis:
    """Merge cluster MAP globals into one full-scan MAP selection."""
    if not cluster_snapshots:
        return GlobalHypothesis(leaf_nodes_by_track_id={}, log_weight=0.0)

    merged_nodes: dict[int, TrackHypothesisNode] = {}
    merged_log = 0.0
    for snapshot in cluster_snapshots:
        if snapshot.map_global is None:
            continue
        merged_nodes.update(snapshot.map_global.leaf_nodes_by_track_id)
        merged_log += float(snapshot.map_global.log_weight)

    return GlobalHypothesis(
        leaf_nodes_by_track_id=merged_nodes,
        log_weight=float(merged_log),
    )


def _rebuild_one_cluster(
    *,
    cluster: ClusterWorkItem,
    ctx: ScanContext,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    cluster_solver: ClusterSolver,
) -> tuple[_ClusterRebuildResult, OverloadSplitSummary | None]:
    """Solve one original cluster and map retained globals to a snapshot."""
    leaf_options = cluster_leaf_options(
        track_ids=cluster.track_ids,
        tree_store=tree_store,
    )
    leaf_options_by_track_id = {
        track_id: leaf_options[idx] for idx, track_id in enumerate(cluster.track_ids)
    }
    projected_combinations = projected_combination_count(leaf_options)

    cluster_universe: set[DetectionKey] = set()
    for keys in cluster.current_scan_det_keys_by_track_id.values():
        cluster_universe |= keys

    solve_input = ClusterSolveInput(
        cluster=cluster,
        ctx=ctx,
        leaf_options=leaf_options,
        cluster_universe=cluster_universe,
    )
    solve_outcome = solve_cluster_globals(
        solve_input=solve_input,
        leaf_options_by_track_id=leaf_options_by_track_id,
        tree_store=tree_store,
        params=params,
        cluster_solver=cluster_solver,
        projected_combinations=projected_combinations,
    )
    map_global = solve_outcome.kept_globals[0] if solve_outcome.kept_globals else None

    return (
        _ClusterRebuildResult(
            snapshot=ClusterRebuildSnapshot(
                cluster_id=cluster.cluster_id,
                track_ids=cluster.track_ids,
                current_scan_conflict_det_keys=frozenset(cluster_universe),
                conflict_links=cluster.conflict_links,
                rebuilt_globals=solve_outcome.kept_globals,
                map_global=map_global,
                feasible_combinations=solve_outcome.feasible_combinations,
                evaluated_combinations=solve_outcome.combinations_evaluated,
                overload_split_origin_cluster_id=None,
            )
        ),
        solve_outcome.overload_split_summary,
    )


def rebuild_cluster_globals(
    *,
    clusters: list[ClusterWorkItem],
    ctx: ScanContext,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    cluster_solver: ClusterSolver,
) -> tuple[list[ClusterRebuildSnapshot], RebuildStats]:
    """Rebuild all clusters, hiding overload subproblems inside each snapshot."""
    if not clusters:
        return [], RebuildStats()

    clusters_for_rebuild = [
        ClusterWorkItem(
            cluster_id=cluster_id,
            track_ids=cluster.track_ids,
            current_scan_det_keys_by_track_id=(
                cluster.current_scan_det_keys_by_track_id
            ),
            conflict_links=cluster.conflict_links,
            overload_split_origin_cluster_id=None,
        )
        for cluster_id, cluster in enumerate(clusters)
    ]
    rebuild_results_and_summaries = [
        _rebuild_one_cluster(
            cluster=cluster,
            ctx=ctx,
            tree_store=tree_store,
            params=params,
            cluster_solver=cluster_solver,
        )
        for cluster in clusters_for_rebuild
    ]
    rebuild_results = [item[0] for item in rebuild_results_and_summaries]
    split_summaries = [
        item[1] for item in rebuild_results_and_summaries if item[1] is not None
    ]
    for split_summary in split_summaries:
        log_overload_split_summary(
            scan_index=ctx.scan_index,
            summary=split_summary,
        )

    snapshots = [result.snapshot for result in rebuild_results]
    return (
        snapshots,
        RebuildStats(
            cluster_count=len(snapshots),
            combinations_evaluated=sum(s.evaluated_combinations for s in snapshots),
            feasible_combinations=sum(s.feasible_combinations for s in snapshots),
            rebuilt_globals_stored=sum(len(s.rebuilt_globals) for s in snapshots),
            nscan_disagreement_total=0,
            overload_split_clusters=len(split_summaries),
            overload_split_operations=sum(
                len(summary.removed_edges) for summary in split_summaries
            ),
        ),
    )
