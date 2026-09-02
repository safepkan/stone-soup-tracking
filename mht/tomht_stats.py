"""Passive TO-MHT scan stats structures and summary reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from statistics import median
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .tomht_model import (
    ClusterRebuildSnapshot,
    GlobalHypothesis,
    NScanCommitmentSnapshot,
    ScanContext,
)
from .tomht_tree_store import TrackTreeStore

if TYPE_CHECKING:
    from .tomht_expansion import ExpansionCallStats
    from .tomht_pruning import SupportedLeafPruningStats


@dataclass(frozen=True)
class BirthStats:
    """End-of-scan residual and retained internal-birth counts."""

    residual_detections_considered: int = 0
    birth_tracks_created: int = 0
    birth_tracks_kept: int = 0


@dataclass(frozen=True)
class RebuildStats:
    cluster_count: int = 0
    combinations_evaluated: int = 0
    feasible_combinations: int = 0
    rebuilt_globals_stored: int = 0
    nscan_disagreement_total: int = 0
    overload_split_clusters: int = 0
    overload_split_operations: int = 0


@dataclass(frozen=True)
class FrontierStageCounts:
    """Active tree/leaf counts sampled at one scan-pipeline boundary."""

    active_trees: int = 0
    active_leaves: int = 0


@dataclass(frozen=True)
class ExpansionFrontierStats:
    """Per-scan aggregate expansion volume and ordered frontier counters."""

    leaves_before_expansion: int = 0
    leaves_after_expansion: int = 0
    leaves_after_empty_tree_removal: int = 0
    leaves_after_post_solve_supported_pruning: int = 0
    leaves_after_n_scan_pruning: int = 0
    leaves_after_lifecycle: int = 0
    # Frontier after end-of-scan residual processing and birth insertion.
    leaves_after_end_of_scan_births: int = 0
    trees_before_expansion: int = 0
    trees_after_expansion: int = 0
    trees_after_empty_tree_removal: int = 0
    trees_after_post_solve_supported_pruning: int = 0
    trees_after_n_scan_pruning: int = 0
    trees_after_lifecycle: int = 0
    # Tree count after end-of-scan residual processing and birth insertion.
    trees_after_end_of_scan_births: int = 0
    expanded_leaf_count: int = 0
    expanded_leaves_tentative: int = 0
    expanded_leaves_confirmed: int = 0
    local_child_candidates_total: int = 0
    local_children_created_total: int = 0
    local_children_retained_total: int = 0
    local_miss_children_created: int = 0
    local_detection_children_created: int = 0
    map_selected_leaf_count: int = 0
    retained_topk_supported_leaf_count: int = 0
    unsupported_leaf_count_pruned: int = 0


@dataclass(frozen=True)
class ScanTimingBreakdown:
    """Per-scan wall-time breakdown across major update pipeline phases."""

    prep_ctx_ms: float = 0.0
    pre_expand_validate_ms: float = 0.0
    expand_ms: float = 0.0
    expand_hypothesise_calls: int = 0
    expand_hypothesise_ms: float = 0.0
    expand_track_reconstruct_calls: int = 0
    expand_track_reconstruct_ms: float = 0.0
    expand_default_state_fast_path_calls: int = 0
    expand_update_calls: int = 0
    expand_update_ms: float = 0.0
    post_expand_prune_validate_ms: float = 0.0
    cluster_build_and_solve_ms: float = 0.0
    post_solve_prune_ms: float = 0.0
    map_merge_ms: float = 0.0
    nscan_prune_ms: float = 0.0
    lifecycle_ms: float = 0.0
    lifecycle_deleter_track_reconstruct_calls: int = 0
    lifecycle_deleter_track_reconstruct_ms: float = 0.0
    lifecycle_default_miss_deleter_fast_path_calls: int = 0
    lifecycle_deleter_check_ms: float = 0.0
    # End-of-scan residual extraction, initiation, root insertion, MAP merge, and
    # post-insertion confirmation time.
    births_ms: float = 0.0
    publication_ms: float = 0.0
    cleanup_ms: float = 0.0


@dataclass(frozen=True)
class ScanStats:
    scan_index: int
    timestamp: datetime.datetime
    scan_wall_ms: float
    maxrss_mb: float
    node_count_total: int
    active_trees: int
    active_tentative_trees: int
    active_confirmed_trees: int
    active_leaves: int
    num_detections: int
    cluster_count: int
    combinations_evaluated: int
    feasible_combinations: int
    rebuilt_globals_stored: int
    nscan_disagreement_total: int
    overload_split_clusters: int
    overload_split_operations: int
    nscan_boundary_scan_index: int
    nscan_tracks_in_scope: int
    nscan_tracks_committed: int
    birth_candidates: int
    birth_tracks_created: int
    birth_tracks_kept: int
    map_tracks: int
    map_published_tracks: int
    map_unpublished_tracks: int
    map_used: int
    map_unused: int
    map_miss_hist: dict[int, int]
    map_mean_hit_rate: float
    timing_breakdown: ScanTimingBreakdown = field(default_factory=ScanTimingBreakdown)
    expansion_frontier: ExpansionFrontierStats = field(
        default_factory=ExpansionFrontierStats
    )


def frontier_stage_counts_for_store(
    *,
    tree_store: TrackTreeStore,
) -> FrontierStageCounts:
    """Sample cheap active tree/leaf counts from the persistent tree store."""
    return FrontierStageCounts(
        active_trees=tree_store.active_tree_count(),
        active_leaves=tree_store.active_leaf_count(),
    )


def _map_selected_leaf_ids(
    cluster_snapshots: list[ClusterRebuildSnapshot],
) -> set[int]:
    out: set[int] = set()
    for snapshot in cluster_snapshots:
        if snapshot.map_global is None:
            continue
        out.update(
            int(leaf.node_id)
            for leaf in snapshot.map_global.leaf_nodes_by_track_id.values()
        )
    return out


def _retained_topk_supported_leaf_ids(
    cluster_snapshots: list[ClusterRebuildSnapshot],
) -> set[int]:
    out: set[int] = set()
    for snapshot in cluster_snapshots:
        for rebuilt_global in snapshot.rebuilt_globals:
            out.update(
                int(leaf.node_id)
                for leaf in rebuilt_global.leaf_nodes_by_track_id.values()
            )
    return out


def build_expansion_frontier_stats(
    *,
    before_expansion: FrontierStageCounts,
    after_expansion: FrontierStageCounts,
    after_empty_tree_removal: FrontierStageCounts,
    after_post_solve_supported_pruning: FrontierStageCounts,
    after_n_scan_pruning: FrontierStageCounts,
    after_lifecycle: FrontierStageCounts,
    after_end_of_scan_births: FrontierStageCounts,
    expansion_call_stats: ExpansionCallStats,
    supported_pruning_stats: SupportedLeafPruningStats,
    cluster_snapshots: list[ClusterRebuildSnapshot],
) -> ExpansionFrontierStats:
    """Assemble expansion/frontier counters from phase-local instrumentation."""
    return ExpansionFrontierStats(
        leaves_before_expansion=before_expansion.active_leaves,
        leaves_after_expansion=after_expansion.active_leaves,
        leaves_after_empty_tree_removal=after_empty_tree_removal.active_leaves,
        leaves_after_post_solve_supported_pruning=(
            after_post_solve_supported_pruning.active_leaves
        ),
        leaves_after_n_scan_pruning=after_n_scan_pruning.active_leaves,
        leaves_after_lifecycle=after_lifecycle.active_leaves,
        leaves_after_end_of_scan_births=(after_end_of_scan_births.active_leaves),
        trees_before_expansion=before_expansion.active_trees,
        trees_after_expansion=after_expansion.active_trees,
        trees_after_empty_tree_removal=after_empty_tree_removal.active_trees,
        trees_after_post_solve_supported_pruning=(
            after_post_solve_supported_pruning.active_trees
        ),
        trees_after_n_scan_pruning=after_n_scan_pruning.active_trees,
        trees_after_lifecycle=after_lifecycle.active_trees,
        trees_after_end_of_scan_births=(after_end_of_scan_births.active_trees),
        expanded_leaf_count=int(expansion_call_stats.expanded_leaf_count),
        expanded_leaves_tentative=int(expansion_call_stats.expanded_leaves_tentative),
        expanded_leaves_confirmed=int(expansion_call_stats.expanded_leaves_confirmed),
        local_child_candidates_total=int(
            expansion_call_stats.local_child_candidates_total
        ),
        local_children_created_total=int(
            expansion_call_stats.local_children_created_total
        ),
        local_children_retained_total=int(
            expansion_call_stats.local_children_retained_total
        ),
        local_miss_children_created=int(
            expansion_call_stats.local_miss_children_created
        ),
        local_detection_children_created=int(
            expansion_call_stats.local_detection_children_created
        ),
        map_selected_leaf_count=len(_map_selected_leaf_ids(cluster_snapshots)),
        retained_topk_supported_leaf_count=len(
            _retained_topk_supported_leaf_ids(cluster_snapshots)
        ),
        unsupported_leaf_count_pruned=int(
            supported_pruning_stats.unsupported_leaf_count_pruned
        ),
    )


def display_cluster_rebuilds(
    *,
    cluster_snapshots: list[ClusterRebuildSnapshot],
    debug_globals_max: int,
) -> None:
    """Print retained rebuilt globals for last scan inspection."""
    for snapshot in cluster_snapshots:
        split_from = snapshot.overload_split_origin_cluster_id
        split_tag = "" if split_from is None else f" split_from={split_from}"
        print(
            f"cluster={snapshot.cluster_id} tracks={list(snapshot.track_ids)}{split_tag} "
            f"globals={len(snapshot.rebuilt_globals)} "
            f"comb_eval={snapshot.evaluated_combinations} "
            f"comb_feas={snapshot.feasible_combinations} "
            f"disagree={snapshot.disagreement_count}"
        )
        for gh in snapshot.rebuilt_globals[:debug_globals_max]:
            tids = sorted(gh.leaf_nodes_by_track_id.keys())
            print(f"  logW={gh.log_weight:.3f} tids={tids}")


def maybe_display_scan_debug_output(
    *,
    ctx: ScanContext,
    cluster_snapshots: list[ClusterRebuildSnapshot],
    debug_display_detections: bool,
    debug_display_hypotheses: bool,
    debug_globals_max: int,
) -> None:
    """Emit optional per-scan debug displays before stats logging."""
    if debug_display_detections:
        print(f"\nDetections at timestamp {ctx.timestamp}:")
        for det in ctx.detections:
            print(f"  {det.state_vector}")

    if debug_display_hypotheses:
        print(f"\nCluster rebuilds scan={ctx.scan_index} t={ctx.timestamp}:")
        display_cluster_rebuilds(
            cluster_snapshots=cluster_snapshots,
            debug_globals_max=debug_globals_max,
        )


def map_stats_for_current_map(
    *,
    map_global: GlobalHypothesis | None,
    tree_store: TrackTreeStore,
    detection_count: int,
    scan_index: int,
) -> tuple[int, int, int, int, int, dict[int, int], float]:
    """Compute lightweight MAP-level counters for scan stats reporting."""
    map_tracks = 0
    map_published_tracks = 0
    map_unpublished_tracks = 0
    map_used = 0
    map_unused = int(detection_count)
    map_miss_hist: dict[int, int] = {}
    map_mean_hit_rate = 0.0
    if map_global is None:
        return (
            map_tracks,
            map_published_tracks,
            map_unpublished_tracks,
            map_used,
            map_unused,
            map_miss_hist,
            map_mean_hit_rate,
        )

    leaf_nodes_by_track_id = map_global.leaf_nodes_by_track_id
    map_tracks = len(leaf_nodes_by_track_id)
    track_trees_by_track_id = tree_store.track_trees_by_track_id
    for track_id in leaf_nodes_by_track_id:
        tree = track_trees_by_track_id.get(track_id)
        if tree is not None and tree.publication_state == "published":
            map_published_tracks += 1
    map_unpublished_tracks = map_tracks - map_published_tracks

    used_keys = {
        leaf.used_det_key
        for leaf in leaf_nodes_by_track_id.values()
        if leaf.used_det_key is not None
        and int(leaf.used_det_key.scan_index) == scan_index
    }
    map_used = len(used_keys)
    map_unused = int(detection_count) - map_used

    hit_rates: list[float] = []
    for leaf_node in leaf_nodes_by_track_id.values():
        misses = int(leaf_node.missed_count)
        map_miss_hist[misses] = map_miss_hist.get(misses, 0) + 1
        age = int(leaf_node.age)
        if age > 0:
            hits = int(leaf_node.hits)
            hit_rates.append(float(hits) / float(age))
    if hit_rates:
        map_mean_hit_rate = float(np.mean(hit_rates))
    return (
        map_tracks,
        map_published_tracks,
        map_unpublished_tracks,
        map_used,
        map_unused,
        map_miss_hist,
        map_mean_hit_rate,
    )


def active_lifecycle_tree_counts(
    *,
    tree_store: TrackTreeStore,
) -> tuple[int, int]:
    """Return active-tree counts by tentative/confirmed lifecycle state."""
    tentative = 0
    confirmed = 0
    for tree in tree_store.track_trees_by_track_id.values():
        if tree.lifecycle_state == "confirmed":
            confirmed += 1
        else:
            tentative += 1
    return tentative, confirmed


def build_scan_stats(
    *,
    ctx: ScanContext,
    map_global: GlobalHypothesis | None,
    tree_store: TrackTreeStore,
    scan_wall_ms: float,
    maxrss_mb: float,
    node_count_total: int,
    active_trees: int,
    active_leaves: int,
    rebuild_stats: RebuildStats,
    nscan_boundary_scan_index: int,
    nscan_tracks_in_scope: int,
    nscan_tracks_committed: int,
    birth_stats: BirthStats,
    timing_breakdown: ScanTimingBreakdown,
    expansion_frontier_stats: ExpansionFrontierStats | None = None,
) -> ScanStats:
    """Assemble one immutable per-scan ScanStats record."""
    (
        map_tracks,
        map_published_tracks,
        map_unpublished_tracks,
        map_used,
        map_unused,
        map_miss_hist,
        map_mean_hit_rate,
    ) = map_stats_for_current_map(
        map_global=map_global,
        tree_store=tree_store,
        detection_count=len(ctx.detections),
        scan_index=ctx.scan_index,
    )
    active_tentative_trees, active_confirmed_trees = active_lifecycle_tree_counts(
        tree_store=tree_store
    )
    return ScanStats(
        scan_index=int(ctx.scan_index),
        timestamp=ctx.timestamp,
        scan_wall_ms=float(scan_wall_ms),
        maxrss_mb=float(maxrss_mb),
        node_count_total=int(node_count_total),
        active_trees=int(active_trees),
        active_tentative_trees=active_tentative_trees,
        active_confirmed_trees=active_confirmed_trees,
        active_leaves=int(active_leaves),
        num_detections=len(ctx.detections),
        cluster_count=rebuild_stats.cluster_count,
        combinations_evaluated=rebuild_stats.combinations_evaluated,
        feasible_combinations=rebuild_stats.feasible_combinations,
        rebuilt_globals_stored=rebuild_stats.rebuilt_globals_stored,
        nscan_disagreement_total=rebuild_stats.nscan_disagreement_total,
        overload_split_clusters=rebuild_stats.overload_split_clusters,
        overload_split_operations=rebuild_stats.overload_split_operations,
        nscan_boundary_scan_index=nscan_boundary_scan_index,
        nscan_tracks_in_scope=nscan_tracks_in_scope,
        nscan_tracks_committed=nscan_tracks_committed,
        birth_candidates=birth_stats.residual_detections_considered,
        birth_tracks_created=birth_stats.birth_tracks_created,
        birth_tracks_kept=birth_stats.birth_tracks_kept,
        map_tracks=map_tracks,
        map_published_tracks=map_published_tracks,
        map_unpublished_tracks=map_unpublished_tracks,
        map_used=map_used,
        map_unused=map_unused,
        map_miss_hist=map_miss_hist,
        map_mean_hit_rate=map_mean_hit_rate,
        timing_breakdown=timing_breakdown,
        expansion_frontier=(
            ExpansionFrontierStats()
            if expansion_frontier_stats is None
            else expansion_frontier_stats
        ),
    )


def print_scan_stats(
    *,
    timestamp: datetime.datetime,
    scan_stats: ScanStats,
    nscan_snapshot: NScanCommitmentSnapshot,
    debug_display_map_miss_hist: bool,
) -> None:
    """Print one per-scan instrumentation block from precomputed ScanStats."""
    breakdown = scan_stats.timing_breakdown
    phase_accounted_ms = (
        float(breakdown.prep_ctx_ms)
        + float(breakdown.pre_expand_validate_ms)
        + float(breakdown.expand_ms)
        + float(breakdown.post_expand_prune_validate_ms)
        + float(breakdown.births_ms)
        + float(breakdown.cluster_build_and_solve_ms)
        + float(breakdown.post_solve_prune_ms)
        + float(breakdown.map_merge_ms)
        + float(breakdown.nscan_prune_ms)
        + float(breakdown.lifecycle_ms)
        + float(breakdown.publication_ms)
        + float(breakdown.cleanup_ms)
    )
    phase_other_ms = max(float(scan_stats.scan_wall_ms) - phase_accounted_ms, 0.0)
    expand_other_ms = max(
        float(breakdown.expand_ms)
        - float(breakdown.expand_hypothesise_ms)
        - float(breakdown.expand_track_reconstruct_ms)
        - float(breakdown.expand_update_ms),
        0.0,
    )
    lifecycle_other_ms = max(
        float(breakdown.lifecycle_ms)
        - float(breakdown.lifecycle_deleter_track_reconstruct_ms)
        - float(breakdown.lifecycle_deleter_check_ms),
        0.0,
    )

    print(
        f"SCAN scan={scan_stats.scan_index} t={timestamp} "
        f"det={scan_stats.num_detections} "
        f"trees={scan_stats.active_trees} leaves={scan_stats.active_leaves} "
        f"tentative={scan_stats.active_tentative_trees} "
        f"confirmed={scan_stats.active_confirmed_trees} "
        f"clusters={scan_stats.cluster_count} "
        f"comb_eval={scan_stats.combinations_evaluated} "
        f"comb_feas={scan_stats.feasible_combinations} "
        f"rebuilt_globals={scan_stats.rebuilt_globals_stored} "
        f"split_clusters={scan_stats.overload_split_clusters} "
        f"split_ops={scan_stats.overload_split_operations} "
        f"nscan boundary={scan_stats.nscan_boundary_scan_index} "
        f"in_scope={scan_stats.nscan_tracks_in_scope} "
        f"committed_now={scan_stats.nscan_tracks_committed} "
        f"committed_total={len(nscan_snapshot.committed_boundary_by_track_id)} "
        f"disagree={scan_stats.nscan_disagreement_total} "
        f"births cand={scan_stats.birth_candidates} "
        f"tracks_created={scan_stats.birth_tracks_created} "
        f"tracks_kept={scan_stats.birth_tracks_kept} "
        f"MAP tracks={scan_stats.map_tracks} "
        f"published={scan_stats.map_published_tracks} "
        f"unpublished={scan_stats.map_unpublished_tracks} "
        f"used={scan_stats.map_used} unused={scan_stats.map_unused} "
        f"hit_rate={scan_stats.map_mean_hit_rate:.2f}"
    )
    print(f"SCAN_TIMING t={timestamp} wall_ms={scan_stats.scan_wall_ms:.3f}")
    print(
        f"SCAN_TIMING_PHASES t={timestamp} "
        f"prep_ctx_ms={breakdown.prep_ctx_ms:.3f} "
        f"pre_expand_validate_ms={breakdown.pre_expand_validate_ms:.3f} "
        f"expand_ms={breakdown.expand_ms:.3f} "
        f"expand_hypothesise_calls={breakdown.expand_hypothesise_calls} "
        f"expand_hypothesise_ms={breakdown.expand_hypothesise_ms:.3f} "
        "expand_track_reconstruct_calls="
        f"{breakdown.expand_track_reconstruct_calls} "
        "expand_track_reconstruct_ms="
        f"{breakdown.expand_track_reconstruct_ms:.3f} "
        "expand_default_state_fast_path_calls="
        f"{breakdown.expand_default_state_fast_path_calls} "
        f"expand_update_calls={breakdown.expand_update_calls} "
        f"expand_update_ms={breakdown.expand_update_ms:.3f} "
        f"expand_other_ms={expand_other_ms:.3f} "
        "post_expand_prune_validate_ms="
        f"{breakdown.post_expand_prune_validate_ms:.3f} "
        f"cluster_build_solve_ms={breakdown.cluster_build_and_solve_ms:.3f} "
        f"post_solve_prune_ms={breakdown.post_solve_prune_ms:.3f} "
        f"map_merge_ms={breakdown.map_merge_ms:.3f} "
        f"nscan_prune_ms={breakdown.nscan_prune_ms:.3f} "
        f"lifecycle_ms={breakdown.lifecycle_ms:.3f} "
        "lifecycle_deleter_track_reconstruct_calls="
        f"{breakdown.lifecycle_deleter_track_reconstruct_calls} "
        "lifecycle_deleter_track_reconstruct_ms="
        f"{breakdown.lifecycle_deleter_track_reconstruct_ms:.3f} "
        "lifecycle_default_miss_deleter_fast_path_calls="
        f"{breakdown.lifecycle_default_miss_deleter_fast_path_calls} "
        "lifecycle_deleter_check_ms="
        f"{breakdown.lifecycle_deleter_check_ms:.3f} "
        f"lifecycle_other_ms={lifecycle_other_ms:.3f} "
        f"births_ms={breakdown.births_ms:.3f} "
        f"publication_ms={breakdown.publication_ms:.3f} "
        f"cleanup_ms={breakdown.cleanup_ms:.3f} "
        f"other_ms={phase_other_ms:.3f}"
    )
    print(
        f"SCAN_MEMORY t={timestamp} "
        f"nodes={scan_stats.node_count_total} "
        f"maxrss_mb={scan_stats.maxrss_mb:.1f}"
    )
    if nscan_snapshot.latest_committed_ancestor_by_track_id:
        committed_pairs = ", ".join(
            f"{track_id}->node{ancestor.node_id}@s{ancestor.scan_index}"
            for track_id, ancestor in sorted(
                nscan_snapshot.latest_committed_ancestor_by_track_id.items()
            )
        )
        print(
            "SCAN_NSCAN_COMMITTED "
            f"scan={scan_stats.scan_index} "
            f"t={timestamp} boundary={nscan_snapshot.boundary_scan_index} "
            f"{committed_pairs}"
        )
    if debug_display_map_miss_hist:
        print(f"SCAN_MAP_MISS_HIST t={timestamp} miss_hist={scan_stats.map_miss_hist}")


def print_expansion_frontier_stats(
    *,
    timestamp: datetime.datetime,
    scan_stats: ScanStats,
) -> None:
    """Print compact opt-in expansion/frontier usefulness diagnostics."""
    stats = scan_stats.expansion_frontier
    print(
        f"EXPANSION_FRONTIER scan={scan_stats.scan_index} t={timestamp} "
        f"leaves_before={stats.leaves_before_expansion} "
        f"leaves_after_expansion={stats.leaves_after_expansion} "
        f"leaves_after_empty={stats.leaves_after_empty_tree_removal} "
        "leaves_after_supported_prune="
        f"{stats.leaves_after_post_solve_supported_pruning} "
        f"leaves_after_nscan={stats.leaves_after_n_scan_pruning} "
        f"leaves_after_lifecycle={stats.leaves_after_lifecycle} "
        "leaves_after_end_scan_births="
        f"{stats.leaves_after_end_of_scan_births} "
        f"trees_before={stats.trees_before_expansion} "
        f"trees_after_lifecycle={stats.trees_after_lifecycle} "
        f"expanded={stats.expanded_leaf_count} "
        f"expanded_tentative={stats.expanded_leaves_tentative} "
        f"expanded_confirmed={stats.expanded_leaves_confirmed} "
        f"child_candidates={stats.local_child_candidates_total} "
        f"children_created={stats.local_children_created_total} "
        f"children_retained={stats.local_children_retained_total} "
        f"miss_children={stats.local_miss_children_created} "
        f"detection_children={stats.local_detection_children_created} "
        "track_reconstruct_calls="
        f"{scan_stats.timing_breakdown.expand_track_reconstruct_calls} "
        "default_state_fast_path_calls="
        f"{scan_stats.timing_breakdown.expand_default_state_fast_path_calls} "
        f"topk_supported={stats.retained_topk_supported_leaf_count} "
        f"map_selected={stats.map_selected_leaf_count} "
        f"unsupported_pruned={stats.unsupported_leaf_count_pruned}"
    )


def print_summary_stats(
    *,
    stats: list[ScanStats],
    last_nscan_boundary_scan_index: int | None,
    committed_boundary_by_track_id: Mapping[int, int],
    debug_display_expansion_frontier: bool = False,
) -> None:
    """Print aggregate instrumentation summaries from collected ScanStats."""
    if not stats:
        print("SUMMARY scans=0 (no collected ScanStats)")
        return

    num_scans = len(stats)

    def _mean(values: list[int] | list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values)) / float(len(values))

    def _percentile(values: list[int] | list[float], quantile: float) -> float:
        """Linear-interpolated percentile in [0.0, 1.0]."""
        if not values:
            return 0.0
        sorted_values = sorted(float(value) for value in values)
        if quantile <= 0.0:
            return sorted_values[0]
        if quantile >= 1.0:
            return sorted_values[-1]
        rank = (len(sorted_values) - 1) * quantile
        lower_idx = int(rank)
        upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
        fraction = rank - float(lower_idx)
        return (
            sorted_values[lower_idx] * (1.0 - fraction)
            + sorted_values[upper_idx] * fraction
        )

    trees = [s.active_trees for s in stats]
    tentative_trees = [s.active_tentative_trees for s in stats]
    confirmed_trees = [s.active_confirmed_trees for s in stats]
    leaves = [s.active_leaves for s in stats]
    clusters = [s.cluster_count for s in stats]
    comb_eval = [s.combinations_evaluated for s in stats]
    comb_feas = [s.feasible_combinations for s in stats]
    rebuilt = [s.rebuilt_globals_stored for s in stats]
    split_clusters = [s.overload_split_clusters for s in stats]
    split_ops = [s.overload_split_operations for s in stats]
    disagree = [s.nscan_disagreement_total for s in stats]

    birth_created = [s.birth_tracks_created for s in stats]
    birth_kept = [s.birth_tracks_kept for s in stats]
    birth_cand = [s.birth_candidates for s in stats]

    map_tracks = [s.map_tracks for s in stats]
    map_published_tracks = [s.map_published_tracks for s in stats]
    map_unpublished_tracks = [s.map_unpublished_tracks for s in stats]
    map_unused = [s.map_unused for s in stats]
    map_used = [s.map_used for s in stats]
    map_hit_rate = [s.map_mean_hit_rate for s in stats]

    scan_wall_ms = [s.scan_wall_ms for s in stats]
    prep_ctx_ms = [s.timing_breakdown.prep_ctx_ms for s in stats]
    pre_expand_validate_ms = [s.timing_breakdown.pre_expand_validate_ms for s in stats]
    expand_ms = [s.timing_breakdown.expand_ms for s in stats]
    expand_hypothesise_ms = [s.timing_breakdown.expand_hypothesise_ms for s in stats]
    expand_track_reconstruct_ms = [
        s.timing_breakdown.expand_track_reconstruct_ms for s in stats
    ]
    expand_update_ms = [s.timing_breakdown.expand_update_ms for s in stats]
    expand_other_ms = [
        max(
            float(s.timing_breakdown.expand_ms)
            - float(s.timing_breakdown.expand_hypothesise_ms)
            - float(s.timing_breakdown.expand_track_reconstruct_ms)
            - float(s.timing_breakdown.expand_update_ms),
            0.0,
        )
        for s in stats
    ]
    expand_hypothesise_calls = [
        float(s.timing_breakdown.expand_hypothesise_calls) for s in stats
    ]
    expand_track_reconstruct_calls = [
        float(s.timing_breakdown.expand_track_reconstruct_calls) for s in stats
    ]
    expand_default_state_fast_path_calls = [
        float(s.timing_breakdown.expand_default_state_fast_path_calls) for s in stats
    ]
    expand_update_calls = [float(s.timing_breakdown.expand_update_calls) for s in stats]
    post_expand_prune_validate_ms = [
        s.timing_breakdown.post_expand_prune_validate_ms for s in stats
    ]
    births_ms = [s.timing_breakdown.births_ms for s in stats]
    cluster_build_and_solve_ms = [
        s.timing_breakdown.cluster_build_and_solve_ms for s in stats
    ]
    post_solve_prune_ms = [s.timing_breakdown.post_solve_prune_ms for s in stats]
    map_merge_ms = [s.timing_breakdown.map_merge_ms for s in stats]
    nscan_prune_ms = [s.timing_breakdown.nscan_prune_ms for s in stats]
    lifecycle_ms = [s.timing_breakdown.lifecycle_ms for s in stats]
    lifecycle_deleter_track_reconstruct_ms = [
        s.timing_breakdown.lifecycle_deleter_track_reconstruct_ms for s in stats
    ]
    lifecycle_deleter_check_ms = [
        s.timing_breakdown.lifecycle_deleter_check_ms for s in stats
    ]
    lifecycle_deleter_track_reconstruct_calls = [
        float(s.timing_breakdown.lifecycle_deleter_track_reconstruct_calls)
        for s in stats
    ]
    lifecycle_default_miss_deleter_fast_path_calls = [
        float(s.timing_breakdown.lifecycle_default_miss_deleter_fast_path_calls)
        for s in stats
    ]
    lifecycle_other_ms = [
        max(
            float(s.timing_breakdown.lifecycle_ms)
            - float(s.timing_breakdown.lifecycle_deleter_track_reconstruct_ms)
            - float(s.timing_breakdown.lifecycle_deleter_check_ms),
            0.0,
        )
        for s in stats
    ]
    publication_ms = [s.timing_breakdown.publication_ms for s in stats]
    cleanup_ms = [s.timing_breakdown.cleanup_ms for s in stats]
    other_ms = [
        max(
            float(s.scan_wall_ms)
            - (
                float(s.timing_breakdown.prep_ctx_ms)
                + float(s.timing_breakdown.pre_expand_validate_ms)
                + float(s.timing_breakdown.expand_ms)
                + float(s.timing_breakdown.post_expand_prune_validate_ms)
                + float(s.timing_breakdown.births_ms)
                + float(s.timing_breakdown.cluster_build_and_solve_ms)
                + float(s.timing_breakdown.post_solve_prune_ms)
                + float(s.timing_breakdown.map_merge_ms)
                + float(s.timing_breakdown.nscan_prune_ms)
                + float(s.timing_breakdown.lifecycle_ms)
                + float(s.timing_breakdown.publication_ms)
                + float(s.timing_breakdown.cleanup_ms)
            ),
            0.0,
        )
        for s in stats
    ]
    maxrss_mb = [s.maxrss_mb for s in stats]
    node_count_total = [s.node_count_total for s in stats]

    nscan_tracks_in_scope = [s.nscan_tracks_in_scope for s in stats]
    nscan_tracks_committed = [s.nscan_tracks_committed for s in stats]

    print(
        "SUMMARY "
        f"scans={num_scans} "
        f"det_total={sum(s.num_detections for s in stats)} "
        f"det_mean={_mean([s.num_detections for s in stats]):.2f}"
    )
    print(
        "SUMMARY trees "
        f"active med={median(trees):.1f} max={max(trees)} "
        f"tentative med={median(tentative_trees):.1f} max={max(tentative_trees)} "
        f"confirmed med={median(confirmed_trees):.1f} max={max(confirmed_trees)} "
        f"leaves med={median(leaves):.1f} max={max(leaves)}"
    )
    if debug_display_expansion_frontier:
        expansion_stats = [s.expansion_frontier for s in stats]
        leaves_before = [s.leaves_before_expansion for s in expansion_stats]
        leaves_after_expansion = [s.leaves_after_expansion for s in expansion_stats]
        leaves_after_supported = [
            s.leaves_after_post_solve_supported_pruning for s in expansion_stats
        ]
        leaves_after_lifecycle = [s.leaves_after_lifecycle for s in expansion_stats]
        expanded = [s.expanded_leaf_count for s in expansion_stats]
        expanded_tentative = [s.expanded_leaves_tentative for s in expansion_stats]
        expanded_confirmed = [s.expanded_leaves_confirmed for s in expansion_stats]
        child_candidates = [s.local_child_candidates_total for s in expansion_stats]
        children_created = [s.local_children_created_total for s in expansion_stats]
        children_retained = [s.local_children_retained_total for s in expansion_stats]
        topk_supported = [s.retained_topk_supported_leaf_count for s in expansion_stats]
        map_selected = [s.map_selected_leaf_count for s in expansion_stats]
        unsupported_pruned = [s.unsupported_leaf_count_pruned for s in expansion_stats]
        print(
            "SUMMARY expansion_frontier "
            f"leaves_before med={median(leaves_before):.1f} max={max(leaves_before)} "
            "leaves_after_expansion "
            f"med={median(leaves_after_expansion):.1f} max={max(leaves_after_expansion)} "
            "leaves_after_supported_prune "
            f"med={median(leaves_after_supported):.1f} max={max(leaves_after_supported)} "
            "leaves_after_lifecycle "
            f"med={median(leaves_after_lifecycle):.1f} max={max(leaves_after_lifecycle)} "
            f"expanded sum={sum(expanded)} mean={_mean(expanded):.2f} "
            f"expanded_tentative sum={sum(expanded_tentative)} "
            f"expanded_confirmed sum={sum(expanded_confirmed)} "
            f"child_candidates sum={sum(child_candidates)} mean={_mean(child_candidates):.2f} "
            f"children_created sum={sum(children_created)} mean={_mean(children_created):.2f} "
            f"children_retained sum={sum(children_retained)} mean={_mean(children_retained):.2f} "
            f"topk_supported sum={sum(topk_supported)} mean={_mean(topk_supported):.2f} "
            f"map_selected sum={sum(map_selected)} mean={_mean(map_selected):.2f} "
            f"unsupported_pruned sum={sum(unsupported_pruned)}"
        )
    print(
        "SUMMARY clusters "
        f"count med={median(clusters):.1f} max={max(clusters)} "
        f"comb_eval med={median(comb_eval):.1f} max={max(comb_eval)} "
        f"comb_feas med={median(comb_feas):.1f} max={max(comb_feas)} "
        f"globals med={median(rebuilt):.1f} max={max(rebuilt)} "
        f"split_clusters med={median(split_clusters):.1f} max={max(split_clusters)} "
        f"split_ops med={median(split_ops):.1f} max={max(split_ops)}"
    )
    print(
        "SUMMARY timing "
        f"scan_wall_ms med={median(scan_wall_ms):.1f} "
        f"mean={_mean(scan_wall_ms):.1f} "
        f"p65={_percentile(scan_wall_ms, 0.65):.1f} "
        f"p80={_percentile(scan_wall_ms, 0.80):.1f} "
        f"p90={_percentile(scan_wall_ms, 0.90):.1f} "
        f"p95={_percentile(scan_wall_ms, 0.95):.1f} "
        f"max={max(scan_wall_ms):.1f}"
    )
    print(
        "SUMMARY timing_phases "
        f"prep_ctx_ms med={median(prep_ctx_ms):.1f} p65={_percentile(prep_ctx_ms, 0.65):.1f} p80={_percentile(prep_ctx_ms, 0.80):.1f} p90={_percentile(prep_ctx_ms, 0.90):.1f} p95={_percentile(prep_ctx_ms, 0.95):.1f} max={max(prep_ctx_ms):.1f} "
        "pre_expand_validate_ms "
        f"med={median(pre_expand_validate_ms):.1f} p65={_percentile(pre_expand_validate_ms, 0.65):.1f} p80={_percentile(pre_expand_validate_ms, 0.80):.1f} p90={_percentile(pre_expand_validate_ms, 0.90):.1f} p95={_percentile(pre_expand_validate_ms, 0.95):.1f} max={max(pre_expand_validate_ms):.1f} "
        f"expand_ms med={median(expand_ms):.1f} p65={_percentile(expand_ms, 0.65):.1f} p80={_percentile(expand_ms, 0.80):.1f} p90={_percentile(expand_ms, 0.90):.1f} p95={_percentile(expand_ms, 0.95):.1f} max={max(expand_ms):.1f} "
        "expand_hypothesise_ms "
        f"med={median(expand_hypothesise_ms):.1f} p65={_percentile(expand_hypothesise_ms, 0.65):.1f} p80={_percentile(expand_hypothesise_ms, 0.80):.1f} p90={_percentile(expand_hypothesise_ms, 0.90):.1f} p95={_percentile(expand_hypothesise_ms, 0.95):.1f} max={max(expand_hypothesise_ms):.1f} "
        "expand_track_reconstruct_ms "
        f"med={median(expand_track_reconstruct_ms):.1f} p65={_percentile(expand_track_reconstruct_ms, 0.65):.1f} p80={_percentile(expand_track_reconstruct_ms, 0.80):.1f} p90={_percentile(expand_track_reconstruct_ms, 0.90):.1f} p95={_percentile(expand_track_reconstruct_ms, 0.95):.1f} max={max(expand_track_reconstruct_ms):.1f} "
        "expand_update_ms "
        f"med={median(expand_update_ms):.1f} p65={_percentile(expand_update_ms, 0.65):.1f} p80={_percentile(expand_update_ms, 0.80):.1f} p90={_percentile(expand_update_ms, 0.90):.1f} p95={_percentile(expand_update_ms, 0.95):.1f} max={max(expand_update_ms):.1f} "
        "expand_other_ms "
        f"med={median(expand_other_ms):.1f} p65={_percentile(expand_other_ms, 0.65):.1f} p80={_percentile(expand_other_ms, 0.80):.1f} p90={_percentile(expand_other_ms, 0.90):.1f} p95={_percentile(expand_other_ms, 0.95):.1f} max={max(expand_other_ms):.1f} "
        "expand_hypothesise_calls "
        f"med={median(expand_hypothesise_calls):.1f} mean={_mean(expand_hypothesise_calls):.2f} max={max(expand_hypothesise_calls):.1f} "
        "expand_track_reconstruct_calls "
        f"med={median(expand_track_reconstruct_calls):.1f} mean={_mean(expand_track_reconstruct_calls):.2f} max={max(expand_track_reconstruct_calls):.1f} "
        "expand_default_state_fast_path_calls "
        f"med={median(expand_default_state_fast_path_calls):.1f} mean={_mean(expand_default_state_fast_path_calls):.2f} max={max(expand_default_state_fast_path_calls):.1f} "
        "expand_update_calls "
        f"med={median(expand_update_calls):.1f} mean={_mean(expand_update_calls):.2f} max={max(expand_update_calls):.1f} "
        "post_expand_prune_validate_ms "
        f"med={median(post_expand_prune_validate_ms):.1f} p65={_percentile(post_expand_prune_validate_ms, 0.65):.1f} p80={_percentile(post_expand_prune_validate_ms, 0.80):.1f} p90={_percentile(post_expand_prune_validate_ms, 0.90):.1f} p95={_percentile(post_expand_prune_validate_ms, 0.95):.1f} max={max(post_expand_prune_validate_ms):.1f} "
        "cluster_build_solve_ms "
        f"med={median(cluster_build_and_solve_ms):.1f} p65={_percentile(cluster_build_and_solve_ms, 0.65):.1f} p80={_percentile(cluster_build_and_solve_ms, 0.80):.1f} p90={_percentile(cluster_build_and_solve_ms, 0.90):.1f} p95={_percentile(cluster_build_and_solve_ms, 0.95):.1f} max={max(cluster_build_and_solve_ms):.1f} "
        f"post_solve_prune_ms med={median(post_solve_prune_ms):.1f} p65={_percentile(post_solve_prune_ms, 0.65):.1f} p80={_percentile(post_solve_prune_ms, 0.80):.1f} p90={_percentile(post_solve_prune_ms, 0.90):.1f} p95={_percentile(post_solve_prune_ms, 0.95):.1f} max={max(post_solve_prune_ms):.1f} "
        f"map_merge_ms med={median(map_merge_ms):.1f} p65={_percentile(map_merge_ms, 0.65):.1f} p80={_percentile(map_merge_ms, 0.80):.1f} p90={_percentile(map_merge_ms, 0.90):.1f} p95={_percentile(map_merge_ms, 0.95):.1f} max={max(map_merge_ms):.1f} "
        f"nscan_prune_ms med={median(nscan_prune_ms):.1f} p65={_percentile(nscan_prune_ms, 0.65):.1f} p80={_percentile(nscan_prune_ms, 0.80):.1f} p90={_percentile(nscan_prune_ms, 0.90):.1f} p95={_percentile(nscan_prune_ms, 0.95):.1f} max={max(nscan_prune_ms):.1f} "
        f"lifecycle_ms med={median(lifecycle_ms):.1f} p65={_percentile(lifecycle_ms, 0.65):.1f} p80={_percentile(lifecycle_ms, 0.80):.1f} p90={_percentile(lifecycle_ms, 0.90):.1f} p95={_percentile(lifecycle_ms, 0.95):.1f} max={max(lifecycle_ms):.1f} "
        "lifecycle_deleter_track_reconstruct_ms "
        f"med={median(lifecycle_deleter_track_reconstruct_ms):.1f} p65={_percentile(lifecycle_deleter_track_reconstruct_ms, 0.65):.1f} p80={_percentile(lifecycle_deleter_track_reconstruct_ms, 0.80):.1f} p90={_percentile(lifecycle_deleter_track_reconstruct_ms, 0.90):.1f} p95={_percentile(lifecycle_deleter_track_reconstruct_ms, 0.95):.1f} max={max(lifecycle_deleter_track_reconstruct_ms):.1f} "
        "lifecycle_deleter_track_reconstruct_calls "
        f"med={median(lifecycle_deleter_track_reconstruct_calls):.1f} mean={_mean(lifecycle_deleter_track_reconstruct_calls):.2f} max={max(lifecycle_deleter_track_reconstruct_calls):.1f} "
        "lifecycle_default_miss_deleter_fast_path_calls "
        f"med={median(lifecycle_default_miss_deleter_fast_path_calls):.1f} mean={_mean(lifecycle_default_miss_deleter_fast_path_calls):.2f} max={max(lifecycle_default_miss_deleter_fast_path_calls):.1f} "
        "lifecycle_deleter_check_ms "
        f"med={median(lifecycle_deleter_check_ms):.1f} p65={_percentile(lifecycle_deleter_check_ms, 0.65):.1f} p80={_percentile(lifecycle_deleter_check_ms, 0.80):.1f} p90={_percentile(lifecycle_deleter_check_ms, 0.90):.1f} p95={_percentile(lifecycle_deleter_check_ms, 0.95):.1f} max={max(lifecycle_deleter_check_ms):.1f} "
        "lifecycle_other_ms "
        f"med={median(lifecycle_other_ms):.1f} p65={_percentile(lifecycle_other_ms, 0.65):.1f} p80={_percentile(lifecycle_other_ms, 0.80):.1f} p90={_percentile(lifecycle_other_ms, 0.90):.1f} p95={_percentile(lifecycle_other_ms, 0.95):.1f} max={max(lifecycle_other_ms):.1f} "
        f"births_ms med={median(births_ms):.1f} p65={_percentile(births_ms, 0.65):.1f} p80={_percentile(births_ms, 0.80):.1f} p90={_percentile(births_ms, 0.90):.1f} p95={_percentile(births_ms, 0.95):.1f} max={max(births_ms):.1f} "
        f"publication_ms med={median(publication_ms):.1f} p65={_percentile(publication_ms, 0.65):.1f} p80={_percentile(publication_ms, 0.80):.1f} p90={_percentile(publication_ms, 0.90):.1f} p95={_percentile(publication_ms, 0.95):.1f} max={max(publication_ms):.1f} "
        f"cleanup_ms med={median(cleanup_ms):.1f} p65={_percentile(cleanup_ms, 0.65):.1f} p80={_percentile(cleanup_ms, 0.80):.1f} p90={_percentile(cleanup_ms, 0.90):.1f} p95={_percentile(cleanup_ms, 0.95):.1f} max={max(cleanup_ms):.1f} "
        f"other_ms med={median(other_ms):.1f} p65={_percentile(other_ms, 0.65):.1f} p80={_percentile(other_ms, 0.80):.1f} p90={_percentile(other_ms, 0.90):.1f} p95={_percentile(other_ms, 0.95):.1f} max={max(other_ms):.1f}"
    )
    print(
        "SUMMARY memory "
        f"nodes_total med={median(node_count_total):.1f} "
        f"max={max(node_count_total)} "
        f"maxrss_mb final={maxrss_mb[-1]:.1f} "
        f"peak={max(maxrss_mb):.1f}"
    )
    print(
        "SUMMARY births "
        f"candidates med={median(birth_cand):.1f} max={max(birth_cand)} "
        f"tracks_created med={median(birth_created):.1f} mean={_mean(birth_created):.2f} max={max(birth_created)} "
        f"tracks_kept med={median(birth_kept):.1f} mean={_mean(birth_kept):.2f} max={max(birth_kept)}"
    )
    print(
        "SUMMARY nscan "
        f"tracks_in_scope med={median(nscan_tracks_in_scope):.1f} mean={_mean(nscan_tracks_in_scope):.2f} "
        f"committed_now med={median(nscan_tracks_committed):.1f} mean={_mean(nscan_tracks_committed):.2f} "
        f"disagreement med={median(disagree):.1f} mean={_mean(disagree):.2f} max={max(disagree)} "
        f"latest_boundary={last_nscan_boundary_scan_index} "
        f"committed_tracks_total={len(committed_boundary_by_track_id)}"
    )

    miss_hist_all: dict[int, int] = {}
    for scan in stats:
        for misses, count in scan.map_miss_hist.items():
            miss_hist_all[misses] = miss_hist_all.get(misses, 0) + count
    miss_hist_str = (
        "{" + ", ".join(f"{k}: {miss_hist_all[k]}" for k in sorted(miss_hist_all)) + "}"
    )
    print(
        "SUMMARY map "
        f"tracks med={median(map_tracks):.1f} mean={_mean(map_tracks):.2f} "
        f"published med={median(map_published_tracks):.1f} mean={_mean(map_published_tracks):.2f} "
        f"unpublished med={median(map_unpublished_tracks):.1f} mean={_mean(map_unpublished_tracks):.2f} "
        f"used med={median(map_used):.1f} mean={_mean(map_used):.2f} "
        f"unused med={median(map_unused):.1f} mean={_mean(map_unused):.2f} "
        f"hit_rate mean={_mean(map_hit_rate):.3f} "
        f"miss_hist={miss_hist_str}"
    )
