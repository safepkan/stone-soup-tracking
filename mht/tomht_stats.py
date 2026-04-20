"""Passive TO-MHT scan stats structures and summary reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from statistics import median
from typing import Mapping

from .tomht_model import NScanCommitmentSnapshot


@dataclass(frozen=True)
class BirthStats:
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
    historical_relaxation_attempts: int = 0
    historical_relaxation_successes: int = 0
    historical_relaxed_keys_total: int = 0


@dataclass(frozen=True)
class ScanTimingBreakdown:
    """Per-scan wall-time breakdown across major update pipeline phases."""

    prep_ctx_ms: float = 0.0
    pre_expand_validate_ms: float = 0.0
    expand_ms: float = 0.0
    expand_hypothesise_calls: int = 0
    expand_hypothesise_ms: float = 0.0
    expand_update_calls: int = 0
    expand_update_ms: float = 0.0
    post_expand_prune_validate_ms: float = 0.0
    births_ms: float = 0.0
    cluster_build_and_solve_ms: float = 0.0
    post_solve_prune_ms: float = 0.0
    map_merge_ms: float = 0.0
    nscan_lifecycle_ms: float = 0.0
    cleanup_ms: float = 0.0


@dataclass(frozen=True)
class ScanStats:
    scan_index: int
    timestamp: datetime.datetime
    scan_wall_ms: float
    maxrss_mb: float
    node_count_total: int
    active_trees: int
    active_leaves: int
    num_detections: int
    cluster_count: int
    combinations_evaluated: int
    feasible_combinations: int
    rebuilt_globals_stored: int
    nscan_disagreement_total: int
    overload_split_clusters: int
    overload_split_operations: int
    historical_relaxation_attempts: int
    historical_relaxation_successes: int
    historical_relaxed_keys_total: int
    nscan_boundary_scan_index: int
    nscan_tracks_in_scope: int
    nscan_tracks_committed: int
    birth_candidates: int
    birth_tracks_created: int
    birth_tracks_kept: int
    map_tracks: int
    map_used: int
    map_unused: int
    map_miss_hist: dict[int, int]
    map_mean_hit_rate: float
    timing_breakdown: ScanTimingBreakdown = field(default_factory=ScanTimingBreakdown)


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
        + float(breakdown.nscan_lifecycle_ms)
        + float(breakdown.cleanup_ms)
    )
    phase_other_ms = max(float(scan_stats.scan_wall_ms) - phase_accounted_ms, 0.0)
    expand_other_ms = max(
        float(breakdown.expand_ms)
        - float(breakdown.expand_hypothesise_ms)
        - float(breakdown.expand_update_ms),
        0.0,
    )

    print(
        f"SCAN scan={scan_stats.scan_index} t={timestamp} "
        f"det={scan_stats.num_detections} "
        f"trees={scan_stats.active_trees} leaves={scan_stats.active_leaves} "
        f"clusters={scan_stats.cluster_count} "
        f"comb_eval={scan_stats.combinations_evaluated} "
        f"comb_feas={scan_stats.feasible_combinations} "
        f"rebuilt_globals={scan_stats.rebuilt_globals_stored} "
        f"split_clusters={scan_stats.overload_split_clusters} "
        f"split_ops={scan_stats.overload_split_operations} "
        f"hist_relax_attempts={scan_stats.historical_relaxation_attempts} "
        f"hist_relax_ok={scan_stats.historical_relaxation_successes} "
        f"hist_relax_keys={scan_stats.historical_relaxed_keys_total} "
        f"nscan boundary={scan_stats.nscan_boundary_scan_index} "
        f"in_scope={scan_stats.nscan_tracks_in_scope} "
        f"committed_now={scan_stats.nscan_tracks_committed} "
        f"committed_total={len(nscan_snapshot.committed_boundary_by_track_id)} "
        f"disagree={scan_stats.nscan_disagreement_total} "
        f"births cand={scan_stats.birth_candidates} "
        f"tracks_created={scan_stats.birth_tracks_created} "
        f"tracks_kept={scan_stats.birth_tracks_kept} "
        f"MAP tracks={scan_stats.map_tracks} "
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
        f"expand_update_calls={breakdown.expand_update_calls} "
        f"expand_update_ms={breakdown.expand_update_ms:.3f} "
        f"expand_other_ms={expand_other_ms:.3f} "
        "post_expand_prune_validate_ms="
        f"{breakdown.post_expand_prune_validate_ms:.3f} "
        f"births_ms={breakdown.births_ms:.3f} "
        f"cluster_build_solve_ms={breakdown.cluster_build_and_solve_ms:.3f} "
        f"post_solve_prune_ms={breakdown.post_solve_prune_ms:.3f} "
        f"map_merge_ms={breakdown.map_merge_ms:.3f} "
        f"nscan_lifecycle_ms={breakdown.nscan_lifecycle_ms:.3f} "
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


def print_summary_stats(
    *,
    stats: list[ScanStats],
    max_global_hypotheses: int,
    last_nscan_boundary_scan_index: int | None,
    committed_boundary_by_track_id: Mapping[int, int],
) -> None:
    """Print aggregate instrumentation summaries from collected ScanStats."""
    del max_global_hypotheses
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
    leaves = [s.active_leaves for s in stats]
    clusters = [s.cluster_count for s in stats]
    comb_eval = [s.combinations_evaluated for s in stats]
    comb_feas = [s.feasible_combinations for s in stats]
    rebuilt = [s.rebuilt_globals_stored for s in stats]
    split_clusters = [s.overload_split_clusters for s in stats]
    split_ops = [s.overload_split_operations for s in stats]
    hist_relax_attempts = [s.historical_relaxation_attempts for s in stats]
    hist_relax_ok = [s.historical_relaxation_successes for s in stats]
    hist_relax_keys = [s.historical_relaxed_keys_total for s in stats]
    disagree = [s.nscan_disagreement_total for s in stats]

    birth_created = [s.birth_tracks_created for s in stats]
    birth_kept = [s.birth_tracks_kept for s in stats]
    birth_cand = [s.birth_candidates for s in stats]

    map_tracks = [s.map_tracks for s in stats]
    map_unused = [s.map_unused for s in stats]
    map_used = [s.map_used for s in stats]
    map_hit_rate = [s.map_mean_hit_rate for s in stats]

    scan_wall_ms = [s.scan_wall_ms for s in stats]
    prep_ctx_ms = [s.timing_breakdown.prep_ctx_ms for s in stats]
    pre_expand_validate_ms = [s.timing_breakdown.pre_expand_validate_ms for s in stats]
    expand_ms = [s.timing_breakdown.expand_ms for s in stats]
    expand_hypothesise_ms = [s.timing_breakdown.expand_hypothesise_ms for s in stats]
    expand_update_ms = [s.timing_breakdown.expand_update_ms for s in stats]
    expand_other_ms = [
        max(
            float(s.timing_breakdown.expand_ms)
            - float(s.timing_breakdown.expand_hypothesise_ms)
            - float(s.timing_breakdown.expand_update_ms),
            0.0,
        )
        for s in stats
    ]
    expand_hypothesise_calls = [
        float(s.timing_breakdown.expand_hypothesise_calls) for s in stats
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
    nscan_lifecycle_ms = [s.timing_breakdown.nscan_lifecycle_ms for s in stats]
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
                + float(s.timing_breakdown.nscan_lifecycle_ms)
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
        f"leaves med={median(leaves):.1f} max={max(leaves)}"
    )
    print(
        "SUMMARY clusters "
        f"count med={median(clusters):.1f} max={max(clusters)} "
        f"comb_eval med={median(comb_eval):.1f} max={max(comb_eval)} "
        f"comb_feas med={median(comb_feas):.1f} max={max(comb_feas)} "
        f"globals med={median(rebuilt):.1f} max={max(rebuilt)} "
        f"split_clusters med={median(split_clusters):.1f} max={max(split_clusters)} "
        f"split_ops med={median(split_ops):.1f} max={max(split_ops)} "
        "hist_relax_attempts med="
        f"{median(hist_relax_attempts):.1f} max={max(hist_relax_attempts)} "
        f"hist_relax_ok med={median(hist_relax_ok):.1f} max={max(hist_relax_ok)} "
        f"hist_relax_keys med={median(hist_relax_keys):.1f} max={max(hist_relax_keys)}"
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
        "expand_update_ms "
        f"med={median(expand_update_ms):.1f} p65={_percentile(expand_update_ms, 0.65):.1f} p80={_percentile(expand_update_ms, 0.80):.1f} p90={_percentile(expand_update_ms, 0.90):.1f} p95={_percentile(expand_update_ms, 0.95):.1f} max={max(expand_update_ms):.1f} "
        "expand_other_ms "
        f"med={median(expand_other_ms):.1f} p65={_percentile(expand_other_ms, 0.65):.1f} p80={_percentile(expand_other_ms, 0.80):.1f} p90={_percentile(expand_other_ms, 0.90):.1f} p95={_percentile(expand_other_ms, 0.95):.1f} max={max(expand_other_ms):.1f} "
        "expand_hypothesise_calls "
        f"med={median(expand_hypothesise_calls):.1f} mean={_mean(expand_hypothesise_calls):.2f} max={max(expand_hypothesise_calls):.1f} "
        "expand_update_calls "
        f"med={median(expand_update_calls):.1f} mean={_mean(expand_update_calls):.2f} max={max(expand_update_calls):.1f} "
        "post_expand_prune_validate_ms "
        f"med={median(post_expand_prune_validate_ms):.1f} p65={_percentile(post_expand_prune_validate_ms, 0.65):.1f} p80={_percentile(post_expand_prune_validate_ms, 0.80):.1f} p90={_percentile(post_expand_prune_validate_ms, 0.90):.1f} p95={_percentile(post_expand_prune_validate_ms, 0.95):.1f} max={max(post_expand_prune_validate_ms):.1f} "
        f"births_ms med={median(births_ms):.1f} p65={_percentile(births_ms, 0.65):.1f} p80={_percentile(births_ms, 0.80):.1f} p90={_percentile(births_ms, 0.90):.1f} p95={_percentile(births_ms, 0.95):.1f} max={max(births_ms):.1f} "
        "cluster_build_solve_ms "
        f"med={median(cluster_build_and_solve_ms):.1f} p65={_percentile(cluster_build_and_solve_ms, 0.65):.1f} p80={_percentile(cluster_build_and_solve_ms, 0.80):.1f} p90={_percentile(cluster_build_and_solve_ms, 0.90):.1f} p95={_percentile(cluster_build_and_solve_ms, 0.95):.1f} max={max(cluster_build_and_solve_ms):.1f} "
        f"post_solve_prune_ms med={median(post_solve_prune_ms):.1f} p65={_percentile(post_solve_prune_ms, 0.65):.1f} p80={_percentile(post_solve_prune_ms, 0.80):.1f} p90={_percentile(post_solve_prune_ms, 0.90):.1f} p95={_percentile(post_solve_prune_ms, 0.95):.1f} max={max(post_solve_prune_ms):.1f} "
        f"map_merge_ms med={median(map_merge_ms):.1f} p65={_percentile(map_merge_ms, 0.65):.1f} p80={_percentile(map_merge_ms, 0.80):.1f} p90={_percentile(map_merge_ms, 0.90):.1f} p95={_percentile(map_merge_ms, 0.95):.1f} max={max(map_merge_ms):.1f} "
        f"nscan_lifecycle_ms med={median(nscan_lifecycle_ms):.1f} p65={_percentile(nscan_lifecycle_ms, 0.65):.1f} p80={_percentile(nscan_lifecycle_ms, 0.80):.1f} p90={_percentile(nscan_lifecycle_ms, 0.90):.1f} p95={_percentile(nscan_lifecycle_ms, 0.95):.1f} max={max(nscan_lifecycle_ms):.1f} "
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
        f"used med={median(map_used):.1f} mean={_mean(map_used):.2f} "
        f"unused med={median(map_unused):.1f} mean={_mean(map_unused):.2f} "
        f"hit_rate mean={_mean(map_hit_rate):.3f} "
        f"miss_hist={miss_hist_str}"
    )
