"""Passive TO-MHT scan stats structures and summary reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from statistics import median
from typing import Mapping

from mht.tomht_model import NScanCommitmentSnapshot


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


@dataclass(frozen=True)
class ScanStats:
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


def print_scan_stats(
    *,
    timestamp: datetime.datetime,
    scan_stats: ScanStats,
    nscan_snapshot: NScanCommitmentSnapshot,
    debug_display_map_miss_hist: bool,
) -> None:
    """Print one per-scan instrumentation block from precomputed ScanStats."""
    print(
        f"SCAN t={timestamp} det={scan_stats.num_detections} "
        f"trees={scan_stats.active_trees} leaves={scan_stats.active_leaves} "
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
        f"used={scan_stats.map_used} unused={scan_stats.map_unused} "
        f"hit_rate={scan_stats.map_mean_hit_rate:.2f}"
    )
    print(f"SCAN_TIMING t={timestamp} wall_ms={scan_stats.scan_wall_ms:.3f}")
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

    trees = [s.active_trees for s in stats]
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
    map_unused = [s.map_unused for s in stats]
    map_used = [s.map_used for s in stats]
    map_hit_rate = [s.map_mean_hit_rate for s in stats]

    scan_wall_ms = [s.scan_wall_ms for s in stats]
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
        f"split_ops med={median(split_ops):.1f} max={max(split_ops)}"
    )
    print(
        "SUMMARY timing "
        f"scan_wall_ms med={median(scan_wall_ms):.1f} "
        f"mean={_mean(scan_wall_ms):.1f} "
        f"max={max(scan_wall_ms):.1f}"
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
