"""Passive TO-MHT scan stats structures and summary reporting helpers."""

from dataclasses import dataclass
import datetime
from statistics import median
from typing import Mapping


@dataclass(frozen=True)
class BirthStats:
    residual_detections_considered: int = 0
    birth_tracks_created: int = 0
    birth_tracks_kept: int = 0
    birth_track_instances_in_beam: int = 0
    globals_with_birth: int = 0
    globals_before_births: int = 0
    globals_after_births: int = 0


@dataclass(frozen=True)
class ScanStats:
    timestamp: datetime.datetime
    scan_wall_ms: float
    maxrss_mb: float
    node_count_total: int
    leaf_instances_in_beam: int
    num_detections: int
    globals_in: int
    globals_expanded: int
    globals_after_unused: int
    globals_after_dedupe: int
    globals_after_beam: int
    nscan_boundary_scan_index: int
    nscan_tracks_in_scope: int
    nscan_tracks_committed: int
    globals_after_births: int
    birth_candidates: int
    birth_tracks_created: int
    birth_tracks_kept: int
    birth_track_instances_in_beam: int
    globals_with_birth: int
    map_tracks: int
    map_used: int
    map_unused: int
    map_miss_hist: dict[int, int]
    map_mean_hit_rate: float


def print_summary_stats(
    *,
    stats: list[ScanStats],
    max_global_hypotheses: int,
    last_nscan_boundary_scan_index: int | None,
    committed_boundary_by_track_id: Mapping[int, int],
) -> None:
    """Print aggregate instrumentation summaries from collected ScanStats."""
    if not stats:
        print("SUMMARY scans=0 (no collected ScanStats)")
        return

    num_scans = len(stats)
    expanded = [s.globals_expanded for s in stats]
    deduped = [s.globals_after_dedupe for s in stats]
    beamed = [s.globals_after_beam for s in stats]
    after_births = [s.globals_after_births for s in stats]
    birth_created = [s.birth_tracks_created for s in stats]
    birth_kept = [s.birth_tracks_kept for s in stats]
    map_tracks = [s.map_tracks for s in stats]
    map_unused = [s.map_unused for s in stats]
    map_used = [s.map_used for s in stats]
    map_hit_rate = [s.map_mean_hit_rate for s in stats]
    scan_wall_ms = [s.scan_wall_ms for s in stats]
    maxrss_mb = [s.maxrss_mb for s in stats]
    node_count_total = [s.node_count_total for s in stats]
    leaf_instances_in_beam = [s.leaf_instances_in_beam for s in stats]
    nscan_tracks_in_scope = [s.nscan_tracks_in_scope for s in stats]
    nscan_tracks_committed = [s.nscan_tracks_committed for s in stats]

    def _mean(values: list[int] | list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values)) / float(len(values))

    max_globals = max_global_hypotheses
    beam_full_pre_births = sum(1 for s in stats if s.globals_after_beam == max_globals)
    beam_full_post_births = sum(
        1 for s in stats if s.globals_after_births == max_globals
    )
    scans_with_births = sum(1 for s in stats if s.birth_tracks_created > 0)
    scans_with_birth_globals = sum(1 for s in stats if s.globals_with_birth > 0)
    scans_birth_push_to_full = sum(
        1
        for s in stats
        if s.globals_after_beam < max_globals
        and s.globals_after_births == max_globals
        and s.globals_with_birth > 0
    )

    print(
        "SUMMARY "
        f"scans={num_scans} "
        f"det_total={sum(s.num_detections for s in stats)} "
        f"det_mean={_mean([s.num_detections for s in stats]):.2f}"
    )
    print(
        "SUMMARY globals "
        f"expanded med={median(expanded):.1f} max={max(expanded)} "
        f"dedup med={median(deduped):.1f} max={max(deduped)} "
        f"beam med={median(beamed):.1f} max={max(beamed)}"
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
        f"leaf_instances_beam med={median(leaf_instances_in_beam):.1f} "
        f"max={max(leaf_instances_in_beam)} "
        f"maxrss_mb final={maxrss_mb[-1]:.1f} "
        f"peak={max(maxrss_mb):.1f}"
    )
    print(
        "SUMMARY beam "
        f"after_births med={median(after_births):.1f} max={max(after_births)} "
        f"full_pre_births={beam_full_pre_births}/{num_scans} ({beam_full_pre_births / num_scans:.1%}) "
        f"full_post_births={beam_full_post_births}/{num_scans} ({beam_full_post_births / num_scans:.1%})"
    )
    print(
        "SUMMARY births "
        f"active={scans_with_births}/{num_scans} ({scans_with_births / num_scans:.1%}) "
        f"tracks_created med={median(birth_created):.1f} mean={_mean(birth_created):.2f} max={max(birth_created)} "
        f"tracks_kept med={median(birth_kept):.1f} mean={_mean(birth_kept):.2f} max={max(birth_kept)} "
        f"globals_with_birth={scans_with_birth_globals}/{num_scans} ({scans_with_birth_globals / num_scans:.1%}) "
        f"birth_pushes_to_full={scans_birth_push_to_full}/{num_scans} ({scans_birth_push_to_full / num_scans:.1%})"
    )
    print(
        "SUMMARY nscan "
        f"tracks_in_scope med={median(nscan_tracks_in_scope):.1f} mean={_mean(nscan_tracks_in_scope):.2f} "
        f"committed_now med={median(nscan_tracks_committed):.1f} mean={_mean(nscan_tracks_committed):.2f} "
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
