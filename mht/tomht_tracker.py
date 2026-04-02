"""Track-oriented MHT with persistent track trees and per-scan rebuilt globals.

Typical usage pattern:
```python
tracker = TOMHTTracker(hypothesiser,updater,initiator=initiator,params=params)
t,map_tracks = tracker.update_tracker(time,detections)
tracker.add_external_starts(t,confirmed_starts_at_t)  # optional, same timestamp
map_tracks = tracker.tracks  # or use ``map_tracks`` from update_tracker()
```

Core control APIs:
- ``update_tracker(time,detections)``: process one scan and return MAP output.
- ``tracks``: current MAP output as Stone Soup ``Track`` objects.
- ``add_external_starts(time,starts)``: inject confirmed external starts after the
  same-timestamp ``update_tracker()`` call.

Track-oriented architecture in this phase:
- persistent state is explicit ``TrackTree`` objects and their active leaves,
- globals are rebuilt per cluster on every scan from current leaves,
- the previous scan's explicit global list is not the persistent search frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import datetime
import heapq
import os
import resource
import sys
import time as wall_clock
from itertools import product
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ordered_set import OrderedSet

import numpy as np

from stonesoup.base import Property
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.base import Initiator
from stonesoup.types.detection import Detection
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.tracker.base import Tracker, _TrackerMixInUpdate
from stonesoup.updater.base import Updater

from mht.tomht_model import (
    ClusterRebuildSnapshot,
    DetectionKey,
    GlobalHypothesis,
    MAPHypothesisSnapshot,
    NScanCommitmentSnapshot,
    TrackHypothesisNode,
    TrackTree,
)
from mht.tomht_output import reconstruct_track_from_leaf_node
from mht.tomht_scoring import (
    ScoringModel,
    make_default_scoring_model,
    maybe_log_scoring_diagnostics,
)
from mht.tomht_stats import (
    BirthStats,
    RebuildStats,
    ScanStats,
    print_scan_stats as print_scan_stats_report,
    print_summary_stats as print_summary_stats_report,
)


# ============================================================================
# Tracker-Local Support Structures / Params
# ============================================================================


@dataclass(frozen=True)
class TOMHTParams:
    """Flat tracker configuration for the track-oriented TO-MHT implementation.

    Stable operational controls:
    - per-leaf local branching and local frontier safety-valves,
    - whole-track miss termination after N-scan pruning,
    - MAP-only N-scan pruning window,
    - optional debug/stat visibility toggles.

    Compatibility note:
    ``max_global_hypotheses`` is retained as a cap for how many rebuilt globals
    are kept per cluster for debug/snapshot storage; it is no longer a persistent
    beam frontier carried scan-to-scan.
    """

    # Local expansion / lifecycle controls.
    max_children_per_track: int = 5
    # Optional pre-solve per-tree frontier cap used only as a safety valve.
    # The high default keeps this in a tractability guardrail role, not as the
    # primary pruning mechanism.
    max_leaves_per_track_tree: int | None = 500
    # Base miss threshold used for post-N-scan whole-track termination.
    # Effective threshold uses an N-scan-aware floor (see helper below).
    max_missed: int = 5
    # Whole-track miss termination mode applied after N-scan pruning.
    # - "all_active_leaves": terminate only if all active leaves exceed threshold
    # - "map_leaf": terminate if MAP leaf exceeds threshold
    # - "global_k_leaves": terminate if all retained rebuilt-global leaves exceed
    #   threshold (fallback to active leaves if unavailable after N-scan)
    track_miss_termination_mode: str = "map_leaf"

    # Rebuilt-global storage cap (debug/inspection cap, not persistent beam state).
    max_global_hypotheses: int = 20
    # Optional hard cap for one cluster's projected Cartesian leaf combinations.
    # If exceeded, cluster rebuild fails explicitly (no adaptive trimming/retry).
    max_projected_cluster_combinations: int | None = None
    # Optional approximate overload mitigation:
    # when a cluster's projected Cartesian combinations exceed this threshold,
    # iteratively sever weakest conflict edges and solve resulting subclusters.
    overload_split_enabled: bool = True
    overload_split_projected_combination_threshold: int | None = 500_000
    overload_split_max_edge_removals_per_cluster: int | None = None
    # Narrow safety-net: if an exact cluster is infeasible, allow relaxation only
    # for forced historical keys that are already shared across tracks.
    historical_conflict_relaxation_enabled: bool = True

    # Scoring / numerical behavior.
    scoring_mode: str = "beta_ratio"
    log_epsilon: float = 1e-12
    prob_gate: float = 0.99

    # MAP-only N-scan pruning: boundary is b = k - N.
    ns_scan_window: int = 6

    # Internal birth handling (kept intentionally simple in this phase).
    max_births_per_scan: int = 2
    birth_log_penalty: float = 8.0
    unused_det_log_penalty: float = 0.2
    # Birth load guards: skip births once frontier growth is already high.
    birth_skip_if_active_trees_above: int | None = 40
    birth_skip_if_active_leaves_above: int | None = 200

    # Birth sanity guards.
    birth_max_abs_pos: float = 1e5
    birth_max_covar_trace: float = 1e12

    # Debug / instrumentation toggles.
    debug_display_detections: bool = False
    debug_display_scan_stats: bool = True
    debug_display_hypotheses: bool = True
    debug_display_births: bool = True
    debug_display_map_miss_hist: bool = False
    debug_births_max: int = 5
    debug_globals_max: int = 5
    collect_stats: bool = True


@dataclass(frozen=True)
class ScanContext:
    """Per-scan context passed into scoring and pipeline helpers."""

    scan_index: int
    timestamp: datetime.datetime
    detections: list[Detection]
    det_index_by_obj: dict[int, int]


@dataclass(frozen=True)
class LocalChildCandidate:
    """One retained local child candidate produced from one leaf expansion."""

    track_id: int
    child_node: TrackHypothesisNode
    used_det_key: DetectionKey | None
    log_delta: float


@dataclass(frozen=True)
class _ClusterWorkItem:
    """Transient per-scan cluster build input."""

    cluster_id: int
    track_ids: tuple[int, ...]
    current_scan_det_keys_by_track_id: dict[int, set[DetectionKey]]
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...]
    overload_split_origin_cluster_id: int | None = None


@dataclass(frozen=True)
class _ClusterUnusedScoreContext:
    """Precomputed cluster-local context for unused-detection scoring."""

    local_ctx: ScanContext
    local_slot_by_global_det_index: dict[int, int]


@dataclass(frozen=True)
class _OverloadSplitRemovedEdge:
    """One removed conflict-graph edge during overload decomposition."""

    left_track_id: int
    right_track_id: int
    shared_history_key_count: int


@dataclass(frozen=True)
class _OverloadSplitSummary:
    """Compact instrumentation for one original cluster overload split pass."""

    original_cluster_id: int
    original_track_ids: tuple[int, ...]
    projected_before: int
    projected_threshold: int
    removed_edges: tuple[_OverloadSplitRemovedEdge, ...]
    resulting_subclusters: tuple[tuple[int, ...], ...]
    projected_after_by_subcluster: tuple[int, ...]
    stopping_reason: str


@dataclass(frozen=True)
class _ClusterRebuildResult:
    """Cluster rebuild result with narrow historical-relaxation bookkeeping."""

    snapshot: ClusterRebuildSnapshot
    historical_relaxation_attempted: bool = False
    historical_relaxation_succeeded: bool = False
    historical_relaxed_key_count: int = 0


@dataclass(frozen=True)
class _ClusterSolveInput:
    """Prepared inputs for one internal cluster-solver call."""

    cluster: _ClusterWorkItem
    ctx: ScanContext
    leaf_options: list[list[TrackHypothesisNode]]
    cluster_universe: set[DetectionKey]
    unused_score_context: _ClusterUnusedScoreContext | None


@dataclass(frozen=True)
class _ClusterSolveOutcome:
    """Solver outcome with optional historical-relaxation bookkeeping."""

    kept_globals: tuple[GlobalHypothesis, ...]
    combinations_evaluated: int
    feasible_combinations: int
    historical_relaxation_attempted: bool = False
    historical_relaxation_succeeded: bool = False
    historical_relaxed_keys: frozenset[DetectionKey] = frozenset()


@dataclass(frozen=True)
class _MapNScanPruningPlan:
    """Planned MAP-only N-scan choices and diagnostics before mutation."""

    boundary_scan_index: int
    root_before_by_track_id: dict[int, TrackHypothesisNode]
    map_choice_by_track_id: dict[int, int]
    disagreement_total: int
    updated_snapshots: list[ClusterRebuildSnapshot]


type _ClusterTopKHeap = list[tuple[float, int, GlobalHypothesis]]


# ============================================================================
# Tracker Implementation
# ============================================================================


class TOMHTTracker(_TrackerMixInUpdate, Tracker):
    """Public TO-MHT tracker contract with Stone Soup boundary objects.

    Operational APIs:
    - ``update_tracker(time,detections)``
    - ``tracks``
    - ``add_external_starts(time,starts)``

    Core per-scan pipeline order:
    1. Sort detections deterministically.
    2. Expand active leaves in every persistent ``TrackTree``.
    3. Apply simple local filtering (drop trees with no surviving active leaves).
    4. Optionally create internal birth trees from detections unused by the union
       of surviving active leaves after Step 3.
    5. Recompute measurement-exclusivity clusters from current trees.
    6. Rebuild feasible globals per cluster via exhaustive enumeration and choose
       MAP per cluster; overloaded clusters may first be approximately decomposed
       by severing weakest full-history conflict edges.
    7. Post-solve prune each cluster tree frontier to leaves supported by at
       least one retained rebuilt top-K global for that cluster.
    8. Merge cluster MAP selections into full-scan MAP, then apply MAP-only
       N-scan tree pruning and whole-track miss-based lifecycle.
    9. Keep last-scan debug snapshots and return MAP output tracks.

    Behavior notes for readability:
    - Exact behavior: cluster feasibility checks and exclusivity constraints use
      full detection-history keys on active leaves.
    - Safety valves: pre-solve per-tree leaf capping and birth load guards.
    - Approximation paths: overload cluster decomposition and optional narrow
      historical-conflict relaxation.
    - Inspection/debug retention: last-scan cluster snapshots and scan stats.
    """

    ASSOC_PAD = -1
    ASSOC_MISS = -2

    hypothesiser: PDAHypothesiser = Property(
        PDAHypothesiser, doc="Hypothesiser used to branch per-track hypotheses."
    )
    updater: Updater = Property(
        Updater, doc="Updater used to generate posteriors from selected hypotheses."
    )

    _last_nscan_boundary_scan_index: int | None
    _last_nscan_tracks_in_scope: int
    _last_nscan_committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]
    _committed_boundary_by_track_id: dict[int, int]
    _committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]

    # =========================================================================
    # Public API
    # =========================================================================

    def __init__(
        self,
        hypothesiser: PDAHypothesiser,
        updater: Updater,
        *,
        detector: Any | None = None,
        initiator: Initiator | None = None,
        params: TOMHTParams = TOMHTParams(),
        scoring_model: ScoringModel | None = None,
    ) -> None:
        """Construct the tracker with Stone Soup components and TO-MHT params."""
        super().__init__(hypothesiser, updater)
        self.detector = detector
        self.params = params
        self.initiator = initiator
        if scoring_model is None:
            scoring_model = make_default_scoring_model(
                hypothesiser=hypothesiser,
                scoring_mode=params.scoring_mode,
                log_epsilon=params.log_epsilon,
                prob_gate=params.prob_gate,
                unused_det_log_penalty=params.unused_det_log_penalty,
                birth_log_penalty=params.birth_log_penalty,
            )
        self.scoring_model = scoring_model
        maybe_log_scoring_diagnostics(self.scoring_model)

        self._next_track_id = 0
        self._next_node_id = 0

        # Persistent tracker state.
        self._nodes_by_id: dict[int, TrackHypothesisNode] = {}
        self.track_trees_by_track_id: dict[int, TrackTree] = {}

        # Last-scan rebuilt artifacts retained for inspection only.
        self._last_cluster_snapshots: list[ClusterRebuildSnapshot] = []
        self.global_hypotheses: list[GlobalHypothesis] = [
            GlobalHypothesis(leaf_nodes_by_track_id={}, log_weight=0.0)
        ]
        self._last_map_global: GlobalHypothesis = self.global_hypotheses[0]

        # N-scan bookkeeping snapshots.
        self._last_nscan_boundary_scan_index = None
        self._last_nscan_tracks_in_scope = 0
        self._last_nscan_committed_ancestor_by_track_id = {}
        self._committed_boundary_by_track_id = {}
        self._committed_ancestor_by_track_id = {}

        self._last_update_timestamp: datetime.datetime | None = None
        self._last_scan_index: int | None = None
        self._last_unused_detections: list[Detection] = []

        self.last_scan_stats: ScanStats | None = None
        self._stats: list[ScanStats] = []
        self.reset_stats()

    def reset_stats(self) -> None:
        """Clear collected ScanStats and the last per-scan snapshot."""
        self._stats = []
        self.last_scan_stats = None

    @property
    def tracks(self) -> set[Track]:
        """Current MAP output as Stone Soup ``Track`` objects."""
        return self.get_map_output_tracks()

    def update_tracker(
        self,
        time: datetime.datetime,
        detections: Iterable[Detection],
    ) -> tuple[datetime.datetime, set[Track]]:
        """Run one scan update and return ``(time, MAP tracks)``."""
        scan_wall_start_ns = wall_clock.perf_counter_ns()

        scan_index = (
            0 if self._last_scan_index is None else int(self._last_scan_index) + 1
        )
        det_list = self._sorted_detections(detections)
        det_index_by_obj = {id(det): i for i, det in enumerate(det_list)}
        ctx = ScanContext(
            scan_index=scan_index,
            timestamp=time,
            detections=det_list,
            det_index_by_obj=det_index_by_obj,
        )

        self._maybe_validate_pruning_feasibility(
            stage="pre_local_expansion",
            ctx=ctx,
        )

        # 1) Expand every tree locally.
        self._expand_all_track_trees(ctx)

        # 2) Simple lifecycle handling.
        self._remove_empty_trees()
        self._maybe_validate_pruning_feasibility(
            stage="post_local_pruning",
            ctx=ctx,
        )

        # 3) Internal births from Step-2 residual detections.
        birth_stats = self._run_internal_births(ctx)

        # 4) Build clusters and rebuild globals per cluster (fresh each scan).
        cluster_work = self._build_track_clusters(ctx)
        cluster_snapshots, rebuild_stats = self._rebuild_cluster_globals(
            cluster_work, ctx
        )

        # 5) Post-solve cluster-local supported-leaf pruning from rebuilt top-K.
        self._apply_post_solve_supported_leaf_pruning(cluster_snapshots)
        self._maybe_validate_pruning_feasibility(
            stage="post_supported_leaf_pruning",
            ctx=ctx,
        )

        map_global = self._merge_cluster_map_globals(cluster_snapshots)

        # 6) MAP-only N-scan pruning on explicit trees + disagreement stats.
        (
            nscan_boundary_scan_index,
            nscan_tracks_in_scope,
            nscan_tracks_committed,
            disagreement_total,
            cluster_snapshots,
        ) = self._apply_map_n_scan_pruning(
            scan_index=scan_index,
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
        )
        map_global = self._apply_post_n_scan_track_miss_lifecycle(
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
            scan_index=scan_index,
        )
        self._maybe_validate_pruning_feasibility(
            stage="post_n_scan_pruning",
            ctx=ctx,
        )

        rebuild_stats = replace(
            rebuild_stats,
            nscan_disagreement_total=disagreement_total,
        )
        self._last_cluster_snapshots = cluster_snapshots

        # Keep one full-scan MAP global in compatibility slot for old inspection paths.
        self._last_map_global = map_global
        self.global_hypotheses = [map_global]

        # 7) Reclaim node storage not reachable from surviving roots/leaves/commitments.
        self._cleanup_unreachable_nodes()

        # 8) Post-scan instrumentation.
        scan_wall_ms = (wall_clock.perf_counter_ns() - scan_wall_start_ns) / 1e6
        maxrss_mb = self._get_process_maxrss_mb()
        node_count_total = len(self._nodes_by_id)
        active_leaves = sum(
            len(tree.active_leaf_node_ids)
            for tree in self.track_trees_by_track_id.values()
        )

        self._run_scan_instrumentation(
            ctx=ctx,
            scan_wall_ms=scan_wall_ms,
            maxrss_mb=maxrss_mb,
            node_count_total=node_count_total,
            active_trees=len(self.track_trees_by_track_id),
            active_leaves=active_leaves,
            rebuild_stats=rebuild_stats,
            nscan_boundary_scan_index=nscan_boundary_scan_index,
            nscan_tracks_in_scope=nscan_tracks_in_scope,
            nscan_tracks_committed=nscan_tracks_committed,
            birth_stats=birth_stats,
        )

        self._last_update_timestamp = time
        self._last_scan_index = scan_index
        map_output_tracks = self.get_map_output_tracks()
        return time, map_output_tracks

    def add_external_starts(
        self,
        time: datetime.datetime,
        starts: Iterable[Track],
    ) -> None:
        """Insert externally confirmed starts as new single-node track trees."""
        self._validate_external_starts_timestamp(time)
        start_list = list(starts)
        if not start_list:
            return

        for start in start_list:
            root = self._make_external_start_root(start, time)
            tree = TrackTree(
                track_id=root.track_id,
                root_node_id=root.node_id,
                active_leaf_node_ids={root.node_id},
                root_source="external_start",
            )
            self.track_trees_by_track_id[root.track_id] = tree

        # External starts are assumed to be from currently unused detections,
        # so add them directly to the last MAP view.
        merged = dict(self._last_map_global.leaf_nodes_by_track_id)
        for track_id, tree in self.track_trees_by_track_id.items():
            if track_id in merged:
                continue
            if len(tree.active_leaf_node_ids) != 1:
                continue
            only_leaf_id = next(iter(tree.active_leaf_node_ids))
            merged[track_id] = self._nodes_by_id[only_leaf_id]

        self._last_map_global = GlobalHypothesis(
            leaf_nodes_by_track_id=merged,
            log_weight=self._last_map_global.log_weight,
        )
        self.global_hypotheses = [self._last_map_global]

    def get_unused_detections(self) -> list[Detection]:
        """Return residual detections from the most recent completed update."""
        if self._last_update_timestamp is None:
            raise RuntimeError(
                "get_unused_detections() requires a completed update_tracker() first."
            )
        return list(self._last_unused_detections)

    # --- Read-only inspection / reporting helpers ---

    def get_map_output_tracks(self) -> set[Track]:
        """Return current MAP outputs as Stone Soup ``Track`` objects."""
        map_snapshot = self.get_map_hypothesis_snapshot()
        if map_snapshot is None:
            return set()
        return {
            reconstruct_track_from_leaf_node(leaf_node)
            for leaf_node in map_snapshot.leaf_nodes_by_track_id.values()
        }

    def get_map_hypothesis_snapshot(self) -> MAPHypothesisSnapshot | None:
        """Return read-only node-native MAP state for inspection/debug."""
        if self._last_map_global is None:
            return None
        return MAPHypothesisSnapshot(
            log_weight=float(self._last_map_global.log_weight),
            leaf_nodes_by_track_id=MappingProxyType(
                dict(self._last_map_global.leaf_nodes_by_track_id)
            ),
        )

    def get_n_scan_commitment_snapshot(self) -> NScanCommitmentSnapshot:
        """Return read-only MAP-based N-scan pruning bookkeeping."""
        return NScanCommitmentSnapshot(
            boundary_scan_index=self._last_nscan_boundary_scan_index,
            tracks_in_scope=int(self._last_nscan_tracks_in_scope),
            latest_committed_ancestor_by_track_id=dict(
                self._last_nscan_committed_ancestor_by_track_id
            ),
            committed_boundary_by_track_id=dict(self._committed_boundary_by_track_id),
            committed_ancestor_by_track_id=dict(self._committed_ancestor_by_track_id),
        )

    def get_last_cluster_snapshots(self) -> tuple[ClusterRebuildSnapshot, ...]:
        """Return the most recent per-scan rebuilt-cluster snapshots."""
        return tuple(self._last_cluster_snapshots)

    def get_track_tree_snapshot(self) -> Mapping[int, dict[str, object]]:
        """Return a read-only snapshot of current persistent tree roots/leaves."""
        out: dict[int, dict[str, object]] = {}
        for track_id, tree in sorted(self.track_trees_by_track_id.items()):
            out[track_id] = {
                "root_node_id": int(tree.root_node_id),
                "active_leaf_node_ids": tuple(sorted(tree.active_leaf_node_ids)),
                "root_source": tree.root_source,
            }
        return MappingProxyType(out)

    def print_summary_stats(self) -> None:
        """Print aggregate instrumentation summaries from collected ScanStats."""
        print_summary_stats_report(
            stats=self._stats,
            max_global_hypotheses=self.params.max_global_hypotheses,
            last_nscan_boundary_scan_index=self._last_nscan_boundary_scan_index,
            committed_boundary_by_track_id=self._committed_boundary_by_track_id,
        )

    # =========================================================================
    # Scan Pipeline Utilities
    # =========================================================================

    @staticmethod
    def _det_sort_key(det: Detection) -> tuple:
        """Stable per-scan ordering key for detections."""
        ts = getattr(det, "timestamp", None)
        ts_key: tuple[int, float | str]
        if ts is None:
            ts_key = (0, 0.0)
        elif hasattr(ts, "timestamp"):
            ts_key = (1, float(ts.timestamp()))
        elif isinstance(ts, (int, float)):
            ts_key = (2, float(ts))
        else:
            ts_key = (3, str(ts))

        vec = np.asarray(det.state_vector).ravel()

        def _elem_key(x) -> tuple[int, float | str]:
            try:
                xf = float(x)
                if np.isfinite(xf):
                    return (0, xf)
                return (0, float("inf"))
            except Exception:
                return (1, str(x))

        vec_key = tuple(_elem_key(x) for x in vec)
        return (ts_key, len(vec_key), vec_key)

    def _sorted_detections(self, detections: Iterable[Detection]) -> list[Detection]:
        """Return a deterministic list copy of scan detections."""
        det_list = list(detections)
        det_list.sort(key=self._det_sort_key)
        return det_list

    @staticmethod
    def _get_process_maxrss_mb() -> float:
        """Return process peak RSS in MB with platform-normalized units."""
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return float(ru.ru_maxrss) / (1024.0 * 1024.0)
        return float(ru.ru_maxrss) / 1024.0

    @staticmethod
    def _current_scan_det_indices_from_keys(
        keys: Iterable[DetectionKey], scan_index: int
    ) -> set[int]:
        """Return detection indices from keys that belong to ``scan_index``."""
        return {det_idx for (key_scan, det_idx) in keys if key_scan == scan_index}

    # =========================================================================
    # Node/Tree Construction Helpers
    # =========================================================================

    def _allocate_node_id(self) -> int:
        """Allocate the next stable node ID in this tracker instance."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _register_node(self, node: TrackHypothesisNode) -> TrackHypothesisNode:
        """Store one node in the persistent node table and return it."""
        self._nodes_by_id[node.node_id] = node
        return node

    def _create_track_hypothesis_node(
        self,
        *,
        track_id: int,
        parent: TrackHypothesisNode | None,
        scan_index: int,
        timestamp: datetime.datetime,
        state: State,
        state_kind: str,
        used_det_key: DetectionKey | None,
        assoc_label: int,
        log_delta: float,
        age: int,
        hits: int,
        missed_count: int,
        last_det_key: DetectionKey | None,
        last_det_hit: bool,
        root_source: str,
        birth_scan_index: int,
    ) -> TrackHypothesisNode:
        """Create and register one persistent hypothesis node."""
        if parent is not None and parent.track_id != track_id:
            raise ValueError(
                "TrackHypothesisNode parent.track_id must match child track_id."
            )

        if parent is None:
            history_keys: frozenset[DetectionKey]
            if used_det_key is None:
                history_keys = frozenset()
            else:
                history_keys = frozenset({used_det_key})
            accumulated_log_score = float(log_delta)
        else:
            if used_det_key is None:
                history_keys = parent.detection_history_keys
            else:
                history_keys = parent.detection_history_keys | {used_det_key}
            accumulated_log_score = float(parent.accumulated_log_score) + float(
                log_delta
            )

        node = TrackHypothesisNode(
            node_id=self._allocate_node_id(),
            track_id=int(track_id),
            parent=parent,
            scan_index=int(scan_index),
            timestamp=timestamp,
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=int(assoc_label),
            log_delta=float(log_delta),
            accumulated_log_score=float(accumulated_log_score),
            detection_history_keys=history_keys,
            age=int(age),
            hits=int(hits),
            missed_count=int(missed_count),
            last_det_key=last_det_key,
            last_det_hit=bool(last_det_hit),
            root_source=root_source,
            birth_scan_index=int(birth_scan_index),
        )
        self._register_node(node)
        if parent is not None:
            parent.child_node_ids.add(node.node_id)
        return node

    def _create_root_node(
        self,
        *,
        track_id: int,
        scan_index: int,
        timestamp: datetime.datetime,
        state: State,
        state_kind: str,
        used_det_key: DetectionKey | None,
        assoc_label: int,
        log_delta: float,
        age: int,
        hits: int,
        root_source: str,
    ) -> TrackHypothesisNode:
        """Create a root node wrapper for births/external starts."""
        return self._create_track_hypothesis_node(
            track_id=track_id,
            parent=None,
            scan_index=scan_index,
            timestamp=timestamp,
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=assoc_label,
            log_delta=log_delta,
            age=age,
            hits=hits,
            missed_count=0,
            last_det_key=used_det_key,
            last_det_hit=used_det_key is not None,
            root_source=root_source,
            birth_scan_index=scan_index,
        )

    # =========================================================================
    # Local Expansion and Simple Lifecycle
    # =========================================================================

    def _candidate_from_hypothesis(
        self,
        *,
        leaf_node: TrackHypothesisNode,
        hypothesis: Any,
        ctx: ScanContext,
        log_delta: float,
    ) -> LocalChildCandidate:
        """Map one Stone Soup hypothesis to one child node candidate."""
        if not hypothesis:
            state = getattr(hypothesis, "prediction")
            used_det_key = None
            assoc_label = TOMHTTracker.ASSOC_MISS
            state_kind = "prediction"
            missed_count = int(leaf_node.missed_count) + 1
            last_det_key = leaf_node.last_det_key
        else:
            state = self.updater.update(hypothesis)
            det_index = int(ctx.det_index_by_obj[id(hypothesis.measurement)])
            used_det_key = (ctx.scan_index, det_index)
            assoc_label = det_index
            state_kind = "update"
            missed_count = 0
            last_det_key = used_det_key

        child_node = self._create_track_hypothesis_node(
            track_id=leaf_node.track_id,
            parent=leaf_node,
            scan_index=ctx.scan_index,
            timestamp=getattr(state, "timestamp", ctx.timestamp),
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=assoc_label,
            log_delta=log_delta,
            age=int(leaf_node.age) + 1,
            hits=int(leaf_node.hits) + (0 if used_det_key is None else 1),
            missed_count=missed_count,
            last_det_key=last_det_key,
            last_det_hit=used_det_key is not None,
            root_source=leaf_node.root_source,
            birth_scan_index=leaf_node.birth_scan_index,
        )
        return LocalChildCandidate(
            track_id=leaf_node.track_id,
            child_node=child_node,
            used_det_key=used_det_key,
            log_delta=log_delta,
        )

    def _candidates_for_track_leaf(
        self,
        leaf_node: TrackHypothesisNode,
        ctx: ScanContext,
    ) -> list[LocalChildCandidate]:
        """Build retained local continuation candidates for one active leaf."""
        track = reconstruct_track_from_leaf_node(leaf_node)
        multi_hypotheses = self.hypothesiser.hypothesise(
            track, ctx.detections, ctx.timestamp
        )
        hypotheses = list(multi_hypotheses)

        hyp_scores = self.scoring_model.score_track_hypotheses(
            track=track,
            hypotheses=hypotheses,
            ctx=ctx,
        )

        def _score_for_sort(hyp) -> float:
            score = hyp_scores.get(id(hyp))
            if score is not None:
                return float(score)
            p = getattr(hyp, "probability", None)
            if p is not None:
                try:
                    return float(p)
                except Exception:
                    return 0.0
            w = getattr(hyp, "weight", 0.0)
            try:
                return float(w)
            except Exception:
                return 0.0

        def _sort_key(hyp) -> tuple[float, int]:
            p = _score_for_sort(hyp)
            if not hyp:
                return (p, -1)
            return (p, -ctx.det_index_by_obj.get(id(hyp.measurement), 10**9))

        sorted_hypotheses = sorted(hypotheses, key=_sort_key, reverse=True)
        kept = sorted_hypotheses[: self.params.max_children_per_track]
        miss = next((h for h in sorted_hypotheses if not h), None)
        if miss is not None and miss not in kept:
            kept.append(miss)

        out = [
            self._candidate_from_hypothesis(
                leaf_node=leaf_node,
                hypothesis=hyp,
                ctx=ctx,
                log_delta=float(hyp_scores.get(id(hyp), 0.0)),
            )
            for hyp in kept
        ]
        out.sort(key=lambda c: c.log_delta, reverse=True)
        return out

    def _expand_one_track_tree(self, tree: TrackTree, ctx: ScanContext) -> None:
        """Expand all active leaves in one tree, then apply pre-solve cap guardrail."""
        new_leaf_ids: set[int] = set()

        for leaf_id in sorted(tree.active_leaf_node_ids):
            leaf = self._nodes_by_id[leaf_id]
            candidates = self._candidates_for_track_leaf(leaf, ctx)
            for cand in candidates:
                new_leaf_ids.add(cand.child_node.node_id)

        tree.active_leaf_node_ids = self._apply_pre_solve_leaf_cap_guardrail(
            new_leaf_ids
        )

    def _apply_pre_solve_leaf_cap_guardrail(
        self,
        leaf_node_ids: set[int],
    ) -> set[int]:
        """Apply optional local leaf capping only as a pre-solve tractability valve."""
        max_leaves = self.params.max_leaves_per_track_tree
        if max_leaves is None or len(leaf_node_ids) <= max_leaves:
            return leaf_node_ids

        ranked = sorted(
            (self._nodes_by_id[node_id] for node_id in leaf_node_ids),
            key=lambda node: (
                float(node.accumulated_log_score),
                -int(node.node_id),
            ),
            reverse=True,
        )
        return {node.node_id for node in ranked[: int(max_leaves)]}

    def _expand_all_track_trees(self, ctx: ScanContext) -> None:
        """Run local expansion for all current persistent track trees."""
        for tree in self.track_trees_by_track_id.values():
            self._expand_one_track_tree(tree, ctx)

    def _remove_empty_trees(self) -> None:
        """Drop any tree that has no surviving active leaves."""
        dead_track_ids = [
            track_id
            for track_id, tree in self.track_trees_by_track_id.items()
            if not tree.active_leaf_node_ids
        ]
        for track_id in dead_track_ids:
            self.track_trees_by_track_id.pop(track_id, None)

    # =========================================================================
    # Internal Birth Handling (Simple Phase-D Policy)
    # =========================================================================

    def _active_leaf_nodes(self) -> list[TrackHypothesisNode]:
        """Return all active leaves across all persistent track trees."""
        out: list[TrackHypothesisNode] = []
        for tree in self.track_trees_by_track_id.values():
            out.extend(
                self._nodes_by_id[node_id] for node_id in tree.active_leaf_node_ids
            )
        return out

    def _birth_used_key(
        self,
        tr: Track,
        *,
        scan_index: int,
        det_index_by_obj: dict[int, int],
    ) -> DetectionKey | None:
        """Best-effort extraction of a current-scan used detection key for a birth."""
        try:
            last = tr.states[-1]
            hyp = getattr(last, "hypothesis", None)
            meas = getattr(hyp, "measurement", None) if hyp is not None else None
            if meas is None:
                return None
            det_index = det_index_by_obj.get(id(meas))
            if det_index is None:
                return None
            return (scan_index, int(det_index))
        except Exception:
            return None

    def _birth_is_sane(self, tr: Track) -> bool:
        """Apply simple numeric sanity checks before accepting an internal birth."""
        st = tr.states[-1]
        sv = np.asarray(st.state_vector, dtype=float)

        x = float(sv[0, 0])
        y = float(sv[2, 0])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        if (
            abs(x) > self.params.birth_max_abs_pos
            or abs(y) > self.params.birth_max_abs_pos
        ):
            return False

        cov = getattr(st, "covar", None)
        if cov is None:
            return False
        cov = np.asarray(cov, dtype=float)
        if not np.all(np.isfinite(cov)):
            return False
        if float(np.trace(cov)) > self.params.birth_max_covar_trace:
            return False

        return True

    def _residual_detection_indices_after_step2(self, ctx: ScanContext) -> list[int]:
        """Return current-scan detection indices unused after local expansion."""
        used_current_scan_det_indices: set[int] = set()
        for leaf in self._active_leaf_nodes():
            used_current_scan_det_indices |= self._current_scan_det_indices_from_keys(
                leaf.detection_history_keys,
                ctx.scan_index,
            )
        return [
            i
            for i in range(len(ctx.detections))
            if i not in used_current_scan_det_indices
        ]

    def _birth_guardrail_block_reason(self) -> str | None:
        """Return a reason when simple load guards should block internal births."""
        active_trees = len(self.track_trees_by_track_id)
        active_leaves = sum(
            len(tree.active_leaf_node_ids)
            for tree in self.track_trees_by_track_id.values()
        )

        trees_cap = self.params.birth_skip_if_active_trees_above
        if trees_cap is not None and active_trees > int(trees_cap):
            return f"active tree count above cap ({active_trees}>{int(trees_cap)})"

        leaves_cap = self.params.birth_skip_if_active_leaves_above
        if leaves_cap is not None and active_leaves > int(leaves_cap):
            return f"active leaf count above cap ({active_leaves}>{int(leaves_cap)})"

        return None

    def _run_internal_births(self, ctx: ScanContext) -> BirthStats:
        """Create simple internal birth trees from Step-2 residual detections."""
        residual_det_indices = self._residual_detection_indices_after_step2(ctx)
        residual_detections = [ctx.detections[i] for i in residual_det_indices]

        if self.initiator is None:
            self._last_unused_detections = residual_detections
            return BirthStats(
                residual_detections_considered=len(residual_detections),
                birth_tracks_created=0,
                birth_tracks_kept=0,
            )

        if not residual_detections:
            self._last_unused_detections = []
            return BirthStats(
                residual_detections_considered=0,
                birth_tracks_created=0,
                birth_tracks_kept=0,
            )

        birth_block_reason = self._birth_guardrail_block_reason()
        if birth_block_reason is not None:
            self._last_unused_detections = residual_detections
            if self.params.debug_display_births:
                print(
                    "\nINTERNAL_BIRTH_GUARDRAIL "
                    f"t={ctx.timestamp} reason={birth_block_reason} "
                    f"residual={len(residual_detections)}"
                )
            return BirthStats(
                residual_detections_considered=len(residual_detections),
                birth_tracks_created=0,
                birth_tracks_kept=0,
            )

        self._last_unused_detections = []

        initiated_tracks = list(
            self.initiator.initiate(OrderedSet(residual_detections), ctx.timestamp)
        )
        birth_tracks_created = len(initiated_tracks)

        # Keep this phase intentionally simple: numeric sanity + fixed cap.
        kept_birth_tracks = [
            track for track in initiated_tracks if self._birth_is_sane(track)
        ]
        if len(kept_birth_tracks) > self.params.max_births_per_scan:
            kept_birth_tracks = kept_birth_tracks[: self.params.max_births_per_scan]
        birth_tracks_kept = len(kept_birth_tracks)

        if self.params.debug_display_births and kept_birth_tracks:
            print(f"\nInternal births at {ctx.timestamp}: kept={birth_tracks_kept}")
            for track in kept_birth_tracks[: self.params.debug_births_max]:
                # Debug-only display retained for quick replay inspection.
                state = track.states[-1].state_vector
                print(f"  birth_state={self._fmt_state_xyvxvy(state)}")

        for birth_track in kept_birth_tracks:
            track_id = self._next_track_id
            self._next_track_id += 1
            state = birth_track.states[-1]
            used_key = self._birth_used_key(
                birth_track,
                scan_index=ctx.scan_index,
                det_index_by_obj=ctx.det_index_by_obj,
            )
            age = max(len(birth_track), 1)
            hits = 1 if used_key is not None else 0
            root_log_delta = self.scoring_model.score_birth(
                birth_track=birth_track,
                used_det_key=(None if used_key is None else int(used_key[1])),
                ctx=ctx,
            )
            root = self._create_root_node(
                track_id=track_id,
                scan_index=ctx.scan_index,
                timestamp=getattr(state, "timestamp", ctx.timestamp),
                state=state,
                state_kind="internal_birth",
                used_det_key=used_key,
                assoc_label=(
                    TOMHTTracker.ASSOC_PAD if used_key is None else int(used_key[1])
                ),
                log_delta=float(root_log_delta),
                age=age,
                hits=hits,
                root_source="internal_birth",
            )
            self.track_trees_by_track_id[track_id] = TrackTree(
                track_id=track_id,
                root_node_id=root.node_id,
                active_leaf_node_ids={root.node_id},
                root_source="internal_birth",
            )

        return BirthStats(
            residual_detections_considered=len(residual_detections),
            birth_tracks_created=birth_tracks_created,
            birth_tracks_kept=birth_tracks_kept,
        )

    # =========================================================================
    # Per-Scan Clustering + Global Rebuild
    # =========================================================================

    # --- Cluster graph construction from current tree frontiers ---

    def _current_scan_candidate_keys_for_tree(
        self,
        tree: TrackTree,
        scan_index: int,
    ) -> set[DetectionKey]:
        """Return current-scan detection keys present in one tree frontier."""
        keys: set[DetectionKey] = set()
        for leaf_id in tree.active_leaf_node_ids:
            leaf = self._nodes_by_id[leaf_id]
            if (
                leaf.used_det_key is not None
                and int(leaf.used_det_key[0]) == scan_index
            ):
                keys.add(leaf.used_det_key)
        return keys

    def _history_conflict_keys_for_tree(
        self,
        tree: TrackTree,
    ) -> set[DetectionKey]:
        """Return all detection-history keys present in this tree's active leaves."""
        keys: set[DetectionKey] = set()
        for leaf_id in tree.active_leaf_node_ids:
            leaf = self._nodes_by_id[leaf_id]
            keys |= set(leaf.detection_history_keys)
        return keys

    def _build_track_clusters(self, ctx: ScanContext) -> list[_ClusterWorkItem]:
        """Build independent clusters from shared active-leaf history detections."""
        track_ids = sorted(self.track_trees_by_track_id.keys())
        if not track_ids:
            return []

        history_keys_by_track: dict[int, set[DetectionKey]] = {
            track_id: self._history_conflict_keys_for_tree(
                self.track_trees_by_track_id[track_id],
            )
            for track_id in track_ids
        }
        current_scan_keys_by_track: dict[int, set[DetectionKey]] = {
            track_id: self._current_scan_candidate_keys_for_tree(
                self.track_trees_by_track_id[track_id],
                ctx.scan_index,
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

        out: list[_ClusterWorkItem] = []
        for cluster_id, component in enumerate(
            sorted(components, key=lambda c: tuple(c))
        ):
            comp_track_ids = tuple(component)
            comp_track_set = set(comp_track_ids)
            comp_links = tuple(
                link
                for link in conflict_links
                if link[0] in comp_track_set and link[1] in comp_track_set
            )
            out.append(
                _ClusterWorkItem(
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

    def _build_cluster_unused_score_context(
        self,
        *,
        cluster_universe: set[DetectionKey],
        ctx: ScanContext,
    ) -> _ClusterUnusedScoreContext | None:
        """Build one reusable cluster-local context for unused-detection scoring."""
        if not cluster_universe:
            return None

        det_indices = sorted(
            det_idx
            for (scan_idx, det_idx) in cluster_universe
            if scan_idx == ctx.scan_index
        )
        if not det_indices:
            return None

        detections_subset = [ctx.detections[idx] for idx in det_indices]
        local_det_index_by_obj = {
            id(det): local_idx for local_idx, det in enumerate(detections_subset)
        }
        local_ctx = ScanContext(
            scan_index=ctx.scan_index,
            timestamp=ctx.timestamp,
            detections=detections_subset,
            det_index_by_obj=local_det_index_by_obj,
        )

        local_slot_by_global_det_index = {
            global_det_idx: local_idx
            for local_idx, global_det_idx in enumerate(det_indices)
        }
        return _ClusterUnusedScoreContext(
            local_ctx=local_ctx,
            local_slot_by_global_det_index=local_slot_by_global_det_index,
        )

    # --- Solver guardrails / debug-only feasibility checks ---

    def _score_unused_cluster_current_scan_term(
        self,
        *,
        selected_used_current_scan_keys: set[DetectionKey],
        score_context: _ClusterUnusedScoreContext | None,
        scan_index: int,
    ) -> float:
        """Compute explicit per-combination cluster-local unused-detection term."""
        if score_context is None:
            return 0.0

        used_local_slots = {
            score_context.local_slot_by_global_det_index[det_idx]
            for (key_scan_idx, det_idx) in selected_used_current_scan_keys
            if (
                key_scan_idx == scan_index
                and det_idx in score_context.local_slot_by_global_det_index
            )
        }
        return self.scoring_model.score_unused_detections(
            used_det_keys=used_local_slots,
            ctx=score_context.local_ctx,
        )

    @staticmethod
    def _pruning_feasibility_validation_enabled() -> bool:
        """Return whether debug-only pruning feasibility validation is enabled."""
        raw = os.getenv("TOMHT_DEBUG_VALIDATE_PRUNING_FEASIBILITY")
        if raw is None:
            return False
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _has_any_feasible_cluster_combination(
        leaf_options: list[list[TrackHypothesisNode]],
    ) -> bool:
        """Return whether at least one cluster leaf-product combination is feasible."""
        prepared = [
            [(leaf, set(leaf.detection_history_keys)) for leaf in leaves]
            for leaves in leaf_options
        ]
        for picked in product(*prepared):
            used_keys: set[DetectionKey] = set()
            feasible = True
            for _, leaf_keys in picked:
                if used_keys & leaf_keys:
                    feasible = False
                    break
                used_keys |= leaf_keys
            if feasible:
                return True
        return False

    def _maybe_validate_pruning_feasibility(
        self,
        *,
        stage: str,
        ctx: ScanContext,
    ) -> None:
        """Debug-only guard: fail fast if any cluster is infeasible after pruning."""
        if not self._pruning_feasibility_validation_enabled():
            return
        if not self.track_trees_by_track_id:
            return

        clusters = self._build_track_clusters(ctx)
        for cluster in clusters:
            leaf_options = self._cluster_leaf_options(cluster.track_ids)
            if self._has_any_feasible_cluster_combination(leaf_options):
                continue
            dbg = self._infeasible_cluster_debug_summary(
                cluster=cluster,
                leaf_options=leaf_options,
                ctx=ctx,
            )
            raise RuntimeError(
                "Pruning feasibility check failed. " f"stage={stage}; {dbg}"
            )

    @staticmethod
    def _projected_combination_count(
        leaf_options: list[list[TrackHypothesisNode]],
    ) -> int:
        """Return projected Cartesian product size for one leaf-option set."""
        projected = 1
        for leaves in leaf_options:
            projected *= len(leaves)
        return projected

    def _projected_combination_count_for_track_ids(
        self,
        track_ids: tuple[int, ...],
    ) -> int:
        """Projected Cartesian leaf combinations for one track-id tuple."""
        projected = 1
        for track_id in track_ids:
            tree = self.track_trees_by_track_id[track_id]
            leaf_count = len(tree.active_leaf_node_ids)
            if leaf_count <= 0:
                raise RuntimeError(
                    "Cluster rebuild encountered a tree with no active leaves. "
                    "Lifecycle filtering should remove empty trees before clustering."
                )
            projected *= leaf_count
        return projected

    @staticmethod
    def _edge_pair(
        left_track_id: int,
        right_track_id: int,
    ) -> tuple[int, int]:
        """Return canonical undirected edge ordering for track-id pairs."""
        if left_track_id <= right_track_id:
            return (left_track_id, right_track_id)
        return (right_track_id, left_track_id)

    # --- Approximation path: overload split before exact subcluster rebuild ---

    @staticmethod
    def _connected_components_from_pairs(
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

    @staticmethod
    def _cluster_edge_strengths(
        cluster: _ClusterWorkItem,
    ) -> dict[tuple[int, int], int]:
        """Return conflict-edge strengths = shared full-history key counts."""
        strengths: dict[tuple[int, int], int] = {}
        for left_track_id, right_track_id, shared_keys in cluster.conflict_links:
            strengths[TOMHTTracker._edge_pair(left_track_id, right_track_id)] = len(
                shared_keys
            )
        return strengths

    def _split_overloaded_cluster(
        self,
        *,
        cluster: _ClusterWorkItem,
        projected_before: int,
        threshold: int,
    ) -> tuple[list[_ClusterWorkItem], _OverloadSplitSummary]:
        """Approximate one overloaded cluster by severing weakest conflict edges."""
        remaining_edge_keys_by_pair: dict[tuple[int, int], tuple[DetectionKey, ...]] = {
            self._edge_pair(left_track_id, right_track_id): tuple(shared_keys)
            for left_track_id, right_track_id, shared_keys in cluster.conflict_links
        }
        edge_strengths = self._cluster_edge_strengths(cluster)
        removed_edges: list[_OverloadSplitRemovedEdge] = []

        stopping_reason = "all_components_under_threshold"
        max_removals = self.params.overload_split_max_edge_removals_per_cluster
        max_removals_int = None if max_removals is None else int(max_removals)

        while True:
            components = self._connected_components_from_pairs(
                cluster.track_ids,
                remaining_edge_keys_by_pair.keys(),
            )
            overloaded_components: list[tuple[int, ...]] = []
            for component_track_ids in components:
                projected = self._projected_combination_count_for_track_ids(
                    component_track_ids
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
                _OverloadSplitRemovedEdge(
                    left_track_id=left_track_id,
                    right_track_id=right_track_id,
                    shared_history_key_count=strength,
                )
            )

        final_components = self._connected_components_from_pairs(
            cluster.track_ids,
            remaining_edge_keys_by_pair.keys(),
        )
        subclusters: list[_ClusterWorkItem] = []
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
                _ClusterWorkItem(
                    cluster_id=-1,
                    track_ids=component_track_ids,
                    current_scan_det_keys_by_track_id={
                        track_id: set(
                            cluster.current_scan_det_keys_by_track_id[track_id]
                        )
                        for track_id in component_track_ids
                    },
                    conflict_links=component_links,
                    overload_split_origin_cluster_id=cluster.cluster_id,
                )
            )
            projected_after_by_subcluster.append(
                self._projected_combination_count_for_track_ids(component_track_ids)
            )

        summary = _OverloadSplitSummary(
            original_cluster_id=cluster.cluster_id,
            original_track_ids=cluster.track_ids,
            projected_before=projected_before,
            projected_threshold=threshold,
            removed_edges=tuple(removed_edges),
            resulting_subclusters=tuple(
                subcluster.track_ids for subcluster in subclusters
            ),
            projected_after_by_subcluster=tuple(projected_after_by_subcluster),
            stopping_reason=stopping_reason,
        )
        return subclusters, summary

    def _maybe_split_cluster_under_overload(
        self,
        *,
        cluster: _ClusterWorkItem,
    ) -> tuple[list[_ClusterWorkItem], _OverloadSplitSummary | None]:
        """Split one cluster only when projected Cartesian size exceeds threshold."""
        if not self.params.overload_split_enabled:
            return [cluster], None

        threshold_raw = self.params.overload_split_projected_combination_threshold
        if threshold_raw is None:
            return [cluster], None
        threshold = int(threshold_raw)
        if threshold <= 0:
            raise ValueError(
                "overload_split_projected_combination_threshold must be positive "
                "when overload splitting is enabled."
            )

        projected_before = self._projected_combination_count_for_track_ids(
            cluster.track_ids
        )
        if projected_before <= threshold:
            return [cluster], None

        return self._split_overloaded_cluster(
            cluster=cluster,
            projected_before=projected_before,
            threshold=threshold,
        )

    @staticmethod
    def _log_overload_split_summary(
        *,
        scan_index: int,
        summary: _OverloadSplitSummary,
    ) -> None:
        """Print one compact overload-split instrumentation line."""
        removed_edges_str = (
            "["
            + ", ".join(
                (
                    f"{edge.left_track_id}-{edge.right_track_id}:"
                    f"{edge.shared_history_key_count}"
                )
                for edge in summary.removed_edges
            )
            + "]"
            if summary.removed_edges
            else "[]"
        )
        projected_after = list(summary.projected_after_by_subcluster)
        print(
            "OVERLOAD_SPLIT "
            f"scan={scan_index} "
            f"cluster={summary.original_cluster_id} "
            f"track_ids={list(summary.original_track_ids)} "
            f"projected_before={summary.projected_before} "
            f"threshold={summary.projected_threshold} "
            f"split_ops={len(summary.removed_edges)} "
            f"stop={summary.stopping_reason} "
            f"removed_edges={removed_edges_str} "
            f"subclusters={[list(c) for c in summary.resulting_subclusters]} "
            f"projected_after={projected_after}"
        )

    # --- Exact cluster rebuild helpers ---

    def _cluster_leaf_options(
        self,
        track_ids: tuple[int, ...],
    ) -> list[list[TrackHypothesisNode]]:
        """Materialize sorted active-leaf options for each track in a cluster."""
        out: list[list[TrackHypothesisNode]] = []
        for track_id in track_ids:
            tree = self.track_trees_by_track_id[track_id]
            leaves = [
                self._nodes_by_id[node_id]
                for node_id in sorted(tree.active_leaf_node_ids)
            ]
            if not leaves:
                raise RuntimeError(
                    "Cluster rebuild encountered a tree with no active leaves. "
                    "Lifecycle filtering should remove empty trees before clustering."
                )
            out.append(leaves)
        return out

    def _push_top_k_global(
        self,
        *,
        top_k_heap: _ClusterTopKHeap,
        candidate: GlobalHypothesis,
        insertion_order: int,
        k: int,
    ) -> None:
        """Streaming top-K maintenance for rebuilt globals.

        Heap entries are ``(log_weight, -insertion_order, global)``.
        For equal ``log_weight``, this keeps earlier-enumerated combinations and
        evicts later-enumerated ties, matching the previous stable-sort behavior.
        """
        if k <= 0:
            return

        entry = (
            float(candidate.log_weight),
            -int(insertion_order),
            candidate,
        )
        if len(top_k_heap) < k:
            heapq.heappush(top_k_heap, entry)
            return
        if entry > top_k_heap[0]:
            heapq.heapreplace(top_k_heap, entry)

    @staticmethod
    def _finalize_top_k_globals(
        top_k_heap: _ClusterTopKHeap,
    ) -> tuple[GlobalHypothesis, ...]:
        """Return retained rebuilt globals sorted best-first."""
        top_k_heap.sort(
            key=lambda item: (
                float(item[0]),  # log_weight
                int(-item[1]),  # insertion_order (ascending for tie stability)
            ),
            reverse=True,
        )
        return tuple(item[2] for item in top_k_heap)

    def _infeasible_cluster_debug_summary(
        self,
        *,
        cluster: _ClusterWorkItem,
        leaf_options: list[list[TrackHypothesisNode]],
        ctx: ScanContext,
    ) -> str:
        """Build compact debug context for a cluster with no feasible combinations."""
        parts: list[str] = []
        parts.append(f"scan_index={ctx.scan_index}")
        parts.append(f"cluster_id={cluster.cluster_id}")
        parts.append(f"track_ids={list(cluster.track_ids)}")

        leaf_count_by_track_id = {
            track_id: len(leaf_options[idx])
            for idx, track_id in enumerate(cluster.track_ids)
        }
        parts.append(f"leaf_counts={leaf_count_by_track_id}")

        # Pairwise overlap counts on full detection histories indicate how "hard"
        # the incompatibilities are between tree frontiers.
        pairwise_overlap_counts: list[str] = []
        for i, left_track_id in enumerate(cluster.track_ids):
            left_leaves = leaf_options[i]
            for j, right_track_id in enumerate(cluster.track_ids[i + 1 :], start=i + 1):
                right_leaves = leaf_options[j]
                conflicting_pairs = 0
                for left_leaf in left_leaves:
                    left_hist = set(left_leaf.detection_history_keys)
                    for right_leaf in right_leaves:
                        if left_hist & set(right_leaf.detection_history_keys):
                            conflicting_pairs += 1
                total_pairs = len(left_leaves) * len(right_leaves)
                pairwise_overlap_counts.append(
                    f"{left_track_id}-{right_track_id}:{conflicting_pairs}/{total_pairs}"
                )
        if pairwise_overlap_counts:
            parts.append(
                "pairwise_conflicts={" + ", ".join(pairwise_overlap_counts) + "}"
            )

        return "; ".join(parts)

    @staticmethod
    def _format_detection_key_sample(
        keys: set[DetectionKey],
        *,
        max_items: int = 6,
    ) -> str:
        """Return compact stable formatting for detection-key debug samples."""
        if not keys:
            return "[]"
        ordered = sorted(keys)
        if len(ordered) <= max_items:
            return str(ordered)
        head = ordered[:max_items]
        return f"{head}...(+{len(ordered) - max_items})"

    @staticmethod
    def _forced_detection_history_keys(
        leaves: list[TrackHypothesisNode],
    ) -> set[DetectionKey]:
        """Return detection keys present in every active leaf for one track tree."""
        forced = set(leaves[0].detection_history_keys)
        for leaf in leaves[1:]:
            forced &= set(leaf.detection_history_keys)
        return forced

    def _historical_relaxed_conflict_keys_for_cluster(
        self,
        *,
        cluster: _ClusterWorkItem,
        leaf_options: list[list[TrackHypothesisNode]],
        ctx: ScanContext,
    ) -> set[DetectionKey]:
        """Return forced committed historical keys shared by multiple tracks."""
        boundary_scan_index = int(ctx.scan_index) - int(self.params.ns_scan_window)
        key_track_count: dict[DetectionKey, int] = {}
        for idx, track_id in enumerate(cluster.track_ids):
            leaves = leaf_options[idx]
            forced_keys = self._forced_detection_history_keys(leaves)
            tree = self.track_trees_by_track_id[track_id]
            root = self._nodes_by_id[tree.root_node_id]
            root_keys = set(root.detection_history_keys)
            forced_committed_keys = {
                key
                for key in (forced_keys & root_keys)
                if int(key[0]) <= boundary_scan_index
            }
            for key in forced_committed_keys:
                key_track_count[key] = key_track_count.get(key, 0) + 1

        return {key for key, count in key_track_count.items() if count > 1}

    def _log_historical_relaxation(
        self,
        *,
        cluster: _ClusterWorkItem,
        ctx: ScanContext,
        relaxed_keys: set[DetectionKey],
        feasible_before: int,
        feasible_after: int,
    ) -> None:
        """Emit compact instrumentation for historical-conflict relaxation events."""
        print(
            "HIST_RELAX "
            f"scan={ctx.scan_index} "
            f"cluster={cluster.cluster_id} "
            f"track_ids={list(cluster.track_ids)} "
            f"relaxed_keys={len(relaxed_keys)} "
            "relaxed_sample="
            f"{self._format_detection_key_sample(relaxed_keys)} "
            f"feasible_before={feasible_before} "
            f"feasible_after={feasible_after} "
            f"status={'enabled' if feasible_after > 0 else 'failed'}"
        )

    def _enumerate_cluster_globals(
        self,
        *,
        cluster: _ClusterWorkItem,
        leaf_options: list[list[TrackHypothesisNode]],
        ctx: ScanContext,
        cluster_universe: set[DetectionKey],
        unused_score_context: _ClusterUnusedScoreContext | None,
        relaxed_conflict_keys: frozenset[DetectionKey],
    ) -> tuple[_ClusterTopKHeap, int, int]:
        """Enumerate feasible globals with optional relaxed historical conflicts."""
        top_k_heap: _ClusterTopKHeap = []
        combinations_evaluated = 0
        feasible_combinations = 0
        k = int(self.params.max_global_hypotheses)
        relaxed_keys_set = set(relaxed_conflict_keys)

        for picked in product(*leaf_options):
            combinations_evaluated += 1
            selected = list(picked)

            feasible = True
            used_keys: set[DetectionKey] = set()
            for leaf in selected:
                leaf_keys = set(leaf.detection_history_keys)
                overlap = (used_keys & leaf_keys) - relaxed_keys_set
                if overlap:
                    feasible = False
                    break
                used_keys |= leaf_keys
            if not feasible:
                continue

            feasible_combinations += 1
            leaf_nodes_by_track_id = {
                track_id: selected[idx]
                for idx, track_id in enumerate(cluster.track_ids)
            }
            leaf_score_sum = sum(float(leaf.accumulated_log_score) for leaf in selected)

            used_current_scan_keys = {
                key
                for key in used_keys
                if int(key[0]) == ctx.scan_index and key in cluster_universe
            }
            unused_term = self._score_unused_cluster_current_scan_term(
                selected_used_current_scan_keys=used_current_scan_keys,
                score_context=unused_score_context,
                scan_index=ctx.scan_index,
            )
            candidate = GlobalHypothesis(
                leaf_nodes_by_track_id=leaf_nodes_by_track_id,
                log_weight=float(leaf_score_sum + unused_term),
            )
            self._push_top_k_global(
                top_k_heap=top_k_heap,
                candidate=candidate,
                insertion_order=feasible_combinations,
                k=k,
            )

        return top_k_heap, combinations_evaluated, feasible_combinations

    def _solve_cluster_exact_exhaustive(
        self,
        *,
        solve_input: _ClusterSolveInput,
        relaxed_conflict_keys: frozenset[DetectionKey],
    ) -> tuple[_ClusterTopKHeap, int, int]:
        """Run one exhaustive cluster-solver pass under fixed conflict rules."""
        return self._enumerate_cluster_globals(
            cluster=solve_input.cluster,
            leaf_options=solve_input.leaf_options,
            ctx=solve_input.ctx,
            cluster_universe=solve_input.cluster_universe,
            unused_score_context=solve_input.unused_score_context,
            relaxed_conflict_keys=relaxed_conflict_keys,
        )

    def _raise_cluster_infeasible_error(
        self,
        *,
        solve_input: _ClusterSolveInput,
        relaxed_historical_keys: set[DetectionKey],
    ) -> None:
        """Raise the existing cluster infeasibility error with optional relax debug."""
        dbg = self._infeasible_cluster_debug_summary(
            cluster=solve_input.cluster,
            leaf_options=solve_input.leaf_options,
            ctx=solve_input.ctx,
        )
        relaxation_dbg = ""
        if relaxed_historical_keys:
            relaxation_dbg = (
                "; "
                f"relaxed_historical_keys={len(relaxed_historical_keys)} "
                "relaxed_sample="
                f"{self._format_detection_key_sample(relaxed_historical_keys)}"
            )
        raise RuntimeError(
            "Cluster rebuild found no feasible combination. "
            "Expected at least one feasible joint assignment. "
            f"{dbg}{relaxation_dbg}"
        )

    def _solve_with_optional_historical_relaxation(
        self,
        *,
        solve_input: _ClusterSolveInput,
    ) -> _ClusterSolveOutcome:
        """Solve one cluster, retrying once with optional historical relaxation."""
        (
            top_k_heap,
            combinations_evaluated,
            feasible_combinations,
        ) = self._solve_cluster_exact_exhaustive(
            solve_input=solve_input,
            relaxed_conflict_keys=frozenset(),
        )

        historical_relaxation_attempted = False
        historical_relaxation_succeeded = False
        relaxed_historical_keys: set[DetectionKey] = set()
        if (
            feasible_combinations == 0
            and self.params.historical_conflict_relaxation_enabled
        ):
            relaxed_historical_keys = (
                self._historical_relaxed_conflict_keys_for_cluster(
                    cluster=solve_input.cluster,
                    leaf_options=solve_input.leaf_options,
                    ctx=solve_input.ctx,
                )
            )
            if relaxed_historical_keys:
                historical_relaxation_attempted = True
                (
                    top_k_heap,
                    relaxed_combinations_evaluated,
                    feasible_combinations,
                ) = self._solve_cluster_exact_exhaustive(
                    solve_input=solve_input,
                    relaxed_conflict_keys=frozenset(relaxed_historical_keys),
                )
                combinations_evaluated += relaxed_combinations_evaluated
                historical_relaxation_succeeded = feasible_combinations > 0
                self._log_historical_relaxation(
                    cluster=solve_input.cluster,
                    ctx=solve_input.ctx,
                    relaxed_keys=relaxed_historical_keys,
                    feasible_before=0,
                    feasible_after=feasible_combinations,
                )

        if feasible_combinations == 0:
            self._raise_cluster_infeasible_error(
                solve_input=solve_input,
                relaxed_historical_keys=relaxed_historical_keys,
            )

        kept_globals = self._finalize_top_k_globals(top_k_heap)
        return _ClusterSolveOutcome(
            kept_globals=kept_globals,
            combinations_evaluated=combinations_evaluated,
            feasible_combinations=feasible_combinations,
            historical_relaxation_attempted=historical_relaxation_attempted,
            historical_relaxation_succeeded=historical_relaxation_succeeded,
            historical_relaxed_keys=frozenset(relaxed_historical_keys),
        )

    def _solve_cluster(
        self,
        *,
        solve_input: _ClusterSolveInput,
    ) -> _ClusterSolveOutcome:
        """Cluster-solver boundary wrapper for future backend swaps."""
        return self._solve_with_optional_historical_relaxation(solve_input=solve_input)

    def _rebuild_one_cluster(
        self,
        cluster: _ClusterWorkItem,
        ctx: ScanContext,
    ) -> _ClusterRebuildResult:
        """Exhaustively enumerate and score feasible globals for one cluster."""
        leaf_options = self._cluster_leaf_options(cluster.track_ids)
        projected_combinations = self._projected_combination_count(leaf_options)
        projected_cap = self.params.max_projected_cluster_combinations
        if projected_cap is not None and projected_combinations > int(projected_cap):
            raise RuntimeError(
                "Cluster rebuild projected Cartesian combinations exceed guardrail: "
                f"cluster={cluster.cluster_id} "
                f"projected={projected_combinations} "
                f"cap={int(projected_cap)}"
            )

        cluster_universe: set[DetectionKey] = set()
        for keys in cluster.current_scan_det_keys_by_track_id.values():
            cluster_universe |= keys
        unused_score_context = self._build_cluster_unused_score_context(
            cluster_universe=cluster_universe,
            ctx=ctx,
        )

        solve_input = _ClusterSolveInput(
            cluster=cluster,
            ctx=ctx,
            leaf_options=leaf_options,
            cluster_universe=cluster_universe,
            unused_score_context=unused_score_context,
        )
        solve_outcome = self._solve_cluster(solve_input=solve_input)
        map_global = (
            solve_outcome.kept_globals[0] if solve_outcome.kept_globals else None
        )

        return _ClusterRebuildResult(
            snapshot=ClusterRebuildSnapshot(
                cluster_id=cluster.cluster_id,
                track_ids=cluster.track_ids,
                current_scan_conflict_det_keys=frozenset(cluster_universe),
                conflict_links=cluster.conflict_links,
                rebuilt_globals=solve_outcome.kept_globals,
                map_global=map_global,
                feasible_combinations=solve_outcome.feasible_combinations,
                evaluated_combinations=solve_outcome.combinations_evaluated,
                overload_split_origin_cluster_id=cluster.overload_split_origin_cluster_id,
            ),
            historical_relaxation_attempted=(
                solve_outcome.historical_relaxation_attempted
            ),
            historical_relaxation_succeeded=(
                solve_outcome.historical_relaxation_succeeded
            ),
            historical_relaxed_key_count=len(solve_outcome.historical_relaxed_keys),
        )

    def _rebuild_cluster_globals(
        self,
        clusters: list[_ClusterWorkItem],
        ctx: ScanContext,
    ) -> tuple[list[ClusterRebuildSnapshot], RebuildStats]:
        """Rebuild all clusters and aggregate per-scan rebuild instrumentation."""
        if not clusters:
            return [], RebuildStats()

        clusters_for_rebuild_raw: list[_ClusterWorkItem] = []
        split_summaries: list[_OverloadSplitSummary] = []
        for cluster in clusters:
            subclusters, split_summary = self._maybe_split_cluster_under_overload(
                cluster=cluster
            )
            clusters_for_rebuild_raw.extend(subclusters)
            if split_summary is not None:
                split_summaries.append(split_summary)
                self._log_overload_split_summary(
                    scan_index=ctx.scan_index,
                    summary=split_summary,
                )

        clusters_for_rebuild = [
            replace(cluster, cluster_id=cluster_id)
            for cluster_id, cluster in enumerate(clusters_for_rebuild_raw)
        ]
        rebuild_results = [
            self._rebuild_one_cluster(cluster, ctx) for cluster in clusters_for_rebuild
        ]
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
                historical_relaxation_attempts=sum(
                    1
                    for result in rebuild_results
                    if result.historical_relaxation_attempted
                ),
                historical_relaxation_successes=sum(
                    1
                    for result in rebuild_results
                    if result.historical_relaxation_succeeded
                ),
                historical_relaxed_keys_total=sum(
                    result.historical_relaxed_key_count for result in rebuild_results
                ),
            ),
        )

    # --- Post-solve pruning + full-scan MAP merge ---

    @staticmethod
    def _supported_leaf_ids_by_track_from_rebuilt_globals(
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

    def _apply_post_solve_supported_leaf_pruning(
        self,
        cluster_snapshots: list[ClusterRebuildSnapshot],
    ) -> None:
        """Prune each cluster tree to leaves supported by retained rebuilt globals."""
        for snapshot in cluster_snapshots:
            # Overload-decomposed clusters are approximate; keep their current
            # frontiers to avoid over-pruning branches that may be needed once
            # severed weak links reconnect in later scans.
            if snapshot.overload_split_origin_cluster_id is not None:
                continue

            # Keep k=0 behavior non-destructive for compatibility/debug edge cases.
            if not snapshot.rebuilt_globals:
                continue

            supported_by_track_id = (
                self._supported_leaf_ids_by_track_from_rebuilt_globals(snapshot)
            )
            for track_id in snapshot.track_ids:
                tree = self.track_trees_by_track_id.get(track_id)
                if tree is None:
                    continue
                supported_leaf_ids = supported_by_track_id.get(track_id, set())
                if not supported_leaf_ids:
                    raise RuntimeError(
                        "Post-solve supported-leaf pruning found no retained leaves "
                        f"for cluster={snapshot.cluster_id} track_id={track_id}."
                    )
                tree.active_leaf_node_ids = set(supported_leaf_ids)

    @staticmethod
    def _merge_cluster_map_globals(
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

    # =========================================================================
    # MAP-Only N-Scan Pruning on Explicit Trees
    # =========================================================================

    def _child_of_root_on_path(
        self,
        *,
        root: TrackHypothesisNode,
        leaf: TrackHypothesisNode,
    ) -> TrackHypothesisNode | None:
        """Return the root child that lies on the root->leaf path."""
        if root.node_id == leaf.node_id:
            return None

        node = leaf
        while node.parent is not None and node.parent.node_id != root.node_id:
            node = node.parent
        if node.parent is None:
            return None
        return node

    def _is_descendant_of(
        self,
        *,
        node: TrackHypothesisNode,
        ancestor: TrackHypothesisNode,
    ) -> bool:
        cur: TrackHypothesisNode | None = node
        while cur is not None:
            if cur.node_id == ancestor.node_id:
                return True
            cur = cur.parent
        return False

    def _compute_cluster_pruning_disagreement(
        self,
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
                alt_child = self._child_of_root_on_path(root=root_before, leaf=leaf)
                alt_child_id = None if alt_child is None else alt_child.node_id
                if alt_child_id != map_child_id:
                    disagrees = True
                    break
            if disagrees:
                disagreement_count += 1

        return map_choice_for_cluster, disagreement_count

    def _annotate_cluster_snapshots_with_map_pruning_disagreement(
        self,
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
                self._compute_cluster_pruning_disagreement(
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

    def _plan_map_n_scan_pruning(
        self,
        *,
        boundary_scan_index: int,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
    ) -> _MapNScanPruningPlan:
        """Plan MAP child commits and disagreement diagnostics without mutation."""
        root_before_by_track_id: dict[int, TrackHypothesisNode] = {
            track_id: self._nodes_by_id[tree.root_node_id]
            for track_id, tree in self.track_trees_by_track_id.items()
        }

        map_choice_by_track_id: dict[int, int] = {}
        for track_id, tree in sorted(self.track_trees_by_track_id.items()):
            root_before = root_before_by_track_id[track_id]
            if int(root_before.scan_index) >= boundary_scan_index:
                continue
            map_leaf = map_global.leaf_nodes_by_track_id.get(track_id)
            if map_leaf is None:
                continue
            child = self._child_of_root_on_path(root=root_before, leaf=map_leaf)
            if child is None:
                continue
            map_choice_by_track_id[track_id] = child.node_id

        updated_snapshots, disagreement_total = (
            self._annotate_cluster_snapshots_with_map_pruning_disagreement(
                cluster_snapshots=cluster_snapshots,
                root_before_by_track_id=root_before_by_track_id,
                map_choice_by_track_id=map_choice_by_track_id,
            )
        )
        return _MapNScanPruningPlan(
            boundary_scan_index=boundary_scan_index,
            root_before_by_track_id=root_before_by_track_id,
            map_choice_by_track_id=map_choice_by_track_id,
            disagreement_total=disagreement_total,
            updated_snapshots=updated_snapshots,
        )

    def _apply_planned_map_n_scan_pruning(
        self,
        *,
        plan: _MapNScanPruningPlan,
    ) -> int:
        """Apply one precomputed N-scan pruning plan to trees and bookkeeping."""
        committed_count = 0
        for track_id, chosen_child_id in plan.map_choice_by_track_id.items():
            current_tree = self.track_trees_by_track_id.get(track_id)
            if current_tree is None:
                continue
            root_before = plan.root_before_by_track_id[track_id]
            chosen_child = self._nodes_by_id[chosen_child_id]

            current_tree.root_node_id = chosen_child.node_id
            chosen_child.parent = None

            retained_leaf_ids = {
                leaf_id
                for leaf_id in current_tree.active_leaf_node_ids
                if self._is_descendant_of(
                    node=self._nodes_by_id[leaf_id],
                    ancestor=chosen_child,
                )
            }
            if not retained_leaf_ids:
                retained_leaf_ids = {chosen_child.node_id}
            current_tree.active_leaf_node_ids = retained_leaf_ids

            self._last_nscan_committed_ancestor_by_track_id[track_id] = chosen_child
            prev_boundary = self._committed_boundary_by_track_id.get(track_id)
            if prev_boundary is None or plan.boundary_scan_index > prev_boundary:
                self._committed_boundary_by_track_id[track_id] = (
                    plan.boundary_scan_index
                )
                self._committed_ancestor_by_track_id[track_id] = chosen_child
            committed_count += 1

            # Root promotion detaches the old root lineage from this tree.
            root_before.child_node_ids = {
                child_id
                for child_id in root_before.child_node_ids
                if child_id == chosen_child_id
            }

        self._remove_empty_trees()
        return committed_count

    def _apply_map_n_scan_pruning(
        self,
        *,
        scan_index: int,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
    ) -> tuple[int, int, int, int, list[ClusterRebuildSnapshot]]:
        """Apply MAP-only N-scan root-child promotion and disagreement bookkeeping."""
        boundary_scan_index = int(scan_index) - int(self.params.ns_scan_window)
        self._last_nscan_boundary_scan_index = boundary_scan_index
        self._last_nscan_tracks_in_scope = 0
        self._last_nscan_committed_ancestor_by_track_id = {}

        if boundary_scan_index < 0 or not self.track_trees_by_track_id:
            return boundary_scan_index, 0, 0, 0, cluster_snapshots

        plan = self._plan_map_n_scan_pruning(
            boundary_scan_index=boundary_scan_index,
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
        )
        self._last_nscan_tracks_in_scope = len(plan.map_choice_by_track_id)
        committed_count = self._apply_planned_map_n_scan_pruning(plan=plan)
        return (
            boundary_scan_index,
            len(plan.map_choice_by_track_id),
            committed_count,
            plan.disagreement_total,
            plan.updated_snapshots,
        )

    # =========================================================================
    # Post-N-Scan Whole-Track Miss Lifecycle
    # =========================================================================

    @staticmethod
    def _normalized_track_miss_termination_mode(mode_raw: str) -> str:
        """Normalize and validate track-level miss termination mode."""
        mode = str(mode_raw).strip().lower()
        valid = {"all_active_leaves", "map_leaf", "global_k_leaves"}
        if mode not in valid:
            raise ValueError(
                "Invalid TOMHTParams.track_miss_termination_mode. "
                f"Expected one of {sorted(valid)}, got {mode_raw!r}."
            )
        return mode

    def _effective_track_miss_threshold(self) -> int:
        """Track-level miss termination threshold with N-scan safety floor."""
        return max(
            int(self.params.max_missed),
            int(self.params.ns_scan_window) + 1,
        )

    def _track_miss_termination_leaves(
        self,
        *,
        track_id: int,
        tree: TrackTree,
        mode: str,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
    ) -> list[TrackHypothesisNode]:
        """Return leaves to evaluate for whole-track miss termination."""
        root = self._nodes_by_id[tree.root_node_id]

        if mode == "map_leaf":
            map_leaf = map_global.leaf_nodes_by_track_id.get(track_id)
            if map_leaf is not None and self._is_descendant_of(
                node=map_leaf,
                ancestor=root,
            ):
                return [map_leaf]

        if mode == "global_k_leaves":
            candidate_node_ids: set[int] = set()
            for snapshot in cluster_snapshots:
                if track_id not in snapshot.track_ids:
                    continue
                for rebuilt_global in snapshot.rebuilt_globals:
                    leaf = rebuilt_global.leaf_nodes_by_track_id.get(track_id)
                    if leaf is not None:
                        candidate_node_ids.add(int(leaf.node_id))

            candidate_leaves: list[TrackHypothesisNode] = []
            for node_id in sorted(candidate_node_ids):
                leaf = self._nodes_by_id.get(node_id)
                if leaf is None:
                    continue
                if self._is_descendant_of(node=leaf, ancestor=root):
                    candidate_leaves.append(leaf)
            if candidate_leaves:
                return candidate_leaves

        # Default and safe fallback for empty map/global-k sets.
        return [
            self._nodes_by_id[node_id] for node_id in sorted(tree.active_leaf_node_ids)
        ]

    def _filter_map_global_to_live_trees(
        self,
        map_global: GlobalHypothesis,
    ) -> GlobalHypothesis:
        """Drop map entries for tracks that no longer have active trees."""
        filtered_nodes = {
            track_id: leaf
            for track_id, leaf in map_global.leaf_nodes_by_track_id.items()
            if track_id in self.track_trees_by_track_id
        }
        return GlobalHypothesis(
            leaf_nodes_by_track_id=filtered_nodes,
            log_weight=float(map_global.log_weight),
        )

    def _apply_post_n_scan_track_miss_lifecycle(
        self,
        *,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
        scan_index: int,
    ) -> GlobalHypothesis:
        """Apply whole-track miss termination after N-scan pruning."""
        del scan_index  # reserved for potential future diagnostics.
        mode = self._normalized_track_miss_termination_mode(
            self.params.track_miss_termination_mode
        )
        threshold = self._effective_track_miss_threshold()

        terminated_track_ids: list[int] = []
        for track_id, tree in sorted(self.track_trees_by_track_id.items()):
            leaves = self._track_miss_termination_leaves(
                track_id=track_id,
                tree=tree,
                mode=mode,
                map_global=map_global,
                cluster_snapshots=cluster_snapshots,
            )
            if not leaves:
                continue
            if all(int(leaf.missed_count) >= threshold for leaf in leaves):
                terminated_track_ids.append(track_id)

        if terminated_track_ids:
            for track_id in terminated_track_ids:
                self.track_trees_by_track_id.pop(track_id, None)
            print(
                "TRACK_LIFECYCLE "
                f"mode={mode} miss_threshold={threshold} "
                f"terminated={terminated_track_ids}"
            )

        self._remove_empty_trees()
        return self._filter_map_global_to_live_trees(map_global)

    # =========================================================================
    # Node Retention / Cleanup
    # =========================================================================

    def _reachable_node_ids_from_seeds(
        self,
        seeds: Iterable[TrackHypothesisNode],
    ) -> set[int]:
        """Return node IDs reachable via parent links from supplied seeds."""
        reachable: set[int] = set()
        stack = list(seeds)
        while stack:
            node = stack.pop()
            node_id = int(node.node_id)
            if node_id in reachable:
                continue
            reachable.add(node_id)
            if node.parent is not None:
                stack.append(node.parent)
        return reachable

    def _cleanup_unreachable_nodes(self) -> None:
        """Reclaim nodes no longer reachable from roots/leaves/commitment refs."""
        if not self._nodes_by_id:
            return

        seeds: list[TrackHypothesisNode] = []
        for tree in self.track_trees_by_track_id.values():
            seeds.append(self._nodes_by_id[tree.root_node_id])
            seeds.extend(
                self._nodes_by_id[node_id] for node_id in tree.active_leaf_node_ids
            )
        seeds.extend(self._last_nscan_committed_ancestor_by_track_id.values())
        seeds.extend(self._committed_ancestor_by_track_id.values())

        if not seeds:
            self._nodes_by_id.clear()
            return

        retained_node_ids = self._reachable_node_ids_from_seeds(seeds)
        if len(retained_node_ids) == len(self._nodes_by_id):
            return

        self._nodes_by_id = {
            node_id: node
            for node_id, node in self._nodes_by_id.items()
            if node_id in retained_node_ids
        }
        for node in self._nodes_by_id.values():
            node.child_node_ids = {
                child_id
                for child_id in node.child_node_ids
                if child_id in retained_node_ids
            }

    # =========================================================================
    # External-Start Helpers
    # =========================================================================

    def _make_external_start_root(
        self,
        start: Track,
        time: datetime.datetime,
    ) -> TrackHypothesisNode:
        """Convert one confirmed external start Track into a root node."""
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

        track_id = self._next_track_id
        self._next_track_id += 1

        age = max(int(start.metadata.get("age", len(start))), 1)
        hits = int(start.metadata.get("hits", age))
        hits = min(max(hits, 1), age)
        state = start.states[-1]
        if self._last_scan_index is None:
            raise RuntimeError(
                "External starts require at least one completed update_tracker() call."
            )

        return self._create_root_node(
            track_id=track_id,
            scan_index=int(self._last_scan_index),
            timestamp=getattr(state, "timestamp", time),
            state=state,
            state_kind="external_start",
            used_det_key=None,
            assoc_label=TOMHTTracker.ASSOC_PAD,
            log_delta=0.0,
            age=age,
            hits=hits,
            root_source="external_start",
        )

    def _validate_external_starts_timestamp(self, time: datetime.datetime) -> None:
        """Validate add_external_starts(...) ordering and timestamp match."""
        if not isinstance(time, datetime.datetime):
            raise TypeError(
                "add_external_starts() time must be a datetime.datetime instance."
            )
        if self._last_update_timestamp is None or self._last_scan_index is None:
            raise RuntimeError(
                "add_external_starts() requires a completed update_tracker() first."
            )
        if time != self._last_update_timestamp:
            raise ValueError(
                "add_external_starts() time must match the most recent "
                f"completed update_tracker() timestamp. Expected {self._last_update_timestamp!r}, "
                f"got {time!r}."
            )

    # =========================================================================
    # Diagnostics / Instrumentation Helpers
    # =========================================================================

    def _run_scan_instrumentation(
        self,
        *,
        ctx: ScanContext,
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
    ) -> None:
        """Build/store per-scan stats and emit optional debug displays."""
        self._maybe_display_scan_debug_output(ctx)
        scan_stats = self._build_scan_stats(
            ctx=ctx,
            scan_wall_ms=scan_wall_ms,
            maxrss_mb=maxrss_mb,
            node_count_total=node_count_total,
            active_trees=active_trees,
            active_leaves=active_leaves,
            rebuild_stats=rebuild_stats,
            nscan_boundary_scan_index=nscan_boundary_scan_index,
            nscan_tracks_in_scope=nscan_tracks_in_scope,
            nscan_tracks_committed=nscan_tracks_committed,
            birth_stats=birth_stats,
        )
        self.last_scan_stats = scan_stats
        if self.params.collect_stats:
            self._stats.append(scan_stats)
        self._maybe_display_scan_stats(timestamp=ctx.timestamp, scan_stats=scan_stats)

    def _maybe_display_scan_debug_output(self, ctx: ScanContext) -> None:
        """Emit optional per-scan debug displays before stats logging."""
        if self.params.debug_display_detections:
            print(f"\nDetections at timestamp {ctx.timestamp}:")
            for det in ctx.detections:
                print(f"  {det.state_vector}")

        if self.params.debug_display_hypotheses:
            print(f"\nCluster rebuilds at timestamp {ctx.timestamp}:")
            self._display_cluster_rebuilds()

    def _display_cluster_rebuilds(self) -> None:
        """Print retained rebuilt globals for last scan (inspection only)."""
        for snapshot in self._last_cluster_snapshots:
            split_from = snapshot.overload_split_origin_cluster_id
            split_tag = "" if split_from is None else f" split_from={split_from}"
            print(
                f"cluster={snapshot.cluster_id} tracks={list(snapshot.track_ids)} "
                f"{split_tag}"
                f"globals={len(snapshot.rebuilt_globals)} "
                f"comb_eval={snapshot.evaluated_combinations} "
                f"comb_feas={snapshot.feasible_combinations} "
                f"disagree={snapshot.disagreement_count}"
            )
            for gh in snapshot.rebuilt_globals[: self.params.debug_globals_max]:
                tids = sorted(gh.leaf_nodes_by_track_id.keys())
                print(f"  logW={gh.log_weight:.3f} tids={tids}")

    def _map_stats_for_current_map(
        self,
        detections: list[Detection],
        scan_index: int,
    ) -> tuple[int, int, int, dict[int, int], float]:
        """Compute lightweight MAP-level counters for scan stats reporting."""
        map_snapshot = self.get_map_hypothesis_snapshot()
        map_tracks = 0
        map_used = 0
        map_unused = len(detections)
        map_miss_hist: dict[int, int] = {}
        map_mean_hit_rate = 0.0
        if map_snapshot is None:
            return map_tracks, map_used, map_unused, map_miss_hist, map_mean_hit_rate

        map_tracks = len(map_snapshot.leaf_nodes_by_track_id)
        used_keys = {
            leaf.used_det_key
            for leaf in map_snapshot.leaf_nodes_by_track_id.values()
            if leaf.used_det_key is not None and int(leaf.used_det_key[0]) == scan_index
        }
        map_used = len(used_keys)
        map_unused = len(detections) - map_used

        hit_rates: list[float] = []
        for leaf_node in map_snapshot.leaf_nodes_by_track_id.values():
            misses = int(leaf_node.missed_count)
            map_miss_hist[misses] = map_miss_hist.get(misses, 0) + 1
            age = int(leaf_node.age)
            if age > 0:
                hits = int(leaf_node.hits)
                hit_rates.append(float(hits) / float(age))
        if hit_rates:
            map_mean_hit_rate = float(np.mean(hit_rates))
        return map_tracks, map_used, map_unused, map_miss_hist, map_mean_hit_rate

    def _build_scan_stats(
        self,
        *,
        ctx: ScanContext,
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
    ) -> ScanStats:
        """Assemble one immutable per-scan ScanStats record."""
        map_tracks, map_used, map_unused, map_miss_hist, map_mean_hit_rate = (
            self._map_stats_for_current_map(ctx.detections, ctx.scan_index)
        )
        return ScanStats(
            timestamp=ctx.timestamp,
            scan_wall_ms=float(scan_wall_ms),
            maxrss_mb=float(maxrss_mb),
            node_count_total=int(node_count_total),
            active_trees=int(active_trees),
            active_leaves=int(active_leaves),
            num_detections=len(ctx.detections),
            cluster_count=rebuild_stats.cluster_count,
            combinations_evaluated=rebuild_stats.combinations_evaluated,
            feasible_combinations=rebuild_stats.feasible_combinations,
            rebuilt_globals_stored=rebuild_stats.rebuilt_globals_stored,
            nscan_disagreement_total=rebuild_stats.nscan_disagreement_total,
            overload_split_clusters=rebuild_stats.overload_split_clusters,
            overload_split_operations=rebuild_stats.overload_split_operations,
            historical_relaxation_attempts=rebuild_stats.historical_relaxation_attempts,
            historical_relaxation_successes=rebuild_stats.historical_relaxation_successes,
            historical_relaxed_keys_total=rebuild_stats.historical_relaxed_keys_total,
            nscan_boundary_scan_index=nscan_boundary_scan_index,
            nscan_tracks_in_scope=nscan_tracks_in_scope,
            nscan_tracks_committed=nscan_tracks_committed,
            birth_candidates=birth_stats.residual_detections_considered,
            birth_tracks_created=birth_stats.birth_tracks_created,
            birth_tracks_kept=birth_stats.birth_tracks_kept,
            map_tracks=map_tracks,
            map_used=map_used,
            map_unused=map_unused,
            map_miss_hist=map_miss_hist,
            map_mean_hit_rate=map_mean_hit_rate,
        )

    def _maybe_display_scan_stats(
        self,
        *,
        timestamp: datetime.datetime,
        scan_stats: ScanStats,
    ) -> None:
        """Emit optional human-readable scan stats summary output."""
        if not self.params.debug_display_scan_stats:
            return
        nscan_snapshot = self.get_n_scan_commitment_snapshot()
        print_scan_stats_report(
            timestamp=timestamp,
            scan_stats=scan_stats,
            nscan_snapshot=nscan_snapshot,
            debug_display_map_miss_hist=self.params.debug_display_map_miss_hist,
        )

    @staticmethod
    def _fmt_state_xyvxvy(state_vector) -> str:
        """Format `[x,vx,y,vy]` state vectors for compact debug output."""
        sv = np.asarray(state_vector, dtype=float)
        x = float(sv[0, 0])
        vx = float(sv[1, 0])
        y = float(sv[2, 0])
        vy = float(sv[3, 0])
        return f"(x={x:.1f}, vx={vx:.2f}, y={y:.1f}, vy={vy:.2f})"


# ============================================================================
# Public TOMHT Utility
# ============================================================================


def get_tomht_track_id(track: Track) -> int:
    """Return the stable TOMHT logical track ID from a TOMHT output track."""
    try:
        return int(track.metadata["track_id"])
    except KeyError as exc:
        raise KeyError(
            "Track metadata does not contain TOMHT 'track_id'. "
            "Use this helper only with TOMHTTracker-produced tracks."
        ) from exc
