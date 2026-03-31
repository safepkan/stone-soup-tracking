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
    - per-leaf local branching and miss tolerance,
    - MAP-only N-scan pruning window,
    - optional debug/stat visibility toggles.

    Compatibility note:
    ``max_global_hypotheses`` is retained as a cap for how many rebuilt globals
    are kept per cluster for debug/snapshot storage; it is no longer a persistent
    beam frontier carried scan-to-scan.
    """

    # Local expansion / lifecycle controls.
    max_children_per_track: int = 5
    # Optional per-tree frontier cap applied after local expansion.
    max_leaves_per_track_tree: int | None = 50
    max_missed: int = 5

    # Rebuilt-global storage cap (debug/inspection cap, not persistent beam state).
    max_global_hypotheses: int = 20

    # Scoring / numerical behavior.
    scoring_mode: str = "beta_ratio"
    log_epsilon: float = 1e-12
    prob_gate: float = 0.99

    # MAP-only N-scan pruning: boundary is b = k - N.
    ns_scan_window: int = 3

    # Internal birth handling (kept intentionally simple in this phase).
    max_births_per_scan: int = 2
    birth_log_penalty: float = 8.0
    unused_det_log_penalty: float = 0.2

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
    3. Apply simple lifecycle filtering (drop leaves with miss budget exceeded,
       drop trees with no surviving active leaves).
    4. Optionally create internal birth trees from detections unused by the union
       of surviving active leaves after Step 3.
    5. Recompute measurement-exclusivity clusters from current trees.
    6. Rebuild feasible globals per cluster via exhaustive enumeration and choose
       MAP per cluster.
    7. Merge cluster MAP selections into full-scan MAP, then apply MAP-only
       N-scan tree pruning.
    8. Keep last-scan debug snapshots and return MAP output tracks.
    """

    ASSOC_PAD = -1
    ASSOC_MISS = -2

    hypothesiser: PDAHypothesiser = Property(
        doc="Hypothesiser used to branch per-track hypotheses."
    )
    updater: Updater = Property(
        doc="Updater used to generate posteriors from selected hypotheses."
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

        # 1) Expand every tree locally.
        self._expand_all_track_trees(ctx)

        # 2) Simple lifecycle handling.
        self._remove_empty_trees()

        # 3) Internal births from Step-2 residual detections.
        birth_stats = self._run_internal_births(ctx)

        # 4) Build clusters and rebuild globals per cluster (fresh each scan).
        cluster_work = self._build_track_clusters(ctx)
        cluster_snapshots, rebuild_stats = self._rebuild_cluster_globals(
            cluster_work, ctx
        )

        map_global = self._merge_cluster_map_globals(cluster_snapshots)
        self._last_map_global = map_global

        # 5) MAP-only N-scan pruning on explicit trees + disagreement stats.
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

        rebuild_stats = replace(
            rebuild_stats,
            nscan_disagreement_total=disagreement_total,
        )
        self._last_cluster_snapshots = cluster_snapshots

        # Keep one full-scan MAP global in compatibility slot for old inspection paths.
        self.global_hypotheses = [map_global]

        # 6) Reclaim node storage not reachable from surviving roots/leaves/commitments.
        self._cleanup_unreachable_nodes()

        # 7) Post-scan instrumentation.
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
        det_list = list(detections)
        det_list.sort(key=self._det_sort_key)
        return det_list

    @staticmethod
    def _get_process_maxrss_mb() -> float:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return float(ru.ru_maxrss) / (1024.0 * 1024.0)
        return float(ru.ru_maxrss) / 1024.0

    @staticmethod
    def _current_scan_det_indices_from_keys(
        keys: Iterable[DetectionKey], scan_index: int
    ) -> set[int]:
        return {det_idx for (key_scan, det_idx) in keys if key_scan == scan_index}

    # =========================================================================
    # Node/Tree Construction Helpers
    # =========================================================================

    def _allocate_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _register_node(self, node: TrackHypothesisNode) -> TrackHypothesisNode:
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
        multi = self.hypothesiser.hypothesise(track, ctx.detections, ctx.timestamp)
        singles = list(multi)

        hyp_scores = self.scoring_model.score_track_hypotheses(
            track=track,
            hypotheses=singles,
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

        singles_sorted = sorted(singles, key=_sort_key, reverse=True)
        kept = singles_sorted[: self.params.max_children_per_track]
        miss = next((h for h in singles_sorted if not h), None)
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
        """Expand all active leaves in one persistent track tree."""
        new_leaf_ids: set[int] = set()

        for leaf_id in sorted(tree.active_leaf_node_ids):
            leaf = self._nodes_by_id[leaf_id]
            candidates = self._candidates_for_track_leaf(leaf, ctx)
            for cand in candidates:
                if int(cand.child_node.missed_count) > self.params.max_missed:
                    continue
                new_leaf_ids.add(cand.child_node.node_id)

        max_leaves = self.params.max_leaves_per_track_tree
        if max_leaves is not None and len(new_leaf_ids) > max_leaves:
            ranked = sorted(
                (self._nodes_by_id[node_id] for node_id in new_leaf_ids),
                key=lambda node: (
                    float(node.accumulated_log_score),
                    -int(node.node_id),
                ),
                reverse=True,
            )
            new_leaf_ids = {node.node_id for node in ranked[: int(max_leaves)]}

        tree.active_leaf_node_ids = new_leaf_ids

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

        self._last_unused_detections = []
        if not residual_detections:
            return BirthStats(
                residual_detections_considered=0,
                birth_tracks_created=0,
                birth_tracks_kept=0,
            )

        born = list(
            self.initiator.initiate(OrderedSet(residual_detections), ctx.timestamp)
        )
        birth_tracks_created = len(born)

        # Keep this phase intentionally simple: numeric sanity + fixed cap.
        born = [tr for tr in born if self._birth_is_sane(tr)]
        if len(born) > self.params.max_births_per_scan:
            born = born[: self.params.max_births_per_scan]
        birth_tracks_kept = len(born)

        if self.params.debug_display_births and born:
            print(f"\nInternal births at {ctx.timestamp}: kept={birth_tracks_kept}")
            for tr in born[: self.params.debug_births_max]:
                state = tr.states[-1].state_vector
                print(f"  birth_state={self._fmt_state_xyvxvy(state)}")

        for tr in born:
            track_id = self._next_track_id
            self._next_track_id += 1
            state = tr.states[-1]
            used_key = self._birth_used_key(
                tr,
                scan_index=ctx.scan_index,
                det_index_by_obj=ctx.det_index_by_obj,
            )
            age = max(len(tr), 1)
            hits = 1 if used_key is not None else 0
            root_log_delta = self.scoring_model.score_birth(
                birth_track=tr,
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

    def _current_scan_candidate_keys_for_tree(
        self,
        tree: TrackTree,
        scan_index: int,
    ) -> set[DetectionKey]:
        keys: set[DetectionKey] = set()
        for leaf_id in tree.active_leaf_node_ids:
            leaf = self._nodes_by_id[leaf_id]
            if (
                leaf.used_det_key is not None
                and int(leaf.used_det_key[0]) == scan_index
            ):
                keys.add(leaf.used_det_key)
        return keys

    def _build_track_clusters(self, ctx: ScanContext) -> list[_ClusterWorkItem]:
        """Build current independent clusters from shared current-scan detections."""
        track_ids = sorted(self.track_trees_by_track_id.keys())
        if not track_ids:
            return []

        keys_by_track: dict[int, set[DetectionKey]] = {
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
                shared = keys_by_track[left_track_id] & keys_by_track[right_track_id]
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
                        track_id: set(keys_by_track[track_id])
                        for track_id in comp_track_ids
                    },
                    conflict_links=comp_links,
                )
            )
        return out

    def _score_unused_cluster_current_scan_term(
        self,
        *,
        cluster_universe: set[DetectionKey],
        selected_used_current_scan_keys: set[DetectionKey],
        ctx: ScanContext,
    ) -> float:
        """Compute explicit per-combination cluster-local unused-detection term."""
        if not cluster_universe:
            return 0.0

        det_indices = sorted(
            det_idx
            for (scan_idx, det_idx) in cluster_universe
            if scan_idx == ctx.scan_index
        )
        if not det_indices:
            return 0.0

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
        used_local_slots = {
            local_slot_by_global_det_index[det_idx]
            for (scan_idx, det_idx) in selected_used_current_scan_keys
            if scan_idx == ctx.scan_index and det_idx in local_slot_by_global_det_index
        }
        return self.scoring_model.score_unused_detections(
            used_det_keys=used_local_slots,
            ctx=local_ctx,
        )

    def _cluster_leaf_options(
        self,
        track_ids: tuple[int, ...],
    ) -> list[list[TrackHypothesisNode]]:
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
        top_k_heap: list[tuple[float, int, GlobalHypothesis]],
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
        top_k_heap: list[tuple[float, int, GlobalHypothesis]],
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

    def _rebuild_one_cluster(
        self,
        cluster: _ClusterWorkItem,
        ctx: ScanContext,
    ) -> ClusterRebuildSnapshot:
        """Exhaustively enumerate and score feasible globals for one cluster."""
        leaf_options = self._cluster_leaf_options(cluster.track_ids)
        cluster_universe: set[DetectionKey] = set()
        for keys in cluster.current_scan_det_keys_by_track_id.values():
            cluster_universe |= keys

        top_k_heap: list[tuple[float, int, GlobalHypothesis]] = []
        combinations_evaluated = 0
        feasible_combinations = 0
        k = int(self.params.max_global_hypotheses)

        for picked in product(*leaf_options):
            combinations_evaluated += 1
            selected = list(picked)

            feasible = True
            used_keys: set[DetectionKey] = set()
            for leaf in selected:
                overlap = used_keys & set(leaf.detection_history_keys)
                if overlap:
                    feasible = False
                    break
                used_keys |= set(leaf.detection_history_keys)
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
                cluster_universe=cluster_universe,
                selected_used_current_scan_keys=used_current_scan_keys,
                ctx=ctx,
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

        if feasible_combinations == 0:
            raise RuntimeError(
                "Cluster rebuild found no feasible combination. "
                "Expected at least one feasible joint assignment."
            )

        kept_globals = self._finalize_top_k_globals(top_k_heap)
        map_global = kept_globals[0] if kept_globals else None

        return ClusterRebuildSnapshot(
            cluster_id=cluster.cluster_id,
            track_ids=cluster.track_ids,
            current_scan_conflict_det_keys=frozenset(cluster_universe),
            conflict_links=cluster.conflict_links,
            rebuilt_globals=kept_globals,
            map_global=map_global,
            feasible_combinations=feasible_combinations,
            evaluated_combinations=combinations_evaluated,
        )

    def _rebuild_cluster_globals(
        self,
        clusters: list[_ClusterWorkItem],
        ctx: ScanContext,
    ) -> tuple[list[ClusterRebuildSnapshot], RebuildStats]:
        if not clusters:
            return [], RebuildStats()

        snapshots = [self._rebuild_one_cluster(cluster, ctx) for cluster in clusters]
        return (
            snapshots,
            RebuildStats(
                cluster_count=len(snapshots),
                combinations_evaluated=sum(s.evaluated_combinations for s in snapshots),
                feasible_combinations=sum(s.feasible_combinations for s in snapshots),
                rebuilt_globals_stored=sum(len(s.rebuilt_globals) for s in snapshots),
                nscan_disagreement_total=0,
            ),
        )

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

        self._last_nscan_tracks_in_scope = len(map_choice_by_track_id)

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

        committed_count = 0
        for track_id, chosen_child_id in map_choice_by_track_id.items():
            current_tree = self.track_trees_by_track_id.get(track_id)
            if current_tree is None:
                continue
            root_before = root_before_by_track_id[track_id]
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
            if prev_boundary is None or boundary_scan_index > prev_boundary:
                self._committed_boundary_by_track_id[track_id] = boundary_scan_index
                self._committed_ancestor_by_track_id[track_id] = chosen_child
            committed_count += 1

            # Root promotion detaches the old root lineage from this tree.
            root_before.child_node_ids = {
                child_id
                for child_id in root_before.child_node_ids
                if child_id == chosen_child_id
            }

        self._remove_empty_trees()
        return (
            boundary_scan_index,
            len(map_choice_by_track_id),
            committed_count,
            disagreement_total,
            updated_snapshots,
        )

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
        if self.params.debug_display_detections:
            print(f"\nDetections at timestamp {ctx.timestamp}:")
            for det in ctx.detections:
                print(f"  {det.state_vector}")

        if self.params.debug_display_hypotheses:
            print(f"\nCluster rebuilds at timestamp {ctx.timestamp}:")
            self._display_cluster_rebuilds()

    def _display_cluster_rebuilds(self) -> None:
        for snapshot in self._last_cluster_snapshots:
            print(
                f"cluster={snapshot.cluster_id} tracks={list(snapshot.track_ids)} "
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
