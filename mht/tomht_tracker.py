"""Track-oriented MHT with persistent track trees and per-scan rebuilt globals.

Typical usage pattern:
```python
tracker = TOMHTTracker(
    updater=updater,
    predictor=predictor,
    initiator=initiator,
    params=params,
)
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

from dataclasses import dataclass, fields, replace
import datetime
import os
import sys
import time as wall_clock
from itertools import product
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from stonesoup.base import Property
from stonesoup.deleter.base import Deleter
from stonesoup.hypothesiser.base import Hypothesiser
from stonesoup.initiator.base import Initiator
from stonesoup.predictor.base import Predictor
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.tracker.base import Tracker, _TrackerMixInUpdate
from stonesoup.updater.base import Updater

from .tomht_births import (
    run_internal_births_after_expansion,
)
from .tomht_clustering import (
    ClusterWorkItem as _ClusterWorkItem,
    build_track_clusters,
)
from .tomht_cluster_rebuild import (
    cluster_leaf_options,
    infeasible_cluster_debug_summary,
    rebuild_cluster_globals,
)
from .tomht_cluster_solver import ClusterSolver
from .tomht_cluster_solver_branch_and_bound import BranchAndBoundClusterSolver
from .tomht_cluster_solver_exhaustive import ExhaustiveClusterSolver
from .tomht_expansion import (
    ExpansionCallStats as _ExpansionCallStats,
    expand_all_track_trees,
)
from .tomht_model import (
    ClusterRebuildSnapshot,
    DetectionKey,
    GlobalHypothesis,
    MAPHypothesisSnapshot,
    NScanCommitmentSnapshot,
    TrackHypothesisNode,
    TrackTree,
)
from .tomht_output import (
    reconstruct_track_from_committed_prefix_and_leaf_node,
)
from .tomht_params import TOMHTParams
from .tomht_pruning import apply_post_solve_supported_leaf_pruning
from .tomht_types import ScanContext
from .tomht_hypothesiser import TrackerOwnedNLLDistanceHypothesiser
from .tomht_scoring import (
    NLLScoringModel,
    _existence_probability_to_log_odds,
    maybe_log_scoring_diagnostics,
)
from .tomht_stats import (
    BirthStats,
    RebuildStats,
    ScanStats,
    ScanTimingBreakdown,
    print_scan_stats as print_scan_stats_report,
    print_summary_stats as print_summary_stats_report,
)
from .tomht_utils import (
    sorted_detections,
)
from .tomht_tree_store import TrackTreeStore
from .tomht_tree_utils import child_of_root_on_path, is_descendant_of
from .utils import get_process_maxrss_mb

# ============================================================================
# Tracker-Local Support Structures
# ============================================================================


class _DensePublishedTrackIdMapper:
    """Assign dense public IDs in first-publication order."""

    def __init__(self) -> None:
        self._next_public_track_id = 0

    def __call__(self, internal_track_id: int) -> int:
        del internal_track_id
        public_track_id = self._next_public_track_id
        self._next_public_track_id += 1
        return public_track_id


@dataclass(frozen=True)
class _MapNScanPruningPlan:
    """Planned MAP-only N-scan choices and diagnostics before mutation."""

    boundary_scan_index: int
    root_before_by_track_id: dict[int, TrackHypothesisNode]
    map_choice_by_track_id: dict[int, int]
    disagreement_total: int
    updated_snapshots: list[ClusterRebuildSnapshot]


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
    6. Rebuild feasible globals per cluster through the exact cluster-solver
       contract (default backend = branch-and-bound exact search) and choose MAP per
       cluster; overloaded clusters may first be approximately decomposed by
       severing weakest full-history conflict edges.
    7. Post-solve prune each cluster tree frontier to leaves supported by at
       least one retained rebuilt top-K global for that cluster.
    8. Merge cluster MAP selections into full-scan MAP.
    9. Apply MAP-only N-scan tree pruning: root promotion, committed states,
       active leaves, and disagreement stats.
    10. Apply whole-track lifecycle: sticky score-based confirmation, then
        post-N-scan termination. Score deletion always runs; node-native miss
        policy is used by default, and an optional Stone Soup deleter can replace
        the miss lane as a domain-specific hook.
    11. Update sticky output-publication state for MAP-selected live trees.
    12. Keep last-scan debug snapshots and return published MAP output tracks.

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

    # Stone Soup 1.8 under Python 3.14 can miss PEP-563 style class annotations
    # when resolving Property types, so keep explicit cls there.
    if sys.version_info >= (3, 14):
        updater = Property(
            Updater,
            doc="Updater supplied by caller. Required for tracker runtime use.",
        )
        predictor = Property(
            Predictor | None,
            default=None,
            doc="Predictor supplied by caller (or None when using custom hypothesiser).",
        )
        hypothesiser = Property(
            Hypothesiser | None,
            default=None,
            doc=(
                "Optional custom distance hypothesiser supplied by caller "
                "(or None when using predictor-driven default construction)."
            ),
        )
    else:
        updater: Updater = Property(
            doc="Updater supplied by caller. Required for tracker runtime use.",
        )
        predictor: Predictor | None = Property(
            default=None,
            doc="Predictor supplied by caller (or None when using custom hypothesiser).",
        )
        hypothesiser: Hypothesiser | None = Property(
            default=None,
            doc=(
                "Optional custom distance hypothesiser supplied by caller "
                "(or None when using predictor-driven default construction)."
            ),
        )

    _updater: Updater
    _hypothesiser: Hypothesiser
    _deleter: Deleter | None
    _output_track_id_mapper: Callable[[int], object]
    _external_start_initial_log_delta: float
    _track_confirmation_log_odds_threshold: float
    _track_deletion_log_odds_threshold: float
    _publish_lifecycle_states: frozenset[str]
    _publish_min_existence_log_odds_threshold: float | None

    # =========================================================================
    # Public API
    # =========================================================================

    def __init__(
        self,
        updater: Updater,
        predictor: Predictor | None = None,
        hypothesiser: Hypothesiser | None = None,
        *,
        detector: Any | None = None,
        initiator: Initiator | None = None,
        deleter: Deleter | None = None,
        params: TOMHTParams = TOMHTParams(),
        params_overrides: Mapping[str, Any] | None = None,
        output_track_id_mapper: Callable[[int], object] | None = None,
    ) -> None:
        """Construct the tracker with Stone Soup components and TO-MHT params.

        Parameters
        ----------
        updater : Updater
            Updater used for posterior state generation from selected hypotheses.
            This is always required.
        predictor : Predictor | None
            Predictor used to construct the tracker-owned default distance
            hypothesiser.
        hypothesiser : Hypothesiser | None
            Explicit custom Stone Soup hypothesiser returning distance
            hypotheses. Exactly one of ``predictor``
            or ``hypothesiser`` must be provided. For detection hypotheses,
            ``distance`` is expected to be ``-log p(z|x)`` at the predicted
            measurement (NLL only), without detection-probability or
            clutter-density factors.
        detector : Any | None
            Optional detector used when iterating over the tracker.
        initiator : Initiator | None
            Optional initiator for internal birth track creation.
        deleter : Deleter | None
            Optional Stone Soup deleter used for post-N-scan whole-track
            termination decisions. When provided, deleter-based lifecycle
            supersedes node-native miss-threshold lifecycle, but score-based
            deletion still runs.
        params : TOMHTParams
            Tracker configuration.
        params_overrides : Mapping[str, Any] | None
            Optional field-level overrides applied onto ``params``.
        output_track_id_mapper : Callable[[int], object] | None
            Optional mapping from the tracker-internal integer logical track ID
            to the public Stone Soup ``Track.id`` object assigned when a tree
            first becomes published. Defaults to dense integer IDs in
            first-publication order.
        """
        params = self._apply_params_overrides(params, params_overrides)
        external_start_initial_log_delta = _existence_probability_to_log_odds(
            params.external_start_initial_existence_probability,
            parameter_name="external_start_initial_existence_probability",
        )
        track_confirmation_log_odds_threshold = _existence_probability_to_log_odds(
            params.track_confirmation_existence_probability,
            parameter_name="track_confirmation_existence_probability",
        )
        track_deletion_log_odds_threshold = _existence_probability_to_log_odds(
            params.track_deletion_existence_probability,
            parameter_name="track_deletion_existence_probability",
        )
        publish_min_existence_log_odds_threshold: float | None
        if float(params.publish_min_existence_probability) <= 0.0:
            publish_min_existence_log_odds_threshold = None
        else:
            publish_min_existence_log_odds_threshold = (
                _existence_probability_to_log_odds(
                    params.publish_min_existence_probability,
                    parameter_name="publish_min_existence_probability",
                )
            )
        super().__init__(
            predictor=predictor,
            updater=updater,
            hypothesiser=hypothesiser,
        )
        self._updater = self.updater
        self._hypothesiser = self._resolve_hypothesiser(params=params)
        self._output_track_id_mapper = self._resolve_output_track_id_mapper(
            output_track_id_mapper
        )
        if deleter is not None and not hasattr(deleter, "check_for_deletion"):
            raise TypeError(
                "deleter must provide check_for_deletion(track, **kwargs) when provided."
            )
        self.detector = detector
        self.params = params
        self._external_start_initial_log_delta = external_start_initial_log_delta
        self._track_confirmation_log_odds_threshold = (
            track_confirmation_log_odds_threshold
        )
        self._track_deletion_log_odds_threshold = track_deletion_log_odds_threshold
        self._publish_lifecycle_states = frozenset(params.publish_lifecycle_states)
        self._publish_min_existence_log_odds_threshold = (
            publish_min_existence_log_odds_threshold
        )
        self.initiator = initiator
        self._deleter = deleter
        self.scoring_model = NLLScoringModel(
            prob_detect=float(params.prob_detect),
            clutter_density=float(params.clutter_density),
            log_epsilon=params.log_epsilon,
        )
        maybe_log_scoring_diagnostics(self.scoring_model)
        # Exact cluster-solver backend behind a narrow solver-facing contract.
        self._cluster_solver: ClusterSolver = self._make_cluster_solver(
            self.params.cluster_solver_backend
        )
        self._maybe_print_config()

        # Persistent tracker state.
        self._tree_store = TrackTreeStore()

        # Last-scan rebuilt artifacts retained for inspection only.
        self._last_cluster_snapshots: list[ClusterRebuildSnapshot] = []
        self.global_hypotheses: list[GlobalHypothesis] = [
            GlobalHypothesis(leaf_nodes_by_track_id={}, log_weight=0.0)
        ]
        self._last_map_global: GlobalHypothesis = self.global_hypotheses[0]

        # N-scan bookkeeping snapshots.
        self._nscan_commitment_snapshot = NScanCommitmentSnapshot(
            boundary_scan_index=None,
            tracks_in_scope=0,
            latest_committed_ancestor_by_track_id={},
            committed_boundary_by_track_id={},
            committed_ancestor_by_track_id={},
        )

        self._last_update_timestamp: datetime.datetime | None = None
        self._last_scan_index: int | None = None
        self._last_unused_detections: list[Detection] = []

        self.last_scan_stats: ScanStats | None = None
        self._stats: list[ScanStats] = []
        self.reset_stats()

    @property
    def _nodes_by_id(self) -> dict[int, TrackHypothesisNode]:
        """Compatibility view of the persistent node table owned by the store."""
        return self._tree_store.nodes_by_id

    @_nodes_by_id.setter
    def _nodes_by_id(self, value: dict[int, TrackHypothesisNode]) -> None:
        self._tree_store.nodes_by_id = value

    @property
    def track_trees_by_track_id(self) -> dict[int, TrackTree]:
        """Compatibility view of the persistent track-tree table owned by the store."""
        return self._tree_store.track_trees_by_track_id

    @track_trees_by_track_id.setter
    def track_trees_by_track_id(self, value: dict[int, TrackTree]) -> None:
        self._tree_store.track_trees_by_track_id = value

    @staticmethod
    def _make_cluster_solver(cluster_solver_backend: str) -> ClusterSolver:
        """Construct one exact cluster-solver backend by configured name."""
        backend = str(cluster_solver_backend).strip().lower()
        if backend == "exhaustive":
            return ExhaustiveClusterSolver()
        if backend in {"branch_and_bound", "branch-and-bound", "bnb"}:
            return BranchAndBoundClusterSolver()
        if backend in {"ortools", "ortools_cp_sat", "cp_sat"}:
            from .tomht_cluster_solver_ortools import ORToolsClusterSolver

            return ORToolsClusterSolver()
        raise ValueError(
            "Unknown cluster solver backend. "
            f"cluster_solver_backend={cluster_solver_backend!r}"
        )

    def _maybe_print_config(self) -> None:
        """Print a one-time resolved tracker config snapshot when enabled."""
        if not self.params.debug_display_config:
            return
        print("TOMHT_CONFIG resolved parameters:")
        for param_field in fields(TOMHTParams):
            value = getattr(self.params, param_field.name)
            print(f"  {param_field.name}={value!r}")
        print(
            "  "
            f"scoring_model={type(self.scoring_model).__name__} "
            f"cluster_solver={type(self._cluster_solver).__name__}"
        )

    def _resolve_hypothesiser(
        self,
        *,
        params: TOMHTParams,
    ) -> Hypothesiser:
        """Resolve constructor mode into the active runtime hypothesiser."""
        predictor = self.predictor
        hypothesiser = self.hypothesiser

        has_predictor = predictor is not None
        has_hypothesiser = hypothesiser is not None
        if has_predictor == has_hypothesiser:
            raise TypeError("Provide exactly one of predictor or hypothesiser.")

        if has_predictor:
            assert predictor is not None
            return TrackerOwnedNLLDistanceHypothesiser(
                predictor=predictor,
                updater=self.updater,
                mahalanobis_gate_threshold=params.mahalanobis_gate_threshold,
            )

        assert hypothesiser is not None
        resolved_predictor = getattr(hypothesiser, "predictor", None)
        if resolved_predictor is None:
            raise TypeError(
                "A provided hypothesiser must expose predictor for tracker wiring."
            )
        return hypothesiser

    @staticmethod
    def _apply_params_overrides(
        params: TOMHTParams,
        params_overrides: Mapping[str, Any] | None,
    ) -> TOMHTParams:
        """Apply JSON-style parameter overrides onto a frozen ``TOMHTParams``."""
        if params_overrides is None:
            return params
        if not isinstance(params_overrides, Mapping):
            raise TypeError(
                "params_overrides must be a mapping of TOMHTParams field names to values."
            )
        overrides = dict(params_overrides)
        if not overrides:
            return params
        non_string_keys = [key for key in overrides if not isinstance(key, str)]
        if non_string_keys:
            non_string_keys_str = ", ".join(repr(key) for key in non_string_keys)
            raise TypeError(
                "params_overrides keys must be strings matching TOMHTParams fields; "
                f"got: {non_string_keys_str}."
            )
        valid_keys = {field.name for field in fields(TOMHTParams)}
        invalid_keys = sorted(set(overrides).difference(valid_keys))
        if invalid_keys:
            invalid_keys_str = ", ".join(invalid_keys)
            raise ValueError(
                f"Unknown TOMHTParams override key(s): {invalid_keys_str}."
            )
        return replace(params, **overrides)

    @staticmethod
    def _resolve_output_track_id_mapper(
        output_track_id_mapper: Callable[[int], object] | None,
    ) -> Callable[[int], object]:
        """Resolve and validate output Track.id mapping strategy."""
        if output_track_id_mapper is None:
            return _DensePublishedTrackIdMapper()
        if not callable(output_track_id_mapper):
            raise TypeError("output_track_id_mapper must be callable when provided.")
        return output_track_id_mapper

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
        phase_start_ns = scan_wall_start_ns

        scan_index = (
            0 if self._last_scan_index is None else int(self._last_scan_index) + 1
        )
        det_list = sorted_detections(detections)
        det_index_by_obj = {id(det): i for i, det in enumerate(det_list)}
        ctx = ScanContext(
            scan_index=scan_index,
            timestamp=time,
            detections=det_list,
            det_index_by_obj=det_index_by_obj,
        )
        prep_ctx_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        self._maybe_validate_pruning_feasibility(
            stage="pre_local_expansion",
            ctx=ctx,
        )
        pre_expand_validate_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        # 1) Expand every tree locally.
        expansion_call_stats = _ExpansionCallStats()
        self._expand_all_track_trees(ctx, expansion_call_stats=expansion_call_stats)
        expand_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        # 2) Simple lifecycle handling.
        self._tree_store.remove_empty_trees()
        self._maybe_validate_pruning_feasibility(
            stage="post_local_pruning",
            ctx=ctx,
        )
        post_expand_prune_validate_ms = (
            wall_clock.perf_counter_ns() - phase_start_ns
        ) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        # 3) Internal births from Step-2 residual detections.
        birth_stats = self._run_internal_births(ctx)
        births_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        # 4) Build clusters and rebuild globals per cluster (fresh each scan).
        cluster_work = self._build_track_clusters(ctx)
        cluster_snapshots, rebuild_stats = self._rebuild_cluster_globals(
            cluster_work, ctx
        )
        cluster_build_and_solve_ms = (
            wall_clock.perf_counter_ns() - phase_start_ns
        ) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        # 5) Post-solve cluster-local supported-leaf pruning from rebuilt top-K.
        self._apply_post_solve_supported_leaf_pruning(cluster_snapshots)
        self._maybe_validate_pruning_feasibility(
            stage="post_supported_leaf_pruning",
            ctx=ctx,
        )
        post_solve_prune_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        map_global = self._merge_cluster_map_globals(cluster_snapshots)
        map_merge_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

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

        # 7) Whole-track lifecycle: sticky score-based confirmation,
        # then post-N-scan termination policy.
        self._apply_score_based_track_confirmation()
        map_global = self._apply_post_n_scan_track_lifecycle(
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
            scan_index=scan_index,
            timestamp=ctx.timestamp,
        )
        self._maybe_validate_pruning_feasibility(
            stage="post_n_scan_pruning",
            ctx=ctx,
        )

        # 8) Sticky output publication state for MAP-selected live trees.
        self._apply_output_publication(map_global)

        rebuild_stats = replace(
            rebuild_stats,
            nscan_disagreement_total=disagreement_total,
        )
        self._last_cluster_snapshots = cluster_snapshots

        # Keep one full-scan MAP global in compatibility slot for old inspection paths.
        self._last_map_global = map_global
        self.global_hypotheses = [map_global]
        nscan_lifecycle_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6
        phase_start_ns = wall_clock.perf_counter_ns()

        # 9) Reclaim node storage not reachable from surviving roots/leaves/commitments.
        nscan_snapshot = self._nscan_commitment_snapshot
        cleanup_seed_nodes = list(
            nscan_snapshot.latest_committed_ancestor_by_track_id.values()
        )
        cleanup_seed_nodes.extend(
            nscan_snapshot.committed_ancestor_by_track_id.values()
        )
        self._tree_store.cleanup_unreachable_nodes(
            extra_seed_nodes=cleanup_seed_nodes,
        )
        cleanup_ms = (wall_clock.perf_counter_ns() - phase_start_ns) / 1e6

        # 10) Post-scan instrumentation.
        scan_wall_ms = (wall_clock.perf_counter_ns() - scan_wall_start_ns) / 1e6
        maxrss_mb = get_process_maxrss_mb()
        node_count_total = len(self._tree_store.nodes_by_id)
        active_leaves = self._tree_store.active_leaf_count()

        timing_breakdown = ScanTimingBreakdown(
            prep_ctx_ms=float(prep_ctx_ms),
            pre_expand_validate_ms=float(pre_expand_validate_ms),
            expand_ms=float(expand_ms),
            expand_hypothesise_calls=int(expansion_call_stats.hypothesise_calls),
            expand_hypothesise_ms=float(expansion_call_stats.hypothesise_wall_ns / 1e6),
            expand_update_calls=int(expansion_call_stats.update_calls),
            expand_update_ms=float(expansion_call_stats.update_wall_ns / 1e6),
            post_expand_prune_validate_ms=float(post_expand_prune_validate_ms),
            births_ms=float(births_ms),
            cluster_build_and_solve_ms=float(cluster_build_and_solve_ms),
            post_solve_prune_ms=float(post_solve_prune_ms),
            map_merge_ms=float(map_merge_ms),
            nscan_lifecycle_ms=float(nscan_lifecycle_ms),
            cleanup_ms=float(cleanup_ms),
        )

        self._run_scan_instrumentation(
            ctx=ctx,
            scan_wall_ms=scan_wall_ms,
            maxrss_mb=maxrss_mb,
            node_count_total=node_count_total,
            active_trees=self._tree_store.active_tree_count(),
            active_leaves=active_leaves,
            rebuild_stats=rebuild_stats,
            nscan_boundary_scan_index=nscan_boundary_scan_index,
            nscan_tracks_in_scope=nscan_tracks_in_scope,
            nscan_tracks_committed=nscan_tracks_committed,
            birth_stats=birth_stats,
            timing_breakdown=timing_breakdown,
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

        new_roots = [
            self._make_external_start_root(start, time) for start in start_list
        ]

        # External starts are assumed to be from currently unused detections,
        # so add them directly to the last MAP view.
        tree_store = self._tree_store
        merged = dict(self._last_map_global.leaf_nodes_by_track_id)
        for track_id, tree in tree_store.track_trees_by_track_id.items():
            if track_id in merged:
                continue
            if len(tree.active_leaf_node_ids) != 1:
                continue
            only_leaf_id = next(iter(tree.active_leaf_node_ids))
            merged[track_id] = tree_store.nodes_by_id[only_leaf_id]

        self._last_map_global = GlobalHypothesis(
            leaf_nodes_by_track_id=merged,
            log_weight=float(self._last_map_global.log_weight)
            + sum(float(root.log_delta) for root in new_roots),
        )
        self._apply_output_publication(self._last_map_global)
        self.global_hypotheses = [self._last_map_global]

    def get_unused_detections(self) -> list[Detection]:
        """Return residual detections from the most recent completed update."""
        if self._last_update_timestamp is None:
            raise RuntimeError(
                "get_unused_detections() requires a completed update_tracker() first."
            )
        return list(self._last_unused_detections)

    # --- Read-only inspection / reporting helpers ---

    def get_map_output_tracks(
        self,
        *,
        include_unpublished: bool = False,
    ) -> set[Track]:
        """Return current MAP outputs as Stone Soup ``Track`` objects.

        By default this is the published output boundary. With
        ``include_unpublished=True``, all internal MAP-selected live trees are
        reconstructed for inspection and carry publication metadata.
        """
        map_snapshot = self.get_map_hypothesis_snapshot()
        if map_snapshot is None:
            return set()
        tree_store = self._tree_store
        output_tracks: set[Track] = set()
        for leaf_node in map_snapshot.leaf_nodes_by_track_id.values():
            tree = tree_store.track_trees_by_track_id.get(int(leaf_node.track_id))
            if tree is None:
                continue
            is_published = tree.publication_state == "published"
            if not include_unpublished and not is_published:
                continue
            if is_published:
                public_track_id = self._ensure_public_track_id(tree)
                output_track_id = public_track_id
            else:
                public_track_id = None
                output_track_id = int(leaf_node.track_id)
            committed_states = list(tree.committed_states)
            output_tracks.add(
                reconstruct_track_from_committed_prefix_and_leaf_node(
                    committed_states=committed_states,
                    leaf_node=leaf_node,
                    output_track_id=output_track_id,
                    lifecycle_state=tree.lifecycle_state,
                    publication_state=tree.publication_state,
                    public_track_id=public_track_id,
                )
            )
        return output_tracks

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
        snapshot = self._nscan_commitment_snapshot
        return NScanCommitmentSnapshot(
            boundary_scan_index=snapshot.boundary_scan_index,
            tracks_in_scope=int(snapshot.tracks_in_scope),
            latest_committed_ancestor_by_track_id=dict(
                snapshot.latest_committed_ancestor_by_track_id
            ),
            committed_boundary_by_track_id=dict(
                snapshot.committed_boundary_by_track_id
            ),
            committed_ancestor_by_track_id=dict(
                snapshot.committed_ancestor_by_track_id
            ),
        )

    def get_last_cluster_snapshots(self) -> tuple[ClusterRebuildSnapshot, ...]:
        """Return the most recent per-scan rebuilt-cluster snapshots."""
        return tuple(self._last_cluster_snapshots)

    def get_track_tree_snapshot(self) -> Mapping[int, dict[str, object]]:
        """Return a read-only snapshot of current persistent tree roots/leaves."""
        out: dict[int, dict[str, object]] = {}
        for track_id, tree in sorted(self._tree_store.track_trees_by_track_id.items()):
            out[track_id] = {
                "root_node_id": int(tree.root_node_id),
                "active_leaf_node_ids": tuple(sorted(tree.active_leaf_node_ids)),
                "root_source": tree.root_source,
                "lifecycle_state": tree.lifecycle_state,
                "publication_state": tree.publication_state,
                "public_track_id": tree.public_track_id,
            }
        return MappingProxyType(out)

    def print_summary_stats(self) -> None:
        """Print aggregate instrumentation summaries from collected ScanStats."""
        nscan_snapshot = self._nscan_commitment_snapshot
        print_summary_stats_report(
            stats=self._stats,
            max_global_hypotheses=self.params.max_global_hypotheses,
            last_nscan_boundary_scan_index=nscan_snapshot.boundary_scan_index,
            committed_boundary_by_track_id=nscan_snapshot.committed_boundary_by_track_id,
        )

    # =========================================================================
    # Local Expansion and Simple Lifecycle
    # =========================================================================

    def _expand_all_track_trees(
        self,
        ctx: ScanContext,
        *,
        expansion_call_stats: _ExpansionCallStats,
    ) -> None:
        """Run local expansion for all current persistent track trees."""
        expand_all_track_trees(
            tree_store=self._tree_store,
            ctx=ctx,
            hypothesiser=self._hypothesiser,
            updater=self._updater,
            scoring_model=self.scoring_model,
            params=self.params,
            assoc_miss_label=TOMHTTracker.ASSOC_MISS,
            expansion_call_stats=expansion_call_stats,
        )

    # =========================================================================
    # Internal Birth Handling
    # =========================================================================

    def _run_internal_births(self, ctx: ScanContext) -> BirthStats:
        """Create simple internal birth trees from Step-2 residual detections."""
        result = run_internal_births_after_expansion(
            ctx=ctx,
            initiator=self.initiator,
            tree_store=self._tree_store,
            params=self.params,
            assoc_pad_label=TOMHTTracker.ASSOC_PAD,
        )
        self._last_unused_detections = result.unused_detections
        return result.stats

    # =========================================================================
    # Per-Scan Clustering + Global Rebuild
    # =========================================================================

    def _build_track_clusters(self, ctx: ScanContext) -> list[_ClusterWorkItem]:
        """Build independent clusters from shared active-leaf history detections."""
        return build_track_clusters(
            tree_store=self._tree_store,
            scan_index=ctx.scan_index,
        )

    def _rebuild_cluster_globals(
        self,
        clusters: list[_ClusterWorkItem],
        ctx: ScanContext,
    ) -> tuple[list[ClusterRebuildSnapshot], RebuildStats]:
        """Thin wrapper around cluster rebuild orchestration."""
        return rebuild_cluster_globals(
            clusters=clusters,
            ctx=ctx,
            tree_store=self._tree_store,
            params=self.params,
            cluster_solver=self._cluster_solver,
        )

    # --- Solver guardrails / debug-only feasibility checks ---

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
        if not self._tree_store.track_trees_by_track_id:
            return

        clusters = self._build_track_clusters(ctx)
        for cluster in clusters:
            leaf_options = cluster_leaf_options(
                track_ids=cluster.track_ids,
                tree_store=self._tree_store,
            )
            if self._has_any_feasible_cluster_combination(leaf_options):
                continue
            dbg = infeasible_cluster_debug_summary(
                cluster=cluster,
                leaf_options=leaf_options,
                ctx=ctx,
            )
            raise RuntimeError(
                "Pruning feasibility check failed. " f"stage={stage}; {dbg}"
            )

    # --- Post-solve pruning + full-scan MAP merge ---

    def _apply_post_solve_supported_leaf_pruning(
        self,
        cluster_snapshots: list[ClusterRebuildSnapshot],
    ) -> None:
        """Prune each cluster tree to leaves supported by retained rebuilt globals."""
        apply_post_solve_supported_leaf_pruning(
            cluster_snapshots=cluster_snapshots,
            tree_store=self._tree_store,
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
                alt_child = child_of_root_on_path(root=root_before, leaf=leaf)
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
        tree_store = self._tree_store
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
        tree_store = self._tree_store
        snapshot = self._nscan_commitment_snapshot
        latest_committed_ancestor_by_track_id = dict(
            snapshot.latest_committed_ancestor_by_track_id
        )
        committed_boundary_by_track_id = dict(snapshot.committed_boundary_by_track_id)
        committed_ancestor_by_track_id = dict(snapshot.committed_ancestor_by_track_id)
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

        self._nscan_commitment_snapshot = replace(
            snapshot,
            latest_committed_ancestor_by_track_id=latest_committed_ancestor_by_track_id,
            committed_boundary_by_track_id=committed_boundary_by_track_id,
            committed_ancestor_by_track_id=committed_ancestor_by_track_id,
        )
        self._tree_store.remove_empty_trees()
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
        self._nscan_commitment_snapshot = replace(
            self._nscan_commitment_snapshot,
            boundary_scan_index=boundary_scan_index,
            tracks_in_scope=0,
            latest_committed_ancestor_by_track_id={},
        )

        if boundary_scan_index < 0 or not self._tree_store.track_trees_by_track_id:
            return boundary_scan_index, 0, 0, 0, cluster_snapshots

        plan = self._plan_map_n_scan_pruning(
            boundary_scan_index=boundary_scan_index,
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
        )
        self._nscan_commitment_snapshot = replace(
            self._nscan_commitment_snapshot,
            tracks_in_scope=len(plan.map_choice_by_track_id),
        )
        committed_count = self._apply_planned_map_n_scan_pruning(plan=plan)
        return (
            boundary_scan_index,
            len(plan.map_choice_by_track_id),
            committed_count,
            plan.disagreement_total,
            plan.updated_snapshots,
        )

    # =========================================================================
    # Tree-Level Confirmation Lifecycle
    # =========================================================================

    def _apply_score_based_track_confirmation(self) -> int:
        """Promote tentative trees whose active frontier score crosses threshold."""
        confirmed_count = 0
        threshold = float(self._track_confirmation_log_odds_threshold)
        for _, tree in sorted(self._tree_store.track_trees_by_track_id.items()):
            if tree.lifecycle_state != "tentative":
                continue
            tree_score = self._tree_store.active_tree_max_accumulated_log_score(tree)
            if tree_score is None:
                continue
            if float(tree_score) >= threshold:
                tree.lifecycle_state = "confirmed"
                confirmed_count += 1
        return confirmed_count

    # =========================================================================
    # Output Publication State
    # =========================================================================

    def _map_leaf_satisfies_publication_policy(
        self,
        *,
        tree: TrackTree,
        leaf: TrackHypothesisNode,
    ) -> bool:
        """Return whether a MAP leaf can first transition to published output."""
        if tree.lifecycle_state not in self._publish_lifecycle_states:
            return False
        if int(leaf.hits) < int(self.params.publish_min_hits):
            return False
        if int(leaf.age) < int(self.params.publish_min_age):
            return False

        threshold = self._publish_min_existence_log_odds_threshold
        if threshold is not None and float(leaf.accumulated_log_score) < threshold:
            return False
        return True

    def _apply_output_publication(self, map_global: GlobalHypothesis) -> int:
        """Stickily publish MAP-selected trees that satisfy output policy."""
        published_count = 0
        track_trees_by_track_id = self._tree_store.track_trees_by_track_id
        for track_id, leaf in sorted(map_global.leaf_nodes_by_track_id.items()):
            tree = track_trees_by_track_id.get(track_id)
            if tree is None:
                continue
            if tree.publication_state == "published":
                self._ensure_public_track_id(tree)
                continue
            if not self._map_leaf_satisfies_publication_policy(tree=tree, leaf=leaf):
                continue
            self._ensure_public_track_id(tree)
            tree.publication_state = "published"
            published_count += 1
        return published_count

    def _ensure_public_track_id(self, tree: TrackTree) -> object:
        """Return an existing public ID, assigning one if publication needs repair."""
        if tree.public_track_id is not None:
            return tree.public_track_id
        public_track_id = self._output_track_id_mapper(int(tree.track_id))
        if public_track_id is None:
            raise ValueError(
                "output_track_id_mapper returned None for a published TOMHT track; "
                "None is reserved for unpublished inspection tracks."
            )
        tree.public_track_id = public_track_id
        return public_track_id

    # =========================================================================
    # Whole-Track Lifecycle
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
        nodes_by_id = self._tree_store.nodes_by_id
        root = nodes_by_id[tree.root_node_id]

        if mode == "map_leaf":
            map_leaf = map_global.leaf_nodes_by_track_id.get(track_id)
            if map_leaf is not None and is_descendant_of(
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
                leaf = nodes_by_id.get(node_id)
                if leaf is None:
                    continue
                if is_descendant_of(node=leaf, ancestor=root):
                    candidate_leaves.append(leaf)
            if candidate_leaves:
                return candidate_leaves

        # Default and safe fallback for empty map/global-k sets.
        return [nodes_by_id[node_id] for node_id in sorted(tree.active_leaf_node_ids)]

    def _filter_map_global_to_live_trees(
        self,
        map_global: GlobalHypothesis,
    ) -> GlobalHypothesis:
        """Drop map entries for tracks that no longer have active trees."""
        track_trees_by_track_id = self._tree_store.track_trees_by_track_id
        filtered_nodes = {
            track_id: leaf
            for track_id, leaf in map_global.leaf_nodes_by_track_id.items()
            if track_id in track_trees_by_track_id
        }
        return GlobalHypothesis(
            leaf_nodes_by_track_id=filtered_nodes,
            log_weight=float(map_global.log_weight),
        )

    @staticmethod
    def _add_track_termination_reason(
        termination_reasons_by_track_id: dict[int, set[str]],
        *,
        track_id: int,
        reason: str,
    ) -> None:
        """Record one deterministic whole-track termination reason."""
        termination_reasons_by_track_id.setdefault(int(track_id), set()).add(reason)

    def _collect_score_track_termination_reasons(self) -> dict[int, set[str]]:
        """Return whole-tree score deletions from max active-leaf score."""
        threshold = float(self._track_deletion_log_odds_threshold)
        termination_reasons_by_track_id: dict[int, set[str]] = {}
        for track_id, tree in sorted(self._tree_store.track_trees_by_track_id.items()):
            tree_score = self._tree_store.active_tree_max_accumulated_log_score(tree)
            if tree_score is None:
                continue
            if float(tree_score) <= threshold:
                self._add_track_termination_reason(
                    termination_reasons_by_track_id,
                    track_id=track_id,
                    reason="score",
                )
        return termination_reasons_by_track_id

    @staticmethod
    def _format_track_termination_reasons(
        termination_reasons_by_track_id: dict[int, set[str]],
    ) -> str:
        """Format track termination reasons in stable reason/track order."""
        parts: list[str] = []
        for reason in ("score", "miss", "deleter"):
            track_ids = [
                track_id
                for track_id in sorted(termination_reasons_by_track_id)
                if reason in termination_reasons_by_track_id[track_id]
            ]
            if track_ids:
                parts.append(f"{reason}:{track_ids}")
        return ";".join(parts)

    def _remove_terminated_track_trees(
        self,
        *,
        termination_reasons_by_track_id: dict[int, set[str]],
        mode: str,
        miss_threshold: int | None,
        deleter: Deleter | None,
    ) -> None:
        """Remove terminated trees and emit one deterministic lifecycle diagnostic."""
        if not termination_reasons_by_track_id:
            return

        terminated_track_ids = sorted(termination_reasons_by_track_id)
        for track_id in terminated_track_ids:
            self._tree_store.track_trees_by_track_id.pop(track_id, None)

        diagnostic_parts = [
            "TRACK_LIFECYCLE",
            f"mode={mode}",
            f"score_threshold={self._track_deletion_log_odds_threshold:.6g}",
        ]
        if miss_threshold is not None:
            diagnostic_parts.append(f"miss_threshold={miss_threshold}")
        if deleter is not None:
            diagnostic_parts.append(f"deleter={type(deleter).__name__}")
        diagnostic_parts.append(f"terminated={terminated_track_ids}")
        diagnostic_parts.append(
            "reasons="
            + self._format_track_termination_reasons(termination_reasons_by_track_id)
        )
        print(" ".join(diagnostic_parts))

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

        termination_reasons_by_track_id = (
            self._collect_score_track_termination_reasons()
        )
        track_trees_by_track_id = self._tree_store.track_trees_by_track_id
        for track_id, tree in sorted(track_trees_by_track_id.items()):
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
                self._add_track_termination_reason(
                    termination_reasons_by_track_id,
                    track_id=track_id,
                    reason="miss",
                )

        self._remove_terminated_track_trees(
            termination_reasons_by_track_id=termination_reasons_by_track_id,
            mode=mode,
            miss_threshold=threshold,
            deleter=None,
        )

        self._tree_store.remove_empty_trees()
        return self._filter_map_global_to_live_trees(map_global)

    def _apply_post_n_scan_track_deleter_lifecycle(
        self,
        *,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
        scan_index: int,
        timestamp: datetime.datetime,
    ) -> GlobalHypothesis:
        """Apply whole-track lifecycle using the configured Stone Soup deleter."""
        del scan_index  # reserved for potential future diagnostics.
        deleter = self._deleter
        if deleter is None:
            raise RuntimeError(
                "Deleter lifecycle requested without a configured deleter."
            )
        mode = self._normalized_track_miss_termination_mode(
            self.params.track_miss_termination_mode
        )

        termination_reasons_by_track_id = (
            self._collect_score_track_termination_reasons()
        )
        track_trees_by_track_id = self._tree_store.track_trees_by_track_id
        for track_id, tree in sorted(track_trees_by_track_id.items()):
            leaves = self._track_miss_termination_leaves(
                track_id=track_id,
                tree=tree,
                mode=mode,
                map_global=map_global,
                cluster_snapshots=cluster_snapshots,
            )
            if not leaves:
                continue

            committed_states = list(tree.committed_states)
            leaf_delete_decisions: list[bool] = []
            for leaf in leaves:
                candidate_track = reconstruct_track_from_committed_prefix_and_leaf_node(
                    committed_states=committed_states,
                    leaf_node=leaf,
                    output_track_id=int(leaf.track_id),
                    lifecycle_state=tree.lifecycle_state,
                    public_track_id=tree.public_track_id,
                )
                should_delete = bool(
                    deleter.check_for_deletion(candidate_track, timestamp=timestamp)
                )
                leaf_delete_decisions.append(should_delete)
            if leaf_delete_decisions and all(leaf_delete_decisions):
                self._add_track_termination_reason(
                    termination_reasons_by_track_id,
                    track_id=track_id,
                    reason="deleter",
                )

        self._remove_terminated_track_trees(
            termination_reasons_by_track_id=termination_reasons_by_track_id,
            mode=mode,
            miss_threshold=None,
            deleter=deleter,
        )

        self._tree_store.remove_empty_trees()
        return self._filter_map_global_to_live_trees(map_global)

    def _apply_post_n_scan_track_lifecycle(
        self,
        *,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
        scan_index: int,
        timestamp: datetime.datetime,
    ) -> GlobalHypothesis:
        """Apply post-N-scan whole-track lifecycle using the configured lane."""
        if self._deleter is not None:
            return self._apply_post_n_scan_track_deleter_lifecycle(
                map_global=map_global,
                cluster_snapshots=cluster_snapshots,
                scan_index=scan_index,
                timestamp=timestamp,
            )
        return self._apply_post_n_scan_track_miss_lifecycle(
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
            scan_index=scan_index,
        )

    # =========================================================================
    # External-Start Helpers
    # =========================================================================

    def _make_external_start_root(
        self,
        start: Track,
        time: datetime.datetime,
    ) -> TrackHypothesisNode:
        """Convert one confirmed external start Track into an inserted root node."""
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

        age = max(int(start.metadata.get("age", len(start))), 1)
        hits = int(start.metadata.get("hits", age))
        hits = min(max(hits, 1), age)
        state = start.states[-1]
        if self._last_scan_index is None:
            raise RuntimeError(
                "External starts require at least one completed update_tracker() call."
            )
        log_delta = self._external_start_initial_log_delta
        # Input metadata["existence_probability"] is an optional upstream initial
        # prior. Output metadata["existence_probability"] is tracker score-implied.
        metadata_existence_probability = start.metadata.get("existence_probability")
        if metadata_existence_probability is not None:
            try:
                log_delta = _existence_probability_to_log_odds(
                    metadata_existence_probability,
                    parameter_name="external start metadata['existence_probability']",
                )
            except ValueError:
                log_delta = self._external_start_initial_log_delta

        return self._tree_store.create_root_tree_for_new_track(
            scan_index=int(self._last_scan_index),
            timestamp=getattr(state, "timestamp", time),
            state=state,
            state_kind="external_start",
            used_det_key=None,
            assoc_label=TOMHTTracker.ASSOC_PAD,
            log_delta=log_delta,
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
        timing_breakdown: ScanTimingBreakdown,
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
            timing_breakdown=timing_breakdown,
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
            print(f"\nCluster rebuilds scan={ctx.scan_index} t={ctx.timestamp}:")
            self._display_cluster_rebuilds()

    def _display_cluster_rebuilds(self) -> None:
        """Print retained rebuilt globals for last scan (inspection only)."""
        for snapshot in self._last_cluster_snapshots:
            split_from = snapshot.overload_split_origin_cluster_id
            split_tag = "" if split_from is None else f" split_from={split_from}"
            print(
                f"cluster={snapshot.cluster_id} tracks={list(snapshot.track_ids)}{split_tag} "
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
    ) -> tuple[int, int, int, int, int, dict[int, int], float]:
        """Compute lightweight MAP-level counters for scan stats reporting."""
        map_snapshot = self.get_map_hypothesis_snapshot()
        map_tracks = 0
        map_published_tracks = 0
        map_unpublished_tracks = 0
        map_used = 0
        map_unused = len(detections)
        map_miss_hist: dict[int, int] = {}
        map_mean_hit_rate = 0.0
        if map_snapshot is None:
            return (
                map_tracks,
                map_published_tracks,
                map_unpublished_tracks,
                map_used,
                map_unused,
                map_miss_hist,
                map_mean_hit_rate,
            )

        map_tracks = len(map_snapshot.leaf_nodes_by_track_id)
        track_trees_by_track_id = self._tree_store.track_trees_by_track_id
        for track_id in map_snapshot.leaf_nodes_by_track_id:
            tree = track_trees_by_track_id.get(track_id)
            if tree is not None and tree.publication_state == "published":
                map_published_tracks += 1
        map_unpublished_tracks = map_tracks - map_published_tracks

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
        return (
            map_tracks,
            map_published_tracks,
            map_unpublished_tracks,
            map_used,
            map_unused,
            map_miss_hist,
            map_mean_hit_rate,
        )

    def _active_lifecycle_tree_counts(self) -> tuple[int, int]:
        """Return active-tree counts by tentative/confirmed lifecycle state."""
        tentative = 0
        confirmed = 0
        for tree in self._tree_store.track_trees_by_track_id.values():
            if tree.lifecycle_state == "confirmed":
                confirmed += 1
            else:
                tentative += 1
        return tentative, confirmed

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
        timing_breakdown: ScanTimingBreakdown,
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
        ) = self._map_stats_for_current_map(ctx.detections, ctx.scan_index)
        active_tentative_trees, active_confirmed_trees = (
            self._active_lifecycle_tree_counts()
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
            map_published_tracks=map_published_tracks,
            map_unpublished_tracks=map_unpublished_tracks,
            map_used=map_used,
            map_unused=map_unused,
            map_miss_hist=map_miss_hist,
            map_mean_hit_rate=map_mean_hit_rate,
            timing_breakdown=timing_breakdown,
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


# ============================================================================
# Public TOMHT Utility
# ============================================================================


def get_tomht_track_id(track: Track) -> int:
    """Return the internal TOMHT logical track ID from a TOMHT output track."""
    try:
        return int(track.metadata["track_id"])
    except KeyError as exc:
        raise KeyError(
            "Track metadata does not contain TOMHT 'track_id'. "
            "Use this helper only with TOMHTTracker-produced tracks."
        ) from exc
