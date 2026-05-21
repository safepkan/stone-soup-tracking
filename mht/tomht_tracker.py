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

from dataclasses import fields, replace
import datetime
import os
import sys
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

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
    has_any_feasible_cluster_combination,
    infeasible_cluster_debug_summary,
    merge_cluster_map_globals,
    rebuild_cluster_globals,
)
from .tomht_cluster_solver import ClusterSolver
from .tomht_cluster_solver_factory import make_cluster_solver
from .tomht_external_starts import (
    insert_external_start_trees,
    validate_external_starts_timestamp,
)
from .tomht_expansion import (
    ExpansionCallStats as _ExpansionCallStats,
    expand_all_track_trees,
)
from .tomht_model import (
    ClusterRebuildSnapshot,
    GlobalHypothesis,
    MAPHypothesisSnapshot,
    NScanCommitmentSnapshot,
    ScanContext,
    TrackHypothesisNode,
    TrackTree,
)
from .tomht_lifecycle import (
    DeleterWithMetadata,
    apply_post_n_scan_track_lifecycle,
    apply_score_based_track_confirmation,
    internal_track_id_for_deleter_candidate,
    resolve_deleter_with_metadata,
)
from .tomht_output import (
    apply_output_publication,
    ensure_public_track_id,
    reconstruct_map_output_tracks,
    resolve_output_track_id_mapper,
)
from .tomht_params import TOMHTParams, apply_params_overrides
from .tomht_pruning import (
    SupportedLeafPruningStats,
    apply_map_n_scan_pruning,
    apply_post_solve_supported_leaf_pruning,
)
from .tomht_hypothesiser import TrackerOwnedNLLDistanceHypothesiser
from .tomht_scoring import (
    ConstantDetectionProbabilityModel,
    DetectionProbabilityModel,
    NLLScoringModel,
    _existence_probability_to_log_odds,
    maybe_log_scoring_diagnostics,
)
from .tomht_stats import (
    BirthStats,
    ExpansionFrontierStats,
    RebuildStats,
    ScanStats,
    ScanTimingBreakdown,
    build_expansion_frontier_stats,
    build_scan_stats,
    frontier_stage_counts_for_store,
    maybe_display_scan_debug_output,
    print_expansion_frontier_stats,
    print_scan_stats as print_scan_stats_report,
    print_summary_stats as print_summary_stats_report,
)
from .tomht_utils import (
    sorted_detections,
)
from .tomht_tree_store import TrackTreeStore
from .utils import elapsed_ms, get_process_maxrss_mb, ns_to_ms, start_timer

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
       severing weakest live conflict edges.
    7. Post-solve prune each cluster tree frontier to leaves supported by at
       least one retained rebuilt top-K global for that cluster.
    8. Merge cluster MAP selections into full-scan MAP.
    9. Apply MAP-only N-scan tree pruning: root promotion, committed states,
       active leaves, and disagreement stats.
    10. Apply whole-track lifecycle: sticky score-based confirmation, then
        post-N-scan termination. Score deletion always runs; TOMHT resolves an
        internal miss-count deleter by default, and an optional Stone Soup
        deleter can replace that default as a domain-specific hook.
    11. Update sticky output-publication state for MAP-selected live trees.
    12. Keep last-scan debug snapshots and return published MAP output tracks.

    Behavior notes for readability:
    - Exact behavior: cluster feasibility checks and exclusivity constraints use
      live unresolved detection keys on active leaves.
    - Safety valves: pre-solve per-tree leaf capping and birth load guards.
    - Approximation paths: overload cluster decomposition.
    - Inspection/debug retention: last-scan cluster snapshots and scan stats.
    """

    ASSOC_PAD = -1
    ASSOC_MISS = -2

    # =========================================================================
    # Stone Soup Constructor Properties
    # =========================================================================

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

    _hypothesiser: Hypothesiser

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
        detection_probability_model: DetectionProbabilityModel | None = None,
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
            Optional Stone Soup deleter used for post-N-scan tree deletion
            decisions. When omitted, TOMHT resolves an internal miss-count
            deleter from ``params``. Score-based deletion always runs.
        params : TOMHTParams
            Tracker configuration.
        params_overrides : Mapping[str, Any] | None
            Optional field-level overrides applied onto ``params``.
        output_track_id_mapper : Callable[[int], object] | None
            Optional mapping from the tracker-internal integer logical track ID
            to the public Stone Soup ``Track.id`` object assigned when a tree
            first becomes published. Defaults to dense integer IDs in
            first-publication order.
        detection_probability_model : DetectionProbabilityModel | None
            Optional dynamic model for per-hypothesis ``P_D`` and clutter
            density. When omitted, scalar ``TOMHTParams.prob_detect`` and
            ``TOMHTParams.clutter_density`` are wrapped in
            ``ConstantDetectionProbabilityModel`` to preserve default scoring.
        """
        super().__init__(
            predictor=predictor,
            updater=updater,
            hypothesiser=hypothesiser,
        )
        if deleter is not None and not hasattr(deleter, "check_for_deletion"):
            raise TypeError(
                "deleter must provide check_for_deletion(track, **kwargs) when provided."
            )

        self.detector = detector
        self.initiator = initiator
        self.deleter = deleter

        self._output_track_id_mapper = resolve_output_track_id_mapper(
            output_track_id_mapper
        )

        params = apply_params_overrides(params, params_overrides)
        self.params = params
        self._deleter_with_metadata: DeleterWithMetadata = (
            resolve_deleter_with_metadata(params=params, deleter=deleter)
        )
        self._hypothesiser = self._resolve_hypothesiser(params=params)

        self._external_start_initial_log_delta: float = (
            _existence_probability_to_log_odds(
                params.external_start_initial_existence_probability,
                parameter_name="external_start_initial_existence_probability",
            )
        )
        self._track_confirmation_log_odds_threshold: float = (
            _existence_probability_to_log_odds(
                params.track_confirmation_existence_probability,
                parameter_name="track_confirmation_existence_probability",
            )
        )
        self._track_deletion_log_odds_threshold: float = (
            _existence_probability_to_log_odds(
                params.track_deletion_existence_probability,
                parameter_name="track_deletion_existence_probability",
            )
        )
        self._publish_lifecycle_states: frozenset[str] = frozenset(
            params.publish_lifecycle_states
        )
        self._publish_min_existence_log_odds_threshold: float | None
        if float(params.publish_min_existence_probability) <= 0.0:
            self._publish_min_existence_log_odds_threshold = None
        else:
            self._publish_min_existence_log_odds_threshold = (
                _existence_probability_to_log_odds(
                    params.publish_min_existence_probability,
                    parameter_name="publish_min_existence_probability",
                )
            )

        resolved_dpm = detection_probability_model
        if resolved_dpm is None:
            resolved_dpm = ConstantDetectionProbabilityModel(
                prob_detect=float(params.prob_detect),
                clutter_density=float(params.clutter_density),
            )
        self.scoring_model = NLLScoringModel(
            detection_probability_model=resolved_dpm,
            log_epsilon=params.log_epsilon,
        )
        maybe_log_scoring_diagnostics(self.scoring_model)
        self._cluster_solver: ClusterSolver = make_cluster_solver(
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
    def tracks(self) -> set[Track]:
        """Current MAP output as Stone Soup ``Track`` objects."""
        return self.get_map_output_tracks()

    def update_tracker(
        self,
        time: datetime.datetime,
        detections: Iterable[Detection],
        *,
        caller_scan_context: object | None = None,
    ) -> tuple[datetime.datetime, set[Track]]:
        """Run one single-sensor scan update and return ``(time, MAP tracks)``.

        One call is expected to contain detections from one sensor / one
        measurement space. Multi-sensor applications should call this once per
        sensor update with the corresponding opaque ``caller_scan_context``.
        That caller context is threaded to the DetectionProbabilityModel and is
        distinct from TOMHT's internal ``ScanContext`` bookkeeping.
        """
        scan_wall_start_ns = start_timer()
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
            caller_scan_context=caller_scan_context,
        )
        prep_ctx_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

        self._maybe_validate_pruning_feasibility(
            stage="pre_local_expansion",
            ctx=ctx,
        )
        pre_expand_validate_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

        # 1) Expand every tree locally.
        frontier_before_expansion = frontier_stage_counts_for_store(
            tree_store=self._tree_store
        )
        expansion_call_stats = _ExpansionCallStats()
        self._expand_all_track_trees(ctx, expansion_call_stats=expansion_call_stats)
        frontier_after_expansion = frontier_stage_counts_for_store(
            tree_store=self._tree_store
        )
        expand_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

        # 2) Simple lifecycle handling.
        self._tree_store.remove_empty_trees()
        frontier_after_empty_tree_removal = frontier_stage_counts_for_store(
            tree_store=self._tree_store
        )
        self._maybe_validate_pruning_feasibility(
            stage="post_local_pruning",
            ctx=ctx,
        )
        post_expand_prune_validate_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

        # 3) Internal births from Step-2 residual detections.
        birth_stats = self._run_internal_births(ctx)
        frontier_after_births = frontier_stage_counts_for_store(
            tree_store=self._tree_store
        )
        births_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

        # 4) Build clusters and rebuild globals per cluster (fresh each scan).
        cluster_work = self._build_track_clusters(ctx)
        cluster_snapshots, rebuild_stats = self._rebuild_cluster_globals(
            cluster_work, ctx
        )
        cluster_build_and_solve_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

        # 5) Post-solve cluster-local supported-leaf pruning from rebuilt top-K.
        supported_pruning_stats = self._apply_post_solve_supported_leaf_pruning(
            cluster_snapshots
        )
        frontier_after_post_solve_supported_pruning = frontier_stage_counts_for_store(
            tree_store=self._tree_store
        )
        self._maybe_validate_pruning_feasibility(
            stage="post_supported_leaf_pruning",
            ctx=ctx,
        )
        post_solve_prune_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

        map_global = merge_cluster_map_globals(cluster_snapshots)
        map_merge_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

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
        frontier_after_n_scan_pruning = frontier_stage_counts_for_store(
            tree_store=self._tree_store
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
        frontier_after_lifecycle = frontier_stage_counts_for_store(
            tree_store=self._tree_store
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
        nscan_lifecycle_ms = elapsed_ms(phase_start_ns)
        phase_start_ns = start_timer()

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
        cleanup_ms = elapsed_ms(phase_start_ns)

        # 10) Post-scan instrumentation.
        scan_wall_ms = elapsed_ms(scan_wall_start_ns)
        maxrss_mb = get_process_maxrss_mb()
        node_count_total = len(self._tree_store.nodes_by_id)
        active_leaves = self._tree_store.active_leaf_count()
        expansion_frontier_stats = build_expansion_frontier_stats(
            before_expansion=frontier_before_expansion,
            after_expansion=frontier_after_expansion,
            after_empty_tree_removal=frontier_after_empty_tree_removal,
            after_births=frontier_after_births,
            after_post_solve_supported_pruning=(
                frontier_after_post_solve_supported_pruning
            ),
            after_n_scan_pruning=frontier_after_n_scan_pruning,
            after_lifecycle=frontier_after_lifecycle,
            expansion_call_stats=expansion_call_stats,
            supported_pruning_stats=supported_pruning_stats,
            cluster_snapshots=cluster_snapshots,
        )

        timing_breakdown = ScanTimingBreakdown(
            prep_ctx_ms=float(prep_ctx_ms),
            pre_expand_validate_ms=float(pre_expand_validate_ms),
            expand_ms=float(expand_ms),
            expand_hypothesise_calls=int(expansion_call_stats.hypothesise_calls),
            expand_hypothesise_ms=ns_to_ms(expansion_call_stats.hypothesise_wall_ns),
            expand_update_calls=int(expansion_call_stats.update_calls),
            expand_update_ms=ns_to_ms(expansion_call_stats.update_wall_ns),
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
            expansion_frontier_stats=expansion_frontier_stats,
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
        validate_external_starts_timestamp(
            time=time,
            last_update_timestamp=self._last_update_timestamp,
            last_scan_index=self._last_scan_index,
        )
        start_list = list(starts)
        if not start_list:
            return

        result = insert_external_start_trees(
            time=time,
            starts=start_list,
            tree_store=self._tree_store,
            last_scan_index=self._last_scan_index,
            last_map_global=self._last_map_global,
            external_start_default_log_delta=self._external_start_initial_log_delta,
            assoc_pad_label=TOMHTTracker.ASSOC_PAD,
        )
        self._last_map_global = result.map_global
        self._apply_score_based_track_confirmation()
        self._apply_output_publication(self._last_map_global)
        self.global_hypotheses = [self._last_map_global]

    def get_unused_detections(self) -> list[Detection]:
        """Return residual detections from the most recent completed update."""
        if self._last_update_timestamp is None:
            raise RuntimeError(
                "get_unused_detections() requires a completed update_tracker() first."
            )
        return list(self._last_unused_detections)

    # =========================================================================
    # Inspection and Stats API
    # =========================================================================

    def reset_stats(self) -> None:
        """Clear collected ScanStats and the last per-scan snapshot."""
        self._stats = []
        self.last_scan_stats = None

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
        return reconstruct_map_output_tracks(
            tree_store=self._tree_store,
            map_snapshot=self.get_map_hypothesis_snapshot(),
            include_unpublished=include_unpublished,
            output_track_id_mapper=self._output_track_id_mapper,
        )

    def get_map_hypothesis_snapshot(self) -> MAPHypothesisSnapshot:
        """Return read-only node-native MAP state for inspection/debug."""
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
            debug_display_expansion_frontier=(self._expansion_frontier_debug_enabled()),
        )

    # =========================================================================
    # Compatibility Inspection Views
    # =========================================================================

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

    # =========================================================================
    # Constructor Helpers
    # =========================================================================

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
            updater=self.updater,
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
        """Build independent clusters from shared active-leaf live detections."""
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

    # =========================================================================
    # Post-Solve Supported-Leaf Pruning
    # =========================================================================

    def _apply_post_solve_supported_leaf_pruning(
        self,
        cluster_snapshots: list[ClusterRebuildSnapshot],
    ) -> SupportedLeafPruningStats:
        """Prune each cluster tree to leaves supported by retained rebuilt globals."""
        return apply_post_solve_supported_leaf_pruning(
            cluster_snapshots=cluster_snapshots,
            tree_store=self._tree_store,
        )

    # =========================================================================
    # MAP-Only N-Scan Pruning on Explicit Trees
    # =========================================================================

    def _apply_map_n_scan_pruning(
        self,
        *,
        scan_index: int,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
    ) -> tuple[int, int, int, int, list[ClusterRebuildSnapshot]]:
        """Apply MAP-only N-scan root-child promotion and disagreement bookkeeping."""
        result = apply_map_n_scan_pruning(
            scan_index=scan_index,
            ns_scan_window=self.params.ns_scan_window,
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
            tree_store=self._tree_store,
            nscan_commitment_snapshot=self._nscan_commitment_snapshot,
        )
        self._nscan_commitment_snapshot = result.nscan_commitment_snapshot
        return (
            result.boundary_scan_index,
            result.tracks_in_scope,
            result.committed_count,
            result.disagreement_total,
            result.updated_snapshots,
        )

    # =========================================================================
    # Tree-Level Confirmation Lifecycle
    # =========================================================================

    def _apply_score_based_track_confirmation(self) -> int:
        """Promote tentative trees whose active frontier score crosses threshold."""
        return apply_score_based_track_confirmation(
            tree_store=self._tree_store,
            confirmation_log_odds_threshold=self._track_confirmation_log_odds_threshold,
        )

    # =========================================================================
    # Output Publication State
    # =========================================================================

    def _apply_output_publication(self, map_global: GlobalHypothesis) -> int:
        """Stickily publish MAP-selected trees that satisfy output policy."""
        return apply_output_publication(
            tree_store=self._tree_store,
            map_global=map_global,
            publish_lifecycle_states=self._publish_lifecycle_states,
            publish_min_hits=self.params.publish_min_hits,
            publish_min_age=self.params.publish_min_age,
            publish_min_existence_log_odds_threshold=(
                self._publish_min_existence_log_odds_threshold
            ),
            output_track_id_mapper=self._output_track_id_mapper,
        )

    def _ensure_public_track_id(self, tree: TrackTree) -> object:
        """Return an existing public ID, assigning one if publication needs repair."""
        return ensure_public_track_id(
            tree=tree,
            output_track_id_mapper=self._output_track_id_mapper,
        )

    # =========================================================================
    # Whole-Track Lifecycle
    # =========================================================================

    def _apply_post_n_scan_track_lifecycle(
        self,
        *,
        map_global: GlobalHypothesis,
        cluster_snapshots: list[ClusterRebuildSnapshot],
        scan_index: int,
        timestamp: datetime.datetime,
    ) -> GlobalHypothesis:
        """Apply post-N-scan whole-track lifecycle using the configured deleter."""
        del scan_index  # reserved for potential future diagnostics.
        return apply_post_n_scan_track_lifecycle(
            tree_store=self._tree_store,
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
            params=self.params,
            deletion_log_odds_threshold=self._track_deletion_log_odds_threshold,
            deleter_with_metadata=self._deleter_with_metadata,
            output_track_id_for_deleter=internal_track_id_for_deleter_candidate,
            timestamp=timestamp,
        )

    # =========================================================================
    # Debug Validation Helpers
    # =========================================================================

    @staticmethod
    def _pruning_feasibility_validation_enabled() -> bool:
        """Return whether debug-only pruning feasibility validation is enabled."""
        raw = os.getenv("TOMHT_DEBUG_VALIDATE_PRUNING_FEASIBILITY")
        if raw is None:
            return False
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _expansion_frontier_debug_enabled(self) -> bool:
        """Return whether opt-in expansion/frontier diagnostics are enabled."""
        if self.params.debug_display_expansion_frontier:
            return True
        raw = os.getenv("TOMHT_DEBUG_EXPANSION_FRONTIER")
        if raw is None:
            return False
        return raw.strip().lower() in {"1", "true", "yes", "on"}

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
            if has_any_feasible_cluster_combination(
                cluster=cluster,
                leaf_options=leaf_options,
                tree_store=self._tree_store,
            ):
                continue
            dbg = infeasible_cluster_debug_summary(
                cluster=cluster,
                leaf_options=leaf_options,
                tree_store=self._tree_store,
                ctx=ctx,
            )
            raise RuntimeError(
                "Pruning feasibility check failed. " f"stage={stage}; {dbg}"
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
        expansion_frontier_stats: ExpansionFrontierStats,
    ) -> None:
        """Build/store per-scan stats and emit optional debug displays."""
        maybe_display_scan_debug_output(
            ctx=ctx,
            cluster_snapshots=self._last_cluster_snapshots,
            debug_display_detections=self.params.debug_display_detections,
            debug_display_hypotheses=self.params.debug_display_hypotheses,
            debug_globals_max=self.params.debug_globals_max,
        )
        scan_stats = build_scan_stats(
            ctx=ctx,
            map_global=self._last_map_global,
            tree_store=self._tree_store,
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
            expansion_frontier_stats=expansion_frontier_stats,
        )
        self.last_scan_stats = scan_stats
        if self.params.collect_stats:
            self._stats.append(scan_stats)
        debug_display_expansion_frontier = self._expansion_frontier_debug_enabled()
        if not self.params.debug_display_scan_stats:
            if debug_display_expansion_frontier:
                print_expansion_frontier_stats(
                    timestamp=ctx.timestamp,
                    scan_stats=scan_stats,
                )
            return
        nscan_snapshot = self.get_n_scan_commitment_snapshot()
        print_scan_stats_report(
            timestamp=ctx.timestamp,
            scan_stats=scan_stats,
            nscan_snapshot=nscan_snapshot,
            debug_display_map_miss_hist=self.params.debug_display_map_miss_hist,
        )
        if debug_display_expansion_frontier:
            print_expansion_frontier_stats(
                timestamp=ctx.timestamp,
                scan_stats=scan_stats,
            )
