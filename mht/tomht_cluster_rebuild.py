"""Cluster rebuild orchestration for the track-oriented MHT tracker."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .tomht_clustering import (
    ClusterWorkItem,
    OverloadSplitSummary,
    apply_overload_splitting_to_clusters,
)
from .tomht_cluster_solver import (
    ClusterSolver,
    ClusterSolverDiagnostics,
    ClusterSolverLeafOption,
    ClusterSolverProblem,
    ClusterSolverResult,
    ClusterSolverTrackOptions,
    missing_cluster_solver_diagnostics,
)
from .tomht_model import (
    ClusterRebuildSnapshot,
    DetectionKey,
    GlobalHypothesis,
    TrackHypothesisNode,
)
from .tomht_params import TOMHTParams
from .tomht_scoring import ScoringModel
from .tomht_stats import RebuildStats
from .tomht_tree_store import TrackTreeStore
from .tomht_types import ScanContext
from .tomht_utils import format_detection_key_sample


@dataclass(frozen=True)
class _ClusterUnusedScoreContext:
    """Precomputed cluster-local context for unused-detection scoring."""

    local_ctx: ScanContext


@dataclass(frozen=True)
class _ClusterCurrentScanScoreDecomposition:
    """Per-hit + constant decomposition for cluster-local current-scan scoring."""

    per_hit_log_bonus: float
    constant_log_offset: float


@dataclass(frozen=True)
class _ClusterRebuildResult:
    """Cluster rebuild result with narrow historical-relaxation bookkeeping."""

    snapshot: ClusterRebuildSnapshot
    historical_relaxation_attempted: bool = False
    historical_relaxation_succeeded: bool = False
    historical_relaxed_key_count: int = 0


@dataclass(frozen=True)
class _ClusterSolveInput:
    """Tracker-side prepared cluster-solve inputs before policy wrappers."""

    cluster: ClusterWorkItem
    ctx: ScanContext
    leaf_options: list[list[TrackHypothesisNode]]
    cluster_universe: set[DetectionKey]
    unused_score_context: _ClusterUnusedScoreContext | None


@dataclass(frozen=True)
class _PreparedClusterSolveProblem:
    """One solver-facing cluster problem plus leaf-ID mapping back to nodes."""

    problem: ClusterSolverProblem
    leaf_node_by_leaf_id: dict[int, TrackHypothesisNode]


@dataclass(frozen=True)
class _ClusterSolveOutcome:
    """Solver outcome with optional historical-relaxation bookkeeping."""

    kept_globals: tuple[GlobalHypothesis, ...]
    combinations_evaluated: int
    feasible_combinations: int
    historical_relaxation_attempted: bool = False
    historical_relaxation_succeeded: bool = False
    historical_relaxed_keys: frozenset[DetectionKey] = frozenset()


def _build_cluster_unused_score_context(
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

    return _ClusterUnusedScoreContext(local_ctx=local_ctx)


def _build_cluster_current_scan_score_decomposition(
    *,
    score_context: _ClusterUnusedScoreContext | None,
    scoring_model: ScoringModel,
) -> _ClusterCurrentScanScoreDecomposition | None:
    """Build per-hit + constant decomposition for tracker scoring."""
    if score_context is None:
        return None

    used_none: set[int] = set()
    total_current_scan_det_count = len(score_context.local_ctx.detections)
    if total_current_scan_det_count <= 0:
        return None

    baseline = float(
        scoring_model.score_unused_detections(
            used_det_keys=used_none,
            ctx=score_context.local_ctx,
        )
    )

    # Derive per-used slope from two points and verify the linear shape for
    # the current scoring contract that this solver interface supports.
    first_key = 0
    first_used_score = float(
        scoring_model.score_unused_detections(
            used_det_keys={first_key},
            ctx=score_context.local_ctx,
        )
    )
    per_used = float(first_used_score - baseline)

    if total_current_scan_det_count >= 2:
        second_used_score = float(
            scoring_model.score_unused_detections(
                used_det_keys={0, 1},
                ctx=score_context.local_ctx,
            )
        )
        second_increment = float(second_used_score - first_used_score)
        if not np.isclose(second_increment, per_used, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "Current cluster-solver leaf-score pre-baking assumes a linear "
                "per-hit current-scan clutter correction. Tracker scoring model "
                "produced non-linear cluster unused-detection behavior."
            )

    predicted_all_used = baseline + per_used * float(total_current_scan_det_count)
    all_used_score = float(
        scoring_model.score_unused_detections(
            used_det_keys=set(range(total_current_scan_det_count)),
            ctx=score_context.local_ctx,
        )
    )
    if not np.isclose(predicted_all_used, all_used_score, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Current cluster-solver leaf-score pre-baking assumes a linear "
            "per-hit current-scan clutter correction. Tracker scoring model "
            "produced inconsistent endpoint scores."
        )

    return _ClusterCurrentScanScoreDecomposition(
        per_hit_log_bonus=per_used,
        constant_log_offset=baseline,
    )


def _projected_combination_count(
    leaf_options: list[list[TrackHypothesisNode]],
) -> int:
    """Return projected Cartesian product size for one leaf-option set."""
    projected = 1
    for leaves in leaf_options:
        projected *= len(leaves)
    return projected


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


def _build_cluster_solver_problem(
    *,
    cluster: ClusterWorkItem,
    leaf_options: list[list[TrackHypothesisNode]],
    ctx: ScanContext,
    cluster_universe: set[DetectionKey],
    unused_score_context: _ClusterUnusedScoreContext | None,
    relaxed_conflict_keys: frozenset[DetectionKey],
    params: TOMHTParams,
    scoring_model: ScoringModel,
) -> _PreparedClusterSolveProblem:
    """Build one solver-facing exact cluster problem from tracker state."""
    relaxed_key_set = set(relaxed_conflict_keys)
    score_decomposition = _build_cluster_current_scan_score_decomposition(
        score_context=unused_score_context,
        scoring_model=scoring_model,
    )
    per_hit_log_bonus = (
        0.0 if score_decomposition is None else score_decomposition.per_hit_log_bonus
    )
    constant_log_offset = (
        0.0 if score_decomposition is None else score_decomposition.constant_log_offset
    )
    leaf_node_by_leaf_id: dict[int, TrackHypothesisNode] = {}
    track_options: list[ClusterSolverTrackOptions] = []

    for idx, track_id in enumerate(cluster.track_ids):
        solver_leaf_options: list[ClusterSolverLeafOption] = []
        for leaf in leaf_options[idx]:
            leaf_id = int(leaf.node_id)
            leaf_node_by_leaf_id[leaf_id] = leaf

            conflict_keys = set(leaf.detection_history_keys)
            if relaxed_key_set:
                conflict_keys -= relaxed_key_set

            used_current_scan_keys = sorted(
                key
                for key in leaf.detection_history_keys
                if key[0] == ctx.scan_index and key in cluster_universe
            )
            if len(used_current_scan_keys) > 1:
                raise RuntimeError(
                    "Cluster solver contract requires at most one current-scan "
                    "detection per leaf option. "
                    f"track_id={track_id} leaf_id={leaf_id} "
                    f"current_scan_keys={used_current_scan_keys}"
                )
            uses_current_scan_detection = bool(used_current_scan_keys)

            if (
                uses_current_scan_detection
                and used_current_scan_keys[0] not in conflict_keys
            ):
                raise RuntimeError(
                    "Current-scan detections cannot be relaxed out of "
                    "full-history conflict keys for exact cluster solving. "
                    f"track_id={track_id} leaf_id={leaf_id} "
                    f"det_key={used_current_scan_keys[0]}"
                )

            solver_leaf_options.append(
                ClusterSolverLeafOption(
                    leaf_id=leaf_id,
                    track_id=int(track_id),
                    score=float(leaf.accumulated_log_score)
                    + (
                        float(per_hit_log_bonus) if uses_current_scan_detection else 0.0
                    ),
                    full_history_conflict_keys=frozenset(conflict_keys),
                )
            )

        track_options.append(
            ClusterSolverTrackOptions(
                track_id=int(track_id),
                leaf_options=tuple(solver_leaf_options),
            )
        )

    return _PreparedClusterSolveProblem(
        problem=ClusterSolverProblem(
            track_options=tuple(track_options),
            max_results=int(params.max_global_hypotheses),
            constant_score_offset=float(constant_log_offset),
        ),
        leaf_node_by_leaf_id=leaf_node_by_leaf_id,
    )


def _solver_result_to_globals(
    *,
    solver_result: ClusterSolverResult,
    leaf_node_by_leaf_id: Mapping[int, TrackHypothesisNode],
) -> tuple[GlobalHypothesis, ...]:
    """Map solver-facing leaf IDs back to node-native rebuilt globals."""
    out: list[GlobalHypothesis] = []
    for solution in solver_result.solutions:
        leaf_nodes_by_track_id: dict[int, TrackHypothesisNode] = {}
        for track_id, leaf_id in solution.selected_leaf_id_by_track_id.items():
            leaf_node = leaf_node_by_leaf_id.get(int(leaf_id))
            if leaf_node is None:
                raise RuntimeError(
                    "Cluster solver returned an unknown leaf ID. "
                    f"track_id={track_id} leaf_id={leaf_id}"
                )
            leaf_nodes_by_track_id[int(track_id)] = leaf_node
        out.append(
            GlobalHypothesis(
                leaf_nodes_by_track_id=leaf_nodes_by_track_id,
                log_weight=float(solution.score),
            )
        )
    return tuple(out)


def _solve_cluster_exact(
    *,
    prepared_problem: _PreparedClusterSolveProblem,
    cluster_solver: ClusterSolver,
) -> tuple[tuple[GlobalHypothesis, ...], ClusterSolverDiagnostics]:
    """Run one exact cluster solve call through the solver interface."""
    solver_result = cluster_solver.solve(prepared_problem.problem)
    kept_globals = _solver_result_to_globals(
        solver_result=solver_result,
        leaf_node_by_leaf_id=prepared_problem.leaf_node_by_leaf_id,
    )
    diagnostics = cluster_solver.get_last_diagnostics()
    if diagnostics is None:
        diagnostics = missing_cluster_solver_diagnostics()
    return kept_globals, diagnostics


def infeasible_cluster_debug_summary(
    *,
    cluster: ClusterWorkItem,
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
        parts.append("pairwise_conflicts={" + ", ".join(pairwise_overlap_counts) + "}")

    return "; ".join(parts)


def _forced_detection_history_keys(
    leaves: list[TrackHypothesisNode],
) -> set[DetectionKey]:
    """Return detection keys present in every active leaf for one track tree."""
    forced = set(leaves[0].detection_history_keys)
    for leaf in leaves[1:]:
        forced &= set(leaf.detection_history_keys)
    return forced


def _historical_relaxed_conflict_keys_for_cluster(
    *,
    cluster: ClusterWorkItem,
    leaf_options: list[list[TrackHypothesisNode]],
    ctx: ScanContext,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
) -> set[DetectionKey]:
    """Return forced committed historical keys shared by multiple tracks."""
    boundary_scan_index = int(ctx.scan_index) - int(params.ns_scan_window)
    key_track_count: dict[DetectionKey, int] = {}
    for idx, track_id in enumerate(cluster.track_ids):
        leaves = leaf_options[idx]
        forced_keys = _forced_detection_history_keys(leaves)
        tree = tree_store.track_trees_by_track_id[track_id]
        root = tree_store.nodes_by_id[tree.root_node_id]
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
    *,
    cluster: ClusterWorkItem,
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
        f"{format_detection_key_sample(relaxed_keys)} "
        f"feasible_before={feasible_before} "
        f"feasible_after={feasible_after} "
        f"status={'enabled' if feasible_after > 0 else 'failed'}"
    )


def _raise_cluster_infeasible_error(
    *,
    solve_input: _ClusterSolveInput,
    relaxed_historical_keys: set[DetectionKey],
) -> None:
    """Raise the existing cluster infeasibility error with optional relax debug."""
    dbg = infeasible_cluster_debug_summary(
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
            f"{format_detection_key_sample(relaxed_historical_keys)}"
        )
    raise RuntimeError(
        "Cluster rebuild found no feasible combination. "
        "Expected at least one feasible joint assignment. "
        f"{dbg}{relaxation_dbg}"
    )


def _solve_with_optional_historical_relaxation(
    *,
    solve_input: _ClusterSolveInput,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    scoring_model: ScoringModel,
    cluster_solver: ClusterSolver,
) -> _ClusterSolveOutcome:
    """Solve one cluster, with optional relaxed-key retry around exact solve."""
    prepared_problem = _build_cluster_solver_problem(
        cluster=solve_input.cluster,
        leaf_options=solve_input.leaf_options,
        ctx=solve_input.ctx,
        cluster_universe=solve_input.cluster_universe,
        unused_score_context=solve_input.unused_score_context,
        relaxed_conflict_keys=frozenset(),
        params=params,
        scoring_model=scoring_model,
    )
    kept_globals, solve_diagnostics = _solve_cluster_exact(
        prepared_problem=prepared_problem,
        cluster_solver=cluster_solver,
    )
    combinations_evaluated = int(solve_diagnostics.combinations_evaluated)
    feasible_combinations = int(solve_diagnostics.feasible_combinations)

    historical_relaxation_attempted = False
    historical_relaxation_succeeded = False
    relaxed_historical_keys: set[DetectionKey] = set()
    if feasible_combinations == 0 and params.historical_conflict_relaxation_enabled:
        relaxed_historical_keys = _historical_relaxed_conflict_keys_for_cluster(
            cluster=solve_input.cluster,
            leaf_options=solve_input.leaf_options,
            ctx=solve_input.ctx,
            tree_store=tree_store,
            params=params,
        )
        if relaxed_historical_keys:
            historical_relaxation_attempted = True
            relaxed_problem = _build_cluster_solver_problem(
                cluster=solve_input.cluster,
                leaf_options=solve_input.leaf_options,
                ctx=solve_input.ctx,
                cluster_universe=solve_input.cluster_universe,
                unused_score_context=solve_input.unused_score_context,
                relaxed_conflict_keys=frozenset(relaxed_historical_keys),
                params=params,
                scoring_model=scoring_model,
            )
            kept_globals, relaxed_diagnostics = _solve_cluster_exact(
                prepared_problem=relaxed_problem,
                cluster_solver=cluster_solver,
            )
            combinations_evaluated += int(relaxed_diagnostics.combinations_evaluated)
            feasible_combinations = int(relaxed_diagnostics.feasible_combinations)
            historical_relaxation_succeeded = feasible_combinations > 0
            _log_historical_relaxation(
                cluster=solve_input.cluster,
                ctx=solve_input.ctx,
                relaxed_keys=relaxed_historical_keys,
                feasible_before=0,
                feasible_after=feasible_combinations,
            )

    if feasible_combinations == 0:
        _raise_cluster_infeasible_error(
            solve_input=solve_input,
            relaxed_historical_keys=relaxed_historical_keys,
        )

    return _ClusterSolveOutcome(
        kept_globals=kept_globals,
        combinations_evaluated=combinations_evaluated,
        feasible_combinations=feasible_combinations,
        historical_relaxation_attempted=historical_relaxation_attempted,
        historical_relaxation_succeeded=historical_relaxation_succeeded,
        historical_relaxed_keys=frozenset(relaxed_historical_keys),
    )


def _solve_cluster(
    *,
    solve_input: _ClusterSolveInput,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    scoring_model: ScoringModel,
    cluster_solver: ClusterSolver,
) -> _ClusterSolveOutcome:
    """Tracker-side policy wrapper around exact cluster-solver calls."""
    return _solve_with_optional_historical_relaxation(
        solve_input=solve_input,
        tree_store=tree_store,
        params=params,
        scoring_model=scoring_model,
        cluster_solver=cluster_solver,
    )


def _rebuild_one_cluster(
    *,
    cluster: ClusterWorkItem,
    ctx: ScanContext,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    scoring_model: ScoringModel,
    cluster_solver: ClusterSolver,
) -> _ClusterRebuildResult:
    """Solve one exact cluster problem and map solver results to snapshots."""
    leaf_options = cluster_leaf_options(
        track_ids=cluster.track_ids,
        tree_store=tree_store,
    )
    projected_combinations = _projected_combination_count(leaf_options)
    projected_cap = params.max_projected_cluster_combinations
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
    unused_score_context = _build_cluster_unused_score_context(
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
    solve_outcome = _solve_cluster(
        solve_input=solve_input,
        tree_store=tree_store,
        params=params,
        scoring_model=scoring_model,
        cluster_solver=cluster_solver,
    )
    map_global = solve_outcome.kept_globals[0] if solve_outcome.kept_globals else None

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
        historical_relaxation_attempted=solve_outcome.historical_relaxation_attempted,
        historical_relaxation_succeeded=solve_outcome.historical_relaxation_succeeded,
        historical_relaxed_key_count=len(solve_outcome.historical_relaxed_keys),
    )


def _log_overload_split_summary(
    *,
    scan_index: int,
    summary: OverloadSplitSummary,
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


def rebuild_cluster_globals(
    *,
    clusters: list[ClusterWorkItem],
    ctx: ScanContext,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    scoring_model: ScoringModel,
    cluster_solver: ClusterSolver,
) -> tuple[list[ClusterRebuildSnapshot], RebuildStats]:
    """Rebuild all clusters with explicit pre-solve overload-split policy."""
    if not clusters:
        return [], RebuildStats()

    leaf_count_by_track_id = {
        track_id: len(tree.active_leaf_node_ids)
        for track_id, tree in tree_store.track_trees_by_track_id.items()
    }
    clusters_for_rebuild_raw, split_summaries = apply_overload_splitting_to_clusters(
        clusters=clusters,
        leaf_count_by_track_id=leaf_count_by_track_id,
        params=params,
    )
    for split_summary in split_summaries:
        _log_overload_split_summary(
            scan_index=ctx.scan_index,
            summary=split_summary,
        )

    clusters_for_rebuild = [
        replace(cluster, cluster_id=cluster_id)
        for cluster_id, cluster in enumerate(clusters_for_rebuild_raw)
    ]
    rebuild_results = [
        _rebuild_one_cluster(
            cluster=cluster,
            ctx=ctx,
            tree_store=tree_store,
            params=params,
            scoring_model=scoring_model,
            cluster_solver=cluster_solver,
        )
        for cluster in clusters_for_rebuild
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
