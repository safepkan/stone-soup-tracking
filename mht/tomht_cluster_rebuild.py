"""Cluster rebuild orchestration for the track-oriented MHT tracker."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Mapping

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
from .tomht_stats import RebuildStats
from .tomht_tree_store import TrackTreeStore
from .tomht_tree_utils import live_conflict_keys_for_leaf
from .tomht_types import ScanContext


@dataclass(frozen=True)
class _ClusterRebuildResult:
    """Cluster rebuild result for one solved cluster."""

    snapshot: ClusterRebuildSnapshot


@dataclass(frozen=True)
class _ClusterSolveInput:
    """Tracker-side prepared cluster-solve inputs before policy wrappers."""

    cluster: ClusterWorkItem
    ctx: ScanContext
    leaf_options: list[list[TrackHypothesisNode]]
    cluster_universe: set[DetectionKey]


@dataclass(frozen=True)
class _PreparedClusterSolveProblem:
    """One solver-facing cluster problem plus leaf-ID mapping back to nodes."""

    problem: ClusterSolverProblem
    leaf_node_by_leaf_id: dict[int, TrackHypothesisNode]


@dataclass(frozen=True)
class _ClusterSolveOutcome:
    """Solver outcome for one exact cluster solve."""

    kept_globals: tuple[GlobalHypothesis, ...]
    combinations_evaluated: int
    feasible_combinations: int


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


def has_any_feasible_cluster_combination(
    *,
    cluster: ClusterWorkItem,
    leaf_options: list[list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
) -> bool:
    """Return whether at least one cluster leaf-product combination is feasible."""
    prepared: list[list[tuple[TrackHypothesisNode, set[DetectionKey]]]] = []
    for idx, track_id in enumerate(cluster.track_ids):
        tree = tree_store.track_trees_by_track_id[track_id]
        prepared.append(
            [
                (leaf, set(live_conflict_keys_for_leaf(leaf=leaf, tree=tree)))
                for leaf in leaf_options[idx]
            ]
        )
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


def _build_cluster_solver_problem(
    *,
    cluster: ClusterWorkItem,
    leaf_options: list[list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
    ctx: ScanContext,
    cluster_universe: set[DetectionKey],
    params: TOMHTParams,
) -> _PreparedClusterSolveProblem:
    """Build one solver-facing exact cluster problem from tracker state."""
    leaf_node_by_leaf_id: dict[int, TrackHypothesisNode] = {}
    track_options: list[ClusterSolverTrackOptions] = []

    for idx, track_id in enumerate(cluster.track_ids):
        tree = tree_store.track_trees_by_track_id[track_id]
        solver_leaf_options: list[ClusterSolverLeafOption] = []
        for leaf in leaf_options[idx]:
            leaf_id = int(leaf.node_id)
            leaf_node_by_leaf_id[leaf_id] = leaf

            conflict_keys = set(live_conflict_keys_for_leaf(leaf=leaf, tree=tree))

            used_current_scan_keys = sorted(
                key
                for key in conflict_keys
                if key[0] == ctx.scan_index and key in cluster_universe
            )
            if len(used_current_scan_keys) > 1:
                raise RuntimeError(
                    "Cluster solver contract requires at most one current-scan "
                    "detection per leaf option. "
                    f"track_id={track_id} leaf_id={leaf_id} "
                    f"current_scan_keys={used_current_scan_keys}"
                )
            solver_leaf_options.append(
                ClusterSolverLeafOption(
                    leaf_id=leaf_id,
                    track_id=int(track_id),
                    score=float(leaf.accumulated_log_score),
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
    tree_store: TrackTreeStore,
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

    # Pairwise overlap counts on live conflict keys indicate how "hard" the
    # unresolved incompatibilities are between tree frontiers.
    pairwise_overlap_counts: list[str] = []
    for i, left_track_id in enumerate(cluster.track_ids):
        left_leaves = leaf_options[i]
        left_tree = tree_store.track_trees_by_track_id[left_track_id]
        for j, right_track_id in enumerate(cluster.track_ids[i + 1 :], start=i + 1):
            right_leaves = leaf_options[j]
            right_tree = tree_store.track_trees_by_track_id[right_track_id]
            conflicting_pairs = 0
            for left_leaf in left_leaves:
                left_keys = set(
                    live_conflict_keys_for_leaf(leaf=left_leaf, tree=left_tree)
                )
                for right_leaf in right_leaves:
                    right_keys = set(
                        live_conflict_keys_for_leaf(
                            leaf=right_leaf,
                            tree=right_tree,
                        )
                    )
                    if left_keys & right_keys:
                        conflicting_pairs += 1
            total_pairs = len(left_leaves) * len(right_leaves)
            pairwise_overlap_counts.append(
                f"{left_track_id}-{right_track_id}:{conflicting_pairs}/{total_pairs}"
            )
    if pairwise_overlap_counts:
        parts.append("pairwise_conflicts={" + ", ".join(pairwise_overlap_counts) + "}")

    return "; ".join(parts)


def _raise_cluster_infeasible_error(
    *,
    solve_input: _ClusterSolveInput,
    tree_store: TrackTreeStore,
) -> None:
    """Raise the existing cluster infeasibility error with live-key debug."""
    dbg = infeasible_cluster_debug_summary(
        cluster=solve_input.cluster,
        leaf_options=solve_input.leaf_options,
        tree_store=tree_store,
        ctx=solve_input.ctx,
    )
    raise RuntimeError(
        "Cluster rebuild found no feasible combination. "
        "Expected at least one feasible joint assignment. "
        f"{dbg}"
    )


def _solve_cluster(
    *,
    solve_input: _ClusterSolveInput,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    cluster_solver: ClusterSolver,
) -> _ClusterSolveOutcome:
    """Tracker-side policy wrapper around one exact cluster-solver call."""
    prepared_problem = _build_cluster_solver_problem(
        cluster=solve_input.cluster,
        leaf_options=solve_input.leaf_options,
        tree_store=tree_store,
        ctx=solve_input.ctx,
        cluster_universe=solve_input.cluster_universe,
        params=params,
    )
    kept_globals, solve_diagnostics = _solve_cluster_exact(
        prepared_problem=prepared_problem,
        cluster_solver=cluster_solver,
    )
    combinations_evaluated = int(solve_diagnostics.combinations_evaluated)
    feasible_combinations = int(solve_diagnostics.feasible_combinations)

    if feasible_combinations == 0:
        _raise_cluster_infeasible_error(
            solve_input=solve_input,
            tree_store=tree_store,
        )

    return _ClusterSolveOutcome(
        kept_globals=kept_globals,
        combinations_evaluated=combinations_evaluated,
        feasible_combinations=feasible_combinations,
    )


def _rebuild_one_cluster(
    *,
    cluster: ClusterWorkItem,
    ctx: ScanContext,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
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

    solve_input = _ClusterSolveInput(
        cluster=cluster,
        ctx=ctx,
        leaf_options=leaf_options,
        cluster_universe=cluster_universe,
    )
    solve_outcome = _solve_cluster(
        solve_input=solve_input,
        tree_store=tree_store,
        params=params,
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
        )
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
                f"{edge.shared_live_key_count}"
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
        ),
    )
