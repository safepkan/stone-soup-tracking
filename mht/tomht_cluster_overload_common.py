"""Shared mechanics for TO-MHT cluster overload solving.

This module contains data containers and solver plumbing used by the public
cluster overload entry point plus the recursive overload strategies. It is
internal in spirit; policy choices live in ``tomht_cluster_split_policy`` and
strategy-specific recursion lives in the greedy/conditional modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

from .tomht_clustering import (
    ClusterWorkItem,
    OverloadSplitRemovedEdge,
    OverloadSplitSummary,
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
from .tomht_cluster_solver_search import has_any_feasible_solver_combination
from .tomht_model import DetectionKey, GlobalHypothesis, ScanContext
from .tomht_model import TrackHypothesisNode
from .tomht_params import TOMHTParams
from .tomht_tree_store import TrackTreeStore
from .tomht_tree_utils import live_conflict_keys_for_leaf


@dataclass(frozen=True)
class ClusterSolveInput:
    """Tracker-side prepared cluster-solve inputs before policy wrappers."""

    cluster: ClusterWorkItem
    ctx: ScanContext
    leaf_options: list[list[TrackHypothesisNode]]
    cluster_universe: set[DetectionKey]


@dataclass(frozen=True)
class ClusterSolveOutcome:
    """Solved original-cluster globals plus aggregate diagnostics."""

    kept_globals: tuple[GlobalHypothesis, ...]
    combinations_evaluated: int
    feasible_combinations: int
    overload_split_summary: OverloadSplitSummary | None = None


@dataclass(frozen=True)
class _PreparedClusterSolveProblem:
    """One solver-facing cluster problem plus leaf-ID mapping back to nodes."""

    problem: ClusterSolverProblem
    leaf_node_by_leaf_id: dict[int, TrackHypothesisNode]


@dataclass(frozen=True)
class _BinaryClusterSplit:
    """One internal binary decomposition of an overloaded cluster."""

    left_cluster: ClusterWorkItem
    right_cluster: ClusterWorkItem
    cut_keys: frozenset[DetectionKey]
    removed_edges: tuple[OverloadSplitRemovedEdge, ...]


_RecursiveCacheKey: TypeAlias = tuple[tuple[int, ...], frozenset[DetectionKey]]


_RecursiveSolveCache: TypeAlias = dict[_RecursiveCacheKey, tuple[GlobalHypothesis, ...]]


@dataclass
class _RecursiveSolveAccumulator:
    """Mutable diagnostics accumulated across one original-cluster solve."""

    original_projected_combinations: int
    projected_threshold: int
    removed_edges: list[OverloadSplitRemovedEdge] = field(default_factory=list)
    removed_edge_keys: set[tuple[int, int, int]] = field(default_factory=set)
    final_exact_track_ids: set[tuple[int, ...]] = field(default_factory=set)
    exact_combinations_evaluated: int = 0
    exact_feasible_combinations: int = 0
    split_operations: int = 0
    conditional_branches_attempted: int = 0
    conditional_branches_without_solution: int = 0
    recombination_candidates_considered: int = 0
    infeasible_recombination_candidates_skipped: int = 0
    branch_recombined_globals_retained: int = 0
    recombination_failures: int = 0
    interface_assignment_cap_fallbacks: int = 0
    recursive_cache_hits: int = 0
    recursive_cache_misses: int = 0
    max_recursion_depth: int = 0
    max_cut_key_count: int = 0
    total_interface_assignments: int = 0
    max_recombination_product_size: int = 0
    final_recombined_globals_retained: int = 0
    greedy_partition_splits: int = 0
    greedy_partition_fallbacks: int = 0
    greedy_cut_keys_assigned_left: int = 0
    greedy_cut_keys_assigned_right: int = 0
    greedy_cut_keys_assigned_neither: int = 0
    greedy_cut_keys_released: int = 0


def projected_combination_count(
    leaf_options: list[list[TrackHypothesisNode]],
) -> int:
    """Return projected Cartesian product size for one leaf-option set."""
    projected = 1
    for leaves in leaf_options:
        projected *= len(leaves)
    return projected


def has_any_feasible_cluster_combination(
    *,
    cluster: ClusterWorkItem,
    leaf_options: list[list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
) -> bool:
    """Return whether at least one cluster leaf-product combination is feasible."""
    track_options: list[ClusterSolverTrackOptions] = []
    for idx, track_id in enumerate(cluster.track_ids):
        tree = tree_store.track_trees_by_track_id[track_id]
        solver_leaf_options = tuple(
            ClusterSolverLeafOption(
                leaf_id=int(leaf.node_id),
                track_id=int(track_id),
                score=float(leaf.accumulated_log_score),
                full_history_conflict_keys=frozenset(
                    live_conflict_keys_for_leaf(leaf=leaf, tree=tree)
                ),
            )
            for leaf in leaf_options[idx]
        )
        if not solver_leaf_options:
            return False
        track_options.append(
            ClusterSolverTrackOptions(
                track_id=int(track_id),
                leaf_options=solver_leaf_options,
            )
        )

    return has_any_feasible_solver_combination(
        ClusterSolverProblem(track_options=tuple(track_options), max_results=1)
    )


def is_global_feasible_under_live_conflicts(
    *,
    global_hypothesis: GlobalHypothesis,
    tree_store: TrackTreeStore,
) -> bool:
    """Return whether one selected-leaf global satisfies live-key exclusivity."""
    used_keys: set[DetectionKey] = set()
    for track_id, leaf in sorted(global_hypothesis.leaf_nodes_by_track_id.items()):
        tree = tree_store.track_trees_by_track_id[int(track_id)]
        leaf_keys = set(live_conflict_keys_for_leaf(leaf=leaf, tree=tree))
        if used_keys & leaf_keys:
            return False
        used_keys |= leaf_keys
    return True


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

    # Pairwise overlap counts on live conflict keys indicate how hard the
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
                if key.scan_index == ctx.scan_index and key in cluster_universe
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


def _global_selection_key(
    global_hypothesis: GlobalHypothesis,
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic identity key for a selected-leaf combination."""
    return tuple(
        (int(track_id), int(leaf.node_id))
        for track_id, leaf in sorted(global_hypothesis.leaf_nodes_by_track_id.items())
    )


def _global_sort_key(
    global_hypothesis: GlobalHypothesis,
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Sort globals by descending score, then stable selected-leaf identity."""
    return (
        -float(global_hypothesis.log_weight),
        _global_selection_key(global_hypothesis),
    )


def _sort_dedupe_and_cap_globals(
    globals_in: list[GlobalHypothesis],
    *,
    max_results: int,
) -> tuple[GlobalHypothesis, ...]:
    """Keep one global per selected-leaf combination in deterministic top-K order."""
    best_by_selection: dict[tuple[tuple[int, int], ...], GlobalHypothesis] = {}
    for global_hypothesis in globals_in:
        selection_key = _global_selection_key(global_hypothesis)
        previous = best_by_selection.get(selection_key)
        if previous is None or _global_sort_key(global_hypothesis) < _global_sort_key(
            previous
        ):
            best_by_selection[selection_key] = global_hypothesis

    return tuple(
        sorted(best_by_selection.values(), key=_global_sort_key)[: int(max_results)]
    )


def _filter_leaf_options_by_forbidden_keys(
    *,
    cluster: ClusterWorkItem,
    leaf_options_by_track_id: Mapping[int, list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
    forbidden_keys: frozenset[DetectionKey],
) -> list[list[TrackHypothesisNode]] | None:
    """Filter each track's leaves to options that do not claim forbidden keys."""
    filtered_options: list[list[TrackHypothesisNode]] = []
    for track_id in cluster.track_ids:
        tree = tree_store.track_trees_by_track_id[int(track_id)]
        filtered_leaves = [
            leaf
            for leaf in leaf_options_by_track_id[int(track_id)]
            if not (
                set(live_conflict_keys_for_leaf(leaf=leaf, tree=tree))
                & set(forbidden_keys)
            )
        ]
        if not filtered_leaves:
            return None
        filtered_options.append(filtered_leaves)
    return filtered_options


def _conflict_links_from_leaf_options(
    *,
    cluster: ClusterWorkItem,
    leaf_options: list[list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
) -> tuple[tuple[int, int, tuple[DetectionKey, ...]], ...]:
    """Rebuild live conflict links from a filtered leaf-option set."""
    live_keys_by_track_id: dict[int, set[DetectionKey]] = {}
    for idx, track_id in enumerate(cluster.track_ids):
        tree = tree_store.track_trees_by_track_id[int(track_id)]
        keys: set[DetectionKey] = set()
        for leaf in leaf_options[idx]:
            keys |= set(live_conflict_keys_for_leaf(leaf=leaf, tree=tree))
        live_keys_by_track_id[int(track_id)] = keys

    conflict_links: list[tuple[int, int, tuple[DetectionKey, ...]]] = []
    for i, left_track_id in enumerate(cluster.track_ids):
        for right_track_id in cluster.track_ids[i + 1 :]:
            shared = (
                live_keys_by_track_id[int(left_track_id)]
                & live_keys_by_track_id[int(right_track_id)]
            )
            if shared:
                conflict_links.append(
                    (int(left_track_id), int(right_track_id), tuple(sorted(shared)))
                )
    return tuple(conflict_links)


def _recombine_global_solutions(
    *,
    left_solutions: tuple[GlobalHypothesis, ...],
    right_solutions: tuple[GlobalHypothesis, ...],
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    accumulator: _RecursiveSolveAccumulator,
) -> tuple[GlobalHypothesis, ...]:
    """Recombine subcluster solutions into feasible original-cluster globals."""
    product_size = len(left_solutions) * len(right_solutions)
    accumulator.max_recombination_product_size = max(
        accumulator.max_recombination_product_size,
        product_size,
    )
    recombined: list[GlobalHypothesis] = []
    for left_solution in left_solutions:
        for right_solution in right_solutions:
            accumulator.recombination_candidates_considered += 1
            leaf_nodes_by_track_id = dict(left_solution.leaf_nodes_by_track_id)
            overlapping_track_ids = set(leaf_nodes_by_track_id) & set(
                right_solution.leaf_nodes_by_track_id
            )
            if overlapping_track_ids:
                raise RuntimeError(
                    "Recursive cluster recombination received overlapping tracks: "
                    f"{sorted(overlapping_track_ids)}."
                )
            leaf_nodes_by_track_id.update(right_solution.leaf_nodes_by_track_id)
            candidate = GlobalHypothesis(
                leaf_nodes_by_track_id=leaf_nodes_by_track_id,
                log_weight=(
                    float(left_solution.log_weight) + float(right_solution.log_weight)
                ),
            )
            if not is_global_feasible_under_live_conflicts(
                global_hypothesis=candidate,
                tree_store=tree_store,
            ):
                accumulator.infeasible_recombination_candidates_skipped += 1
                continue
            recombined.append(candidate)

    capped = _sort_dedupe_and_cap_globals(
        recombined,
        max_results=int(params.max_global_hypotheses),
    )
    accumulator.branch_recombined_globals_retained += len(capped)
    if not capped and left_solutions and right_solutions:
        accumulator.recombination_failures += 1
    return capped


def _raise_cluster_infeasible_error(
    *,
    solve_input: ClusterSolveInput,
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


def _solve_cluster_exact_leaf_problem(
    *,
    cluster: ClusterWorkItem,
    ctx: ScanContext,
    filtered_leaf_options: list[list[TrackHypothesisNode]],
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...],
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    cluster_solver: ClusterSolver,
    accumulator: _RecursiveSolveAccumulator,
    recursive_cache: _RecursiveSolveCache,
    cache_key: _RecursiveCacheKey,
    allow_empty: bool,
) -> tuple[GlobalHypothesis, ...]:
    """Solve a filtered leaf-product problem through the exact solver backend."""
    projected_cap = params.max_projected_cluster_combinations
    projected_combinations = projected_combination_count(filtered_leaf_options)
    if projected_cap is not None and projected_combinations > int(projected_cap):
        raise RuntimeError(
            "Cluster rebuild projected Cartesian combinations exceed guardrail: "
            f"cluster={cluster.cluster_id} "
            f"projected={projected_combinations} "
            f"cap={int(projected_cap)}"
        )

    exact_cluster = ClusterWorkItem(
        cluster_id=cluster.cluster_id,
        track_ids=cluster.track_ids,
        current_scan_det_keys_by_track_id=cluster.current_scan_det_keys_by_track_id,
        conflict_links=conflict_links,
        overload_split_origin_cluster_id=None,
    )
    cluster_universe: set[DetectionKey] = set()
    for keys in exact_cluster.current_scan_det_keys_by_track_id.values():
        cluster_universe |= keys
    solve_input = ClusterSolveInput(
        cluster=exact_cluster,
        ctx=ctx,
        leaf_options=filtered_leaf_options,
        cluster_universe=cluster_universe,
    )
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
    accumulator.exact_combinations_evaluated += combinations_evaluated
    accumulator.exact_feasible_combinations += feasible_combinations
    accumulator.final_exact_track_ids.add(tuple(exact_cluster.track_ids))

    if feasible_combinations == 0:
        if allow_empty:
            recursive_cache[cache_key] = ()
            return ()
        _raise_cluster_infeasible_error(
            solve_input=solve_input,
            tree_store=tree_store,
        )

    recursive_cache[cache_key] = kept_globals
    return kept_globals
