"""Public overload-aware cluster solving entry point.

The main reader path is intentionally small: prepare solve diagnostics, choose
``greedy_partition`` or ``conditional_exact``, verify original-cluster
feasibility, build overload diagnostics, and return a ``ClusterSolveOutcome``.
Implementation details live in the common, policy, greedy, and conditional
modules.
"""

from __future__ import annotations

from typing import Mapping

from .tomht_clustering import OverloadSplitSummary
from . import tomht_cluster_overload_common as _common
from . import tomht_cluster_overload_conditional as _conditional
from . import tomht_cluster_overload_greedy as _greedy
from .tomht_cluster_overload_common import (
    ClusterSolveInput,
    ClusterSolveOutcome,
)
from .tomht_cluster_solver import ClusterSolver
from .tomht_model import TrackHypothesisNode
from .tomht_params import TOMHTParams
from .tomht_tree_store import TrackTreeStore

__all__ = [
    "ClusterSolveInput",
    "ClusterSolveOutcome",
    "log_overload_split_summary",
    "solve_cluster_globals",
]


def solve_cluster_globals(
    *,
    solve_input: ClusterSolveInput,
    leaf_options_by_track_id: Mapping[int, list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    cluster_solver: ClusterSolver,
    projected_combinations: int,
) -> ClusterSolveOutcome:
    """Solve one original cluster while keeping overload splits internal."""
    threshold = params.overload_split_projected_combination_threshold
    threshold_for_summary = 0 if threshold is None else int(threshold)
    accumulator = _common._RecursiveSolveAccumulator(
        original_projected_combinations=int(projected_combinations),
        projected_threshold=threshold_for_summary,
    )
    solution_mode = params.overload_split_solution_mode
    if solution_mode == "conditional_exact":
        recursive_cache: _common._RecursiveSolveCache = {}
        kept_globals = _conditional._solve_cluster_conditional_exact_recursive(
            cluster=solve_input.cluster,
            ctx=solve_input.ctx,
            leaf_options_by_track_id=leaf_options_by_track_id,
            tree_store=tree_store,
            params=params,
            cluster_solver=cluster_solver,
            inherited_forbidden_keys=frozenset(),
            accumulator=accumulator,
            recursive_cache=recursive_cache,
            recursion_depth=0,
            allow_empty=False,
        )
    elif solution_mode == "greedy_partition":
        greedy_recursive_cache: _common._RecursiveSolveCache = {}
        exact_recursive_cache: _common._RecursiveSolveCache = {}
        kept_globals = _greedy._solve_cluster_greedy_partition_recursive(
            cluster=solve_input.cluster,
            ctx=solve_input.ctx,
            leaf_options_by_track_id=leaf_options_by_track_id,
            tree_store=tree_store,
            params=params,
            cluster_solver=cluster_solver,
            inherited_forbidden_keys=frozenset(),
            accumulator=accumulator,
            greedy_recursive_cache=greedy_recursive_cache,
            exact_recursive_cache=exact_recursive_cache,
            recursion_depth=0,
            allow_empty=False,
        )
    else:
        raise ValueError(f"Unknown overload split solution mode: {solution_mode!r}.")
    if not kept_globals:
        _common._raise_cluster_infeasible_error(
            solve_input=solve_input,
            tree_store=tree_store,
        )

    for rebuilt_global in kept_globals:
        if not _common.is_global_feasible_under_live_conflicts(
            global_hypothesis=rebuilt_global,
            tree_store=tree_store,
        ):
            raise RuntimeError(
                "Cluster rebuild emitted an infeasible rebuilt global under live "
                f"conflicts. cluster={solve_input.cluster.cluster_id} "
                f"selection={_common._global_selection_key(rebuilt_global)}"
            )

    accumulator.final_recombined_globals_retained = len(kept_globals)
    overload_split_summary = _build_overload_split_summary(
        solve_input=solve_input,
        leaf_options_by_track_id=leaf_options_by_track_id,
        solution_mode=solution_mode,
        accumulator=accumulator,
    )

    return ClusterSolveOutcome(
        kept_globals=kept_globals,
        combinations_evaluated=(
            accumulator.exact_combinations_evaluated
            + accumulator.recombination_candidates_considered
        ),
        feasible_combinations=(
            accumulator.exact_feasible_combinations
            + accumulator.branch_recombined_globals_retained
        ),
        overload_split_summary=overload_split_summary,
    )


def _build_overload_split_summary(
    *,
    solve_input: ClusterSolveInput,
    leaf_options_by_track_id: Mapping[int, list[TrackHypothesisNode]],
    solution_mode: str,
    accumulator: _common._RecursiveSolveAccumulator,
) -> OverloadSplitSummary | None:
    """Build overload diagnostics only when recursion actually split."""
    if accumulator.split_operations <= 0:
        return None

    final_components = tuple(sorted(accumulator.final_exact_track_ids))
    projected_after_by_subcluster = tuple(
        _common.projected_combination_count(
            [leaf_options_by_track_id[track_id] for track_id in component]
        )
        for component in final_components
    )
    return OverloadSplitSummary(
        original_cluster_id=solve_input.cluster.cluster_id,
        original_track_ids=solve_input.cluster.track_ids,
        projected_before=accumulator.original_projected_combinations,
        projected_threshold=accumulator.projected_threshold,
        removed_edges=tuple(accumulator.removed_edges),
        resulting_subclusters=final_components,
        projected_after_by_subcluster=projected_after_by_subcluster,
        stopping_reason=(
            "recursive_conditioning"
            if solution_mode == "conditional_exact"
            else "greedy_partition"
        ),
        conditional_branches_attempted=(accumulator.conditional_branches_attempted),
        conditional_branches_without_solution=(
            accumulator.conditional_branches_without_solution
        ),
        recombination_candidates_considered=(
            accumulator.recombination_candidates_considered
        ),
        infeasible_recombination_candidates_skipped=(
            accumulator.infeasible_recombination_candidates_skipped
        ),
        branch_recombined_globals_retained=(
            accumulator.branch_recombined_globals_retained
        ),
        recombination_failures=accumulator.recombination_failures,
        interface_assignment_cap_fallbacks=(
            accumulator.interface_assignment_cap_fallbacks
        ),
        recursive_cache_hits=accumulator.recursive_cache_hits,
        recursive_cache_misses=accumulator.recursive_cache_misses,
        max_recursion_depth=accumulator.max_recursion_depth,
        max_cut_key_count=accumulator.max_cut_key_count,
        total_interface_assignments=accumulator.total_interface_assignments,
        max_recombination_product_size=accumulator.max_recombination_product_size,
        final_recombined_globals_retained=(
            accumulator.final_recombined_globals_retained
        ),
        greedy_partition_splits=accumulator.greedy_partition_splits,
        greedy_partition_fallbacks=accumulator.greedy_partition_fallbacks,
        greedy_cut_keys_assigned_left=accumulator.greedy_cut_keys_assigned_left,
        greedy_cut_keys_assigned_right=accumulator.greedy_cut_keys_assigned_right,
        greedy_cut_keys_assigned_neither=accumulator.greedy_cut_keys_assigned_neither,
        greedy_cut_keys_released=accumulator.greedy_cut_keys_released,
    )


def log_overload_split_summary(
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
    greedy_str = ""
    if summary.stopping_reason == "greedy_partition":
        greedy_str = (
            f" greedy_splits={summary.greedy_partition_splits}"
            f" greedy_fallbacks={summary.greedy_partition_fallbacks}"
            f" greedy_assign_l={summary.greedy_cut_keys_assigned_left}"
            f" greedy_assign_r={summary.greedy_cut_keys_assigned_right}"
            f" greedy_assign_none={summary.greedy_cut_keys_assigned_neither}"
            f" greedy_released={summary.greedy_cut_keys_released}"
        )
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
        f"projected_after={projected_after} "
        f"branches={summary.conditional_branches_attempted} "
        f"empty_branches={summary.conditional_branches_without_solution} "
        f"recomb_considered={summary.recombination_candidates_considered} "
        "recomb_infeasible="
        f"{summary.infeasible_recombination_candidates_skipped} "
        f"branch_recomb_retained={summary.branch_recombined_globals_retained} "
        f"recomb_failures={summary.recombination_failures} "
        f"recursive_cache_hits={summary.recursive_cache_hits} "
        f"recursive_cache_misses={summary.recursive_cache_misses} "
        f"max_recursion_depth={summary.max_recursion_depth} "
        f"max_cut_key_count={summary.max_cut_key_count} "
        f"total_interface_assignments={summary.total_interface_assignments} "
        "max_recombination_product_size="
        f"{summary.max_recombination_product_size} "
        "final_recomb_retained="
        f"{summary.final_recombined_globals_retained} "
        "interface_assignment_cap_fallbacks="
        f"{summary.interface_assignment_cap_fallbacks}"
        f"{greedy_str}"
    )
