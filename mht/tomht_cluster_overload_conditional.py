"""Reference recursive conditional-exact overload strategy.

``conditional_exact`` is the reference / higher-compute overload mode. It
enumerates cut-key forbiddance assignments, recursively solves each side, and
recombines feasible K-best-oriented solutions across the cut interface. This can
be expensive, so readers interested in the default operational path should start
with ``tomht_cluster_overload_greedy`` instead.
"""

from __future__ import annotations

from itertools import product
from typing import Mapping

from .tomht_clustering import ClusterWorkItem
from .tomht_cluster_overload_common import (
    _RecursiveCacheKey,
    _RecursiveSolveAccumulator,
    _RecursiveSolveCache,
    _conflict_links_from_leaf_options,
    _filter_leaf_options_by_forbidden_keys,
    _recombine_global_solutions,
    _solve_cluster_exact_leaf_problem,
    _sort_dedupe_and_cap_globals,
    projected_combination_count,
)
from .tomht_cluster_solver import ClusterSolver
from .tomht_cluster_split_policy import (
    _choose_binary_overload_split,
    _record_removed_edges_once,
    _should_try_overload_split,
)
from .tomht_model import DetectionKey, GlobalHypothesis, ScanContext
from .tomht_model import TrackHypothesisNode
from .tomht_params import TOMHTParams
from .tomht_tree_store import TrackTreeStore

_MAX_EXHAUSTIVE_INTERFACE_ASSIGNMENT_KEYS = 8


def _cut_key_assignments(
    *,
    cut_keys: frozenset[DetectionKey],
    accumulator: _RecursiveSolveAccumulator,
) -> tuple[tuple[frozenset[DetectionKey], frozenset[DetectionKey]], ...]:
    """Enumerate left/right forbidden-key assignments for one cut interface."""
    sorted_cut_keys = tuple(sorted(cut_keys))
    accumulator.max_cut_key_count = max(
        accumulator.max_cut_key_count,
        len(sorted_cut_keys),
    )
    if not sorted_cut_keys:
        accumulator.total_interface_assignments += 1
        return ((frozenset(), frozenset()),)

    if len(sorted_cut_keys) > _MAX_EXHAUSTIVE_INTERFACE_ASSIGNMENT_KEYS:
        accumulator.interface_assignment_cap_fallbacks += 1
        all_cut_keys = frozenset(sorted_cut_keys)
        # This fallback is intentionally incomplete but sound: it avoids the
        # exponential assignment explosion without allowing infeasible globals.
        # If it fires and recombination finds no solution, treat that as a
        # search-limit diagnostic rather than proof that no feasible joint
        # assignment exists.
        fallback_assignments: tuple[
            tuple[frozenset[DetectionKey], frozenset[DetectionKey]],
            ...,
        ] = (
            (frozenset(), all_cut_keys),
            (all_cut_keys, frozenset()),
            (all_cut_keys, all_cut_keys),
        )
        accumulator.total_interface_assignments += len(fallback_assignments)
        return fallback_assignments

    assignments: list[tuple[frozenset[DetectionKey], frozenset[DetectionKey]]] = []
    for choices in product((0, 1, 2), repeat=len(sorted_cut_keys)):
        left_forbidden: set[DetectionKey] = set()
        right_forbidden: set[DetectionKey] = set()
        for cut_key, choice in zip(sorted_cut_keys, choices):
            if choice == 0:
                right_forbidden.add(cut_key)
            elif choice == 1:
                left_forbidden.add(cut_key)
            else:
                left_forbidden.add(cut_key)
                right_forbidden.add(cut_key)
        assignments.append((frozenset(left_forbidden), frozenset(right_forbidden)))
    accumulator.total_interface_assignments += len(assignments)
    return tuple(assignments)


def _solve_cluster_conditional_exact_recursive(
    *,
    cluster: ClusterWorkItem,
    ctx: ScanContext,
    leaf_options_by_track_id: Mapping[int, list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    cluster_solver: ClusterSolver,
    inherited_forbidden_keys: frozenset[DetectionKey],
    accumulator: _RecursiveSolveAccumulator,
    recursive_cache: _RecursiveSolveCache,
    recursion_depth: int,
    allow_empty: bool,
) -> tuple[GlobalHypothesis, ...]:
    """Recursively enumerate cut assignments for the reference exact mode."""
    accumulator.max_recursion_depth = max(
        accumulator.max_recursion_depth,
        int(recursion_depth),
    )
    cache_key: _RecursiveCacheKey = (
        tuple(cluster.track_ids),
        inherited_forbidden_keys,
    )
    cached_result = recursive_cache.get(cache_key)
    if cached_result is not None:
        accumulator.recursive_cache_hits += 1
        return cached_result
    accumulator.recursive_cache_misses += 1

    filtered_leaf_options = _filter_leaf_options_by_forbidden_keys(
        cluster=cluster,
        leaf_options_by_track_id=leaf_options_by_track_id,
        tree_store=tree_store,
        forbidden_keys=inherited_forbidden_keys,
    )
    if filtered_leaf_options is None:
        recursive_cache[cache_key] = ()
        return ()

    projected_combinations = projected_combination_count(filtered_leaf_options)
    should_try_split = _should_try_overload_split(
        params=params,
        projected_combinations=projected_combinations,
    )
    conflict_links = _conflict_links_from_leaf_options(
        cluster=cluster,
        leaf_options=filtered_leaf_options,
        tree_store=tree_store,
    )

    if should_try_split:
        split = _choose_binary_overload_split(
            cluster=cluster,
            conflict_links=conflict_links,
            accumulator=accumulator,
            params=params,
        )
        if split is not None:
            accumulator.split_operations += 1
            _record_removed_edges_once(
                accumulator=accumulator,
                removed_edges=split.removed_edges,
            )
            branch_globals: list[GlobalHypothesis] = []
            for left_forbidden, right_forbidden in _cut_key_assignments(
                cut_keys=split.cut_keys,
                accumulator=accumulator,
            ):
                accumulator.conditional_branches_attempted += 1
                left_solutions = _solve_cluster_conditional_exact_recursive(
                    cluster=split.left_cluster,
                    ctx=ctx,
                    leaf_options_by_track_id=leaf_options_by_track_id,
                    tree_store=tree_store,
                    params=params,
                    cluster_solver=cluster_solver,
                    inherited_forbidden_keys=(
                        inherited_forbidden_keys | left_forbidden
                    ),
                    accumulator=accumulator,
                    recursive_cache=recursive_cache,
                    recursion_depth=recursion_depth + 1,
                    allow_empty=True,
                )
                if not left_solutions:
                    accumulator.conditional_branches_without_solution += 1
                    continue

                right_solutions = _solve_cluster_conditional_exact_recursive(
                    cluster=split.right_cluster,
                    ctx=ctx,
                    leaf_options_by_track_id=leaf_options_by_track_id,
                    tree_store=tree_store,
                    params=params,
                    cluster_solver=cluster_solver,
                    inherited_forbidden_keys=(
                        inherited_forbidden_keys | right_forbidden
                    ),
                    accumulator=accumulator,
                    recursive_cache=recursive_cache,
                    recursion_depth=recursion_depth + 1,
                    allow_empty=True,
                )
                if not right_solutions:
                    accumulator.conditional_branches_without_solution += 1
                    continue

                branch_globals.extend(
                    _recombine_global_solutions(
                        left_solutions=left_solutions,
                        right_solutions=right_solutions,
                        tree_store=tree_store,
                        params=params,
                        accumulator=accumulator,
                    )
                )

            result = _sort_dedupe_and_cap_globals(
                branch_globals,
                max_results=int(params.max_global_hypotheses),
            )
            recursive_cache[cache_key] = result
            return result

    result = _solve_cluster_exact_leaf_problem(
        cluster=cluster,
        ctx=ctx,
        filtered_leaf_options=filtered_leaf_options,
        conflict_links=conflict_links,
        tree_store=tree_store,
        params=params,
        cluster_solver=cluster_solver,
        accumulator=accumulator,
        recursive_cache=recursive_cache,
        cache_key=cache_key,
        allow_empty=allow_empty,
    )
    return result
