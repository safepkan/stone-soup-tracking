"""Default greedy-partition overload strategy.

``greedy_partition`` is the default operational overload fallback. It preserves
original-cluster feasibility, but it does not preserve full original-cluster
K-best optimality. For each split, it assigns cut keys by the best local
claiming-leaf score, solves the side with more assigned cut keys first, releases
assigned keys that no retained first-side global uses, solves the second side,
recombines the two sides, and verifies feasibility under the original live
conflicts. If a greedy branch cannot produce feasible parent globals, it falls
back to ``conditional_exact`` for that branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .tomht_clustering import ClusterWorkItem
from .tomht_cluster_overload_common import (
    _BinaryClusterSplit,
    _RecursiveCacheKey,
    _RecursiveSolveAccumulator,
    _RecursiveSolveCache,
    _conflict_links_from_leaf_options,
    _filter_leaf_options_by_forbidden_keys,
    _recombine_global_solutions,
    is_global_feasible_under_live_conflicts,
    projected_combination_count,
)
from .tomht_cluster_overload_conditional import (
    _solve_cluster_conditional_exact_recursive,
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
from .tomht_tree_utils import live_conflict_keys_for_leaf


@dataclass(frozen=True)
class _GreedyCutKeyOwnership:
    """Greedy side ownership for one split's cut interface."""

    left_assigned: frozenset[DetectionKey]
    right_assigned: frozenset[DetectionKey]
    neither_assigned: frozenset[DetectionKey]


def _best_claiming_leaf_score_for_cut_key(
    *,
    cut_key: DetectionKey,
    track_ids: tuple[int, ...],
    leaf_options_by_track_id: Mapping[int, list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
) -> float | None:
    """Return the best leaf score on one side that claims ``cut_key``."""
    best_score: float | None = None
    for track_id in track_ids:
        tree = tree_store.track_trees_by_track_id[int(track_id)]
        for leaf in leaf_options_by_track_id[int(track_id)]:
            if cut_key not in live_conflict_keys_for_leaf(leaf=leaf, tree=tree):
                continue
            score = float(leaf.accumulated_log_score)
            if best_score is None or score > best_score:
                best_score = score
    return best_score


def _assign_greedy_cut_key_ownership(
    *,
    split: _BinaryClusterSplit,
    leaf_options_by_track_id: Mapping[int, list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
    accumulator: _RecursiveSolveAccumulator,
) -> _GreedyCutKeyOwnership:
    """Assign each cut key to the side with the best local claiming leaf."""
    left_assigned: set[DetectionKey] = set()
    right_assigned: set[DetectionKey] = set()
    neither_assigned: set[DetectionKey] = set()

    for cut_key in sorted(split.cut_keys):
        left_score = _best_claiming_leaf_score_for_cut_key(
            cut_key=cut_key,
            track_ids=split.left_cluster.track_ids,
            leaf_options_by_track_id=leaf_options_by_track_id,
            tree_store=tree_store,
        )
        right_score = _best_claiming_leaf_score_for_cut_key(
            cut_key=cut_key,
            track_ids=split.right_cluster.track_ids,
            leaf_options_by_track_id=leaf_options_by_track_id,
            tree_store=tree_store,
        )

        if left_score is None and right_score is None:
            neither_assigned.add(cut_key)
        elif right_score is None:
            left_assigned.add(cut_key)
        elif left_score is None:
            right_assigned.add(cut_key)
        elif left_score > right_score:
            left_assigned.add(cut_key)
        elif right_score > left_score:
            right_assigned.add(cut_key)
        elif split.left_cluster.track_ids <= split.right_cluster.track_ids:
            left_assigned.add(cut_key)
        else:
            right_assigned.add(cut_key)

    accumulator.greedy_cut_keys_assigned_left += len(left_assigned)
    accumulator.greedy_cut_keys_assigned_right += len(right_assigned)
    accumulator.greedy_cut_keys_assigned_neither += len(neither_assigned)

    return _GreedyCutKeyOwnership(
        left_assigned=frozenset(left_assigned),
        right_assigned=frozenset(right_assigned),
        neither_assigned=frozenset(neither_assigned),
    )


def _claimed_live_keys_by_globals(
    *,
    globals_in: tuple[GlobalHypothesis, ...],
    tree_store: TrackTreeStore,
) -> set[DetectionKey]:
    """Return live conflict keys claimed by at least one retained global."""
    claimed: set[DetectionKey] = set()
    for global_hypothesis in globals_in:
        for track_id, leaf in global_hypothesis.leaf_nodes_by_track_id.items():
            tree = tree_store.track_trees_by_track_id[int(track_id)]
            claimed |= set(live_conflict_keys_for_leaf(leaf=leaf, tree=tree))
    return claimed


def _solve_cluster_greedy_partition_recursive(
    *,
    cluster: ClusterWorkItem,
    ctx: ScanContext,
    leaf_options_by_track_id: Mapping[int, list[TrackHypothesisNode]],
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    cluster_solver: ClusterSolver,
    inherited_forbidden_keys: frozenset[DetectionKey],
    accumulator: _RecursiveSolveAccumulator,
    greedy_recursive_cache: _RecursiveSolveCache,
    exact_recursive_cache: _RecursiveSolveCache,
    recursion_depth: int,
    allow_empty: bool,
) -> tuple[GlobalHypothesis, ...]:
    """Recursively solve overloads with greedy cut ownership and exact fallback."""
    accumulator.max_recursion_depth = max(
        accumulator.max_recursion_depth,
        int(recursion_depth),
    )
    cache_key: _RecursiveCacheKey = (
        tuple(cluster.track_ids),
        inherited_forbidden_keys,
    )
    cached_result = greedy_recursive_cache.get(cache_key)
    if cached_result is not None:
        accumulator.recursive_cache_hits += 1
        return cached_result
    accumulator.recursive_cache_misses += 1

    def _fall_back_to_conditional_exact() -> tuple[GlobalHypothesis, ...]:
        accumulator.greedy_partition_fallbacks += 1
        fallback_result = _solve_cluster_conditional_exact_recursive(
            cluster=cluster,
            ctx=ctx,
            leaf_options_by_track_id=leaf_options_by_track_id,
            tree_store=tree_store,
            params=params,
            cluster_solver=cluster_solver,
            inherited_forbidden_keys=inherited_forbidden_keys,
            accumulator=accumulator,
            recursive_cache=exact_recursive_cache,
            recursion_depth=recursion_depth,
            allow_empty=allow_empty,
        )
        greedy_recursive_cache[cache_key] = fallback_result
        return fallback_result

    filtered_leaf_options = _filter_leaf_options_by_forbidden_keys(
        cluster=cluster,
        leaf_options_by_track_id=leaf_options_by_track_id,
        tree_store=tree_store,
        forbidden_keys=inherited_forbidden_keys,
    )
    if filtered_leaf_options is None:
        greedy_recursive_cache[cache_key] = ()
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
            accumulator.greedy_partition_splits += 1
            accumulator.max_cut_key_count = max(
                accumulator.max_cut_key_count,
                len(split.cut_keys),
            )
            _record_removed_edges_once(
                accumulator=accumulator,
                removed_edges=split.removed_edges,
            )

            filtered_leaf_options_by_track_id = {
                track_id: filtered_leaf_options[idx]
                for idx, track_id in enumerate(cluster.track_ids)
            }
            ownership = _assign_greedy_cut_key_ownership(
                split=split,
                leaf_options_by_track_id=filtered_leaf_options_by_track_id,
                tree_store=tree_store,
                accumulator=accumulator,
            )
            left_assigned = ownership.left_assigned
            right_assigned = ownership.right_assigned
            first_is_left = len(left_assigned) >= len(right_assigned)
            if first_is_left:
                first_cluster = split.left_cluster
                second_cluster = split.right_cluster
                first_assigned = left_assigned
                second_assigned = right_assigned
            else:
                first_cluster = split.right_cluster
                second_cluster = split.left_cluster
                first_assigned = right_assigned
                second_assigned = left_assigned

            first_solutions = _solve_cluster_greedy_partition_recursive(
                cluster=first_cluster,
                ctx=ctx,
                leaf_options_by_track_id=leaf_options_by_track_id,
                tree_store=tree_store,
                params=params,
                cluster_solver=cluster_solver,
                inherited_forbidden_keys=(inherited_forbidden_keys | second_assigned),
                accumulator=accumulator,
                greedy_recursive_cache=greedy_recursive_cache,
                exact_recursive_cache=exact_recursive_cache,
                recursion_depth=recursion_depth + 1,
                allow_empty=True,
            )
            if not first_solutions:
                return _fall_back_to_conditional_exact()

            claimed_by_first = _claimed_live_keys_by_globals(
                globals_in=first_solutions,
                tree_store=tree_store,
            ) & set(first_assigned)
            accumulator.greedy_cut_keys_released += len(
                set(first_assigned) - claimed_by_first
            )
            second_solutions = _solve_cluster_greedy_partition_recursive(
                cluster=second_cluster,
                ctx=ctx,
                leaf_options_by_track_id=leaf_options_by_track_id,
                tree_store=tree_store,
                params=params,
                cluster_solver=cluster_solver,
                inherited_forbidden_keys=(
                    inherited_forbidden_keys | frozenset(claimed_by_first)
                ),
                accumulator=accumulator,
                greedy_recursive_cache=greedy_recursive_cache,
                exact_recursive_cache=exact_recursive_cache,
                recursion_depth=recursion_depth + 1,
                allow_empty=True,
            )
            if not second_solutions:
                return _fall_back_to_conditional_exact()

            if first_is_left:
                left_solutions = first_solutions
                right_solutions = second_solutions
            else:
                left_solutions = second_solutions
                right_solutions = first_solutions
            result = _recombine_global_solutions(
                left_solutions=left_solutions,
                right_solutions=right_solutions,
                tree_store=tree_store,
                params=params,
                accumulator=accumulator,
            )
            if not result:
                return _fall_back_to_conditional_exact()
            if any(
                not is_global_feasible_under_live_conflicts(
                    global_hypothesis=rebuilt_global,
                    tree_store=tree_store,
                )
                for rebuilt_global in result
            ):
                return _fall_back_to_conditional_exact()

            greedy_recursive_cache[cache_key] = result
            return result

    exact_result = _solve_cluster_conditional_exact_recursive(
        cluster=cluster,
        ctx=ctx,
        leaf_options_by_track_id=leaf_options_by_track_id,
        tree_store=tree_store,
        params=params,
        cluster_solver=cluster_solver,
        inherited_forbidden_keys=inherited_forbidden_keys,
        accumulator=accumulator,
        recursive_cache=exact_recursive_cache,
        recursion_depth=recursion_depth,
        allow_empty=allow_empty,
    )
    greedy_recursive_cache[cache_key] = exact_result
    return exact_result
