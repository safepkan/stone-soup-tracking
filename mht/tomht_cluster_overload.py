"""Internal overload split solving for TO-MHT cluster rebuilds.

Both overload modes are internal strategies for solving one original live
cluster. They may decompose that cluster recursively, but downstream still sees
only feasible globals for the original cluster:

- ``greedy_partition`` is the default operational fallback. It greedily assigns
  contested cut keys to one side, solves one deterministic partition, verifies
  recombined globals under the original live conflicts, and falls back to
  ``conditional_exact`` if the greedy partition cannot produce feasible parent
  globals. It preserves feasibility, not full original-cluster K-best optimality.
- ``conditional_exact`` is the reference / higher-compute recursive conditional
  mode. It enumerates cut-key forbiddance assignments and is K-best-oriented
  across the cut interface, but can be substantially more expensive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Mapping, TypeAlias

from .tomht_clustering import (
    ClusterWorkItem,
    OverloadSplitRemovedEdge,
    OverloadSplitSummary,
    canonical_edge_pair,
    connected_components_from_pairs,
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
from .tomht_model import DetectionKey, GlobalHypothesis, ScanContext
from .tomht_model import TrackHypothesisNode
from .tomht_params import TOMHTParams
from .tomht_tree_store import TrackTreeStore
from .tomht_tree_utils import live_conflict_keys_for_leaf

_MAX_EXHAUSTIVE_INTERFACE_ASSIGNMENT_KEYS = 8


# Public solve inputs / outputs and private recursion state.


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


@dataclass(frozen=True)
class _GreedyCutKeyOwnership:
    """Greedy side ownership for one split's cut interface."""

    left_assigned: frozenset[DetectionKey]
    right_assigned: frozenset[DetectionKey]
    neither_assigned: frozenset[DetectionKey]


# Generic live-conflict feasibility helpers.


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


# Exact solver bridge for one filtered leaf-product problem.


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


# Binary split selection and cut-interface ownership.


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


def _subcluster_from_track_ids(
    *,
    cluster: ClusterWorkItem,
    track_ids: tuple[int, ...],
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...],
) -> ClusterWorkItem:
    """Build an internal subcluster for recursive solving."""
    track_id_set = set(track_ids)
    return ClusterWorkItem(
        cluster_id=cluster.cluster_id,
        track_ids=track_ids,
        current_scan_det_keys_by_track_id={
            track_id: set(cluster.current_scan_det_keys_by_track_id[track_id])
            for track_id in track_ids
        },
        conflict_links=tuple(
            link
            for link in conflict_links
            if link[0] in track_id_set and link[1] in track_id_set
        ),
        overload_split_origin_cluster_id=None,
    )


def _partition_binary_subclusters(
    *,
    cluster: ClusterWorkItem,
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...],
    left_track_ids: tuple[int, ...],
    right_track_ids: tuple[int, ...],
    removed_edges: tuple[OverloadSplitRemovedEdge, ...],
) -> _BinaryClusterSplit:
    """Create left/right subclusters and collect all cross-cut live keys."""
    left_track_set = set(left_track_ids)
    right_track_set = set(right_track_ids)
    cut_keys: set[DetectionKey] = set()
    for left_track_id, right_track_id, shared_keys in conflict_links:
        if (left_track_id in left_track_set and right_track_id in right_track_set) or (
            left_track_id in right_track_set and right_track_id in left_track_set
        ):
            cut_keys |= set(shared_keys)

    return _BinaryClusterSplit(
        left_cluster=_subcluster_from_track_ids(
            cluster=cluster,
            track_ids=left_track_ids,
            conflict_links=conflict_links,
        ),
        right_cluster=_subcluster_from_track_ids(
            cluster=cluster,
            track_ids=right_track_ids,
            conflict_links=conflict_links,
        ),
        cut_keys=frozenset(cut_keys),
        removed_edges=removed_edges,
    )


def _choose_binary_overload_split(
    *,
    cluster: ClusterWorkItem,
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...],
    accumulator: _RecursiveSolveAccumulator,
    params: TOMHTParams,
) -> _BinaryClusterSplit | None:
    """Choose one deterministic binary split for an overloaded solve branch."""
    if len(cluster.track_ids) <= 1:
        return None

    remaining_edge_keys_by_pair: dict[tuple[int, int], tuple[DetectionKey, ...]] = {
        canonical_edge_pair(left_track_id, right_track_id): tuple(shared_keys)
        for left_track_id, right_track_id, shared_keys in conflict_links
    }
    components = connected_components_from_pairs(
        cluster.track_ids,
        remaining_edge_keys_by_pair.keys(),
    )
    if len(components) > 1:
        left_track_ids = components[0]
        right_track_ids = tuple(
            track_id for component in components[1:] for track_id in component
        )
        return _partition_binary_subclusters(
            cluster=cluster,
            conflict_links=conflict_links,
            left_track_ids=left_track_ids,
            right_track_ids=tuple(sorted(right_track_ids)),
            removed_edges=(),
        )

    if not remaining_edge_keys_by_pair:
        split_at = max(1, len(cluster.track_ids) // 2)
        return _partition_binary_subclusters(
            cluster=cluster,
            conflict_links=conflict_links,
            left_track_ids=tuple(cluster.track_ids[:split_at]),
            right_track_ids=tuple(cluster.track_ids[split_at:]),
            removed_edges=(),
        )

    max_removals = params.overload_split_max_edge_removals_per_cluster
    removed_edges: list[OverloadSplitRemovedEdge] = []
    while remaining_edge_keys_by_pair:
        if max_removals is not None and len(accumulator.removed_edges) + len(
            removed_edges
        ) >= int(max_removals):
            return None

        left_track_id, right_track_id = min(
            remaining_edge_keys_by_pair,
            key=lambda pair: (
                len(remaining_edge_keys_by_pair[pair]),
                int(pair[0]),
                int(pair[1]),
            ),
        )
        shared_keys = remaining_edge_keys_by_pair.pop((left_track_id, right_track_id))
        removed_edges.append(
            OverloadSplitRemovedEdge(
                left_track_id=left_track_id,
                right_track_id=right_track_id,
                shared_live_key_count=len(shared_keys),
            )
        )

        components = connected_components_from_pairs(
            cluster.track_ids,
            remaining_edge_keys_by_pair.keys(),
        )
        if len(components) <= 1:
            continue

        left_track_ids = components[0]
        right_track_ids = tuple(
            track_id for component in components[1:] for track_id in component
        )
        return _partition_binary_subclusters(
            cluster=cluster,
            conflict_links=conflict_links,
            left_track_ids=left_track_ids,
            right_track_ids=tuple(sorted(right_track_ids)),
            removed_edges=tuple(removed_edges),
        )

    return None


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


# Recombination, diagnostics, and exact leaf-problem fallback.


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


def _record_removed_edges_once(
    *,
    accumulator: _RecursiveSolveAccumulator,
    removed_edges: tuple[OverloadSplitRemovedEdge, ...],
) -> None:
    """Record unique removed-edge diagnostics across recursive branches."""
    for edge in removed_edges:
        edge_key = (
            int(edge.left_track_id),
            int(edge.right_track_id),
            int(edge.shared_live_key_count),
        )
        if edge_key in accumulator.removed_edge_keys:
            continue
        accumulator.removed_edge_keys.add(edge_key)
        accumulator.removed_edges.append(edge)


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


# Recursive overload solution modes.


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


def _should_try_overload_split(
    *,
    params: TOMHTParams,
    projected_combinations: int,
) -> bool:
    """Return whether the projected solve size exceeds the overload threshold."""
    threshold = params.overload_split_projected_combination_threshold
    if not params.overload_split_enabled or threshold is None:
        return False
    threshold_int = int(threshold)
    if threshold_int <= 0:
        raise ValueError(
            "overload_split_projected_combination_threshold must be positive "
            "when overload splitting is enabled."
        )
    return int(projected_combinations) > threshold_int


# Public overload solve entry point and log emission.


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
    accumulator = _RecursiveSolveAccumulator(
        original_projected_combinations=int(projected_combinations),
        projected_threshold=threshold_for_summary,
    )
    solution_mode = params.overload_split_solution_mode
    if solution_mode == "conditional_exact":
        recursive_cache: _RecursiveSolveCache = {}
        kept_globals = _solve_cluster_conditional_exact_recursive(
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
        greedy_recursive_cache: _RecursiveSolveCache = {}
        exact_recursive_cache: _RecursiveSolveCache = {}
        kept_globals = _solve_cluster_greedy_partition_recursive(
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
        _raise_cluster_infeasible_error(
            solve_input=solve_input,
            tree_store=tree_store,
        )

    for rebuilt_global in kept_globals:
        if not is_global_feasible_under_live_conflicts(
            global_hypothesis=rebuilt_global,
            tree_store=tree_store,
        ):
            raise RuntimeError(
                "Cluster rebuild emitted an infeasible rebuilt global under live "
                f"conflicts. cluster={solve_input.cluster.cluster_id} "
                f"selection={_global_selection_key(rebuilt_global)}"
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
    accumulator: _RecursiveSolveAccumulator,
) -> OverloadSplitSummary | None:
    """Build overload diagnostics only when recursion actually split."""
    if accumulator.split_operations <= 0:
        return None

    final_components = tuple(sorted(accumulator.final_exact_track_ids))
    projected_after_by_subcluster = tuple(
        projected_combination_count(
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
