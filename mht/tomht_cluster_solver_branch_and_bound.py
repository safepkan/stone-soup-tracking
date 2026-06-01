"""Exact depth-first branch-and-bound cluster solver backend (the default)."""

from __future__ import annotations

from .tomht_cluster_solver import (
    ClusterSolverDiagnostics,
    ClusterSolverProblem,
    ClusterSolverResult,
    ClusterSolverSolution,
    TopKSolutionHeap,
    validate_cluster_solver_problem,
)
from .tomht_cluster_solver_search import (
    prepare_ordered_tracks_for_search,
    suffix_best_score_by_depth,
)


class BranchAndBoundClusterSolver:
    """Exact backend: deterministic depth-first branch-and-bound search."""

    def __init__(self) -> None:
        self._last_diagnostics: ClusterSolverDiagnostics | None = None

    def solve(self, problem: ClusterSolverProblem) -> ClusterSolverResult:
        validate_cluster_solver_problem(problem)

        max_results = int(problem.max_results)
        if max_results <= 0:
            self._last_diagnostics = ClusterSolverDiagnostics(
                combinations_evaluated=0,
                feasible_combinations=0,
                backend="branch_and_bound",
                optimal=True,
                solutions_returned=0,
                terminated_early=False,
                early_stop_reason="max_results_is_zero",
                search_nodes_visited=0,
                branches_pruned_conflict=0,
                branches_pruned_bound=0,
                complete_feasible_solutions=0,
            )
            return ClusterSolverResult(solutions=())

        if len(problem.track_options) == 1:
            return self._solve_one_track_fast_path(problem)

        ordered_tracks = prepare_ordered_tracks_for_search(problem)
        suffix_best_score = suffix_best_score_by_depth(ordered_tracks)
        track_count = len(ordered_tracks)
        ordered_track_ids = tuple(track.track_id for track in ordered_tracks)

        top_k = TopKSolutionHeap(k=max_results)
        top_k_push = top_k.push

        search_nodes_visited = 0
        branches_pruned_conflict = 0
        branches_pruned_bound = 0
        complete_feasible_solutions = 0
        selected_leaf_id_by_depth = [0 for _ in range(track_count)]
        retention_floor_score: float | None = None

        def depth_first_search(
            depth: int, partial_score: float, used_conflict_mask: int
        ) -> None:
            nonlocal search_nodes_visited
            nonlocal branches_pruned_conflict
            nonlocal branches_pruned_bound
            nonlocal complete_feasible_solutions
            nonlocal retention_floor_score

            if depth >= track_count:
                complete_feasible_solutions += 1
                selected_leaf_id_by_track_id = {
                    ordered_track_ids[index]: selected_leaf_id_by_depth[index]
                    for index in range(track_count)
                }
                top_k_push(
                    candidate=ClusterSolverSolution(
                        selected_leaf_id_by_track_id=selected_leaf_id_by_track_id,
                        score=float(partial_score),
                    ),
                    insertion_order=complete_feasible_solutions,
                )
                retention_floor_score = top_k.retention_floor_score()
                return

            if retention_floor_score is not None:
                node_upper_bound = float(partial_score + suffix_best_score[depth])
                if node_upper_bound <= retention_floor_score:
                    branches_pruned_bound += 1
                    return

            track = ordered_tracks[depth]
            for leaf_id, leaf_score, leaf_conflict_mask in track.leaf_options:
                search_nodes_visited += 1

                if used_conflict_mask & leaf_conflict_mask:
                    branches_pruned_conflict += 1
                    continue

                candidate_partial_score = partial_score + leaf_score
                if retention_floor_score is not None:
                    branch_upper_bound = float(
                        candidate_partial_score + suffix_best_score[depth + 1]
                    )
                    if branch_upper_bound <= retention_floor_score:
                        branches_pruned_bound += 1
                        continue

                selected_leaf_id_by_depth[depth] = leaf_id
                depth_first_search(
                    depth + 1,
                    candidate_partial_score,
                    used_conflict_mask | leaf_conflict_mask,
                )

        depth_first_search(0, 0.0, 0)

        solutions = top_k.finalize()
        if len(solutions) < max_results:
            early_stop_reason = "feasible_set_exhausted"
        else:
            early_stop_reason = "max_results_reached"

        self._last_diagnostics = ClusterSolverDiagnostics(
            combinations_evaluated=search_nodes_visited,
            feasible_combinations=complete_feasible_solutions,
            backend="branch_and_bound",
            optimal=True,
            solutions_returned=len(solutions),
            terminated_early=len(solutions) < max_results,
            early_stop_reason=early_stop_reason,
            search_nodes_visited=search_nodes_visited,
            branches_pruned_conflict=branches_pruned_conflict,
            branches_pruned_bound=branches_pruned_bound,
            complete_feasible_solutions=complete_feasible_solutions,
        )
        return ClusterSolverResult(solutions=solutions)

    def get_last_diagnostics(self) -> ClusterSolverDiagnostics | None:
        return self._last_diagnostics

    def _solve_one_track_fast_path(
        self, problem: ClusterSolverProblem
    ) -> ClusterSolverResult:
        """Solve one-track problems directly via deterministic score sort."""
        track = problem.track_options[0]
        max_results = int(problem.max_results)
        top_k = TopKSolutionHeap(k=max_results)
        top_k_push = top_k.push

        sorted_leaf_options = tuple(
            leaf
            for _, leaf in sorted(
                enumerate(track.leaf_options),
                key=lambda item: (-float(item[1].score), int(item[0])),
            )
        )

        search_nodes_visited = 0
        branches_pruned_bound = 0
        complete_feasible_solutions = 0
        retention_floor_score: float | None = None
        for leaf in sorted_leaf_options:
            search_nodes_visited += 1
            candidate_score = float(leaf.score)
            if (
                retention_floor_score is not None
                and candidate_score <= retention_floor_score
            ):
                branches_pruned_bound += (
                    len(sorted_leaf_options) - search_nodes_visited + 1
                )
                break

            complete_feasible_solutions += 1
            top_k_push(
                candidate=ClusterSolverSolution(
                    selected_leaf_id_by_track_id={
                        int(track.track_id): int(leaf.leaf_id)
                    },
                    score=candidate_score,
                ),
                insertion_order=complete_feasible_solutions,
            )
            retention_floor_score = top_k.retention_floor_score()

        solutions = top_k.finalize()
        if len(solutions) < max_results:
            early_stop_reason = "feasible_set_exhausted"
        else:
            early_stop_reason = "max_results_reached"

        self._last_diagnostics = ClusterSolverDiagnostics(
            combinations_evaluated=search_nodes_visited,
            feasible_combinations=complete_feasible_solutions,
            backend="branch_and_bound",
            optimal=True,
            solutions_returned=len(solutions),
            terminated_early=len(solutions) < max_results,
            early_stop_reason=early_stop_reason,
            search_nodes_visited=search_nodes_visited,
            branches_pruned_conflict=0,
            branches_pruned_bound=branches_pruned_bound,
            complete_feasible_solutions=complete_feasible_solutions,
        )
        return ClusterSolverResult(solutions=solutions)
