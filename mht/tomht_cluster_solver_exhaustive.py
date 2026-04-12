"""Exhaustive exact backend for TO-MHT cluster-solver problems."""

from __future__ import annotations

from itertools import product

from mht.tomht_model import DetectionKey

from mht.tomht_cluster_solver import (
    ClusterSolverDiagnostics,
    ClusterSolverProblem,
    ClusterSolverResult,
    ClusterSolverSolution,
    TopKSolutionHeap,
    validate_cluster_solver_problem,
)


class ExhaustiveClusterSolver:
    """Exact backend: exhaustive enumeration with deterministic top-K."""

    def __init__(self) -> None:
        self._last_diagnostics: ClusterSolverDiagnostics | None = None

    def solve(self, problem: ClusterSolverProblem) -> ClusterSolverResult:
        validate_cluster_solver_problem(problem)

        track_option_lists = [track.leaf_options for track in problem.track_options]
        top_k = TopKSolutionHeap(k=int(problem.max_results))
        combinations_evaluated = 0
        feasible_combinations = 0
        constant_offset = float(problem.constant_score_offset)

        for picked in product(*track_option_lists):
            combinations_evaluated += 1
            used_history_keys: set[DetectionKey] = set()
            selected_leaf_id_by_track_id: dict[int, int] = {}
            leaf_score_sum = 0.0

            feasible = True
            for option in picked:
                overlap = used_history_keys & option.full_history_conflict_keys
                if overlap:
                    feasible = False
                    break
                used_history_keys |= option.full_history_conflict_keys
                selected_leaf_id_by_track_id[int(option.track_id)] = int(option.leaf_id)
                leaf_score_sum += float(option.score)
            if not feasible:
                continue

            feasible_combinations += 1
            candidate = ClusterSolverSolution(
                selected_leaf_id_by_track_id=selected_leaf_id_by_track_id,
                score=float(leaf_score_sum + constant_offset),
            )
            top_k.push(
                candidate=candidate,
                insertion_order=feasible_combinations,
            )

        solutions = top_k.finalize()
        max_results = int(problem.max_results)
        if max_results <= 0:
            early_stop_reason = "max_results_is_zero"
        elif len(solutions) < max_results:
            early_stop_reason = "feasible_set_exhausted"
        else:
            early_stop_reason = "max_results_reached"

        self._last_diagnostics = ClusterSolverDiagnostics(
            combinations_evaluated=combinations_evaluated,
            feasible_combinations=feasible_combinations,
            backend="exhaustive",
            optimal=True,
            solutions_returned=len(solutions),
            terminated_early=len(solutions) < max_results,
            early_stop_reason=early_stop_reason,
        )
        return ClusterSolverResult(
            solutions=solutions,
        )

    def get_last_diagnostics(self) -> ClusterSolverDiagnostics | None:
        return self._last_diagnostics
