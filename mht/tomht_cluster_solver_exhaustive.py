"""Exhaustive exact backend for TO-MHT cluster-solver problems."""

from __future__ import annotations

import heapq
from itertools import product

from mht.tomht_model import DetectionKey

from mht.tomht_cluster_solver import (
    ClusterSolverDiagnostics,
    ClusterSolverProblem,
    ClusterSolverResult,
    ClusterSolverSolution,
    validate_cluster_solver_problem,
)


class TopKSolutionHeap:
    """Deterministic streaming top-K retention for scored solver solutions."""

    def __init__(self, *, k: int) -> None:
        self._k = int(k)
        self._heap: list[tuple[float, int, ClusterSolverSolution]] = []

    def push(
        self,
        *,
        candidate: ClusterSolverSolution,
        insertion_order: int,
    ) -> None:
        """Consider one candidate for top-K retention."""
        if self._k <= 0:
            return

        entry = (
            float(candidate.score),
            -int(insertion_order),
            candidate,
        )
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, entry)
            return
        if entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def finalize(self) -> tuple[ClusterSolverSolution, ...]:
        """Return retained solutions sorted best-first with deterministic tie order."""
        self._heap.sort(
            key=lambda item: (
                float(item[0]),  # score
                int(-item[1]),  # deterministic tie order by insertion
            ),
            reverse=True,
        )
        return tuple(item[2] for item in self._heap)


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

        self._last_diagnostics = ClusterSolverDiagnostics(
            combinations_evaluated=combinations_evaluated,
            feasible_combinations=feasible_combinations,
        )
        return ClusterSolverResult(
            solutions=top_k.finalize(),
        )

    def get_last_diagnostics(self) -> ClusterSolverDiagnostics | None:
        return self._last_diagnostics
