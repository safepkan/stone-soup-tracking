"""Cluster-solver contract and shared helpers for TO-MHT cluster rebuilds.

This module defines the exact per-cluster K-best problem in a solver-facing form
that is independent of tracker tree/node objects. The tracker prepares this
problem from current tree frontiers and maps solved leaf IDs back to nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import prod
from typing import Mapping, Protocol

from mht.tomht_model import DetectionKey


@dataclass(frozen=True)
class ClusterSolverLeafOption:
    """One solver-facing leaf option for one track."""

    leaf_id: int
    track_id: int
    score: float
    full_history_conflict_keys: frozenset[DetectionKey]


@dataclass(frozen=True)
class ClusterSolverTrackOptions:
    """All solver-facing leaf options for one track."""

    track_id: int
    leaf_options: tuple[ClusterSolverLeafOption, ...]


@dataclass(frozen=True)
class ClusterSolverProblem:
    """Exact cluster K-best problem passed to a solver backend.

    Semantics:
    - choose exactly one leaf option per track,
    - reject combinations with overlapping full-history conflict keys,
    - score = sum(selected leaf scores) + cluster-constant offset,
    - keep up to ``max_results`` best feasible combinations.
    """

    track_options: tuple[ClusterSolverTrackOptions, ...]
    max_results: int
    # Ranking-inert cluster constant retained temporarily to ease comparison with
    # previous versions. Intended to be dropped once validated.
    constant_score_offset: float = 0.0


@dataclass(frozen=True)
class ClusterSolverSolution:
    """One feasible solved cluster selection."""

    selected_leaf_id_by_track_id: dict[int, int]
    score: float


@dataclass(frozen=True)
class ClusterSolverResult:
    """K-best solutions for one cluster solve call."""

    solutions: tuple[ClusterSolverSolution, ...]


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

    def is_full(self) -> bool:
        """Return whether the heap currently retains exactly K candidates."""
        return self._k > 0 and len(self._heap) >= self._k

    def retention_floor_score(self) -> float | None:
        """Return the current worst retained score when heap is full."""
        if not self.is_full():
            return None
        return float(self._heap[0][0])

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


@dataclass(frozen=True)
class ClusterSolverDiagnostics:
    """Optional backend diagnostics for one cluster solve call."""

    # Required by the tracker/statistics path.
    combinations_evaluated: int
    feasible_combinations: int

    # Common fields that all backends should report explicitly.
    # Values may be ``None`` only for compatibility/fallback paths.
    backend: str | None
    optimal: bool | None
    solutions_returned: int | None
    terminated_early: bool | None
    early_stop_reason: str | None

    # Backend-specific fields (populate only when meaningful).
    solves_attempted: int | None = None
    search_nodes_visited: int | None = None
    branches_pruned_conflict: int | None = None
    branches_pruned_bound: int | None = None
    complete_feasible_solutions: int | None = None


class ClusterSolver(Protocol):
    """Backend-agnostic exact solver interface for one cluster solve problem."""

    def solve(self, problem: ClusterSolverProblem) -> ClusterSolverResult:
        """Return up to K feasible solutions in descending score order."""

    def get_last_diagnostics(self) -> ClusterSolverDiagnostics | None:
        """Return diagnostics for the most recent ``solve`` call, when available."""


def missing_cluster_solver_diagnostics() -> ClusterSolverDiagnostics:
    """Return fallback diagnostics when a solver does not report diagnostics."""
    return ClusterSolverDiagnostics(
        combinations_evaluated=0,
        feasible_combinations=0,
        backend=None,
        optimal=None,
        solutions_returned=None,
        terminated_early=None,
        early_stop_reason="missing_solver_diagnostics",
    )


def validate_cluster_solver_problem(problem: ClusterSolverProblem) -> None:
    """Validate generic cluster-solver contract invariants."""
    if int(problem.max_results) < 0:
        raise ValueError("ClusterSolverProblem.max_results must be >= 0.")
    for track in problem.track_options:
        if not track.leaf_options:
            raise ValueError(
                "ClusterSolverProblem requires at least one leaf option per track."
            )


def projected_track_combination_count(problem: ClusterSolverProblem) -> int:
    """Return Cartesian track-choice count for one cluster-solver problem."""
    return int(prod(len(track.leaf_options) for track in problem.track_options))


def score_selected_leaf_ids_if_exact_feasible(
    *,
    problem: ClusterSolverProblem,
    selected_leaf_id_by_track_id: Mapping[int, int],
    leaf_option_by_leaf_id: Mapping[int, ClusterSolverLeafOption],
) -> float | None:
    """Return exact score for a proposed track->leaf selection, if feasible.

    This evaluates one candidate selection under the exact cluster-solver
    contract:
    - exactly one selected leaf per track in ``problem.track_options``,
    - each selected leaf exists and belongs to that track,
    - selected leaves are pairwise conflict-free under
      ``full_history_conflict_keys``.

    Returns:
    - ``float`` exact score when all checks pass,
    - ``None`` when the proposal violates any feasibility constraint.
    """
    if len(selected_leaf_id_by_track_id) != len(problem.track_options):
        return None

    used_history_keys: set[DetectionKey] = set()
    leaf_score_sum = 0.0

    for track in problem.track_options:
        track_id = int(track.track_id)
        leaf_id = selected_leaf_id_by_track_id.get(track_id)
        if leaf_id is None:
            return None
        leaf = leaf_option_by_leaf_id.get(int(leaf_id))
        if leaf is None or int(leaf.track_id) != track_id:
            return None

        overlap = used_history_keys & leaf.full_history_conflict_keys
        if overlap:
            return None

        used_history_keys |= set(leaf.full_history_conflict_keys)
        leaf_score_sum += float(leaf.score)

    return float(leaf_score_sum + float(problem.constant_score_offset))
