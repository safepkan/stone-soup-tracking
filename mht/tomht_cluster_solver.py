"""Cluster-solver contract and exhaustive backend for TO-MHT cluster rebuilds.

This module defines the exact per-cluster K-best problem in a solver-facing form
that is independent of tracker tree/node objects. The tracker prepares this
problem from current tree frontiers and maps solved leaf IDs back to nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import product
from typing import Callable, Protocol

from mht.tomht_model import DetectionKey


type UnusedCurrentScanScoreFn = Callable[[frozenset[int]], float]


@dataclass(frozen=True)
class ClusterSolverLeafOption:
    """One solver-facing leaf option for one track."""

    leaf_id: int
    track_id: int
    accumulated_log_score: float
    full_history_conflict_keys: frozenset[DetectionKey]
    used_current_scan_det_indices: frozenset[int]


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
    - score = sum(selected leaf accumulated scores) + optional cluster-local
      unused-detection term from the current-scan used-detection union,
    - keep up to ``max_results`` best feasible combinations.
    """

    track_options: tuple[ClusterSolverTrackOptions, ...]
    max_results: int
    score_unused_current_scan_detections: UnusedCurrentScanScoreFn | None = None


@dataclass(frozen=True)
class ClusterSolverSolution:
    """One feasible solved cluster selection."""

    selected_leaf_id_by_track_id: dict[int, int]
    log_weight: float


@dataclass(frozen=True)
class ClusterSolverResult:
    """K-best result plus enumeration counters for one cluster solve call."""

    solutions: tuple[ClusterSolverSolution, ...]
    combinations_evaluated: int
    feasible_combinations: int


class ClusterSolver(Protocol):
    """Backend-agnostic exact solver interface for one cluster solve problem."""

    def solve(self, problem: ClusterSolverProblem) -> ClusterSolverResult:
        """Return up to K feasible solutions in descending score order."""


type _TopKHeap = list[tuple[float, int, ClusterSolverSolution]]


class ExhaustiveClusterSolver:
    """Current exact backend: exhaustive enumeration with deterministic top-K."""

    @staticmethod
    def _push_top_k(
        *,
        top_k_heap: _TopKHeap,
        candidate: ClusterSolverSolution,
        insertion_order: int,
        k: int,
    ) -> None:
        """Streaming top-K with existing tie retention (earlier feasible combos kept)."""
        if k <= 0:
            return

        entry = (
            float(candidate.log_weight),
            -int(insertion_order),
            candidate,
        )
        if len(top_k_heap) < k:
            heapq.heappush(top_k_heap, entry)
            return
        if entry > top_k_heap[0]:
            heapq.heapreplace(top_k_heap, entry)

    @staticmethod
    def _finalize_top_k(top_k_heap: _TopKHeap) -> tuple[ClusterSolverSolution, ...]:
        """Return retained solutions sorted best-first with existing tie ordering."""
        top_k_heap.sort(
            key=lambda item: (
                float(item[0]),  # log_weight
                int(-item[1]),  # existing deterministic tie order by insertion
            ),
            reverse=True,
        )
        return tuple(item[2] for item in top_k_heap)

    @staticmethod
    def _validate_problem(problem: ClusterSolverProblem) -> None:
        if int(problem.max_results) < 0:
            raise ValueError("ClusterSolverProblem.max_results must be >= 0.")
        for track in problem.track_options:
            if not track.leaf_options:
                raise ValueError(
                    "ClusterSolverProblem requires at least one leaf option per track."
                )

    def solve(self, problem: ClusterSolverProblem) -> ClusterSolverResult:
        self._validate_problem(problem)

        track_option_lists = [track.leaf_options for track in problem.track_options]
        top_k_heap: _TopKHeap = []
        combinations_evaluated = 0
        feasible_combinations = 0
        k = int(problem.max_results)
        score_unused = problem.score_unused_current_scan_detections

        for picked in product(*track_option_lists):
            combinations_evaluated += 1
            used_history_keys: set[DetectionKey] = set()
            used_current_scan_det_indices: set[int] = set()
            selected_leaf_id_by_track_id: dict[int, int] = {}
            leaf_score_sum = 0.0

            feasible = True
            for option in picked:
                overlap = used_history_keys & option.full_history_conflict_keys
                if overlap:
                    feasible = False
                    break
                used_history_keys |= option.full_history_conflict_keys
                used_current_scan_det_indices |= option.used_current_scan_det_indices
                selected_leaf_id_by_track_id[int(option.track_id)] = int(option.leaf_id)
                leaf_score_sum += float(option.accumulated_log_score)
            if not feasible:
                continue

            feasible_combinations += 1
            unused_term = 0.0
            if score_unused is not None:
                unused_term = float(
                    score_unused(frozenset(used_current_scan_det_indices))
                )

            candidate = ClusterSolverSolution(
                selected_leaf_id_by_track_id=selected_leaf_id_by_track_id,
                log_weight=float(leaf_score_sum + unused_term),
            )
            self._push_top_k(
                top_k_heap=top_k_heap,
                candidate=candidate,
                insertion_order=feasible_combinations,
                k=k,
            )

        return ClusterSolverResult(
            solutions=self._finalize_top_k(top_k_heap),
            combinations_evaluated=combinations_evaluated,
            feasible_combinations=feasible_combinations,
        )
