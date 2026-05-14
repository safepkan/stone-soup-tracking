"""Factory for configured TO-MHT cluster solver backends."""

from __future__ import annotations

from .tomht_cluster_solver import ClusterSolver
from .tomht_cluster_solver_branch_and_bound import BranchAndBoundClusterSolver
from .tomht_cluster_solver_exhaustive import ExhaustiveClusterSolver


def make_cluster_solver(cluster_solver_backend: str) -> ClusterSolver:
    """Construct one exact cluster-solver backend by configured name."""
    backend = str(cluster_solver_backend).strip().lower()
    if backend == "exhaustive":
        return ExhaustiveClusterSolver()
    if backend in {"branch_and_bound", "branch-and-bound", "bnb"}:
        return BranchAndBoundClusterSolver()
    if backend in {"ortools", "ortools_cp_sat", "cp_sat"}:
        from .tomht_cluster_solver_ortools import ORToolsClusterSolver

        return ORToolsClusterSolver()
    raise ValueError(
        "Unknown cluster solver backend. "
        f"cluster_solver_backend={cluster_solver_backend!r}"
    )
