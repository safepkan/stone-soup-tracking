from __future__ import annotations

import unittest

from mht.tomht_cluster_solver_branch_and_bound import BranchAndBoundClusterSolver
from mht.tomht_cluster_solver_exhaustive import ExhaustiveClusterSolver
from mht.tomht_cluster_solver_factory import make_cluster_solver


class TOMHTClusterSolverFactoryTest(unittest.TestCase):
    def test_make_cluster_solver_resolves_supported_aliases(self) -> None:
        self.assertIsInstance(
            make_cluster_solver("exhaustive"),
            ExhaustiveClusterSolver,
        )
        self.assertIsInstance(
            make_cluster_solver("branch_and_bound"),
            BranchAndBoundClusterSolver,
        )
        self.assertIsInstance(
            make_cluster_solver("branch-and-bound"),
            BranchAndBoundClusterSolver,
        )
        self.assertIsInstance(
            make_cluster_solver("bnb"),
            BranchAndBoundClusterSolver,
        )

    def test_make_cluster_solver_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown cluster solver backend"):
            make_cluster_solver("unknown")
