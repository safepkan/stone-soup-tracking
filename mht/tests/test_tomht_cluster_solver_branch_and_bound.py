from __future__ import annotations

import unittest

from mht.tomht_cluster_solver import (
    ClusterSolverLeafOption,
    ClusterSolverProblem,
    ClusterSolverResult,
    ClusterSolverTrackOptions,
)
from mht.tomht_cluster_solver_branch_and_bound import BranchAndBoundClusterSolver
from mht.tomht_cluster_solver_exhaustive import ExhaustiveClusterSolver


def _leaf(
    *,
    leaf_id: int,
    track_id: int,
    score: float,
    conflict_keys: set[tuple[int, int]],
) -> ClusterSolverLeafOption:
    return ClusterSolverLeafOption(
        leaf_id=leaf_id,
        track_id=track_id,
        score=score,
        full_history_conflict_keys=frozenset(conflict_keys),
    )


def _selection_key(result: ClusterSolverResult) -> list[tuple[tuple[int, int], ...]]:
    return [
        tuple(sorted(solution.selected_leaf_id_by_track_id.items()))
        for solution in result.solutions
    ]


def _score_map(result: ClusterSolverResult) -> dict[tuple[tuple[int, int], ...], float]:
    return {
        tuple(sorted(solution.selected_leaf_id_by_track_id.items())): float(
            solution.score
        )
        for solution in result.solutions
    }


class TOMHTClusterSolverBranchAndBoundTest(unittest.TestCase):
    def test_matches_exhaustive_top_k_and_scores_on_small_exact_problem(self) -> None:
        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=10,
                            track_id=0,
                            score=1.5,
                            conflict_keys={(2, 0)},
                        ),
                        _leaf(
                            leaf_id=11,
                            track_id=0,
                            score=0.3,
                            conflict_keys=set(),
                        ),
                    ),
                ),
                ClusterSolverTrackOptions(
                    track_id=1,
                    leaf_options=(
                        _leaf(
                            leaf_id=20,
                            track_id=1,
                            score=2.2,
                            conflict_keys={(2, 1)},
                        ),
                        _leaf(
                            leaf_id=21,
                            track_id=1,
                            score=1.1,
                            conflict_keys={(2, 0)},
                        ),
                    ),
                ),
                ClusterSolverTrackOptions(
                    track_id=2,
                    leaf_options=(
                        _leaf(
                            leaf_id=30,
                            track_id=2,
                            score=0.9,
                            conflict_keys={(2, 2)},
                        ),
                        _leaf(
                            leaf_id=31,
                            track_id=2,
                            score=0.5,
                            conflict_keys={(2, 1)},
                        ),
                    ),
                ),
            ),
            max_results=5,
            constant_score_offset=-0.25,
        )

        exhaustive_result = ExhaustiveClusterSolver().solve(problem)
        bnb_solver = BranchAndBoundClusterSolver()
        bnb_result = bnb_solver.solve(problem)

        self.assertEqual(_selection_key(exhaustive_result), _selection_key(bnb_result))
        self.assertEqual(_score_map(exhaustive_result), _score_map(bnb_result))

        diagnostics = bnb_solver.get_last_diagnostics()
        assert diagnostics is not None
        self.assertEqual("branch_and_bound", diagnostics.backend)
        self.assertTrue(diagnostics.optimal)
        self.assertEqual(
            diagnostics.feasible_combinations, diagnostics.complete_feasible_solutions
        )
        self.assertGreaterEqual(int(diagnostics.search_nodes_visited or 0), 1)
        self.assertGreaterEqual(int(diagnostics.branches_pruned_bound or 0), 0)
        self.assertGreaterEqual(int(diagnostics.branches_pruned_conflict or 0), 0)

    def test_returns_fewer_than_k_when_feasible_set_is_small(self) -> None:
        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=1,
                            track_id=0,
                            score=0.0,
                            conflict_keys={(0, 0)},
                        ),
                        _leaf(
                            leaf_id=2,
                            track_id=0,
                            score=0.0,
                            conflict_keys={(0, 1)},
                        ),
                    ),
                ),
                ClusterSolverTrackOptions(
                    track_id=1,
                    leaf_options=(
                        _leaf(
                            leaf_id=3,
                            track_id=1,
                            score=0.0,
                            conflict_keys={(0, 1)},
                        ),
                    ),
                ),
            ),
            max_results=5,
        )

        exhaustive_result = ExhaustiveClusterSolver().solve(problem)
        bnb_solver = BranchAndBoundClusterSolver()
        bnb_result = bnb_solver.solve(problem)

        self.assertEqual(_selection_key(exhaustive_result), _selection_key(bnb_result))
        self.assertEqual(_score_map(exhaustive_result), _score_map(bnb_result))
        self.assertEqual(1, len(bnb_result.solutions))

        diagnostics = bnb_solver.get_last_diagnostics()
        assert diagnostics is not None
        self.assertTrue(diagnostics.terminated_early)
        self.assertEqual("feasible_set_exhausted", diagnostics.early_stop_reason)

    def test_conflict_exclusion_matches_exhaustive(self) -> None:
        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=10,
                            track_id=0,
                            score=5.0,
                            conflict_keys={(9, 0)},
                        ),
                        _leaf(
                            leaf_id=11,
                            track_id=0,
                            score=1.0,
                            conflict_keys=set(),
                        ),
                    ),
                ),
                ClusterSolverTrackOptions(
                    track_id=1,
                    leaf_options=(
                        _leaf(
                            leaf_id=20,
                            track_id=1,
                            score=4.0,
                            conflict_keys={(9, 0)},
                        ),
                        _leaf(
                            leaf_id=21,
                            track_id=1,
                            score=2.0,
                            conflict_keys=set(),
                        ),
                    ),
                ),
                ClusterSolverTrackOptions(
                    track_id=2,
                    leaf_options=(
                        _leaf(
                            leaf_id=30,
                            track_id=2,
                            score=3.0,
                            conflict_keys=set(),
                        ),
                    ),
                ),
            ),
            max_results=3,
        )

        exhaustive_result = ExhaustiveClusterSolver().solve(problem)
        bnb_result = BranchAndBoundClusterSolver().solve(problem)

        self.assertEqual(_selection_key(exhaustive_result), _selection_key(bnb_result))
        self.assertEqual(_score_map(exhaustive_result), _score_map(bnb_result))

    def test_one_track_fast_path_matches_exhaustive(self) -> None:
        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=7,
                    leaf_options=(
                        _leaf(
                            leaf_id=700,
                            track_id=7,
                            score=3.0,
                            conflict_keys=set(),
                        ),
                        _leaf(
                            leaf_id=701,
                            track_id=7,
                            score=1.0,
                            conflict_keys=set(),
                        ),
                        _leaf(
                            leaf_id=702,
                            track_id=7,
                            score=1.0,
                            conflict_keys=set(),
                        ),
                    ),
                ),
            ),
            max_results=2,
        )

        exhaustive_result = ExhaustiveClusterSolver().solve(problem)
        bnb_solver = BranchAndBoundClusterSolver()
        bnb_result = bnb_solver.solve(problem)

        self.assertEqual(_selection_key(exhaustive_result), _selection_key(bnb_result))
        self.assertEqual(_score_map(exhaustive_result), _score_map(bnb_result))
        diagnostics = bnb_solver.get_last_diagnostics()
        assert diagnostics is not None
        self.assertGreaterEqual(int(diagnostics.branches_pruned_bound or 0), 1)

    def test_tie_case_matches_solution_set_even_if_order_differs(self) -> None:
        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=100,
                            track_id=0,
                            score=0.0,
                            conflict_keys=set(),
                        ),
                        _leaf(
                            leaf_id=101,
                            track_id=0,
                            score=0.0,
                            conflict_keys=set(),
                        ),
                    ),
                ),
                ClusterSolverTrackOptions(
                    track_id=1,
                    leaf_options=(
                        _leaf(
                            leaf_id=200,
                            track_id=1,
                            score=0.0,
                            conflict_keys=set(),
                        ),
                        _leaf(
                            leaf_id=201,
                            track_id=1,
                            score=0.0,
                            conflict_keys=set(),
                        ),
                    ),
                ),
            ),
            max_results=4,
        )

        exhaustive_result = ExhaustiveClusterSolver().solve(problem)
        bnb_result = BranchAndBoundClusterSolver().solve(problem)

        self.assertEqual(
            set(_selection_key(exhaustive_result)),
            set(_selection_key(bnb_result)),
        )
        self.assertEqual(_score_map(exhaustive_result), _score_map(bnb_result))


if __name__ == "__main__":
    unittest.main()
