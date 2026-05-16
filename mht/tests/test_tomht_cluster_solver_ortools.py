from __future__ import annotations

import unittest

from mht.tomht_cluster_solver import (
    ClusterSolverLeafOption,
    ClusterSolverProblem,
    ClusterSolverResult,
    ClusterSolverTrackOptions,
)
from mht.tomht_cluster_solver_exhaustive import ExhaustiveClusterSolver
from mht.tomht_cluster_solver_ortools import ORToolsClusterSolver
from mht.tomht_model import DetectionKey


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
        full_history_conflict_keys=frozenset(
            DetectionKey(scan_index=scan_index, det_index=det_index)
            for scan_index, det_index in conflict_keys
        ),
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


class TOMHTClusterSolverORToolsTest(unittest.TestCase):
    def test_constructor_rejects_negative_extra_iterations(self) -> None:
        with self.assertRaises(ValueError):
            ORToolsClusterSolver(extra_k_best_iterations=-1)

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
                            score=0.4,
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
                            score=2.0,
                            conflict_keys={(2, 1)},
                        ),
                        _leaf(
                            leaf_id=21,
                            track_id=1,
                            score=1.2,
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
                            score=0.7,
                            conflict_keys={(2, 1)},
                        ),
                    ),
                ),
            ),
            max_results=5,
        )

        exhaustive_solver = ExhaustiveClusterSolver()
        ortools_solver = ORToolsClusterSolver(score_scale=1_000_000)

        exhaustive_result = exhaustive_solver.solve(problem)
        ortools_result = ortools_solver.solve(problem)

        self.assertEqual(
            _selection_key(exhaustive_result), _selection_key(ortools_result)
        )
        for exhaustive_solution, ortools_solution in zip(
            exhaustive_result.solutions,
            ortools_result.solutions,
            strict=True,
        ):
            self.assertAlmostEqual(exhaustive_solution.score, ortools_solution.score)

        diagnostics = ortools_solver.get_last_diagnostics()
        assert diagnostics is not None
        self.assertEqual("ortools_cp_sat", diagnostics.backend)
        self.assertTrue(diagnostics.optimal)
        self.assertEqual(len(ortools_result.solutions), diagnostics.solutions_returned)

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

        exhaustive_solver = ExhaustiveClusterSolver()
        ortools_solver = ORToolsClusterSolver(score_scale=1_000_000)

        exhaustive_result = exhaustive_solver.solve(problem)
        ortools_result = ortools_solver.solve(problem)

        self.assertEqual(
            _selection_key(exhaustive_result), _selection_key(ortools_result)
        )
        self.assertEqual(1, len(ortools_result.solutions))

        diagnostics = ortools_solver.get_last_diagnostics()
        assert diagnostics is not None
        self.assertTrue(diagnostics.terminated_early)
        self.assertEqual("infeasible_or_exhausted", diagnostics.early_stop_reason)

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
        ortools_result = ORToolsClusterSolver(score_scale=1_000_000).solve(problem)

        self.assertEqual(
            _selection_key(exhaustive_result), _selection_key(ortools_result)
        )
        self.assertEqual(_score_map(exhaustive_result), _score_map(ortools_result))

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
        ortools_result = ORToolsClusterSolver(score_scale=1_000_000).solve(problem)

        self.assertEqual(
            set(_selection_key(exhaustive_result)),
            set(_selection_key(ortools_result)),
        )
        self.assertEqual(_score_map(exhaustive_result), _score_map(ortools_result))

    def test_extra_k_best_iterations_extends_solve_budget(self) -> None:
        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=10,
                            track_id=0,
                            score=4.0,
                            conflict_keys=set(),
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
                            score=3.0,
                            conflict_keys=set(),
                        ),
                        _leaf(
                            leaf_id=21,
                            track_id=1,
                            score=2.0,
                            conflict_keys=set(),
                        ),
                    ),
                ),
            ),
            max_results=1,
        )

        ortools_solver = ORToolsClusterSolver(
            score_scale=1_000_000, extra_k_best_iterations=2
        )
        result = ortools_solver.solve(problem)

        self.assertEqual(1, len(result.solutions))
        self.assertEqual(
            {0: 10, 1: 20},
            result.solutions[0].selected_leaf_id_by_track_id,
        )

        diagnostics = ortools_solver.get_last_diagnostics()
        assert diagnostics is not None
        self.assertEqual(3, diagnostics.solves_attempted)
        self.assertEqual(3, diagnostics.feasible_combinations)
        self.assertEqual("solve_budget_reached", diagnostics.early_stop_reason)


if __name__ == "__main__":
    unittest.main()
