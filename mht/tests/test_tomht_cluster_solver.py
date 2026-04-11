from __future__ import annotations

import unittest

from mht.tomht_cluster_solver import (
    ClusterSolverLeafOption,
    ClusterSolverProblem,
    ClusterSolverTrackOptions,
    ExhaustiveClusterSolver,
)


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


class TOMHTClusterSolverTest(unittest.TestCase):
    def test_conflicts_and_unused_term_match_current_objective(self) -> None:
        solver = ExhaustiveClusterSolver()

        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=10,
                            track_id=0,
                            score=5.5,
                            conflict_keys={(1, 0)},
                        ),
                        _leaf(
                            leaf_id=11,
                            track_id=0,
                            score=3.0,
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
                            score=4.5,
                            conflict_keys={(1, 1)},
                        ),
                        _leaf(
                            leaf_id=21,
                            track_id=1,
                            score=2.5,
                            conflict_keys={(1, 0)},
                        ),
                    ),
                ),
            ),
            max_results=2,
            # Ranking-inert cluster constant retained for backward comparability.
            constant_score_offset=-1.0,
        )
        result = solver.solve(problem)
        diagnostics = solver.get_last_diagnostics()
        assert diagnostics is not None

        self.assertEqual(4, diagnostics.combinations_evaluated)
        self.assertEqual(3, diagnostics.feasible_combinations)
        self.assertEqual(2, len(result.solutions))
        self.assertEqual(
            {0: 10, 1: 20}, result.solutions[0].selected_leaf_id_by_track_id
        )
        self.assertEqual(9.0, result.solutions[0].score)
        self.assertEqual(
            {0: 11, 1: 20},
            result.solutions[1].selected_leaf_id_by_track_id,
        )
        self.assertEqual(6.5, result.solutions[1].score)

    def test_tie_order_is_deterministic_by_enumeration_order(self) -> None:
        solver = ExhaustiveClusterSolver()
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
            max_results=2,
        )

        result = solver.solve(problem)
        diagnostics = solver.get_last_diagnostics()
        assert diagnostics is not None

        self.assertEqual(4, diagnostics.combinations_evaluated)
        self.assertEqual(4, diagnostics.feasible_combinations)
        self.assertEqual(2, len(result.solutions))
        self.assertEqual(
            {0: 100, 1: 201},
            result.solutions[0].selected_leaf_id_by_track_id,
        )
        self.assertEqual(
            {0: 100, 1: 200},
            result.solutions[1].selected_leaf_id_by_track_id,
        )

    def test_solver_can_return_fewer_than_k_when_feasible_set_is_small(self) -> None:
        solver = ExhaustiveClusterSolver()
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

        result = solver.solve(problem)
        diagnostics = solver.get_last_diagnostics()
        assert diagnostics is not None
        self.assertEqual(2, diagnostics.combinations_evaluated)
        self.assertEqual(1, diagnostics.feasible_combinations)
        self.assertEqual(1, len(result.solutions))
        self.assertEqual({0: 1, 1: 3}, result.solutions[0].selected_leaf_id_by_track_id)

    def test_linear_used_detection_term_matches_unused_form_ranking(self) -> None:
        solver = ExhaustiveClusterSolver()
        # old callback form: -0.4 * (3 - used_count) = -1.2 + 0.4 * used_count.
        # The 0.4*used_count part is pre-baked into hit-leaf scores.
        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=10,
                            track_id=0,
                            score=2.4,
                            conflict_keys={(1, 0)},
                        ),
                        _leaf(
                            leaf_id=11,
                            track_id=0,
                            score=1.8,
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
                            score=2.4,
                            conflict_keys={(1, 1)},
                        ),
                        _leaf(
                            leaf_id=21,
                            track_id=1,
                            score=2.1,
                            conflict_keys={(1, 0)},
                        ),
                    ),
                ),
            ),
            max_results=4,
            constant_score_offset=-1.2,
        )

        result = solver.solve(problem)

        # Reference ranking under old callback form.
        def old_unused_callback(*, used_count: int) -> float:
            return -0.4 * float(3 - used_count)

        expected_scores_by_selection = {
            ((0, 10), (1, 20)): 2.0 + 2.0 + old_unused_callback(used_count=2),
            ((0, 11), (1, 20)): 1.8 + 2.0 + old_unused_callback(used_count=1),
            ((0, 11), (1, 21)): 1.8 + 1.7 + old_unused_callback(used_count=1),
        }
        expected_order = sorted(
            expected_scores_by_selection.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        observed_order = [
            (
                tuple(sorted(solution.selected_leaf_id_by_track_id.items())),
                solution.score,
            )
            for solution in result.solutions
        ]

        self.assertEqual(
            [selection for selection, _ in expected_order],
            [selection for selection, _ in observed_order],
        )


if __name__ == "__main__":
    unittest.main()
