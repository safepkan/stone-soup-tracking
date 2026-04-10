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
    used_det_indices: set[int],
) -> ClusterSolverLeafOption:
    return ClusterSolverLeafOption(
        leaf_id=leaf_id,
        track_id=track_id,
        accumulated_log_score=score,
        full_history_conflict_keys=frozenset(conflict_keys),
        used_current_scan_det_indices=frozenset(used_det_indices),
    )


class TOMHTClusterSolverTest(unittest.TestCase):
    def test_conflicts_and_unused_term_match_current_objective(self) -> None:
        solver = ExhaustiveClusterSolver()

        def score_unused(used: frozenset[int]) -> float:
            return -0.5 * float(2 - len(used))

        problem = ClusterSolverProblem(
            track_options=(
                ClusterSolverTrackOptions(
                    track_id=0,
                    leaf_options=(
                        _leaf(
                            leaf_id=10,
                            track_id=0,
                            score=5.0,
                            conflict_keys={(1, 0)},
                            used_det_indices={0},
                        ),
                        _leaf(
                            leaf_id=11,
                            track_id=0,
                            score=3.0,
                            conflict_keys=set(),
                            used_det_indices=set(),
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
                            conflict_keys={(1, 1)},
                            used_det_indices={1},
                        ),
                        _leaf(
                            leaf_id=21,
                            track_id=1,
                            score=2.0,
                            conflict_keys={(1, 0)},
                            used_det_indices={0},
                        ),
                    ),
                ),
            ),
            max_results=2,
            score_unused_current_scan_detections=score_unused,
        )
        result = solver.solve(problem)

        self.assertEqual(4, result.combinations_evaluated)
        self.assertEqual(3, result.feasible_combinations)
        self.assertEqual(2, len(result.solutions))
        self.assertEqual(
            {0: 10, 1: 20}, result.solutions[0].selected_leaf_id_by_track_id
        )
        self.assertEqual(9.0, result.solutions[0].log_weight)
        self.assertEqual(
            {0: 11, 1: 20},
            result.solutions[1].selected_leaf_id_by_track_id,
        )
        self.assertEqual(6.5, result.solutions[1].log_weight)

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
                            used_det_indices=set(),
                        ),
                        _leaf(
                            leaf_id=101,
                            track_id=0,
                            score=0.0,
                            conflict_keys=set(),
                            used_det_indices=set(),
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
                            used_det_indices=set(),
                        ),
                        _leaf(
                            leaf_id=201,
                            track_id=1,
                            score=0.0,
                            conflict_keys=set(),
                            used_det_indices=set(),
                        ),
                    ),
                ),
            ),
            max_results=2,
            score_unused_current_scan_detections=None,
        )

        result = solver.solve(problem)

        self.assertEqual(4, result.combinations_evaluated)
        self.assertEqual(4, result.feasible_combinations)
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
                            used_det_indices=set(),
                        ),
                        _leaf(
                            leaf_id=2,
                            track_id=0,
                            score=0.0,
                            conflict_keys={(0, 1)},
                            used_det_indices=set(),
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
                            used_det_indices=set(),
                        ),
                    ),
                ),
            ),
            max_results=5,
        )

        result = solver.solve(problem)
        self.assertEqual(2, result.combinations_evaluated)
        self.assertEqual(1, result.feasible_combinations)
        self.assertEqual(1, len(result.solutions))
        self.assertEqual({0: 1, 1: 3}, result.solutions[0].selected_leaf_id_by_track_id)


if __name__ == "__main__":
    unittest.main()
