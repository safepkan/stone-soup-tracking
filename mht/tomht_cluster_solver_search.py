"""Shared compact search helpers for exact cluster-solver problems."""

from __future__ import annotations

from dataclasses import dataclass

from .tomht_cluster_solver import (
    ClusterSolverProblem,
    validate_cluster_solver_problem,
)
from .tomht_model import DetectionKey


@dataclass(frozen=True)
class OrderedTrackOptions:
    """One track's search metadata after deterministic ordering."""

    track_id: int
    # Per-leaf tuple layout: (leaf_id, score, conflict_mask).
    leaf_options: tuple[tuple[int, float, int], ...]
    conflict_burden: int


def prepare_ordered_tracks_for_search(
    problem: ClusterSolverProblem,
) -> tuple[OrderedTrackOptions, ...]:
    """Order tracks and precompute compact per-leaf search tuples."""
    key_to_track_ids: dict[DetectionKey, set[int]] = {}
    keys_by_track_id: dict[int, set[DetectionKey]] = {}

    for track in problem.track_options:
        track_id = int(track.track_id)
        track_keys = keys_by_track_id.setdefault(track_id, set())
        for leaf in track.leaf_options:
            for key in leaf.full_history_conflict_keys:
                track_keys.add(key)
                key_to_track_ids.setdefault(key, set()).add(track_id)

    shared_history_key_bit_by_key: dict[DetectionKey, int] = {}
    next_bit_index = 0
    for key, track_ids in key_to_track_ids.items():
        if len(track_ids) <= 1:
            continue
        shared_history_key_bit_by_key[key] = 1 << next_bit_index
        next_bit_index += 1

    ordered_tracks: list[OrderedTrackOptions] = []
    for track in problem.track_options:
        track_id = int(track.track_id)
        conflict_burden = sum(
            1
            for key in keys_by_track_id.get(track_id, set())
            if len(key_to_track_ids.get(key, set())) > 1
        )
        ordered_leaf_options_list: list[tuple[int, float, int]] = []
        for _, leaf in sorted(
            enumerate(track.leaf_options),
            key=lambda item: (-float(item[1].score), int(item[0])),
        ):
            conflict_mask = 0
            for key in leaf.full_history_conflict_keys:
                bit = shared_history_key_bit_by_key.get(key)
                if bit is None:
                    continue
                conflict_mask |= bit
            ordered_leaf_options_list.append(
                (
                    int(leaf.leaf_id),
                    float(leaf.score),
                    conflict_mask,
                )
            )
        ordered_leaf_options = tuple(ordered_leaf_options_list)
        ordered_tracks.append(
            OrderedTrackOptions(
                track_id=track_id,
                leaf_options=ordered_leaf_options,
                conflict_burden=conflict_burden,
            )
        )

    ordered_tracks.sort(
        key=lambda track: (
            len(track.leaf_options),
            -int(track.conflict_burden),
            int(track.track_id),
        )
    )
    return tuple(ordered_tracks)


def suffix_best_score_by_depth(
    ordered_tracks: tuple[OrderedTrackOptions, ...],
) -> tuple[float, ...]:
    """Compute optimistic suffix bounds by remaining search depth."""
    suffix_best_score = [0.0 for _ in range(len(ordered_tracks) + 1)]
    for depth in range(len(ordered_tracks) - 1, -1, -1):
        track = ordered_tracks[depth]
        best_track_score = float(track.leaf_options[0][1])
        suffix_best_score[depth] = suffix_best_score[depth + 1] + best_track_score
    return tuple(suffix_best_score)


def has_any_feasible_solver_combination(problem: ClusterSolverProblem) -> bool:
    """Return whether one feasible solver selection exists.

    This uses the same compact conflict-mask representation and track ordering
    as the branch-and-bound backend, but it is an existential check rather than
    a K-best solve. It stops as soon as one complete feasible assignment is
    found.
    """
    validate_cluster_solver_problem(problem)

    ordered_tracks = prepare_ordered_tracks_for_search(problem)
    track_count = len(ordered_tracks)

    def depth_first_search(depth: int, used_conflict_mask: int) -> bool:
        if depth >= track_count:
            return True

        track = ordered_tracks[depth]
        for _, _, leaf_conflict_mask in track.leaf_options:
            if used_conflict_mask & leaf_conflict_mask:
                continue
            if depth_first_search(depth + 1, used_conflict_mask | leaf_conflict_mask):
                return True
        return False

    return depth_first_search(0, 0)
