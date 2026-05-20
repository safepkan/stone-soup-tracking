"""Local expansion helpers for the track-oriented TOMHT tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from stonesoup.hypothesiser.base import Hypothesiser
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis, SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.updater.base import Updater

from .tomht_model import DetectionKey, ScanContext, TrackHypothesisNode, TrackTree
from .tomht_output import reconstruct_track_from_leaf_node
from .tomht_params import TOMHTParams
from .tomht_scoring import ScoringModel
from .tomht_tree_store import TrackTreeStore
from .utils import elapsed_ns, start_timer


@dataclass(frozen=True)
class LocalChildCandidate:
    """One retained local child candidate produced from one leaf expansion."""

    track_id: int
    child_node: TrackHypothesisNode
    used_det_key: DetectionKey | None
    log_delta: float


@dataclass
class ExpansionCallStats:
    """Per-scan expansion timing/call counters for key Stone Soup callbacks."""

    hypothesise_calls: int = 0
    hypothesise_wall_ns: int = 0
    update_calls: int = 0
    update_wall_ns: int = 0
    expanded_leaf_count: int = 0
    expanded_leaves_tentative: int = 0
    expanded_leaves_confirmed: int = 0
    local_child_candidates_total: int = 0
    local_children_created_total: int = 0
    local_children_retained_total: int = 0
    local_miss_children_created: int = 0
    local_detection_children_created: int = 0


def validate_distance_hypothesis(
    hyp: SingleHypothesis,
) -> SingleDistanceHypothesis:
    """Validate one local distance hypothesis from the configured hypothesiser."""
    if not isinstance(hyp, SingleDistanceHypothesis):
        raise TypeError(
            "Distance hypothesiser must yield SingleDistanceHypothesis items."
        )
    if not isinstance(hyp.measurement, (Detection, MissedDetection)):
        raise TypeError(
            "Distance hypothesis measurement must be Detection or MissedDetection."
        )
    if not np.isfinite(float(hyp.distance)):
        raise ValueError("Distance hypothesis distance must be finite.")
    return hyp


def candidate_from_distance_hypothesis(
    *,
    leaf_node: TrackHypothesisNode,
    hypothesis: SingleDistanceHypothesis,
    log_delta: float,
    ctx: ScanContext,
    updater: Updater,
    tree_store: TrackTreeStore,
    assoc_miss_label: int,
    expansion_call_stats: ExpansionCallStats,
) -> LocalChildCandidate:
    """Map one distance hypothesis to one child node candidate."""
    if isinstance(hypothesis.measurement, MissedDetection):
        state = hypothesis.prediction
        used_det_key = None
        assoc_label = assoc_miss_label
        state_kind = "prediction"
        missed_count = int(leaf_node.missed_count) + 1
        last_det_key = leaf_node.last_det_key
    else:
        detection = hypothesis.measurement
        det_index_raw = ctx.det_index_by_obj.get(id(detection))
        if det_index_raw is None:
            raise ValueError(
                "Distance detection hypothesis must reference one of the "
                "input detection objects for this scan."
            )
        det_index = int(det_index_raw)
        update_start_ns = start_timer()
        state = updater.update(hypothesis)
        expansion_call_stats.update_calls += 1
        expansion_call_stats.update_wall_ns += elapsed_ns(update_start_ns)
        used_det_key = DetectionKey(scan_index=ctx.scan_index, det_index=det_index)
        assoc_label = det_index
        state_kind = "update"
        missed_count = 0
        last_det_key = used_det_key

    child_node = tree_store.create_track_hypothesis_node(
        track_id=leaf_node.track_id,
        parent=leaf_node,
        scan_index=ctx.scan_index,
        timestamp=getattr(state, "timestamp", ctx.timestamp),
        state=state,
        state_kind=state_kind,
        used_det_key=used_det_key,
        assoc_label=assoc_label,
        log_delta=log_delta,
        age=int(leaf_node.age) + 1,
        hits=int(leaf_node.hits) + (0 if used_det_key is None else 1),
        missed_count=missed_count,
        last_det_key=last_det_key,
        last_det_hit=used_det_key is not None,
        root_source=leaf_node.root_source,
        birth_scan_index=leaf_node.birth_scan_index,
    )
    return LocalChildCandidate(
        track_id=leaf_node.track_id,
        child_node=child_node,
        used_det_key=used_det_key,
        log_delta=log_delta,
    )


def candidates_for_track_leaf(
    *,
    leaf_node: TrackHypothesisNode,
    tree: TrackTree,
    ctx: ScanContext,
    hypothesiser: Hypothesiser,
    updater: Updater,
    scoring_model: ScoringModel,
    params: TOMHTParams,
    tree_store: TrackTreeStore,
    assoc_miss_label: int,
    expansion_call_stats: ExpansionCallStats,
) -> list[LocalChildCandidate]:
    """Build retained local continuation candidates for one active leaf."""
    track = reconstruct_track_from_leaf_node(
        leaf_node,
        lifecycle_state=tree.lifecycle_state,
        publication_state=tree.publication_state,
        public_track_id=tree.public_track_id,
    )
    hypothesise_start_ns = start_timer()
    raw_hypotheses = hypothesiser.hypothesise(track, ctx.detections, ctx.timestamp)
    expansion_call_stats.hypothesise_calls += 1
    expansion_call_stats.hypothesise_wall_ns += elapsed_ns(hypothesise_start_ns)
    if not isinstance(raw_hypotheses, MultipleHypothesis):
        raise TypeError(
            "Distance hypothesiser must return stonesoup MultipleHypothesis."
        )
    hypotheses = [
        validate_distance_hypothesis(hyp) for hyp in raw_hypotheses.single_hypotheses
    ]
    expansion_call_stats.local_child_candidates_total += len(hypotheses)
    local_log_deltas = scoring_model.score_track_hypotheses(
        hypotheses=hypotheses,
        ctx=ctx,
        track_id=tree.public_track_id,
    )
    if len(local_log_deltas) != len(hypotheses):
        raise RuntimeError(
            "score_track_hypotheses must return one score per hypothesis."
        )
    if any(not np.isfinite(float(score)) for score in local_log_deltas):
        raise ValueError("score_track_hypotheses produced a non-finite score.")

    miss_hypotheses = [
        hyp for hyp in hypotheses if isinstance(hyp.measurement, MissedDetection)
    ]
    if len(miss_hypotheses) != 1:
        raise ValueError(
            "Distance hypothesiser must return exactly one missed-detection "
            "SingleDistanceHypothesis."
        )

    scored_rows = list(enumerate(zip(hypotheses, local_log_deltas)))
    miss_row_index = next(
        row_index
        for row_index, (hyp, _) in scored_rows
        if isinstance(hyp.measurement, MissedDetection)
    )

    def _sort_key(
        row: tuple[int, tuple[SingleDistanceHypothesis, float]],
    ) -> tuple[float, int]:
        _, (hyp, score) = row
        if isinstance(hyp.measurement, MissedDetection):
            return (float(score), -1)
        return (
            float(score),
            -ctx.det_index_by_obj.get(id(hyp.measurement), 10**9),
        )

    sorted_rows = sorted(scored_rows, key=_sort_key, reverse=True)
    kept_rows = sorted_rows[: params.max_children_per_leaf]
    kept_row_indices = {row_index for row_index, _ in kept_rows}
    if miss_row_index not in kept_row_indices:
        kept_rows.append(scored_rows[miss_row_index])

    out = [
        candidate_from_distance_hypothesis(
            leaf_node=leaf_node,
            hypothesis=hyp,
            log_delta=float(log_delta),
            ctx=ctx,
            updater=updater,
            tree_store=tree_store,
            assoc_miss_label=assoc_miss_label,
            expansion_call_stats=expansion_call_stats,
        )
        for _, (hyp, log_delta) in kept_rows
    ]
    out.sort(key=lambda c: c.log_delta, reverse=True)
    expansion_call_stats.local_children_created_total += len(out)
    expansion_call_stats.local_miss_children_created += sum(
        1 for cand in out if cand.used_det_key is None
    )
    expansion_call_stats.local_detection_children_created += sum(
        1 for cand in out if cand.used_det_key is not None
    )
    return out


def apply_pre_solve_leaf_cap_guardrail(
    *,
    leaf_node_ids: set[int],
    nodes_by_id: Mapping[int, TrackHypothesisNode],
    params: TOMHTParams,
) -> set[int]:
    """Apply optional local leaf capping only as a pre-solve tractability valve."""
    max_leaves = params.max_leaves_per_track_tree
    if max_leaves is None or len(leaf_node_ids) <= max_leaves:
        return leaf_node_ids

    ranked = sorted(
        (nodes_by_id[node_id] for node_id in leaf_node_ids),
        key=lambda node: (
            float(node.accumulated_log_score),
            -int(node.node_id),
        ),
        reverse=True,
    )
    return {node.node_id for node in ranked[: int(max_leaves)]}


def expand_one_track_tree(
    *,
    tree: TrackTree,
    nodes_by_id: Mapping[int, TrackHypothesisNode],
    ctx: ScanContext,
    hypothesiser: Hypothesiser,
    updater: Updater,
    scoring_model: ScoringModel,
    params: TOMHTParams,
    tree_store: TrackTreeStore,
    assoc_miss_label: int,
    expansion_call_stats: ExpansionCallStats,
) -> None:
    """Expand all active leaves in one tree, then apply pre-solve cap guardrail."""
    new_leaf_ids: set[int] = set()
    active_leaf_ids = sorted(tree.active_leaf_node_ids)
    expansion_call_stats.expanded_leaf_count += len(active_leaf_ids)
    if tree.lifecycle_state == "confirmed":
        expansion_call_stats.expanded_leaves_confirmed += len(active_leaf_ids)
    else:
        expansion_call_stats.expanded_leaves_tentative += len(active_leaf_ids)

    for leaf_id in active_leaf_ids:
        leaf = nodes_by_id[leaf_id]
        candidates = candidates_for_track_leaf(
            leaf_node=leaf,
            tree=tree,
            ctx=ctx,
            hypothesiser=hypothesiser,
            updater=updater,
            scoring_model=scoring_model,
            params=params,
            tree_store=tree_store,
            assoc_miss_label=assoc_miss_label,
            expansion_call_stats=expansion_call_stats,
        )
        for cand in candidates:
            new_leaf_ids.add(cand.child_node.node_id)

    tree.active_leaf_node_ids = apply_pre_solve_leaf_cap_guardrail(
        leaf_node_ids=new_leaf_ids,
        nodes_by_id=nodes_by_id,
        params=params,
    )
    expansion_call_stats.local_children_retained_total += len(tree.active_leaf_node_ids)


def expand_all_track_trees(
    *,
    tree_store: TrackTreeStore,
    ctx: ScanContext,
    hypothesiser: Hypothesiser,
    updater: Updater,
    scoring_model: ScoringModel,
    params: TOMHTParams,
    assoc_miss_label: int,
    expansion_call_stats: ExpansionCallStats,
) -> None:
    """Run local expansion for all current persistent track trees."""
    for tree in tree_store.track_trees_by_track_id.values():
        expand_one_track_tree(
            tree=tree,
            nodes_by_id=tree_store.nodes_by_id,
            ctx=ctx,
            hypothesiser=hypothesiser,
            updater=updater,
            scoring_model=scoring_model,
            params=params,
            tree_store=tree_store,
            assoc_miss_label=assoc_miss_label,
            expansion_call_stats=expansion_call_stats,
        )
