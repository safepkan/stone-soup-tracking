"""Stone Soup output-adapter helpers for TOMHT node-based internals."""

from collections.abc import Callable
from math import exp

from stonesoup.types.track import Track
from stonesoup.types.state import State

from .tomht_model import (
    TrackHypothesisNode,
    TrackLifecycleState,
    TrackPublicationState,
)


def lineage_from_leaf_node(leaf_node: TrackHypothesisNode) -> list[TrackHypothesisNode]:
    """Return same-track ancestry from root to ``leaf_node`` (inclusive)."""
    lineage: list[TrackHypothesisNode] = []
    node: TrackHypothesisNode | None = leaf_node
    while node is not None:
        lineage.append(node)
        node = node.parent
    lineage.reverse()
    return lineage


def _sigmoid_from_log_odds(log_odds: float) -> float:
    """Convert log-odds to probability without overflow for large scores."""
    x = float(log_odds)
    if x >= 0.0:
        exp_neg = exp(-x)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = exp(x)
    return exp_pos / (1.0 + exp_pos)


def output_track_metadata_from_leaf_node(
    leaf_node: TrackHypothesisNode,
    *,
    lifecycle_state: TrackLifecycleState | None = None,
    publication_state: TrackPublicationState | None = None,
) -> dict[str, object]:
    """
    Build the explicit TOMHT-owned metadata projection for a reconstructed output
    Track.

    This helper defines the metadata contract for Track objects returned from the
    tracker. The metadata is derived from the current leaf node and is intended
    for lightweight observability, debugging, and downstream inspection of the
    MAP output.

    Important:
    - This is an explicit projection from the internal node state.
    - Arbitrary metadata from input birth tracks or external-start tracks is
      intentionally not propagated.
    - ``track_id`` is the stable logical-track identifier exposed on output
      tracks.
    - ``existence_probability`` is currently score-implied from accumulated
      log-odds, not a fully calibrated probability; birth scoring is still
      under review.
    - ``lifecycle_state`` is included when tree-level lifecycle context is
      available at the output boundary.
    - ``publication_state`` is included when tree-level publication context is
      available at the output boundary.
    - The remaining keys are diagnostic/inspection fields describing the current
      leaf node and its cached maintenance/provenance state.

    Keeping this projection explicit makes the Stone Soup boundary easier to
    reason about and avoids carrying opaque metadata through the internal
    node-based hypothesis structure.
    """
    metadata: dict[str, object] = {
        "track_id": int(leaf_node.track_id),
        "node_id": int(leaf_node.node_id),
        "age": int(leaf_node.age),
        "hits": int(leaf_node.hits),
        "missed_count": int(leaf_node.missed_count),
        "last_det_key": leaf_node.last_det_key,
        "last_det_hit": bool(leaf_node.last_det_hit),
        "root_source": leaf_node.root_source,
        "birth_scan_index": int(leaf_node.birth_scan_index),
        "existence_log_odds": float(leaf_node.accumulated_log_score),
        "existence_probability": _sigmoid_from_log_odds(
            leaf_node.accumulated_log_score
        ),
    }
    if lifecycle_state is not None:
        metadata["lifecycle_state"] = lifecycle_state
    if publication_state is not None:
        metadata["publication_state"] = publication_state
    return metadata


def reconstruct_track_from_leaf_node(leaf_node: TrackHypothesisNode) -> Track:
    """
    Reconstruct a Stone Soup Track compatibility view from node ancestry.

    Internal branch identity stays node-based (leaf node IDs + parent links).
    This adapter exists for APIs that currently expect Track instances.
    """
    lineage = lineage_from_leaf_node(leaf_node)
    tr = Track([node.state for node in lineage], id=int(leaf_node.track_id))
    tr.metadata.update(output_track_metadata_from_leaf_node(leaf_node))
    return tr


def reconstruct_track_from_committed_prefix_and_leaf_node(
    *,
    committed_states: list[State],
    leaf_node: TrackHypothesisNode,
    output_track_id_mapper: Callable[[int], object] | None = None,
    lifecycle_state: TrackLifecycleState | None = None,
    publication_state: TrackPublicationState | None = None,
) -> Track:
    """Reconstruct output track from committed prefix plus unresolved lineage."""
    lineage = lineage_from_leaf_node(leaf_node)
    unresolved_states = [node.state for node in lineage]
    internal_track_id = int(leaf_node.track_id)
    mapped_output_track_id = (
        internal_track_id
        if output_track_id_mapper is None
        else output_track_id_mapper(internal_track_id)
    )
    tr = Track([*committed_states, *unresolved_states], id=mapped_output_track_id)
    tr.metadata.update(
        output_track_metadata_from_leaf_node(
            leaf_node,
            lifecycle_state=lifecycle_state,
            publication_state=publication_state,
        )
    )
    return tr
