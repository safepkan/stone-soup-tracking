"""Stone Soup output-adapter helpers for TOMHT node-based internals."""

from stonesoup.types.track import Track

from mht.tomht_model import TrackHypothesisNode


def lineage_from_leaf_node(leaf_node: TrackHypothesisNode) -> list[TrackHypothesisNode]:
    """Return same-track ancestry from root to ``leaf_node`` (inclusive)."""
    lineage: list[TrackHypothesisNode] = []
    node: TrackHypothesisNode | None = leaf_node
    while node is not None:
        lineage.append(node)
        node = node.parent
    lineage.reverse()
    return lineage


def output_track_metadata_from_leaf_node(
    leaf_node: TrackHypothesisNode,
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
    - The remaining keys are diagnostic/inspection fields describing the current
      leaf node and its cached maintenance/provenance state.

    Keeping this projection explicit makes the Stone Soup boundary easier to
    reason about and avoids carrying opaque metadata through the internal
    node-based hypothesis structure.
    """
    return {
        "track_id": int(leaf_node.track_id),
        "node_id": int(leaf_node.node_id),
        "age": int(leaf_node.age),
        "hits": int(leaf_node.hits),
        "missed_count": int(leaf_node.missed_count),
        "last_det_key": leaf_node.last_det_key,
        "last_det_hit": bool(leaf_node.last_det_hit),
        "root_source": leaf_node.root_source,
        "birth_scan_index": int(leaf_node.birth_scan_index),
    }


def reconstruct_track_from_leaf_node(leaf_node: TrackHypothesisNode) -> Track:
    """
    Reconstruct a Stone Soup Track compatibility view from node ancestry.

    Internal branch identity stays node-based (leaf node IDs + parent links).
    This adapter exists for APIs that currently expect Track instances.
    """
    lineage = lineage_from_leaf_node(leaf_node)
    tr = Track([node.state for node in lineage])
    tr.metadata.update(output_track_metadata_from_leaf_node(leaf_node))
    return tr
