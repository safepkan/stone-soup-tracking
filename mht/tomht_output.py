"""Stone Soup output-adapter helpers for TOMHT node-based internals."""

from math import exp
from typing import Callable

from stonesoup.types.track import Track
from stonesoup.types.state import State

from .tomht_model import (
    GlobalHypothesis,
    MAPHypothesisSnapshot,
    TrackHypothesisNode,
    TrackLifecycleState,
    TrackPublicationState,
    TrackTree,
)
from .tomht_tree_store import TrackTreeStore


class DensePublishedTrackIdMapper:
    """Assign dense public IDs in first-publication order."""

    def __init__(self) -> None:
        self._next_public_track_id = 0

    def __call__(self, internal_track_id: int) -> int:
        del internal_track_id
        public_track_id = self._next_public_track_id
        self._next_public_track_id += 1
        return public_track_id


def resolve_output_track_id_mapper(
    output_track_id_mapper: Callable[[int], object] | None,
) -> Callable[[int], object]:
    """Resolve and validate output Track.id mapping strategy."""
    if output_track_id_mapper is None:
        return DensePublishedTrackIdMapper()
    if not callable(output_track_id_mapper):
        raise TypeError("output_track_id_mapper must be callable when provided.")
    return output_track_id_mapper


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
    public_track_id: object | None = None,
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
    - ``internal_track_id`` is the stable internal TOMHT logical-track
      identifier.
    - ``track_id`` is a deprecated compatibility alias for ``internal_track_id``.
    - ``public_track_id`` mirrors ``Track.id`` for published tracks and is
      ``None`` for unpublished inspection tracks.
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
        "internal_track_id": int(leaf_node.track_id),
        "public_track_id": public_track_id,
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
    output_track_id: object | None = None,
    public_track_id: object | None = None,
    lifecycle_state: TrackLifecycleState | None = None,
    publication_state: TrackPublicationState | None = None,
) -> Track:
    """Reconstruct output track from committed prefix plus unresolved lineage."""
    lineage = lineage_from_leaf_node(leaf_node)
    unresolved_states = [node.state for node in lineage]
    internal_track_id = int(leaf_node.track_id)
    track_id = internal_track_id if output_track_id is None else output_track_id
    tr = Track([*committed_states, *unresolved_states], id=track_id)
    tr.metadata.update(
        output_track_metadata_from_leaf_node(
            leaf_node,
            lifecycle_state=lifecycle_state,
            publication_state=publication_state,
            public_track_id=public_track_id,
        )
    )
    return tr


def ensure_public_track_id(
    *,
    tree: TrackTree,
    output_track_id_mapper: Callable[[int], object],
) -> object:
    """Return an existing public ID, assigning one if publication needs repair."""
    if tree.public_track_id is not None:
        return tree.public_track_id
    public_track_id = output_track_id_mapper(int(tree.track_id))
    if public_track_id is None:
        raise ValueError(
            "output_track_id_mapper returned None for a published TOMHT track; "
            "None is reserved for unpublished inspection tracks."
        )
    tree.public_track_id = public_track_id
    return public_track_id


def map_leaf_satisfies_publication_policy(
    *,
    tree: TrackTree,
    leaf: TrackHypothesisNode,
    publish_lifecycle_states: frozenset[str],
    publish_min_hits: int,
    publish_min_age: int,
    publish_min_existence_log_odds_threshold: float | None,
) -> bool:
    """Return whether a MAP leaf can first transition to published output."""
    if tree.lifecycle_state not in publish_lifecycle_states:
        return False
    if int(leaf.hits) < int(publish_min_hits):
        return False
    if int(leaf.age) < int(publish_min_age):
        return False

    threshold = publish_min_existence_log_odds_threshold
    if threshold is not None and float(leaf.accumulated_log_score) < threshold:
        return False
    return True


def apply_output_publication(
    *,
    tree_store: TrackTreeStore,
    map_global: GlobalHypothesis,
    publish_lifecycle_states: frozenset[str],
    publish_min_hits: int,
    publish_min_age: int,
    publish_min_existence_log_odds_threshold: float | None,
    output_track_id_mapper: Callable[[int], object],
) -> int:
    """Stickily publish MAP-selected trees that satisfy output policy."""
    published_count = 0
    track_trees_by_track_id = tree_store.track_trees_by_track_id
    for track_id, leaf in sorted(map_global.leaf_nodes_by_track_id.items()):
        tree = track_trees_by_track_id.get(track_id)
        if tree is None:
            continue
        if tree.publication_state == "published":
            ensure_public_track_id(
                tree=tree,
                output_track_id_mapper=output_track_id_mapper,
            )
            continue
        if not map_leaf_satisfies_publication_policy(
            tree=tree,
            leaf=leaf,
            publish_lifecycle_states=publish_lifecycle_states,
            publish_min_hits=publish_min_hits,
            publish_min_age=publish_min_age,
            publish_min_existence_log_odds_threshold=(
                publish_min_existence_log_odds_threshold
            ),
        ):
            continue
        ensure_public_track_id(
            tree=tree,
            output_track_id_mapper=output_track_id_mapper,
        )
        tree.publication_state = "published"
        published_count += 1
    return published_count


def reconstruct_map_output_tracks(
    *,
    tree_store: TrackTreeStore,
    map_snapshot: MAPHypothesisSnapshot | None,
    include_unpublished: bool,
    output_track_id_mapper: Callable[[int], object],
) -> set[Track]:
    """Reconstruct current MAP outputs as Stone Soup ``Track`` objects."""
    if map_snapshot is None:
        return set()

    output_tracks: set[Track] = set()
    for leaf_node in map_snapshot.leaf_nodes_by_track_id.values():
        tree = tree_store.track_trees_by_track_id.get(int(leaf_node.track_id))
        if tree is None:
            continue
        is_published = tree.publication_state == "published"
        if not include_unpublished and not is_published:
            continue
        if is_published:
            public_track_id = ensure_public_track_id(
                tree=tree,
                output_track_id_mapper=output_track_id_mapper,
            )
            output_track_id = public_track_id
        else:
            public_track_id = None
            output_track_id = int(leaf_node.track_id)
        committed_states = list(tree.committed_states)
        output_tracks.add(
            reconstruct_track_from_committed_prefix_and_leaf_node(
                committed_states=committed_states,
                leaf_node=leaf_node,
                output_track_id=output_track_id,
                lifecycle_state=tree.lifecycle_state,
                publication_state=tree.publication_state,
                public_track_id=public_track_id,
            )
        )
    return output_tracks
