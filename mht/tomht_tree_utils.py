"""Pure helpers for TOMHT track-hypothesis tree nodes."""

from __future__ import annotations

from .tomht_model import DetectionKey, TrackHypothesisNode, TrackTree


def live_conflict_keys_for_leaf(
    *,
    leaf: TrackHypothesisNode,
    tree: TrackTree,
) -> frozenset[DetectionKey]:
    """Return unresolved-window conflict keys for one active leaf."""
    return frozenset(leaf.detection_history_keys - tree.committed_detection_keys)


def child_of_root_on_path(
    *,
    root: TrackHypothesisNode,
    leaf: TrackHypothesisNode,
) -> TrackHypothesisNode | None:
    """Return the root child that lies on the root->leaf path."""
    if root.node_id == leaf.node_id:
        return None

    node = leaf
    while node.parent is not None and node.parent.node_id != root.node_id:
        node = node.parent
    if node.parent is None:
        return None
    return node


def is_descendant_of(
    *,
    node: TrackHypothesisNode,
    ancestor: TrackHypothesisNode,
) -> bool:
    """Return whether ``node`` is equal to or below ``ancestor``."""
    cur: TrackHypothesisNode | None = node
    while cur is not None:
        if cur.node_id == ancestor.node_id:
            return True
        cur = cur.parent
    return False
