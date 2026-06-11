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


def trim_detection_history_keys_to_scan_window(
    *,
    history_keys: frozenset[DetectionKey],
    scan_index: int,
    scan_window: int,
) -> frozenset[DetectionKey]:
    """Return detection keys no older than the configured N-scan horizon.

    The cutoff is inclusive. With ``N=1`` and a node at scan ``k``, the node
    keeps keys from scans ``k-1`` and ``k`` because the solver may need the
    boundary scan before the current scan's N-scan commit masks it.
    """
    min_scan_index = int(scan_index) - int(scan_window)
    return trim_detection_history_keys_before_scan(
        history_keys=history_keys,
        min_scan_index=min_scan_index,
    )


def trim_detection_history_keys_before_scan(
    *,
    history_keys: frozenset[DetectionKey],
    min_scan_index: int,
) -> frozenset[DetectionKey]:
    """Return detection keys with scan index at or after ``min_scan_index``."""
    min_scan_index_int = int(min_scan_index)
    if all(int(key.scan_index) >= min_scan_index_int for key in history_keys):
        return history_keys
    return frozenset(
        key for key in history_keys if int(key.scan_index) >= min_scan_index_int
    )


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
