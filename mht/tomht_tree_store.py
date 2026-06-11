"""Mutable persistent tree/node store for the track-oriented TOMHT tracker."""

from __future__ import annotations

import datetime
from typing import Iterable

from stonesoup.types.state import State

from .tomht_model import DetectionKey, TrackHypothesisNode, TrackTree
from .tomht_tree_utils import trim_detection_history_keys_to_scan_window


class TrackTreeStore:
    """Own persistent track-tree/node tables and stable ID allocation."""

    def __init__(
        self,
        *,
        detection_history_scan_window: int,
    ) -> None:
        if int(detection_history_scan_window) < 0:
            raise ValueError("detection_history_scan_window must be >= 0.")
        self.detection_history_scan_window = int(detection_history_scan_window)
        self._next_track_id = 0
        self._next_node_id = 0
        self.nodes_by_id: dict[int, TrackHypothesisNode] = {}
        self.track_trees_by_track_id: dict[int, TrackTree] = {}

    def allocate_track_id(self) -> int:
        """Allocate the next stable logical track ID."""
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def allocate_node_id(self) -> int:
        """Allocate the next stable hypothesis-node ID."""
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def register_node(self, node: TrackHypothesisNode) -> TrackHypothesisNode:
        """Store one node in the persistent node table and return it."""
        self.nodes_by_id[node.node_id] = node
        return node

    def create_track_hypothesis_node(
        self,
        *,
        track_id: int,
        parent: TrackHypothesisNode | None,
        scan_index: int,
        timestamp: datetime.datetime,
        state: State,
        state_kind: str,
        used_det_key: DetectionKey | None,
        assoc_label: int,
        log_delta: float,
        age: int,
        hits: int,
        missed_count: int,
        last_det_key: DetectionKey | None,
        last_det_hit: bool,
        root_source: str,
        birth_scan_index: int,
    ) -> TrackHypothesisNode:
        """Create and register one persistent hypothesis node."""
        if parent is not None and parent.track_id != track_id:
            raise ValueError(
                "TrackHypothesisNode parent.track_id must match child track_id."
            )

        if parent is None:
            history_keys: frozenset[DetectionKey]
            if used_det_key is None:
                history_keys = frozenset()
            else:
                history_keys = frozenset({used_det_key})
            accumulated_log_score = float(log_delta)
        else:
            if used_det_key is None:
                history_keys = parent.detection_history_keys
            else:
                history_keys = parent.detection_history_keys | {used_det_key}
            accumulated_log_score = float(parent.accumulated_log_score) + float(
                log_delta
            )

        tree = self.track_trees_by_track_id.get(int(track_id))
        if tree is not None and tree.committed_detection_keys:
            history_keys = frozenset(history_keys - tree.committed_detection_keys)
        history_keys = trim_detection_history_keys_to_scan_window(
            history_keys=history_keys,
            scan_index=scan_index,
            scan_window=self.detection_history_scan_window,
        )

        node = TrackHypothesisNode(
            node_id=self.allocate_node_id(),
            track_id=int(track_id),
            parent=parent,
            scan_index=int(scan_index),
            timestamp=timestamp,
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=int(assoc_label),
            log_delta=float(log_delta),
            accumulated_log_score=float(accumulated_log_score),
            detection_history_keys=history_keys,
            age=int(age),
            hits=int(hits),
            missed_count=int(missed_count),
            last_det_key=last_det_key,
            last_det_hit=bool(last_det_hit),
            root_source=root_source,
            birth_scan_index=int(birth_scan_index),
        )
        self.register_node(node)
        if parent is not None:
            parent.child_node_ids.add(node.node_id)
        return node

    def create_root_node(
        self,
        *,
        track_id: int,
        scan_index: int,
        timestamp: datetime.datetime,
        state: State,
        state_kind: str,
        used_det_key: DetectionKey | None,
        assoc_label: int,
        log_delta: float,
        age: int,
        hits: int,
        root_source: str,
    ) -> TrackHypothesisNode:
        """Create a root node for an internal birth or external start."""
        return self.create_track_hypothesis_node(
            track_id=track_id,
            parent=None,
            scan_index=scan_index,
            timestamp=timestamp,
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=assoc_label,
            log_delta=log_delta,
            age=age,
            hits=hits,
            missed_count=0,
            last_det_key=used_det_key,
            last_det_hit=used_det_key is not None,
            root_source=root_source,
            birth_scan_index=scan_index,
        )

    def add_track_tree(self, tree: TrackTree) -> TrackTree:
        """Insert or replace one persistent track tree."""
        self.track_trees_by_track_id[int(tree.track_id)] = tree
        return tree

    def add_track_tree_for_root(
        self,
        root: TrackHypothesisNode,
        *,
        root_source: str,
    ) -> TrackTree:
        """Insert a single-root tree for a newly created root node."""
        return self.add_track_tree(
            TrackTree(
                track_id=int(root.track_id),
                root_node_id=int(root.node_id),
                active_leaf_node_ids={int(root.node_id)},
                root_source=root_source,
            )
        )

    def create_root_tree_for_new_track(
        self,
        *,
        scan_index: int,
        timestamp: datetime.datetime,
        state: State,
        state_kind: str,
        used_det_key: DetectionKey | None,
        assoc_label: int,
        log_delta: float,
        age: int,
        hits: int,
        root_source: str,
    ) -> TrackHypothesisNode:
        """Create a new logical track root and insert its single-root tree."""
        root = self.create_root_node(
            track_id=self.allocate_track_id(),
            scan_index=scan_index,
            timestamp=timestamp,
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=assoc_label,
            log_delta=log_delta,
            age=age,
            hits=hits,
            root_source=root_source,
        )
        self.add_track_tree_for_root(
            root,
            root_source=root_source,
        )
        return root

    def active_leaf_nodes(self) -> list[TrackHypothesisNode]:
        """Return current active leaves across all persistent track trees."""
        out: list[TrackHypothesisNode] = []
        for tree in self.track_trees_by_track_id.values():
            out.extend(
                self.nodes_by_id[node_id] for node_id in tree.active_leaf_node_ids
            )
        return out

    def active_tree_count(self) -> int:
        """Return the number of currently active track trees."""
        return len(self.track_trees_by_track_id)

    def active_leaf_count(self) -> int:
        """Return the total active leaf count across all track trees."""
        return sum(
            len(tree.active_leaf_node_ids)
            for tree in self.track_trees_by_track_id.values()
        )

    def active_tree_max_accumulated_log_score(
        self,
        tree: TrackTree,
    ) -> float | None:
        """Return max accumulated score over one tree's active leaves."""
        scores = [
            float(self.nodes_by_id[node_id].accumulated_log_score)
            for node_id in sorted(tree.active_leaf_node_ids)
            if node_id in self.nodes_by_id
        ]
        if not scores:
            return None
        return max(scores)

    def remove_empty_trees(self) -> None:
        """Drop any tree that has no surviving active leaves."""
        dead_track_ids = [
            track_id
            for track_id, tree in self.track_trees_by_track_id.items()
            if not tree.active_leaf_node_ids
        ]
        for track_id in dead_track_ids:
            self.track_trees_by_track_id.pop(track_id, None)

    @staticmethod
    def reachable_node_ids_from_seeds(
        seeds: Iterable[TrackHypothesisNode],
    ) -> set[int]:
        """Return node IDs reachable via parent links from supplied seeds."""
        reachable: set[int] = set()
        stack = list(seeds)
        while stack:
            node = stack.pop()
            node_id = int(node.node_id)
            if node_id in reachable:
                continue
            reachable.add(node_id)
            if node.parent is not None:
                stack.append(node.parent)
        return reachable

    def cleanup_unreachable_nodes(
        self,
        *,
        extra_seed_nodes: Iterable[TrackHypothesisNode] = (),
    ) -> None:
        """Reclaim nodes no longer reachable from roots, leaves, or extra seeds."""
        if not self.nodes_by_id:
            return

        seeds: list[TrackHypothesisNode] = []
        for tree in self.track_trees_by_track_id.values():
            seeds.append(self.nodes_by_id[tree.root_node_id])
            seeds.extend(
                self.nodes_by_id[node_id] for node_id in tree.active_leaf_node_ids
            )
        seeds.extend(extra_seed_nodes)

        if not seeds:
            self.nodes_by_id.clear()
            return

        retained_node_ids = self.reachable_node_ids_from_seeds(seeds)
        if len(retained_node_ids) == len(self.nodes_by_id):
            return

        self.nodes_by_id = {
            node_id: node
            for node_id, node in self.nodes_by_id.items()
            if node_id in retained_node_ids
        }
        for node in self.nodes_by_id.values():
            node.child_node_ids = {
                child_id
                for child_id in node.child_node_ids
                if child_id in retained_node_ids
            }
