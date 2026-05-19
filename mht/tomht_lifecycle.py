"""Whole-track lifecycle helpers for the track-oriented TOMHT tracker."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
import sys
from typing import Callable

from stonesoup.base import Property
from stonesoup.deleter.base import Deleter
from stonesoup.types.track import Track

from .tomht_model import (
    ClusterRebuildSnapshot,
    GlobalHypothesis,
    TrackHypothesisNode,
    TrackTree,
)
from .tomht_output import reconstruct_track_from_committed_prefix_and_leaf_node
from .tomht_params import TOMHTParams
from .tomht_tree_store import TrackTreeStore
from .tomht_tree_utils import is_descendant_of


class TOMHTMissCountDeleter(Deleter):
    """Internal Stone Soup-style deleter for TOMHT consecutive-miss deletion."""

    if sys.version_info >= (3, 14):
        threshold = Property(
            int,
            doc="Delete when reconstructed track metadata missed_count >= threshold.",
        )
    else:
        threshold: int = Property(
            doc="Delete when reconstructed track metadata missed_count >= threshold."
        )

    def check_for_deletion(self, track: Track, **kwargs) -> bool:
        """Return whether the reconstructed TOMHT track has enough misses."""
        del kwargs
        return int(track.metadata["missed_count"]) >= int(self.threshold)


@dataclass(frozen=True)
class DeleterWithMetadata:
    """Configured deleter plus TOMHT diagnostic metadata."""

    deleter: Deleter
    reason: str
    display_name: str | None = None
    miss_threshold: int | None = None


def apply_score_based_track_confirmation(
    *,
    tree_store: TrackTreeStore,
    confirmation_log_odds_threshold: float,
) -> int:
    """Promote tentative trees whose active frontier score crosses threshold."""
    confirmed_count = 0
    threshold = float(confirmation_log_odds_threshold)
    for _, tree in sorted(tree_store.track_trees_by_track_id.items()):
        if tree.lifecycle_state != "tentative":
            continue
        tree_score = tree_store.active_tree_max_accumulated_log_score(tree)
        if tree_score is None:
            continue
        if float(tree_score) >= threshold:
            tree.lifecycle_state = "confirmed"
            confirmed_count += 1
    return confirmed_count


def normalized_track_miss_termination_mode(mode_raw: str) -> str:
    """Normalize and validate track-level miss termination mode."""
    mode = str(mode_raw).strip().lower()
    valid = {"all_active_leaves", "map_leaf", "global_k_leaves"}
    if mode not in valid:
        raise ValueError(
            "Invalid TOMHTParams.track_miss_termination_mode. "
            f"Expected one of {sorted(valid)}, got {mode_raw!r}."
        )
    return mode


def effective_track_miss_threshold(*, params: TOMHTParams) -> int:
    """Track-level miss termination threshold with N-scan safety floor."""
    return max(
        int(params.max_missed),
        int(params.ns_scan_window) + 1,
    )


def resolve_deleter_with_metadata(
    *,
    params: TOMHTParams,
    deleter: Deleter | None,
) -> DeleterWithMetadata:
    """Resolve caller/default deleter configuration and diagnostics."""
    if deleter is not None:
        return DeleterWithMetadata(
            deleter=deleter,
            reason="deleter",
            display_name=type(deleter).__name__,
        )

    threshold = effective_track_miss_threshold(params=params)
    return DeleterWithMetadata(
        deleter=TOMHTMissCountDeleter(threshold=threshold),
        reason="miss",
        miss_threshold=threshold,
    )


def track_miss_termination_leaves(
    *,
    track_id: int,
    tree: TrackTree,
    tree_store: TrackTreeStore,
    mode: str,
    map_global: GlobalHypothesis,
    cluster_snapshots: list[ClusterRebuildSnapshot],
) -> list[TrackHypothesisNode]:
    """Return leaves to evaluate for configured deleter deletion."""
    nodes_by_id = tree_store.nodes_by_id
    root = nodes_by_id[tree.root_node_id]

    if mode == "map_leaf":
        map_leaf = map_global.leaf_nodes_by_track_id.get(track_id)
        if map_leaf is not None and is_descendant_of(
            node=map_leaf,
            ancestor=root,
        ):
            return [map_leaf]

    if mode == "global_k_leaves":
        candidate_node_ids: set[int] = set()
        for snapshot in cluster_snapshots:
            if track_id not in snapshot.track_ids:
                continue
            for rebuilt_global in snapshot.rebuilt_globals:
                leaf = rebuilt_global.leaf_nodes_by_track_id.get(track_id)
                if leaf is not None:
                    candidate_node_ids.add(int(leaf.node_id))

        candidate_leaves: list[TrackHypothesisNode] = []
        for node_id in sorted(candidate_node_ids):
            leaf = nodes_by_id.get(node_id)
            if leaf is None:
                continue
            if is_descendant_of(node=leaf, ancestor=root):
                candidate_leaves.append(leaf)
        if candidate_leaves:
            return candidate_leaves

    # Default and safe fallback for empty map/global-k sets.
    return [nodes_by_id[node_id] for node_id in sorted(tree.active_leaf_node_ids)]


def filter_map_global_to_live_trees(
    *,
    map_global: GlobalHypothesis,
    tree_store: TrackTreeStore,
) -> GlobalHypothesis:
    """Drop map entries for tracks that no longer have active trees."""
    track_trees_by_track_id = tree_store.track_trees_by_track_id
    filtered_nodes = {
        track_id: leaf
        for track_id, leaf in map_global.leaf_nodes_by_track_id.items()
        if track_id in track_trees_by_track_id
    }
    return GlobalHypothesis(
        leaf_nodes_by_track_id=filtered_nodes,
        log_weight=float(map_global.log_weight),
    )


def add_track_termination_reason(
    termination_reasons_by_track_id: dict[int, set[str]],
    *,
    track_id: int,
    reason: str,
) -> None:
    """Record one deterministic whole-track termination reason."""
    termination_reasons_by_track_id.setdefault(int(track_id), set()).add(reason)


def collect_score_track_termination_reasons(
    *,
    tree_store: TrackTreeStore,
    deletion_log_odds_threshold: float,
) -> dict[int, set[str]]:
    """Return whole-tree score deletions from max active-leaf score."""
    threshold = float(deletion_log_odds_threshold)
    termination_reasons_by_track_id: dict[int, set[str]] = {}
    for track_id, tree in sorted(tree_store.track_trees_by_track_id.items()):
        tree_score = tree_store.active_tree_max_accumulated_log_score(tree)
        if tree_score is None:
            continue
        if float(tree_score) <= threshold:
            add_track_termination_reason(
                termination_reasons_by_track_id,
                track_id=track_id,
                reason="score",
            )
    return termination_reasons_by_track_id


def format_track_termination_reasons(
    termination_reasons_by_track_id: dict[int, set[str]],
) -> str:
    """Format track termination reasons in stable reason/track order."""
    parts: list[str] = []
    for reason in ("score", "miss", "deleter"):
        track_ids = [
            track_id
            for track_id in sorted(termination_reasons_by_track_id)
            if reason in termination_reasons_by_track_id[track_id]
        ]
        if track_ids:
            parts.append(f"{reason}:{track_ids}")
    return ";".join(parts)


def remove_terminated_track_trees(
    *,
    tree_store: TrackTreeStore,
    termination_reasons_by_track_id: dict[int, set[str]],
    mode: str,
    deletion_log_odds_threshold: float,
    deleter_with_metadata: DeleterWithMetadata,
) -> None:
    """Remove terminated trees and emit one deterministic lifecycle diagnostic."""
    if not termination_reasons_by_track_id:
        return

    terminated_track_ids = sorted(termination_reasons_by_track_id)
    for track_id in terminated_track_ids:
        tree_store.track_trees_by_track_id.pop(track_id, None)

    diagnostic_parts = [
        "TRACK_LIFECYCLE",
        f"mode={mode}",
        f"score_threshold={deletion_log_odds_threshold:.6g}",
    ]
    if deleter_with_metadata.miss_threshold is not None:
        diagnostic_parts.append(
            f"miss_threshold={deleter_with_metadata.miss_threshold}"
        )
    if deleter_with_metadata.reason == "deleter":
        diagnostic_parts.append(f"deleter={deleter_with_metadata.display_name}")
    diagnostic_parts.append(f"terminated={terminated_track_ids}")
    diagnostic_parts.append(
        "reasons=" + format_track_termination_reasons(termination_reasons_by_track_id)
    )
    print(" ".join(diagnostic_parts))


def internal_track_id_for_deleter_candidate(
    leaf_node: TrackHypothesisNode,
) -> object:
    """Return the Stone Soup Track.id used for deleter candidate evaluation."""
    return int(leaf_node.track_id)


def apply_post_n_scan_track_lifecycle(
    *,
    tree_store: TrackTreeStore,
    map_global: GlobalHypothesis,
    cluster_snapshots: list[ClusterRebuildSnapshot],
    params: TOMHTParams,
    deletion_log_odds_threshold: float,
    deleter_with_metadata: DeleterWithMetadata,
    output_track_id_for_deleter: Callable[[TrackHypothesisNode], object | None],
    timestamp: datetime.datetime,
) -> GlobalHypothesis:
    """Apply post-N-scan score deletion plus the configured deleter."""
    mode = normalized_track_miss_termination_mode(params.track_miss_termination_mode)

    termination_reasons_by_track_id = collect_score_track_termination_reasons(
        tree_store=tree_store,
        deletion_log_odds_threshold=deletion_log_odds_threshold,
    )
    track_trees_by_track_id = tree_store.track_trees_by_track_id
    for track_id, tree in sorted(track_trees_by_track_id.items()):
        leaves = track_miss_termination_leaves(
            track_id=track_id,
            tree=tree,
            tree_store=tree_store,
            mode=mode,
            map_global=map_global,
            cluster_snapshots=cluster_snapshots,
        )
        if not leaves:
            continue

        committed_states = list(tree.committed_states)
        leaf_delete_decisions: list[bool] = []
        for leaf in leaves:
            candidate_track = reconstruct_track_from_committed_prefix_and_leaf_node(
                committed_states=committed_states,
                leaf_node=leaf,
                output_track_id=output_track_id_for_deleter(leaf),
                lifecycle_state=tree.lifecycle_state,
                public_track_id=tree.public_track_id,
            )
            should_delete = bool(
                deleter_with_metadata.deleter.check_for_deletion(
                    candidate_track,
                    timestamp=timestamp,
                )
            )
            leaf_delete_decisions.append(should_delete)
        if leaf_delete_decisions and all(leaf_delete_decisions):
            add_track_termination_reason(
                termination_reasons_by_track_id,
                track_id=track_id,
                reason=deleter_with_metadata.reason,
            )

    remove_terminated_track_trees(
        tree_store=tree_store,
        termination_reasons_by_track_id=termination_reasons_by_track_id,
        mode=mode,
        deletion_log_odds_threshold=deletion_log_odds_threshold,
        deleter_with_metadata=deleter_with_metadata,
    )

    tree_store.remove_empty_trees()
    return filter_map_global_to_live_trees(
        map_global=map_global,
        tree_store=tree_store,
    )
