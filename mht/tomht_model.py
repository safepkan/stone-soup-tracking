"""Core TO-MHT data structures for tree state, per-scan rebuilds, and snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Literal, Mapping, NamedTuple, TypeAlias

from stonesoup.types.detection import Detection
from stonesoup.types.state import State


class DetectionKey(NamedTuple):
    """Detection key used for exclusivity/conflict checks."""

    # Keys are unique across the unresolved tree window and cannot collide when
    # multiple scans are still represented.
    scan_index: int
    det_index: int


TrackLifecycleState: TypeAlias = Literal["tentative", "confirmed"]
TrackPublicationState: TypeAlias = Literal["unpublished", "published"]


@dataclass
class TrackHypothesisNode:
    """One hypothesis node within one logical track tree.

    Nodes are mutable to allow direct child-link maintenance
    (``child_node_ids``). Parent links remain same-track only.
    """

    node_id: int
    track_id: int
    parent: TrackHypothesisNode | None
    scan_index: int
    timestamp: datetime.datetime
    state: State
    state_kind: str
    used_det_key: DetectionKey | None
    assoc_label: int
    log_delta: float
    accumulated_log_score: float
    # Detection keys cached for conflict checks. New descendants omit keys that
    # were already committed when they were created; keys committed later may
    # remain here and are masked by TrackTree.committed_detection_keys.
    detection_history_keys: frozenset[DetectionKey]

    age: int
    hits: int
    missed_count: int
    last_det_key: DetectionKey | None
    last_det_hit: bool

    root_source: str
    birth_scan_index: int

    # Mutable child links for explicit tree navigation / pruning operations.
    child_node_ids: set[int] = field(default_factory=set)


@dataclass
class TrackTree:
    """Persistent logical-track tree/family container."""

    track_id: int
    root_node_id: int
    active_leaf_node_ids: set[int]
    root_source: str
    lifecycle_state: TrackLifecycleState = "tentative"
    publication_state: TrackPublicationState = "unpublished"
    public_track_id: object | None = None
    committed_states: list[State] = field(default_factory=list)
    committed_detection_keys: frozenset[DetectionKey] = field(default_factory=frozenset)


@dataclass(frozen=True)
class GlobalHypothesis:
    """One rebuilt globally consistent leaf selection for a cluster/full scan."""

    leaf_nodes_by_track_id: dict[int, TrackHypothesisNode]
    log_weight: float


@dataclass(frozen=True)
class ScanContext:
    """Internal per-scan TOMHT bookkeeping.

    ``caller_scan_context`` is opaque caller-provided scan data threaded to the
    DetectionProbabilityModel. It is intentionally separate from the internal
    bookkeeping fields in this dataclass.
    """

    scan_index: int
    timestamp: datetime.datetime
    detections: list[Detection]
    det_index_by_obj: dict[int, int]
    caller_scan_context: object | None = None


@dataclass(frozen=True)
class ClusterRebuildSnapshot:
    """Read-only per-cluster rebuild snapshot retained for inspection/debug."""

    cluster_id: int
    track_ids: tuple[int, ...]
    current_scan_conflict_det_keys: frozenset[DetectionKey]
    conflict_links: tuple[tuple[int, int, tuple[DetectionKey, ...]], ...]
    rebuilt_globals: tuple[GlobalHypothesis, ...]
    map_global: GlobalHypothesis | None
    feasible_combinations: int
    evaluated_combinations: int
    overload_split_origin_cluster_id: int | None = None
    map_pruning_child_by_track_id: dict[int, int] = field(default_factory=dict)
    disagreement_count: int = 0


@dataclass(frozen=True)
class NScanCommitmentSnapshot:
    """Read-only MAP-based N-scan pruning bookkeeping snapshot."""

    boundary_scan_index: int | None
    tracks_in_scope: int
    latest_committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]
    committed_boundary_by_track_id: dict[int, int]
    committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]


@dataclass(frozen=True)
class MAPHypothesisSnapshot:
    """Read-only copy of the latest full-scan MAP rebuilt global."""

    log_weight: float
    leaf_nodes_by_track_id: Mapping[int, TrackHypothesisNode]
