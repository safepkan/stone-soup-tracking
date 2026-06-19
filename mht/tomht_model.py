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
AssociationStatus: TypeAlias = Literal["committed", "tentative"]


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
    # Detection keys cached for conflict checks. This is not a lifetime audit
    # log: tracker-created nodes retain only the N-scan conflict horizon. New
    # descendants omit keys that were already committed when they were created;
    # keys committed later may remain here and are masked by
    # TrackTree.committed_detection_keys until descendants are materialized.
    detection_history_keys: frozenset[DetectionKey]

    age: int
    hits: int
    missed_count: int
    last_det_key: DetectionKey | None
    last_det_hit: bool

    root_source: str
    birth_scan_index: int
    # Caller-facing index from enumerate(detections) in update_tracker(...).
    # This is intentionally separate from DetectionKey.det_index, which is the
    # tracker-internal sorted scan index used for deterministic solving.
    used_input_det_index: int | None = None

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
    caller_metadata: dict[str, object] = field(default_factory=dict)
    committed_states: list[State] = field(default_factory=list)
    # Bounded masking set for recently committed detection keys that can still
    # appear in retained node histories. This is not a full committed detection
    # history.
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
    det_input_index_by_obj: dict[int, int] = field(default_factory=dict)
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


@dataclass(frozen=True)
class MapAssociationStep:
    """One association decision in a MAP-selected track suffix.

    Public fields: scan_index, timestamp, association_status, and
    input_detection_index.
    """

    scan_index: int
    timestamp: datetime.datetime
    association_status: AssociationStatus
    input_detection_index: int | None

    # Diagnostic fields.
    internal_detection_index: int | None
    node_id: int
    state_kind: str
    detection_key: DetectionKey | None


@dataclass(frozen=True)
class MapTrackAssociationHistory:
    """Association-history suffix for one MAP-selected logical track.

    All fields are part of the public association-history contract. The nested
    MapAssociationStep values carry their own public-vs-diagnostic distinction.
    """

    internal_track_id: int
    public_track_id: object | None
    lifecycle_state: TrackLifecycleState
    publication_state: TrackPublicationState
    committed_boundary_scan_index: int | None
    steps: tuple[MapAssociationStep, ...]


@dataclass(frozen=True)
class MAPAssociationHistorySnapshot:
    """Read-only MAP association-history view for inspection/integration.

    All fields are part of the public association-history contract.
    """

    selection: Literal["map"]
    scan_index: int | None
    timestamp: datetime.datetime | None
    include_unpublished: bool
    histories: tuple[MapTrackAssociationHistory, ...]
