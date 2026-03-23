"""Core passive TO-MHT data structures and read-only snapshots."""

from dataclasses import dataclass
import datetime
from typing import Mapping

from stonesoup.types.state import State


@dataclass(frozen=True)
class TrackHypothesisNode:
    """One node in one logical track's same-track hypothesis ancestry chain.

    Parent links are same-track only. A global hypothesis chooses one active leaf
    per ``track_id``; identity and ancestry come from node linkage, not caches.

    Core structural fields (core hypothesis identity/ancestry/payload):
    - ``node_id``: Unique node identity used for structural references/deduping.
    - ``track_id``: Stable logical track identity this node belongs to.
    - ``parent``: Previous node for the same ``track_id``; ``None`` at roots.
    - ``scan_index``: Scan index where this node's step was created.
    - ``timestamp``: Scan timestamp for this node's step.
    - ``state``: Per-step Stone Soup ``State`` payload used for reconstruction/update flow.
    - ``state_kind``: Step kind tag (for example ``"update"``, ``"prediction"``).
    - ``used_det_key``: Concrete detection index used for exclusivity/scoring;
      ``None`` for miss/no-detection steps.
    - ``assoc_label``: Per-step association label/structural marker (index or
      sentinels such as miss/pad for root-like cases).
    - ``log_delta``: Per-step log-score increment relative to ``parent``.

    Cached operational/convenience fields (tracker-logic support, not identity):
    - ``age``: Total step count accumulated along this track chain.
    - ``hits``: Total detection-hit count accumulated along this chain.
    - ``missed_count``: Current consecutive-miss streak for miss-limit logic.
    - ``last_det_key``: Most recent detection index seen on this chain.
    - ``last_det_hit``: Whether the most recent step was a hit.

    Instrumentation/provenance fields (debug and inspection context):
    - ``root_source``: Origin label (for example internal birth vs external start).
    - ``birth_scan_index``: Scan index where this logical track chain started.

    """

    # Core structural identity + ancestry + per-step payload.
    node_id: int
    track_id: int
    parent: "TrackHypothesisNode | None"
    scan_index: int
    timestamp: datetime.datetime
    state: State
    state_kind: str
    used_det_key: int | None
    assoc_label: int
    log_delta: float

    # Cached operational/convenience metadata for common tracker logic.
    age: int
    hits: int
    missed_count: int
    last_det_key: int | None
    last_det_hit: bool

    # Instrumentation/provenance fields (useful in debug output and snapshots).
    root_source: str
    birth_scan_index: int


@dataclass(frozen=True)
class GlobalHypothesis:
    """One joint hypothesis over currently active logical tracks.

    In this implementation, one global means one active leaf node per logical
    ``track_id`` plus a cumulative score/weight.

    Fields:
    - ``leaf_nodes_by_track_id``: ``{track_id: leaf_node}`` map selecting the
      active leaf for each track in this global. Shared ancestry is carried by
      each leaf node's same-track ``parent`` chain.
    - ``log_weight``: Cumulative log score for this joint assignment.
    """

    leaf_nodes_by_track_id: dict[int, TrackHypothesisNode]
    log_weight: float


@dataclass(frozen=True)
class ChildCandidate:
    track_id: int
    child_node: TrackHypothesisNode
    used_det_key: int | None
    log_delta: float


@dataclass(frozen=True)
class NScanCommitmentSnapshot:
    """Read-only copy of current ancestor-identity N-scan commitment state."""

    boundary_scan_index: int | None
    tracks_in_scope: int
    latest_committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]
    committed_boundary_by_track_id: dict[int, int]
    committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]


@dataclass(frozen=True)
class MAPHypothesisSnapshot:
    """Read-only copy of current MAP global hypothesis in node-native form."""

    log_weight: float
    leaf_nodes_by_track_id: Mapping[int, TrackHypothesisNode]
