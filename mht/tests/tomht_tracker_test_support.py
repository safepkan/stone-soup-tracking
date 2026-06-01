from __future__ import annotations

import datetime
from math import exp, log, log1p
import sys
from typing import Iterable, cast

import numpy as np
from stonesoup.base import Property
from stonesoup.deleter.base import Deleter
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import Update

from mht.tomht_model import (
    ClusterRebuildSnapshot,
    DetectionKey,
    GlobalHypothesis,
    TrackHypothesisNode,
)
from mht.tomht_clustering import ClusterWorkItem
from mht.tomht_lifecycle import LifecycleDeleterStats
from mht.tomht_scoring import DetectionProbabilityModel
from mht.tomht_tree_store import TrackTreeStore
from mht.tomht_tree_utils import live_conflict_keys_for_leaf
from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class _ScriptedHypothesiser:
    """Per-(timestamp,track_id) scripted local hypotheses for deterministic tests."""

    def __init__(self) -> None:
        self.predictor = _NoopPredictor()
        self._script: dict[
            tuple[datetime.datetime, int], list[tuple[int | None, float]]
        ] = {}

    def set_options(
        self,
        *,
        timestamp: datetime.datetime,
        track_id: int,
        options: list[tuple[int | None, float]],
    ) -> None:
        self._script[(timestamp, track_id)] = list(options)

    def hypothesise(
        self,
        track: Track,
        detections: Iterable[Detection],
        timestamp: datetime.datetime,
        **kwargs,
    ) -> MultipleHypothesis:
        del kwargs
        detections_list = list(detections)
        track_id = int(track.metadata["track_id"])
        options = self._script.get((timestamp, track_id), [(None, 0.0)])
        prediction = cast(GaussianState, track.states[-1])

        out: list[SingleDistanceHypothesis] = []
        for det_index, log_delta in options:
            if det_index is None:
                out.append(
                    SingleDistanceHypothesis(
                        prediction=prediction,
                        measurement=MissedDetection(timestamp=timestamp),
                        distance=-float(log_delta),
                    )
                )
                continue

            detection = detections_list[int(det_index)]
            out.append(
                SingleDistanceHypothesis(
                    prediction=prediction,
                    measurement=detection,
                    distance=-float(log_delta),
                )
            )
        return MultipleHypothesis(out, normalise=False)


class _RecordingMetadataHypothesiser(_ScriptedHypothesiser):
    """Scripted hypothesiser that records TOMHT track metadata at expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.track_metadata: list[dict[str, object]] = []
        self.track_objects: list[Track] = []

    def hypothesise(
        self,
        track: Track,
        detections: Iterable[Detection],
        timestamp: datetime.datetime,
        **kwargs,
    ) -> MultipleHypothesis:
        self.track_objects.append(track)
        self.track_metadata.append(dict(track.metadata))
        return super().hypothesise(track, detections, timestamp, **kwargs)


class _ScriptedPredictingHypothesiser(_ScriptedHypothesiser):
    """Scripted hypotheses that build fresh per-scan prediction states."""

    def hypothesise(
        self,
        track: Track,
        detections: Iterable[Detection],
        timestamp: datetime.datetime,
        **kwargs,
    ) -> MultipleHypothesis:
        del kwargs
        detections_list = list(detections)
        track_id = int(track.metadata["track_id"])
        options = self._script.get((timestamp, track_id), [(None, 0.0)])
        prior = track.states[-1]
        prediction = GaussianState(
            state_vector=np.asarray(prior.state_vector, dtype=float).copy(),
            covar=np.asarray(prior.covar, dtype=float).copy(),
            timestamp=timestamp,
        )

        out: list[SingleDistanceHypothesis] = []
        for det_index, log_delta in options:
            if det_index is None:
                out.append(
                    SingleDistanceHypothesis(
                        prediction=prediction,
                        measurement=MissedDetection(timestamp=timestamp),
                        distance=-float(log_delta),
                    )
                )
                continue

            detection = detections_list[int(det_index)]
            out.append(
                SingleDistanceHypothesis(
                    prediction=prediction,
                    measurement=detection,
                    distance=-float(log_delta),
                )
            )
        return MultipleHypothesis(out, normalise=False)


class _NoopPredictor:
    def predict(self, prior, timestamp=None, **kwargs):
        del prior, timestamp, kwargs
        raise RuntimeError("No prediction expected for scripted hypotheses")


class _RecordingPriorPredictor:
    def __init__(self) -> None:
        self.priors: list[object] = []

    def predict(self, prior, timestamp=None, **kwargs):
        del kwargs
        self.priors.append(prior)
        return GaussianState(
            state_vector=np.asarray(prior.state_vector, dtype=float).copy(),
            covar=np.asarray(prior.covar, dtype=float).copy(),
            timestamp=timestamp,
        )


class _ScriptedUpdater:
    def update(self, hypothesis) -> GaussianState:
        measurement = getattr(hypothesis, "measurement", None)
        if not isinstance(measurement, Detection):
            raise RuntimeError("Detection-associated update expected.")
        timestamp = measurement.timestamp
        if timestamp is None:
            prediction = getattr(hypothesis, "prediction", None)
            timestamp = getattr(prediction, "timestamp", None)
        if timestamp is None:
            raise RuntimeError("Timestamp required for scripted updater.")
        return _state_from_detection(measurement, timestamp)


class _ScriptedUpdaterWithUpdateStates(_ScriptedUpdater):
    def update(self, hypothesis):
        measurement = getattr(hypothesis, "measurement", None)
        if not isinstance(measurement, Detection):
            raise RuntimeError("Detection-associated update expected.")
        timestamp = measurement.timestamp
        if timestamp is None:
            prediction = getattr(hypothesis, "prediction", None)
            timestamp = getattr(prediction, "timestamp", None)
        if timestamp is None:
            raise RuntimeError("Timestamp required for scripted updater.")
        posterior_state = _state_from_detection(measurement, timestamp)
        return Update.from_state(
            posterior_state,
            hypothesis=hypothesis,
            timestamp=timestamp,
        )


class _CaptureInitiator:
    def __init__(self, born: list[Track]) -> None:
        self._born = born
        self.last_received: list[Detection] = []

    def initiate(self, detections, timestamp):
        del timestamp
        self.last_received = list(detections)
        return list(self._born)


class _CaptureDetectionProbabilityModel:
    def __init__(self, *, clutter_density: float) -> None:
        self._clutter_density = clutter_density
        self.detection_probability_calls: list[
            tuple[object | None, object, object | None]
        ] = []
        self.clutter_density_calls: list[
            tuple[object, Detection | None, object | None]
        ] = []

    def detection_probability(
        self,
        *,
        track_id: object | None,
        prediction,
        caller_scan_context: object | None,
    ) -> float:
        self.detection_probability_calls.append(
            (track_id, prediction, caller_scan_context)
        )
        return 0.0

    def clutter_density(
        self,
        *,
        prediction,
        detection: Detection | None,
        caller_scan_context: object | None,
    ) -> float:
        self.clutter_density_calls.append((prediction, detection, caller_scan_context))
        return self._clutter_density


class _NeutralDetectionProbabilityModel:
    def __init__(self, *, clutter_density: float) -> None:
        self._clutter_density = clutter_density

    def detection_probability(
        self,
        *,
        track_id: object | None,
        prediction,
        caller_scan_context: object | None,
    ) -> float:
        del track_id, prediction, caller_scan_context
        return 0.0

    def clutter_density(
        self,
        *,
        prediction,
        detection: Detection | None,
        caller_scan_context: object | None,
    ) -> float:
        del prediction, detection, caller_scan_context
        return self._clutter_density


class _MetadataMissCountDeleter(Deleter):
    if sys.version_info >= (3, 14):
        threshold = Property(
            int, default=1, doc="Delete when metadata missed_count >= threshold."
        )
    else:
        threshold: int = Property(
            default=1, doc="Delete when metadata missed_count >= threshold."
        )

    def check_for_deletion(self, track: Track, **kwargs) -> bool:
        del kwargs
        misses = int(track.metadata.get("missed_count", 0))
        return misses >= int(self.threshold)


class _RecordingMetadataMissCountDeleter(_MetadataMissCountDeleter):
    if sys.version_info >= (3, 14):
        track_ids = Property(list, default=None, doc="Recorded candidate track IDs.")
        missed_counts = Property(
            list, default=None, doc="Recorded candidate missed counts."
        )
        track_state_counts = Property(
            list, default=None, doc="Recorded candidate track state counts."
        )
        track_is_track = Property(
            list, default=None, doc="Recorded candidate Track type checks."
        )
    else:
        track_ids: list[object] | None = Property(
            default=None, doc="Recorded candidate track IDs."
        )
        missed_counts: list[int] | None = Property(
            default=None, doc="Recorded candidate missed counts."
        )
        track_state_counts: list[int] | None = Property(
            default=None, doc="Recorded candidate track state counts."
        )
        track_is_track: list[bool] | None = Property(
            default=None, doc="Recorded candidate Track type checks."
        )

    def check_for_deletion(self, track: Track, **kwargs) -> bool:
        del kwargs
        if self.track_ids is None:
            self.track_ids = []
        if self.missed_counts is None:
            self.missed_counts = []
        if self.track_state_counts is None:
            self.track_state_counts = []
        if self.track_is_track is None:
            self.track_is_track = []
        self.track_ids.append(track.id)
        missed_count = int(track.metadata["missed_count"])
        self.missed_counts.append(missed_count)
        self.track_state_counts.append(len(track.states))
        self.track_is_track.append(isinstance(track, Track))
        return missed_count >= int(self.threshold)


def _state(x: float, timestamp: datetime.datetime) -> GaussianState:
    return GaussianState(
        [x, 0.0, x, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )


def _state_from_detection(
    det: Detection, timestamp: datetime.datetime
) -> GaussianState:
    vec = np.asarray(det.state_vector, dtype=float).reshape(-1)
    x = float(vec[0])
    y = float(vec[1])
    return GaussianState(
        [x, 0.0, y, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )


def _detection(x: float, y: float, timestamp: datetime.datetime) -> Detection:
    return Detection(np.array([[x], [y]]), timestamp=timestamp)


def _track_start(x: float, timestamp: datetime.datetime) -> Track:
    return Track([_state(x, timestamp)])


def _sigmoid(log_odds: float) -> float:
    return 1.0 / (1.0 + exp(-log_odds))


def _logit(probability: float) -> float:
    return log(probability) - log1p(-probability)


def _build_tracker(
    *,
    hypothesiser: _ScriptedHypothesiser,
    updater: _ScriptedUpdater,
    initiator: SimpleMeasurementInitiator | None = None,
    deleter: Deleter | None = None,
    params: TOMHTParams | None = None,
    detection_probability_model: DetectionProbabilityModel | None = None,
) -> TOMHTTracker:
    if params is None:
        params = TOMHTParams(
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )
    if detection_probability_model is None:
        # Test scripts encode hit scores in the hypothesiser distance and expect
        # miss alternatives to be score-neutral. A dynamic DPM with P_D = 0 keeps
        # that behavior, and its clutter density matches log_epsilon so the hit
        # term reduces to -NLL. This avoids relying on the scalar scoring params,
        # which are ignored once a DPM is supplied.
        detection_probability_model = _NeutralDetectionProbabilityModel(
            clutter_density=params.log_epsilon,
        )
    return TOMHTTracker(
        hypothesiser=hypothesiser,
        updater=updater,
        initiator=initiator,
        deleter=deleter,
        params=params,
        detection_probability_model=detection_probability_model,
    )


def _initiator_birth_root_for_metadata(
    metadata: dict[str, object],
    *,
    default_probability: float = 0.8,
) -> TrackHypothesisNode:
    timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
    birth = _track_start(99.0, timestamp)
    birth.metadata.update(metadata)
    capture_initiator = _CaptureInitiator([birth])
    tracker = _build_tracker(
        hypothesiser=_ScriptedHypothesiser(),
        updater=_ScriptedUpdater(),
        initiator=cast(SimpleMeasurementInitiator, capture_initiator),
        params=TOMHTParams(
            initiator_start_initial_existence_probability=default_probability,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        ),
    )

    tracker.update_tracker(timestamp, [_detection(1.0, 1.0, timestamp)])

    tree = next(iter(tracker.track_trees_by_track_id.values()))
    return tracker.nodes_by_id[tree.root_node_id]


def _run_post_n_scan_lifecycle(
    tracker: TOMHTTracker,
    *,
    timestamp: datetime.datetime,
    map_global: GlobalHypothesis | None = None,
    cluster_snapshots: list[ClusterRebuildSnapshot] | None = None,
    lifecycle_deleter_stats: LifecycleDeleterStats | None = None,
) -> GlobalHypothesis:
    if map_global is None:
        map_global = tracker._last_map_global
    if cluster_snapshots is None:
        cluster_snapshots = []
    scan_index = (
        0 if tracker._last_scan_index is None else int(tracker._last_scan_index)
    )
    filtered = tracker._apply_post_n_scan_track_lifecycle(
        map_global=map_global,
        cluster_snapshots=cluster_snapshots,
        scan_index=scan_index,
        timestamp=timestamp,
        lifecycle_deleter_stats=lifecycle_deleter_stats,
    )
    tracker._last_map_global = filtered
    return filtered


def _set_track_active_leaf_scores(
    tracker: TOMHTTracker,
    *,
    track_id: int,
    scores: list[float],
) -> list[TrackHypothesisNode]:
    tree = tracker.track_trees_by_track_id[track_id]
    leaf_ids = sorted(tree.active_leaf_node_ids)
    if len(leaf_ids) != len(scores):
        raise AssertionError(
            f"Expected {len(scores)} active leaves for track {track_id}, "
            f"got {len(leaf_ids)}."
        )
    leaves = [tracker.nodes_by_id[leaf_id] for leaf_id in leaf_ids]
    for leaf, score in zip(leaves, scores):
        leaf.accumulated_log_score = float(score)
    return leaves


def _replace_active_leaves_with_scores(
    tracker: TOMHTTracker,
    *,
    track_id: int,
    scores: list[float],
    timestamp: datetime.datetime,
) -> list[TrackHypothesisNode]:
    tree = tracker.track_trees_by_track_id[track_id]
    root = tracker.nodes_by_id[tree.root_node_id]
    leaves: list[TrackHypothesisNode] = []
    for score in scores:
        leaf = tracker._tree_store.create_track_hypothesis_node(
            track_id=track_id,
            parent=root,
            scan_index=int(root.scan_index) + 1,
            timestamp=timestamp,
            state=root.state,
            state_kind="manual_test_leaf",
            used_det_key=None,
            assoc_label=TOMHTTracker.ASSOC_MISS,
            log_delta=float(score) - float(root.accumulated_log_score),
            age=int(root.age) + 1,
            hits=int(root.hits),
            missed_count=0,
            last_det_key=None,
            last_det_hit=False,
            root_source=root.root_source,
            birth_scan_index=root.birth_scan_index,
        )
        leaves.append(leaf)
    tree.active_leaf_node_ids = {leaf.node_id for leaf in leaves}
    return leaves


def _single_track_rebuild_snapshot(
    *,
    track_id: int,
    supported_leaves: list[TrackHypothesisNode],
    overload_split_origin_cluster_id: int | None,
) -> ClusterRebuildSnapshot:
    rebuilt_globals = tuple(
        GlobalHypothesis(
            leaf_nodes_by_track_id={track_id: leaf},
            log_weight=float(leaf.accumulated_log_score),
        )
        for leaf in supported_leaves
    )
    map_global = rebuilt_globals[0] if rebuilt_globals else None
    return ClusterRebuildSnapshot(
        cluster_id=101,
        track_ids=(track_id,),
        current_scan_conflict_det_keys=frozenset(),
        conflict_links=(),
        rebuilt_globals=rebuilt_globals,
        map_global=map_global,
        feasible_combinations=len(rebuilt_globals),
        evaluated_combinations=len(rebuilt_globals),
        overload_split_origin_cluster_id=overload_split_origin_cluster_id,
    )


def _tracker_with_two_miss_candidate_leaves(
    *,
    mode: str,
    deleter: Deleter | None = None,
) -> tuple[
    TOMHTTracker,
    datetime.datetime,
    list[TrackHypothesisNode],
    GlobalHypothesis,
    list[ClusterRebuildSnapshot],
]:
    timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
    tracker = _build_tracker(
        hypothesiser=_ScriptedHypothesiser(),
        updater=_ScriptedUpdater(),
        deleter=deleter,
        params=TOMHTParams(
            max_missed=1,
            ns_scan_window=0,
            track_miss_termination_mode=mode,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        ),
    )
    tracker.update_tracker(timestamp, [])
    tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
    leaves = _replace_active_leaves_with_scores(
        tracker,
        track_id=0,
        scores=[10.0, 9.0],
        timestamp=timestamp,
    )
    leaves[0].missed_count = 1
    leaves[1].missed_count = 0
    map_global = GlobalHypothesis(
        leaf_nodes_by_track_id={0: leaves[0]},
        log_weight=float(leaves[0].accumulated_log_score),
    )
    cluster_snapshots = [
        _single_track_rebuild_snapshot(
            track_id=0,
            supported_leaves=[leaves[0]],
            overload_split_origin_cluster_id=None,
        )
    ]
    return tracker, timestamp, leaves, map_global, cluster_snapshots


def _tracker_with_manual_frontier() -> tuple[TOMHTTracker, list[TrackHypothesisNode]]:
    timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
    tracker = _build_tracker(
        hypothesiser=_ScriptedHypothesiser(),
        updater=_ScriptedUpdater(),
        params=TOMHTParams(
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        ),
    )
    tracker.update_tracker(timestamp, [])
    tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
    leaves = _replace_active_leaves_with_scores(
        tracker,
        track_id=0,
        scores=[10.0, 5.0, 1.0],
        timestamp=timestamp,
    )
    return tracker, leaves


def _add_manual_tree_with_live_options(
    store: TrackTreeStore,
    *,
    root_x: float,
    live_options: list[tuple[DetectionKey | None, float]],
) -> tuple[int, list[TrackHypothesisNode]]:
    t1 = datetime.datetime(2026, 3, 28, 10, 0, 1)
    t2 = t1 + datetime.timedelta(seconds=1)
    track_id = store.allocate_track_id()
    root_key = DetectionKey(scan_index=1, det_index=track_id)
    root = store.create_root_node(
        track_id=track_id,
        scan_index=1,
        timestamp=t1,
        state=_state(root_x, t1),
        state_kind="manual_root",
        used_det_key=root_key,
        assoc_label=int(root_key.det_index),
        log_delta=0.0,
        age=1,
        hits=1,
        root_source="manual",
    )
    tree = store.add_track_tree_for_root(root, root_source="manual")
    tree.committed_detection_keys = frozenset({root_key})

    leaves: list[TrackHypothesisNode] = []
    for option_index, (live_det_key, score) in enumerate(live_options):
        if live_det_key is None:
            assoc_label = TOMHTTracker.ASSOC_MISS
            state = _state(root_x, t2)
            hits = 1
            missed_count = 1
            last_det_key = root.last_det_key
            last_det_hit = False
        else:
            assoc_label = int(live_det_key.det_index)
            state = _state(root_x + float(option_index + 1), t2)
            hits = 2
            missed_count = 0
            last_det_key = live_det_key
            last_det_hit = True

        leaf = store.create_track_hypothesis_node(
            track_id=track_id,
            parent=root,
            scan_index=2,
            timestamp=t2,
            state=state,
            state_kind="manual_live",
            used_det_key=live_det_key,
            assoc_label=assoc_label,
            log_delta=float(score),
            age=2,
            hits=hits,
            missed_count=missed_count,
            last_det_key=last_det_key,
            last_det_hit=last_det_hit,
            root_source=root.root_source,
            birth_scan_index=root.birth_scan_index,
        )
        leaves.append(leaf)

    tree.active_leaf_node_ids = {leaf.node_id for leaf in leaves}
    return track_id, leaves


def _manual_cluster_for_track_ids(
    *,
    store: TrackTreeStore,
    track_ids: tuple[int, ...],
    scan_index: int,
) -> ClusterWorkItem:
    current_scan_keys_by_track_id: dict[int, set[DetectionKey]] = {}
    live_keys_by_track_id: dict[int, set[DetectionKey]] = {}
    for track_id in track_ids:
        tree = store.track_trees_by_track_id[track_id]
        current_scan_keys: set[DetectionKey] = set()
        live_keys: set[DetectionKey] = set()
        for leaf_id in tree.active_leaf_node_ids:
            leaf = store.nodes_by_id[leaf_id]
            live_keys |= set(live_conflict_keys_for_leaf(leaf=leaf, tree=tree))
            if (
                leaf.used_det_key is not None
                and int(leaf.used_det_key.scan_index) == scan_index
            ):
                current_scan_keys.add(leaf.used_det_key)
        current_scan_keys_by_track_id[track_id] = current_scan_keys
        live_keys_by_track_id[track_id] = live_keys

    conflict_links: list[tuple[int, int, tuple[DetectionKey, ...]]] = []
    for index, left_track_id in enumerate(track_ids):
        for right_track_id in track_ids[index + 1 :]:
            shared = (
                live_keys_by_track_id[left_track_id]
                & live_keys_by_track_id[right_track_id]
            )
            if shared:
                conflict_links.append(
                    (left_track_id, right_track_id, tuple(sorted(shared)))
                )

    return ClusterWorkItem(
        cluster_id=0,
        track_ids=track_ids,
        current_scan_det_keys_by_track_id=current_scan_keys_by_track_id,
        conflict_links=tuple(conflict_links),
    )


def _add_manual_tree_with_committed_prefix(
    store: TrackTreeStore,
    *,
    root_x: float,
    live_hit_det_key: DetectionKey | None,
    live_hit_score: float,
    live_miss_score: float | None = None,
) -> tuple[int, TrackHypothesisNode | None, TrackHypothesisNode | None]:
    t1 = datetime.datetime(2026, 3, 28, 10, 0, 1)
    t2 = t1 + datetime.timedelta(seconds=1)
    root = store.create_root_tree_for_new_track(
        scan_index=1,
        timestamp=t1,
        state=_state(root_x, t1),
        state_kind="manual_root",
        used_det_key=DetectionKey(scan_index=1, det_index=0),
        assoc_label=0,
        log_delta=0.0,
        age=1,
        hits=1,
        root_source="manual",
    )
    tree = store.track_trees_by_track_id[root.track_id]
    self_key = DetectionKey(scan_index=1, det_index=0)
    tree.committed_detection_keys = frozenset({self_key})

    hit_leaf: TrackHypothesisNode | None = None
    miss_leaf: TrackHypothesisNode | None = None
    active_leaf_ids: set[int] = set()
    if live_hit_det_key is not None:
        hit_leaf = store.create_track_hypothesis_node(
            track_id=root.track_id,
            parent=root,
            scan_index=2,
            timestamp=t2,
            state=_state(root_x + 1.0, t2),
            state_kind="manual_hit",
            used_det_key=live_hit_det_key,
            assoc_label=int(live_hit_det_key.det_index),
            log_delta=live_hit_score,
            age=2,
            hits=2,
            missed_count=0,
            last_det_key=live_hit_det_key,
            last_det_hit=True,
            root_source=root.root_source,
            birth_scan_index=root.birth_scan_index,
        )
        active_leaf_ids.add(hit_leaf.node_id)
    if live_miss_score is not None:
        miss_leaf = store.create_track_hypothesis_node(
            track_id=root.track_id,
            parent=root,
            scan_index=2,
            timestamp=t2,
            state=_state(root_x, t2),
            state_kind="manual_miss",
            used_det_key=None,
            assoc_label=TOMHTTracker.ASSOC_MISS,
            log_delta=live_miss_score,
            age=2,
            hits=1,
            missed_count=1,
            last_det_key=root.last_det_key,
            last_det_hit=False,
            root_source=root.root_source,
            birth_scan_index=root.birth_scan_index,
        )
        active_leaf_ids.add(miss_leaf.node_id)
    tree.active_leaf_node_ids = active_leaf_ids
    return root.track_id, hit_leaf, miss_leaf
