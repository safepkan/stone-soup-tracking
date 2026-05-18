from __future__ import annotations

import contextlib
from dataclasses import fields, replace
import datetime
import io
from math import exp, log, log1p
import unittest
from typing import Iterable, cast

import numpy as np
from stonesoup.base import Property
from stonesoup.deleter.base import Deleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeDeleter, UpdateTimeStepsDeleter
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
    ScanContext,
    TrackHypothesisNode,
)
from mht.tomht_clustering import ClusterWorkItem, build_track_clusters
from mht.tomht_cluster_rebuild import (
    is_global_feasible_under_live_conflicts,
    rebuild_cluster_globals,
)
from mht.tomht_cluster_solver_factory import make_cluster_solver
from mht.tomht_pruning import apply_post_solve_supported_leaf_pruning
from mht.tomht_scoring import (
    ConstantDetectionProbabilityModel,
    DetectionProbabilityModel,
)
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


class _MetadataMissCountDeleter(Deleter):
    threshold: int = Property(
        default=1, doc="Delete when metadata missed_count >= threshold."
    )

    def check_for_deletion(self, track: Track, **kwargs) -> bool:
        del kwargs
        misses = int(track.metadata.get("missed_count", 0))
        return misses >= int(self.threshold)


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
    params = replace(
        params,
        prob_detect=0.0,
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
    return tracker._nodes_by_id[tree.root_node_id]


def _run_post_n_scan_lifecycle(
    tracker: TOMHTTracker,
    *,
    timestamp: datetime.datetime,
    map_global: GlobalHypothesis | None = None,
) -> GlobalHypothesis:
    if map_global is None:
        map_global = tracker._last_map_global
    scan_index = (
        0 if tracker._last_scan_index is None else int(tracker._last_scan_index)
    )
    filtered = tracker._apply_post_n_scan_track_lifecycle(
        map_global=map_global,
        cluster_snapshots=[],
        scan_index=scan_index,
        timestamp=timestamp,
    )
    tracker._last_map_global = filtered
    tracker.global_hypotheses = [filtered]
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
    leaves = [tracker._nodes_by_id[leaf_id] for leaf_id in leaf_ids]
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
    root = tracker._nodes_by_id[tree.root_node_id]
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


class TOMHTTrackOrientedArchitectureTest(unittest.TestCase):
    def test_explicit_trees_clustering_and_rebuilt_globals(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0), _track_start(10.0, t0)])

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 5.0), (None, 0.0)],
        )
        hypothesiser.set_options(
            timestamp=t1,
            track_id=1,
            options=[(0, 4.0), (1, 3.0), (None, 0.0)],
        )

        det0 = _detection(1.0, 1.0, t1)
        det1 = _detection(2.0, 2.0, t1)
        tracker.update_tracker(t1, [det0, det1])

        self.assertEqual({0, 1}, set(tracker.track_trees_by_track_id.keys()))
        for tree in tracker.track_trees_by_track_id.values():
            root = tracker._nodes_by_id[tree.root_node_id]
            self.assertGreater(len(root.child_node_ids), 0)
            self.assertGreater(len(tree.active_leaf_node_ids), 0)

        clusters = tracker.get_last_cluster_snapshots()
        self.assertEqual(1, len(clusters))
        cluster = clusters[0]
        self.assertEqual((0, 1), cluster.track_ids)
        self.assertGreaterEqual(len(cluster.conflict_links), 1)
        self.assertGreaterEqual(len(cluster.rebuilt_globals), 2)

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(2, len(map_snapshot.leaf_nodes_by_track_id))
        used_det_indices = {
            int(leaf.used_det_key.det_index)
            for leaf in map_snapshot.leaf_nodes_by_track_id.values()
            if leaf.used_det_key is not None
        }
        self.assertEqual({0, 1}, used_det_indices)

        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual(2, len(output_tracks))
        for output_track in output_tracks:
            track_id = int(output_track.metadata["track_id"])
            leaf = map_snapshot.leaf_nodes_by_track_id[track_id]
            existence_log_odds = float(leaf.accumulated_log_score)
            self.assertAlmostEqual(
                existence_log_odds,
                output_track.metadata["existence_log_odds"],
            )
            self.assertAlmostEqual(
                _sigmoid(existence_log_odds),
                output_track.metadata["existence_probability"],
            )

        tree_snapshot = tracker.get_track_tree_snapshot()
        self.assertEqual({0, 1}, set(tree_snapshot.keys()))

    def test_track_tree_committed_detection_keys_start_empty(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual(frozenset(), tree.committed_detection_keys)

    def test_clustering_ignores_committed_prefix_only_conflicts(self) -> None:
        store = TrackTreeStore()
        _add_manual_tree_with_committed_prefix(
            store,
            root_x=0.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=0),
            live_hit_score=5.0,
        )
        _add_manual_tree_with_committed_prefix(
            store,
            root_x=10.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=1),
            live_hit_score=5.0,
        )

        clusters = build_track_clusters(tree_store=store, scan_index=2)

        self.assertEqual([(0,), (1,)], [cluster.track_ids for cluster in clusters])
        self.assertTrue(all(not cluster.conflict_links for cluster in clusters))

    def test_live_unresolved_conflicts_are_enforced_by_cluster_solver(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        track0_id, track0_hit, track0_miss = _add_manual_tree_with_committed_prefix(
            store,
            root_x=0.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=0),
            live_hit_score=10.0,
            live_miss_score=9.0,
        )
        track1_id, track1_hit, _ = _add_manual_tree_with_committed_prefix(
            store,
            root_x=10.0,
            live_hit_det_key=DetectionKey(scan_index=2, det_index=0),
            live_hit_score=10.0,
        )
        self.assertIsNotNone(track0_hit)
        self.assertIsNotNone(track0_miss)
        self.assertIsNotNone(track1_hit)

        clusters = build_track_clusters(tree_store=store, scan_index=2)
        self.assertEqual(1, len(clusters))
        self.assertEqual((track0_id, track1_id), clusters[0].track_ids)
        self.assertEqual(
            ((track0_id, track1_id, ((2, 0),)),), clusters[0].conflict_links
        )

        ctx = ScanContext(
            scan_index=2,
            timestamp=timestamp,
            detections=[_detection(1.0, 1.0, timestamp)],
            det_index_by_obj={},
        )
        snapshots, stats = rebuild_cluster_globals(
            clusters=clusters,
            ctx=ctx,
            tree_store=store,
            params=TOMHTParams(
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        self.assertEqual(1, len(snapshots))
        self.assertEqual(1, stats.feasible_combinations)
        map_global = snapshots[0].map_global
        self.assertIsNotNone(map_global)
        assert map_global is not None
        self.assertIs(map_global.leaf_nodes_by_track_id[track0_id], track0_miss)
        self.assertIs(map_global.leaf_nodes_by_track_id[track1_id], track1_hit)

    def test_overload_solve_returns_one_original_cluster_snapshot(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        track0_id, _ = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        track1_id, _ = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (None, 0.0)],
        )
        clusters = build_track_clusters(tree_store=store, scan_index=2)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            snapshots, stats = rebuild_cluster_globals(
                clusters=clusters,
                ctx=ScanContext(
                    scan_index=2,
                    timestamp=timestamp,
                    detections=[_detection(1.0, 1.0, timestamp)],
                    det_index_by_obj={},
                ),
                tree_store=store,
                params=TOMHTParams(
                    max_global_hypotheses=3,
                    overload_split_projected_combination_threshold=1,
                    overload_split_solution_mode="conditional_exact",
                    debug_display_scan_stats=False,
                    debug_display_hypotheses=False,
                    debug_display_births=False,
                    collect_stats=False,
                ),
                cluster_solver=make_cluster_solver("branch_and_bound"),
            )
        overload_log = stdout.getvalue()

        self.assertEqual(1, len(snapshots))
        snapshot = snapshots[0]
        self.assertEqual((track0_id, track1_id), snapshot.track_ids)
        self.assertIsNone(snapshot.overload_split_origin_cluster_id)
        self.assertGreaterEqual(stats.overload_split_clusters, 1)
        self.assertGreaterEqual(stats.overload_split_operations, 1)
        self.assertTrue(snapshot.rebuilt_globals)
        for rebuilt_global in snapshot.rebuilt_globals:
            self.assertTrue(
                is_global_feasible_under_live_conflicts(
                    global_hypothesis=rebuilt_global,
                    tree_store=store,
                )
            )
        self.assertRegex(overload_log, r"recursive_cache_hits=[1-9][0-9]*")
        self.assertIn("recursive_cache_misses=", overload_log)
        self.assertIn("max_recursion_depth=1", overload_log)
        self.assertIn("max_cut_key_count=1", overload_log)
        self.assertIn("total_interface_assignments=3", overload_log)
        self.assertIn("max_recombination_product_size=", overload_log)
        self.assertIn("branch_recomb_retained=", overload_log)
        self.assertIn(
            f"final_recomb_retained={len(snapshot.rebuilt_globals)}",
            overload_log,
        )
        self.assertIn("interface_assignment_cap_fallbacks=0", overload_log)

    def test_overload_recombination_recovers_beyond_naive_subcluster_top1(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        track0_id, track0_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        track1_id, track1_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (None, 0.0)],
        )
        clusters = build_track_clusters(tree_store=store, scan_index=2)

        snapshots, _ = rebuild_cluster_globals(
            clusters=clusters,
            ctx=ScanContext(
                scan_index=2,
                timestamp=timestamp,
                detections=[_detection(1.0, 1.0, timestamp)],
                det_index_by_obj={},
            ),
            tree_store=store,
            params=TOMHTParams(
                max_global_hypotheses=1,
                overload_split_projected_combination_threshold=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        self.assertEqual(1, len(snapshots))
        map_global = snapshots[0].map_global
        self.assertIsNotNone(map_global)
        assert map_global is not None
        self.assertIs(map_global.leaf_nodes_by_track_id[track0_id], track0_leaves[0])
        self.assertIs(map_global.leaf_nodes_by_track_id[track1_id], track1_leaves[1])
        self.assertTrue(
            is_global_feasible_under_live_conflicts(
                global_hypothesis=map_global,
                tree_store=store,
            )
        )

    def test_overload_supported_leaf_pruning_uses_original_feasible_globals(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        track0_id, track0_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (None, 0.0)],
        )
        track1_id, track1_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (None, 0.0)],
        )

        snapshots, _ = rebuild_cluster_globals(
            clusters=build_track_clusters(tree_store=store, scan_index=2),
            ctx=ScanContext(
                scan_index=2,
                timestamp=timestamp,
                detections=[_detection(1.0, 1.0, timestamp)],
                det_index_by_obj={},
            ),
            tree_store=store,
            params=TOMHTParams(
                max_global_hypotheses=1,
                overload_split_projected_combination_threshold=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        stats = apply_post_solve_supported_leaf_pruning(
            cluster_snapshots=snapshots,
            tree_store=store,
        )

        self.assertEqual(2, stats.unsupported_leaf_count_pruned)
        self.assertEqual(
            {track0_leaves[0].node_id},
            store.track_trees_by_track_id[track0_id].active_leaf_node_ids,
        )
        self.assertEqual(
            {track1_leaves[1].node_id},
            store.track_trees_by_track_id[track1_id].active_leaf_node_ids,
        )

    def test_overload_recombined_globals_are_deterministically_ordered(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        track0_id, track0_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(None, 1.0), (None, 1.0)],
        )
        track1_id, track1_leaves = _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(None, 0.0)],
        )
        cluster = _manual_cluster_for_track_ids(
            store=store,
            track_ids=(track0_id, track1_id),
            scan_index=2,
        )

        snapshots, _ = rebuild_cluster_globals(
            clusters=[cluster],
            ctx=ScanContext(
                scan_index=2,
                timestamp=timestamp,
                detections=[],
                det_index_by_obj={},
            ),
            tree_store=store,
            params=TOMHTParams(
                max_global_hypotheses=2,
                overload_split_projected_combination_threshold=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
            cluster_solver=make_cluster_solver("branch_and_bound"),
        )

        selections = [
            (
                global_hypothesis.leaf_nodes_by_track_id[track0_id].node_id,
                global_hypothesis.leaf_nodes_by_track_id[track1_id].node_id,
            )
            for global_hypothesis in snapshots[0].rebuilt_globals
        ]
        self.assertEqual(
            [
                (track0_leaves[0].node_id, track1_leaves[0].node_id),
                (track0_leaves[1].node_id, track1_leaves[0].node_id),
            ],
            selections,
        )

    def test_overload_no_feasible_conditional_branch_reports_clear_error(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 2)
        store = TrackTreeStore()
        shared_key = DetectionKey(scan_index=2, det_index=0)
        _add_manual_tree_with_live_options(
            store,
            root_x=0.0,
            live_options=[(shared_key, 10.0), (shared_key, 8.0)],
        )
        _add_manual_tree_with_live_options(
            store,
            root_x=10.0,
            live_options=[(shared_key, 9.0), (shared_key, 7.0)],
        )

        with self.assertRaisesRegex(RuntimeError, "no feasible combination"):
            rebuild_cluster_globals(
                clusters=build_track_clusters(tree_store=store, scan_index=2),
                ctx=ScanContext(
                    scan_index=2,
                    timestamp=timestamp,
                    detections=[_detection(1.0, 1.0, timestamp)],
                    det_index_by_obj={},
                ),
                tree_store=store,
                params=TOMHTParams(
                    max_global_hypotheses=1,
                    overload_split_projected_combination_threshold=1,
                    debug_display_scan_stats=False,
                    debug_display_hypotheses=False,
                    debug_display_births=False,
                    collect_stats=False,
                ),
                cluster_solver=make_cluster_solver("branch_and_bound"),
            )

    def test_track_confirmation_uses_max_active_leaf_score(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=True,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        self.assertEqual(
            "tentative",
            tracker.track_trees_by_track_id[0].lifecycle_state,
        )

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, -1.0), (1, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
        )

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("confirmed", tree.lifecycle_state)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        self.assertEqual(0, tracker.last_scan_stats.active_tentative_trees)
        self.assertEqual(1, tracker.last_scan_stats.active_confirmed_trees)
        self.assertEqual(1, tracker.last_scan_stats.map_tracks)
        self.assertEqual(1, tracker.last_scan_stats.map_published_tracks)
        self.assertEqual(0, tracker.last_scan_stats.map_unpublished_tracks)

        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual("confirmed", output_track.metadata["lifecycle_state"])

    def test_track_confirmation_is_sticky_if_score_later_drops(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        self.assertEqual(
            "confirmed",
            tracker.track_trees_by_track_id[0].lifecycle_state,
        )

        tree = tracker.track_trees_by_track_id[0]
        for leaf_id in tree.active_leaf_node_ids:
            tracker._nodes_by_id[leaf_id].accumulated_log_score = _logit(0.1)
        tracker._apply_score_based_track_confirmation()

        self.assertEqual("confirmed", tree.lifecycle_state)
        tree_score = tracker._tree_store.active_tree_max_accumulated_log_score(tree)
        self.assertIsNotNone(tree_score)
        assert tree_score is not None
        self.assertLess(tree_score, _logit(0.8))

    def test_default_publication_keeps_tentative_map_track_internal(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=True,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t1, [])

        self.assertIn(0, tracker.track_trees_by_track_id)
        self.assertEqual(
            "tentative",
            tracker.track_trees_by_track_id[0].lifecycle_state,
        )
        self.assertEqual(
            "unpublished",
            tracker.track_trees_by_track_id[0].publication_state,
        )
        self.assertIsNone(tracker.track_trees_by_track_id[0].public_track_id)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        self.assertEqual(1, tracker.last_scan_stats.active_tentative_trees)
        self.assertEqual(0, tracker.last_scan_stats.active_confirmed_trees)
        self.assertEqual(1, tracker.last_scan_stats.map_tracks)
        self.assertEqual(0, tracker.last_scan_stats.map_published_tracks)
        self.assertEqual(1, tracker.last_scan_stats.map_unpublished_tracks)

        self.assertEqual(set(), tracker.get_map_output_tracks())
        self.assertEqual(set(), tracker.tracks)

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(1, len(map_snapshot.leaf_nodes_by_track_id))

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(0, inspection_track.id)
        self.assertEqual(0, inspection_track.metadata["track_id"])
        self.assertEqual(0, inspection_track.metadata["internal_track_id"])
        self.assertIsNone(inspection_track.metadata["public_track_id"])
        self.assertEqual("tentative", inspection_track.metadata["lifecycle_state"])
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )

    def test_default_publication_publishes_confirmed_map_output(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
                debug_display_scan_stats=True,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("tentative", tree.lifecycle_state)
        self.assertEqual("unpublished", tree.publication_state)
        self.assertEqual(set(), tracker.get_map_output_tracks())
        self.assertEqual(set(), tracker.tracks)

        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual(1, len(map_snapshot.leaf_nodes_by_track_id))

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        with contextlib.redirect_stdout(log_stream):
            _, output_tracks = tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("confirmed", tree.lifecycle_state)
        self.assertEqual("published", tree.publication_state)
        self.assertEqual(0, tree.public_track_id)
        self.assertIn(
            "MAP tracks=1 published=1 unpublished=0",
            log_stream.getvalue(),
        )
        self.assertEqual(1, len(output_tracks))
        self.assertEqual(1, len(tracker.tracks))
        output_track = next(iter(output_tracks))
        self.assertEqual(0, output_track.id)
        self.assertEqual(0, output_track.metadata["track_id"])
        self.assertEqual(0, output_track.metadata["internal_track_id"])
        self.assertEqual(0, output_track.metadata["public_track_id"])
        self.assertEqual("confirmed", output_track.metadata["lifecycle_state"])
        self.assertEqual("published", output_track.metadata["publication_state"])

    def test_default_public_track_ids_are_dense_in_publication_order(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                track_confirmation_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0), _track_start(10.0, t0)])

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual({0, 1}, {int(track.id) for track in inspection_tracks})
        self.assertEqual(
            {None},
            {track.metadata["public_track_id"] for track in inspection_tracks},
        )

        hypothesiser.set_options(
            timestamp=t1,
            track_id=1,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(10.0, 10.0, t1)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertIsNone(tracker.track_trees_by_track_id[0].public_track_id)
        self.assertEqual(
            "published", tracker.track_trees_by_track_id[1].publication_state
        )
        self.assertEqual(0, tracker.track_trees_by_track_id[1].public_track_id)

        tracker.add_external_starts(t1, [_track_start(20.0, t1)])
        self.assertIn(2, tracker.track_trees_by_track_id)
        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[2].publication_state
        )
        self.assertIsNone(tracker.track_trees_by_track_id[2].public_track_id)

        hypothesiser.set_options(
            timestamp=t2,
            track_id=2,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t2, [_detection(20.0, 20.0, t2)])

        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual({0, 1}, {int(track.id) for track in output_tracks})
        self.assertEqual(
            {1, 2},
            {int(track.metadata["internal_track_id"]) for track in output_tracks},
        )
        self.assertEqual(
            {1, 2},
            {int(track.metadata["track_id"]) for track in output_tracks},
        )
        self.assertEqual(
            {0, 1},
            {int(track.metadata["public_track_id"]) for track in output_tracks},
        )
        for track in output_tracks:
            self.assertEqual(track.id, track.metadata["public_track_id"])

        tracker.update_tracker(t3, [])
        self.assertEqual(0, tracker.track_trees_by_track_id[1].public_track_id)
        self.assertEqual(1, tracker.track_trees_by_track_id[2].public_track_id)

    def test_dpm_receives_public_track_id_and_caller_scan_context(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        hit_context = {"sensor": "radar-a", "mode": "track"}
        empty_context = {"sensor": "radar-a", "mode": "coast"}

        hypothesiser = _ScriptedHypothesiser()
        params = TOMHTParams(
            external_start_initial_existence_probability=0.6,
            track_confirmation_existence_probability=0.8,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )
        dpm = _CaptureDetectionProbabilityModel(
            clutter_density=params.log_epsilon,
        )
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=params,
            detection_probability_model=dpm,
        )

        tracker.update_tracker(t0, [], caller_scan_context={"sensor": "bootstrap"})
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        detection = _detection(1.0, 1.0, t1)
        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [detection],
            caller_scan_context=hit_context,
        )

        calls_after_hit = len(dpm.detection_probability_calls)
        self.assertGreaterEqual(calls_after_hit, 2)
        for track_id, _, caller_context in dpm.detection_probability_calls:
            self.assertIsNone(track_id)
            self.assertIs(hit_context, caller_context)
        self.assertEqual(1, len(dpm.clutter_density_calls))
        self.assertIs(detection, dpm.clutter_density_calls[0][1])
        self.assertIs(hit_context, dpm.clutter_density_calls[0][2])
        self.assertEqual(0, tracker.track_trees_by_track_id[0].public_track_id)

        tracker.update_tracker(t2, [], caller_scan_context=empty_context)

        empty_scan_calls = dpm.detection_probability_calls[calls_after_hit:]
        self.assertGreaterEqual(len(empty_scan_calls), 1)
        for track_id, _, caller_context in empty_scan_calls:
            self.assertEqual(0, track_id)
            self.assertIs(empty_context, caller_context)
        self.assertEqual(
            1,
            len(dpm.clutter_density_calls),
            "empty-detection miss scans must not request hit clutter density",
        )

    def test_explicit_constant_dpm_matches_default_tracker_scoring(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        params = TOMHTParams(
            external_start_initial_existence_probability=0.6,
            track_confirmation_existence_probability=0.8,
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )

        default_hypothesiser = _ScriptedHypothesiser()
        default_tracker = _build_tracker(
            hypothesiser=default_hypothesiser,
            updater=_ScriptedUpdater(),
            params=params,
        )
        explicit_hypothesiser = _ScriptedHypothesiser()
        explicit_tracker = _build_tracker(
            hypothesiser=explicit_hypothesiser,
            updater=_ScriptedUpdater(),
            params=params,
            detection_probability_model=ConstantDetectionProbabilityModel(
                prob_detect=0.0,
                clutter_density=params.log_epsilon,
            ),
        )

        for tracker in (default_tracker, explicit_tracker):
            tracker.update_tracker(t0, [])
            tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        detection = _detection(1.0, 1.0, t1)
        for hypothesiser in (default_hypothesiser, explicit_hypothesiser):
            hypothesiser.set_options(
                timestamp=t1,
                track_id=0,
                options=[(0, 2.0), (None, 0.0)],
            )
        default_tracker.update_tracker(t1, [detection])
        explicit_tracker.update_tracker(t1, [detection])

        default_snapshot = default_tracker.get_map_hypothesis_snapshot()
        explicit_snapshot = explicit_tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(default_snapshot)
        self.assertIsNotNone(explicit_snapshot)
        assert default_snapshot is not None
        assert explicit_snapshot is not None
        self.assertAlmostEqual(
            default_snapshot.log_weight, explicit_snapshot.log_weight
        )
        default_leaf = next(iter(default_snapshot.leaf_nodes_by_track_id.values()))
        explicit_leaf = next(iter(explicit_snapshot.leaf_nodes_by_track_id.values()))
        self.assertAlmostEqual(
            default_leaf.accumulated_log_score,
            explicit_leaf.accumulated_log_score,
        )

    def test_publish_min_hits_gates_initial_publication(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                publish_min_hits=2,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertEqual(set(), tracker.get_map_output_tracks())

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 1.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(2, output_track.metadata["hits"])

    def test_publish_min_age_gates_initial_publication(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                publish_min_age=2,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertEqual(set(), tracker.get_map_output_tracks())

        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t1, [])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(2, output_track.metadata["age"])

    def test_publish_min_existence_probability_gates_initial_publication(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                publish_min_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[0].publication_state
        )
        self.assertEqual(set(), tracker.get_map_output_tracks())

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertGreaterEqual(
            float(output_track.metadata["existence_probability"]),
            0.8,
        )

    def test_publication_is_sticky_if_score_later_drops(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.6,
                publish_min_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (None, 0.0)],
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", tree.publication_state)
        for leaf_id in tree.active_leaf_node_ids:
            tracker._nodes_by_id[leaf_id].accumulated_log_score = _logit(0.1)

        tracker._apply_output_publication(tracker._last_map_global)

        self.assertEqual("published", tree.publication_state)
        output_tracks = tracker.get_map_output_tracks()
        self.assertEqual(1, len(output_tracks))

    def test_score_deletion_removes_tentative_tree_below_threshold(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        self.assertEqual(
            "tentative", tracker.track_trees_by_track_id[0].lifecycle_state
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual({}, filtered.leaf_nodes_by_track_id)
        self.assertEqual(set(), tracker.get_map_output_tracks(include_unpublished=True))

    def test_score_deletion_removes_confirmed_tree_below_threshold(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        tracker.track_trees_by_track_id[0].lifecycle_state = "confirmed"
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.get_map_output_tracks(include_unpublished=True))

    def test_score_deletion_keeps_tree_above_threshold(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.02)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertIn(0, tracker.track_trees_by_track_id)
        self.assertEqual({0}, set(filtered.leaf_nodes_by_track_id))

    def test_score_deletion_uses_max_active_leaf_score(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        low_leaf, high_leaf = _replace_active_leaves_with_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005), _logit(0.02)],
            timestamp=timestamp,
        )
        map_global = GlobalHypothesis(
            leaf_nodes_by_track_id={0: low_leaf},
            log_weight=float(low_leaf.accumulated_log_score),
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(
                tracker,
                timestamp=timestamp,
                map_global=map_global,
            )

        self.assertIn(0, tracker.track_trees_by_track_id)
        self.assertEqual(low_leaf.node_id, filtered.leaf_nodes_by_track_id[0].node_id)
        tree_score = tracker._tree_store.active_tree_max_accumulated_log_score(
            tracker.track_trees_by_track_id[0]
        )
        self.assertIsNotNone(tree_score)
        assert tree_score is not None
        self.assertAlmostEqual(
            float(high_leaf.accumulated_log_score),
            float(tree_score),
        )

    def test_score_deletion_filters_current_map_global_to_live_trees(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(
            timestamp,
            [_track_start(0.0, timestamp), _track_start(10.0, timestamp)],
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=1,
            scores=[_logit(0.02)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({1}, set(tracker.track_trees_by_track_id))
        self.assertEqual({1}, set(filtered.leaf_nodes_by_track_id))
        map_snapshot = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(map_snapshot)
        assert map_snapshot is not None
        self.assertEqual({1}, set(map_snapshot.leaf_nodes_by_track_id))

    def test_score_deletion_composes_with_miss_deletion_diagnostics(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                max_missed=0,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(
            timestamp,
            [_track_start(0.0, timestamp), _track_start(10.0, timestamp)],
        )
        score_leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )[0]
        miss_leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=1,
            scores=[_logit(0.02)],
        )[0]
        self.assertEqual(0, score_leaf.missed_count)
        miss_leaf.missed_count = 1

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual({}, filtered.leaf_nodes_by_track_id)
        log_output = log_stream.getvalue()
        self.assertIn("TRACK_LIFECYCLE", log_output)
        self.assertIn("terminated=[0, 1]", log_output)
        self.assertIn("score:[0]", log_output)
        self.assertIn("miss:[1]", log_output)

    def test_score_deletion_runs_with_configured_deleter(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            deleter=_MetadataMissCountDeleter(threshold=99),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        leaf = _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )[0]
        leaf.missed_count = 0

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertIn("deleter=_MetadataMissCountDeleter", log_stream.getvalue())
        self.assertIn("score:[0]", log_stream.getvalue())

    def test_score_deletion_does_not_reuse_public_ids(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                publish_lifecycle_states=("tentative", "confirmed"),
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(timestamp, [_track_start(0.0, timestamp)])
        first_tree = tracker.track_trees_by_track_id[0]
        self.assertEqual("published", first_tree.publication_state)
        self.assertEqual(0, first_tree.public_track_id)

        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)
        self.assertEqual({}, tracker.track_trees_by_track_id)

        tracker.add_external_starts(timestamp, [_track_start(10.0, timestamp)])

        self.assertEqual({1}, set(tracker.track_trees_by_track_id))
        second_tree = tracker.track_trees_by_track_id[1]
        self.assertEqual("published", second_tree.publication_state)
        self.assertEqual(1, second_tree.public_track_id)
        output_track = next(iter(tracker.get_map_output_tracks()))
        self.assertEqual(1, output_track.id)
        self.assertEqual(1, output_track.metadata["public_track_id"])

    def test_score_deletion_removes_published_and_unpublished_trees(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                external_start_initial_existence_probability=0.5,
                track_deletion_existence_probability=0.01,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [])
        tracker.add_external_starts(
            timestamp,
            [_track_start(0.0, timestamp), _track_start(10.0, timestamp)],
        )
        published_tree = tracker.track_trees_by_track_id[0]
        tracker._ensure_public_track_id(published_tree)
        published_tree.publication_state = "published"
        self.assertEqual(
            "unpublished", tracker.track_trees_by_track_id[1].publication_state
        )

        _set_track_active_leaf_scores(
            tracker,
            track_id=0,
            scores=[_logit(0.005)],
        )
        _set_track_active_leaf_scores(
            tracker,
            track_id=1,
            scores=[_logit(0.005)],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            filtered = _run_post_n_scan_lifecycle(tracker, timestamp=timestamp)

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual({}, filtered.leaf_nodes_by_track_id)
        self.assertEqual(set(), tracker.get_map_output_tracks(include_unpublished=True))

    def test_max_missed_drops_leaf_and_removes_tree(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                max_missed=0,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])

        tracker.update_tracker(t1, [])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_configured_deleter_lane_overrides_native_max_missed_policy(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            deleter=_MetadataMissCountDeleter(threshold=99),
            params=TOMHTParams(
                max_missed=0,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])

        tracker.update_tracker(t1, [])

        self.assertIn(0, tracker.track_trees_by_track_id)
        leaf_id = next(iter(tracker.track_trees_by_track_id[0].active_leaf_node_ids))
        self.assertEqual(1, tracker._nodes_by_id[leaf_id].missed_count)

    def test_configured_deleter_lane_can_delete_tracks(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            deleter=_MetadataMissCountDeleter(threshold=1),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])

        tracker.update_tracker(t1, [])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_builtin_covariance_based_deleter_deletes_track(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            deleter=CovarianceBasedDeleter(covar_trace_thresh=0.1),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(timestamp=t1, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t1, [])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_builtin_update_time_steps_deleter_behavior(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedPredictingHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdaterWithUpdateStates(),
            deleter=UpdateTimeStepsDeleter(time_steps_since_update=2),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t2, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t2, [])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t3, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t3, [])
        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_builtin_update_time_deleter_behavior(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedPredictingHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdaterWithUpdateStates(),
            deleter=UpdateTimeDeleter(time_since_update=datetime.timedelta(seconds=1)),
            params=TOMHTParams(
                max_missed=999,
                ns_scan_window=0,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t2, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t2, [])
        self.assertIn(0, tracker.track_trees_by_track_id)

        hypothesiser.set_options(timestamp=t3, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t3, [])
        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual(set(), tracker.tracks)

    def test_map_only_n_scan_pruning_promotes_map_branch_and_collects_disagreement(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                ns_scan_window=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 5.0), (1, 1.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
        )

        scan1_map = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(scan1_map)
        assert scan1_map is not None
        scan1_leaf = scan1_map.leaf_nodes_by_track_id[0]
        self.assertEqual((1, 0), scan1_leaf.used_det_key)

        hypothesiser.set_options(timestamp=t2, track_id=0, options=[(None, 0.0)])
        tracker.update_tracker(t2, [])

        tree = tracker.track_trees_by_track_id[0]
        self.assertEqual(scan1_leaf.node_id, tree.root_node_id)

        nscan = tracker.get_n_scan_commitment_snapshot()
        self.assertEqual(1, nscan.boundary_scan_index)
        self.assertIn(0, nscan.latest_committed_ancestor_by_track_id)
        self.assertEqual(
            scan1_leaf.node_id,
            nscan.latest_committed_ancestor_by_track_id[0].node_id,
        )

        clusters = tracker.get_last_cluster_snapshots()
        self.assertEqual(1, len(clusters))
        self.assertGreaterEqual(clusters[0].disagreement_count, 1)
        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        self.assertGreaterEqual(tracker.last_scan_stats.nscan_disagreement_total, 1)
        timings = tracker.last_scan_stats.timing_breakdown
        phase_sum = (
            timings.prep_ctx_ms
            + timings.pre_expand_validate_ms
            + timings.expand_ms
            + timings.post_expand_prune_validate_ms
            + timings.births_ms
            + timings.cluster_build_and_solve_ms
            + timings.post_solve_prune_ms
            + timings.map_merge_ms
            + timings.nscan_lifecycle_ms
            + timings.cleanup_ms
        )
        self.assertGreaterEqual(timings.cluster_build_and_solve_ms, 0.0)
        self.assertGreaterEqual(timings.expand_ms, 0.0)
        self.assertGreaterEqual(timings.expand_hypothesise_calls, 0)
        self.assertGreaterEqual(timings.expand_update_calls, 0)
        self.assertLessEqual(
            timings.expand_hypothesise_ms + timings.expand_update_ms,
            timings.expand_ms + 1.0,
        )
        self.assertLessEqual(
            phase_sum, tracker.last_scan_stats.scan_wall_ms + 1.0
        )  # keep tolerance for measurement overhead/noise

    def test_expansion_frontier_stats_are_populated_on_simple_scan(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (1, 1.0), (None, 0.0)],
        )
        tracker.update_tracker(
            t1,
            [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
        )

        self.assertIsNotNone(tracker.last_scan_stats)
        assert tracker.last_scan_stats is not None
        frontier = tracker.last_scan_stats.expansion_frontier
        for stat_field in fields(frontier):
            self.assertGreaterEqual(getattr(frontier, stat_field.name), 0)

        self.assertEqual(1, frontier.leaves_before_expansion)
        self.assertEqual(3, frontier.leaves_after_expansion)
        self.assertEqual(3, frontier.leaves_after_empty_tree_removal)
        self.assertEqual(3, frontier.leaves_after_births)
        self.assertEqual(3, frontier.leaves_after_post_solve_supported_pruning)
        self.assertEqual(3, frontier.leaves_after_n_scan_pruning)
        self.assertEqual(3, frontier.leaves_after_lifecycle)
        self.assertEqual(1, frontier.expanded_leaf_count)
        self.assertEqual(0, frontier.expanded_leaves_tentative)
        self.assertEqual(1, frontier.expanded_leaves_confirmed)
        self.assertEqual(3, frontier.local_child_candidates_total)
        self.assertEqual(3, frontier.local_children_created_total)
        self.assertEqual(3, frontier.local_children_retained_total)
        self.assertEqual(1, frontier.local_miss_children_created)
        self.assertEqual(2, frontier.local_detection_children_created)
        self.assertEqual(1, frontier.map_selected_leaf_count)
        self.assertEqual(3, frontier.retained_topk_supported_leaf_count)
        self.assertEqual(0, frontier.unsupported_leaf_count_pruned)

    def test_supported_leaf_pruning_applies_to_all_snapshots(self) -> None:
        tracker, leaves = _tracker_with_manual_frontier()
        snapshot = _single_track_rebuild_snapshot(
            track_id=0,
            supported_leaves=[leaves[0]],
            overload_split_origin_cluster_id=10,
        )

        stats = tracker._apply_post_solve_supported_leaf_pruning([snapshot])

        active_leaf_ids = tracker.track_trees_by_track_id[0].active_leaf_node_ids
        self.assertEqual({leaves[0].node_id}, active_leaf_ids)
        self.assertEqual(2, stats.unsupported_leaf_count_pruned)
        for stat_field in fields(stats):
            self.assertGreaterEqual(getattr(stats, stat_field.name), 0)

    def test_supported_leaf_pruning_is_deterministic(self) -> None:
        tracker, leaves = _tracker_with_manual_frontier()
        snapshot = _single_track_rebuild_snapshot(
            track_id=0,
            supported_leaves=[leaves[0], leaves[2]],
            overload_split_origin_cluster_id=10,
        )

        first_stats = tracker._apply_post_solve_supported_leaf_pruning([snapshot])
        active_leaf_ids = tracker.track_trees_by_track_id[0].active_leaf_node_ids

        self.assertEqual({leaves[0].node_id, leaves[2].node_id}, active_leaf_ids)
        self.assertEqual(1, first_stats.unsupported_leaf_count_pruned)
        for stat_field in fields(first_stats):
            self.assertGreaterEqual(getattr(first_stats, stat_field.name), 0)

        tracker_again, leaves_again = _tracker_with_manual_frontier()
        snapshot_again = _single_track_rebuild_snapshot(
            track_id=0,
            supported_leaves=[leaves_again[0], leaves_again[2]],
            overload_split_origin_cluster_id=10,
        )
        second_stats = tracker_again._apply_post_solve_supported_leaf_pruning(
            [snapshot_again]
        )
        self.assertEqual(first_stats, second_stats)

    def test_expansion_frontier_debug_flag_does_not_change_behavior(self) -> None:
        def run_case(
            *,
            debug_display_expansion_frontier: bool,
        ) -> tuple[TOMHTTracker, str]:
            t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
            t1 = t0 + datetime.timedelta(seconds=1)
            hypothesiser = _ScriptedHypothesiser()
            tracker = _build_tracker(
                hypothesiser=hypothesiser,
                updater=_ScriptedUpdater(),
                params=TOMHTParams(
                    debug_display_scan_stats=False,
                    debug_display_hypotheses=False,
                    debug_display_births=False,
                    debug_display_expansion_frontier=(debug_display_expansion_frontier),
                    collect_stats=False,
                ),
            )
            log_stream = io.StringIO()
            with contextlib.redirect_stdout(log_stream):
                tracker.update_tracker(t0, [])
            tracker.add_external_starts(t0, [_track_start(0.0, t0)])
            hypothesiser.set_options(
                timestamp=t1,
                track_id=0,
                options=[(0, 2.0), (1, 1.0), (None, 0.0)],
            )
            with contextlib.redirect_stdout(log_stream):
                tracker.update_tracker(
                    t1,
                    [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
                )
            return tracker, log_stream.getvalue()

        plain_tracker, plain_log = run_case(debug_display_expansion_frontier=False)
        debug_tracker, debug_log = run_case(debug_display_expansion_frontier=True)

        plain_map = plain_tracker.get_map_hypothesis_snapshot()
        debug_map = debug_tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(plain_map)
        self.assertIsNotNone(debug_map)
        assert plain_map is not None
        assert debug_map is not None
        plain_keys = {
            track_id: leaf.used_det_key
            for track_id, leaf in plain_map.leaf_nodes_by_track_id.items()
        }
        debug_keys = {
            track_id: leaf.used_det_key
            for track_id, leaf in debug_map.leaf_nodes_by_track_id.items()
        }
        self.assertEqual(plain_keys, debug_keys)
        self.assertIsNotNone(plain_tracker.last_scan_stats)
        self.assertIsNotNone(debug_tracker.last_scan_stats)
        assert plain_tracker.last_scan_stats is not None
        assert debug_tracker.last_scan_stats is not None
        self.assertEqual(
            plain_tracker.last_scan_stats.expansion_frontier,
            debug_tracker.last_scan_stats.expansion_frontier,
        )
        del plain_log

        debug_lines = [
            line
            for line in debug_log.splitlines()
            if line.startswith("EXPANSION_FRONTIER ")
        ]
        self.assertEqual(2, len(debug_lines))
        self.assertEqual(
            "EXPANSION_FRONTIER scan=1 t=2026-03-28 10:00:01 "
            "leaves_before=1 leaves_after_expansion=3 leaves_after_empty=3 "
            "leaves_after_births=3 leaves_after_supported_prune=3 "
            "leaves_after_nscan=3 leaves_after_lifecycle=3 trees_before=1 "
            "trees_after_lifecycle=1 expanded=1 expanded_tentative=0 "
            "expanded_confirmed=1 child_candidates=3 children_created=3 "
            "children_retained=3 miss_children=1 detection_children=2 "
            "topk_supported=3 map_selected=1 unsupported_pruned=0",
            debug_lines[-1],
        )

    def test_expansion_frontier_summary_output_is_deterministic(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                debug_display_expansion_frontier=True,
                collect_stats=True,
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])
        hypothesiser.set_options(
            timestamp=t1,
            track_id=0,
            options=[(0, 2.0), (1, 1.0), (None, 0.0)],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tracker.update_tracker(
                t1,
                [_detection(1.0, 1.0, t1), _detection(2.0, 2.0, t1)],
            )

        first = io.StringIO()
        second = io.StringIO()
        with contextlib.redirect_stdout(first):
            tracker.print_summary_stats()
        with contextlib.redirect_stdout(second):
            tracker.print_summary_stats()

        self.assertEqual(first.getvalue(), second.getvalue())
        self.assertIn("SUMMARY expansion_frontier ", first.getvalue())

    def test_map_output_reconstruction_keeps_committed_prefix_across_pruning(
        self,
    ) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)
        t2 = t1 + datetime.timedelta(seconds=1)
        t3 = t2 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            params=TOMHTParams(
                ns_scan_window=1,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t1, [_detection(1.0, 1.0, t1)])
        scan1_map = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(scan1_map)
        assert scan1_map is not None
        scan1_leaf = scan1_map.leaf_nodes_by_track_id[0]
        tree_before_detection_commit = tracker.track_trees_by_track_id[0]
        self.assertEqual(
            frozenset(),
            tree_before_detection_commit.committed_detection_keys,
        )
        self.assertEqual(
            scan1_leaf.detection_history_keys,
            live_conflict_keys_for_leaf(
                leaf=scan1_leaf,
                tree=tree_before_detection_commit,
            ),
        )

        hypothesiser.set_options(
            timestamp=t2, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t2, [_detection(2.0, 2.0, t2)])

        tree_after_first_prune = tracker.track_trees_by_track_id[0]
        self.assertEqual(1, len(tree_after_first_prune.committed_states))
        self.assertEqual(
            frozenset({(1, 0)}),
            tree_after_first_prune.committed_detection_keys,
        )

        hypothesiser.set_options(
            timestamp=t3, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t3, [_detection(3.0, 3.0, t3)])

        tree_after_second_prune = tracker.track_trees_by_track_id[0]
        self.assertEqual(2, len(tree_after_second_prune.committed_states))
        self.assertEqual(
            frozenset({(1, 0), (2, 0)}),
            tree_after_second_prune.committed_detection_keys,
        )
        scan3_map = tracker.get_map_hypothesis_snapshot()
        self.assertIsNotNone(scan3_map)
        assert scan3_map is not None
        scan3_leaf = scan3_map.leaf_nodes_by_track_id[0]
        self.assertEqual(
            frozenset({(3, 0)}),
            live_conflict_keys_for_leaf(
                leaf=scan3_leaf,
                tree=tree_after_second_prune,
            ),
        )
        committed_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in tree_after_second_prune.committed_states
        ]
        self.assertEqual([0.0, 1.0], committed_x)

        output_track = next(iter(tracker.get_map_output_tracks()))
        output_x = [
            float(np.asarray(state.state_vector, dtype=float).reshape(-1)[0])
            for state in output_track.states
        ]
        self.assertEqual([0.0, 1.0, 2.0, 3.0], output_x)
        self.assertEqual(1, output_x.count(1.0))
        self.assertEqual(4, len(output_track))

    def test_internal_births_use_step2_residual_detections(self) -> None:
        t0 = datetime.datetime(2026, 3, 28, 10, 0, 0)
        t1 = t0 + datetime.timedelta(seconds=1)

        hypothesiser = _ScriptedHypothesiser()
        capture_initiator = _CaptureInitiator([_track_start(99.0, t1)])
        tracker = _build_tracker(
            hypothesiser=hypothesiser,
            updater=_ScriptedUpdater(),
            initiator=cast(SimpleMeasurementInitiator, capture_initiator),
        )

        tracker.update_tracker(t0, [])
        tracker.add_external_starts(t0, [_track_start(0.0, t0)])

        hypothesiser.set_options(
            timestamp=t1, track_id=0, options=[(0, 4.0), (None, 0.0)]
        )

        det0 = _detection(1.0, 1.0, t1)
        det1 = _detection(2.0, 2.0, t1)
        tracker.update_tracker(t1, [det0, det1])

        self.assertEqual([det1], capture_initiator.last_received)
        self.assertIn(1, tracker.track_trees_by_track_id)
        self.assertEqual([], tracker.get_unused_detections())

    def test_no_initiator_leaves_residuals_available(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        detection = _detection(1.0, 1.0, timestamp)
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
        )

        tracker.update_tracker(timestamp, [detection])

        self.assertEqual({}, tracker.track_trees_by_track_id)
        self.assertEqual([detection], tracker.get_unused_detections())

    def test_initiator_birth_score_uses_initiator_start_prior(self) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        capture_initiator = _CaptureInitiator([_track_start(99.0, timestamp)])
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            initiator=cast(SimpleMeasurementInitiator, capture_initiator),
            params=TOMHTParams(
                initiator_start_initial_existence_probability=0.7,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        _, output_tracks = tracker.update_tracker(
            timestamp,
            [_detection(1.0, 1.0, timestamp)],
        )

        tree = next(iter(tracker.track_trees_by_track_id.values()))
        root = tracker._nodes_by_id[tree.root_node_id]
        expected_log_delta = _logit(0.7)
        self.assertEqual("internal_birth", root.root_source)
        self.assertEqual("tentative", tree.lifecycle_state)
        self.assertEqual("unpublished", tree.publication_state)
        self.assertAlmostEqual(expected_log_delta, root.log_delta)
        self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)
        self.assertEqual(set(), output_tracks)

        inspection_tracks = tracker.get_map_output_tracks(include_unpublished=True)
        self.assertEqual(1, len(inspection_tracks))
        inspection_track = next(iter(inspection_tracks))
        self.assertEqual(
            "unpublished",
            inspection_track.metadata["publication_state"],
        )

    def test_initiator_birth_metadata_existence_probability_overrides_default(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
        birth = _track_start(99.0, timestamp)
        birth.metadata["existence_probability"] = 0.6
        capture_initiator = _CaptureInitiator([birth])
        tracker = _build_tracker(
            hypothesiser=_ScriptedHypothesiser(),
            updater=_ScriptedUpdater(),
            initiator=cast(SimpleMeasurementInitiator, capture_initiator),
            params=TOMHTParams(
                initiator_start_initial_existence_probability=0.8,
                debug_display_scan_stats=False,
                debug_display_hypotheses=False,
                debug_display_births=False,
                collect_stats=False,
            ),
        )

        tracker.update_tracker(timestamp, [_detection(1.0, 1.0, timestamp)])

        tree = next(iter(tracker.track_trees_by_track_id.values()))
        root = tracker._nodes_by_id[tree.root_node_id]
        expected_log_delta = _logit(0.6)
        self.assertAlmostEqual(expected_log_delta, root.log_delta)
        self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)

    def test_initiator_birth_metadata_existence_log_odds_overrides_default(
        self,
    ) -> None:
        for value in (-2.5, 1000.0):
            with self.subTest(value=value):
                root = _initiator_birth_root_for_metadata(
                    {"existence_log_odds": value},
                )

                self.assertAlmostEqual(value, root.log_delta)
                self.assertAlmostEqual(value, root.accumulated_log_score)

    def test_initiator_birth_log_odds_metadata_precedes_probability(self) -> None:
        root = _initiator_birth_root_for_metadata(
            {
                "existence_log_odds": 1.25,
                "existence_probability": 0.6,
            },
        )

        self.assertAlmostEqual(1.25, root.log_delta)
        self.assertAlmostEqual(1.25, root.accumulated_log_score)

    def test_initiator_birth_invalid_log_odds_metadata_falls_back_to_probability(
        self,
    ) -> None:
        invalid_values: list[object] = [
            "not-a-number",
            float("nan"),
            float("inf"),
            float("-inf"),
            None,
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                root = _initiator_birth_root_for_metadata(
                    {
                        "existence_log_odds": value,
                        "existence_probability": 0.6,
                    },
                )

                expected_log_delta = _logit(0.6)
                self.assertAlmostEqual(expected_log_delta, root.log_delta)
                self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)

    def test_initiator_birth_invalid_existence_metadata_falls_back(self) -> None:
        invalid_metadata: list[dict[str, object]] = [
            {
                "existence_log_odds": "not-a-number",
                "existence_probability": "also-not-a-number",
            },
            {
                "existence_log_odds": float("nan"),
                "existence_probability": 0.0,
            },
            {
                "existence_log_odds": float("inf"),
                "existence_probability": 1.0,
            },
        ]
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                root = _initiator_birth_root_for_metadata(metadata)

                expected_log_delta = _logit(0.8)
                self.assertAlmostEqual(expected_log_delta, root.log_delta)
                self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)

    def test_initiator_birth_invalid_metadata_existence_probability_falls_back(
        self,
    ) -> None:
        invalid_values = ["not-a-number", float("nan"), float("inf"), -0.1, 1.1]
        for value in invalid_values:
            with self.subTest(value=value):
                timestamp = datetime.datetime(2026, 3, 28, 10, 0, 0)
                birth = _track_start(99.0, timestamp)
                birth.metadata["existence_probability"] = value
                capture_initiator = _CaptureInitiator([birth])
                tracker = _build_tracker(
                    hypothesiser=_ScriptedHypothesiser(),
                    updater=_ScriptedUpdater(),
                    initiator=cast(SimpleMeasurementInitiator, capture_initiator),
                    params=TOMHTParams(
                        initiator_start_initial_existence_probability=0.8,
                        debug_display_scan_stats=False,
                        debug_display_hypotheses=False,
                        debug_display_births=False,
                        collect_stats=False,
                    ),
                )

                tracker.update_tracker(timestamp, [_detection(1.0, 1.0, timestamp)])

                tree = next(iter(tracker.track_trees_by_track_id.values()))
                root = tracker._nodes_by_id[tree.root_node_id]
                expected_log_delta = _logit(0.8)
                self.assertAlmostEqual(expected_log_delta, root.log_delta)
                self.assertAlmostEqual(expected_log_delta, root.accumulated_log_score)


if __name__ == "__main__":
    unittest.main()
