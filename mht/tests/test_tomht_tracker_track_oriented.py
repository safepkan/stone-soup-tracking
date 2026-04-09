from __future__ import annotations

import datetime
import unittest
from typing import cast

import numpy as np
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from mht.tomht_tracker import TOMHTParams, TOMHTTracker


class _ScriptedHypothesis:
    def __init__(
        self,
        *,
        measurement,
        prediction: GaussianState,
        updated_state: GaussianState,
        log_delta: float,
        is_miss: bool,
    ) -> None:
        self.measurement = measurement
        self.prediction = prediction
        self.updated_state = updated_state
        self.log_delta = float(log_delta)
        self.probability = 1.0
        self.weight = 1.0
        self._is_miss = bool(is_miss)

    def __bool__(self) -> bool:
        return not self._is_miss


class _ScriptedHypothesiser:
    """Per-(timestamp,track_id) scripted local hypotheses for deterministic tests."""

    def __init__(self) -> None:
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

    def hypothesise(self, track: Track, detections, timestamp: datetime.datetime):
        track_id = int(track.metadata["track_id"])
        options = self._script.get((timestamp, track_id), [(None, 0.0)])
        prediction = cast(GaussianState, track.states[-1])

        out = []
        for det_index, log_delta in options:
            if det_index is None:
                out.append(
                    _ScriptedHypothesis(
                        measurement=MissedDetection(timestamp=timestamp),
                        prediction=prediction,
                        updated_state=prediction,
                        log_delta=log_delta,
                        is_miss=True,
                    )
                )
                continue

            detection = detections[int(det_index)]
            out.append(
                _ScriptedHypothesis(
                    measurement=detection,
                    prediction=prediction,
                    updated_state=_state_from_detection(detection, timestamp),
                    log_delta=log_delta,
                    is_miss=False,
                )
            )
        return out


class _NoopPredictor:
    def predict(self, prior, timestamp=None, **kwargs):
        del prior, timestamp, kwargs
        raise RuntimeError("No prediction expected for scripted hypotheses")


class _ScriptedUpdater:
    def update(self, hypothesis: _ScriptedHypothesis) -> GaussianState:
        return hypothesis.updated_state


class _ManualScoringModel:
    def __init__(self, *, per_unused_penalty: float = 0.5) -> None:
        self.per_unused_penalty = float(per_unused_penalty)

    def score_track_hypotheses(self, *, track, hypotheses, ctx) -> dict[int, float]:
        del track, ctx
        return {id(hyp): float(hyp.log_delta) for hyp in hypotheses}

    def score_unused_detections(self, *, used_det_keys: set[int], ctx) -> float:
        unused = len(ctx.detections) - len(used_det_keys)
        return -float(unused) * self.per_unused_penalty

    def score_birth(
        self, *, birth_track: Track, used_det_key: int | None, ctx
    ) -> float:
        del birth_track, used_det_key, ctx
        return -1.0


class _CaptureInitiator:
    def __init__(self, born: list[Track]) -> None:
        self._born = born
        self.last_received: list[Detection] = []

    def initiate(self, detections, timestamp):
        del timestamp
        self.last_received = list(detections)
        return list(self._born)


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


def _build_tracker(
    *,
    hypothesiser: _ScriptedHypothesiser,
    updater: _ScriptedUpdater,
    initiator: SimpleMeasurementInitiator | None = None,
    params: TOMHTParams | None = None,
) -> TOMHTTracker:
    if params is None:
        params = TOMHTParams(
            debug_display_scan_stats=False,
            debug_display_hypotheses=False,
            debug_display_births=False,
            collect_stats=False,
        )
    return TOMHTTracker(
        predictor=_NoopPredictor(),
        updater=updater,
        hypothesis_generator=hypothesiser,
        initiator=initiator,
        params=params,
        scoring_model=_ManualScoringModel(per_unused_penalty=0.5),
    )


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
            int(leaf.used_det_key[1])
            for leaf in map_snapshot.leaf_nodes_by_track_id.values()
            if leaf.used_det_key is not None
        }
        self.assertEqual({0, 1}, used_det_indices)

        tree_snapshot = tracker.get_track_tree_snapshot()
        self.assertEqual({0, 1}, set(tree_snapshot.keys()))

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


if __name__ == "__main__":
    unittest.main()
