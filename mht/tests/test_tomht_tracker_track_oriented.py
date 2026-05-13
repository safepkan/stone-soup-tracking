from __future__ import annotations

import datetime
from math import exp
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


class _ManualScoringModel:
    def __init__(self, *, per_unused_penalty: float = 0.5) -> None:
        self.per_unused_penalty = float(per_unused_penalty)

    def score_track_hypotheses(self, *, hypotheses, ctx) -> list[float]:
        del ctx
        return [-float(hypothesis.distance) for hypothesis in hypotheses]

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


def _build_tracker(
    *,
    hypothesiser: _ScriptedHypothesiser,
    updater: _ScriptedUpdater,
    initiator: SimpleMeasurementInitiator | None = None,
    deleter: Deleter | None = None,
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
        hypothesiser=hypothesiser,
        updater=updater,
        initiator=initiator,
        deleter=deleter,
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

        hypothesiser.set_options(
            timestamp=t2, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t2, [_detection(2.0, 2.0, t2)])

        tree_after_first_prune = tracker.track_trees_by_track_id[0]
        self.assertEqual(1, len(tree_after_first_prune.committed_states))

        hypothesiser.set_options(
            timestamp=t3, track_id=0, options=[(0, 5.0), (None, 0.0)]
        )
        tracker.update_tracker(t3, [_detection(3.0, 3.0, t3)])

        tree_after_second_prune = tracker.track_trees_by_track_id[0]
        self.assertEqual(2, len(tree_after_second_prune.committed_states))
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


if __name__ == "__main__":
    unittest.main()
