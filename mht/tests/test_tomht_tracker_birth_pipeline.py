from __future__ import annotations

import datetime
import unittest
from typing import cast
from unittest.mock import patch

import numpy as np
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.updater.base import Updater

from mht.tomht_tracker import (
    ASSOC_PAD,
    GlobalHypothesis,
    ScanContext,
    TOMHTParams,
    TOMHTTracker,
)


class _BirthPenaltyScoringModel:
    def __init__(self, birth_penalty: float) -> None:
        self._birth_penalty = float(birth_penalty)

    def score_track_hypotheses(self, *, track, hypotheses, ctx) -> dict[int, float]:
        del track, hypotheses, ctx
        return {}

    def score_unused_detections(self, *, used_det_keys: set[int], ctx) -> float:
        del used_det_keys, ctx
        return 0.0

    def score_birth(
        self, *, birth_track: Track, used_det_key: int | None, ctx
    ) -> float:
        del birth_track, used_det_key, ctx
        return -self._birth_penalty


class _MultiBirthInitiator:
    def __init__(self, births: list[Track]) -> None:
        self._births = births

    def initiate(self, detections, timestamp):
        del detections, timestamp
        return list(self._births)


def _state(timestamp: datetime.datetime) -> GaussianState:
    return GaussianState(
        [0.0, 0.0, 0.0, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )


def _birth_track(timestamp: datetime.datetime, label: str) -> Track:
    track = Track([_state(timestamp)])
    track.metadata["label"] = label
    return track


def _existing_track(
    timestamp: datetime.datetime,
    *,
    track_id: int,
    last_det_key: int | None,
) -> Track:
    track = Track([_state(timestamp)])
    track.metadata["track_id"] = track_id
    track.metadata["age"] = 1
    track.metadata["hits"] = 0
    track.metadata["missed_count"] = 0
    track.metadata["last_det_key"] = last_det_key
    track.metadata["last_det_hit"] = last_det_key is not None
    return track


def _scan_context(timestamp: datetime.datetime) -> ScanContext:
    det0 = Detection(np.array([[1.0], [2.0]]), timestamp=timestamp)
    det1 = Detection(np.array([[3.0], [4.0]]), timestamp=timestamp)
    return ScanContext(
        scan_index=0,
        timestamp=timestamp,
        detections=[det0, det1],
        det_index_by_obj={id(det0): 0, id(det1): 1},
    )


def _build_tracker(
    *,
    initiator: _MultiBirthInitiator,
    params: TOMHTParams,
    scoring_model: _BirthPenaltyScoringModel,
) -> TOMHTTracker:
    return TOMHTTracker(
        hypothesiser=cast(PDAHypothesiser, object()),
        updater=cast(Updater, object()),
        initiator=cast(SimpleMeasurementInitiator, initiator),
        params=params,
        scoring_model=scoring_model,
    )


def _seed_existing_tracks(tracker: TOMHTTracker, tracks: list[Track]) -> None:
    nodes_by_track_id = {}
    for track in tracks:
        track_id = int(track.metadata["track_id"])
        state = track.states[-1]
        last_det_key = track.metadata.get("last_det_key")
        node = tracker._create_track_hypothesis_node(
            track_id=track_id,
            parent=None,
            scan_index=0,
            timestamp=getattr(state, "timestamp", None),
            state=state,
            state_kind="seed_existing",
            used_det_key=int(last_det_key) if last_det_key is not None else None,
            assoc_label=(ASSOC_PAD if last_det_key is None else int(last_det_key)),
            log_delta=0.0,
            age=int(track.metadata.get("age", len(track))),
            hits=int(track.metadata.get("hits", 0)),
            missed_count=int(track.metadata.get("missed_count", 0)),
            last_det_key=int(last_det_key) if last_det_key is not None else None,
            last_det_hit=bool(track.metadata.get("last_det_hit", False)),
            root_source="seed_existing",
            birth_scan_index=0,
            track_metadata=dict(track.metadata),
        )
        nodes_by_track_id[track_id] = node
    tracker.global_hypotheses = [
        GlobalHypothesis(
            leaf_nodes_by_track_id=nodes_by_track_id,
            log_weight=0.0,
        )
    ]
    tracker._next_track_id = (max(nodes_by_track_id) + 1) if nodes_by_track_id else 0


class TOMHTTrackerBirthPipelineTest(unittest.TestCase):
    def test_birth_ranking_and_limit_keep_top_candidates(self) -> None:
        timestamp = datetime.datetime(2026, 3, 16, 10, 0, 0)
        rank_a = _birth_track(timestamp, "A")
        rank_b = _birth_track(timestamp, "B")
        rank_c = _birth_track(timestamp, "C")

        rank_a.metadata["sam"] = (3, 3, 0)
        rank_b.metadata["sam"] = (2, 2, 0)
        rank_c.metadata["sam"] = (2, 3, 1)
        rank_a.metadata["cov"] = 10.0
        rank_b.metadata["cov"] = 2.0
        rank_c.metadata["cov"] = 1.0
        rank_a.metadata["used"] = 0
        rank_b.metadata["used"] = 1
        rank_c.metadata["used"] = 0

        tracker = _build_tracker(
            initiator=_MultiBirthInitiator([rank_c, rank_b, rank_a]),
            params=TOMHTParams(
                max_births_per_scan=2,
                max_global_hypotheses=20,
                debug_display_births=False,
                debug_display_hypotheses=False,
                debug_display_scan_stats=False,
                collect_stats=False,
            ),
            scoring_model=_BirthPenaltyScoringModel(0.0),
        )
        ctx = _scan_context(timestamp)

        with (
            patch.object(
                tracker,
                "_birth_support_age_misses",
                side_effect=lambda tr: tr.metadata["sam"],
            ),
            patch.object(
                tracker,
                "_birth_covar_trace",
                side_effect=lambda tr: float(tr.metadata["cov"]),
            ),
            patch.object(
                tracker,
                "_birth_used_key",
                side_effect=lambda tr, det_index_by_obj: tr.metadata["used"],
            ),
        ):
            stats = tracker._branch_globals_with_births(ctx)

        labels_in_beam = {
            node.track_metadata.get("label")
            for gh in tracker.global_hypotheses
            for node in gh.leaf_nodes_by_track_id.values()
        }
        self.assertEqual({"A", "B"}, labels_in_beam)
        self.assertNotIn("C", labels_in_beam)
        self.assertEqual(3, len(tracker.global_hypotheses))
        self.assertEqual(2, stats.residual_detections_considered)
        self.assertEqual(3, stats.birth_tracks_created)
        self.assertEqual(2, stats.birth_tracks_kept)
        self.assertEqual(4, stats.birth_track_instances_in_beam)
        self.assertEqual(3, stats.globals_with_birth)

    def test_birth_compatibility_blocks_conflicting_detection_keys(self) -> None:
        timestamp = datetime.datetime(2026, 3, 16, 10, 0, 0)
        birth_conflict = _birth_track(timestamp, "CONFLICTS")
        birth_ok = _birth_track(timestamp, "OK")
        birth_conflict.metadata["sam"] = (1, 1, 0)
        birth_ok.metadata["sam"] = (1, 1, 0)
        birth_conflict.metadata["cov"] = 1.0
        birth_ok.metadata["cov"] = 1.0
        birth_conflict.metadata["used"] = 0
        birth_ok.metadata["used"] = 1

        tracker = _build_tracker(
            initiator=_MultiBirthInitiator([birth_conflict, birth_ok]),
            params=TOMHTParams(
                max_births_per_scan=2,
                max_global_hypotheses=20,
                debug_display_births=False,
                debug_display_hypotheses=False,
                debug_display_scan_stats=False,
                collect_stats=False,
            ),
            scoring_model=_BirthPenaltyScoringModel(1.0),
        )
        _seed_existing_tracks(
            tracker,
            [_existing_track(timestamp, track_id=100, last_det_key=0)],
        )
        ctx = _scan_context(timestamp)

        with (
            patch.object(
                tracker,
                "_birth_support_age_misses",
                side_effect=lambda tr: tr.metadata["sam"],
            ),
            patch.object(
                tracker,
                "_birth_covar_trace",
                side_effect=lambda tr: float(tr.metadata["cov"]),
            ),
            patch.object(
                tracker,
                "_birth_used_key",
                side_effect=lambda tr, det_index_by_obj: tr.metadata["used"],
            ),
        ):
            stats = tracker._branch_globals_with_births(ctx)

        self.assertEqual(2, len(tracker.global_hypotheses))
        self.assertEqual(
            [0.0, -1.0],
            [gh.log_weight for gh in tracker.global_hypotheses],
        )
        labels_in_beam = {
            node.track_metadata.get("label")
            for gh in tracker.global_hypotheses
            for node in gh.leaf_nodes_by_track_id.values()
            if node.track_metadata.get("label") is not None
        }
        self.assertEqual({"OK"}, labels_in_beam)
        self.assertNotIn("CONFLICTS", labels_in_beam)
        self.assertEqual(2, stats.birth_tracks_created)
        self.assertEqual(2, stats.birth_tracks_kept)
        self.assertEqual(1, stats.birth_track_instances_in_beam)
        self.assertEqual(1, stats.globals_with_birth)

    def test_birth_branching_keeps_no_birth_one_birth_and_two_birth_variants(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 16, 10, 0, 0)
        birth_a = _birth_track(timestamp, "A")
        birth_b = _birth_track(timestamp, "B")
        birth_a.metadata["sam"] = (1, 1, 0)
        birth_b.metadata["sam"] = (1, 1, 0)
        birth_a.metadata["cov"] = 1.0
        birth_b.metadata["cov"] = 1.0
        birth_a.metadata["used"] = 0
        birth_b.metadata["used"] = 1

        tracker = _build_tracker(
            initiator=_MultiBirthInitiator([birth_a, birth_b]),
            params=TOMHTParams(
                max_births_per_scan=2,
                max_global_hypotheses=20,
                debug_display_births=False,
                debug_display_hypotheses=False,
                debug_display_scan_stats=False,
                collect_stats=False,
            ),
            scoring_model=_BirthPenaltyScoringModel(1.0),
        )
        _seed_existing_tracks(
            tracker,
            [_existing_track(timestamp, track_id=7, last_det_key=None)],
        )
        ctx = _scan_context(timestamp)

        with (
            patch.object(
                tracker,
                "_birth_support_age_misses",
                side_effect=lambda tr: tr.metadata["sam"],
            ),
            patch.object(
                tracker,
                "_birth_covar_trace",
                side_effect=lambda tr: float(tr.metadata["cov"]),
            ),
            patch.object(
                tracker,
                "_birth_used_key",
                side_effect=lambda tr, det_index_by_obj: tr.metadata["used"],
            ),
        ):
            stats = tracker._branch_globals_with_births(ctx)

        self.assertEqual(4, len(tracker.global_hypotheses))
        self.assertEqual(
            [0.0, -1.0, -1.0, -2.0],
            [gh.log_weight for gh in tracker.global_hypotheses],
        )
        self.assertEqual(2, stats.birth_tracks_created)
        self.assertEqual(2, stats.birth_tracks_kept)
        self.assertEqual(4, stats.birth_track_instances_in_beam)
        self.assertEqual(3, stats.globals_with_birth)


if __name__ == "__main__":
    unittest.main()
