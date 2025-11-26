from __future__ import annotations

from dataclasses import dataclass
from math import log
from ordered_set import OrderedSet
from typing import Iterable, TypeAlias

import numpy as np

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.models.transition.base import TransitionModel
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from stonesoup.initiator.simple import SimpleMeasurementInitiator

from mht_experiments.helpers.hypothesiser import RobustPDAHypothesiser


PredictorT: TypeAlias = KalmanPredictor | UnscentedKalmanPredictor
UpdaterT: TypeAlias = KalmanUpdater | UnscentedKalmanUpdater


@dataclass(frozen=True)
class TOMHTParams:
    max_global_hypotheses: int = 20
    max_children_per_track: int = 5
    max_missed: int = 5
    log_epsilon: float = 1e-12

    prob_gate: float = 0.99

    max_births_per_scan: int = 2
    birth_log_penalty: float = (
        8.0  # subtract this from log-weight per birth (i.e. add -8.0)
    )
    births_k: int = 5  # how many top globals are used to define "residual"
    unused_det_log_penalty: float = 0.2

    birth_max_abs_pos: float = 1e5  # safety: reject absurd positions
    birth_max_covar_trace: float = 1e12  # safety: reject absurd uncertainty

    debug_display_detections: bool = False
    debug_display_hypotheses: bool = True
    debug_display_births: bool = True
    debug_births_max: int = 5
    debug_globals_max: int = 5


@dataclass(frozen=True)
class ChildCandidate:
    track_id: int
    child_track: Track
    used_det_key: int | None
    log_delta: float


@dataclass(frozen=True)
class GlobalHypothesis:
    """One global hypothesis = one leaf per track_id + cumulative log weight."""

    tracks_by_id: dict[int, Track]
    log_weight: float


class TOMHTTracker:
    """
    Track-Oriented MHT with K-best global hypotheses (beam search).

    - Maintains a list of GlobalHypothesis objects of size <= K.
    - Each scan: branch each track (per global hyp), then form consistent globals
      (one child per track_id, no shared detections).
    """

    def __init__(
        self,
        hypothesiser: PDAHypothesiser,
        updater: UpdaterT,
        tracks: Iterable[Track],
        *,
        initiator: SimpleMeasurementInitiator | None = None,
        params: TOMHTParams = TOMHTParams(),
    ) -> None:
        self.hypothesiser = hypothesiser
        self.updater = updater
        self.params = params
        self.initiator = initiator

        init_tracks_by_id: dict[int, Track] = {}
        max_tid = -1
        for i, tr in enumerate(list(tracks)):
            tr.metadata.setdefault("track_id", i)
            tr.metadata.setdefault("missed_count", 0)
            init_tracks_by_id[int(tr.metadata["track_id"])] = tr
            max_tid = max(max_tid, int(tr.metadata["track_id"]))

        self._next_track_id = max_tid + 1

        self.global_hypotheses: list[GlobalHypothesis] = [
            GlobalHypothesis(tracks_by_id=init_tracks_by_id, log_weight=0.0)
        ]

    @staticmethod
    def _fmt_state_xyvxvy(state_vector) -> str:
        sv = np.asarray(state_vector, dtype=float)
        x = float(sv[0, 0])
        vx = float(sv[1, 0])
        y = float(sv[2, 0])
        vy = float(sv[3, 0])
        return f"(x={x:.1f}, vx={vx:.2f}, y={y:.1f}, vy={vy:.2f})"

    def _hyp_probability(self, hyp) -> float:
        p = getattr(hyp, "probability", None)
        if p is None:
            w = getattr(hyp, "weight", 0.0)
            try:
                return float(w)
            except Exception:
                return 0.0
        try:
            return float(p)
        except Exception:
            return 0.0

    def _hyp_log_delta(self, hyp) -> float:
        return log(max(self._hyp_probability(hyp), self.params.log_epsilon))

    @staticmethod
    def _copy_track(track: Track) -> Track:
        # Track is list-like; copying states is enough for our simple bookkeeping
        return Track(list(track.states))

    def _used_det_key_for_track(self, tr: Track) -> int | None:
        # Deterministic per-scan key assigned in _candidates_for_track
        val = tr.metadata.get("last_det_key", None)
        return int(val) if val is not None else None

    def _used_det_keys_for_tracks(self, tracks_by_id: dict[int, Track]) -> set[int]:
        used: set[int] = set()
        for tr in tracks_by_id.values():
            dk = self._used_det_key_for_track(tr)
            if dk is not None:
                used.add(dk)
        return used

    def _used_det_keys_in_global(self, gh: GlobalHypothesis) -> set[int]:
        used: set[int] = set()
        for tr in gh.tracks_by_id.values():
            dk = self._used_det_key_for_track(tr)
            if dk is not None:
                used.add(dk)
        return used

    def _residual_detections(
        self,
        globals: list[GlobalHypothesis],
        detections: list[Detection],
    ) -> list[Detection]:
        k = max(1, min(self.params.births_k, len(globals)))
        used: set[int] = set()
        for gh in globals[:k]:
            used |= self._used_det_keys_in_global(gh)

        out = []
        for i, d in enumerate(detections):
            if i not in used:
                out.append(d)
        return out

    def _apply_unused_detection_penalty(
        self,
        gh: GlobalHypothesis,
        detections: list[Detection],
    ) -> GlobalHypothesis:
        if not detections:
            return gh
        used = self._used_det_keys_for_tracks(gh.tracks_by_id)
        unused = len(detections) - len(used)
        if unused <= 0:
            return gh
        return GlobalHypothesis(
            tracks_by_id=gh.tracks_by_id,
            log_weight=gh.log_weight
            - self.params.unused_det_log_penalty * float(unused),
        )

    def _birth_support_points(self, birth: Track) -> int:
        holding = birth.metadata.get("holding_track", None)
        hist = holding if isinstance(holding, Track) else birth
        # updates_only semantics (you use updates_only=True in your initiator)
        return sum(1 for s in hist.states if isinstance(s, Update))

    def _birth_covar_trace(self, birth: Track) -> float:
        st = birth.states[-1]
        cov = getattr(st, "covar", None)
        if cov is None:
            return float("inf")
        return float(np.trace(np.asarray(cov, dtype=float)))

    def _birth_holding_track(self, birth: Track) -> Track:
        holding = birth.metadata.get("holding_track", None)
        return holding if isinstance(holding, Track) else birth

    def _birth_support_age_misses(self, birth: Track) -> tuple[int, int, int]:
        holding = self._birth_holding_track(birth)
        age = len(holding)  # number of steps in holding life
        support = self._birth_support_points(birth)  # update-count (hits)
        misses = max(age - support, 0)
        return support, age, misses

    def _candidates_for_track(
        self,
        track_id: int,
        track: Track,
        detections: list[Detection],
        timestamp,
        det_index_by_obj: dict[int, int],
    ) -> list[ChildCandidate]:
        multi = self.hypothesiser.hypothesise(track, detections, timestamp)
        singles = list(multi)

        def sort_key(hyp) -> tuple[float, int]:
            p = self._hyp_probability(hyp)
            if not hyp:
                return (p, -1)  # deterministic position for misses among ties
            return (p, -det_index_by_obj.get(id(hyp.measurement), 10**9))

        # Sort best-first and cap. Always keep a "miss" if present.
        singles_sorted = sorted(singles, key=sort_key, reverse=True)
        kept = singles_sorted[: self.params.max_children_per_track]
        miss = next((h for h in singles_sorted if not h), None)
        if miss is not None and miss not in kept:
            kept.append(miss)

        candidates: list[ChildCandidate] = []
        for hyp in kept:
            child = self._copy_track(track)
            child.metadata.update(track.metadata)

            if not hyp:
                child.append(hyp.prediction)
                child.metadata["missed_count"] = (
                    int(track.metadata.get("missed_count", 0)) + 1
                )
                used = None
            else:
                upd = self.updater.update(hyp)
                child.append(upd)
                child.metadata["missed_count"] = 0
                used = det_index_by_obj[id(hyp.measurement)]

            child.metadata["age"] = int(track.metadata.get("age", len(track))) + 1
            child.metadata["hits"] = int(track.metadata.get("hits", 0)) + (
                1 if hyp else 0
            )
            child.metadata["last_det_key"] = used
            child.metadata["last_det_hit"] = used is not None

            candidates.append(
                ChildCandidate(
                    track_id=track_id,
                    child_track=child,
                    used_det_key=used,
                    log_delta=self._hyp_log_delta(hyp),
                )
            )

        candidates.sort(key=lambda c: c.log_delta, reverse=True)
        return candidates

    def _expand_global_hypothesis(
        self,
        gh: GlobalHypothesis,
        detections: list[Detection],
        timestamp,
        det_index_by_obj: dict[int, int],
    ) -> list[GlobalHypothesis]:
        track_ids = sorted(gh.tracks_by_id.keys())

        per_track_candidates: list[list[ChildCandidate]] = []
        for tid in track_ids:
            per_track_candidates.append(
                self._candidates_for_track(
                    tid,
                    gh.tracks_by_id[tid],
                    detections,
                    timestamp,
                    det_index_by_obj,
                )
            )

        new_globals: list[GlobalHypothesis] = []

        def backtrack(
            i: int, used: set[int], acc_tracks: dict[int, Track], acc_log: float
        ) -> None:
            if i == len(track_ids):
                new_globals.append(
                    GlobalHypothesis(tracks_by_id=dict(acc_tracks), log_weight=acc_log)
                )
                return

            tid = track_ids[i]
            for cand in per_track_candidates[i]:
                if cand.used_det_key is not None and cand.used_det_key in used:
                    continue

                child = cand.child_track
                missed = int(child.metadata.get("missed_count", 0))
                if missed > self.params.max_missed:
                    acc_tracks.pop(tid, None)
                    backtrack(i + 1, used, acc_tracks, acc_log + cand.log_delta)
                    # restore for other branches
                    acc_tracks[tid] = child
                    continue

                if cand.used_det_key is not None:
                    used.add(cand.used_det_key)
                acc_tracks[tid] = child
                backtrack(i + 1, used, acc_tracks, acc_log + cand.log_delta)
                if cand.used_det_key is not None:
                    used.remove(cand.used_det_key)

        backtrack(0, set(), dict(), gh.log_weight)
        return new_globals

    def _birth_used_key(
        self, tr: Track, det_index_by_obj: dict[int, int]
    ) -> int | None:
        # Try to recover which detection was used to create this initiated track.
        try:
            last = tr.states[-1]
            hyp = getattr(last, "hypothesis", None)
            meas = getattr(hyp, "measurement", None) if hyp is not None else None
            if meas is None:
                return None
            return det_index_by_obj.get(id(meas))
        except Exception:
            return None

    def _birth_is_sane(self, tr: Track) -> bool:
        st = tr.states[-1]
        sv = np.asarray(st.state_vector, dtype=float)

        x = float(sv[0, 0])
        y = float(sv[2, 0])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        if (
            abs(x) > self.params.birth_max_abs_pos
            or abs(y) > self.params.birth_max_abs_pos
        ):
            return False

        cov = getattr(st, "covar", None)
        if cov is None:
            return False
        cov = np.asarray(cov, dtype=float)
        if not np.all(np.isfinite(cov)):
            return False
        if float(np.trace(cov)) > self.params.birth_max_covar_trace:
            return False

        return True

    def _branch_globals_with_births(
        self,
        detections: list[Detection],
        det_index_by_obj: dict[int, int],
        timestamp,
    ) -> None:
        if self.initiator is not None and self.global_hypotheses:
            residual = self._residual_detections(self.global_hypotheses, detections)

            born = (
                list(self.initiator.initiate(OrderedSet(residual), timestamp))
                if residual
                else []
            )

            born = [tr for tr in born if self._birth_is_sane(tr)]

            born_scored: list[tuple[tuple, Track]] = []
            for tr in born:
                used = self._birth_used_key(tr, det_index_by_obj)
                support, age, misses = self._birth_support_age_misses(tr)
                covtr = self._birth_covar_trace(tr)

                # Prefer: more support, fewer misses, shorter holding age (i.e. confirmed quickly),
                # then tighter covariance, then deterministic tie-break.
                key = (
                    -support,
                    misses,
                    age,
                    covtr,
                    used if used is not None else 10**9,
                )

                born_scored.append((key, tr))

            born_scored.sort(key=lambda kt: kt[0])
            born = [tr for _, tr in born_scored]

            if self.params.debug_display_births:
                print(
                    f"\nBirth candidates at {timestamp} (pre-limit): {len(born_scored)}"
                )
                for key, tr in born_scored[: self.params.debug_births_max]:
                    used = self._birth_used_key(tr, det_index_by_obj)
                    support, age, misses = self._birth_support_age_misses(tr)
                    covtr = self._birth_covar_trace(tr)
                    last = tr.states[-1].state_vector
                    print(
                        f"  used={used}, support={support}, age={age}, misses={misses}, covtr={covtr:.2g}, "
                        f"last={self._fmt_state_xyvxvy(last)}"
                    )

            born = born[: self.params.max_births_per_scan]

            if self.params.debug_display_births:
                print(f"Births kept (post-limit): {len(born)}")
                for tr in born[: self.params.debug_births_max]:
                    used = self._birth_used_key(tr, det_index_by_obj)
                    support, age, misses = self._birth_support_age_misses(tr)
                    covtr = self._birth_covar_trace(tr)
                    last = tr.states[-1].state_vector
                    print(
                        f"  used={used}, support={support}, age={age}, misses={misses}, covtr={covtr:.2g}, "
                        f"last={self._fmt_state_xyvxvy(last)}"
                    )

            if born:
                # Allocate stable IDs once for these births (shared across variants)
                birth_templates: list[tuple[int, Track, int | None]] = []
                for tr in born:
                    tid = self._next_track_id
                    self._next_track_id += 1
                    used_key = self._birth_used_key(tr, det_index_by_obj)
                    birth_templates.append((tid, tr, used_key))

                new_globals: list[GlobalHypothesis] = []
                for gh in self.global_hypotheses:
                    # If there are no tracks yet, don't keep the empty hypothesis once we have births to add.
                    if gh.tracks_by_id:
                        new_globals.append(gh)

                    used_in_gh = self._used_det_keys_in_global(gh)
                    compatible = [
                        (tid, tr, used)
                        for (tid, tr, used) in birth_templates
                        if used is None or used not in used_in_gh
                    ]

                    if self.params.debug_display_births:
                        print(
                            f"  GH logW={gh.log_weight:.3f}: compatible_births={len(compatible)}"
                        )

                    # Always allow "no birth" variant (except for empty start heuristic, see above)
                    # and then branch with births one-by-one.
                    for tid, template, used in compatible:
                        tracks_by_id = dict(gh.tracks_by_id)

                        tr_copy = self._copy_track(template)
                        tr_copy.metadata["track_id"] = tid
                        tr_copy.metadata["age"] = 1
                        tr_copy.metadata["hits"] = 1 if used is not None else 0
                        tr_copy.metadata["missed_count"] = 0
                        tr_copy.metadata["last_det_key"] = used
                        tr_copy.metadata["last_det_hit"] = used is not None

                        tracks_by_id[tid] = tr_copy
                        new_globals.append(
                            GlobalHypothesis(
                                tracks_by_id=tracks_by_id,
                                log_weight=gh.log_weight
                                - self.params.birth_log_penalty,
                            )
                        )

                    # Optional: also include the "two births at once" variant when exactly 2 are compatible.
                    use_two_births = True
                    if use_two_births:
                        if (
                            len(compatible) >= 2
                            and self.params.max_births_per_scan >= 2
                        ):
                            (tid1, t1, u1), (tid2, t2, u2) = (
                                compatible[0],
                                compatible[1],
                            )
                            if (
                                u1 is None or u2 is None or u1 != u2
                            ):  # should always hold, but be safe
                                tracks_by_id = dict(gh.tracks_by_id)

                                for tid, template, used in [
                                    (tid1, t1, u1),
                                    (tid2, t2, u2),
                                ]:
                                    tr_copy = self._copy_track(template)
                                    tr_copy.metadata["track_id"] = tid
                                    tr_copy.metadata["age"] = 1
                                    tr_copy.metadata["hits"] = (
                                        1 if used is not None else 0
                                    )
                                    tr_copy.metadata["missed_count"] = 0
                                    tr_copy.metadata["last_det_key"] = used
                                    tr_copy.metadata["last_det_hit"] = used is not None
                                    tracks_by_id[tid] = tr_copy

                                new_globals.append(
                                    GlobalHypothesis(
                                        tracks_by_id=tracks_by_id,
                                        log_weight=gh.log_weight
                                        - 2.0 * self.params.birth_log_penalty,
                                    )
                                )

                new_globals.sort(key=lambda g: g.log_weight, reverse=True)
                self.global_hypotheses = new_globals[
                    : self.params.max_global_hypotheses
                ]

    def _dedupe_globals_by_last_keys(
        self, globals: list[GlobalHypothesis]
    ) -> list[GlobalHypothesis]:
        """Keep best log_weight per (track_id -> last_det_key) signature."""
        best: dict[tuple[tuple[int, int | None], ...], GlobalHypothesis] = {}
        for gh in globals:
            sig = tuple(
                sorted(
                    (tid, gh.tracks_by_id[tid].metadata.get("last_det_key", None))
                    for tid in gh.tracks_by_id
                )
            )
            prev = best.get(sig)
            if prev is None or gh.log_weight > prev.log_weight:
                best[sig] = gh
        return list(best.values())

    def step(self, detections: Iterable[Detection], timestamp) -> set[Track]:
        det_list = list(detections)
        det_index_by_obj = {id(det): i for i, det in enumerate(det_list)}

        # Expand globals with current set of detections
        expanded: list[GlobalHypothesis] = []
        for gh in self.global_hypotheses:
            expanded.extend(
                self._expand_global_hypothesis(
                    gh, det_list, timestamp, det_index_by_obj
                )
            )

        # Apply unused detection penalty
        expanded = [
            self._apply_unused_detection_penalty(gh, det_list) for gh in expanded
        ]

        # Dedupe
        expanded = self._dedupe_globals_by_last_keys(expanded)

        # Keep top-K globals (beam)
        expanded.sort(key=lambda g: g.log_weight, reverse=True)
        self.global_hypotheses = expanded[: self.params.max_global_hypotheses]

        # Births: run initiator once on residual detections, then branch globals with/without births.
        self._branch_globals_with_births(det_list, det_index_by_obj, timestamp)

        if self.params.debug_display_detections:
            print(f"\nDetections at timestamp {timestamp}:")
            for det in det_list:
                print(f"  {det.state_vector}")

        if self.params.debug_display_hypotheses:
            print(f"\nGlobal hypotheses at timestamp {timestamp}:")
            for gh in self.global_hypotheses[: self.params.debug_globals_max]:
                used = len(self._used_det_keys_for_tracks(gh.tracks_by_id))
                unused = len(det_list) - used
                print(
                    f"logW={gh.log_weight:.3f}, "
                    f"tracks={len(gh.tracks_by_id)}, "
                    f"used={used}, unused={unused}, "
                    f"ids={sorted(gh.tracks_by_id.keys())}"
                )

                for tid, tr in sorted(gh.tracks_by_id.items()):
                    last = tr.states[-1].state_vector
                    ldk = tr.metadata.get("last_det_key", None)
                    miss = int(tr.metadata.get("missed_count", 0))
                    dk = self._used_det_key_for_track(tr)
                    used_str = "MISS" if dk is None else "HIT"
                    age = int(tr.metadata.get("age", len(tr)))
                    hits = int(tr.metadata.get("hits", 0))
                    print(
                        f"  id={tid}, {used_str}, age={age}, hits={hits}, miss={miss}, ldk={ldk}, last={self._fmt_state_xyvxvy(last)}"
                    )

        # Output MAP global hypothesis
        if not self.global_hypotheses:
            return set()

        best = self.global_hypotheses[0]
        return set(best.tracks_by_id.values())


def build_tomht_linear(
    transition_model: TransitionModel,
    measurement_model: MeasurementModel,
    *,
    prob_detect: float,
    clutter_density: float,
    tracks: Iterable[Track],
    initiator: SimpleMeasurementInitiator | None = None,
    params: TOMHTParams = TOMHTParams(),
) -> TOMHTTracker:
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = PDAHypothesiser(
        predictor,
        updater,
        clutter_density,
        prob_gate=params.prob_gate,
        prob_detect=prob_detect,
    )
    return TOMHTTracker(
        hypothesiser, updater, tracks, initiator=initiator, params=params
    )


def build_tomht_ukf(
    transition_model: TransitionModel,
    measurement_model: MeasurementModel,
    *,
    prob_detect: float,
    clutter_density: float,
    tracks: Iterable[Track],
    initiator: SimpleMeasurementInitiator | None = None,
    params: TOMHTParams = TOMHTParams(),
) -> TOMHTTracker:
    predictor = UnscentedKalmanPredictor(transition_model)
    updater = UnscentedKalmanUpdater(measurement_model)
    hypothesiser = RobustPDAHypothesiser(
        predictor,
        updater,
        clutter_density,
        prob_gate=params.prob_gate,
        prob_detect=prob_detect,
    )
    return TOMHTTracker(
        hypothesiser, updater, tracks, initiator=initiator, params=params
    )
