from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import log
from statistics import median
from ordered_set import OrderedSet
from typing import Iterable, Mapping, Protocol

import numpy as np

from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.types.detection import MissedDetection
from stonesoup.models.measurement.base import MeasurementModel
from stonesoup.models.transition.base import TransitionModel
from stonesoup.predictor.kalman import KalmanPredictor, UnscentedKalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.updater.kalman import KalmanUpdater, UnscentedKalmanUpdater
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.updater.base import Updater

from mht_experiments.helpers.hypothesiser import RobustPDAHypothesiser

ASSOC_PAD = -1
ASSOC_MISS = -2


@dataclass(frozen=True)
class TOMHTParams:
    max_global_hypotheses: int = 20
    max_children_per_track: int = 5
    max_missed: int = 5
    log_epsilon: float = 1e-12
    scoring_mode: str = "beta_ratio"  # Only beta_ratio is supported.

    assoc_history_len: int = 3
    ns_scan_window: int = 0  # default set in __post_init__

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
    debug_display_scan_stats: bool = True
    debug_display_hypotheses: bool = True
    debug_display_births: bool = True
    debug_display_map_miss_hist: bool = False
    debug_births_max: int = 5
    debug_globals_max: int = 5
    collect_stats: bool = True

    def __post_init__(self) -> None:
        # Default the N-scan window to the stored history length.
        if self.ns_scan_window <= 0:
            object.__setattr__(self, "ns_scan_window", self.assoc_history_len)


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


@dataclass(frozen=True)
class ScanContext:
    """Per-scan context passed into scoring models."""

    timestamp: object
    detections: list[Detection]
    det_index_by_obj: dict[int, int]


@dataclass(frozen=True)
class BirthStats:
    residual_detections_considered: int = 0
    birth_tracks_created: int = 0
    birth_tracks_kept: int = 0
    birth_track_instances_in_beam: int = 0
    globals_with_birth: int = 0
    globals_before_births: int = 0
    globals_after_births: int = 0


@dataclass(frozen=True)
class ScanStats:
    timestamp: object
    num_detections: int
    globals_in: int
    globals_expanded: int
    globals_after_unused: int
    globals_after_dedupe: int
    globals_after_beam: int
    globals_after_births: int
    birth_candidates: int
    birth_tracks_created: int
    birth_tracks_kept: int
    birth_track_instances_in_beam: int
    globals_with_birth: int
    map_tracks: int
    map_used: int
    map_unused: int
    map_miss_hist: dict[int, int]
    map_mean_hit_rate: float


class ScoringModel(Protocol):
    def score_track_hypotheses(
        self,
        *,
        track: Track,
        hypotheses: Iterable,
        ctx: ScanContext,
    ) -> Mapping[int, float]:
        """Return {id(hypothesis_object): log_delta} for each hypothesis."""

    def score_unused_detections(
        self,
        *,
        used_det_keys: set[int],
        ctx: ScanContext,
    ) -> float:
        """Return a log_delta to add for clutter / unused detections."""

    def score_birth(
        self,
        *,
        birth_track: Track,
        used_det_key: int | None,
        ctx: ScanContext,
    ) -> float:
        """Return a log_delta to add for a birth (usually negative)."""


@dataclass(frozen=True)
class BetaRatioScoringModel:
    """Approximate MHT-like scoring using PDA β ratios."""

    prob_detect: float
    prob_gate: float
    clutter_density: float
    log_epsilon: float
    fallback_unused_det_log_penalty: float
    birth_log_penalty: float

    def _per_unused_log_delta(self) -> float:
        """Return the per-unused log increment used by clutter scoring."""
        lam = max(self.clutter_density, 0.0)
        if lam <= 0.0:
            return -self.fallback_unused_det_log_penalty
        return log(max(lam, self.log_epsilon))

    def _beta_values(
        self, hypotheses: Iterable
    ) -> tuple[float, Mapping[int, tuple[float, bool]]]:
        beta0 = None
        scores: dict[int, tuple[float, bool]] = {}
        for hyp in hypotheses:
            p = getattr(hyp, "probability", None)
            try:
                p_float = float(p) if p is not None else 0.0
            except Exception:
                p_float = 0.0

            is_miss = isinstance(
                getattr(hyp, "measurement", None), MissedDetection
            ) or (not hyp)
            if is_miss:
                beta0 = p_float if beta0 is None else beta0 + p_float
            scores[id(hyp)] = (p_float, is_miss)

        beta0_val = beta0 if beta0 is not None else self.log_epsilon
        beta0_val = max(beta0_val, self.log_epsilon)
        return beta0_val, scores

    def score_track_hypotheses(
        self, *, track: Track, hypotheses: Iterable, ctx: ScanContext
    ) -> Mapping[int, float]:
        beta0, prob_map = self._beta_values(hypotheses)
        pd_pg = min(1.0, max(0.0, self.prob_detect * self.prob_gate))
        common = log(max(1.0 - pd_pg, self.log_epsilon))

        out: dict[int, float] = {}
        for hyp_id, (beta, is_miss) in prob_map.items():
            beta_clamped = max(beta, self.log_epsilon)
            beta0_clamped = max(beta0, self.log_epsilon)
            if is_miss:
                out[hyp_id] = common
            else:
                out[hyp_id] = log(beta_clamped) - log(beta0_clamped) + common
        return out

    def score_unused_detections(
        self, *, used_det_keys: set[int], ctx: ScanContext
    ) -> float:
        unused = len(ctx.detections) - len(used_det_keys)
        if unused <= 0:
            return 0.0
        return float(unused) * self._per_unused_log_delta()

    def score_birth(
        self, *, birth_track: Track, used_det_key: int | None, ctx: ScanContext
    ) -> float:
        return -self.birth_log_penalty


class TOMHTTracker:
    """
    Track-Oriented MHT with K-best global hypotheses (beam search).

    - Maintains a list of GlobalHypothesis objects of size <= K.
    - Each scan: branch each track (per global hyp), then form consistent globals
      (one child per track_id, no shared detections).
    """

    def _new_assoc_history(self, last_entry: int | None) -> deque[int]:
        hist = deque(
            [ASSOC_PAD] * max(self.params.assoc_history_len - 1, 0),
            maxlen=self.params.assoc_history_len,
        )
        hist.append(ASSOC_PAD if last_entry is None else int(last_entry))
        return hist

    def _append_assoc_history(self, track: Track, value: int) -> None:
        hist = track.metadata.get("assoc_history")
        if isinstance(hist, deque):
            hist = deque(hist, maxlen=self.params.assoc_history_len)
        else:
            hist = deque([], maxlen=self.params.assoc_history_len)
        track.metadata["assoc_history"] = hist
        hist.append(int(value))

    def _history_tail(self, track: Track) -> tuple[int, ...]:
        hist = track.metadata.get("assoc_history")
        hist = deque(
            hist if isinstance(hist, deque) else [],
            maxlen=self.params.assoc_history_len,
        )
        tail_len = min(
            int(self.params.ns_scan_window), int(self.params.assoc_history_len)
        )
        if len(hist) < tail_len:
            pad_needed = tail_len - len(hist)
            return tuple([ASSOC_PAD] * pad_needed + list(hist))
        return tuple(list(hist)[-tail_len:])

    def __init__(
        self,
        hypothesiser: PDAHypothesiser,
        updater: Updater,
        initial_tracks: Iterable[Track],
        *,
        initiator: SimpleMeasurementInitiator | None = None,
        params: TOMHTParams = TOMHTParams(),
        scoring_model: ScoringModel | None = None,
    ) -> None:
        self.hypothesiser = hypothesiser
        self.updater = updater
        self.params = params
        self.initiator = initiator
        if scoring_model is None:
            if params.scoring_mode == "beta_ratio":
                # Try to read clutter density attribute name used by different hyp types.
                clutter = getattr(hypothesiser, "clutter_density", None)
                if clutter is None:
                    clutter = getattr(hypothesiser, "clutter_spatial_density", 0.0)
                prob_detect = getattr(hypothesiser, "prob_detect", 0.9)
                prob_gate = getattr(hypothesiser, "prob_gate", params.prob_gate)
                scoring_model = BetaRatioScoringModel(
                    prob_detect=float(prob_detect),
                    prob_gate=float(prob_gate),
                    clutter_density=float(clutter) if clutter is not None else 0.0,
                    log_epsilon=params.log_epsilon,
                    fallback_unused_det_log_penalty=params.unused_det_log_penalty,
                    birth_log_penalty=params.birth_log_penalty,
                )
            else:
                raise ValueError(
                    f"Unsupported scoring_mode='{params.scoring_mode}'. "
                    "Only 'beta_ratio' is available now."
                )
        self.scoring_model = scoring_model
        # Optional sanity log for clutter term.
        if isinstance(self.scoring_model, BetaRatioScoringModel):
            clutter = self.scoring_model.clutter_density
            per_unused = self.scoring_model._per_unused_log_delta()
            if per_unused > 0.0:
                print(
                    "[WARN] per_unused_delta is positive; unused detections are rewarded. "
                    "Check clutter_density units/config."
                )
            print(
                f"[Scoring] beta_ratio: clutter_density={clutter}, "
                f"per_unused_delta={per_unused:+.3f}"
            )

        init_tracks_by_id: dict[int, Track] = {}
        max_tid = -1
        for i, tr in enumerate(list(initial_tracks)):
            tid = int(tr.metadata.get("track_id", i))
            last_det = tr.metadata.get("last_det_key", None)
            # Keep constructor defaults as before (age from len(track), hits default 0),
            # but write tracker-owned fields via the shared metadata path.
            self._write_track_maintenance_metadata(
                tr,
                track_id=tid,
                age=int(tr.metadata.get("age", len(tr))),
                hits=int(tr.metadata.get("hits", 0)),
                missed_count=int(tr.metadata.get("missed_count", 0)),
                last_det_key=last_det,
                last_det_hit=tr.metadata.get("last_det_hit", None),
            )
            init_tracks_by_id[tid] = tr
            max_tid = max(max_tid, tid)

        self._next_track_id = max_tid + 1

        self.global_hypotheses: list[GlobalHypothesis] = [
            GlobalHypothesis(tracks_by_id=init_tracks_by_id, log_weight=0.0)
        ]
        self._last_step_timestamp: object | None = None
        self.last_scan_stats: ScanStats | None = None
        self._stats: list[ScanStats] = []
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats = []
        self.last_scan_stats = None

    def print_summary_stats(self) -> None:
        stats = self._stats
        if not stats:
            print("SUMMARY scans=0 (no collected ScanStats)")
            return

        num_scans = len(stats)
        expanded = [s.globals_expanded for s in stats]
        deduped = [s.globals_after_dedupe for s in stats]
        beamed = [s.globals_after_beam for s in stats]
        after_births = [s.globals_after_births for s in stats]
        birth_created = [s.birth_tracks_created for s in stats]
        birth_kept = [s.birth_tracks_kept for s in stats]
        map_tracks = [s.map_tracks for s in stats]
        map_unused = [s.map_unused for s in stats]
        map_used = [s.map_used for s in stats]
        map_hit_rate = [s.map_mean_hit_rate for s in stats]

        def _mean(values: list[int] | list[float]) -> float:
            if not values:
                return 0.0
            return float(sum(values)) / float(len(values))

        max_globals = self.params.max_global_hypotheses
        beam_full_pre_births = sum(
            1 for s in stats if s.globals_after_beam == max_globals
        )
        beam_full_post_births = sum(
            1 for s in stats if s.globals_after_births == max_globals
        )
        scans_with_births = sum(1 for s in stats if s.birth_tracks_created > 0)
        scans_with_birth_globals = sum(1 for s in stats if s.globals_with_birth > 0)
        scans_birth_push_to_full = sum(
            1
            for s in stats
            if s.globals_after_beam < max_globals
            and s.globals_after_births == max_globals
            and s.globals_with_birth > 0
        )

        print(
            "SUMMARY "
            f"scans={num_scans} "
            f"det_total={sum(s.num_detections for s in stats)} "
            f"det_mean={_mean([s.num_detections for s in stats]):.2f}"
        )
        print(
            "SUMMARY globals "
            f"expanded med={median(expanded):.1f} max={max(expanded)} "
            f"dedup med={median(deduped):.1f} max={max(deduped)} "
            f"beam med={median(beamed):.1f} max={max(beamed)}"
        )
        print(
            "SUMMARY beam "
            f"after_births med={median(after_births):.1f} max={max(after_births)} "
            f"full_pre_births={beam_full_pre_births}/{num_scans} ({beam_full_pre_births / num_scans:.1%}) "
            f"full_post_births={beam_full_post_births}/{num_scans} ({beam_full_post_births / num_scans:.1%})"
        )
        print(
            "SUMMARY births "
            f"active={scans_with_births}/{num_scans} ({scans_with_births / num_scans:.1%}) "
            f"tracks_created med={median(birth_created):.1f} mean={_mean(birth_created):.2f} max={max(birth_created)} "
            f"tracks_kept med={median(birth_kept):.1f} mean={_mean(birth_kept):.2f} max={max(birth_kept)} "
            f"globals_with_birth={scans_with_birth_globals}/{num_scans} ({scans_with_birth_globals / num_scans:.1%}) "
            f"birth_pushes_to_full={scans_birth_push_to_full}/{num_scans} ({scans_birth_push_to_full / num_scans:.1%})"
        )
        miss_hist_all: dict[int, int] = {}
        for scan in stats:
            for misses, count in scan.map_miss_hist.items():
                miss_hist_all[misses] = miss_hist_all.get(misses, 0) + count
        miss_hist_str = (
            "{"
            + ", ".join(f"{k}: {miss_hist_all[k]}" for k in sorted(miss_hist_all))
            + "}"
        )
        print(
            "SUMMARY map "
            f"tracks med={median(map_tracks):.1f} mean={_mean(map_tracks):.2f} "
            f"used med={median(map_used):.1f} mean={_mean(map_used):.2f} "
            f"unused med={median(map_unused):.1f} mean={_mean(map_unused):.2f} "
            f"hit_rate mean={_mean(map_hit_rate):.3f} "
            f"miss_hist={miss_hist_str}"
        )

    @staticmethod
    def _det_sort_key(det: Detection) -> tuple:
        """Stable per-scan ordering key for detections."""
        ts = getattr(det, "timestamp", None)
        ts_key: tuple[int, float | str]
        if ts is None:
            ts_key = (0, 0.0)
        elif hasattr(ts, "timestamp"):
            ts_key = (1, float(ts.timestamp()))
        elif isinstance(ts, (int, float)):
            ts_key = (2, float(ts))
        else:
            ts_key = (3, str(ts))

        vec = np.asarray(det.state_vector).ravel()

        def _elem_key(x) -> tuple[int, float | str]:
            try:
                xf = float(x)
                if np.isfinite(xf):
                    return (0, xf)
                return (0, float("inf"))
            except Exception:
                return (1, str(x))

        vec_key = tuple(_elem_key(x) for x in vec)

        # Note: if two detections are perfect duplicates (same timestamp + same state_vector),
        # they are indistinguishable by content. Their relative order will then fall back to the
        # input iterable’s iteration order. Python’s sort is stable, so if the input order is
        # deterministic (e.g. a list), the result is deterministic; if the input is an unordered
        # container (e.g. a set), duplicate ordering may vary between runs.
        return (ts_key, len(vec_key), vec_key)

    def _sorted_detections(self, detections: Iterable[Detection]) -> list[Detection]:
        det_list = list(detections)
        det_list.sort(key=self._det_sort_key)
        return det_list

    def _display_global_hypotheses(self, det_list: list[Detection]) -> None:
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

    def _display_births(
        self, born: list[Track], det_index_by_obj: dict[int, int]
    ) -> None:
        for tr in born[: self.params.debug_births_max]:
            used = self._birth_used_key(tr, det_index_by_obj)
            support, age, misses = self._birth_support_age_misses(tr)
            covtr = self._birth_covar_trace(tr)
            last = tr.states[-1].state_vector
            print(
                f"  used={used}, support={support}, age={age}, misses={misses}, covtr={covtr:.2g}, "
                f"last={self._fmt_state_xyvxvy(last)}"
            )

    @staticmethod
    def _fmt_state_xyvxvy(state_vector) -> str:
        sv = np.asarray(state_vector, dtype=float)
        x = float(sv[0, 0])
        vx = float(sv[1, 0])
        y = float(sv[2, 0])
        vy = float(sv[3, 0])
        return f"(x={x:.1f}, vx={vx:.2f}, y={y:.1f}, vy={vy:.2f})"

    @staticmethod
    def _copy_track(track: Track) -> Track:
        """Copy track states and metadata; deep-copy assoc_history deque if present."""
        child = Track(list(track.states))
        child.metadata.update(track.metadata)

        hist = track.metadata.get("assoc_history")
        if isinstance(hist, deque):
            child.metadata["assoc_history"] = deque(hist, maxlen=hist.maxlen)
        return child

    def _write_track_maintenance_metadata(
        self,
        track: Track,
        *,
        track_id: int,
        age: int,
        hits: int,
        missed_count: int,
        last_det_key: int | None,
        last_det_hit: bool | None = None,
    ) -> None:
        """Write tracker-owned maintenance metadata fields in one place."""
        track.metadata["track_id"] = int(track_id)
        track.metadata["age"] = int(age)
        track.metadata["hits"] = int(hits)
        track.metadata["missed_count"] = int(missed_count)
        track.metadata["last_det_key"] = last_det_key
        track.metadata["last_det_hit"] = (
            last_det_key is not None if last_det_hit is None else bool(last_det_hit)
        )
        track.metadata["assoc_history"] = self._new_assoc_history(last_det_key)

    def _initialise_inserted_track_metadata(
        self,
        track: Track,
        *,
        track_id: int,
        age: int,
        hits: int,
        last_det_key: int | None,
        last_det_hit: bool | None = None,
    ) -> None:
        age = max(int(age), 1)
        hits = min(max(int(hits), 0), age)
        self._write_track_maintenance_metadata(
            track,
            track_id=track_id,
            age=age,
            hits=hits,
            missed_count=0,
            last_det_key=last_det_key,
            last_det_hit=last_det_hit,
        )

    def _make_external_start_template(self, start: Track, timestamp: object) -> Track:
        if len(start) == 0:
            raise ValueError(
                "External starts must contain at least one state at the current timestamp."
            )

        start_timestamp = getattr(start.states[-1], "timestamp", None)
        if start_timestamp != timestamp:
            raise ValueError(
                "External starts must already be initialised at the supplied "
                f"timestamp. Expected {timestamp!r}, got {start_timestamp!r}."
            )

        track_id = self._next_track_id
        self._next_track_id += 1

        template = self._copy_track(start)
        age = max(int(template.metadata.get("age", len(template))), 1)
        hits = int(template.metadata.get("hits", age))
        hits = min(max(hits, 1), age)
        self._initialise_inserted_track_metadata(
            template,
            track_id=track_id,
            age=age,
            hits=hits,
            last_det_key=None,
            last_det_hit=False,
        )
        return template

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
        ctx: ScanContext,
    ) -> GlobalHypothesis:
        if not ctx.detections:
            return gh
        used = self._used_det_keys_for_tracks(gh.tracks_by_id)
        delta = self.scoring_model.score_unused_detections(used_det_keys=used, ctx=ctx)
        if delta == 0.0:
            return gh
        return GlobalHypothesis(
            tracks_by_id=gh.tracks_by_id, log_weight=gh.log_weight + delta
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
        ctx: ScanContext,
    ) -> list[ChildCandidate]:
        multi = self.hypothesiser.hypothesise(track, ctx.detections, ctx.timestamp)
        singles = list(multi)

        # Precompute scores with the chosen scoring model.
        hyp_scores = self.scoring_model.score_track_hypotheses(
            track=track, hypotheses=singles, ctx=ctx
        )

        def _score_for_sort(hyp) -> float:
            # Sort/prune by log-delta (hyp_scores); fall back to probability/weight only if missing.
            score = hyp_scores.get(id(hyp))
            if score is not None:
                return float(score)
            p = getattr(hyp, "probability", None)
            if p is not None:
                try:
                    return float(p)
                except Exception:
                    return 0.0
            w = getattr(hyp, "weight", 0.0)
            try:
                return float(w)
            except Exception:
                return 0.0

        def sort_key(hyp) -> tuple[float, int]:
            p = _score_for_sort(hyp)
            if not hyp:
                return (p, -1)  # deterministic position for misses among ties
            return (p, -ctx.det_index_by_obj.get(id(hyp.measurement), 10**9))

        # Sort best-first and cap. Always keep a "miss" if present.
        singles_sorted = sorted(singles, key=sort_key, reverse=True)
        kept = singles_sorted[: self.params.max_children_per_track]
        miss = next((h for h in singles_sorted if not h), None)
        if miss is not None and miss not in kept:
            kept.append(miss)

        candidates: list[ChildCandidate] = []
        for hyp in kept:
            child = self._copy_track(track)
            child.metadata.setdefault("track_id", track_id)

            if not hyp:
                child.append(hyp.prediction)
                child.metadata["missed_count"] = (
                    int(track.metadata.get("missed_count", 0)) + 1
                )
                used = None
                self._append_assoc_history(child, ASSOC_MISS)
            else:
                upd = self.updater.update(hyp)
                child.append(upd)
                child.metadata["missed_count"] = 0
                used = ctx.det_index_by_obj[id(hyp.measurement)]
                self._append_assoc_history(child, used)

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
                    log_delta=float(hyp_scores.get(id(hyp), 0.0)),
                )
            )

        candidates.sort(key=lambda c: c.log_delta, reverse=True)
        return candidates

    def _expand_global_hypothesis(
        self,
        gh: GlobalHypothesis,
        ctx: ScanContext,
    ) -> list[GlobalHypothesis]:
        track_ids = sorted(gh.tracks_by_id.keys())

        per_track_candidates: list[list[ChildCandidate]] = []
        for tid in track_ids:
            per_track_candidates.append(
                self._candidates_for_track(
                    tid,
                    gh.tracks_by_id[tid],
                    ctx,
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
        ctx: ScanContext,
    ) -> BirthStats:
        globals_before_births = len(self.global_hypotheses)
        if self.initiator is not None and self.global_hypotheses:
            residual = self._residual_detections(self.global_hypotheses, ctx.detections)
            residual_detections_considered = len(residual)

            born = (
                list(self.initiator.initiate(OrderedSet(residual), ctx.timestamp))
                if residual
                else []
            )
            birth_tracks_created = len(born)

            born = [tr for tr in born if self._birth_is_sane(tr)]

            born_scored: list[tuple[tuple, Track]] = []
            for tr in born:
                used = self._birth_used_key(tr, ctx.det_index_by_obj)
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
                    f"\nBirth candidates at {ctx.timestamp} (pre-limit): {len(born_scored)}"
                )
                self._display_births(born, ctx.det_index_by_obj)

            born = born[: self.params.max_births_per_scan]
            birth_tracks_kept = len(born)

            if self.params.debug_display_births:
                print(f"Births kept (post-limit): {len(born)}")
                self._display_births(born, ctx.det_index_by_obj)

            birth_track_ids: set[int] = set()
            if born:
                # Allocate stable IDs once for these births (shared across variants)
                birth_templates: list[tuple[int, Track, int | None]] = []
                for tr in born:
                    tid = self._next_track_id
                    self._next_track_id += 1
                    used_key = self._birth_used_key(tr, ctx.det_index_by_obj)
                    birth_templates.append((tid, tr, used_key))
                    birth_track_ids.add(tid)

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

                    # Always allow "no birth" variant (except for empty start heuristic, see above)
                    # and then branch with births one-by-one.
                    for tid, template, used in compatible:
                        tracks_by_id = dict(gh.tracks_by_id)

                        tr_copy = self._copy_track(template)
                        self._initialise_inserted_track_metadata(
                            tr_copy,
                            track_id=tid,
                            age=1,
                            hits=1 if used is not None else 0,
                            last_det_key=used,
                        )

                        tracks_by_id[tid] = tr_copy
                        birth_delta = self.scoring_model.score_birth(
                            birth_track=tr_copy,
                            used_det_key=used,
                            ctx=ctx,
                        )
                        new_globals.append(
                            GlobalHypothesis(
                                tracks_by_id=tracks_by_id,
                                log_weight=gh.log_weight + birth_delta,
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
                                birth_delta_total = 0.0

                                for tid, template, used in [
                                    (tid1, t1, u1),
                                    (tid2, t2, u2),
                                ]:
                                    tr_copy = self._copy_track(template)
                                    self._initialise_inserted_track_metadata(
                                        tr_copy,
                                        track_id=tid,
                                        age=1,
                                        hits=1 if used is not None else 0,
                                        last_det_key=used,
                                    )
                                    tracks_by_id[tid] = tr_copy

                                    birth_delta_total += self.scoring_model.score_birth(
                                        birth_track=tr_copy,
                                        used_det_key=used,
                                        ctx=ctx,
                                    )

                                new_globals.append(
                                    GlobalHypothesis(
                                        tracks_by_id=tracks_by_id,
                                        log_weight=gh.log_weight + birth_delta_total,
                                    )
                                )

                new_globals.sort(key=lambda g: g.log_weight, reverse=True)
                self.global_hypotheses = new_globals[
                    : self.params.max_global_hypotheses
                ]

            birth_track_instances_in_beam = 0
            globals_with_birth = 0
            if birth_track_ids:
                birth_track_instances_in_beam = sum(
                    1
                    for gh in self.global_hypotheses
                    for tr in gh.tracks_by_id.values()
                    if int(tr.metadata.get("track_id", -1)) in birth_track_ids
                )
                globals_with_birth = sum(
                    1
                    for gh in self.global_hypotheses
                    if any(
                        int(tr.metadata.get("track_id", -1)) in birth_track_ids
                        for tr in gh.tracks_by_id.values()
                    )
                )

            return BirthStats(
                residual_detections_considered=residual_detections_considered,
                birth_tracks_created=birth_tracks_created,
                birth_tracks_kept=birth_tracks_kept,
                birth_track_instances_in_beam=birth_track_instances_in_beam,
                globals_with_birth=globals_with_birth,
                globals_before_births=globals_before_births,
                globals_after_births=len(self.global_hypotheses),
            )
        return BirthStats(
            globals_before_births=globals_before_births,
            globals_after_births=len(self.global_hypotheses),
        )

    def _dedupe_globals_by_history(
        self, globals: list[GlobalHypothesis]
    ) -> list[GlobalHypothesis]:
        """Keep best log_weight per (track_id, history_tail) signature."""
        best: dict[tuple[tuple[int, tuple[int, ...]], ...], GlobalHypothesis] = {}
        for gh in globals:
            sig = tuple(
                sorted(
                    (tid, self._history_tail(tr)) for tid, tr in gh.tracks_by_id.items()
                )
            )
            prev = best.get(sig)
            if prev is None or gh.log_weight > prev.log_weight:
                best[sig] = gh
        return list(best.values())

    def _validate_external_starts_timestamp(self, timestamp: object) -> None:
        if self._last_step_timestamp is None:
            raise RuntimeError(
                "add_external_starts() requires a completed step() first."
            )
        if timestamp != self._last_step_timestamp:
            raise ValueError(
                "add_external_starts() timestamp must match the most recent "
                f"completed step() timestamp. Expected {self._last_step_timestamp!r}, "
                f"got {timestamp!r}."
            )

    def add_external_starts(self, starts: Iterable[Track], timestamp: object) -> None:
        """
        Insert confirmed external starts into each current global hypothesis.

        Duplicate-like inputs are not deduplicated here: each supplied start is
        treated as a distinct confirmed track and receives a fresh tracker-owned
        track_id.
        """
        self._validate_external_starts_timestamp(timestamp)
        start_list = list(starts)
        if not start_list or not self.global_hypotheses:
            return

        templates = [
            self._make_external_start_template(start, timestamp) for start in start_list
        ]

        new_globals: list[GlobalHypothesis] = []
        for gh in self.global_hypotheses:
            tracks_by_id = dict(gh.tracks_by_id)
            for template in templates:
                tid = int(template.metadata["track_id"])
                tracks_by_id[tid] = self._copy_track(template)
            new_globals.append(
                GlobalHypothesis(
                    tracks_by_id=tracks_by_id,
                    log_weight=gh.log_weight,
                )
            )
        self.global_hypotheses = new_globals

    def step(self, detections: Iterable[Detection], timestamp) -> set[Track]:
        globals_in = len(self.global_hypotheses)
        det_list = self._sorted_detections(detections)
        # Use id(det) because Stone Soup hypotheses keep the original Detection
        # objects; hash/equality isn't defined on Detection, but identity is
        # stable within a scan.
        det_index_by_obj = {id(det): i for i, det in enumerate(det_list)}
        ctx = ScanContext(
            timestamp=timestamp, detections=det_list, det_index_by_obj=det_index_by_obj
        )

        # Expand globals with current batch of detections
        expanded: list[GlobalHypothesis] = []
        for gh in self.global_hypotheses:
            expanded.extend(self._expand_global_hypothesis(gh, ctx))
        globals_expanded = len(expanded)

        # Apply unused detection penalty
        expanded = [self._apply_unused_detection_penalty(gh, ctx) for gh in expanded]
        globals_after_unused = len(expanded)

        # Dedupe
        expanded = self._dedupe_globals_by_history(expanded)
        globals_after_dedupe = len(expanded)

        # Keep top-K globals (beam)
        expanded.sort(key=lambda g: g.log_weight, reverse=True)
        self.global_hypotheses = expanded[: self.params.max_global_hypotheses]
        globals_after_beam = len(self.global_hypotheses)

        # Births: run initiator once on residual detections, then branch globals with/without births.
        birth_stats = self._branch_globals_with_births(ctx)

        if self.params.debug_display_detections:
            print(f"\nDetections at timestamp {timestamp}:")
            for det in det_list:
                print(f"  {det.state_vector}")

        if self.params.debug_display_hypotheses:
            print(f"\nGlobal hypotheses at timestamp {timestamp}:")
            self._display_global_hypotheses(det_list)

        map_tracks = 0
        map_used = 0
        map_unused = len(det_list)
        map_miss_hist: dict[int, int] = {}
        map_mean_hit_rate = 0.0
        if self.global_hypotheses:
            best = self.global_hypotheses[0]
            map_tracks = len(best.tracks_by_id)
            map_used = len(self._used_det_keys_for_tracks(best.tracks_by_id))
            map_unused = len(det_list) - map_used

            hit_rates: list[float] = []
            for tr in best.tracks_by_id.values():
                misses = int(tr.metadata.get("missed_count", 0))
                map_miss_hist[misses] = map_miss_hist.get(misses, 0) + 1
                age = int(tr.metadata.get("age", len(tr)))
                if age > 0:
                    hits = int(tr.metadata.get("hits", 0))
                    hit_rates.append(float(hits) / float(age))
            if hit_rates:
                map_mean_hit_rate = float(np.mean(hit_rates))

        scan_stats = ScanStats(
            timestamp=timestamp,
            num_detections=len(det_list),
            globals_in=globals_in,
            globals_expanded=globals_expanded,
            globals_after_unused=globals_after_unused,
            globals_after_dedupe=globals_after_dedupe,
            globals_after_beam=globals_after_beam,
            globals_after_births=birth_stats.globals_after_births,
            birth_candidates=birth_stats.residual_detections_considered,
            birth_tracks_created=birth_stats.birth_tracks_created,
            birth_tracks_kept=birth_stats.birth_tracks_kept,
            birth_track_instances_in_beam=birth_stats.birth_track_instances_in_beam,
            globals_with_birth=birth_stats.globals_with_birth,
            map_tracks=map_tracks,
            map_used=map_used,
            map_unused=map_unused,
            map_miss_hist=map_miss_hist,
            map_mean_hit_rate=map_mean_hit_rate,
        )
        self.last_scan_stats = scan_stats
        if self.params.collect_stats:
            self._stats.append(scan_stats)

        if self.params.debug_display_scan_stats:
            print(
                f"SCAN t={timestamp} det={scan_stats.num_detections} "
                f"globals in={scan_stats.globals_in} exp={scan_stats.globals_expanded} "
                f"after_unused={scan_stats.globals_after_unused} dedup={scan_stats.globals_after_dedupe} "
                f"beam={scan_stats.globals_after_beam} births cand={scan_stats.birth_candidates} "
                f"tracks_created={scan_stats.birth_tracks_created} tracks_kept={scan_stats.birth_tracks_kept} "
                f"beam_inst={scan_stats.birth_track_instances_in_beam} "
                f"globals_with_birth={scan_stats.globals_with_birth} "
                f"after={scan_stats.globals_after_births} MAP tracks={scan_stats.map_tracks} "
                f"used={scan_stats.map_used} unused={scan_stats.map_unused} "
                f"hit_rate={scan_stats.map_mean_hit_rate:.2f}"
            )
            if self.params.debug_display_map_miss_hist:
                print(
                    f"SCAN_MAP_MISS_HIST t={timestamp} miss_hist={scan_stats.map_miss_hist}"
                )

        self._last_step_timestamp = timestamp

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
