"""Track-oriented MHT with node-based internals and Stone Soup-facing APIs.

Typical usage pattern:
```python
tracker = TOMHTTracker(hypothesiser,updater,initiator=initiator,params=params)
t,map_tracks = tracker.update_tracker(time,detections)
tracker.add_external_starts(t,confirmed_starts_at_t)  # optional, same timestamp
map_tracks = tracker.tracks  # or use ``map_tracks`` from update_tracker()
```

Core control APIs:
- ``update_tracker(time,detections)``: process one scan and return MAP output.
- ``tracks``: current MAP output as Stone Soup ``Track`` objects.
- ``add_external_starts(time,starts)``: inject confirmed external starts after the
  same-timestamp ``update_tracker()`` call.

Read-only inspection helpers (not required for normal operation):
- ``get_map_hypothesis_snapshot()``
- ``get_map_output_tracks()``
- ``get_n_scan_commitment_snapshot()``

Boundary model:
- External interface stays Stone Soup-native (detections/tracks/hypothesiser/updater).
- Internal branch structure is node-native (``track_id -> leaf node`` per global).
- N-scan commitment is maintained by ancestor agreement at boundary ``b = k - N``.
- Stone Soup ``Track`` objects remain the compatibility boundary for output and
  integration-facing interfaces.
"""

from dataclasses import dataclass
import datetime
import resource
import sys
import time as wall_clock
from types import MappingProxyType
from ordered_set import OrderedSet
from typing import Any, Iterable, Mapping

import numpy as np

from stonesoup.base import Property
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.initiator.base import Initiator
from stonesoup.types.state import State
from stonesoup.tracker.base import Tracker, _TrackerMixInUpdate
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.types.update import Update
from stonesoup.updater.base import Updater

from mht.tomht_model import (
    ChildCandidate,
    GlobalHypothesis,
    MAPHypothesisSnapshot,
    NScanCommitmentSnapshot,
    TrackHypothesisNode,
)
from mht.tomht_output import reconstruct_track_from_leaf_node
from mht.tomht_scoring import BetaRatioScoringModel, ScoringModel
from mht.tomht_stats import (
    BirthStats,
    ScanStats,
    print_scan_stats as print_scan_stats_report,
    print_summary_stats as print_summary_stats_report,
)

# ============================================================================
# Tracker-Local Support Structures / Params
# ============================================================================


@dataclass(frozen=True)
class TOMHTParams:
    max_global_hypotheses: int = 20
    max_children_per_track: int = 5
    max_missed: int = 5
    log_epsilon: float = 1e-12
    scoring_mode: str = "beta_ratio"  # Only beta_ratio is supported.

    # Sole N-scan configuration knob.
    ns_scan_window: int = 3

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


@dataclass(frozen=True)
class ScanContext:
    """Per-scan context passed into scoring models."""

    scan_index: int
    timestamp: datetime.datetime
    detections: list[Detection]
    det_index_by_obj: dict[int, int]


@dataclass(frozen=True)
class BirthTemplate:
    track_id: int
    leaf_node: TrackHypothesisNode
    template_track: Track
    used_det_key: int | None


# ============================================================================
# Tracker Implementation
# ============================================================================


class TOMHTTracker(_TrackerMixInUpdate, Tracker):
    """Public TO-MHT tracker contract with Stone Soup boundary objects.

    Core operational APIs:
    - ``update_tracker(time,detections)``: main per-scan update; returns
      ``(time, map_tracks)``.
    - ``tracks``: current MAP output tracks.
    - ``add_external_starts(time,starts)``: inject externally confirmed starts
      for the most recently processed timestamp (call after that update).

    Read-only inspection/debug APIs:
    - ``get_map_hypothesis_snapshot()``: node-native MAP leaf snapshot.
    - ``get_map_output_tracks()``: reconstructed MAP output ``Track`` objects.
    - ``get_n_scan_commitment_snapshot()``: current ancestor-identity N-scan
      commitment bookkeeping.

    Internal model:
    - ``TrackHypothesisNode`` is the branching unit.
    - each global stores one current leaf node per logical ``track_id``.
    - commitment is maintained by ancestor agreement after beam pruning.

    API boundary remains Stone Soup-native; internal hypothesis evolution is
    node-based.
    """

    ASSOC_PAD = -1
    ASSOC_MISS = -2

    hypothesiser: PDAHypothesiser = Property(
        doc="Hypothesiser used to branch per-track hypotheses."
    )
    updater: Updater = Property(
        doc="Updater used to generate posteriors from selected hypotheses."
    )

    _last_nscan_boundary_scan_index: int | None
    _last_nscan_tracks_in_scope: int
    _last_nscan_committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]
    _committed_boundary_by_track_id: dict[int, int]
    _committed_ancestor_by_track_id: dict[int, TrackHypothesisNode]

    # =========================================================================
    # Public API
    # =========================================================================
    #
    # Class roadmap:
    # 1) Public API + top-level scan pipeline.
    # 2) Scan pipeline support (ordering, stats, instrumentation).
    # 3) Continuation/expansion and dedupe/beam/N-scan helpers.
    # 4) Internal birth helpers.
    # 5) Node lifecycle and external-start helpers.
    # 6) Diagnostics/instrumentation helpers.

    def __init__(
        self,
        hypothesiser: PDAHypothesiser,
        updater: Updater,
        *,
        detector: Any | None = None,
        initiator: Initiator | None = None,
        params: TOMHTParams = TOMHTParams(),
        scoring_model: ScoringModel | None = None,
    ) -> None:
        """Construct the tracker with Stone Soup components and TO-MHT params.

        Runtime note: type hints name ``PDAHypothesiser``, but the runtime
        contract used by this tracker is narrower duck-typing around
        ``hypothesise(...)`` outputs and per-hypothesis fields.
        """
        super().__init__(hypothesiser, updater)
        self.detector = detector
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

        self._next_track_id = 0
        self._next_node_id = 0
        self._nodes_by_id: dict[int, TrackHypothesisNode] = {}
        # N-scan commitment bookkeeping; physical node cleanup is intentionally
        # deferred to a later phase.
        self._last_nscan_boundary_scan_index = None
        self._last_nscan_tracks_in_scope = 0
        self._last_nscan_committed_ancestor_by_track_id = {}
        self._committed_boundary_by_track_id = {}
        self._committed_ancestor_by_track_id = {}

        self.global_hypotheses: list[GlobalHypothesis] = [
            GlobalHypothesis(leaf_nodes_by_track_id={}, log_weight=0.0)
        ]
        self._last_update_timestamp: datetime.datetime | None = None
        self._last_scan_index: int | None = None
        self._last_unused_detections: list[Detection] = []
        self.last_scan_stats: ScanStats | None = None
        self._stats: list[ScanStats] = []
        self.reset_stats()

    def reset_stats(self) -> None:
        """Clear collected ScanStats and the last per-scan snapshot."""
        self._stats = []
        self.last_scan_stats = None

    @property
    def tracks(self) -> set[Track]:
        """Current MAP output as Stone Soup ``Track`` objects.

        Equivalent content to ``get_map_output_tracks()``.
        """
        return self.get_map_output_tracks()

    def update_tracker(
        self,
        time: datetime.datetime,
        detections: Iterable[Detection],
    ) -> tuple[datetime.datetime, set[Track]]:
        """Run the main one-scan update and return ``(time, MAP tracks)``.

        This is the primary operational entry point. The returned tracks are the
        current MAP output as reconstructed Stone Soup ``Track`` objects.
        MAP here means maximum a posteriori, i.e. the highest-weight current
        global hypothesis.
        If externally confirmed starts exist at this timestamp, call
        ``add_external_starts(time,starts)`` after this method for the same
        ``time``.

        Pipeline order:
        1. Sort detections deterministically for stable indexing.
        2. Expand globals via per-track child-node candidates.
        3. Apply unused-detection score term.
        4. Dedupe globals by structural leaf-node identity.
        5. Beam prune to top-K globals.
        6. Update ancestor-based N-scan commitment (pre-birth boundary).
        7. Optionally apply internal births from residual detections.
        8. Return current MAP output tracks.
        """
        # Setup / preparation.
        scan_wall_start_ns = wall_clock.perf_counter_ns()
        self._last_unused_detections = []
        globals_in = len(self.global_hypotheses)
        scan_index = (
            0 if self._last_scan_index is None else int(self._last_scan_index) + 1
        )
        det_list = self._sorted_detections(detections)
        # Use id(det) because Stone Soup hypotheses keep the original Detection
        # objects; hash/equality isn't defined on Detection, but identity is
        # stable within a scan.
        det_index_by_obj = {id(det): i for i, det in enumerate(det_list)}
        ctx = ScanContext(
            scan_index=scan_index,
            timestamp=time,
            detections=det_list,
            det_index_by_obj=det_index_by_obj,
        )

        # Core scan pipeline:
        # sort detections -> expand -> unused-score -> dedupe -> beam ->
        # N-scan commitment -> bounded birth phase -> MAP output.

        # Expand globals with current batch of detections
        expanded: list[GlobalHypothesis] = []
        for gh in self.global_hypotheses:
            expanded.extend(self._expand_global_hypothesis(gh, ctx))
        globals_expanded = len(expanded)

        # Apply unused detection penalty
        expanded = [self._apply_unused_detection_penalty(gh, ctx) for gh in expanded]
        globals_after_unused = len(expanded)

        # Dedupe
        expanded = self._dedupe_globals_by_leaf_identity(expanded)
        globals_after_dedupe = len(expanded)

        # Keep top-K globals (beam)
        expanded.sort(key=lambda g: g.log_weight, reverse=True)
        self.global_hypotheses = expanded[: self.params.max_global_hypotheses]
        globals_after_beam = len(self.global_hypotheses)
        (
            nscan_boundary_scan_index,
            nscan_tracks_in_scope,
            nscan_tracks_committed,
        ) = self._update_n_scan_commitment(
            scan_index=scan_index,
            post_beam_globals=self.global_hypotheses,
        )

        # Births run as a separate bounded phase after continuation/beam/N-scan:
        # residual detections are processed once, with per-scan birth limits and
        # post-branch beam truncation enforced in the birth path.
        birth_stats = self._branch_globals_with_births(ctx)

        # Post-pipeline bookkeeping and instrumentation.
        scan_wall_ms = (wall_clock.perf_counter_ns() - scan_wall_start_ns) / 1e6
        maxrss_mb = self._get_process_maxrss_mb()
        node_count_total = len(self._nodes_by_id)
        leaf_instances_in_beam = sum(
            len(gh.leaf_nodes_by_track_id) for gh in self.global_hypotheses
        )

        self._run_scan_instrumentation(
            ctx=ctx,
            scan_wall_ms=scan_wall_ms,
            maxrss_mb=maxrss_mb,
            node_count_total=node_count_total,
            leaf_instances_in_beam=leaf_instances_in_beam,
            globals_in=globals_in,
            globals_expanded=globals_expanded,
            globals_after_unused=globals_after_unused,
            globals_after_dedupe=globals_after_dedupe,
            globals_after_beam=globals_after_beam,
            nscan_boundary_scan_index=nscan_boundary_scan_index,
            nscan_tracks_in_scope=nscan_tracks_in_scope,
            nscan_tracks_committed=nscan_tracks_committed,
            birth_stats=birth_stats,
        )

        self._last_update_timestamp = time
        self._last_scan_index = scan_index

        # Explicit MAP output step (maximum a posteriori / highest-weight global).
        map_output_tracks = self.get_map_output_tracks()
        return time, map_output_tracks

    def add_external_starts(
        self, time: datetime.datetime, starts: Iterable[Track]
    ) -> None:
        """Inject externally confirmed starts for the latest processed timestamp.

        Intended call pattern:
        1. Run ``update_tracker(time,detections)`` for timestamp ``time``.
        2. Call ``add_external_starts(time,starts)`` with confirmed starts at that
           same ``time``; starts are inserted into the current post-update globals.
        Timestamp/order enforcement is handled by
        ``_validate_external_starts_timestamp(...)``.

        Duplicate-like inputs are not deduplicated here: each supplied start is
        treated as a distinct confirmed track and receives a fresh tracker-owned
        track_id.
        """
        self._validate_external_starts_timestamp(time)
        start_list = list(starts)
        if not start_list or not self.global_hypotheses:
            return

        templates = [
            self._make_external_start_template(start, time) for start in start_list
        ]

        new_globals: list[GlobalHypothesis] = []
        for gh in self.global_hypotheses:
            leaf_nodes_by_track_id = dict(gh.leaf_nodes_by_track_id)
            for template in templates:
                leaf_nodes_by_track_id[template.track_id] = template
            new_globals.append(
                GlobalHypothesis(
                    leaf_nodes_by_track_id=leaf_nodes_by_track_id,
                    log_weight=gh.log_weight,
                )
            )
        self.global_hypotheses = new_globals

    def get_unused_detections(self) -> list[Detection]:
        """
        Return residual detections from the most recent completed update_tracker().

        Residual detections are considered consumed when internal births are enabled
        (i.e. ``initiator is not None``), so this returns an empty list in that mode.
        """
        if self._last_update_timestamp is None:
            raise RuntimeError(
                "get_unused_detections() requires a completed update_tracker() first."
            )
        return list(self._last_unused_detections)

    def get_map_hypothesis_snapshot(self) -> MAPHypothesisSnapshot | None:
        """Return read-only node-native MAP state for inspection/debug only."""
        if not self.global_hypotheses:
            return None
        best = self.global_hypotheses[0]
        return MAPHypothesisSnapshot(
            log_weight=float(best.log_weight),
            leaf_nodes_by_track_id=MappingProxyType(dict(best.leaf_nodes_by_track_id)),
        )

    def get_map_output_tracks(self) -> set[Track]:
        """Return current MAP outputs as Stone Soup ``Track`` objects."""
        map_snapshot = self.get_map_hypothesis_snapshot()
        if map_snapshot is None:
            return set()
        return {
            reconstruct_track_from_leaf_node(leaf_node)
            for leaf_node in map_snapshot.leaf_nodes_by_track_id.values()
        }

    def get_n_scan_commitment_snapshot(self) -> NScanCommitmentSnapshot:
        """Return read-only N-scan commitment bookkeeping (inspection/debug)."""
        return NScanCommitmentSnapshot(
            boundary_scan_index=self._last_nscan_boundary_scan_index,
            tracks_in_scope=int(self._last_nscan_tracks_in_scope),
            latest_committed_ancestor_by_track_id=dict(
                self._last_nscan_committed_ancestor_by_track_id
            ),
            committed_boundary_by_track_id=dict(self._committed_boundary_by_track_id),
            committed_ancestor_by_track_id=dict(self._committed_ancestor_by_track_id),
        )

    def print_summary_stats(self) -> None:
        """Print aggregate instrumentation summaries from collected ScanStats."""
        print_summary_stats_report(
            stats=self._stats,
            max_global_hypotheses=self.params.max_global_hypotheses,
            last_nscan_boundary_scan_index=self._last_nscan_boundary_scan_index,
            committed_boundary_by_track_id=self._committed_boundary_by_track_id,
        )

    # =========================================================================
    # Scan Pipeline Support
    # =========================================================================
    # Detection ordering and low-level scan utilities.

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

    @staticmethod
    def _get_process_maxrss_mb() -> float:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss units differ by platform:
        # - macOS: bytes
        # - Linux: KiB
        if sys.platform == "darwin":
            return float(ru.ru_maxrss) / (1024.0 * 1024.0)
        return float(ru.ru_maxrss) / 1024.0

    # Shared detection-usage utilities.

    def _used_det_keys_for_leaf_nodes(
        self, leaf_nodes_by_track_id: Mapping[int, TrackHypothesisNode]
    ) -> set[int]:
        used: set[int] = set()
        for leaf_node in leaf_nodes_by_track_id.values():
            if leaf_node.used_det_key is not None:
                used.add(int(leaf_node.used_det_key))
        return used

    def _used_det_keys_in_global(self, gh: GlobalHypothesis) -> set[int]:
        return self._used_det_keys_for_leaf_nodes(gh.leaf_nodes_by_track_id)

    # =========================================================================
    # Continuation / Expansion Helpers
    # =========================================================================
    # Local per-track continuation candidate generation.

    def _candidate_from_hypothesis(
        self,
        *,
        leaf_node: TrackHypothesisNode,
        hypothesis: Any,
        ctx: ScanContext,
        log_delta: float,
    ) -> ChildCandidate:
        """Map one Stone Soup hypothesis to one node-native child candidate.

        Miss and hit paths are handled explicitly: miss uses the hypothesis
        prediction and increments the miss streak; hit runs updater/update,
        binds a detection key, and resets the miss streak. Cached maintenance
        fields are updated when creating the child node.
        """
        if not hypothesis:
            state = getattr(hypothesis, "prediction")
            used_det_key = None
            assoc_label = TOMHTTracker.ASSOC_MISS
            state_kind = "prediction"
            missed_count = int(leaf_node.missed_count) + 1
            last_det_key = leaf_node.last_det_key
        else:
            state = self.updater.update(hypothesis)
            used_det_key = int(ctx.det_index_by_obj[id(hypothesis.measurement)])
            assoc_label = used_det_key
            state_kind = "update"
            missed_count = 0
            last_det_key = used_det_key

        child_node = self._create_track_hypothesis_node(
            track_id=leaf_node.track_id,
            parent=leaf_node,
            scan_index=ctx.scan_index,
            timestamp=getattr(state, "timestamp", ctx.timestamp),
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=assoc_label,
            log_delta=log_delta,
            age=int(leaf_node.age) + 1,
            hits=int(leaf_node.hits) + (0 if used_det_key is None else 1),
            missed_count=missed_count,
            last_det_key=last_det_key,
            last_det_hit=used_det_key is not None,
            root_source=leaf_node.root_source,
            birth_scan_index=leaf_node.birth_scan_index,
        )
        return ChildCandidate(
            track_id=leaf_node.track_id,
            child_node=child_node,
            used_det_key=used_det_key,
            log_delta=log_delta,
        )

    def _candidates_for_track_leaf(
        self,
        leaf_node: TrackHypothesisNode,
        ctx: ScanContext,
    ) -> list[ChildCandidate]:
        """Build retained local continuation candidates for one track leaf.

        This helper is the continuation boundary adapter: it reconstructs a
        temporary ``Track`` for hypothesiser/updater compatibility, scores and
        prunes local hypotheses to the per-track limit (keeping one miss if
        present), then translates retained hypotheses back to node-native
        ``ChildCandidate`` objects.
        """
        track = reconstruct_track_from_leaf_node(leaf_node)
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

        candidates = [
            self._candidate_from_hypothesis(
                leaf_node=leaf_node,
                hypothesis=hyp,
                ctx=ctx,
                log_delta=float(hyp_scores.get(id(hyp), 0.0)),
            )
            for hyp in kept
        ]
        candidates.sort(key=lambda c: c.log_delta, reverse=True)
        return candidates

    # Joint global expansion under exclusivity constraints.

    def _assemble_joint_globals(
        self,
        *,
        track_ids: list[int],
        per_track_candidates: list[list[ChildCandidate]],
        base_log_weight: float,
    ) -> list[GlobalHypothesis]:
        """Combine local candidates into exclusivity-valid joint globals.

        Backtracks over per-track ``ChildCandidate`` lists, enforces
        one-detection-per-global exclusivity across tracks, drops only tracks
        whose child ``missed_count`` exceeds ``max_missed`` (branch continues),
        and accumulates the resulting global log weights.
        """
        new_globals: list[GlobalHypothesis] = []

        def _backtrack(
            i: int,
            used_det_keys: set[int],
            acc_leaf_nodes: dict[int, TrackHypothesisNode],
            acc_log: float,
        ) -> None:
            if i == len(track_ids):
                new_globals.append(
                    GlobalHypothesis(
                        leaf_nodes_by_track_id=dict(acc_leaf_nodes),
                        log_weight=acc_log,
                    )
                )
                return

            tid = track_ids[i]
            for cand in per_track_candidates[i]:
                if cand.track_id != tid:
                    raise RuntimeError(
                        "ChildCandidate track_id mismatch during global expansion."
                    )
                if cand.used_det_key is not None and cand.used_det_key in used_det_keys:
                    continue

                child = cand.child_node
                missed = int(child.missed_count)
                if missed > self.params.max_missed:
                    # Drop only this logical track from the branch and
                    # continue combining remaining tracks.
                    _backtrack(
                        i + 1,
                        used_det_keys,
                        acc_leaf_nodes,
                        acc_log + cand.log_delta,
                    )
                    continue

                if cand.used_det_key is not None:
                    used_det_keys.add(cand.used_det_key)
                acc_leaf_nodes[tid] = child
                _backtrack(
                    i + 1, used_det_keys, acc_leaf_nodes, acc_log + cand.log_delta
                )
                acc_leaf_nodes.pop(tid, None)
                if cand.used_det_key is not None:
                    used_det_keys.remove(cand.used_det_key)

        _backtrack(0, set(), dict(), base_log_weight)
        return new_globals

    def _expand_global_hypothesis(
        self,
        gh: GlobalHypothesis,
        ctx: ScanContext,
    ) -> list[GlobalHypothesis]:
        """Expand one global by combining per-track local candidates.

        This gathers retained local candidates for each track in ``gh``, then
        assembles exclusivity-valid joint globals while preserving the existing
        drop semantics for tracks that exceed ``max_missed``.
        """
        track_ids = sorted(gh.leaf_nodes_by_track_id.keys())

        per_track_candidates: list[list[ChildCandidate]] = []
        for tid in track_ids:
            per_track_candidates.append(
                self._candidates_for_track_leaf(gh.leaf_nodes_by_track_id[tid], ctx)
            )

        return self._assemble_joint_globals(
            track_ids=track_ids,
            per_track_candidates=per_track_candidates,
            base_log_weight=gh.log_weight,
        )

    # =========================================================================
    # Dedupe / Beam Helpers
    # =========================================================================
    # Sequence here: score expanded globals for unused detections, collapse
    # exact structural duplicates, then beam pruning runs in update_tracker().

    def _apply_unused_detection_penalty(
        self,
        gh: GlobalHypothesis,
        ctx: ScanContext,
    ) -> GlobalHypothesis:
        """Apply the scoring-model term for detections unused by this global.

        Runs after continuation expansion and before dedupe/beam pruning.
        """
        if not ctx.detections:
            return gh
        used = self._used_det_keys_for_leaf_nodes(gh.leaf_nodes_by_track_id)
        delta = self.scoring_model.score_unused_detections(used_det_keys=used, ctx=ctx)
        if delta == 0.0:
            return gh
        return GlobalHypothesis(
            leaf_nodes_by_track_id=gh.leaf_nodes_by_track_id,
            log_weight=gh.log_weight + delta,
        )

    def _leaf_signature_for_global(
        self, gh: GlobalHypothesis
    ) -> tuple[tuple[int, int], ...]:
        """Return this global's structural dedupe signature.

        Signature = active leaf-node identity per track_id.
        """
        return tuple(
            sorted(
                (track_id, leaf_node.node_id)
                for track_id, leaf_node in gh.leaf_nodes_by_track_id.items()
            )
        )

    def _dedupe_globals_by_leaf_identity(
        self, globals: list[GlobalHypothesis]
    ) -> list[GlobalHypothesis]:
        """
        Keep best log_weight per structural leaf signature.

        Two globals are duplicates only when they contain the same leaf node for
        every active track_id.
        """
        best: dict[tuple[tuple[int, int], ...], GlobalHypothesis] = {}
        for gh in globals:
            sig = self._leaf_signature_for_global(gh)
            prev = best.get(sig)
            if prev is None or gh.log_weight > prev.log_weight:
                best[sig] = gh
        return list(best.values())

    # =========================================================================
    # N-Scan Commitment Helpers
    # =========================================================================
    # Ancestor-based commitment at boundary b = k - N.
    # Pipeline note: update_tracker() applies this before the birth phase.

    def _ancestor_at_scan_boundary(
        self,
        leaf_node: TrackHypothesisNode,
        boundary_scan_index: int,
    ) -> TrackHypothesisNode | None:
        """
        Return this track's ancestor node exactly at `boundary_scan_index`.

        Returns None when no exact-boundary ancestor exists, for example:
        - boundary is before this track's birth/root scan, or
        - ancestry has no node exactly at that scan index.
        """
        if boundary_scan_index < 0:
            return None
        node: TrackHypothesisNode | None = leaf_node
        while node is not None and int(node.scan_index) > boundary_scan_index:
            node = node.parent
        if node is None:
            return None
        if int(node.scan_index) != boundary_scan_index:
            return None
        return node

    def _compute_committed_track_ancestors_at_boundary(
        self,
        post_beam_globals: list[GlobalHypothesis],
        boundary_scan_index: int,
    ) -> tuple[dict[int, TrackHypothesisNode], int]:
        """
        Compute per-track ancestor-identity agreement at one N-scan boundary.

        The agreement set is per-track and only uses globals that still contain
        that track_id. Track absence in some globals is not disagreement.

        Conservative choice: if any participating global has no exact-boundary
        ancestor for a track (e.g. track born after boundary), that track is not
        marked committed at this boundary.
        """
        if boundary_scan_index < 0 or not post_beam_globals:
            return {}, 0

        track_ids = sorted(
            {
                track_id
                for gh in post_beam_globals
                for track_id in gh.leaf_nodes_by_track_id.keys()
            }
        )

        committed: dict[int, TrackHypothesisNode] = {}
        tracks_in_scope = 0
        for track_id in track_ids:
            boundary_ancestors: list[TrackHypothesisNode] = []
            missing_exact_boundary = False
            participating_globals = 0

            for gh in post_beam_globals:
                leaf_node = gh.leaf_nodes_by_track_id.get(track_id)
                if leaf_node is None:
                    continue
                participating_globals += 1
                boundary_ancestor = self._ancestor_at_scan_boundary(
                    leaf_node,
                    boundary_scan_index,
                )
                if boundary_ancestor is None:
                    missing_exact_boundary = True
                    break
                boundary_ancestors.append(boundary_ancestor)

            if participating_globals == 0:
                continue
            tracks_in_scope += 1
            if not boundary_ancestors:
                continue
            if missing_exact_boundary:
                continue

            first = boundary_ancestors[0]
            if all(anc.node_id == first.node_id for anc in boundary_ancestors[1:]):
                committed[track_id] = first
        return committed, tracks_in_scope

    def _update_n_scan_commitment(
        self,
        *,
        scan_index: int,
        post_beam_globals: list[GlobalHypothesis],
    ) -> tuple[int, int, int]:
        """
        Update tracker-owned N-scan commitment state for this scan.

        Runs after beam pruning and before births, using boundary b = k - N.
        Physical cleanup/GC is intentionally deferred; this function only records
        commitment agreement state.
        """
        boundary_scan_index = int(scan_index) - int(self.params.ns_scan_window)
        self._last_nscan_boundary_scan_index = boundary_scan_index
        self._last_nscan_committed_ancestor_by_track_id = {}
        self._last_nscan_tracks_in_scope = 0

        committed, tracks_in_scope = (
            self._compute_committed_track_ancestors_at_boundary(
                post_beam_globals,
                boundary_scan_index,
            )
        )
        self._last_nscan_tracks_in_scope = tracks_in_scope
        self._last_nscan_committed_ancestor_by_track_id = committed

        for track_id, ancestor in committed.items():
            prev_boundary = self._committed_boundary_by_track_id.get(track_id)
            if prev_boundary is None or boundary_scan_index > prev_boundary:
                self._committed_boundary_by_track_id[track_id] = boundary_scan_index
                self._committed_ancestor_by_track_id[track_id] = ancestor

        return boundary_scan_index, tracks_in_scope, len(committed)

    # =========================================================================
    # Internal Birth Helpers
    # =========================================================================
    # Internal birth branching from residual detections.

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

    def _generate_birth_candidates(
        self, residual: list[Detection], timestamp: datetime.datetime
    ) -> list[Track]:
        assert self.initiator is not None
        return (
            list(self.initiator.initiate(OrderedSet(residual), timestamp))
            if residual
            else []
        )

    def _filter_birth_candidates(self, born: list[Track]) -> list[Track]:
        return [tr for tr in born if self._birth_is_sane(tr)]

    def _birth_ranking_key(
        self, tr: Track, det_index_by_obj: dict[int, int]
    ) -> tuple[int, int, int, float, int]:
        used = self._birth_used_key(tr, det_index_by_obj)
        support, age, misses = self._birth_support_age_misses(tr)
        covtr = self._birth_covar_trace(tr)

        # Prefer: more support, fewer misses, shorter holding age (i.e. confirmed quickly),
        # then tighter covariance, then deterministic tie-break.
        return (
            -support,
            misses,
            age,
            covtr,
            used if used is not None else 10**9,
        )

    def _score_and_rank_birth_candidates(
        self, born: list[Track], det_index_by_obj: dict[int, int]
    ) -> list[tuple[tuple[int, int, int, float, int], Track]]:
        born_scored = [
            (self._birth_ranking_key(tr, det_index_by_obj), tr) for tr in born
        ]
        born_scored.sort(key=lambda kt: kt[0])
        return born_scored

    def _select_kept_birth_candidates(
        self,
        born_scored: list[tuple[tuple[int, int, int, float, int], Track]],
        *,
        det_index_by_obj: dict[int, int],
        timestamp: datetime.datetime,
    ) -> list[Track]:
        born = [tr for _, tr in born_scored]
        if self.params.debug_display_births:
            print(f"\nBirth candidates at {timestamp} (pre-limit): {len(born_scored)}")
            self._display_births(born, det_index_by_obj)

        born = born[: self.params.max_births_per_scan]

        if self.params.debug_display_births:
            print(f"Births kept (post-limit): {len(born)}")
            self._display_births(born, det_index_by_obj)
        return born

    def _prepare_birth_templates(
        self,
        born: list[Track],
        det_index_by_obj: dict[int, int],
        ctx: ScanContext,
    ) -> tuple[list[BirthTemplate], set[int]]:
        birth_templates: list[BirthTemplate] = []
        birth_track_ids: set[int] = set()
        for tr in born:
            tid = self._next_track_id
            self._next_track_id += 1
            used_key = self._birth_used_key(tr, det_index_by_obj)
            state = tr.states[-1]
            leaf_node = self._create_root_node(
                track_id=tid,
                scan_index=ctx.scan_index,
                timestamp=getattr(state, "timestamp", ctx.timestamp),
                state=state,
                state_kind="birth",
                used_det_key=used_key,
                assoc_label=(
                    TOMHTTracker.ASSOC_PAD if used_key is None else int(used_key)
                ),
                age=1,
                hits=1 if used_key is not None else 0,
                root_source="internal_birth",
            )
            birth_templates.append(
                BirthTemplate(
                    track_id=tid,
                    leaf_node=leaf_node,
                    template_track=tr,
                    used_det_key=used_key,
                )
            )
            birth_track_ids.add(tid)
        return birth_templates, birth_track_ids

    def _insert_birth_templates_into_global(
        self,
        gh: GlobalHypothesis,
        templates: Iterable[BirthTemplate],
        ctx: ScanContext,
    ) -> GlobalHypothesis:
        leaf_nodes_by_track_id = dict(gh.leaf_nodes_by_track_id)
        birth_delta_total = 0.0

        for template in templates:
            leaf_nodes_by_track_id[template.track_id] = template.leaf_node

            birth_delta_total += self.scoring_model.score_birth(
                birth_track=template.template_track,
                used_det_key=template.used_det_key,
                ctx=ctx,
            )

        return GlobalHypothesis(
            leaf_nodes_by_track_id=leaf_nodes_by_track_id,
            log_weight=gh.log_weight + birth_delta_total,
        )

    def _compatible_birth_templates_for_global(
        self, gh: GlobalHypothesis, birth_templates: list[BirthTemplate]
    ) -> list[BirthTemplate]:
        used_in_gh = self._used_det_keys_in_global(gh)
        return [
            template
            for template in birth_templates
            if template.used_det_key is None or template.used_det_key not in used_in_gh
        ]

    # Branching globals with compatible birth templates.

    def _branch_global_with_birth_templates(
        self,
        gh: GlobalHypothesis,
        compatible: list[BirthTemplate],
        ctx: ScanContext,
    ) -> list[GlobalHypothesis]:
        branched: list[GlobalHypothesis] = []

        # If there are no tracks yet, don't keep the empty hypothesis once we have births to add.
        if gh.leaf_nodes_by_track_id:
            branched.append(gh)

        # Always allow one-birth variants for compatible candidates.
        for template in compatible:
            branched.append(
                self._insert_birth_templates_into_global(
                    gh,
                    [template],
                    ctx,
                )
            )

        # Optional: also include the "two births at once" variant when exactly 2 are compatible.
        if len(compatible) >= 2 and self.params.max_births_per_scan >= 2:
            first = compatible[0]
            second = compatible[1]
            if (
                first.used_det_key is None
                or second.used_det_key is None
                or first.used_det_key != second.used_det_key
            ):  # should always hold, but be safe
                branched.append(
                    self._insert_birth_templates_into_global(
                        gh,
                        [first, second],
                        ctx,
                    )
                )
        return branched

    def _branch_globals_with_birth_templates(
        self,
        birth_templates: list[BirthTemplate],
        ctx: ScanContext,
    ) -> None:
        new_globals: list[GlobalHypothesis] = []
        for gh in self.global_hypotheses:
            compatible = self._compatible_birth_templates_for_global(
                gh, birth_templates
            )
            new_globals.extend(
                self._branch_global_with_birth_templates(
                    gh,
                    compatible,
                    ctx,
                )
            )

        new_globals.sort(key=lambda g: g.log_weight, reverse=True)
        self.global_hypotheses = new_globals[: self.params.max_global_hypotheses]

    def _birth_beam_stats(self, birth_track_ids: set[int]) -> tuple[int, int]:
        if not birth_track_ids:
            return 0, 0
        birth_track_instances_in_beam = sum(
            1
            for gh in self.global_hypotheses
            for tid in gh.leaf_nodes_by_track_id
            if tid in birth_track_ids
        )
        globals_with_birth = sum(
            1
            for gh in self.global_hypotheses
            if any(tid in birth_track_ids for tid in gh.leaf_nodes_by_track_id)
        )
        return birth_track_instances_in_beam, globals_with_birth

    # Birth phase entrypoint (invoked once per scan after beam/N-scan).

    def _branch_globals_with_births(
        self,
        ctx: ScanContext,
    ) -> BirthStats:
        globals_before_births = len(self.global_hypotheses)
        residual = self._residual_detections(self.global_hypotheses, ctx.detections)
        if self.initiator is not None and self.global_hypotheses:
            self._last_unused_detections = []
            residual_detections_considered = len(residual)
            born = self._generate_birth_candidates(residual, ctx.timestamp)
            birth_tracks_created = len(born)

            born = self._filter_birth_candidates(born)
            born_scored = self._score_and_rank_birth_candidates(
                born, ctx.det_index_by_obj
            )
            born = self._select_kept_birth_candidates(
                born_scored,
                det_index_by_obj=ctx.det_index_by_obj,
                timestamp=ctx.timestamp,
            )
            birth_tracks_kept = len(born)

            birth_track_ids: set[int] = set()
            if born:
                # Allocate stable IDs once for these births (shared across variants)
                birth_templates, birth_track_ids = self._prepare_birth_templates(
                    born,
                    ctx.det_index_by_obj,
                    ctx,
                )
                self._branch_globals_with_birth_templates(birth_templates, ctx)

            birth_track_instances_in_beam, globals_with_birth = self._birth_beam_stats(
                birth_track_ids
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
        if self.initiator is None:
            self._last_unused_detections = residual
        else:
            self._last_unused_detections = []
        return BirthStats(
            globals_before_births=globals_before_births,
            globals_after_births=len(self.global_hypotheses),
        )

    # =========================================================================
    # Node Lifecycle Helpers
    # =========================================================================

    def _allocate_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _register_node(self, node: TrackHypothesisNode) -> TrackHypothesisNode:
        self._nodes_by_id[node.node_id] = node
        return node

    def _create_track_hypothesis_node(
        self,
        *,
        track_id: int,
        parent: TrackHypothesisNode | None,
        scan_index: int,
        timestamp: datetime.datetime,
        state: State,
        state_kind: str,
        used_det_key: int | None,
        assoc_label: int,
        log_delta: float,
        age: int,
        hits: int,
        missed_count: int,
        last_det_key: int | None,
        last_det_hit: bool,
        root_source: str,
        birth_scan_index: int,
    ) -> TrackHypothesisNode:
        """Create and register one node."""
        if parent is not None and parent.track_id != track_id:
            raise ValueError(
                "TrackHypothesisNode parent.track_id must match child track_id."
            )
        node = TrackHypothesisNode(
            node_id=self._allocate_node_id(),
            track_id=int(track_id),
            parent=parent,
            scan_index=int(scan_index),
            timestamp=timestamp,
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=int(assoc_label),
            log_delta=float(log_delta),
            age=int(age),
            hits=int(hits),
            missed_count=int(missed_count),
            last_det_key=last_det_key,
            last_det_hit=bool(last_det_hit),
            root_source=root_source,
            birth_scan_index=int(birth_scan_index),
        )
        return self._register_node(node)

    def _create_root_node(
        self,
        *,
        track_id: int,
        scan_index: int,
        timestamp: datetime.datetime,
        state: State,
        state_kind: str,
        used_det_key: int | None,
        assoc_label: int,
        age: int,
        hits: int,
        root_source: str,
    ) -> TrackHypothesisNode:
        return self._create_track_hypothesis_node(
            track_id=track_id,
            parent=None,
            scan_index=scan_index,
            timestamp=timestamp,
            state=state,
            state_kind=state_kind,
            used_det_key=used_det_key,
            assoc_label=assoc_label,
            log_delta=0.0,
            age=age,
            hits=hits,
            missed_count=0,
            last_det_key=used_det_key,
            last_det_hit=used_det_key is not None,
            root_source=root_source,
            birth_scan_index=scan_index,
        )

    # =========================================================================
    # External-Start Helpers
    # =========================================================================

    def _make_external_start_template(
        self, start: Track, time: datetime.datetime
    ) -> TrackHypothesisNode:
        """Convert one confirmed external start Track into a root node template."""
        if len(start) == 0:
            raise ValueError(
                "External starts must contain at least one state at the current timestamp."
            )

        start_timestamp = getattr(start.states[-1], "timestamp", None)
        if start_timestamp != time:
            raise ValueError(
                "External starts must already be initialised at the supplied "
                f"timestamp. Expected {time!r}, got {start_timestamp!r}."
            )

        track_id = self._next_track_id
        self._next_track_id += 1

        age = max(int(start.metadata.get("age", len(start))), 1)
        hits = int(start.metadata.get("hits", age))
        hits = min(max(hits, 1), age)
        state = start.states[-1]
        assert self._last_scan_index is not None
        return self._create_root_node(
            track_id=track_id,
            scan_index=int(self._last_scan_index),
            timestamp=getattr(state, "timestamp", time),
            state=state,
            state_kind="external_start",
            used_det_key=None,
            assoc_label=TOMHTTracker.ASSOC_PAD,
            age=age,
            hits=hits,
            root_source="external_start",
        )

    def _validate_external_starts_timestamp(self, time: datetime.datetime) -> None:
        """Validate add_external_starts(...) call ordering and timestamp match."""
        if not isinstance(time, datetime.datetime):
            raise TypeError(
                "add_external_starts() time must be a datetime.datetime instance."
            )
        if self._last_update_timestamp is None or self._last_scan_index is None:
            raise RuntimeError(
                "add_external_starts() requires a completed update_tracker() first."
            )
        if time != self._last_update_timestamp:
            raise ValueError(
                "add_external_starts() time must match the most recent "
                f"completed update_tracker() timestamp. Expected {self._last_update_timestamp!r}, "
                f"got {time!r}."
            )

    # =========================================================================
    # Diagnostics / Instrumentation Helpers
    # =========================================================================
    # Scan-time instrumentation collection and optional debug displays.

    def _run_scan_instrumentation(
        self,
        *,
        ctx: ScanContext,
        scan_wall_ms: float,
        maxrss_mb: float,
        node_count_total: int,
        leaf_instances_in_beam: int,
        globals_in: int,
        globals_expanded: int,
        globals_after_unused: int,
        globals_after_dedupe: int,
        globals_after_beam: int,
        nscan_boundary_scan_index: int,
        nscan_tracks_in_scope: int,
        nscan_tracks_committed: int,
        birth_stats: BirthStats,
    ) -> None:
        self._maybe_display_scan_debug_output(ctx)
        scan_stats = self._build_scan_stats(
            ctx=ctx,
            scan_wall_ms=scan_wall_ms,
            maxrss_mb=maxrss_mb,
            node_count_total=node_count_total,
            leaf_instances_in_beam=leaf_instances_in_beam,
            globals_in=globals_in,
            globals_expanded=globals_expanded,
            globals_after_unused=globals_after_unused,
            globals_after_dedupe=globals_after_dedupe,
            globals_after_beam=globals_after_beam,
            nscan_boundary_scan_index=nscan_boundary_scan_index,
            nscan_tracks_in_scope=nscan_tracks_in_scope,
            nscan_tracks_committed=nscan_tracks_committed,
            birth_stats=birth_stats,
        )
        self.last_scan_stats = scan_stats
        if self.params.collect_stats:
            self._stats.append(scan_stats)
        self._maybe_display_scan_stats(timestamp=ctx.timestamp, scan_stats=scan_stats)

    def _maybe_display_scan_debug_output(self, ctx: ScanContext) -> None:
        if self.params.debug_display_detections:
            print(f"\nDetections at timestamp {ctx.timestamp}:")
            for det in ctx.detections:
                print(f"  {det.state_vector}")

        if self.params.debug_display_hypotheses:
            print(f"\nGlobal hypotheses at timestamp {ctx.timestamp}:")
            self._display_global_hypotheses(ctx.detections)

    def _map_stats_for_current_map(
        self, detections: list[Detection]
    ) -> tuple[int, int, int, dict[int, int], float]:
        map_snapshot = self.get_map_hypothesis_snapshot()
        map_tracks = 0
        map_used = 0
        map_unused = len(detections)
        map_miss_hist: dict[int, int] = {}
        map_mean_hit_rate = 0.0
        if map_snapshot is None:
            return map_tracks, map_used, map_unused, map_miss_hist, map_mean_hit_rate

        map_tracks = len(map_snapshot.leaf_nodes_by_track_id)
        map_used = len(
            self._used_det_keys_for_leaf_nodes(map_snapshot.leaf_nodes_by_track_id)
        )
        map_unused = len(detections) - map_used

        hit_rates: list[float] = []
        for leaf_node in map_snapshot.leaf_nodes_by_track_id.values():
            misses = int(leaf_node.missed_count)
            map_miss_hist[misses] = map_miss_hist.get(misses, 0) + 1
            age = int(leaf_node.age)
            if age > 0:
                hits = int(leaf_node.hits)
                hit_rates.append(float(hits) / float(age))
        if hit_rates:
            map_mean_hit_rate = float(np.mean(hit_rates))
        return map_tracks, map_used, map_unused, map_miss_hist, map_mean_hit_rate

    def _build_scan_stats(
        self,
        *,
        ctx: ScanContext,
        scan_wall_ms: float,
        maxrss_mb: float,
        node_count_total: int,
        leaf_instances_in_beam: int,
        globals_in: int,
        globals_expanded: int,
        globals_after_unused: int,
        globals_after_dedupe: int,
        globals_after_beam: int,
        nscan_boundary_scan_index: int,
        nscan_tracks_in_scope: int,
        nscan_tracks_committed: int,
        birth_stats: BirthStats,
    ) -> ScanStats:
        map_tracks, map_used, map_unused, map_miss_hist, map_mean_hit_rate = (
            self._map_stats_for_current_map(ctx.detections)
        )
        return ScanStats(
            timestamp=ctx.timestamp,
            scan_wall_ms=float(scan_wall_ms),
            maxrss_mb=float(maxrss_mb),
            node_count_total=int(node_count_total),
            leaf_instances_in_beam=int(leaf_instances_in_beam),
            num_detections=len(ctx.detections),
            globals_in=globals_in,
            globals_expanded=globals_expanded,
            globals_after_unused=globals_after_unused,
            globals_after_dedupe=globals_after_dedupe,
            globals_after_beam=globals_after_beam,
            nscan_boundary_scan_index=nscan_boundary_scan_index,
            nscan_tracks_in_scope=nscan_tracks_in_scope,
            nscan_tracks_committed=nscan_tracks_committed,
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

    def _maybe_display_scan_stats(
        self,
        *,
        timestamp: datetime.datetime,
        scan_stats: ScanStats,
    ) -> None:
        if not self.params.debug_display_scan_stats:
            return
        nscan_snapshot = self.get_n_scan_commitment_snapshot()
        print_scan_stats_report(
            timestamp=timestamp,
            scan_stats=scan_stats,
            nscan_snapshot=nscan_snapshot,
            debug_display_map_miss_hist=self.params.debug_display_map_miss_hist,
        )

    # Debug renderers for ad-hoc scan inspection.

    def _display_global_hypotheses(self, det_list: list[Detection]) -> None:
        for gh in self.global_hypotheses[: self.params.debug_globals_max]:
            used = len(self._used_det_keys_for_leaf_nodes(gh.leaf_nodes_by_track_id))
            unused = len(det_list) - used
            print(
                f"logW={gh.log_weight:.3f}, "
                f"tracks={len(gh.leaf_nodes_by_track_id)}, "
                f"used={used}, unused={unused}, "
                f"ids={sorted(gh.leaf_nodes_by_track_id.keys())}"
            )

            for tid, node in sorted(gh.leaf_nodes_by_track_id.items()):
                last = getattr(node.state, "state_vector")
                ldk = node.last_det_key
                miss = int(node.missed_count)
                dk = node.used_det_key
                used_str = "MISS" if dk is None else "HIT"
                age = int(node.age)
                hits = int(node.hits)
                parent_node_id = (
                    None if node.parent is None else int(node.parent.node_id)
                )
                committed_boundary = self._committed_boundary_by_track_id.get(tid)
                committed_node = self._committed_ancestor_by_track_id.get(tid)
                if committed_boundary is None or committed_node is None:
                    committed_str = "none"
                else:
                    committed_str = (
                        f"b={committed_boundary}->node={committed_node.node_id}"
                    )
                print(
                    f"  tid={tid}, leaf={node.node_id}, parent={parent_node_id}, "
                    f"{used_str}, age={age}, hits={hits}, miss={miss}, ldk={ldk}, "
                    f"root={node.root_source}, committed={committed_str}, "
                    f"last={self._fmt_state_xyvxvy(last)}"
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


# ============================================================================
# Public TOMHT Utility
# ============================================================================


def get_tomht_track_id(track: Track) -> int:
    """Return the stable TOMHT logical track ID from a TOMHT output track.

    TOMHTTracker assigns each logical track a stable integer ID that persists
    across scans. The Track objects returned by ``update_tracker()`` and the
    ``tracks`` property are reconstructed each scan, so ``Track.id`` (a UUID)
    is not stable for TOMHT logical identity.

    This helper is TOMHT-specific. It expects TOMHT metadata to be present on
    the supplied ``Track``.
    """
    try:
        return int(track.metadata["track_id"])
    except KeyError as exc:
        raise KeyError(
            "Track metadata does not contain TOMHT 'track_id'. "
            "Use this helper only with TOMHTTracker-produced tracks."
        ) from exc
