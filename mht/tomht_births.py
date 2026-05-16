"""Internal-birth candidate helpers for the track-oriented TOMHT tracker."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from ordered_set import OrderedSet

import numpy as np

from stonesoup.initiator.base import Initiator
from stonesoup.types.detection import Detection
from stonesoup.types.track import Track
from stonesoup.types.update import Update

from .tomht_model import DetectionKey, TrackHypothesisNode
from .tomht_params import TOMHTParams
from .tomht_scoring import (
    _existence_probability_to_log_odds,
    existence_metadata_to_log_odds,
)
from .tomht_stats import BirthStats
from .tomht_tree_store import TrackTreeStore
from .tomht_types import ScanContext


@dataclass(frozen=True)
class InternalBirthResult:
    """Internal-birth side effects plus tracker-owned unused-detection update."""

    stats: BirthStats
    unused_detections: list[Detection]


def birth_used_key(
    tr: Track,
    *,
    scan_index: int,
    det_index_by_obj: dict[int, int],
) -> DetectionKey | None:
    """Best-effort extraction of a current-scan used detection key for a birth."""
    try:
        last = tr.states[-1]
        hyp = getattr(last, "hypothesis", None)
        meas = getattr(hyp, "measurement", None) if hyp is not None else None
        if meas is None:
            return None
        det_index = det_index_by_obj.get(id(meas))
        if det_index is None:
            return None
        return DetectionKey(scan_index=scan_index, det_index=int(det_index))
    except Exception:
        return None


def birth_holding_track(birth: Track) -> Track:
    """Return the initiator holding-track metadata when available."""
    holding = birth.metadata.get("holding_track", None)
    return holding if isinstance(holding, Track) else birth


def birth_support_points(birth: Track) -> int:
    """Return update-count support for one birth candidate."""
    holding = birth_holding_track(birth)
    return sum(1 for state in holding.states if isinstance(state, Update))


def birth_support_age_misses(birth: Track) -> tuple[int, int, int]:
    """Return support/age/miss summary for one birth candidate."""
    holding = birth_holding_track(birth)
    age = len(holding)
    support = birth_support_points(birth)
    misses = max(age - support, 0)
    return support, age, misses


def birth_covar_trace(birth: Track) -> float:
    """Return covariance-trace quality proxy for one birth candidate."""
    state = birth.states[-1]
    cov = getattr(state, "covar", None)
    if cov is None:
        return float("inf")
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        return float("inf")
    trace_val = float(np.trace(cov_arr))
    if not np.isfinite(trace_val):
        return float("inf")
    return trace_val


def birth_existence_probability_sort_value(track: Track) -> float:
    """Return a deterministic ordering hint from initiator confidence metadata."""
    metadata_value = track.metadata.get("existence_probability")
    if metadata_value is None:
        return float("inf")
    try:
        probability = float(metadata_value)
    except (TypeError, ValueError):
        return float("inf")
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        return float("inf")
    return -probability


def _sanitized_flat_state_components(state_vector) -> tuple[float, ...]:
    values = np.asarray(state_vector, dtype=float).reshape(-1)
    components: list[float] = []
    for value in values:
        component = float(value)
        components.append(component if np.isfinite(component) else float("inf"))
    return tuple(components)


def _format_state_component(value: float) -> str:
    return "inf" if not np.isfinite(value) else f"{value:.6g}"


def birth_track_sort_key(
    tr: Track,
    *,
    scan_index: int,
    det_index_by_obj: dict[int, int],
) -> tuple[float, ...]:
    """Return deterministic heuristic ranking key for internal births."""
    used_key = birth_used_key(
        tr,
        scan_index=scan_index,
        det_index_by_obj=det_index_by_obj,
    )
    support, age, misses = birth_support_age_misses(tr)
    cov_trace = birth_covar_trace(tr)

    st = tr.states[-1]
    state_components = _sanitized_flat_state_components(st.state_vector)

    used_idx = float(10**9 if used_key is None else int(used_key.det_index))
    return (
        float(-support),
        float(misses),
        float(age),
        birth_existence_probability_sort_value(tr),
        cov_trace,
        used_idx,
        *state_components,
        float(len(tr.states)),
    )


def residual_detection_indices_after_expansion(
    *,
    active_leaf_nodes: Iterable[TrackHypothesisNode],
    ctx: ScanContext,
) -> list[int]:
    """Return current-scan detection indices unused after local expansion."""
    used_current_scan_det_indices: set[int] = set()
    for leaf in active_leaf_nodes:
        if (
            leaf.used_det_key is not None
            and int(leaf.used_det_key.scan_index) == ctx.scan_index
        ):
            used_current_scan_det_indices.add(int(leaf.used_det_key.det_index))
    return [
        i for i in range(len(ctx.detections)) if i not in used_current_scan_det_indices
    ]


def birth_guardrail_block_reason(
    *,
    active_trees: int,
    active_leaves: int,
    params: TOMHTParams,
) -> str | None:
    """Return a reason when simple load guards should block internal births."""
    trees_cap = params.birth_skip_if_active_trees_above
    if trees_cap is not None and active_trees > int(trees_cap):
        return f"active tree count above cap ({active_trees}>{int(trees_cap)})"

    leaves_cap = params.birth_skip_if_active_leaves_above
    if leaves_cap is not None and active_leaves > int(leaves_cap):
        return f"active leaf count above cap ({active_leaves}>{int(leaves_cap)})"

    return None


def select_internal_birth_candidates(
    *,
    initiated_tracks: list[Track],
    ctx: ScanContext,
    params: TOMHTParams,
) -> list[Track]:
    """Apply deterministic internal-birth quality ordering and per-scan cap."""
    kept_birth_tracks = list(initiated_tracks)
    # Initiators return sets; make cap selection deterministic across runs.
    kept_birth_tracks.sort(
        key=lambda track: birth_track_sort_key(
            track,
            scan_index=ctx.scan_index,
            det_index_by_obj=ctx.det_index_by_obj,
        )
    )
    if len(kept_birth_tracks) > params.max_births_per_scan:
        kept_birth_tracks = kept_birth_tracks[: params.max_births_per_scan]
    return kept_birth_tracks


def format_birth_state_vector(state_vector) -> str:
    """Format flattened state-vector components for compact debug output."""
    components = _sanitized_flat_state_components(state_vector)
    parts = [
        f"{idx}={_format_state_component(value)}"
        for idx, value in enumerate(components)
    ]
    return f"({', '.join(parts)})"


def initiator_start_initial_log_delta(
    birth_track: Track,
    *,
    params: TOMHTParams,
) -> float:
    """Return the initial log-odds score for one black-box initiator start."""
    default_log_delta = _existence_probability_to_log_odds(
        params.initiator_start_initial_existence_probability,
        parameter_name="initiator_start_initial_existence_probability",
    )
    return existence_metadata_to_log_odds(
        birth_track.metadata,
        default_log_odds=default_log_delta,
        source_name="initiator start",
    )


def run_internal_births_from_residuals(
    *,
    residual_detections: list[Detection],
    ctx: ScanContext,
    initiator: Initiator | None,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    assoc_pad_label: int,
) -> InternalBirthResult:
    """Create internal birth trees from Step-2 residual detections."""
    if initiator is None:
        return InternalBirthResult(
            stats=BirthStats(
                residual_detections_considered=len(residual_detections),
                birth_tracks_created=0,
                birth_tracks_kept=0,
            ),
            unused_detections=residual_detections,
        )

    if not residual_detections:
        return InternalBirthResult(
            stats=BirthStats(
                residual_detections_considered=0,
                birth_tracks_created=0,
                birth_tracks_kept=0,
            ),
            unused_detections=[],
        )

    birth_block_reason = birth_guardrail_block_reason(
        active_trees=tree_store.active_tree_count(),
        active_leaves=tree_store.active_leaf_count(),
        params=params,
    )
    if birth_block_reason is not None:
        if params.debug_display_births:
            print(
                "\nINTERNAL_BIRTH_GUARDRAIL "
                f"t={ctx.timestamp} reason={birth_block_reason} "
                f"residual={len(residual_detections)}"
            )
        return InternalBirthResult(
            stats=BirthStats(
                residual_detections_considered=len(residual_detections),
                birth_tracks_created=0,
                birth_tracks_kept=0,
            ),
            unused_detections=residual_detections,
        )

    initiated_tracks = list(
        initiator.initiate(OrderedSet(residual_detections), ctx.timestamp)
    )
    birth_tracks_created = len(initiated_tracks)

    kept_birth_tracks = select_internal_birth_candidates(
        initiated_tracks=initiated_tracks,
        ctx=ctx,
        params=params,
    )
    birth_tracks_kept = len(kept_birth_tracks)

    if params.debug_display_births and kept_birth_tracks:
        print(f"\nInternal births at {ctx.timestamp}: kept={birth_tracks_kept}")
        for track in kept_birth_tracks[: params.debug_births_max]:
            # Debug-only display retained for quick replay inspection.
            state = track.states[-1].state_vector
            print(f"  birth_state={format_birth_state_vector(state)}")

    for birth_track in kept_birth_tracks:
        state = birth_track.states[-1]
        used_key = birth_used_key(
            birth_track,
            scan_index=ctx.scan_index,
            det_index_by_obj=ctx.det_index_by_obj,
        )
        age = max(len(birth_track), 1)
        hits = 1 if used_key is not None else 0
        root_log_delta = initiator_start_initial_log_delta(
            birth_track,
            params=params,
        )
        tree_store.create_root_tree_for_new_track(
            scan_index=ctx.scan_index,
            timestamp=getattr(state, "timestamp", ctx.timestamp),
            state=state,
            state_kind="internal_birth",
            used_det_key=used_key,
            assoc_label=(
                assoc_pad_label if used_key is None else int(used_key.det_index)
            ),
            log_delta=float(root_log_delta),
            age=age,
            hits=hits,
            root_source="internal_birth",
        )

    return InternalBirthResult(
        stats=BirthStats(
            residual_detections_considered=len(residual_detections),
            birth_tracks_created=birth_tracks_created,
            birth_tracks_kept=birth_tracks_kept,
        ),
        unused_detections=[],
    )


def run_internal_births_after_expansion(
    *,
    ctx: ScanContext,
    initiator: Initiator | None,
    tree_store: TrackTreeStore,
    params: TOMHTParams,
    assoc_pad_label: int,
) -> InternalBirthResult:
    """Create internal births from detections unused after local expansion."""
    residual_det_indices = residual_detection_indices_after_expansion(
        active_leaf_nodes=tree_store.active_leaf_nodes(),
        ctx=ctx,
    )
    residual_detections = [ctx.detections[i] for i in residual_det_indices]
    return run_internal_births_from_residuals(
        residual_detections=residual_detections,
        ctx=ctx,
        initiator=initiator,
        tree_store=tree_store,
        params=params,
        assoc_pad_label=assoc_pad_label,
    )
