"""Internal-birth candidate helpers for the track-oriented TOMHT tracker."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from stonesoup.types.track import Track
from stonesoup.types.update import Update

from .tomht_model import DetectionKey, TrackHypothesisNode
from .tomht_params import TOMHTParams
from .tomht_types import ScanContext
from .tomht_utils import current_scan_det_indices_from_keys


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
        return (scan_index, int(det_index))
    except Exception:
        return None


def birth_is_sane(tr: Track, *, params: TOMHTParams) -> bool:
    """Apply simple numeric sanity checks before accepting an internal birth."""
    st = tr.states[-1]
    sv = np.asarray(st.state_vector, dtype=float)

    x = float(sv[0, 0])
    y = float(sv[2, 0])
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    if abs(x) > params.birth_max_abs_pos or abs(y) > params.birth_max_abs_pos:
        return False

    cov = getattr(st, "covar", None)
    if cov is None:
        return False
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return False
    if float(np.trace(cov)) > params.birth_max_covar_trace:
        return False

    return True


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
    sv = np.asarray(st.state_vector, dtype=float).reshape(-1)

    def _state_component(idx: int) -> float:
        if idx >= sv.size:
            return float("inf")
        value = float(sv[idx])
        if not np.isfinite(value):
            return float("inf")
        return value

    used_idx = float(10**9 if used_key is None else int(used_key[1]))
    return (
        float(-support),
        float(misses),
        float(age),
        cov_trace,
        used_idx,
        _state_component(0),  # x
        _state_component(1),  # vx
        _state_component(2),  # y
        _state_component(3),  # vy
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
        used_current_scan_det_indices |= current_scan_det_indices_from_keys(
            leaf.detection_history_keys,
            ctx.scan_index,
        )
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
    """Apply deterministic internal-birth quality filtering and per-scan cap."""
    kept_birth_tracks = [
        track for track in initiated_tracks if birth_is_sane(track, params=params)
    ]
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
