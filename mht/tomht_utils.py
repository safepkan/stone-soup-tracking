"""TOMHT-specific scan and debug utility helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from stonesoup.types.detection import Detection

from .tomht_model import DetectionKey


def det_sort_key(det: Detection) -> tuple:
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
    return (ts_key, len(vec_key), vec_key)


def sorted_detections(detections: Iterable[Detection]) -> list[Detection]:
    """Return a deterministic list copy of scan detections."""
    return [det for _, det in sorted_detections_with_input_indices(detections)]


def sorted_detections_with_input_indices(
    detections: Iterable[Detection],
) -> list[tuple[int, Detection]]:
    """Return deterministic detections with caller iteration indices."""
    indexed_detections = list(enumerate(detections))
    indexed_detections.sort(key=lambda item: det_sort_key(item[1]))
    return indexed_detections


def current_scan_det_indices_from_keys(
    keys: Iterable[DetectionKey], scan_index: int
) -> set[int]:
    """Return detection indices from keys that belong to ``scan_index``."""
    return {det_idx for (key_scan, det_idx) in keys if key_scan == scan_index}


def format_detection_key_sample(
    keys: set[DetectionKey],
    *,
    max_items: int = 6,
) -> str:
    """Return compact stable formatting for detection-key debug samples."""
    if not keys:
        return "[]"
    ordered = sorted(keys)
    if len(ordered) <= max_items:
        return str(ordered)
    head = ordered[:max_items]
    return f"{head}...(+{len(ordered) - max_items})"
