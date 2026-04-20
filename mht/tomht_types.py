"""Shared TO-MHT type/data containers used across core modules."""

from __future__ import annotations

from dataclasses import dataclass
import datetime

from stonesoup.types.detection import Detection


@dataclass(frozen=True)
class ScanContext:
    """Per-scan context passed into scoring and tracker pipeline helpers."""

    scan_index: int
    timestamp: datetime.datetime
    detections: list[Detection]
    det_index_by_obj: dict[int, int]
