"""Shared TO-MHT type/data containers used across core modules."""

from __future__ import annotations

from dataclasses import dataclass
import datetime

from stonesoup.types.detection import Detection


@dataclass(frozen=True)
class ScanContext:
    """Internal per-scan TOMHT bookkeeping.

    ``caller_scan_context`` is opaque caller-provided scan data threaded to the
    DetectionProbabilityModel. It is intentionally separate from the internal
    bookkeeping fields in this dataclass.
    """

    scan_index: int
    timestamp: datetime.datetime
    detections: list[Detection]
    det_index_by_obj: dict[int, int]
    caller_scan_context: object | None = None
