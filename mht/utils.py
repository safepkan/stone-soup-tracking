from __future__ import annotations

import os
import resource
import sys


def env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return default


def env_float(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def ns_to_ms(ns: int) -> float:
    return float(ns) / 1_000_000.0


def get_process_maxrss_mb() -> float:
    """Return process peak RSS in MB."""
    ru_maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # ru_maxrss units differ by platform:
    # - macOS: bytes
    # - Linux: KiB
    ru_maxrss_units = 1.0 if sys.platform == "darwin" else 1024.0
    return ru_maxrss * ru_maxrss_units / (1024.0 * 1024.0)
