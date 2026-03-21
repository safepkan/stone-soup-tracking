from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration parameters that both scenario and tracker care about."""

    prob_detect: float
    prob_gate: float  # TODO: Nove to MFAParams
    clutter_density: float
    v_bounds: np.ndarray
    slide_window: int  # TODO: Nove to MFAParams
