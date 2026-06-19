"""Public integration surface for the track-oriented TO-MHT tracker.

Import the stable public API from here::

    from mht.api import TOMHTTracker, TOMHTParams

``TO_MHT_API.md`` is the integration guide for these names. Everything else --
the internal ``tomht_*`` modules and undocumented debug snapshot details -- is
not part of this stable surface and may change without notice.

Exported names:

- ``TOMHTTracker``: the tracker.
- ``TOMHTParams``: tracker configuration.
- ``DetectionProbabilityModel``: protocol for a custom dynamic ``P_D`` /
  clutter-density model.
- ``ConstantDetectionProbabilityModel``: scalar default detection-probability
  model used when no custom model is supplied.
- ``MAPAssociationHistorySnapshot`` / ``MapTrackAssociationHistory`` /
  ``MapAssociationStep``: public return types for
  ``TOMHTTracker.get_map_association_history(...)``.
- ``AssociationStatus`` / ``TrackLifecycleState`` /
  ``TrackPublicationState``: public literal aliases used by the association
  history return types.
"""

from .tomht_model import (
    AssociationStatus,
    MAPAssociationHistorySnapshot,
    MapAssociationStep,
    MapTrackAssociationHistory,
    TrackLifecycleState,
    TrackPublicationState,
)
from .tomht_params import TOMHTParams
from .tomht_scoring import (
    ConstantDetectionProbabilityModel,
    DetectionProbabilityModel,
)
from .tomht_tracker import TOMHTTracker

__all__ = [
    "TOMHTTracker",
    "TOMHTParams",
    "DetectionProbabilityModel",
    "ConstantDetectionProbabilityModel",
    "MAPAssociationHistorySnapshot",
    "MapTrackAssociationHistory",
    "MapAssociationStep",
    "AssociationStatus",
    "TrackLifecycleState",
    "TrackPublicationState",
]
