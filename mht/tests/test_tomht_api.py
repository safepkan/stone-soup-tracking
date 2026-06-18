from __future__ import annotations

import mht.api as api


def test_public_api_reexports_stable_integration_names() -> None:
    expected_names = {
        "TOMHTTracker",
        "TOMHTParams",
        "DetectionProbabilityModel",
        "ConstantDetectionProbabilityModel",
        "MAPAssociationHistorySnapshot",
        "MapTrackAssociationHistory",
        "MapAssociationStep",
    }

    assert expected_names <= set(api.__all__)
    for name in expected_names:
        assert getattr(api, name) is not None
