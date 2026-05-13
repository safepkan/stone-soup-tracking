from __future__ import annotations

import datetime
from io import StringIO
from contextlib import redirect_stdout
from math import log
import unittest

import numpy as np
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.state import GaussianState

from mht.tomht_scoring import (
    NLLScoringModel,
    make_default_scoring_model,
    maybe_log_scoring_diagnostics,
)
from mht.tomht_types import ScanContext


def _detection(x: float, timestamp: datetime.datetime) -> Detection:
    return Detection(np.array([[x], [x]]), timestamp=timestamp)


def _scan_context(
    *,
    timestamp: datetime.datetime,
    detections: list[Detection],
) -> ScanContext:
    return ScanContext(
        scan_index=0,
        timestamp=timestamp,
        detections=detections,
        det_index_by_obj={id(det): index for index, det in enumerate(detections)},
    )


def _prediction(timestamp: datetime.datetime) -> GaussianState:
    return GaussianState(
        [0.0, 0.0, 0.0, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )


class NLLScoringModelTest(unittest.TestCase):
    def test_hit_and_miss_scores_use_existing_nll_formulas(self) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        detection = _detection(1.0, timestamp)
        ctx = _scan_context(timestamp=timestamp, detections=[detection])
        prediction = _prediction(timestamp)
        hit_nll = 1.5
        model = NLLScoringModel(
            prob_detect=0.8,
            clutter_density=0.25,
            log_epsilon=1e-12,
        )

        scores = model.score_track_hypotheses(
            hypotheses=[
                SingleDistanceHypothesis(
                    prediction=prediction,
                    measurement=detection,
                    distance=hit_nll,
                ),
                SingleDistanceHypothesis(
                    prediction=prediction,
                    measurement=MissedDetection(timestamp=timestamp),
                    distance=999.0,
                ),
            ],
            ctx=ctx,
        )

        self.assertEqual(2, len(scores))
        self.assertAlmostEqual(log(0.8) - log(0.25) - hit_nll, scores[0])
        self.assertAlmostEqual(log(1.0 - 0.8), scores[1])

    def test_default_scoring_construction_still_works(self) -> None:
        scoring_model = make_default_scoring_model(
            scoring_mode="nll",
            prob_detect=0.9,
            log_epsilon=1e-12,
            clutter_density=0.0,
        )

        self.assertIsInstance(scoring_model, NLLScoringModel)

    def test_diagnostics_report_active_nll_terms_only(self) -> None:
        model = NLLScoringModel(
            prob_detect=0.9,
            clutter_density=2.0,
            log_epsilon=1e-12,
        )
        out = StringIO()

        with redirect_stdout(out):
            maybe_log_scoring_diagnostics(model)

        value = out.getvalue()
        self.assertIn("log_hit_base", value)
        self.assertIn("log_miss", value)
        self.assertNotIn("unused", value)


if __name__ == "__main__":
    unittest.main()
