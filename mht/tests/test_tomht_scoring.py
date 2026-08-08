from __future__ import annotations

import datetime
from io import StringIO
from contextlib import redirect_stdout
from math import inf, isfinite, log, nan
import unittest

import numpy as np
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.state import GaussianState

from mht.tomht_scoring import (
    ConstantDetectionProbabilityModel,
    NLLScoringModel,
    maybe_log_scoring_diagnostics,
)
from mht.tomht_model import ScanContext


def _detection(x: float, timestamp: datetime.datetime) -> Detection:
    return Detection(np.array([[x], [x]]), timestamp=timestamp)


def _scan_context(
    *,
    timestamp: datetime.datetime,
    detections: list[Detection],
    caller_scan_context: object | None = None,
) -> ScanContext:
    return ScanContext(
        scan_index=0,
        timestamp=timestamp,
        detections=detections,
        det_index_by_obj={id(det): index for index, det in enumerate(detections)},
        caller_scan_context=caller_scan_context,
    )


def _prediction(timestamp: datetime.datetime) -> GaussianState:
    return GaussianState(
        [0.0, 0.0, 0.0, 0.0],
        covar=np.eye(4),
        timestamp=timestamp,
    )


class _FixedDetectionProbabilityModel:
    def __init__(self, *, prob_detect: float, clutter_density: float) -> None:
        self._prob_detect = prob_detect
        self._clutter_density = clutter_density

    def detection_probability(
        self,
        *,
        track_id: object | None,
        prediction,
        caller_scan_context: object | None,
    ) -> float:
        del track_id, prediction, caller_scan_context
        return self._prob_detect

    def clutter_density(
        self,
        *,
        prediction,
        detection: Detection | None,
        caller_scan_context: object | None,
    ) -> float:
        del prediction, detection, caller_scan_context
        return self._clutter_density


class NLLScoringModelTest(unittest.TestCase):
    def test_constant_model_rejects_invalid_detection_probability(self) -> None:
        for prob_detect in (0.0, 1.0, -0.1, 1.5, nan):
            with self.subTest(prob_detect=prob_detect):
                with self.assertRaisesRegex(
                    ValueError,
                    "prob_detect must satisfy 0.0 < p < 1.0",
                ):
                    ConstantDetectionProbabilityModel(
                        prob_detect=prob_detect,
                        clutter_density=1.0,
                    )

    def test_constant_model_rejects_invalid_clutter_density(self) -> None:
        for clutter_density in (0.0, -1.0, nan, inf):
            with self.subTest(clutter_density=clutter_density):
                with self.assertRaisesRegex(
                    ValueError,
                    "clutter_density must be a positive, finite density",
                ):
                    ConstantDetectionProbabilityModel(
                        prob_detect=0.5,
                        clutter_density=clutter_density,
                    )

    def test_constant_model_preserves_existing_nll_formulas(self) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        detection = _detection(1.0, timestamp)
        ctx = _scan_context(timestamp=timestamp, detections=[detection])
        prediction = _prediction(timestamp)
        hit_nll = 1.5
        model = NLLScoringModel(
            detection_probability_model=ConstantDetectionProbabilityModel(
                prob_detect=0.8,
                clutter_density=0.25,
            ),
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

    def test_dynamic_model_can_vary_detection_probability_by_prediction(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        ctx = _scan_context(timestamp=timestamp, detections=[])
        low_pd_prediction = GaussianState(
            [-1.0, 0.0, 0.0, 0.0],
            covar=np.eye(4),
            timestamp=timestamp,
        )
        high_pd_prediction = GaussianState(
            [1.0, 0.0, 0.0, 0.0],
            covar=np.eye(4),
            timestamp=timestamp,
        )

        class _PredictionDependentDPM:
            def detection_probability(
                self,
                *,
                track_id: object | None,
                prediction,
                caller_scan_context: object | None,
            ) -> float:
                del track_id, caller_scan_context
                x = float(np.asarray(prediction.state_vector).reshape(-1)[0])
                return 0.1 if x < 0.0 else 0.9

            def clutter_density(
                self,
                *,
                prediction,
                detection: Detection | None,
                caller_scan_context: object | None,
            ) -> float:
                del prediction, detection, caller_scan_context
                return 1.0

        model = NLLScoringModel(
            detection_probability_model=_PredictionDependentDPM(),
            log_epsilon=1e-12,
        )

        scores = model.score_track_hypotheses(
            hypotheses=[
                SingleDistanceHypothesis(
                    prediction=low_pd_prediction,
                    measurement=MissedDetection(timestamp=timestamp),
                    distance=999.0,
                ),
                SingleDistanceHypothesis(
                    prediction=high_pd_prediction,
                    measurement=MissedDetection(timestamp=timestamp),
                    distance=999.0,
                ),
            ],
            ctx=ctx,
        )

        self.assertAlmostEqual(log(1.0 - 0.1), scores[0])
        self.assertAlmostEqual(log(1.0 - 0.9), scores[1])

    def test_miss_score_is_near_zero_when_detection_probability_is_zero(self) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        ctx = _scan_context(timestamp=timestamp, detections=[])
        model = NLLScoringModel(
            detection_probability_model=_FixedDetectionProbabilityModel(
                prob_detect=0.0,
                clutter_density=1.0,
            ),
            log_epsilon=1e-12,
        )

        scores = model.score_track_hypotheses(
            hypotheses=[
                SingleDistanceHypothesis(
                    prediction=_prediction(timestamp),
                    measurement=MissedDetection(timestamp=timestamp),
                    distance=999.0,
                ),
            ],
            ctx=ctx,
        )

        self.assertAlmostEqual(0.0, scores[0])

    def test_dynamic_model_rejects_invalid_detection_probability(self) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        ctx = _scan_context(timestamp=timestamp, detections=[])

        for prob_detect in (-0.1, 1.5, nan, inf):
            with self.subTest(prob_detect=prob_detect):
                model = NLLScoringModel(
                    detection_probability_model=_FixedDetectionProbabilityModel(
                        prob_detect=prob_detect,
                        clutter_density=1.0,
                    ),
                    log_epsilon=1e-12,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "detection_probability must return a finite value in " r"\[0, 1\]",
                ):
                    model.score_track_hypotheses(
                        hypotheses=[
                            SingleDistanceHypothesis(
                                prediction=_prediction(timestamp),
                                measurement=MissedDetection(timestamp=timestamp),
                                distance=999.0,
                            ),
                        ],
                        ctx=ctx,
                    )

    def test_dynamic_model_rejects_invalid_clutter_density(self) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        detection = _detection(1.0, timestamp)
        ctx = _scan_context(timestamp=timestamp, detections=[detection])

        for clutter_density in (0.0, -1.0, nan, inf):
            with self.subTest(clutter_density=clutter_density):
                model = NLLScoringModel(
                    detection_probability_model=_FixedDetectionProbabilityModel(
                        prob_detect=0.5,
                        clutter_density=clutter_density,
                    ),
                    log_epsilon=1e-12,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "clutter_density must return a finite, positive density",
                ):
                    model.score_track_hypotheses(
                        hypotheses=[
                            SingleDistanceHypothesis(
                                prediction=_prediction(timestamp),
                                measurement=detection,
                                distance=1.0,
                            ),
                        ],
                        ctx=ctx,
                    )

    def test_dynamic_model_accepts_endpoint_detection_probabilities(self) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        detection = _detection(1.0, timestamp)
        ctx = _scan_context(timestamp=timestamp, detections=[detection])
        prediction = _prediction(timestamp)

        for prob_detect in (0.0, 1.0):
            with self.subTest(prob_detect=prob_detect):
                model = NLLScoringModel(
                    detection_probability_model=_FixedDetectionProbabilityModel(
                        prob_detect=prob_detect,
                        clutter_density=0.25,
                    ),
                    log_epsilon=1e-12,
                )

                scores = model.score_track_hypotheses(
                    hypotheses=[
                        SingleDistanceHypothesis(
                            prediction=prediction,
                            measurement=detection,
                            distance=1.0,
                        ),
                        SingleDistanceHypothesis(
                            prediction=prediction,
                            measurement=MissedDetection(timestamp=timestamp),
                            distance=999.0,
                        ),
                    ],
                    ctx=ctx,
                )

                self.assertTrue(isfinite(scores[0]))
                self.assertTrue(isfinite(scores[1]))
                if prob_detect == 0.0:
                    self.assertAlmostEqual(0.0, scores[1])

    def test_hit_scoring_uses_dynamic_clutter_inputs(self) -> None:
        timestamp = datetime.datetime(2026, 5, 13, 12, 0, 0)
        caller_context = {"sensor": "alpha"}
        detection = _detection(2.0, timestamp)
        prediction = _prediction(timestamp)
        calls: list[tuple[object | None, object, object | None]] = []
        clutter_calls: list[tuple[object, Detection | None, object | None]] = []

        class _CaptureDPM:
            def detection_probability(
                self,
                *,
                track_id: object | None,
                prediction,
                caller_scan_context: object | None,
            ) -> float:
                calls.append((track_id, prediction, caller_scan_context))
                return 0.5

            def clutter_density(
                self,
                *,
                prediction,
                detection: Detection | None,
                caller_scan_context: object | None,
            ) -> float:
                clutter_calls.append((prediction, detection, caller_scan_context))
                return 0.125

        model = NLLScoringModel(
            detection_probability_model=_CaptureDPM(),
            log_epsilon=1e-12,
        )

        scores = model.score_track_hypotheses(
            hypotheses=[
                SingleDistanceHypothesis(
                    prediction=prediction,
                    measurement=detection,
                    distance=1.25,
                ),
            ],
            ctx=_scan_context(
                timestamp=timestamp,
                detections=[detection],
                caller_scan_context=caller_context,
            ),
            track_id="public-7",
        )

        self.assertAlmostEqual(log(0.5) - log(0.125) - 1.25, scores[0])
        self.assertEqual(1, len(calls))
        self.assertEqual("public-7", calls[0][0])
        self.assertIs(prediction, calls[0][1])
        self.assertIs(caller_context, calls[0][2])
        self.assertEqual(1, len(clutter_calls))
        self.assertIs(prediction, clutter_calls[0][0])
        self.assertIs(detection, clutter_calls[0][1])
        self.assertIs(caller_context, clutter_calls[0][2])

    def test_diagnostics_are_silent_when_disabled(self) -> None:
        model = NLLScoringModel(
            detection_probability_model=ConstantDetectionProbabilityModel(
                prob_detect=0.9,
                clutter_density=2.0,
            ),
            log_epsilon=1e-12,
        )
        out = StringIO()

        with redirect_stdout(out):
            maybe_log_scoring_diagnostics(model, enabled=False)

        self.assertEqual("", out.getvalue())

    def test_enabled_diagnostics_report_active_nll_terms_only(self) -> None:
        model = NLLScoringModel(
            detection_probability_model=ConstantDetectionProbabilityModel(
                prob_detect=0.9,
                clutter_density=2.0,
            ),
            log_epsilon=1e-12,
        )
        out = StringIO()

        with redirect_stdout(out):
            maybe_log_scoring_diagnostics(model, enabled=True)

        value = out.getvalue()
        self.assertIn("log_hit_base", value)
        self.assertIn("log_miss", value)
        self.assertNotIn("unused", value)


if __name__ == "__main__":
    unittest.main()
