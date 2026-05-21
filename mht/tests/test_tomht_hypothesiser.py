from __future__ import annotations

import datetime
from dataclasses import dataclass
from math import sqrt
import unittest
from typing import cast

import numpy as np
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from mht.tomht_hypothesiser import TrackerOwnedNLLDistanceHypothesiser


class _CountingPredictor:
    def __init__(self) -> None:
        self.calls: list[datetime.datetime | None] = []
        self.priors: list[object] = []

    def predict(self, prior, timestamp=None, **kwargs) -> GaussianState:
        self.priors.append(prior)
        del kwargs
        self.calls.append(timestamp)
        return GaussianState([0.0, 0.0], covar=np.eye(2), timestamp=timestamp)


@dataclass(frozen=True)
class _MeasurementPrediction:
    mean: np.ndarray
    covar: np.ndarray


class _ConstantMeasurementPredictionUpdater:
    def predict_measurement(self, prediction, measurement_model, **kwargs):
        del prediction, measurement_model, kwargs
        return _MeasurementPrediction(
            mean=np.array([[0.0], [0.0]], dtype=float),
            covar=np.eye(2),
        )


class _CountingMeasurementPredictionUpdater:
    def __init__(self) -> None:
        self.calls = 0

    def predict_measurement(self, prediction, measurement_model, **kwargs):
        del prediction, measurement_model, kwargs
        self.calls += 1
        return _MeasurementPrediction(
            mean=np.array([[0.0], [0.0]], dtype=float),
            covar=np.eye(2),
        )


class TOMHTHypothesiserMathHelpersTest(unittest.TestCase):
    def test_hypothesise_reuses_scan_prediction_for_scan_timestamp_detections(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 4, 16, 12, 0, 0)
        other_timestamp = timestamp + datetime.timedelta(seconds=1)
        predictor = _CountingPredictor()
        hypothesiser = TrackerOwnedNLLDistanceHypothesiser(
            predictor=predictor,
            updater=_ConstantMeasurementPredictionUpdater(),
            mahalanobis_gate_threshold=3.0,
        )
        track = Track([GaussianState([0.0, 0.0], covar=np.eye(2), timestamp=timestamp)])
        detections = [
            Detection(np.array([[0.0], [0.0]]), timestamp=None),
            Detection(np.array([[0.0], [0.0]]), timestamp=other_timestamp),
        ]

        hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

        self.assertGreaterEqual(len(hypotheses.single_hypotheses), 1)
        self.assertEqual([timestamp, other_timestamp], predictor.calls)
        self.assertTrue(all(prior is track.state for prior in predictor.priors))
        self.assertTrue(all(not isinstance(prior, Track) for prior in predictor.priors))

    def test_hypothesise_reuses_measurement_prediction_for_same_object_inputs(
        self,
    ) -> None:
        timestamp = datetime.datetime(2026, 4, 16, 12, 0, 0)
        predictor = _CountingPredictor()
        updater = _CountingMeasurementPredictionUpdater()
        hypothesiser = TrackerOwnedNLLDistanceHypothesiser(
            predictor=predictor,
            updater=updater,
            mahalanobis_gate_threshold=3.0,
        )
        track = Track([GaussianState([0.0, 0.0], covar=np.eye(2), timestamp=timestamp)])
        detections = [
            Detection(np.array([[0.0], [0.0]]), timestamp=None),
            Detection(np.array([[0.0], [0.0]]), timestamp=None),
        ]

        hypotheses = hypothesiser.hypothesise(track, detections, timestamp)

        self.assertGreaterEqual(len(hypotheses.single_hypotheses), 1)
        self.assertEqual([timestamp], predictor.calls)
        self.assertEqual(1, updater.calls)

    def test_prepare_innovation_covariance_includes_cholesky_and_logdet(self) -> None:
        covariance = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=float)
        prepared = TrackerOwnedNLLDistanceHypothesiser._prepare_innovation_covariance(
            covariance
        )
        expected_logdet = 2.0 * float(np.sum(np.log(np.diag(prepared.cholesky_factor))))
        self.assertAlmostEqual(expected_logdet, prepared.logdet)
        np.testing.assert_allclose(
            prepared.cholesky_factor @ prepared.cholesky_factor.T,
            prepared.covariance_spd,
        )

    def test_rectangular_pre_gate_from_covariance_uses_raw_diagonal(self) -> None:
        covariance = np.array([[4.0, 100.0], [100.0, 3.0]], dtype=float)
        gate_threshold_squared = 9.0

        self.assertFalse(
            TrackerOwnedNLLDistanceHypothesiser._fails_rectangular_pre_gate(
                np.array([6.0, 0.0], dtype=float),
                covariance,
                gate_threshold_squared,
            )
        )
        self.assertTrue(
            TrackerOwnedNLLDistanceHypothesiser._fails_rectangular_pre_gate(
                np.array([6.1, 0.0], dtype=float),
                covariance,
                gate_threshold_squared,
            )
        )

    def test_rectangular_pre_gate_rejects_only_outside_axis_bounds(self) -> None:
        covariance = np.array([[4.0, 1.5], [1.5, 3.0]], dtype=float)
        gate_threshold_squared = 9.0

        self.assertFalse(
            TrackerOwnedNLLDistanceHypothesiser._fails_rectangular_pre_gate(
                np.array([6.0, 0.0], dtype=float),
                covariance,
                gate_threshold_squared,
            )
        )
        self.assertTrue(
            TrackerOwnedNLLDistanceHypothesiser._fails_rectangular_pre_gate(
                np.array([6.1, 0.0], dtype=float),
                covariance,
                gate_threshold_squared,
            )
        )

    def test_rectangular_pre_gate_is_conservative_for_inside_ellipsoid(self) -> None:
        covariance = np.array([[2.0, 1.2], [1.2, 1.5]], dtype=float)
        prepared = TrackerOwnedNLLDistanceHypothesiser._prepare_innovation_covariance(
            covariance
        )
        gate_threshold_squared = 9.0
        chol = np.linalg.cholesky(prepared.covariance_spd)
        rng = np.random.default_rng(20260416)

        for _ in range(200):
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            radius = float(rng.random())
            whitened = direction * (radius * sqrt(gate_threshold_squared))
            innovation = chol @ whitened

            _, gate_distance_squared = (
                TrackerOwnedNLLDistanceHypothesiser._log_likelihood_and_gate_distance(
                    innovation, prepared
                )
            )
            self.assertLessEqual(gate_distance_squared, gate_threshold_squared + 1e-9)
            self.assertFalse(
                TrackerOwnedNLLDistanceHypothesiser._fails_rectangular_pre_gate(
                    innovation,
                    covariance,
                    gate_threshold_squared,
                )
            )

    def test_log_likelihood_and_gate_distance_matches_full_solve(self) -> None:
        covariance = np.array([[2.0, 1.2], [1.2, 1.5]], dtype=float)
        innovation = np.array([0.7, -0.3], dtype=float)
        prepared = TrackerOwnedNLLDistanceHypothesiser._prepare_innovation_covariance(
            covariance
        )

        log_likelihood, gate_distance_squared = (
            TrackerOwnedNLLDistanceHypothesiser._log_likelihood_and_gate_distance(
                innovation, prepared
            )
        )

        innovation_col = innovation.reshape(-1, 1)
        full_solve = np.linalg.solve(prepared.covariance_spd, innovation_col)
        expected_gate_distance_squared = float(
            (innovation_col.T @ full_solve).reshape(())
        )
        self.assertAlmostEqual(expected_gate_distance_squared, gate_distance_squared)

        ndim = innovation_col.shape[0]
        expected_log_likelihood = -0.5 * (
            ndim * np.log(2.0 * np.pi)
            + prepared.logdet
            + expected_gate_distance_squared
        )
        self.assertAlmostEqual(expected_log_likelihood, log_likelihood)

    def test_covariance_prep_cache_reuses_only_exact_equal_input(self) -> None:
        hypothesiser = cast(
            TrackerOwnedNLLDistanceHypothesiser,
            object.__new__(TrackerOwnedNLLDistanceHypothesiser),
        )
        covariance = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=float)

        prepared_1, cache_1 = hypothesiser._prepare_covariance(
            covariance=covariance,
            cache=None,
        )
        prepared_2, cache_2 = hypothesiser._prepare_covariance(
            covariance=np.array(covariance, copy=True),
            cache=cache_1,
        )

        self.assertIs(prepared_1, prepared_2)
        self.assertIs(cache_1, cache_2)

        covariance_changed = np.array(covariance, copy=True)
        covariance_changed[0, 0] += 1e-12
        prepared_3, _ = hypothesiser._prepare_covariance(
            covariance=covariance_changed,
            cache=cache_2,
        )

        self.assertIsNot(prepared_2, prepared_3)
