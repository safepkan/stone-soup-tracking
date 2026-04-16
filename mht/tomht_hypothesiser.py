"""Tracker-owned Stone Soup distance hypothesiser for TOMHT local expansion."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from stonesoup.base import Property
from stonesoup.hypothesiser.base import Hypothesiser
from stonesoup.predictor.base import Predictor
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.track import Track
from stonesoup.updater.base import Updater


@dataclass(frozen=True)
class _PreparedCovariance:
    covariance_spd: np.ndarray
    diagonal: np.ndarray
    logdet: float


@dataclass(frozen=True)
class _CovariancePrepCache:
    covariance_input: np.ndarray
    prepared_covariance: _PreparedCovariance


@dataclass(frozen=True)
class _MeasurementPredictionCache:
    prediction: object
    measurement_model: object
    measurement_prediction: Any


class SingleDualDistanceHypothesis(SingleDistanceHypothesis):
    """Distance hypothesis that also carries Mahalanobis gate distance."""

    if sys.version_info >= (3, 14):
        gate_distance = Property(float, doc="Mahalanobis distance (for gating)")
    else:
        gate_distance: float = Property(doc="Mahalanobis distance (for gating)")


class TrackerOwnedNLLDistanceHypothesiser(Hypothesiser):
    """Tracker-owned default distance hypothesiser.

    Behavior:
    - always emits one missed-detection option,
    - gates detections by Mahalanobis distance,
    - emits one option per gated detection with
      ``distance = NLL = -log p(z | x)``.

    Notes:
    - this object does not own local hit-vs-miss-vs-clutter scoring constants,
    - missed-detection distance is a sentinel only and is ignored by the
      tracker scoring model.
    """

    if sys.version_info >= (3, 14):
        predictor = Property(Predictor, doc="Predictor used for state prediction.")
        updater = Property(
            Updater,
            doc="Updater used for measurement prediction in local association.",
        )
        mahalanobis_gate_threshold = Property(
            float,
            default=3.0,
            doc="Mahalanobis threshold used for detection gating.",
        )
    else:
        predictor: Predictor = Property(doc="Predictor used for state prediction.")
        updater: Updater = Property(
            doc="Updater used for measurement prediction in local association."
        )
        mahalanobis_gate_threshold: float = Property(
            default=3.0,
            doc="Mahalanobis threshold used for detection gating.",
        )

    def hypothesise(
        self,
        track: Track,
        detections: Iterable[Detection],
        timestamp,
        **kwargs,
    ) -> MultipleHypothesis:
        """Generate distance hypotheses for one track at one scan timestamp."""
        gate_threshold_mahalanobis = float(self.mahalanobis_gate_threshold)
        if gate_threshold_mahalanobis <= 0.0:
            raise ValueError("mahalanobis_gate_threshold must be > 0.")
        gate_threshold_squared = gate_threshold_mahalanobis**2

        hypotheses: list[SingleDistanceHypothesis] = []
        prep_cache: _CovariancePrepCache | None = None
        meas_pred_cache: _MeasurementPredictionCache | None = None

        scan_prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)

        # Miss hypothesis
        # Miss distance is a sentinel; tracker scoring computes miss score directly.
        miss_distance = 0.0
        hypotheses.append(
            SingleDistanceHypothesis(
                prediction=scan_prediction,
                measurement=MissedDetection(timestamp=timestamp),
                distance=miss_distance,
            )
        )

        # Detection hypotheses
        for detection in detections:
            # Get prediction, with timestamp-matching reuse
            detection_timestamp = (
                timestamp if detection.timestamp is None else detection.timestamp
            )
            if detection_timestamp == timestamp:
                prediction = scan_prediction
            else:
                prediction = self.predictor.predict(
                    track, timestamp=detection_timestamp, **kwargs
                )

            # Get measurement prediction, with object-identity reuse
            measurement_model = detection.measurement_model
            if (
                meas_pred_cache is not None
                and prediction is meas_pred_cache.prediction
                and measurement_model is meas_pred_cache.measurement_model
            ):
                measurement_prediction = meas_pred_cache.measurement_prediction
            else:
                measurement_prediction = self.updater.predict_measurement(
                    prediction, measurement_model, **kwargs
                )
                meas_pred_cache = _MeasurementPredictionCache(
                    prediction=prediction,
                    measurement_model=measurement_model,
                    measurement_prediction=measurement_prediction,
                )

            # Get innovation and covariance
            covariance = np.asarray(measurement_prediction.covar, dtype=float)
            innovation = (
                np.asarray(detection.state_vector, dtype=float)
                - np.asarray(measurement_prediction.mean, dtype=float)
            ).ravel()

            # Rectangular gating before full Mahalanobis/NLL work
            if self._fails_rectangular_pre_gate(
                innovation, covariance, gate_threshold_squared
            ):
                continue

            # Prepare covariance once and reuse prior prep on exact-equal input
            # for log likelihood and Mahalanobis distance computation
            prep_cov, prep_cache = self._prepare_covariance(covariance, prep_cache)
            log_prob, gate_dist_squared = self._log_likelihood_and_gate_distance(
                innovation, prep_cov
            )

            # Full Mahalanobis gating
            if gate_dist_squared > gate_threshold_squared:
                continue

            # Create hypothesis with NLL and Mahalanobis distance
            nll = -log_prob
            gate_distance = sqrt(max(0.0, gate_dist_squared))
            hypotheses.append(
                SingleDualDistanceHypothesis(
                    prediction=prediction,
                    measurement=detection,
                    distance=nll,
                    measurement_prediction=measurement_prediction,
                    gate_distance=gate_distance,
                ),
            )

        ordered: Sequence[SingleDistanceHypothesis] = sorted(hypotheses, reverse=True)
        return MultipleHypothesis(ordered, normalise=False)

    def _prepare_covariance(
        self,
        covariance: np.ndarray,
        cache: _CovariancePrepCache | None,
        relative_eigenvalue_floor: float = 1e-9,
    ) -> tuple[_PreparedCovariance, _CovariancePrepCache]:
        """Prepare covariance once and reuse prior prep on exact-equal input."""
        if cache is not None and np.array_equal(covariance, cache.covariance_input):
            return cache.prepared_covariance, cache
        prepared = self._prepare_innovation_covariance(
            covariance=covariance,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
        )
        return prepared, _CovariancePrepCache(
            covariance_input=np.array(covariance, copy=True),
            prepared_covariance=prepared,
        )

    @staticmethod
    def _fails_rectangular_pre_gate(
        innovation: np.ndarray,
        covariance: np.ndarray,
        gate_threshold_squared: float,
    ) -> bool:
        """Return whether innovation is outside rectangular gate from raw cov diag."""
        if gate_threshold_squared <= 0.0:
            raise ValueError("gate_threshold_squared must be > 0.")
        max_axis_distance_squared = gate_threshold_squared * np.maximum(
            np.diag(covariance).astype(float, copy=False), 0.0
        )
        return bool(np.any(np.square(innovation) > max_axis_distance_squared))

    @staticmethod
    def _prepare_innovation_covariance(
        covariance: np.ndarray,
        relative_eigenvalue_floor: float = 1e-9,
    ) -> _PreparedCovariance:
        """Return SPD covariance prep bundle (matrix, diagonal, and logdet)."""
        if relative_eigenvalue_floor <= 0.0:
            raise ValueError("relative_eigenvalue_floor must be > 0.")
        covariance = 0.5 * (covariance + covariance.T)
        try:
            np.linalg.cholesky(covariance)
            covariance_spd = covariance
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.asarray(eigenvalues, dtype=float)
            largest_eigenvalue = float(np.max(eigenvalues))
            if largest_eigenvalue > 0.0:
                covariance_scale = largest_eigenvalue
            else:
                # Pathological case: non-positive spectrum only. Fall back to the
                # largest magnitude to keep the floor scale-relative.
                covariance_scale = float(np.max(np.abs(eigenvalues)))
            if covariance_scale == 0.0:
                covariance_scale = np.finfo(float).eps

            eigenvalue_floor = relative_eigenvalue_floor * covariance_scale
            floored = np.maximum(eigenvalues, eigenvalue_floor)
            covariance_spd = (eigenvectors * floored) @ eigenvectors.T

        return _PreparedCovariance(
            covariance_spd=covariance_spd,
            diagonal=np.diag(covariance_spd).astype(float, copy=False),
            logdet=float(np.linalg.slogdet(covariance_spd)[1]),
        )

    @staticmethod
    def _log_likelihood_and_gate_distance(
        innovation: np.ndarray,
        prepared_covariance: _PreparedCovariance,
    ) -> tuple[float, float]:
        """Return ``(log_likelihood, squared_mahalanobis)`` from prepared covariance."""
        innovation_col = innovation.reshape(-1, 1)
        solved = np.linalg.solve(prepared_covariance.covariance_spd, innovation_col)
        squared_mahalanobis = float((innovation_col.T @ solved).reshape(()))
        ndim = int(innovation_col.shape[0])
        log_likelihood = -0.5 * (
            ndim * log(2.0 * pi) + prepared_covariance.logdet + squared_mahalanobis
        )
        return log_likelihood, squared_mahalanobis
