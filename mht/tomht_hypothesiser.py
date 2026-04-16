"""Tracker-owned Stone Soup distance hypothesiser for TOMHT local expansion."""

from __future__ import annotations

from math import log, pi, sqrt
import sys
from typing import Iterable, Sequence

import numpy as np

from stonesoup.base import Property
from stonesoup.hypothesiser.base import Hypothesiser
from stonesoup.predictor.base import Predictor
from stonesoup.types.detection import Detection, MissedDetection
from stonesoup.types.hypothesis import SingleDistanceHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.track import Track
from stonesoup.updater.base import Updater


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
        prediction = self.predictor.predict(track, timestamp=timestamp, **kwargs)
        # Miss distance is a sentinel; tracker scoring computes miss score directly.
        miss_distance = 0.0
        hypotheses.append(
            SingleDistanceHypothesis(
                prediction=prediction,
                measurement=MissedDetection(timestamp=timestamp),
                distance=miss_distance,
            )
        )

        for detection in detections:
            detection_timestamp = (
                timestamp if detection.timestamp is None else detection.timestamp
            )
            prediction = self.predictor.predict(
                track, timestamp=detection_timestamp, **kwargs
            )
            measurement_prediction = self.updater.predict_measurement(
                prediction, detection.measurement_model, **kwargs
            )
            innovation = (
                np.asarray(detection.state_vector, dtype=float)
                - np.asarray(measurement_prediction.mean, dtype=float)
            ).ravel()
            log_prob, gate_distance_squared = self._log_likelihood_and_gate_distance(
                innovation,
                np.asarray(measurement_prediction.covar, dtype=float),
            )
            if gate_distance_squared > gate_threshold_squared:
                continue

            nll = -log_prob
            gate_distance = sqrt(max(0.0, gate_distance_squared))
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

    @staticmethod
    def _log_likelihood_and_gate_distance(
        innovation: np.ndarray,
        covariance: np.ndarray,
        relative_eigenvalue_floor: float = 1e-9,
    ) -> tuple[float, float]:
        """Return ``(log_likelihood, squared_mahalanobis)`` for one innovation.

        Baseline/reference behavior is Stone Soup ``PDAHypothesiser``:
        - use innovation covariance as the Gaussian likelihood covariance,
        - compute squared-Mahalanobis distance for gating,
        - compute Gaussian log-likelihood from the same covariance.

        This tracker-owned path keeps that math but makes a few explicit choices:
        - explicit linear algebra:
          we compute both metrics in one place from the same solved system
          and log-determinant, rather than relying on multiple helper layers;
        - reduced duplicate work:
          the returned squared-Mahalanobis distance is reused by caller gating,
          so we avoid recomputing equivalent distance terms in separate paths;
        - SPD repair guardrail:
          if covariance is not numerically SPD, apply a minimal symmetric
          eigenvalue floor (relative to covariance scale) before solve/logdet to
          preserve stability.
        """
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

        innovation_col = innovation.reshape(-1, 1)
        solved = np.linalg.solve(covariance_spd, innovation_col)
        squared_mahalanobis = float((innovation_col.T @ solved).reshape(()))
        _, logdet = np.linalg.slogdet(covariance_spd)
        ndim = int(innovation_col.shape[0])
        log_likelihood = -0.5 * (
            ndim * log(2.0 * pi) + float(logdet) + squared_mahalanobis
        )
        return log_likelihood, squared_mahalanobis
