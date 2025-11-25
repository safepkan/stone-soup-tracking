import numpy as np
from scipy.stats import chi2

from stonesoup.types.detection import MissedDetection
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.track import Track


def _as_float_col(x) -> np.ndarray:
    """Convert Stone Soup vectors/matrices to a float column ndarray."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _as_float_mat(x) -> np.ndarray:
    """Convert Stone Soup matrices to a float ndarray."""
    return np.asarray(x, dtype=float)


class RobustPDAHypothesiser:
    """PDA-style hypothesiser that tolerates near-singular/indefinite covariances."""

    def __init__(
        self,
        predictor,
        updater,
        clutter_spatial_density: float,
        prob_detect: float,
        prob_gate: float,
    ) -> None:
        self.predictor = predictor
        self.updater = updater
        self.clutter_spatial_density = float(clutter_spatial_density)
        self.prob_detect = float(prob_detect)
        self.prob_gate = float(prob_gate)

    @staticmethod
    def _ensure_spd(cov: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, float]:
        cov = 0.5 * (cov + cov.T)
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, eps)
        cov_spd = (v * w) @ v.T
        logdet = float(np.sum(np.log(w)))
        return cov_spd, logdet

    @staticmethod
    def _logpdf(x: np.ndarray, cov: np.ndarray, logdet: float) -> float:
        d = x.shape[0]
        maha = float(x.T @ np.linalg.solve(cov, x))
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)

    def hypothesise(self, track: Track, detections, timestamp, **kwargs):
        prediction = self.predictor.predict(track[-1], timestamp=timestamp)
        meas_pred = self.updater.predict_measurement(prediction)

        mean = getattr(meas_pred, "mean", None)
        if mean is None:
            mean = meas_pred.state_vector
        mean = _as_float_col(mean)

        cov = _as_float_mat(meas_pred.covar)
        cov, logdet = self._ensure_spd(cov)

        dim_z = int(mean.shape[0])
        gate_thresh = float(chi2.ppf(self.prob_gate, dim_z))

        unnorm: list[tuple[object, float]] = []

        # Detection hypotheses
        for det in detections:
            z = _as_float_col(det.state_vector)
            innov = z - mean
            maha = float(innov.T @ np.linalg.solve(cov, innov))
            if maha > gate_thresh:
                continue

            ll = self._logpdf(innov, cov, logdet)
            # PDA-style unnormalised weight (good enough for ranking)
            w = self.prob_detect * np.exp(ll) / max(self.clutter_spatial_density, 1e-12)

            unnorm.append((det, float(w)))

        # Missed detection hypothesis
        miss_w = max(1.0 - self.prob_detect * self.prob_gate, 1e-12)
        miss_det = MissedDetection(timestamp=timestamp)
        unnorm.append((miss_det, float(miss_w)))

        hyps = [
            SingleProbabilityHypothesis(
                prediction=prediction,
                measurement=meas,
                probability=Probability(w),
            )
            for meas, w in unnorm
        ]

        return MultipleHypothesis(hyps, normalise=True)
