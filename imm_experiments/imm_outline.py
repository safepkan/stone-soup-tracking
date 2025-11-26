from __future__ import annotations

from typing import List, Sequence
import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.base import Property
from stonesoup.predictor.base import Predictor
from stonesoup.updater.base import Updater

from stonesoup.types.state import GaussianState, TaggedWeightedGaussianState
from stonesoup.types.prediction import Prediction, GaussianMeasurementPrediction
from stonesoup.types.update import Update, TaggedWeightedGaussianStateUpdate
from stonesoup.types.hypothesis import SingleHypothesis


def _moment_match_gaussians(
    means: Sequence[np.ndarray],
    covars: Sequence[np.ndarray],
    weights: Sequence[float],
):
    """Return mean/covar of Gaussian mixture via moment matching."""
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    m = np.zeros_like(means[0], dtype=float)
    for wi, mi in zip(w, means):
        m += wi * mi

    P = np.zeros_like(covars[0], dtype=float)
    for wi, mi, Pi in zip(w, means, covars):
        d = mi - m
        P += wi * (Pi + d @ d.T)

    return m, P


class IMMState(GaussianState):
    """Gaussian state that also carries per-mode Gaussian components.

    Components are TaggedWeightedGaussianState, where:
      - component.weight == mode probability
      - component.tag    == stable model identifier (string)
    """

    components: List[TaggedWeightedGaussianState] = Property(
        default=None,
        doc="Per-mode Gaussian components (mean/covar + mode probability).",
    )

    def __init__(self, *args, components=None, **kwargs):
        if components is None:
            components = []
        # Compute fused mean/covar from components (moment matched)
        if components:
            weights = [float(c.weight) for c in components]
            means = [np.asarray(c.state_vector, dtype=float) for c in components]
            covars = [np.asarray(c.covar, dtype=float) for c in components]
            mean, covar = _moment_match_gaussians(means, covars, weights)
            kwargs.setdefault("state_vector", mean)
            kwargs.setdefault("covar", covar)
            kwargs.setdefault("timestamp", components[0].timestamp)
        super().__init__(*args, components=components, **kwargs)


class IMMStatePrediction(Prediction, IMMState):
    """CreatableFromState-compatible prediction wrapper for IMMState."""

    pass


class IMMStateUpdate(Update, IMMState):
    """CreatableFromState-compatible update wrapper for IMMState."""

    pass


class IMMPredictor(Predictor):
    """IMM predictor wrapper around existing per-model predictors.

    Assumes all models live in the same state space dimension.
    """

    predictors: Sequence[Predictor] = Property(
        doc="Per-model predictors (e.g. KalmanPredictor)."
    )
    transition_matrix: np.ndarray = Property(
        doc="Model transition matrix P(i->j), shape (M,M)."
    )
    model_tags: Sequence[str] = Property(
        doc="Stable tag per model (len=M), aligns with predictors."
    )

    def predict(self, prior: IMMState, timestamp=None, **kwargs) -> IMMStatePrediction:
        M = len(self.predictors)
        Pij = np.asarray(self.transition_matrix, dtype=float)
        assert Pij.shape == (M, M)

        # Align components to model order
        comp_by_tag = {c.tag: c for c in prior.components}
        comps = [comp_by_tag[tag] for tag in self.model_tags]
        mu = np.asarray(
            [float(c.weight) for c in comps], dtype=float
        )  # mode probs at k-1
        mu = mu / np.sum(mu)

        # IMM mixing: c_j = sum_i mu_i * P(i->j)
        c = mu @ Pij  # shape (M,)
        c = np.clip(c, 1e-300, None)

        mixed_inputs = []
        for j in range(M):
            muj = (mu * Pij[:, j]) / c[j]  # mixing probs mu_{i|j}

            means = [np.asarray(comps[i].state_vector, dtype=float) for i in range(M)]
            covars = [np.asarray(comps[i].covar, dtype=float) for i in range(M)]
            x0j, P0j = _moment_match_gaussians(means, covars, muj)

            mixed_inputs.append(GaussianState(x0j, P0j, timestamp=prior.timestamp))

        # Per-model prediction
        pred_components: List[TaggedWeightedGaussianState] = []
        for j, (tag, pred, mixed_prior) in enumerate(
            zip(self.model_tags, self.predictors, mixed_inputs)
        ):
            pj = pred.predict(mixed_prior, timestamp=timestamp, **kwargs)
            pred_components.append(
                TaggedWeightedGaussianState(
                    state_vector=pj.state_vector,
                    covar=pj.covar,
                    timestamp=pj.timestamp,
                    weight=float(c[j]),  # predicted mode probability (pre-update)
                    tag=tag,
                )
            )

        # Normalise predicted mode probs
        s = sum(float(c.weight) for c in pred_components)
        for cpt in pred_components:
            cpt.weight = float(cpt.weight) / s

        return IMMStatePrediction(
            state_vector=IMMState(components=pred_components).state_vector,
            covar=IMMState(components=pred_components).covar,
            timestamp=timestamp,
            components=pred_components,
            transition_model=self.transition_model,  # not meaningful here, but required by base API
            prior=prior,
        )


class IMMUpdater(Updater):
    """IMM updater wrapper around existing per-model Kalman-type updaters."""

    updaters: Sequence[Updater] = Property(
        doc="Per-model updaters (e.g. KalmanUpdater)."
    )
    model_tags: Sequence[str] = Property(
        doc="Stable tag per model (len=M), aligns with updaters."
    )

    def predict_measurement(
        self,
        predicted_state: IMMStatePrediction,
        measurement_model=None,
        measurement_noise=True,
        **kwargs,
    ) -> GaussianMeasurementPrediction:
        """Moment-match the per-mode predicted measurements into a single GaussianMeasurementPrediction.

        This is the key to making standard data-association (e.g. Mahalanobis distance) work nicely.
        """
        comps = {c.tag: c for c in predicted_state.components}
        meas_preds = []
        weights = []

        for tag, upd in zip(self.model_tags, self.updaters):
            c = comps[tag]
            mp = upd.predict_measurement(
                c,
                measurement_model=measurement_model,
                measurement_noise=measurement_noise,
                **kwargs,
            )
            meas_preds.append(mp)
            weights.append(float(c.weight))

        means = [np.asarray(mp.state_vector, dtype=float) for mp in meas_preds]
        covars = [np.asarray(mp.covar, dtype=float) for mp in meas_preds]
        m, P = _moment_match_gaussians(means, covars, weights)

        return GaussianMeasurementPrediction(
            state_vector=m, covar=P, timestamp=predicted_state.timestamp
        )

    def update(self, hypothesis: SingleHypothesis, **kwargs) -> IMMStateUpdate:
        """Update each mode, then update mode probabilities using measurement likelihood."""
        predicted: IMMStatePrediction = hypothesis.prediction
        detection = hypothesis.measurement

        comps = {c.tag: c for c in predicted.components}

        updated_components: List[TaggedWeightedGaussianStateUpdate] = []
        post_mode_unnorm = []

        for tag, upd in zip(self.model_tags, self.updaters):
            pred_comp = comps[tag]

            # Build per-mode hypothesis and run existing Kalman update
            mode_hyp = SingleHypothesis(prediction=pred_comp, measurement=detection)
            mode_upd = upd.update(mode_hyp, **kwargs)  # typically GaussianStateUpdate

            # Likelihood p(z | mode=j)
            mp = upd.predict_measurement(
                pred_comp, measurement_model=None, measurement_noise=True
            )
            z = np.asarray(detection.state_vector, dtype=float).ravel()
            mean = np.asarray(mp.state_vector, dtype=float).ravel()
            S = np.asarray(mp.covar, dtype=float)
            lh = float(multivariate_normal.pdf(z, mean=mean, cov=S))

            post_mode_unnorm.append(float(pred_comp.weight) * lh)

            updated_components.append(
                TaggedWeightedGaussianStateUpdate(
                    state_vector=mode_upd.state_vector,
                    covar=mode_upd.covar,
                    timestamp=mode_upd.timestamp,
                    hypothesis=hypothesis,  # keep original association
                    weight=0.0,  # filled after normalisation
                    tag=tag,
                )
            )

        post_mode_unnorm = np.asarray(post_mode_unnorm, dtype=float)
        post_mode = post_mode_unnorm / np.sum(post_mode_unnorm)

        for cpt, w in zip(updated_components, post_mode):
            cpt.weight = float(w)

        fused = IMMState(components=updated_components)

        return IMMStateUpdate(
            state_vector=fused.state_vector,
            covar=fused.covar,
            timestamp=updated_components[0].timestamp,
            components=updated_components,
            hypothesis=hypothesis,
        )
