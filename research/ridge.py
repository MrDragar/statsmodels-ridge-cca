"""Compatibility adapter for the original experiments, using the real CanCorr.

The numerical implementation now lives in statsmodels-checkout. This adapter
only keeps the original experiment result names and covariance normalization.
"""

from dataclasses import dataclass

import numpy as np
from statsmodels.multivariate.cancorr import CanCorr


def _views(x, y, min_rows=2):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or not x.shape[1] or not y.shape[1]:
        raise ValueError("Views must be nonempty two-dimensional arrays")
    if len(x) != len(y) or len(x) < min_rows:
        raise ValueError(f"Views must have equal row counts and at least {min_rows} rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Views must contain only finite values")
    return x, y


@dataclass
class RidgeFit:
    """Frozen train preprocessing and coefficients, in covariance normalization.

    ``penalized_roots`` are objective values; ``train_correlations`` are Pearson
    correlations. Coefficients act on centered, optionally standardized data.
    This research result deliberately has no ``cancorr`` or ``corr_test`` API.
    """

    x_mean: np.ndarray
    y_mean: np.ndarray
    x_scale: np.ndarray
    y_scale: np.ndarray
    x_weights: np.ndarray
    y_weights: np.ndarray
    penalized_roots: np.ndarray
    train_correlations: np.ndarray
    lambda_x: float
    lambda_y: float

    def transform(self, x, y):
        """Project paired new observations using train means, scales and weights."""
        x, y = _views(x, y)
        if x.shape[1] != len(self.x_mean) or y.shape[1] != len(self.y_mean):
            raise ValueError("Feature counts must match training views")
        return (((x - self.x_mean) / self.x_scale) @ self.x_weights,
                ((y - self.y_mean) / self.y_scale) @ self.y_weights)


def fit_ridge(x, y, *, alpha_x=1.0, alpha_y=1.0, n_components=1, scale=False):
    """Fit additive ridge with lambda = alpha * trace(sample covariance) / p.

    The thin-SVD solver avoids p-by-p whitening. Positive penalties allow
    collinear/wide views; a zero penalty requires full column rank in that
    view. Components are limited to the observed numerical ranks. Constants
    must be removed in train folds by the caller. No CV or inference is done.
    """
    x, y = _views(x, y, min_rows=3)
    for alpha in (alpha_x, alpha_y):
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Penalties must be finite nonnegative scalars")
    if (isinstance(n_components, (bool, np.bool_))
            or not isinstance(n_components, (int, np.integer)) or n_components < 1):
        raise ValueError("n_components must be a positive integer")

    x_std, y_std = x.std(0, ddof=1), y.std(0, ddof=1)
    if np.any(x_std == 0) or np.any(y_std == 0):
        raise ValueError("Constant training columns must be removed within the train fold")
    model = CanCorr(y, x, penalty=(alpha_x, alpha_y), standardize=scale)
    if n_components > len(model.cancorr):
        raise ValueError("n_components exceeds the observed numerical rank")
    factor = np.sqrt(len(x) - 1)
    a = model.x_cancoef[:, :n_components] * model.x_scale[:, None] * factor
    b = model.y_cancoef[:, :n_components] * model.y_scale[:, None] * factor
    return RidgeFit(model.x_mean, model.y_mean, model.x_scale, model.y_scale,
                    a, b, model.penalized_roots[:n_components],
                    model.cancorr[:n_components], model.lambda_x, model.lambda_y)


def paired_correlations(u, v):
    """Signed column-wise Pearson correlations; undefined scores raise an error."""
    u, v = _views(u, v)
    if u.shape != v.shape:
        raise ValueError("Score arrays must have identical shapes")
    uc, vc = u - u.mean(0), v - v.mean(0)
    norms = np.linalg.norm(uc, axis=0) * np.linalg.norm(vc, axis=0)
    if np.any(norms == 0):
        raise ValueError("Pearson correlation is undefined for constant scores")
    return np.sum(uc * vc, axis=0) / norms
