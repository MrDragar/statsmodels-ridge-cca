"""Gaussian paired data with prespecified population canonical correlations."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticCCAData:
    """Paired observations and their exact population cross-covariance."""

    x: np.ndarray
    y: np.ndarray
    correlations: np.ndarray
    x_weights: np.ndarray
    y_weights: np.ndarray
    cov_xy: np.ndarray


def make_cca_data(n, p, q, correlations, rng=None):
    """Generate Gaussian CCA data with identity marginal covariances.

    Independent Gaussian coordinates are coupled in the first k positions
    using y_j = rho_j*x_j + sqrt(1-rho_j**2)*epsilon_j. Independent orthogonal
    rotations spread the planted directions over the observed features.
    Thus the population canonical correlations are exactly the supplied
    values followed by zeros; finite-sample correlations need not equal them.

    Parameters
    ----------
    n, p, q : int
        Number of paired observations and feature counts in each view.
    correlations : array_like
        Nonincreasing population canonical correlations in [0, 1).
        An empty sequence generates the global null.
    rng : int or numpy.random.Generator, optional
        Seed or local random generator. Global NumPy state is not modified.

    Returns
    -------
    SyntheticCCAData
        Observations, population correlations, canonical directions and
        population cross-covariance. Marginal covariances are identities.
    """
    for value, minimum in [(n, 2), (p, 1), (q, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError("n must be at least 2; p and q must be positive integers")
    rho = np.asarray(correlations, dtype=float)
    if (rho.ndim != 1 or len(rho) > min(p, q) or not np.isfinite(rho).all()
            or np.any(rho < 0) or np.any(rho >= 1) or np.any(np.diff(rho) > 0)):
        raise ValueError("correlations must be finite, nonincreasing, in [0, 1), and fit both views")
    generator = np.random.default_rng(rng)
    qx = np.linalg.qr(generator.standard_normal((p, p)))[0]
    qy = np.linalg.qr(generator.standard_normal((q, q)))[0]
    zx = generator.standard_normal((n, p))
    zy = generator.standard_normal((n, q))
    k = len(rho)
    zy[:, :k] = zx[:, :k] * rho + zy[:, :k] * np.sqrt(1 - rho ** 2)
    ax, ay = qx[:, :k], qy[:, :k]
    return SyntheticCCAData(zx @ qx.T, zy @ qy.T, rho.copy(), ax, ay,
                            (ax * rho) @ ay.T)
