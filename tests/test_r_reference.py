"""Compare the Python core with outputs actually computed by R CCA::rcc.

Reference outputs were generated with Rscript research/r_reference.R.
The saved fixtures allow these tests to run without R or CCA-Zoo installed.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from research.ridge import fit_ridge

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case,alpha,scale", [
    ("classical", (0, 0), False),
    ("ridge_lowdim", (0.3, 1.2), False),
    ("ridge_wide", (0.3, 1.2), False),
])
def test_matches_actual_r_roots_correlations_and_coefficient_subspaces(case, alpha, scale):
    inputs = ROOT / "statsmodels-checkout/statsmodels/multivariate/tests/results/ridge_cca" / case
    x = np.loadtxt(inputs / "x.csv", delimiter=",")
    y = np.loadtxt(inputs / "y.csv", delimiter=",")
    expected = np.loadtxt(inputs / "r_results.csv", delimiter=",", skiprows=1)
    fit = fit_ridge(x, y, alpha_x=alpha[0], alpha_y=alpha[1], n_components=3, scale=scale)
    assert_allclose(fit.penalized_roots, expected[:, 0], atol=1e-8, rtol=1e-6)
    assert_allclose(fit.train_correlations, expected[:, 1], atol=1e-8, rtol=1e-6)
    for view, weights in [("x", fit.x_weights), ("y", fit.y_weights)]:
        r_weights = np.loadtxt(inputs / f"r_{view}_weights.csv", delimiter=",")
        q_python = np.linalg.qr(weights, mode="reduced")[0]
        q_r = np.linalg.qr(r_weights, mode="reduced")[0]
        cosines = np.linalg.svd(q_python.T @ q_r, compute_uv=False)
        assert_allclose(cosines, np.ones(3), atol=1e-8, rtol=1e-6)
