"""Numerical contracts for the research prototype, with independent oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.multivariate.cancorr import CanCorr

from research.ridge import fit_ridge, paired_correlations


def sample(n=90, p=6, q=4):
    rng = np.random.default_rng(431)
    x = rng.normal(size=(n, p))
    y = x[:, :1] @ rng.normal(size=(1, q)) + rng.normal(size=(n, q))
    return x, y


def test_zero_penalty_agrees_with_statsmodels_and_its_coefficient_scale():
    x, y = sample()
    fitted = fit_ridge(x, y, alpha_x=0, alpha_y=0, n_components=4)
    legacy = CanCorr(y, x)
    assert_allclose(fitted.penalized_roots, legacy.cancorr, atol=1e-8, rtol=1e-6)
    assert_allclose(fitted.train_correlations, legacy.cancorr, atol=1e-8)
    # Legacy scores have unit sum of squares; ours have unit sample variance.
    legacy_scores = (x - x.mean(0)) @ legacy.x_cancoef
    scores, _ = fitted.transform(x, y)
    assert_allclose(abs(legacy_scores.T @ scores / np.sqrt(len(x) - 1)),
                    np.eye(4), atol=1e-8)


@pytest.mark.parametrize('shape', [(90, 6, 4), (18, 32, 25)])
def test_ridge_matches_full_covariance_whitening_and_constraints(shape):
    x, y = sample(*shape)
    fitted = fit_ridge(x, y, alpha_x=0.3, alpha_y=1.2, n_components=3)
    xc, yc = x - x.mean(0), y - y.mean(0)
    sxx, syy = xc.T @ xc / (len(x) - 1), yc.T @ yc / (len(x) - 1)
    gxx = sxx + 0.3 * np.trace(sxx) / x.shape[1] * np.eye(x.shape[1])
    gyy = syy + 1.2 * np.trace(syy) / y.shape[1] * np.eye(y.shape[1])

    def invsqrt(a):
        values, vectors = np.linalg.eigh(a)
        return (vectors / np.sqrt(values)) @ vectors.T

    cross = xc.T @ yc / (len(x) - 1)
    roots = np.linalg.svd(invsqrt(gxx) @ cross @ invsqrt(gyy), compute_uv=False)
    assert_allclose(fitted.penalized_roots, roots[:3], atol=1e-8, rtol=1e-6)
    assert_allclose(fitted.x_weights.T @ gxx @ fitted.x_weights, np.eye(3), atol=1e-8)
    assert_allclose(fitted.y_weights.T @ gyy @ fitted.y_weights, np.eye(3), atol=1e-8)
    assert_allclose(fitted.x_weights.T @ cross @ fitted.y_weights,
                    np.diag(fitted.penalized_roots), atol=1e-8)
    assert np.max(abs(fitted.penalized_roots - fitted.train_correlations)) > 0.01


def test_scalar_case_has_hand_derived_penalized_root_but_perfect_pearson():
    x = np.array([[-1.], [0.], [1.]])
    fitted = fit_ridge(x, 2 * x, alpha_x=3, alpha_y=8, n_components=1)
    assert_allclose(fitted.penalized_roots, [1 / 6], atol=1e-12)
    assert_allclose(fitted.train_correlations, [1], atol=1e-12)


def test_transform_preserves_negative_test_correlations_and_training_state():
    x = np.array([[-2.], [-1.], [0.], [1.], [2.]])
    fitted = fit_ridge(x, x, n_components=1, scale=True)
    u, v = fitted.transform(x + 100, -x - 300)
    assert_allclose(paired_correlations(u, v), [-1], atol=1e-12)
    assert_allclose(u.mean(0), [100 / x.std(ddof=1) / np.sqrt(2)], atol=1e-12)
    u_again, v_again = fitted.transform(x, x)
    assert_allclose(paired_correlations(u_again, v_again), [1], atol=1e-12)


def test_trace_relative_ridge_invariant_to_global_units_in_each_view():
    x, y = sample()
    a = fit_ridge(x, y, n_components=3)
    b = fit_ridge(x * 1000, y * 0.01, n_components=3)
    assert_allclose(a.penalized_roots, b.penalized_roots, atol=1e-10)
    assert_allclose(a.train_correlations, b.train_correlations, atol=1e-10)


def test_collinear_view_supported_only_with_its_own_positive_penalty():
    x, y = sample()
    x = np.column_stack([x, x[:, 0]])
    fitted = fit_ridge(x, y, alpha_x=1, alpha_y=0, n_components=2)
    assert np.isfinite(fitted.penalized_roots).all()
    with pytest.raises(ValueError, match='rank'):
        fit_ridge(x, y, alpha_x=0, alpha_y=1, n_components=2)


@pytest.mark.parametrize('kwargs', [
    {'alpha_x': -1}, {'alpha_y': np.inf}, {'alpha_x': np.nan},
    {'n_components': 0}, {'n_components': 5}, {'n_components': 1.5},
    {'n_components': True},
])
def test_rejects_invalid_parameters(kwargs):
    x, y = sample()
    with pytest.raises(ValueError):
        fit_ridge(x, y, **kwargs)


@pytest.mark.parametrize('kind', ['mismatched_rows', 'nan', 'constant', 'one_dim', 'empty', 'few_rows'])
def test_rejects_invalid_training_data(kind):
    x, y = sample()
    if kind == 'mismatched_rows':
        y = y[:-1]
    elif kind == 'nan':
        x[0, 0] = np.nan
    elif kind == 'constant':
        x[:, 0] = 2
    elif kind == 'one_dim':
        x = x[:, 0]
    elif kind == 'empty':
        x = x[:, :0]
    else:
        x, y = x[:2], y[:2]
    with pytest.raises(ValueError):
        fit_ridge(x, y)


def test_transform_rejects_wrong_feature_count():
    x, y = sample()
    fitted = fit_ridge(x, y)
    with pytest.raises(ValueError):
        fitted.transform(x[:, :-1], y)


def test_degenerate_evaluation_correlation_is_not_silently_zero():
    with pytest.raises(ValueError):
        paired_correlations(np.ones((5, 1)), np.arange(5.)[:, None])
