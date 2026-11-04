"""Numerical and leakage checks for the benchmark's package adapters."""

import numpy as np
from numpy.testing import assert_allclose
from statsmodels.multivariate.cancorr import CanCorr, select_cancorr_penalty

from benchmarks.methods import fit_python, tune_python


def test_zoo_parameter_conversion_matches_ridge_on_raw_scales():
    rng = np.random.default_rng(532)
    x = rng.normal(size=(24, 35)) * np.linspace(1, 30, 35)
    y = rng.normal(size=(24, 15)) + x[:, :2] @ rng.normal(size=(2, 15))
    ref = CanCorr(y, x, penalty=(.1, 2), standardize=True)
    fit = fit_python('cca_zoo', x, y, 2, (.1, 2))
    for a, b in [(ref.x_cancoef[:, :2], fit.x_coef),
                 (ref.y_cancoef[:, :2], fit.y_coef)]:
        qa, qb = np.linalg.qr(a)[0], np.linalg.qr(b)[0]
        assert_allclose(np.linalg.svd(qa.T @ qb, compute_uv=False), 1, atol=1e-10)
    u, v = fit.transform(x, y)
    assert_allclose([np.corrcoef(u[:, i], v[:, i])[0, 1] for i in range(2)],
                    ref.cancorr[:2], atol=1e-10)


def test_generic_tuning_matches_train_only_cancorr_helper():
    rng = np.random.default_rng(78)
    x, y = rng.normal(size=(50, 8)), rng.normal(size=(50, 6))
    x[:10] *= 40
    y[:, :2] += x[:, :2]
    grid = [(.01, .1), (.1, 1), (1, 10)]
    ref = select_cancorr_penalty(y, x, grid, cv=5, standardize=True, rng=51)
    result = tune_python('cca_zoo', x, y, 2, grid, 5, 51)
    assert result['penalty'] == ref.penalty
    assert_allclose(result['cv_scores'], ref.cv_scores, atol=1e-10)


def test_correctness_checks_leading_directions_not_only_full_column_space():
    from benchmarks.methods import from_cancorr
    from benchmarks.run_experiments import correctness

    rng = np.random.default_rng(854)
    x = rng.normal(size=(80, 3))
    y = x * [.8, .5, .2] + rng.normal(size=(80, 3))
    model = CanCorr(y, x, penalty=(.1, .3), standardize=True)
    ours = from_cancorr(model, 3)
    # Swapping directions leaves the complete 3D column space unchanged.
    ref = {'x_mean': ours.x_mean, 'y_mean': ours.y_mean,
           'x_coef': ours.x_coef[:, [1, 0, 2]],
           'y_coef': ours.y_coef[:, [1, 0, 2]], 'roots': ours.roots}
    check = correctness(ours, ref, x, y)
    assert check['x_max_sin_angle'] < 1e-12
    assert check['x_leading_max_sin_angle'] > .1
    assert not check['passed']


def test_correctness_handles_additional_zero_roots_from_rank_deficient_r_fit():
    from benchmarks.methods import from_cancorr
    from benchmarks.run_experiments import correctness

    rng = np.random.default_rng(829)
    x = rng.normal(size=(50, 3))
    x = np.column_stack([x, x[:, 0] + x[:, 1]])
    y = rng.normal(size=(50, 5))
    fit = CanCorr(y, x, penalty=(.2, .4), standardize=True)
    ours = from_cancorr(fit, len(fit.cancorr))
    ref = {'x_mean': ours.x_mean, 'y_mean': ours.y_mean,
           'x_coef': np.column_stack([ours.x_coef, np.zeros(4)]),
           'y_coef': np.column_stack([ours.y_coef, np.zeros(5)]),
           'roots': np.r_[ours.roots, 0.]}
    check = correctness(ours, ref, x, y)
    assert check['passed']
    assert check['reference_additional_zero_roots'] == 1
    ref['roots'][-1] = .1
    assert not correctness(ours, ref, x, y)['passed']
