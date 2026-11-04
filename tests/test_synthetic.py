import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from benchmarks.synthetic import make_cca_data


def test_population_canonical_correlations_are_the_requested_values():
    data = make_cca_data(30, 7, 5, [0.9, 0.4], rng=41)
    # Both marginal population covariances are the identity by construction.
    singular_values = np.linalg.svd(data.cov_xy, compute_uv=False)
    assert_allclose(singular_values, [0.9, 0.4, 0, 0, 0], atol=1e-12)
    assert_allclose(data.x_weights.T @ data.cov_xy @ data.y_weights,
                    np.diag([0.9, 0.4]), atol=1e-12)
    assert_allclose(data.x_weights.T @ data.x_weights, np.eye(2), atol=1e-12)
    assert_allclose(data.y_weights.T @ data.y_weights, np.eye(2), atol=1e-12)


def test_large_sample_scores_recover_planted_correlations():
    data = make_cca_data(40000, 4, 3, [0.8, 0.3], rng=993)
    u, v = data.x @ data.x_weights, data.y @ data.y_weights
    actual = [np.corrcoef(u[:, i], v[:, i])[0, 1] for i in range(2)]
    assert_allclose(actual, [0.8, 0.3], atol=0.015)


def test_global_null_has_zero_population_cross_covariance():
    data = make_cca_data(20, 6, 4, [], rng=1)
    assert_array_equal(data.cov_xy, np.zeros((6, 4)))
    assert data.x_weights.shape == (6, 0)


def test_reproducibility_and_no_global_rng_mutation():
    state = np.random.get_state()
    a = make_cca_data(20, 6, 4, [0.7], rng=1)
    b = make_cca_data(20, 6, 4, [0.7], rng=1)
    assert_array_equal(a.x, b.x)
    assert_array_equal(a.y, b.y)
    assert_array_equal(np.random.get_state()[1], state[1])


@pytest.mark.parametrize('n,p,q,rhos', [
    (1, 4, 3, []), (10, 0, 3, []), (10, 4, 3, [-0.1]),
    (10, 4, 3, [1.1]), (10, 4, 3, [np.nan]),
    (10, 2, 3, [0.8, 0.6, 0.4]), (10, 4, 3, [0.3, 0.8]),
])
def test_invalid_population_configuration(n, p, q, rhos):
    with pytest.raises(ValueError):
        make_cca_data(n, p, q, rhos, rng=1)
