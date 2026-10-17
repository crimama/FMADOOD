import numpy as np
import pytest

from fmad.methods.statistical_calibration_estimators import (
    empirical_bayes_scales, covariance_fusion_weights,
)


def test_eb_measurement_uncertainty_shrinks_toward_layer_prior():
    moments = np.array([[1., 1.], [.2, 3.8], [4., 4.]])
    scales = moments.mean(axis=1)
    fitted, audit = empirical_bayes_scales(scales, moments, power=1.)
    assert audit['shrinkage'][0] == 0.
    assert audit['shrinkage'][2] == 0.
    assert audit['shrinkage'][1] > 0.
    assert abs(np.log(fitted[1]) - audit['prior_log_mean']) <= abs(np.log(scales[1]) - audit['prior_log_mean'])
    np.testing.assert_allclose(fitted[[0, 2]], scales[[0, 2]])


def test_eb_is_equivariant_to_distance_units_and_permutations():
    moments = np.array([[1., 2., 3.], [3., 5., 4.], [.2, 4., 2.]])
    scales = np.sqrt(moments.mean(axis=1))
    fitted, _ = empirical_bayes_scales(scales, moments, power=.5)
    rescaled, _ = empirical_bayes_scales(scales * 3, moments * 9, power=.5)
    np.testing.assert_allclose(rescaled, fitted * 3)
    permuted, _ = empirical_bayes_scales(scales[::-1], moments[::-1, ::-1], power=.5)
    np.testing.assert_allclose(permuted, fitted[::-1])
    with pytest.raises(ValueError):
        empirical_bayes_scales([1., 2.], [[1.], [2.]], power=1.)


def test_covariance_weights_simplex_and_permutation_equivariance():
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(16, 4)) * [1., 3., 2., .5]
    weights, audit = covariance_fusion_weights(scores)
    assert np.min(weights) >= 0.
    assert weights.sum() == pytest.approx(1.)
    assert 0 <= audit['shrinkage'] <= 1.
    permuted, _ = covariance_fusion_weights(scores[::-1, ::-1])
    np.testing.assert_allclose(permuted, weights[::-1], atol=1e-7)
    uniform, metadata = covariance_fusion_weights(np.ones((2, 16)))
    np.testing.assert_allclose(uniform, np.full(16, 1/16))
    assert metadata['fallback']
