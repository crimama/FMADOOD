"""Normal-support estimators used by the declared calibration development suite.

Images, rather than correlated tokens, define the replication units. These are
adaptations of statistical estimators, not inferential coverage guarantees.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize
from sklearn.covariance import LedoitWolf


def empirical_bayes_scales(scales, source_moments, *, power):
    """Approximate normal-means EB on log scales, pooled within one layer.

    Delta-method measurement variance uses between-image moment variation.
    The nonnegative between-observation variance is a method-of-moments fit;
    observations share images, so this is explicitly an approximate model.
    """
    scales = np.asarray(scales, dtype=np.float64)
    moments = np.asarray(source_moments, dtype=np.float64)
    if (moments.ndim != 2 or moments.shape[0] != len(scales)
            or moments.shape[1] < 2 or not np.isfinite(moments).all()
            or not np.isfinite(scales).all() or (scales <= 0).any()
            or (moments < 0).any()):
        raise ValueError("EB calibration requires >=2 aligned normal images")
    theta = np.log(scales)
    variance = power**2 * moments.var(axis=1, ddof=1) / (
        moments.shape[1] * np.maximum(moments.mean(axis=1), 1e-8)**2)
    prior_mean = float(theta.mean())
    prior_variance = max(float(theta.var(ddof=1) - variance.mean()), 0.) if len(theta) > 1 else 0.
    denominator = prior_variance + variance
    shrinkage = np.divide(variance, denominator, out=np.zeros_like(variance),
                          where=denominator > 0.)
    fitted = np.maximum(np.exp((1. - shrinkage) * theta + shrinkage * prior_mean), 1e-4)
    return fitted, dict(prior_log_mean=prior_mean, prior_log_variance=prior_variance,
        measurement_log_variance=variance.tolist(), shrinkage=shrinkage.tolist(),
        source_count=moments.shape[1], replication_unit="support_image",
        model="approximate_log_scale_normal_means_method_of_moments")


def covariance_fusion_weights(source_scores):
    """Nonnegative minimum-variance weights using Ledoit-Wolf covariance.

    Columns are resolution/bag observations of the SAME layer, rows are aligned
    normal support images. This changes fusion weights, not marginal calibration.
    """
    x = np.asarray(source_scores, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("covariance fusion requires aligned finite image scores")
    estimator = LedoitWolf().fit(x)
    cov = estimator.covariance_
    n = x.shape[1]
    scale = float(np.trace(cov) / n)
    if scale <= 1e-15:
        return np.full(n, 1. / n), dict(shrinkage=float(estimator.shrinkage_),
            fallback="constant_image_scores_equal_weights", source_count=x.shape[0])
    normalized = cov / scale
    result = optimize.minimize(lambda w: float(w @ normalized @ w),
        np.full(n, 1. / n), jac=lambda w: 2 * normalized @ w,
        bounds=[(0., 1.)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.,
                     "jac": lambda w: np.ones(n)}, method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 1000})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("covariance fusion optimization failed: " + result.message)
    weights = np.maximum(result.x, 0.)
    weights /= weights.sum()
    return weights, dict(shrinkage=float(estimator.shrinkage_), fallback=None,
        source_count=x.shape[0], replication_unit="support_image",
        covariance=cov.tolist(), optimizer="nonnegative_simplex_minimum_variance")
