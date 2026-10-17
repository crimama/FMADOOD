"""Support-only normalization sweep on the unchanged reference-matching pipeline.

All fitted inputs are normal source-LOO distances. Empirical survival is an
exploratory rank mapping, NOT a claim of conformal coverage for dependent tokens.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import optimize, special, stats

from fmad.methods.reference_matching_fast import ReferenceMatchingFast
from fmad.registry import METHOD_REGISTRY
from fmad.methods.statistical_calibration_estimators import (
    empirical_bayes_scales, covariance_fusion_weights,
)


NORMALIZERS = (
    "none", "q95", "mad", "tail", "upper_width", "shrink_q95", "qn",
    "log_mad", "boxcox_mad", "gamma_survival", "lognormal_survival",
    "gpd_survival", "empirical_survival", "gaussian_survival",
    "quantile_anchor", "quantile_barycenter",
    "axis_layer", "axis_resolution", "axis_layer_resolution",
    "multi_quantile_scale", "image_median_quantile_scale",
    "mean_scale", "rms_scale", "std_scale", "zscore",
    "eb_mean_scale", "eb_rms_scale", "weibull_bank_q95", "covariance_q95",
)
MOMENT_NORMALIZERS = {"mean_scale", "rms_scale", "std_scale", "zscore"}
GROUP_ESTIMATORS = {"eb_mean_scale", "eb_rms_scale", "covariance_q95"}
AFFINE_ESTIMATORS = GROUP_ESTIMATORS | {"weibull_bank_q95"}
AXIS_NORMALIZERS = {"axis_layer", "axis_resolution", "axis_layer_resolution"}
QUANTILE_SCALE_NORMALIZERS = {"multi_quantile_scale", "image_median_quantile_scale"}
SCALE_QUANTILES = (.5, .75, .9, .95)


def geometric_axis_factors(q95: np.ndarray) -> dict:
    """Equal observation weighting, axes [resolution, bag, layer]."""
    u = np.asarray(q95, dtype=np.float64)
    if u.ndim != 3 or not all(u.shape):
        raise ValueError("expected nonempty resolution/bag/layer Q95 array")
    fallback = None
    if not np.isfinite(u).all():
        layer, resolution = np.ones(u.shape[2]), np.ones(u.shape[0])
        fallback = "nonfinite_reference_using_identity"
        mu = None
    else:
        logs = np.log(np.maximum(u, 1e-4))
        mu = float(logs.mean())
        layer = np.exp(logs.mean(axis=(0, 1)) - mu)
        resolution = np.exp(logs.mean(axis=(1, 2)) - mu)
    return dict(layer=layer.tolist(), resolution=resolution.tolist(),
                log_global_q95=mu, observations=int(u.size),
                weighting="equal_observation", q95_floor=1e-4, fallback=fallback)


def quantile_scale(quantiles: np.ndarray, target: np.ndarray) -> float:
    """Positive least-squares scale matching fixed support quantiles."""
    q = np.maximum(np.asarray(quantiles, dtype=np.float64), 1e-4)
    t = np.asarray(target, dtype=np.float64)
    if q.shape != t.shape or q.ndim != 1 or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError("invalid calibration quantiles")
    if (t <= 0).any():
        raise ValueError("target quantiles must be positive")
    return max(float(np.dot(q, q) / np.dot(q, t)), 1e-4)

TRANSPORT_GRIDS = {
    "quantile_anchor": (.5, .95, .99),
    "quantile_barycenter": (.01, .05, .10, .25, .50, .75, .90, .95, .99),
}


def _qn(sample: np.ndarray, cap: int = 1024) -> tuple[float, int]:
    # Sorted quantile thinning is deterministic and consumes no bank RNG state.
    x = np.sort(sample)
    if len(x) > cap:
        x = x[np.linspace(0, len(x) - 1, cap).round().astype(int)]
    n = len(x)
    if n < 2:
        return 1e-4, n
    i, j = np.triu_indices(n, 1)
    differences = np.abs(x[i] - x[j])
    h = n // 2 + 1
    k = h * (h - 1) // 2
    # Asymptotic normal consistency constant; no finite-n correction is claimed.
    return max(2.219144465985076 * np.partition(differences, k - 1)[k - 1], 1e-4), n


def _power(x: np.ndarray, lam: float) -> np.ndarray:
    logx = np.log(x)
    return logx if abs(lam) < 1e-7 else np.expm1(lam * logx) / lam


def fit_normalizer(sample: torch.Tensor, name: str, tail: dict) -> dict:
    """Fit one component. Numerical fallbacks are recorded, never label-selected."""
    x = sample.detach().double().cpu().numpy()
    if name not in NORMALIZERS or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("invalid normalizer or reference distances")
    q50, q80, q95 = np.quantile(x, [.5, .8, .95])
    p = dict(kind=name, n_reference=len(x), center=0., scale=1., fallback=None)
    if name in MOMENT_NORMALIZERS | {"eb_mean_scale", "eb_rms_scale"}:
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=0))
        rms = float(np.sqrt(np.mean(np.square(x))))
        raw_scale = {"mean_scale": mean, "rms_scale": rms,
                     "eb_mean_scale": mean, "eb_rms_scale": rms,
                     "std_scale": std, "zscore": std}[name]
        p.update(center=mean if name == "zscore" else 0.,
                 scale=max(raw_scale, 1e-4), reference_mean=mean,
                 reference_std=std, reference_rms=rms, raw_scale=raw_scale,
                 scale_floor=1e-4, variance_ddof=0,
                 fallback="moment_scale_floor_active" if raw_scale < 1e-4 else None)
    elif name in TRANSPORT_GRIDS:
        p["scale"] = max(float(tail["u"]), 1e-4)
    elif name == "gaussian_survival":
        p["tail_mapping"] = "empirical_body_exponential_excess"
    elif name in {"q95", "shrink_q95", "covariance_q95", "weibull_bank_q95"} | QUANTILE_SCALE_NORMALIZERS:
        # Retain the parent's float32 quantile for exact q95 control parity.
        p["scale"] = max(float(tail["u"]), 1e-4)
        if name == "weibull_bank_q95":
            positive = np.maximum(x, 1e-8)
            if np.ptp(positive) < 1e-8:
                p.update(weibull_shape=None, fallback="constant_reference_using_q95")
            else:
                try:
                    shape, _, fitted_scale = stats.weibull_min.fit(positive, floc=0.)
                    if not np.isfinite([shape, fitted_scale]).all() or min(shape, fitted_scale) <= 0:
                        raise ValueError("invalid Weibull fit")
                    p.update(weibull_shape=float(shape), weibull_fit_scale=float(fitted_scale))
                except (ValueError, RuntimeError, FloatingPointError):
                    p.update(weibull_shape=None, fallback="weibull_fit_failed_using_q95")
    elif name == "mad":
        # Legacy MAD uses torch.median (lower middle for even n), without centering.
        median = sample.median()
        p["scale"] = max(float(1.4826 * (sample - median).abs().median()), 1e-4)
    elif name == "upper_width":
        p.update(center=float(q80), scale=max(float(q95 - q80), .05 * q95, 1e-4),
                 width_floor_fraction_q95=.05)
    elif name == "qn":
        scale, n = _qn(x)
        p.update(center=float(q50), scale=float(scale), qn_sample_count=n,
                 qn_quantile_thinning_cap=1024, qn_finite_sample_correction=False)
    elif name in {"log_mad", "boxcox_mad"}:
        unit = max(float(q50), 1e-6)
        positive = np.maximum(x / unit, 1e-6)
        lam = 0.
        if name == "boxcox_mad" and np.ptp(positive) > 1e-8:
            def objective(value):
                transformed = _power(positive, value)
                variance = max(float(np.var(transformed)), 1e-16)
                return .5 * len(x) * np.log(variance) - (value - 1) * np.log(positive).sum()
            result = optimize.minimize_scalar(objective, bounds=(-1., 1.), method="bounded")
            if result.success and np.isfinite(result.fun):
                lam = float(result.x)
            else:
                p["fallback"] = "boxcox_fit_failed_using_log"
        transformed = _power(positive, lam)
        center = float(np.median(transformed))
        p.update(unit=unit, power_lambda=lam, center=center,
                 scale=max(1.4826 * float(np.median(np.abs(transformed - center))), 1e-4),
                 positive_floor=1e-6)
    elif name in {"gamma_survival", "lognormal_survival"}:
        positive = np.maximum(x, 1e-8)
        if np.ptp(positive) < 1e-8:
            p.update(kind="q95", scale=max(float(tail["u"]), 1e-4),
                     fallback="constant_reference_using_q95")
        elif name == "gamma_survival":
            try:
                shape, _, scale = stats.gamma.fit(positive, floc=0.)
                if not np.isfinite([shape, scale]).all() or min(shape, scale) <= 0:
                    raise ValueError("invalid gamma fit")
                p.update(gamma_shape=float(shape), gamma_scale=float(scale))
            except (ValueError, RuntimeError, FloatingPointError):
                p.update(kind="q95", scale=max(float(tail["u"]), 1e-4),
                         fallback="gamma_fit_failed_using_q95")
        else:
            logx = np.log(positive)
            p.update(log_mu=float(logx.mean()), log_sigma=max(float(logx.std()), 1e-4))
    elif name == "gpd_survival":
        excess = x[x > float(tail["u"])] - float(tail["u"])
        beta = max(float(tail["beta"]), 1e-6)
        shape = 0.
        if len(excess) >= 8 and np.ptp(excess) > 1e-8:
            def nll(v):
                xi, logbeta = v
                scaled = excess / np.exp(logbeta)
                term = scaled if xi < 1e-7 else (1 + 1 / xi) * np.log1p(xi * scaled)
                return float(logbeta + term.mean())
            result = optimize.minimize(nll, [.1, math.log(beta)], method="L-BFGS-B",
                bounds=[(0., .5), (math.log(1e-6), math.log(2.))])
            if result.success and np.isfinite(result.fun):
                shape, logbeta = result.x
                beta = float(np.exp(logbeta))
            else:
                p["fallback"] = "gpd_fit_failed_using_exponential_tail"
        else:
            p["fallback"] = "insufficient_tail_using_exponential_tail"
        p.update(gpd_shape=float(shape), gpd_beta=beta, tail_count=len(excess),
                 gpd_shape_bounds=[0., .5])
    return p


def transform_normalizer(distance: torch.Tensor, tail: dict, p: dict) -> torch.Tensor:
    kind = p["kind"]
    if kind in TRANSPORT_GRIDS:
        x = distance / p["scale"]
        if "source_knots" not in p:
            raise RuntimeError("quantile transport requires fitted cross-observation knots")
        source = torch.as_tensor(p["source_knots"], dtype=x.dtype, device=x.device)
        target = torch.as_tensor(p["target_knots"], dtype=x.dtype, device=x.device)
        index = torch.searchsorted(source, x.contiguous(), right=True).clamp(1, len(source)-1)
        lo, hi = source[index-1], source[index]
        slope = (target[index]-target[index-1]) / (hi-lo)
        slope = torch.where(x > source[-1], slope.clamp(max=10.), slope)
        return target[index-1] + slope * (x-lo)
    if kind in {"none", "q95", "shrink_q95", "mad", "upper_width", "qn"} | AXIS_NORMALIZERS | QUANTILE_SCALE_NORMALIZERS | MOMENT_NORMALIZERS | AFFINE_ESTIMATORS:
        return ((distance - p["center"]) / p["scale"]) * p.get("fusion_multiplier", 1.)
    if kind == "gaussian_survival":
        sample = tail["sample"].to(distance.device)
        n = len(sample)
        rank = torch.searchsorted(sample, distance.contiguous(), right=True)
        survival = ((n - rank).to(distance.dtype) / n).clamp_min(float(tail["p_u"]))
        log_survival = -survival.log()
        u = float(tail["u"])
        beta = max(float(tail["beta"]), 1e-6)
        tail_log_survival = -math.log(float(tail["p_u"])) + (distance - u) / beta
        log_survival = torch.where(distance > u, tail_log_survival, log_survival)
        probability = torch.exp(-log_survival.double()).clamp(1e-12, 1. - 1e-7)
        return torch.special.ndtri(1. - probability).to(distance.dtype)
    if kind in {"log_mad", "boxcox_mad"}:
        logx = (distance / p["unit"]).clamp_min(p["positive_floor"]).log()
        lam = p["power_lambda"]
        transformed = logx if abs(lam) < 1e-7 else torch.expm1(lam * logx) / lam
        return (transformed - p["center"]) / p["scale"]
    if kind == "gamma_survival":
        # Use SciPy's log survival in float64; this is a declared CPU scoring cost.
        x = distance.detach().double().cpu().numpy().clip(min=0.)
        shape, scale = p["gamma_shape"], p["gamma_scale"]
        result = np.asarray(-stats.gamma.logsf(x, shape, scale=scale))
        bad = ~np.isfinite(result)
        if bad.any():
            # Upper incomplete gamma asymptotic only where logsf underflows.
            t = np.maximum(x[bad] / scale, 1e-12)
            correction = np.maximum(1 + (shape - 1) / t
                + (shape - 1) * (shape - 2) / t**2, 1e-12)
            result[bad] = t - (shape - 1) * np.log(t) + special.gammaln(shape) - np.log(correction)
        return torch.as_tensor(result, dtype=distance.dtype, device=distance.device)
    if kind == "lognormal_survival":
        z = (distance.clamp_min(1e-8).log() - p["log_mu"]) / p["log_sigma"]
        return -torch.special.log_ndtr(-z)
    sample = tail["sample"].to(distance.device)
    n = len(sample)
    if kind == "empirical_survival":
        # Inclusive ties and add-one smoothing; scores saturate above max support.
        below = torch.searchsorted(sample, distance.contiguous(), right=False)
        return -torch.log((n - below + 1).to(distance.dtype) / (n + 1))
    rank = torch.searchsorted(sample, distance.contiguous(), right=True)
    body = -(1 - rank.to(distance.dtype) / n).clamp_min(1 / (n + 1)).log()
    u = float(tail["u"])
    beta = p["gpd_beta"] if kind == "gpd_survival" else float(tail["beta"])
    shape = p.get("gpd_shape", 0.)
    excess = (distance - u).clamp_min(0.) / beta
    increment = excess if shape < 1e-7 else torch.log1p(shape * excess) / shape
    extrapolated = -math.log(float(tail["p_u"])) + increment
    return torch.where(distance > u, extrapolated, body)


@METHOD_REGISTRY.register("statistical_reference_matching")
class StatisticalReferenceMatching(ReferenceMatchingFast):
    def __init__(self, backbone, config: dict):
        self.statistical_normalization = str(config.get("statistical_normalization", "q95"))
        if self.statistical_normalization not in NORMALIZERS:
            raise ValueError("unknown statistical_normalization")
        self.axis_alpha_layer = float(config.get("axis_alpha_layer",
            0. if self.statistical_normalization == "axis_resolution" else 1.))
        self.axis_alpha_resolution = float(config.get("axis_alpha_resolution",
            0. if self.statistical_normalization == "axis_layer" else 1.))
        if any(not math.isfinite(x) or not 0. <= x <= 1.
               for x in (self.axis_alpha_layer, self.axis_alpha_resolution)):
            raise ValueError("axis exponents must be finite and within [0, 1]")
        if (self.statistical_normalization not in AXIS_NORMALIZERS
                and any(k in config for k in ("axis_alpha_layer", "axis_alpha_resolution"))):
            raise ValueError("axis exponents require an axis normalizer")
        # The existing tail branch dispatches transforms while retaining identical
        # distances, banks, fusion, routing, and raw/normalized audit hooks.
        if (config.get("variant", "R1") != "R1"
                or config.get("ablation_shallow_mode") != "all"
                or config.get("local_reliability", "off") != "off"
                or config.get("tailscale_center_rho", 0.) != 0.
                or config.get("score_normalization", "tail") != "tail"
                or config.get("ablation_feature_matching", "layerwise") != "layerwise"
                or config.get("scale_fusion", "mean") != "mean"
                or config.get("layer_fusion") != "residual_spike"
                or config.get("prediction_postprocess", "none") != "none"):
            raise ValueError("statistical sweep requires R1 linear shallow fusion, tail dispatch, no local/postprocessing")
        audit = config.get("calibration_audit", False)
        if type(audit) is not bool:
            raise ValueError("calibration_audit must be boolean")
        parent = {**config, "score_normalization": "tail", "calibration_audit": False}
        super().__init__(backbone, parent)
        self.calibration_audit = audit

    def _fit_scales_for_resolution(self, resolution):
        super()._fit_scales_for_resolution(resolution)
        scales = []
        for layer_index, tail in enumerate(self._last_tails):
            p = fit_normalizer(tail["sample"], self.statistical_normalization, tail)
            if self.statistical_normalization in GROUP_ESTIMATORS:
                groups = getattr(self, "_last_source_loo_samples", None)
                if groups is None or len(groups[layer_index]) != len(self.records[resolution]):
                    raise RuntimeError("statistical estimator requires >=2 source-LOO groups")
                squared = self.statistical_normalization == "eb_rms_scale"
                p["source_moments"] = [float(g.double().square().mean() if squared
                                             else g.double().mean()) for g in groups[layer_index]]
                p["source_ids"] = [int(record["source"]) for record in self.records[resolution]]
            if self.statistical_normalization == "weibull_bank_q95":
                n = len(self.records[resolution])
                if n < 2:
                    raise ValueError("bank-size correction requires >=2 source images")
                ratio = (n - 1.) / n
                shape = p["weibull_shape"]
                correction = ratio ** (1. / shape) if shape is not None else 1.
                p.update(unadjusted_q95=p["scale"], loo_to_full_bank_ratio=ratio,
                         bank_scale_correction=correction, source_count=n)
                p["scale"] = max(p["scale"] * correction, 1e-4)
            if self.statistical_normalization in QUANTILE_SCALE_NORMALIZERS:
                p["support_quantiles"] = np.quantile(
                    tail["sample"].double().cpu().numpy(), SCALE_QUANTILES).tolist()
                if self.statistical_normalization == "image_median_quantile_scale":
                    groups = getattr(self, "_last_source_loo_samples", None)
                    if groups is None or len(groups[layer_index]) != len(self.records[resolution]):
                        raise RuntimeError("per-source calibration groups unavailable")
                    p["source_quantiles"] = [np.quantile(
                        group.double().cpu().numpy(), SCALE_QUANTILES).tolist()
                        for group in groups[layer_index]]
            tail["statistical"] = p
            scales.append(p["scale"])
        return torch.tensor(scales)

    def fit(self, support_paths, context):
        super().fit(support_paths, context)
        if self.statistical_normalization in GROUP_ESTIMATORS:
            self.group_estimator_audit = {}
            for li in range(len(self.layers)):
                observations = [(r, bi, bag[li]) for r, bags in self.bag_tails.items()
                                for bi, bag in enumerate(bags)]
                params = [t["statistical"] for _, _, t in observations]
                if any(p["source_ids"] != params[0]["source_ids"] for p in params):
                    raise RuntimeError("source identity mismatch across observations")
                scales = np.asarray([p["scale"] for p in params])
                moments = np.asarray([p["source_moments"] for p in params])
                if self.statistical_normalization == "covariance_q95":
                    weights, audit = covariance_fusion_weights((moments / scales[:, None]).T)
                    for p, w in zip(params, weights):
                        p.update(fusion_multiplier=float(len(params) * w), fusion_weight=float(w))
                    audit["changes_fusion_weights"] = True
                    audit["weights"] = weights.tolist()
                else:
                    fitted, audit = empirical_bayes_scales(scales, moments,
                        power=.5 if self.statistical_normalization == "eb_rms_scale" else 1.)
                    for i, (p, scale) in enumerate(zip(params, fitted)):
                        p.update(unshrunk_scale=p["scale"], scale=float(scale),
                            shrinkage=float(audit["shrinkage"][i]),
                            measurement_log_variance=float(audit["measurement_log_variance"][i]))
                audit["pooling"] = "within_layer_across_resolution_and_bag"
                self.group_estimator_audit[str(self.layers[li])] = audit
                for (r, bi, _), p in zip(observations, params):
                    self.bag_scales[r][bi][li] = p["scale"]
        if self.statistical_normalization in QUANTILE_SCALE_NORMALIZERS:
            observations = [(r, bi, li, t)
                            for r, bags in self.bag_tails.items()
                            for bi, bag in enumerate(bags)
                            for li, t in enumerate(bag)]
            quantiles = np.asarray([t["statistical"]["support_quantiles"]
                                    for _, _, _, t in observations], dtype=np.float64)
            ratios = quantiles / np.maximum(quantiles[:, -1:], 1e-4)
            target = np.exp(np.mean(np.log(np.maximum(ratios, 1e-4)), axis=0))
            target[-1] = 1.
            self.quantile_scale_target = target.tolist()
            for r, bi, li, tail in observations:
                p = tail["statistical"]
                if self.statistical_normalization == "multi_quantile_scale":
                    scale = quantile_scale(p["support_quantiles"], target)
                else:
                    source_scales = [quantile_scale(q, target) for q in p["source_quantiles"]]
                    scale = float(np.exp(np.median(np.log(source_scales))))
                    p["source_scales"] = source_scales
                p.update(scale=scale, target_quantiles=target.tolist(),
                         quantile_grid=list(SCALE_QUANTILES))
                self.bag_scales[r][bi][li] = scale
        if self.statistical_normalization in AXIS_NORMALIZERS:
            resolutions = list(self.bag_tails)
            q95 = np.asarray([[[float(t["u"]) for t in bag]
                               for bag in self.bag_tails[r]] for r in resolutions])
            factors = geometric_axis_factors(q95)
            self.axis_factors = dict(factors, resolutions=resolutions, layers=list(self.layers),
                alpha_layer=self.axis_alpha_layer, alpha_resolution=self.axis_alpha_resolution)
            for ri, r in enumerate(resolutions):
                for bi, bag in enumerate(self.bag_tails[r]):
                    for li, t in enumerate(bag):
                        a = factors["layer"][li] ** self.axis_alpha_layer
                        c = factors["resolution"][ri] ** self.axis_alpha_resolution
                        p = t["statistical"]
                        p.update(scale=a*c, source_loo_q95=max(float(t["u"]), 1e-4),
                                 layer_factor=a, resolution_factor=c,
                                 alpha_layer=self.axis_alpha_layer,
                                 alpha_resolution=self.axis_alpha_resolution,
                                 fallback=factors["fallback"])
                        self.bag_scales[r][bi][li] = p["scale"]
        if self.statistical_normalization == "shrink_q95":
            # Fixed alpha=.5, geometric pooling within a layer/resolution only.
            for resolution, bags in self.bag_tails.items():
                for layer in range(len(self.layers)):
                    group = math.exp(np.mean([math.log(b[layer]["statistical"]["scale"]) for b in bags]))
                    for bag_index, bag in enumerate(bags):
                        p = bag[layer]["statistical"]
                        p.update(unshrunk_scale=p["scale"], group_scale=group, shrinkage_alpha=.5)
                        p["scale"] = math.sqrt(p["scale"] * group)
                        self.bag_scales[resolution][bag_index][layer] = p["scale"]
        if self.statistical_normalization in TRANSPORT_GRIDS:
            grid = TRANSPORT_GRIDS[self.statistical_normalization]
            observations = [tail for bags in self.bag_tails.values() for bag in bags for tail in bag]
            quantiles = []
            for tail in observations:
                p = tail["statistical"]
                sample = tail["sample"].double().cpu().numpy() / p["scale"]
                quantiles.append(np.quantile(sample, grid))
            target = np.mean(quantiles, axis=0)
            for tail, source in zip(observations, quantiles):
                p = tail["statistical"]
                # Degenerate few-shot references cannot identify a transport.
                if np.min(np.diff(source)) <= 1e-6:
                    p["fallback"] = "collapsed_quantile_knots_using_q95"
                    p["source_knots"] = [0., 1.]
                    p["target_knots"] = [0., 1.]
                else:
                    p["source_knots"] = source.tolist()
                    p["target_knots"] = target.tolist()
                p["quantile_grid"] = list(grid)
                p["barycenter_observations"] = len(observations)
                p["upper_slope_cap"] = 10.

    def _tail_transform(self, params, distance):
        return transform_normalizer(distance, params, params["statistical"])

    def object_metadata(self):
        metadata = super().object_metadata()
        fitted = {}
        for resolution, bags in self.bag_tails.items():
            fitted[str(resolution)] = [[dict(t["statistical"],
                transformed_q50=float(self._tail_transform(t, torch.quantile(t["sample"], .5))),
                transformed_q95=float(self._tail_transform(t, t["u"]))) for t in bag] for bag in bags]
        metadata.update(score_normalization=self.statistical_normalization,
            normalization="normal_support_source_LOO_" + self.statistical_normalization,
            score_definition="fixed_reference_matching_with_" + self.statistical_normalization,
            statistical_normalizer_parameters=fitted,
            statistical_normalizer_dispatch="tail",
            statistical_normalizer_fallback_count=sum(bool(t["statistical"]["fallback"])
                for bags in self.bag_tails.values() for bag in bags for t in bag),
            conformal_coverage_claimed=False)
        metadata.update(effective_layer_fusion="deep_mean_plus_gamma_shallow",
                        effective_layer_fusion_gamma=self.layer_fusion_gamma,
                        layer_fusion_tau_used=False)
        if self.statistical_normalization in AXIS_NORMALIZERS:
            metadata["geometric_axis_factors"] = self.axis_factors
        if self.statistical_normalization in GROUP_ESTIMATORS:
            metadata["group_estimator_audit"] = self.group_estimator_audit
        if self.statistical_normalization in QUANTILE_SCALE_NORMALIZERS:
            metadata["quantile_scale_target"] = self.quantile_scale_target
        # Parent audit assumes affine division. Replace it with actual transformed
        # reference quantiles, so nonlinear transforms are not mislabeled as MAD.
        for resolution, audit in metadata.get("ablation_calibration", {}).items():
            for b, bag in enumerate(audit["support_loo"]):
                for layer, item in enumerate(bag):
                    p = fitted[resolution][b][layer]
                    item.update(normalized_q50=p["transformed_q50"],
                        normalized_q95=p["transformed_q95"], offset=p["center"])
        return metadata
