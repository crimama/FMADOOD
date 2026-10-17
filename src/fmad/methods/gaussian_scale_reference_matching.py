"""Normal-support joint score calibration with a positive subset Gaussian GLRT.

Only the four-resolution score fusion changes. Frozen features, random banks,
bag averaging and Deep + 0.125 * Shallow remain those of the comparator.
The Gaussian model is an approximation; spatial probes and shared reference
banks are dependent. No independent-sample coverage or optimality is claimed.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from fmad.methods.multiscale_aggregation_reference_matching import (
    MultiscaleAggregationReferenceMatching, combine_resolution_maps,
)
from fmad.methods.reference_matching_fast import _masked_loo_distances
from fmad.registry import METHOD_REGISTRY


def bilinear_probe_indices(grid, side):
    """Indices/weights matching interpolate(score, (side, side), align_corners=False)."""
    h, w = grid
    y = np.clip((np.arange(side) + .5) * h / side - .5, 0, h - 1)
    x = np.clip((np.arange(side) + .5) * w / side - .5, 0, w - 1)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = yy - y0, xx - x0
    indices = np.stack([y0*w+x0, y0*w+x1, y1*w+x0, y1*w+x1], -1)
    weights = np.stack([(1-fy)*(1-fx), (1-fy)*fx, fy*(1-fx), fy*fx], -1)
    return indices.reshape(-1, 4), weights.reshape(-1, 4)


def _moments(groups):
    x = groups.reshape(-1, groups.shape[-1])
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-6)
    z = (x - mean) / std
    covariance = z.T @ z / len(z)
    # A constant channel has no empirical correlation, and receives unit ridge.
    return mean, std, covariance


def fit_gaussian_scale_model(groups, *, diagonal=False):
    """Fit equal-source moments and shrinkage by source-block normal likelihood.

    Inputs [source, aligned spatial probe, resolution] are normal support LOO
    scores. CV treats score groups as blocks; their shared reference-bank
    dependence means this is a predictive criterion, not an independence claim.
    """
    x = np.asarray(groups, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or min(x.shape[1:]) < 2 or not np.isfinite(x).all():
        raise ValueError('need finite scores from >=2 sources, probes and resolutions')
    mean, std, covariance = _moments(x)
    folds = []
    for source in range(len(x)):
        training = np.delete(x, source, axis=0)
        m, s, c = _moments(training)
        folds.append((c, (x[source] - m) / s, s))
    eye = np.eye(x.shape[-1])
    ridge = 1e-6

    def risk(alpha):
        values = []
        for c, z, s in folds:
            shrunk = (1-alpha)*c + alpha*eye + ridge*eye
            sign, logdet = np.linalg.slogdet(shrunk)
            if sign <= 0:
                return float('inf')
            precision = np.linalg.inv(shrunk)
            values.append(.5*(logdet + 2*np.log(s).sum()
                + np.einsum('ni,ij,nj->n', z, precision, z).mean()))
        return float(np.mean(values))

    if diagonal:
        alpha = 1.
    else:
        result = minimize_scalar(risk, bounds=(0., 1.), method='bounded',
                                 options={'xatol': 1e-5})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('normal source-block likelihood optimization failed')
        alpha = min([0., 1., float(result.x)], key=risk)
    shrunk = (1-alpha)*covariance + alpha*eye + ridge*eye
    return dict(mean=mean.tolist(), std=std.tolist(), covariance=shrunk.tolist(),
        precision=np.linalg.inv(shrunk).tolist(), shrinkage=float(alpha),
        source_count=len(x), probes_per_source=x.shape[1],
        fit_objective='equal_source_block_Gaussian_normal_score_log_likelihood',
        cv_negative_log_likelihood=risk(alpha), covariance_ridge=ridge,
        source_score_means=x.mean(1).tolist(),
        raw_correlation=covariance.tolist(),
        independent_probes_claimed=False, independent_CV_reference_banks_claimed=False,
        calibration_bank_size='source_count_minus_one', prediction_bank_size='source_count',
        bank_size_mismatch_corrected=False)


def gaussian_subset_scan(maps, model, *, chunk=131072):
    """sqrt(2 GLR) for H0 N(0,C) vs H1 N(a 1_S,C), a>=0, S!=empty.

    Enumerate all 2**R-1 subsets with equal shift in standardized score units.
    No labeled tuning, hand-picked scale subset or order weights enter the scan.
    """
    stacked = torch.stack(maps)
    r_count = len(maps)
    if stacked.ndim != 3 or len(model['mean']) != r_count:
        raise ValueError('resolution maps and fitted model do not match')
    device, dtype = stacked.device, stacked.dtype
    mean = torch.tensor(model['mean'], device=device, dtype=dtype)
    std = torch.tensor(model['std'], device=device, dtype=dtype)
    precision = torch.tensor(model['precision'], device=device, dtype=dtype)
    directions = torch.tensor(list(itertools.product([0., 1.], repeat=r_count))[1:],
                              device=device, dtype=dtype)
    filters = directions @ precision
    filters /= (filters * directions).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
    pixels = stacked.flatten(1).T
    outputs = []
    for start in range(0, len(pixels), chunk):
        z = (pixels[start:start+chunk] - mean) / std
        outputs.append((z @ filters.T).amax(1).clamp_min(0))
    return torch.cat(outputs).reshape(stacked.shape[1:])


@METHOD_REGISTRY.register('gaussian_scale_reference_matching')
class GaussianScaleReferenceMatching(MultiscaleAggregationReferenceMatching):
    def __init__(self, backbone, config):
        self.gaussian_scale_fusion = config.get('gaussian_scale_fusion', 'scan_shrink')
        if self.gaussian_scale_fusion not in {'zscore_order', 'scan_diagonal', 'scan_shrink'}:
            raise ValueError('unknown gaussian_scale_fusion')
        self.probe_side = int(config.get('gaussian_probe_side', 16))
        if self.probe_side < 2:
            raise ValueError('gaussian_probe_side must be >=2')
        super().__init__(backbone, config)
        if (self.multiscale_aggregation != 'scale_order_linear'
                or self.deep_layer_fusion != 'mean' or self.bank_bags < 2
                or not self.fast_reuse_calibration_bank or len(self.layers) != 3
                or self.layer_fusion_shallow_idx != 0 or self.ablation_shallow_mode != 'all'
                or self.noise_guard_action != 'lowres'):
            raise ValueError('joint calibration requires fixed three-layer linear bagged baseline')
        self._probe_fitting = False

    def _extract(self, image, resolution, pool=None):
        layers, grid = super()._extract(image, resolution, pool)
        if self._probe_fitting:
            indices, weights = bilinear_probe_indices(grid, self.probe_side)
            index = torch.as_tensor(indices.reshape(-1), device=layers[0].device)
            self._probe_features.setdefault(resolution, []).append(
                ([layer[index].detach().cpu() for layer in layers], weights))
        return layers, grid

    def _fit_scales_for_resolution(self, resolution):
        scales = super()._fit_scales_for_resolution(resolution)
        cache = self._fast_full_bank_cache
        if cache is None or cache[:2] != (resolution, id(self.records[resolution])):
            raise RuntimeError('missing matching source-LOO bank cache')
        bank = cache[2]
        owners = torch.as_tensor(bank['owners'], device=self.device)
        source_scores = []
        for source, (features, weights) in enumerate(self._probe_features[resolution]):
            scores = []
            for li, query in enumerate(features):
                distance = _masked_loo_distances(query.to(self.device), bank['layers'][li],
                    owners, source, self.chunk, self.knn_k)
                calibrated = self._tail_transform(self._last_tails[li], distance)
                interpolated = (calibrated.reshape(-1, 4) * torch.as_tensor(
                    weights, device=self.device, dtype=calibrated.dtype)).sum(1)
                scores.append(interpolated)
            fused = .5*(scores[1] + scores[2]) + self.layer_fusion_gamma*scores[0]
            source_scores.append(fused.detach().cpu().numpy())
        bags = self._probe_bags.setdefault(resolution, [])
        bags.append(np.stack(source_scores))
        if len(bags) == self.bank_bags:
            self._probe_features.pop(resolution)
        return scales

    def fit(self, support_paths, context):
        if len(support_paths) < 2:
            raise ValueError('Gaussian scale calibration requires >=2 normal support images')
        self._probe_features, self._probe_bags = {}, {}
        self._probe_fitting = True
        try:
            super().fit(support_paths, context)
        finally:
            self._probe_fitting = False
        probes = np.stack([np.mean(self._probe_bags[r], axis=0)
                           for r in self.resolutions], axis=-1)
        self.gaussian_scale_model = fit_gaussian_scale_model(probes,
            diagonal=self.gaussian_scale_fusion != 'scan_shrink')
        self._probe_features, self._probe_bags = {}, {}

    def _combine_maps(self, maps, resolutions_to_use):
        # The existing noise guard can select a resolution subset. Marginalize
        # the fitted normal model; do not reuse an incompatible precision block.
        indices = [list(self.resolutions).index(r) for r in resolutions_to_use]
        model = self.gaussian_scale_model
        mean = np.asarray(model['mean'])[indices]
        std = np.asarray(model['std'])[indices]
        if self.gaussian_scale_fusion == 'zscore_order':
            standardized = [(m-float(mu))/float(s) for m, mu, s in zip(maps, mean, std)]
            return combine_resolution_maps(standardized, 'scale_order_linear')
        covariance = np.asarray(model['covariance'])[np.ix_(indices, indices)]
        marginal = dict(mean=mean.tolist(), std=std.tolist(),
                        precision=np.linalg.inv(covariance).tolist())
        return gaussian_subset_scan(maps, marginal)

    def object_metadata(self):
        metadata = super().object_metadata()
        metadata.update(gaussian_scale_fusion=self.gaussian_scale_fusion,
            gaussian_scale_model=self.gaussian_scale_model,
            gaussian_probe_side=self.probe_side,
            score_definition='normal_support_joint_scale_' + self.gaussian_scale_fusion,
            query_used_for_fit=False, fit_source='normal_support_source_LOO_only',
            score_model='Gaussian_on_bag_mean_fixed_linear_layer_score_vectors',
            scale_selection='all_nonempty_subsets_positive_mean_Gaussian_GLRT'
                if self.gaussian_scale_fusion.startswith('scan') else 'fixed_order_weights',
            calibrated_p_value_claimed=False)
        return metadata
