"""Explicit label-using diagnostic, isolated from method fitting/prediction.

Positive per-observation multipliers optimize sampled weighted PRO; candidates
must subsequently be selected with the unchanged full-pixel evaluator. This is
a finite supervised search, not a proven optimum or deployable calibration.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import differential_evolution

from fmad.evaluation.streaming import _label8


def sample_coordinates(mask, rng, normal_cap=128, region_cap=16):
    labels, count = _label8(mask)
    indices, normal_weights, region_weights = [], [], []
    for region in range(count + 1):
        eligible = np.flatnonzero(labels == region)
        if not len(eligible):
            continue
        k = min(normal_cap if region == 0 else region_cap, len(eligible))
        indices.extend(rng.choice(eligible, k, replace=False).tolist())
        normal_weights.extend([len(eligible) / k if region == 0 else 0.] * k)
        region_weights.extend([1. / k if region else 0.] * k)
    return np.asarray(indices), np.asarray(normal_weights), np.asarray(region_weights)


def observation_samples(native_bags, shape, indices, gamma=.125, device='cpu'):
    """Interpolate normalized observations at original-image coordinates."""
    height, width = shape
    y, x = np.unravel_index(indices, shape)
    grid = torch.as_tensor(np.stack([2*(x+.5)/width-1, 2*(y+.5)/height-1], -1),
                           dtype=torch.float32, device=device).reshape(1, -1, 1, 2)
    values = []
    for bags in native_bags:
        tensor = torch.as_tensor(bags, dtype=torch.float32, device=device)
        b, layers, h, w = tensor.shape
        if layers != 3:
            raise ValueError('oracle requires declared shallow/deep/deep layers')
        plane = F.grid_sample(tensor.reshape(1, b*layers, h, w), grid,
                              mode='bilinear', padding_mode='border', align_corners=False)
        coefficient = torch.tensor([gamma, .5, .5], device=device).repeat(b)
        values.append((plane[0, :, :, 0].T * coefficient / (b * len(native_bags))).cpu().numpy())
    return np.concatenate(values, axis=1)


def reconstruct_map(native_bags, shape, multipliers, gamma=.125, device='cpu'):
    """Retain parent bag/layer fusion, interpolation, resolution mean order."""
    total, offset = None, 0
    for bags in native_bags:
        tensor = torch.as_tensor(bags, dtype=torch.float32, device=device)
        n, layers, h, w = tensor.shape
        weights = torch.as_tensor(multipliers[offset:offset+n*layers],
                                  dtype=torch.float32, device=device).reshape(n, layers, 1, 1)
        tensor = tensor * weights
        # Preserve each bag's layer reduction before averaging across bags.
        fused = (tensor[:, 1:].mean(1) + gamma * tensor[:, 0]).mean(0)
        native = F.interpolate(fused.reshape(1,1,h,w), size=shape,
                               mode='bilinear', align_corners=False)[0,0]
        total = native if total is None else total + native
        offset += n*layers
    if offset != len(multipliers):
        raise ValueError('oracle observation count mismatch')
    return (total / len(native_bags)).cpu().numpy()


def weighted_partial_pro(scores, normal_weights, region_weights, limit=.05):
    """Exact tied-score weighted PRO integral for a sampled population."""
    scores = np.asarray(scores, np.float16)
    order = np.argsort(-scores.astype(np.float32), kind='stable')
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(order)-1]
    fpr = np.r_[0., np.cumsum(normal_weights[order])[ends] / normal_weights.sum()]
    pro = np.r_[0., np.cumsum(region_weights[order])[ends] / region_weights.sum()]
    from fmad.evaluation.streaming import _bounded_trapezoid
    return float(_bounded_trapezoid(fpr, pro, limit) / limit)


ORACLE_AXES = ('full', 'layer', 'resolution', 'bag', 'resolution_layer',
               'resolution_layer_fixed_layer')


def oracle_parameter_count(mode, shape):
    r, b, l = shape
    return {'full': r*b*l, 'layer': l, 'resolution': r, 'bag': b,
            'resolution_layer': r*l, 'resolution_layer_fixed_layer': r*l}[mode]


def expand_oracle_parameters(logs, mode, shape, bound=8.):
    """Expand bounded group multipliers in resolution/bag/layer order.

    Fixed-layer projection preserves the arithmetic mean coefficient of EVERY
    layer, including shallow .125, and enforces the same multiplier bounds.
    """
    r, b, l = shape
    logs = np.asarray(logs, dtype=np.float64)
    if logs.ndim != 2 or logs.shape[1] != oracle_parameter_count(mode, shape):
        raise ValueError('oracle group parameter count mismatch')
    n = len(logs)
    weights = np.exp(logs)
    if mode == 'full':
        return weights
    if mode == 'layer':
        weights = weights.reshape(n, 1, 1, l)
    elif mode == 'resolution':
        weights = weights.reshape(n, r, 1, 1)
    elif mode == 'bag':
        weights = weights.reshape(n, 1, b, 1)
    else:
        weights = weights.reshape(n, r, 1, l)
        if mode == 'resolution_layer_fixed_layer':
            low = np.zeros((n, 1, 1, l))
            high = 1. / weights.min(axis=1, keepdims=True)
            for _ in range(48):
                mid = (low + high) / 2
                mean = np.clip(weights * mid, 1/bound, bound).mean(axis=1, keepdims=True)
                low = np.where(mean < 1., mid, low)
                high = np.where(mean >= 1., mid, high)
            weights = np.clip(weights * ((low + high) / 2), 1/bound, bound)
    return np.broadcast_to(weights, (n, r, b, l)).reshape(n, -1).copy()


def search_scales(features, normal_weights, region_weights, device='cpu', seed=0,
                  iterations=20, bound=8., mode='full', observation_shape=None,
                  population=None):
    """48-D bounded positive search; retain baseline and top sampled candidates."""
    features = np.asarray(features, np.float32)
    if not np.isfinite(features).all() or normal_weights.sum() <= 0 or region_weights.sum() <= 0:
        raise ValueError('invalid oracle sample')
    x = torch.as_tensor(features, device=device)
    if mode not in ORACLE_AXES:
        raise ValueError('unknown oracle axis mode')
    shape = observation_shape
    if mode != 'full' and (shape is None or np.prod(shape) != features.shape[1]):
        raise ValueError('grouped oracle requires matching observation shape')
    dimensions = features.shape[1] if shape is None else oracle_parameter_count(mode, shape)
    candidates = []
    evaluations = 0
    def objective(logs):
        nonlocal evaluations
        logs = np.asarray(logs)
        single = logs.ndim == 1
        logs = logs[:, None] if single else logs
        weights = np.exp(logs.T) if shape is None else expand_oracle_parameters(logs.T, mode, shape, bound)
        # Chunk population to bound temporary device memory.
        losses = []
        for start in range(0, len(weights), 16):
            scores = (x @ torch.as_tensor(weights[start:start+16].T,
                                         dtype=x.dtype, device=device)).cpu().numpy()
            for j, weight in enumerate(weights[start:start+16]):
                value = weighted_partial_pro(scores[:, j], normal_weights, region_weights)
                losses.append(-value)
                candidates.append((value, weight.copy()))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        del candidates[3:]
        evaluations += len(weights)
        return losses[0] if single else np.asarray(losses)
    baseline = np.ones(features.shape[1])
    objective(np.zeros(dimensions))
    baseline_value = candidates[0][0]
    popsize = 2 if population is None else max(1, int(np.ceil(population / dimensions)))
    differential_evolution(objective, [(-np.log(bound), np.log(bound))]*dimensions,
        maxiter=iterations, popsize=popsize, seed=seed, polish=False, vectorized=True,
        updating='deferred', x0=np.zeros(dimensions), tol=0., atol=0.)
    unique = [('q95', baseline)]
    for i, (_, weights) in enumerate(candidates):
        if not any(np.array_equal(weights, w) for _, w in unique):
            unique.append((f'oracle_candidate_{i}', weights))
    return unique, dict(sampled_q95_aupro=baseline_value,
        sampled_best_aupro=candidates[0][0], evaluations=evaluations,
        optimizer='differential_evolution', iterations=iterations,
        multiplier_bounds=[1/bound, bound], dimensions=features.shape[1],
        oracle_axis_mode=mode, search_parameters=dimensions,
        effective_degrees_of_freedom=(dimensions - shape[2] if mode == 'resolution_layer_fixed_layer' else dimensions - 1),
        requested_population=population,
        normal_samples=int(np.count_nonzero(normal_weights)),
        defect_samples=int(np.count_nonzero(region_weights)),
        interpretation='GT-optimized same-population diagnostic; finite search, no optimum guarantee')
