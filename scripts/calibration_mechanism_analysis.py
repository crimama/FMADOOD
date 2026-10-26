#!/usr/bin/env python3
"""Common normal-only pipeline, then paired mechanism analysis using public GT.

Exact float32 token distances remain remote. Labels are read only after the
common pipeline finishes. Counterfactuals reuse identical features and banks.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from fmad.cli import parse_args, load_config, resolve_runtime_device
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY
from fmad.methods import register_methods
from fmad.pipeline import run_pipeline, _ground_truth_mask
from fmad.protocol import seed_generators
from fmad.evaluation.streaming import StreamingSegmentationAccumulator
from fmad.methods.multiscale_aggregation_reference_matching import combine_resolution_maps


def resolution_maps(item, normalization, device, noise=None):
    maps = []
    for r in item['resolutions']:
        key = 'raw' if normalization == 'none' else 'q95'
        values = torch.as_tensor(item[f'{key}_{r}'], device=device)
        if noise is not None:
            factors = torch.as_tensor(noise[r], device=device, dtype=values.dtype)
            values = values / factors[:, :, None, None]
        bags = [torch.stack([v[1], v[2]]).mean(0) + .125 * v[0] for v in values]
        score = torch.stack(bags).mean(0)
        maps.append(F.interpolate(score[None, None], size=tuple(item['shape']),
                                  mode='bilinear', align_corners=False)[0, 0])
    return maps


def aggregate(maps, alpha):
    mean = combine_resolution_maps(maps, 'mean')
    if alpha == 0:
        return mean
    renyi = combine_resolution_maps(maps, 'scale_order_renyi')
    if alpha == 1:
        return renyi
    if alpha == 2:
        return torch.stack(maps).max(0).values
    return (1 - alpha) * mean + alpha * renyi


def curve(acc):
    active = np.flatnonzero(acc.positive + acc.negative)[::-1]
    fpr = np.r_[0., np.cumsum(acc.negative[active]) / acc.negative.sum()]
    tpr = np.r_[0., np.cumsum(acc.positive[active]) / acc.positive.sum()]
    pro = np.r_[0., np.cumsum(acc.pro_weight[active]) / acc.region_count]
    last = np.r_[np.flatnonzero(np.diff(fpr) != 0), len(fpr) - 1]
    return {str(f): dict(tpr=float(np.interp(f, fpr[last], tpr[last])),
                         pro=float(np.interp(f, fpr[last], pro[last])))
            for f in (.001, .005, .01, .02, .05)}


def main():
    config = load_config(parse_args())
    options = config.get('evaluation', {}).get('mechanism_analysis', {})
    if options.get('use_evaluation_ground_truth') is not True or options.get('save_token_cache') is not True:
        raise ValueError('explicit diagnostic labels and remote token-cache options required')
    mc = config['method_config']
    if (mc['statistical_normalization'] != 'q95' or mc['ablation_shallow_mode'] != 'all'
            or mc['noise_guard_action'] != 'lowres' or mc.get('layer_adaptation', 'fixed') != 'fixed'
            or mc.get('deep_layer_fusion', 'mean') != 'mean'
            or mc['layer_fusion_gamma'] != .125 or mc['feature_layers'] != [5, 11, 23]):
        raise ValueError('analysis requires the final fixed ScopeAD setting')
    device = resolve_runtime_device(config['device'])
    seed_generators(config['seed'])
    register_methods(config['method'])
    dataset = DATASET_REGISTRY.get(config['dataset'])(data_root=config['data_root'],
                                                       config=config.get('dataset_config', {}))
    objects = dataset.get_objects()
    if len(objects) != 1:
        raise ValueError('one class per runner cell required')
    name = objects[0].name
    backbone = BACKBONE_REGISTRY.get(config['backbone'])(model_name=config['backbone'],
                       device=device, **config.get('backbone_config', {}))
    method = METHOD_REGISTRY.get(config['method'])(backbone=backbone, config=mc)
    root = Path(config['output_dir'])
    cache = root / ('mechanism_cache_' + name)
    cache.mkdir(parents=True, exist_ok=False)
    holder, records = {}, {}
    original_observe = method._observe_layer_scores
    original_score = method._score_tokens
    original_predict = method.predict

    def observe(distances, scores, *, tau):
        if holder.get('active'):
            holder['raw'].append(torch.stack(distances).detach().cpu().numpy().copy())
            holder['q95'].append(torch.stack(scores).detach().cpu().numpy().copy())
        return original_observe(distances, scores, tau=tau)

    def score(layers, grid, bank, *, normalize=True):
        if not holder.get('active'):
            return original_score(layers, grid, bank, normalize=normalize)
        holder['raw'], holder['q95'] = [], []
        result = original_score(layers, grid, bank, normalize=normalize)
        resolution = holder['remaining'].pop(0)
        for key in ('raw', 'q95'):
            holder['item'][f'{key}_{resolution}'] = np.stack(holder[key]).reshape(mc['bank_bags'], 3, *grid)
        return result

    original_resolutions = method._predict_resolutions

    def predict_resolutions(image, resolutions):
        if holder.get('active'):
            holder['remaining'] = list(resolutions)
            holder['item'].update(resolutions=np.asarray(resolutions), shape=np.asarray(image.shape[:2]))
        return original_resolutions(image, resolutions)

    def predict(path):
        holder.update(active=True, item={})
        try:
            result = original_predict(path)
            item = holder['item']
            check = aggregate(resolution_maps(item, 'q95', device), 1).detach().cpu().numpy()
            check = method._postprocess_prediction(check, None)
            error = float(np.max(np.abs(check - result[0])))
            if error > 2e-5:
                raise RuntimeError(f'full-resolution replay mismatch: {error}')
            key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
            np.savez(cache / (key + '.npz'), **item)
            records[str(path)] = dict(file=key + '.npz', replay_max_abs_error=error)
            return result
        finally:
            holder['active'] = False

    method._observe_layer_scores = observe
    method._score_tokens = score
    method._predict_resolutions = predict_resolutions
    method.predict = predict
    previous = set(root.rglob('run_manifest.json'))
    run_pipeline(method, dataset, backbone, config, str(root))
    manifests = list(set(root.rglob('run_manifest.json')) - previous)
    if len(manifests) != 1:
        raise RuntimeError('expected one new manifest')
    run_root = manifests[0].parent
    cache.rename(run_root / 'mechanism_cache')
    cache = run_root / 'mechanism_cache'
    index = dict(object=name, records=records, config=config,
                 label_use='AD2 public seen development GT, only after prediction, post-hoc mechanism analysis',
                 precision='exact float32 token distances/scores, native-resolution replay; remote only')
    (run_root / 'mechanism_cache_index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(f'MECHANISM_CAPTURE_COMPLETE {name} images={len(records)}', flush=True)

    arms = {f'{norm}/alpha{alpha:g}': (norm, alpha, None)
            for norm in ('none', 'q95') for alpha in (0., .5, 1., 2.)}
    rng = np.random.default_rng(84391 + config['seed'])
    for sigma in (.1, .2, .3):
        for rep in range(3):
            noise = {r: np.exp(rng.normal(-sigma*sigma/2, sigma, (mc['bank_bags'], 3)))
                     for r in mc['resolutions']}
            for alpha in (0., 1.):
                arms[f'q95/noise{sigma}/rep{rep}/alpha{alpha:g}'] = ('q95', alpha, noise)
    accumulators = {key: StreamingSegmentationAccumulator() for key in arms}
    bank_stats = []
    rank_counts = {norm: {kind: np.zeros((4, 4), dtype=np.int64)
                        for kind in ('normal', 'defect')} for norm in ('none', 'q95')}
    rank_rng = np.random.default_rng(1937 + config['seed'])
    routed_images = 0
    for number, (path, entry) in enumerate(records.items()):
        with np.load(cache / entry['file']) as archive:
            item = {key: archive[key] for key in archive.files}
        mask = _ground_truth_mask(dataset, name, Path(path).parent.name, path, tuple(item['shape']))
        selected = {}
        for kind, population in (('normal', ~mask), ('defect', mask)):
            indices = np.flatnonzero(population)
            selected[kind] = rank_rng.choice(indices, min(4096, len(indices)), replace=False)
        routed_images += int(len(item['resolutions']) != 4)
        map_cache = {}
        for key, (norm, alpha, noise) in arms.items():
            identity = key.rsplit('/alpha', 1)[0]
            if identity not in map_cache:
                map_cache.clear()
                map_cache[identity] = resolution_maps(item, norm, device, noise)
                if noise is None and len(item['resolutions']) == 4:
                    for kind, indices in selected.items():
                        if len(indices):
                            pixels = torch.stack([m.reshape(-1)[torch.as_tensor(indices, device=device)]
                                                  for m in map_cache[identity]]).cpu().numpy()
                            order = np.argsort(-pixels, axis=0, kind='stable')
                            for position in range(4):
                                rank_counts[norm][kind][:, position] += np.bincount(order[position], minlength=4)
            output = aggregate(map_cache[identity], alpha).detach().cpu().numpy()
            output = method._postprocess_prediction(output, None)
            accumulators[key].update([output], [mask])
        for r in item['resolutions']:
            raw = item[f'raw_{r}'].mean(0)
            calibrated = item[f'q95_{r}'].mean(0)
            grid_mask = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                                      size=raw.shape[-2:], mode='nearest')[0, 0].numpy() > .5
            for layer in range(3):
                values = raw[layer]
                normal, defect = values[~grid_mask], values[grid_mask]
                bank_stats.append(dict(image=Path(path).parent.name + '/' + Path(path).name,
                    resolution=int(r), layer=mc['feature_layers'][layer],
                    normal_pixels=int(normal.size), defect_pixels=int(defect.size),
                    raw_normal_q95=float(np.quantile(normal, .95)) if normal.size else None,
                    calibrated_normal_q95=float(np.quantile(calibrated[layer][~grid_mask], .95)) if normal.size else None,
                    raw_defect_mean=float(defect.mean()) if defect.size else None))
        if (number + 1) % 10 == 0:
            print(f'MECHANISM_REPLAY {name} {number+1}/{len(records)} arms={len(arms)}', flush=True)
    evaluated = {}
    for key, acc in accumulators.items():
        result = acc.compute()
        evaluated[key] = dict(seg_AUPRO_0_05=result.seg_aupro_0_05,
            seg_AUROC_0_05=result.seg_auroc_0_05, prediction_map_count=result.prediction_map_count,
            operating_points=curve(acc))
    baseline = json.loads((run_root / 'metrics.json').read_text())[name]['seg_AUPRO_0.05']
    parity = abs(evaluated['q95/alpha1']['seg_AUPRO_0_05'] - baseline)
    if parity > 3e-5:
        raise RuntimeError(f'full metric parity failed: {parity}')
    scales = {str(r): [[float(t['statistical']['scale']) for t in bag] for bag in bags]
              for r, bags in method.bag_tails.items()}
    report = dict(object=name, seed=config['seed'], shots=config['protocol']['shots'],
        metrics=evaluated, baseline_parity_abs=parity, support_q95=scales,
        sampled_rank_counts={n: {k: v.tolist() for k, v in counts.items()} for n, counts in rank_counts.items()},
        rank_sampling='up to 4096 native pixels per image per normal/defect; paired between arms; stable ties',
        routed_images=routed_images,
        bank_grid_diagnostics=bank_stats, completed_at=time.time(),
        label_use=index['label_use'],
        interpretation='alpha0=mean; alpha0.5=midpoint; alpha1=Renyi; alpha2=max (endpoint label)',
        bank_diagnostic_limit='nearest GT sampled on feature grids; tiny regions can disappear',
        noise_limit='synthetic sensitivity, not a confidence interval for actual support estimator error')
    (run_root / 'calibration_mechanism_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'MECHANISM_ANALYSIS_COMPLETE {name} parity={parity} arms={len(arms)}', flush=True)


if __name__ == '__main__':
    main()
