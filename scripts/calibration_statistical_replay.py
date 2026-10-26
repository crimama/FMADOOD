#!/usr/bin/env python3
"""Post-hoc paired interventions on existing AD2 token caches; no fitting."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from calibration_mechanism_analysis import aggregate, curve, resolution_maps
from fmad.cli import load_config, parse_args, resolve_runtime_device
from fmad.evaluation.streaming import StreamingSegmentationAccumulator
from fmad.pipeline import _ground_truth_mask
from fmad.postprocess import component_quantile
from fmad.registry import DATASET_REGISTRY


SOURCE = Path('/workspace/results_local_compatible/scopead_mechanism_20261020')
ARMS = ('raw_rank_raw_value', 'q95_rank_raw_value',
        'raw_rank_q95_value', 'q95_rank_q95_value', 'raw_mean', 'q95_mean')


def rank_value(rank_scores, value_scores):
    """Ascending Rényi weights, stable resolution order for cross-arm ties."""
    n = rank_scores.shape[0]
    weights = rank_scores.new_tensor([sum(1 / (n-j) for j in range(k))
                                     for k in range(1, n+1)])
    weights /= weights.sum()
    order = torch.argsort(rank_scores, dim=0, stable=True)
    return (torch.gather(value_scores, 0, order) * weights[:, None, None]).sum(0)


def sparse_hist(acc, destination):
    codes = np.flatnonzero(acc.positive + acc.negative)
    np.savez_compressed(destination, codes=codes.astype(np.uint16),
                        positive=acc.positive[codes], negative=acc.negative[codes],
                        pro_weight=acc.pro_weight[codes], regions=acc.region_count)


def add_sparse(acc, path):
    with np.load(path) as h:
        codes = h['codes'].astype(np.int64)
        acc.positive[codes] += h['positive']
        acc.negative[codes] += h['negative']
        acc.pro_weight[codes] += h['pro_weight']
        acc.region_count += int(h['regions'])
        acc.prediction_map_count += 1


def threshold_code(negative, fpr):
    """Most permissive >= code threshold with calibration FPR <= target."""
    total = int(negative.sum())
    if not total:
        raise ValueError('no normal calibration pixels')
    tails = np.cumsum(negative[::-1], dtype=np.int64)[::-1]
    valid = np.flatnonzero(tails <= fpr * total)
    return int(valid[0]) if len(valid) else len(negative)


def crossfit(rows, root):
    """Normal-image-only thresholds; each evaluation image is held out once."""
    normal = [i for i, r in enumerate(rows) if r['normal_image']]
    defect = [i for i, r in enumerate(rows) if not r['normal_image']]
    folds = min(5, len(normal), len(defect))
    if folds < 2:
        return dict(status='insufficient_normal_or_defect_images', folds=folds)
    # Shared deterministic image assignments across arms and support seeds.
    fold_ids = {}
    for ids in (normal, defect):
        ordered = sorted(ids, key=lambda i: hashlib.sha256(rows[i]['image'].encode()).hexdigest())
        fold_ids.update({i: j % folds for j, i in enumerate(ordered)})
    output = {}
    for arm in ARMS:
        points = {str(f): dict(fp=0, negatives=0, tp=0, positives=0,
                              pro_mass=0., regions=0, thresholds=[]) for f in (.005, .01, .05)}
        for fold in range(folds):
            cal = StreamingSegmentationAccumulator()
            val = StreamingSegmentationAccumulator()
            for i in normal:
                if fold_ids[i] != fold:
                    add_sparse(cal, root / arm / f'{i:04}.npz')
            for i in range(len(rows)):
                if fold_ids[i] == fold:
                    add_sparse(val, root / arm / f'{i:04}.npz')
            for f, p in points.items():
                code = threshold_code(cal.negative, float(f))
                p['thresholds'].append(code)
                p['fp'] += int(val.negative[code:].sum())
                p['negatives'] += int(val.negative.sum())
                p['tp'] += int(val.positive[code:].sum())
                p['positives'] += int(val.positive.sum())
                p['pro_mass'] += float(val.pro_weight[code:].sum())
                p['regions'] += val.region_count
                for i, row in enumerate(rows):
                    if fold_ids[i] != fold:
                        continue
                    with np.load(root / arm / f'{i:04}.npz') as h:
                        selected = h['codes'].astype(np.int64) >= code
                        row.setdefault('heldout_counts', {}).setdefault(arm, {})[f] = dict(
                            fp=int(h['negative'][selected].sum()), negatives=int(h['negative'].sum()),
                            tp=int(h['positive'][selected].sum()), positives=int(h['positive'].sum()),
                            pro_mass=float(h['pro_weight'][selected].sum()), regions=int(h['regions']))
        for p in points.values():
            p.update(realized_fpr=p['fp']/p['negatives'], tpr=p['tp']/p['positives'],
                     pro=p['pro_mass']/p['regions'])
        output[arm] = points
    for i, row in enumerate(rows):
        row['crossfit_fold'] = fold_ids[i]
    return dict(status='complete', folds=folds, arms=output,
                threshold_rule='>= float16 score code; conservative empirical normal-image FPR',
                calibration='normal evaluation images in other folds; post-hoc public GT',
                evaluation='all pixels in held-out images; pooled across held-out folds')


def sampled_diagnostics(raw, q95, mask, rng):
    output = {}
    r_count = len(raw)
    for kind, population in (('normal', ~mask), ('defect', mask)):
        ids = np.flatnonzero(population)
        ids = rng.choice(ids, min(len(ids), 16384 if kind == 'normal' else 4096), replace=False)
        if not len(ids):
            output[kind] = dict(count=0)
            continue
        index = torch.as_tensor(ids, device=raw.device)
        a, b = [x.reshape(r_count, -1)[:, index] for x in (raw, q95)]
        oa, ob = [torch.argsort(x, dim=0, stable=True) for x in (a, b)]
        quantiles = (.95, .99, .999)
        aq, bq = [np.quantile(x.cpu().numpy(), quantiles, axis=1).T for x in (a, b)]
        # These are image summaries, not independent pixel observations.
        output[kind] = dict(count=len(ids),
            any_rank_change=float((oa != ob).any(0).float().mean()),
            top_resolution_change=float((oa[-1] != ob[-1]).float().mean()),
            raw_quantiles=aq.tolist(), q95_quantiles=bq.tolist(),
            raw_log_quantile_sd=np.std(np.log(np.maximum(aq, 1e-12)), axis=0).tolist(),
            q95_log_quantile_sd=np.std(np.log(np.maximum(bq, 1e-12)), axis=0).tolist(),
            raw_renyi_uplift=float((rank_value(a[:, None], a[:, None]).squeeze(0)-a.mean(0)).mean()),
            q95_renyi_uplift=float((rank_value(b[:, None], b[:, None]).squeeze(0)-b.mean(0)).mean()))
    return output


def main():
    config = load_config(parse_args())
    dataset = DATASET_REGISTRY.get(config['dataset'])(data_root=config['data_root'],
                                      config=config.get('dataset_config', {}))
    objects = dataset.get_objects()
    if len(objects) != 1:
        raise ValueError('one class required')
    name = objects[0].name
    seed, shots = config['seed'], config['protocol']['shots']
    label = f'ad2_mechanism_s{shots:02}_seed{seed}'
    matches = []
    for stage in SOURCE.glob(f'lane_*/{label}'):
        for index in stage.rglob('mechanism_cache_index.json'):
            info = json.loads(index.read_text())
            if info['object'] == name:
                matches.append((index, info))
    if len(matches) != 1:
        raise ValueError(f'expected exactly one source cache: {name}, {label}: {len(matches)}')
    source, info = matches[0]
    original = info['config']
    for key in ('seed', 'protocol', 'method', 'method_config', 'backbone', 'backbone_config', 'dataset_config'):
        if config.get(key) != original.get(key):
            raise ValueError(f'source config mismatch: {key}')
    first_analysis = json.loads(source.with_name('calibration_mechanism_analysis.json').read_text())
    if first_analysis['baseline_parity_abs'] > 3e-5:
        raise ValueError('invalid source parity')
    out = Path(config['output_dir']) / name
    out.mkdir(parents=True, exist_ok=False)
    hist_root = out / 'image_histograms'
    for arm in ARMS:
        (hist_root / arm).mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(config))
    for filename in ('run_manifest.json', 'metrics.json', 'mechanism_cache_index.json'):
        shutil.copy2(source.with_name(filename), out / ('source_' + filename))
    device = resolve_runtime_device(config['device'])
    torch.set_num_threads(4)
    mc = config['method_config']
    if mc['prediction_postprocess'] != 'component_quantile':
        raise ValueError('unsupported postprocessing')
    totals = {arm: StreamingSegmentationAccumulator() for arm in ARMS}
    image_rows = []
    max_map_parity = 0.
    rng = np.random.default_rng(42103 + seed)
    for i, (path, record) in enumerate(info['records'].items()):
        with np.load(source.parent / 'mechanism_cache' / record['file']) as archive:
            item = {k: archive[k] for k in archive.files}
        mask = _ground_truth_mask(dataset, name, Path(path).parent.name, path, tuple(item['shape']))
        raw = torch.stack(resolution_maps(item, 'none', device))
        q95 = torch.stack(resolution_maps(item, 'q95', device))
        pairs = ((raw, raw), (q95, raw), (raw, q95), (q95, q95))
        row = dict(image=str(Path(path).relative_to(Path(config['data_root']))),
                   normal_image=not bool(mask.any()), resolutions=item['resolutions'].tolist(),
                   sampled=sampled_diagnostics(raw, q95, mask, rng), arm_image_summary={})
        for j, arm in enumerate(ARMS):
            if j < 4:
                prediction = rank_value(*pairs[j])
            else:
                prediction = (raw if j == 4 else q95).mean(0)
            if j in (0, 3):
                expected = aggregate(list(raw if j == 0 else q95), 1)
                error = float((prediction-expected).abs().max())
                max_map_parity = max(max_map_parity, error)
                if error > 2e-5:
                    raise ValueError(f'rank/value endpoint mismatch: {error}')
            prediction = component_quantile(prediction.cpu().numpy(),
                percentile=mc['component_percentile'], q=mc['component_q'],
                dilate_size=mc['component_dilate_size'])
            acc = StreamingSegmentationAccumulator()
            acc.update([prediction], [mask])
            sparse_hist(acc, hist_root / arm / f'{i:04}.npz')
            for field in ('positive', 'negative', 'pro_weight'):
                getattr(totals[arm], field)[:] += getattr(acc, field)
            totals[arm].region_count += acc.region_count
            totals[arm].prediction_map_count += 1
            row['arm_image_summary'][arm] = dict(
                normal_mean=float(prediction[~mask].mean()) if (~mask).any() else None,
                defect_mean=float(prediction[mask].mean()) if mask.any() else None)
        image_rows.append(row)
        del raw, q95, pairs, prediction
        if (i+1) % 10 == 0:
            print(f'STATISTICAL_REPLAY {name} {i+1}/{len(info["records"])} arms=6', flush=True)
    metrics = {}
    original_keys = dict(raw_rank_raw_value='none/alpha1', q95_rank_q95_value='q95/alpha1',
                         raw_mean='none/alpha0', q95_mean='q95/alpha0')
    parity = {}
    for arm, acc in totals.items():
        result = acc.compute()
        metrics[arm] = dict(seg_AUPRO_0_05=result.seg_aupro_0_05,
                           seg_AUROC_0_05=result.seg_auroc_0_05,
                           operating_points=curve(acc), image_count=acc.prediction_map_count)
        if arm in original_keys:
            difference = abs(result.seg_aupro_0_05 - first_analysis['metrics'][original_keys[arm]]['seg_AUPRO_0_05'])
            parity[arm] = difference
            if difference > 3e-5:
                raise ValueError(f'original metric mismatch {arm}: {difference}')
    heldout = crossfit(image_rows, hist_root)
    report = dict(object=name, seed=seed, shots=shots, metrics=metrics,
        source_index=str(source), source_index_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metric_parity_abs=parity, map_parity_max_abs=max_map_parity,
        crossfit=heldout, images=image_rows, completed_at=time.time(),
        rank_ties='stable input resolution order; originals have equal values at ties',
        quantile_sampling='paired up to 16384 normal / 4096 defect native pixels per image; image summaries',
        interpretation='post-hoc development interventions; rank/value hybrids are diagnostic, not final methods',
        limitations='crossfit normal test GT is analysis only; FPR may shift on held-out images; 3 support seeds; sampled upper tails')
    temp = out / 'statistical_analysis.json.tmp'
    temp.write_text(json.dumps(report, indent=2)+'\n')
    temp.rename(out / 'statistical_analysis.json')
    print(f'STATISTICAL_ANALYSIS_COMPLETE {name} images={len(image_rows)} parity={max(parity.values())}', flush=True)


if __name__ == '__main__':
    main()
