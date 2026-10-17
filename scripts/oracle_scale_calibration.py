#!/usr/bin/env python3
"""Common normal-only pipeline followed by explicit GT-using scale diagnosis."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from fmad.cli import parse_args, load_config, resolve_runtime_device
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY
from fmad.methods import register_methods
from fmad.pipeline import run_pipeline, _ground_truth_mask
from fmad.protocol import seed_generators
from fmad.evaluation.streaming import StreamingSegmentationAccumulator
from fmad.evaluation.oracle_scale import (
    sample_coordinates, observation_samples, reconstruct_map, search_scales,
)


def main():
    config = load_config(parse_args())
    options = config.get('evaluation', {}).get('oracle_scale_calibration')
    if not isinstance(options, dict) or options.get('use_evaluation_ground_truth') is not True:
        raise ValueError('oracle requires explicit evaluation GT authorization in config')
    device = resolve_runtime_device(config['device'])
    seed_generators(config['seed'])
    register_methods(config['method'])
    dataset = DATASET_REGISTRY.get(config['dataset'])(
        data_root=config['data_root'], config=config.get('dataset_config', {}))
    objects = dataset.get_objects()
    if len(objects) != 1:
        raise ValueError('oracle entrypoint processes exactly one class per runner cell')
    name = objects[0].name
    backbone = BACKBONE_REGISTRY.get(config['backbone'])(model_name=config['backbone'],
        device=device, **config.get('backbone_config', {}))
    method = METHOD_REGISTRY.get(config['method'])(backbone=backbone,
        config=config.get('method_config', {}))
    mc = config['method_config']
    if (mc.get('statistical_normalization') != 'q95'
            or mc.get('ablation_shallow_mode') != 'all'
            or mc.get('prediction_postprocess') != 'none'):
        raise ValueError('oracle requires Q95, linear shallow, no postprocessing')
    gamma = float(mc['layer_fusion_gamma'])
    cached = []
    original_predict = method.predict_batch
    def capture(paths):
        predictions = original_predict(paths)
        for path, prediction in zip(paths, predictions):
            audit = prediction.diagnostics['calibration_audit']
            if audit['resolutions'] != mc['resolutions']:
                raise RuntimeError('oracle fixed observation scope requires no resolution routing')
            # Native components remain in RAM only, full precision. Labels are
            # neither loaded here nor passed to fitting or method prediction.
            native = [np.array(bags[:, 1], copy=True) for bags in audit['native_bags']]
            shape = prediction.score_map.shape
            check = reconstruct_map(native, shape, np.ones(sum(x.shape[0]*x.shape[1] for x in native)),
                                    gamma, device)
            error = float(np.max(np.abs(check - prediction.score_map)))
            if error > 2e-5:
                raise RuntimeError(f'oracle reconstruction mismatch {error}')
            cached.append(dict(path=path, shape=shape, native=native, reconstruction_error=error))
        return predictions
    method.predict_batch = capture
    # Shared pipeline owns support, backbone fitting, provenance and Q95 metrics.
    root = Path(config['output_dir'])
    previous_manifests = set(root.rglob('run_manifest.json'))
    run_pipeline(method, dataset, backbone, config, config['output_dir'])
    manifests = list(set(root.rglob('run_manifest.json')) - previous_manifests)
    if len(manifests) != 1:
        raise RuntimeError('oracle expected exactly one original manifest')
    manifest_path = manifests[0]
    run_root = manifest_path.parent
    original_metrics = json.loads((run_root/'metrics.json').read_text())
    manifest = json.loads(manifest_path.read_text())
    (run_root/'baseline_q95_metrics.json').write_text(json.dumps(original_metrics, indent=2)+'\n')
    baseline = original_metrics[name]
    features, negative, positive = [], [], []
    for item in cached:
        path = item['path']
        anomaly = Path(path).parent.name
        mask = _ground_truth_mask(dataset, name, anomaly, path, item['shape'])
        # All following label access belongs solely to this oracle analysis.
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(str(Path(path).name).encode()).digest()[:8], 'little'))
        indices, nw, pw = sample_coordinates(mask, rng,
            normal_cap=int(options.get('normal_samples_per_image', 128)),
            region_cap=int(options.get('samples_per_region', 16)))
        features.append(observation_samples(item['native'], item['shape'], indices, gamma, device))
        negative.append(nw)
        positive.append(pw)
    modes = options.get('axis_modes', ['full'])
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('oracle axis modes must be unique and nonempty')
    shape = (len(mc['resolutions']), mc['bank_bags'], len(mc['feature_layers']))
    matrix, nw, pw = np.concatenate(features), np.concatenate(negative), np.concatenate(positive)
    candidates = [('q95', np.ones(matrix.shape[1]))]
    searches = {}
    mode_candidates = {}
    for mode in modes:
        values, search = search_scales(matrix, nw, pw, device=device, seed=config['seed'],
            iterations=int(options.get('iterations', 20)), bound=float(options.get('bound', 8.)),
            mode=mode, observation_shape=shape, population=options.get('population'))
        searches[mode] = search
        mode_candidates[mode] = [(f'{mode}/{label}', w) for label, w in values if label != 'q95']
        candidates.extend(mode_candidates[mode])
        print(f'ORACLE_SEARCH_DONE {name} {mode} sampled={search["sampled_best_aupro"]:.6f}', flush=True)
    accumulators = {label: StreamingSegmentationAccumulator() for label, _ in candidates}
    for item in cached:
        path = item['path']
        mask = _ground_truth_mask(dataset, name, Path(path).parent.name, path, item['shape'])
        for label, weights in candidates:
            score = reconstruct_map(item['native'], item['shape'], weights, gamma, device)
            accumulators[label].update([score], [mask])
    evaluated = {}
    for label, accumulator in accumulators.items():
        result = accumulator.compute()
        evaluated[label] = dict(seg_AUPRO_0_05=result.seg_aupro_0_05,
            seg_AUROC_0_05=result.seg_auroc_0_05, seg_F1_oracle=result.seg_f1,
            best_thre=result.best_threshold, seg_AUROC_1_00=result.seg_auroc_1_00,
            seg_AUPRO_0_30=result.seg_aupro_0_30)
    parity = abs(evaluated['q95']['seg_AUPRO_0_05'] - baseline['seg_AUPRO_0.05'])
    if parity > 3e-5:
        raise RuntimeError(f'oracle reconstructed Q95 full metric differs by {parity}')
    selected = {}
    for mode in modes:
        best_label, best_weights, best = 'q95_original', np.ones(matrix.shape[1]), baseline.copy()
        for label, weights in mode_candidates[mode]:
            if evaluated[label]['seg_AUPRO_0_05'] > best['seg_AUPRO_0.05']:
                values = evaluated[label]
                best_label, best_weights = label, weights
                best = {**baseline, 'seg_AUPRO_0.05': values['seg_AUPRO_0_05'],
                    'seg_AUROC_0.05': values['seg_AUROC_0_05'],
                    'seg_AUROC_1.00': values['seg_AUROC_1_00'],
                    'seg_AUPRO_0.30': values['seg_AUPRO_0_30'],
                    'seg_F1_oracle': values['seg_F1_oracle'], 'best_thre': values['best_thre']}
                best.pop('image_AUROC', None)
        selected[mode] = (best_label, best_weights, best)
    primary_mode = 'full' if 'full' in modes else modes[0]
    best_label, best_weights, best = selected[primary_mode]
    search = searches[primary_mode]
    q95_scales = [float(t['statistical']['scale']) for r in mc['resolutions']
                  for bag in method.bag_tails[r] for t in bag]
    record = dict(**search, completed_at=time.time(), selected=best_label,
        primary_axis_mode=primary_mode,
        axis_ablation={mode: dict(search=searches[mode], selected=v[0],
            selected_multipliers=v[1].tolist(), metrics=v[2],
            delta_aupro=v[2]['seg_AUPRO_0.05']-baseline['seg_AUPRO_0.05'])
            for mode,v in selected.items()},
        observation_order=[dict(resolution=r, bag=b, layer=layer) for r in mc['resolutions']
                           for b in range(mc['bank_bags']) for layer in mc['feature_layers']],
        q95_scales=q95_scales,
        selected_divisor_scales=(np.asarray(q95_scales)/best_weights).tolist(),
        selected_multipliers=best_weights.tolist(), full_candidate_metrics=evaluated,
        baseline_aupro=baseline['seg_AUPRO_0.05'], selected_aupro=best['seg_AUPRO_0.05'],
        delta_aupro=best['seg_AUPRO_0.05']-baseline['seg_AUPRO_0.05'],
        q95_reconstruction_aupro_error=parity,
        max_map_reconstruction_error=max(x['reconstruction_error'] for x in cached),
        query_count=len(cached), label_use='test_public GT used for optimization AND full-metric candidate selection',
        source='normal-only Q95 baseline; all model distances/banks/pipeline fixed',
        same_class_same_shot_scale_for_all_queries=True,
        deployable_calibration=False, global_optimum_claimed=False)
    (run_root/'oracle_diagnostic.json').write_text(json.dumps(record, indent=2)+'\n')
    # Final metrics and manifest share one explicit oracle identity. Original Q95
    # evidence is preserved separately. No baseline image metric is misattributed.
    metrics = {name: best,
        'mean_segmentation_aupro_0.05': best['seg_AUPRO_0.05'],
        'mean_segmentation_auroc_0.05': best['seg_AUROC_0.05'],
        'mean_segmentation_f1_oracle': best['seg_F1_oracle'],
        'mean_segmentation_auroc_1.00': best['seg_AUROC_1.00'],
        'mean_segmentation_aupro_0.30': best['seg_AUPRO_0.30']}
    obj = manifest['objects'][name]
    obj.update(streaming_metrics=best, oracle_scale_diagnostic=record,
               score_normalization=f'oracle_GT_{primary_mode}_positive_scales',
               score_definition='GT_optimized_observation_scales_same_public_population')
    for target, content in ((run_root/'metrics.json', metrics), (manifest_path, manifest)):
        target.write_text(json.dumps(content, indent=2)+'\n')
    summary = root/'summary.tsv'
    if summary.is_file():
        with summary.open() as fp:
            reader = csv.DictReader(fp, delimiter='\t'); fields=reader.fieldnames; rows=list(reader)
        for row in rows:
            if row['object'] == name:
                for key in ('seg_AUPRO_0.05','seg_AUROC_0.05','seg_F1_oracle','best_thre'):
                    row[key]=str(best[key])
        with summary.open('w') as fp:
            writer=csv.DictWriter(fp,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerows(rows)
    print(f'ORACLE_COMPLETE {name} baseline={record["baseline_aupro"]:.8f} selected={record["selected_aupro"]:.8f}', flush=True)


if __name__ == '__main__':
    main()
