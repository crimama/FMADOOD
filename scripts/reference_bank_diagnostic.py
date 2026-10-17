#!/usr/bin/env python3
"""Unchanged Q95 evaluation with post-prediction paired reference diagnosis."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from fmad.cli import parse_args, load_config, resolve_runtime_device
from fmad.registry import BACKBONE_REGISTRY, DATASET_REGISTRY, METHOD_REGISTRY
from fmad.methods import register_methods
from fmad.pipeline import run_pipeline, _ground_truth_mask
from fmad.protocol import seed_generators
from fmad.evaluation.reference_bank_diagnostic import (
    paired_reference_distances, summarize_reference_shift,
)


def main():
    config = load_config(parse_args())
    if config.get('evaluation',{}).get('reference_bank_diagnostic') is not True:
        raise ValueError('requires explicit post-prediction diagnostic option')
    mc = config['method_config']
    if (config['method'] != 'statistical_reference_matching'
            or mc.get('statistical_normalization') != 'q95'
            or mc.get('knn_k',1) != 1 or not mc.get('calibration_audit')):
        raise ValueError('requires audited unchanged Q95 cosine 1NN')
    seed_generators(config['seed'])
    register_methods(config['method'])
    dataset = DATASET_REGISTRY.get(config['dataset'])(
        data_root=config['data_root'], config=config.get('dataset_config',{}))
    objects = dataset.get_objects()
    if len(objects) != 1:
        raise ValueError('one class per runner cell')
    name = objects[0].name
    query_types = {p: kind for kind, paths in dataset.get_test_images(name).items() for p in paths}
    backbone = BACKBONE_REGISTRY.get(config['backbone'])(model_name=config['backbone'],
        device=resolve_runtime_device(config['device']), **config.get('backbone_config',{}))
    method = METHOD_REGISTRY.get(config['method'])(backbone=backbone,config=mc)
    original_score, original_predict = method._score_tokens, method.predict_batch
    captured, pools, normal_queries = [], {}, []
    parity_errors = []
    diagnostic_seconds = 0.

    def capture(layers, grid, bank, *, normalize=True):
        result = original_score(layers,grid,bank,normalize=normalize)
        if normalize:
            # Cache bounded feature probes for one image; no labels enter scoring.
            h,w = grid
            yy,xx = np.meshgrid((np.arange(16)+.5)/16, (np.arange(16)+.5)/16,indexing='ij')
            indices = (np.minimum((yy*h).astype(int),h-1)*w+np.minimum((xx*w).astype(int),w-1)).ravel()
            probes = [layer[torch.as_tensor(indices,device=layer.device)].detach().clone() for layer in layers]
            captured.append((bank,probes,indices))
        return result

    def diagnose(paths):
        nonlocal diagnostic_seconds
        predictions = []
        for path in paths:
            captured.clear()
            prediction = original_predict([path])[0]
            # Labels are first consulted after this unchanged prediction returns.
            mask = _ground_truth_mask(dataset,name,query_types[path],path,prediction.score_map.shape)
            if not mask.any():
                started = time.perf_counter()
                audit = prediction.diagnostics['calibration_audit']
                if len(captured) != len(audit['resolutions']):
                    raise RuntimeError('incomplete cached feature probes')
                normal_queries.append('/'.join(Path(path).parts[-2:]))
                for ri,(r,entry) in enumerate(zip(audit['resolutions'],captured)):
                    banks,probes,indices = entry
                    if not isinstance(banks,list):
                        banks=[banks]
                    for bi,bank in enumerate(banks):
                        source_count=len(set(bank['owners']))
                        for li,layer in enumerate(mc['feature_layers']):
                            with torch.no_grad():
                                full,excluded=paired_reference_distances(probes[li],bank['layers'][li],
                                    bank['owners'],source_count)
                            full,excluded=full.cpu().numpy(),excluded.cpu().numpy()
                            original=audit['native_bags'][ri][bi,0,li].reshape(-1)[indices]
                            error=float(np.max(np.abs(full-original)))
                            parity_errors.append(error)
                            if error>2e-5:
                                raise RuntimeError(f'full-bank diagnostic parity error {error}')
                            key=f'r{r}/b{bi}/l{layer}'
                            pool=pools.setdefault(key,dict(full=[],excluded=[],
                                support_q95=float(method.bag_tails[r][bi][li]['statistical']['scale']),
                                support_raw_q95=float(method.bag_tails[r][bi][li]['u'])))
                            pool['full'].append(full)
                            pool['excluded'].append(excluded)
                diagnostic_seconds+=time.perf_counter()-started
            captured.clear()
            predictions.append(prediction)
        return predictions

    method._score_tokens=capture
    method.predict_batch=diagnose
    root=Path(config['output_dir'])
    previous=set(root.rglob('run_manifest.json'))
    run_pipeline(method,dataset,backbone,config,config['output_dir'])
    manifests=list(set(root.rglob('run_manifest.json'))-previous)
    if len(manifests)!=1 or not normal_queries:
        raise RuntimeError('missing unique manifest or normal diagnostic queries')
    observations={k:summarize_reference_shift(v['full'],v['excluded'],v['support_q95'])
                  for k,v in pools.items()}
    for key,value in observations.items():
        value['support_raw_q95']=pools[key]['support_raw_q95']
        value['support_divisor_clamped']=pools[key]['support_q95']!=pools[key]['support_raw_q95']
    expected=len(mc['resolutions'])*mc['bank_bags']*len(mc['feature_layers'])
    if len(observations)!=expected:
        raise RuntimeError('routing changed diagnostic observation scope')
    record=dict(observations=observations,normal_query_identities=normal_queries,
        normal_query_identity_sha256=hashlib.sha256(json.dumps(normal_queries).encode()).hexdigest(),
        normal_query_count=len(normal_queries), support_count=config['protocol']['shots'],
        full_distance_max_parity_error=max(parity_errors),diagnostic_seconds=diagnostic_seconds,
        label_use='GT only after prediction to select wholly normal diagnostic images',
        method_fit_or_prediction_changed=False,deployable_calibration=False,
        scope='seen AD2 public development; native token probes before interpolation or bag averaging',
        interpretation='paired full vs S-1 references isolates reference removal on same query; S-1 vs support also includes population and sampling differences',
        inference_timing_note='wrapper overhead includes diagnostics; not a computational-cost benchmark')
    manifest_path=manifests[0]
    manifest=json.loads(manifest_path.read_text())
    manifest['objects'][name]['reference_bank_diagnostic']=record
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'REFERENCE_DIAGNOSTIC_COMPLETE {name} normal={len(normal_queries)} observations={len(observations)} parity={max(parity_errors):.3g}',flush=True)


if __name__=='__main__':
    main()
