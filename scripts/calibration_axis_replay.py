#!/usr/bin/env python3
"""Controlled log-Q95 factor interventions on fixed, normal-only source banks."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from calibration_mechanism_analysis import aggregate, curve, resolution_maps
from calibration_statistical_replay import SOURCE
from fmad.cli import load_config, parse_args, resolve_runtime_device
from fmad.evaluation.streaming import StreamingSegmentationAccumulator
from fmad.pipeline import _ground_truth_mask
from fmad.postprocess import component_quantile
from fmad.registry import DATASET_REGISTRY

MODES=('none','global','layer','resolution','separable','full')


def factorize(scales):
    """scales[B,L,R]; preserve the same per-bag global factor in every ablation."""
    scales=np.asarray(scales,dtype=np.float64)
    if scales.ndim!=3 or not np.isfinite(scales).all() or not (scales>0).all():
        raise ValueError('positive finite B,L,R source scales required')
    logs=np.log(scales);g=logs.mean(axis=(1,2),keepdims=True)
    layer=logs.mean(axis=2,keepdims=True)-g
    resolution=logs.mean(axis=1,keepdims=True)-g
    interaction=logs-g-layer-resolution
    error=float(np.max(np.abs(logs-(g+layer+resolution+interaction))))
    if error>1e-10:raise ValueError('invalid log-scale identity')
    for term,axis in ((layer,1),(resolution,2),(interaction,1),(interaction,2)):
        if np.max(np.abs(term.mean(axis=axis)))>1e-10:raise ValueError('nonzero marginal axis effect')
    denominators={
        'none':np.ones_like(scales),'global':np.exp(g)*np.ones_like(scales),
        'layer':np.exp(g+layer)*np.ones_like(scales),
        'resolution':np.exp(g+resolution)*np.ones_like(scales),
        'separable':np.exp(g+layer+resolution),'full':scales}
    return denominators,dict(global_log=g.tolist(),layer_log=layer.tolist(),resolution_log=resolution.tolist(),interaction_log=interaction.tolist(),log_identity_max_abs=error)


def intervened_maps(item,denominators,resolutions,device):
    maps=[]
    for r in item['resolutions']:
        values=torch.as_tensor(item[f'raw_{r}'],device=device)
        factors=torch.as_tensor(denominators[:,:,resolutions.index(int(r))],device=device,dtype=values.dtype)
        values=values/factors[:,:,None,None]
        bags=[torch.stack([v[1],v[2]]).mean(0)+.125*v[0] for v in values]
        score=torch.stack(bags).mean(0)
        maps.append(F.interpolate(score[None,None],size=tuple(item['shape']),mode='bilinear',align_corners=False)[0,0])
    return maps


def main():
    config=load_config(parse_args());device=resolve_runtime_device(config['device']);torch.set_num_threads(4)
    dataset=DATASET_REGISTRY.get(config['dataset'])(data_root=config['data_root'],config=config.get('dataset_config',{}))
    objects=dataset.get_objects()
    if len(objects)!=1:raise ValueError('one class per runner cell')
    name=objects[0].name;shot=config['protocol']['shots'];seed=config['seed'];label=f'ad2_mechanism_s{shot:02}_seed{seed}'
    candidates=[]
    for stage in SOURCE.glob('lane_*/'+label):
        for p in stage.rglob('mechanism_cache_index.json'):
            d=json.loads(p.read_text())
            if d['object']==name:candidates.append((p,d))
    if len(candidates)!=1:raise ValueError('missing or ambiguous source index')
    source,index=candidates[0];original=json.loads(source.with_name('calibration_mechanism_analysis.json').read_text())
    for key in ('seed','protocol','method','method_config','backbone','backbone_config','dataset_config'):
        if config.get(key)!=index['config'].get(key):raise ValueError(f'source identity mismatch {key}')
    if original['baseline_parity_abs']>3e-5:raise ValueError('invalid original endpoint')
    mc=config['method_config'];resolutions=list(mc['resolutions'])
    scales=np.stack([np.asarray(original['support_q95'][str(r)]) for r in resolutions],axis=-1)
    denominators,decomposition=factorize(scales)
    out=Path(config['output_dir'])/name;out.mkdir(parents=True,exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(config))
    for filename in ('run_manifest.json','mechanism_cache_index.json'):
        shutil.copy2(source.with_name(filename),out/('source_'+filename))
    accumulators={f'{mode}/{fusion}':StreamingSegmentationAccumulator() for mode in MODES for fusion in ('mean','renyi')}
    rows=[];max_token_error=0.;rng=np.random.default_rng(95723+seed)
    for number,(path,record) in enumerate(index['records'].items()):
        with np.load(source.parent/'mechanism_cache'/record['file']) as archive:item={k:archive[k] for k in archive.files}
        mask=_ground_truth_mask(dataset,name,Path(path).parent.name,path,tuple(item['shape']))
        chosen={}
        for kind,population in [('normal',~mask),('defect',mask)]:
            ids=np.flatnonzero(population);chosen[kind]=rng.choice(ids,min(4096,len(ids)),replace=False)
        row=dict(image=str(Path(path).relative_to(Path(config['data_root']))),normal_image=not bool(mask.any()),resolutions=item['resolutions'].tolist(),excess={})
        for mode in MODES:
            if mode in ('none','full'):
                maps=resolution_maps(item,'none' if mode=='none' else 'q95',device)
                if mode=='full':
                    # Validate source support denominators against the captured Q95 scores.
                    for r in item['resolutions']:
                        expected=np.asarray(item[f'raw_{r}'])/denominators['full'][:,:,resolutions.index(int(r))].astype(np.float32)[:,:,None,None]
                        error=float(np.max(np.abs(expected-item[f'q95_{r}'])));max_token_error=max(max_token_error,error)
                        if error>2e-5:raise ValueError(f'source full token identity failed: {error}')
            else:maps=intervened_maps(item,denominators[mode],resolutions,device)
            mu=aggregate(maps,0);renyi=aggregate(maps,1)
            for kind,ids in chosen.items():
                if not len(ids):row['excess'][mode+'/'+kind]=dict(count=0,relative_mean=None,mu_mean=None,excess_mean=None);continue
                ix=torch.as_tensor(ids,device=device);m=mu.flatten()[ix].cpu().numpy();e=(renyi.flatten()[ix]-mu.flatten()[ix]).cpu().numpy();valid=m>0
                row['excess'][mode+'/'+kind]=dict(count=len(ids),valid_relative_count=int(valid.sum()),relative_mean=float((e[valid]/m[valid]).mean()) if valid.any() else None,mu_mean=float(m.mean()),excess_mean=float(e.mean()))
            for fusion,tensor in [('mean',mu),('renyi',renyi)]:
                prediction=component_quantile(tensor.cpu().numpy(),percentile=mc['component_percentile'],q=mc['component_q'],dilate_size=mc['component_dilate_size'])
                accumulators[mode+'/'+fusion].update([prediction],[mask])
            del maps,mu,renyi,tensor,prediction
        rows.append(row)
        if (number+1)%10==0:print(f'AXIS_REPLAY {name} {number+1}/{len(index["records"])} arms=12',flush=True)
    metrics={};parity={}
    for arm,acc in accumulators.items():
        r=acc.compute();metrics[arm]=dict(seg_AUPRO_0_05=r.seg_aupro_0_05,seg_AUROC_0_05=r.seg_auroc_0_05,image_count=r.prediction_map_count,operating_points=curve(acc))
        mode,fusion=arm.split('/')
        if mode in ('none','full'):
            key=('none' if mode=='none' else 'q95')+('/alpha0' if fusion=='mean' else '/alpha1')
            parity[arm]={metric:abs(metrics[arm][metric]-original['metrics'][key][metric]) for metric in ('seg_AUPRO_0_05','seg_AUROC_0_05')}
            if max(parity[arm].values())>3e-5:raise ValueError(f'endpoint mismatch {arm}: {parity[arm]}')
    report=dict(object=name,shots=shot,seed=seed,source_index=str(source),source_index_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_analysis_sha256=hashlib.sha256(source.with_name('calibration_mechanism_analysis.json').read_bytes()).hexdigest(),metrics=metrics,endpoint_parity_abs=parity,max_full_token_error=max_token_error,decomposition=decomposition,denominators={k:v.tolist() for k,v in denominators.items()},images=rows,completed_at=time.time(),label_use=index['label_use'],interpretation='global retains per-bag scale; axis contrasts are relative to global; full minus separable measures the log-denominator interaction; all diagnostic, no model selection',limits='post-hoc AD2 public; no actual Q95 estimator uncertainty; no isolated bag-axis experiment; real dataset dependence retained')
    p=out/'axis_analysis.json.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.rename(out/'axis_analysis.json')
    print(f'AXIS_ANALYSIS_COMPLETE {name} images={len(rows)} endpoint_parity={max(max(v.values()) for v in parity.values())}',flush=True)


if __name__=='__main__':main()
