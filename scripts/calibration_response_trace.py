#!/usr/bin/env python3
"""Fixed-method replay: Rényi excess, paired detector transitions, GT regions.
Labels enter post-hoc diagnostics only; no method fitting or model selection.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scipy import ndimage
import torch
import yaml

from calibration_mechanism_analysis import aggregate, resolution_maps
from calibration_statistical_replay import SOURCE, sparse_hist, add_sparse, threshold_code
from fmad.cli import load_config, parse_args, resolve_runtime_device
from fmad.evaluation.streaming import (StreamingSegmentationAccumulator,
    _ordered_float16_codes, _code_to_float16, _label8)
from fmad.pipeline import _ground_truth_mask
from fmad.postprocess import component_quantile
from fmad.registry import DATASET_REGISTRY

STAT = Path('/var/tmp/fsad-analysis/scopead_statistical_20261020')
NORMS = ('none', 'q95')
FPRS = (.005, .01, .05)
SOURCE_KEYS = dict(none_mean='raw_mean', none_renyi='raw_rank_raw_value',
                   q95_mean='q95_mean', q95_renyi='q95_rank_q95_value')


def decompose(maps):
    """Exact ascending gap identity for any available resolution count."""
    ordered = torch.sort(maps, dim=0).values
    n = len(maps)
    weights = maps.new_tensor([sum(1/(n-j) for j in range(k)) for k in range(1,n+1)])
    weights /= weights.sum()
    coefficients = torch.stack([(weights[k+1:]-1/n).sum() for k in range(n-1)]) if n>1 else maps.new_empty(0)
    terms = (ordered[1:]-ordered[:-1])*coefficients[:,None,None]
    mu = maps.mean(0)
    excess = terms.sum(0) if n>1 else torch.zeros_like(mu)
    direct = aggregate(list(maps), 1)
    error = float((mu+excess-direct).abs().max())
    if error>2e-5 or float(excess.min()) < -2e-6:
        raise ValueError(f'gap decomposition mismatch {error}')
    # Direct implementation is the endpoint; identity is verified separately.
    return mu, direct-mu, direct, terms, error


def postprocess(score, mc):
    return component_quantile(score, percentile=mc['component_percentile'],
        q=mc['component_q'], dilate_size=mc['component_dilate_size'])


def quantile_from_hist(hist, q):
    return _code_to_float16(int(np.searchsorted(np.cumsum(hist), q*hist.sum(),side='left')))


def summary_values(values):
    values=np.asarray(values,dtype=np.float64)
    values=values[np.isfinite(values)]
    if not len(values): return dict(count=0,mean=None,q50=None,q95=None,q99=None)
    qs=np.quantile(values,[.5,.95,.99])
    return dict(count=len(values),mean=float(values.mean()),q50=float(qs[0]),q95=float(qs[1]),q99=float(qs[2]))


def region_geometry(mask, path):
    labels,count=_label8(mask)
    gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if gray is None: raise ValueError(f'cannot read {path}')
    if gray.shape!=mask.shape: raise ValueError('image/mask native shape mismatch')
    gray=gray.astype(np.float64)/255
    rows=[]
    for label,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None: raise ValueError('noncontiguous region labels')
        ys,xs=sl
        y0,y1=max(0,ys.start-8),min(mask.shape[0],ys.stop+8)
        x0,x1=max(0,xs.start-8),min(mask.shape[1],xs.stop+8)
        region=labels[y0:y1,x0:x1]==label
        expanded=cv2.dilate(region.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))).astype(bool)
        ring=expanded & ~mask[y0:y1,x0:x1]
        values=gray[y0:y1,x0:x1]
        region_mean=float(values[region].mean())
        ring_mean=float(values[ring].mean()) if ring.any() else None
        ring_sd=float(values[ring].std()) if ring.any() else None
        contrast=(region_mean-ring_mean) if ring_mean is not None else None
        size=int(region.sum()); fraction=size/mask.size
        rows.append(dict(region_id=label,area_pixels=size,area_fraction=fraction,
            size_bin='small' if fraction<.001 else ('medium' if fraction<.01 else 'large'),
            bbox=[xs.start,ys.start,xs.stop,ys.stop],defect_type=Path(path).parent.name,
            grayscale_contrast=contrast,standardized_abs_contrast=abs(contrast)/ring_sd if ring_sd is not None and ring_sd>0 else None,
            ring_pixels=int(ring.sum()),scores={},coverage={}))
    return labels,count,rows


def cover(labels,count,prediction):
    return np.bincount(labels.ravel(),weights=prediction.ravel().astype(float),minlength=count+1)[1:]


def transition(a,b,mask):
    gained=b & ~a; lost=a & ~b
    return dict(defect_gain=int((gained & mask).sum()),defect_loss=int((lost & mask).sum()),
        normal_gain=int((gained & ~mask).sum()),normal_loss=int((lost & ~mask).sum()),
        defect_pixels=int(mask.sum()),normal_pixels=int((~mask).sum()))


def main():
    config=load_config(parse_args()); device=resolve_runtime_device(config['device'])
    torch.set_num_threads(4)
    dataset=DATASET_REGISTRY.get(config['dataset'])(data_root=config['data_root'],config=config.get('dataset_config',{}))
    objects=dataset.get_objects()
    if len(objects)!=1: raise ValueError('one class per runner cell')
    name=objects[0].name; seed=config['seed']; shots=config['protocol']['shots']
    label=f'ad2_mechanism_s{shots:02}_seed{seed}'
    matches=[]
    for stage in SOURCE.glob('lane_*/'+label):
        for p in stage.rglob('mechanism_cache_index.json'):
            info=json.loads(p.read_text())
            if info['object']==name: matches.append((p,info))
    if len(matches)!=1: raise ValueError(f'ambiguous source cache {name} {label}')
    source,info=matches[0]
    for key in ('seed','protocol','method','method_config','backbone','backbone_config','dataset_config'):
        if config.get(key)!=info['config'].get(key): raise ValueError(f'source config mismatch {key}')
    previous=list(STAT.glob(f'gpu_*/lane_0/{label}/{name}/statistical_analysis.json'))
    if len(previous)!=1: raise ValueError('missing/ambiguous completed statistical source')
    previous_path=previous[0]; prior=json.loads(previous_path.read_text())
    if prior['source_index_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest(): raise ValueError('cache index hash mismatch')
    if max(prior['metric_parity_abs'].values())>3e-5 or prior['crossfit']['status']!='complete': raise ValueError('invalid predecessor')
    out=Path(config['output_dir'])/name;out.mkdir(parents=True,exist_ok=False)
    (out/'samples').mkdir(); (out/'pre_histograms').mkdir()
    (out/'config.yaml').write_text(yaml.safe_dump(config))
    for filename in ('run_manifest.json','mechanism_cache_index.json'):
        shutil.copy2(source.with_name(filename),out/('source_'+filename))
    mc=config['method_config']; rows=[];totals={}; max_gap_error=0.
    prior_images={row['image']:row for row in prior['images']}
    cache=source.parent/'mechanism_cache'
    arms=list(SOURCE_KEYS)
    for arm in arms:
        totals[arm]=dict(pre=StreamingSegmentationAccumulator(),post=StreamingSegmentationAccumulator())
        (out/'pre_histograms'/arm).mkdir()
    rng=np.random.default_rng(80267+seed)
    # Pass 1: native endpoint validation and normal-image reference histograms.
    for i,(path,record) in enumerate(info['records'].items()):
        with np.load(cache/record['file']) as archive: item={k:archive[k] for k in archive.files}
        mask=_ground_truth_mask(dataset,name,Path(path).parent.name,path,tuple(item['shape']))
        image=str(Path(path).relative_to(Path(config['data_root'])))
        previous_row=prior_images[image]
        selected={}
        for kind,population,cap in (('normal',~mask,8192),('defect',mask,4096)):
            ids=np.flatnonzero(population);selected[kind]=rng.choice(ids,min(len(ids),cap),replace=False)
        row=dict(image=image,normal_image=not bool(mask.any()),fold=previous_row['crossfit_fold'],resolutions=item['resolutions'].tolist(),distribution={},transitions={},margin_terms={},regions=[])
        sample={}
        for norm in NORMS:
            maps=torch.stack(resolution_maps(item,norm,device))
            mu,e,r,terms,error=decompose(maps);max_gap_error=max(max_gap_error,error)
            for kind,ids in selected.items():
                index=torch.as_tensor(ids,device=device)
                for key,tensor in [('mu',mu),('excess',e)]:sample[f'{norm}_{kind}_{key}']=tensor.flatten()[index].cpu().numpy()
                for gap in range(len(terms)):sample[f'{norm}_{kind}_gap{gap}']=terms[gap].flatten()[index].cpu().numpy()
            for mode,tensor in [('mean',mu),('renyi',r)]:
                arm=norm+'_'+mode;pre=tensor.cpu().numpy();post=postprocess(pre,mc)
                acc=StreamingSegmentationAccumulator();acc.update([pre],[mask])
                sparse_hist(acc,out/'pre_histograms'/arm/f'{i:04}.npz')
                totals[arm]['pre'].update([pre],[mask]);totals[arm]['post'].update([post],[mask])
            del maps,mu,e,r,terms,tensor,pre,post
        np.savez_compressed(out/'samples'/f'{i:04}.npz',**sample)
        rows.append(row)
        if (i+1)%10==0: print(f'RESPONSE_CAPTURE {name} {i+1}/{len(info["records"])}',flush=True)
    metrics={};parity={};thresholds={};references={}
    folds=prior['crossfit']['folds']
    for arm in arms:
        metrics[arm]={}
        for stage,acc in totals[arm].items():
            result=acc.compute();metrics[arm][stage]=dict(seg_AUPRO_0_05=result.seg_aupro_0_05,seg_AUROC_0_05=result.seg_auroc_0_05)
        parity[arm]={key:abs(metrics[arm]['post'][key]-prior['metrics'][SOURCE_KEYS[arm]][key]) for key in ('seg_AUPRO_0_05','seg_AUROC_0_05')}
        if max(parity[arm].values())>3e-5: raise ValueError(f'endpoint metric mismatch {arm}: {parity[arm]}')
        thresholds[arm]=dict(pre={},post={})
        for fold in range(folds):
            cal=StreamingSegmentationAccumulator()
            for i,row in enumerate(rows):
                if row['normal_image'] and row['fold']!=fold:add_sparse(cal,out/'pre_histograms'/arm/f'{i:04}.npz')
            thresholds[arm]['pre'][fold]={str(f):threshold_code(cal.negative,f) for f in FPRS}
            thresholds[arm]['post'][fold]={str(f):prior['crossfit']['arms'][SOURCE_KEYS[arm]][str(f)]['thresholds'][fold] for f in FPRS}
            if arm.endswith('_mean'):
                references.setdefault(arm[:-5],{})[fold]=dict(mu_q95=quantile_from_hist(cal.negative,.95),mu_edges=[quantile_from_hist(cal.negative,q) for q in np.arange(.1,1,.1)])
    # Pass 2: identical held-out thresholds, regions, masks and transition locations.
    validation_counts={arm:{str(f):dict(fp=0,tp=0,negatives=0,positives=0,pro_mass=0.,regions=0) for f in FPRS} for arm in arms}
    max_margin_error=0.
    for i,(path,record) in enumerate(info['records'].items()):
        row=rows[i];fold=row['fold']
        with np.load(cache/record['file']) as archive:item={k:archive[k] for k in archive.files}
        mask=_ground_truth_mask(dataset,name,Path(path).parent.name,path,tuple(item['shape']))
        labels,count,regions=region_geometry(mask,path);sizes=np.bincount(labels.ravel(),minlength=count+1)[1:]
        with np.load(out/'samples'/f'{i:04}.npz') as sample:
            for norm in NORMS:
                ref=references[norm][fold]
                for kind in ('normal','defect'):
                    mu=sample[f'{norm}_{kind}_mu'];e=sample[f'{norm}_{kind}_excess'];valid=mu>0
                    bins=np.searchsorted(ref['mu_edges'],mu,side='right')
                    relative=np.divide(e,mu,out=np.full_like(e,np.nan),where=valid)
                    row['distribution'][norm+'/'+kind]=dict(mean=summary_values(mu),excess=summary_values(e),relative_excess=summary_values(relative),reference_scaled_excess=summary_values(e/ref['mu_q95']) if ref['mu_q95']>0 else None,
                        conditional=[dict(bin=j,mu=summary_values(mu[bins==j]),relative_excess=summary_values(relative[bins==j]),reference_scaled_excess=summary_values(e[bins==j]/ref['mu_q95']) if ref['mu_q95']>0 else None) for j in range(10)])
        # Boolean detectors retained only for this image to compare raw vs Q95.
        detectors={}
        for norm in NORMS:
            maps=torch.stack(resolution_maps(item,norm,device))
            mu,e,r,terms,error=decompose(maps);max_gap_error=max(max_gap_error,error)
            pre_mu=mu.cpu().numpy();excess=e.cpu().numpy();pre_r=r.cpu().numpy()
            for region in regions:
                rid=region['region_id'];sl=tuple(slice(a,b) for a,b in [(region['bbox'][1],region['bbox'][3]),(region['bbox'][0],region['bbox'][2])])
                region_mask=labels[sl]==rid
                means=[float(v[sl].cpu().numpy()[region_mask].mean()) for v in maps]
                region['scores'][norm]=dict(mu_mean=float(pre_mu[sl][region_mask].mean()),excess_mean=float(excess[sl][region_mask].mean()),relative_excess=float(excess[sl][region_mask].mean()/pre_mu[sl][region_mask].mean()) if pre_mu[sl][region_mask].mean()>0 else None,resolution_means=means,dominant_resolution=int(item['resolutions'][int(np.argmax(means))]),max_resolution_share=float(max(means)/sum(means)) if sum(means)>0 else None)
            del maps,mu,e,r,terms
            post_mu=postprocess(pre_mu,mc);post_r=postprocess(pre_r,mc)
            codes={('mean','pre'):_ordered_float16_codes(pre_mu),('renyi','pre'):_ordered_float16_codes(pre_r),('mean','post'):_ordered_float16_codes(post_mu),('renyi','post'):_ordered_float16_codes(post_r)}
            for stage in ('pre','post'):
                for f in FPRS:
                    target=str(f)
                    for mode in ('mean','renyi'):
                        arm=norm+'_'+mode;code=thresholds[arm][stage][fold][target]
                        pred=codes[(mode,stage)].astype(np.int32)>=code
                        detectors[(arm,stage,target)]=pred
                        hits=cover(labels,count,pred)
                        for region,hit,size in zip(regions,hits,sizes):region['coverage'][f'{arm}/{stage}/{target}']=float(hit/size)
                        if stage=='post':
                            cell=validation_counts[arm][target];cell['tp']+=int((pred & mask).sum());cell['fp']+=int((pred & ~mask).sum());cell['positives']+=int(mask.sum());cell['negatives']+=int((~mask).sum());cell['regions']+=count;cell['pro_mass']+=float(np.sum(hits/sizes))
                    a=detectors[(norm+'_mean',stage,target)];b=detectors[(norm+'_renyi',stage,target)]
                    row['transitions'][f'{norm}/mean_to_renyi/{stage}/{target}']=transition(a,b,mask)
                    if stage=='post' and f==.01:
                        tm=thresholds[norm+'_mean'][stage][fold][target];tr=thresholds[norm+'_renyi'][stage][fold][target]
                        if max(tm,tr)>=65536: raise ValueError('no finite threshold for margin analysis')
                        threshold_delta=_code_to_float16(tr)-_code_to_float16(tm)
                        cq_delta=(post_r-pre_r)-(post_mu-pre_mu)
                        quant_delta=(post_r.astype(np.float16).astype(np.float32)-post_r)-(post_mu.astype(np.float16).astype(np.float32)-post_mu)
                        observed=(post_r.astype(np.float16).astype(np.float32)-_code_to_float16(tr))-(post_mu.astype(np.float16).astype(np.float32)-_code_to_float16(tm))
                        reconstructed=excess+cq_delta+quant_delta-threshold_delta
                        error=float(np.max(np.abs(observed-reconstructed)));max_margin_error=max(max_margin_error,error)
                        if error>2e-5: raise ValueError(f'margin decomposition mismatch {error}')
                        for kind,population in [('normal',~mask),('defect',mask)]:
                            for change,pixels in [('gained',b & ~a),('lost',a & ~b)]:
                                selection=population & pixels;n=int(selection.sum())
                                row['margin_terms'][f'{norm}/{kind}/{change}']=dict(count=n,threshold_delta=threshold_delta,
                                    excess_mean=float(excess[selection].mean()) if n else None,
                                    cq_delta_mean=float(cq_delta[selection].mean()) if n else None,
                                    quantization_delta_mean=float(quant_delta[selection].mean()) if n else None,
                                    margin_change_mean=float(observed[selection].mean()) if n else None)
            del pre_mu,pre_r,excess,post_mu,post_r,codes
        for stage in ('pre','post'):
            for mode in ('mean','renyi'):
                for f in FPRS:
                    target=str(f);a=detectors[('none_'+mode,stage,target)];b=detectors[('q95_'+mode,stage,target)]
                    row['transitions'][f'{mode}/none_to_q95/{stage}/{target}']=transition(a,b,mask)
        row['regions']=regions
        del detectors,item,mask,labels
        if (i+1)%10==0:print(f'RESPONSE_TRACE {name} {i+1}/{len(rows)}',flush=True)
    crossfit_parity={}
    for arm in arms:
        crossfit_parity[arm]={}
        for f in FPRS:
            target=str(f);cell=validation_counts[arm][target];expected=prior['crossfit']['arms'][SOURCE_KEYS[arm]][target]
            differences={key:abs(cell[key]-expected[key]) for key in cell}
            if any(differences[k] for k in ('fp','tp','negatives','positives','regions')) or differences['pro_mass']>1e-7:
                raise ValueError(f'crossfit count/mass mismatch {arm} {target} {differences}')
            crossfit_parity[arm][target]=differences
    report=dict(object=name,seed=seed,shots=shots,source_index=str(source),source_index_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),statistical_source=str(previous_path),statistical_source_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),metrics=metrics,metric_parity_abs=parity,crossfit_count_mass_parity=crossfit_parity,gap_identity_max_abs=max_gap_error,margin_identity_max_abs=max_margin_error,normal_reference_by_fold=references,threshold_codes=thresholds,images=rows,completed_at=time.time(),label_use='AD2 public seen development GT for post-hoc masks, normal references and region descriptors; no fitting/prediction changes',limits='image/region association only; frozen normal-fold thresholds; sampled distributions; grayscale contrast proxy; CQ decomposition pre/post explicitly separated')
    temp=out/'response_analysis.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.rename(out/'response_analysis.json')
    print(f'RESPONSE_ANALYSIS_COMPLETE {name} images={len(rows)} gap_error={max_gap_error} margin_error={max_margin_error}',flush=True)


if __name__=='__main__':main()
