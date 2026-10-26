#!/usr/bin/env python3
"""Reanalyze existing paired samples and detector traces; no new predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CLASSES=('can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts')
SOURCE=Path('/var/tmp/fsad-analysis/scopead_response_trace_20261020')


def mean(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def describe(mask,raw_mu,raw_e,cal_mu,cal_e,raw_ref,cal_ref):
    n=int(mask.sum())
    if not n:return dict(count=0,relative_delta_pp=None,reference_scaled_delta_pp=None)
    m0,e0,m1,e1=raw_mu[mask],raw_e[mask],cal_mu[mask],cal_e[mask]
    valid=(m0>0)&(m1>0)
    return dict(count=n,valid_relative_count=int(valid.sum()),
        relative_delta_pp=float(np.mean(e1[valid]/m1[valid]-e0[valid]/m0[valid])*100) if valid.any() else None,
        reference_scaled_delta_pp=float(np.mean(e1/cal_ref-e0/raw_ref)*100),
        raw_mu_over_reference=float(np.mean(m0/raw_ref)),q95_mu_over_reference=float(np.mean(m1/cal_ref)))


def effect(values,rng):
    pairs=[(k,v) for k,v in values.items() if v is not None and np.isfinite(v)]
    if not pairs:return dict(mean=None,ci95=None,classes=0,per_class={})
    x=np.array([v for k,v in pairs]);draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist() if len(x)>1 else None,classes=len(x),per_class=dict(pairs),without_fabric=mean([v for k,v in pairs if k!='fabric']))


def diagnosis(images):
    output={};types={}
    for direction in ('none/mean_to_renyi','q95/mean_to_renyi','mean/none_to_q95','renyi/none_to_q95'):
        output[direction]={}
        for f in ('.005','.01','.05'):
            target=str(float(f));cells=[r['transitions'][direction+'/post/'+target] for r in images]
            normal=sum(c['normal_pixels'] for c in cells);defect=sum(c['defect_pixels'] for c in cells)
            output[direction][target]=dict(normal_gain=sum(c['normal_gain'] for c in cells),normal_loss=sum(c['normal_loss'] for c in cells),defect_gain=sum(c['defect_gain'] for c in cells),defect_loss=sum(c['defect_loss'] for c in cells),normal_pixels=normal,defect_pixels=defect,
                net_fp_pp=(sum(c['normal_gain']-c['normal_loss'] for c in cells)/normal*100) if normal else None,
                net_tp_pp=(sum(c['defect_gain']-c['defect_loss'] for c in cells)/defect*100) if defect else None)
    for row in images:
        for r in row['regions']:
            t=types.setdefault(r['defect_type'],dict(region_areas=[],image_effects={},interaction_images={},concentration=[],size_bins={k:0 for k in ('small','medium','large')}))
            t['region_areas'].append(r['area_fraction']);t['concentration'].append(r['scores']['q95']['max_resolution_share']);t['size_bins'][r['size_bin']]+=1
            effect0=(r['coverage']['q95_renyi/post/0.01']-r['coverage']['none_renyi/post/0.01'])*100
            inter=(r['coverage']['q95_renyi/post/0.01']-r['coverage']['q95_mean/post/0.01']-r['coverage']['none_renyi/post/0.01']+r['coverage']['none_mean/post/0.01'])*100
            t['image_effects'].setdefault(row['image'],[]).append(effect0);t['interaction_images'].setdefault(row['image'],[]).append(inter)
    compact={}
    for k,t in types.items():
        compact[k]=dict(regions=len(t['region_areas']),images=len(t['image_effects']),median_area_fraction=float(np.median(t['region_areas'])),q95_resolution_concentration=mean(t['concentration']),size_bins=t['size_bins'],image_mean_coverage_delta_pp=mean([mean(v) for v in t['image_effects'].values()]),image_mean_interaction_pp=mean([mean(v) for v in t['interaction_images'].values()]))
    return dict(transitions=output,defect_types=compact,limits='same fixed normal thresholds; actual FPR differs; paired image/region description, no new fitting')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--export-root',type=Path);args=parser.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=False)
    paths=sorted(SOURCE.rglob('response_analysis.json'));identities=set();reports=[]
    for path in paths:
        source=json.loads(path.read_text());identity=(source['shots'],source['seed'],source['object'])
        if identity in identities:raise ValueError('duplicate source')
        identities.add(identity)
        if max(max(v.values()) for v in source['metric_parity_abs'].values())>3e-5:raise ValueError('invalid source parity')
        rows=[]
        for i,row in enumerate(source['images']):
            entry=dict(image=row['image'],normal_image=row['normal_image'],fold=row['fold'],groups={},paired_contrast={})
            with np.load(path.parent/'samples'/f'{i:04}.npz') as sample:
                rawref=source['normal_reference_by_fold']['none'][str(row['fold'])];calref=source['normal_reference_by_fold']['q95'][str(row['fold'])]
                if min(rawref['mu_q95'],calref['mu_q95'])<=0:raise ValueError('nonpositive source normal reference')
                for kind in ('normal','defect'):
                    group=('normal_image' if row['normal_image'] else 'background') if kind=='normal' else 'defect'
                    arrays=[sample[norm+'_'+kind+'_'+key] for norm in ('none','q95') for key in ('mu','excess')]
                    if len({len(v) for v in arrays})!=1:raise ValueError('unpaired sample lengths')
                    if len(arrays[0])!=row['distribution']['none/'+kind]['mean']['count']:raise ValueError('sample count provenance mismatch')
                    bins=np.searchsorted(rawref['mu_edges'],arrays[0],side='right')
                    # Crucial: exactly the same raw-mu membership for both score conditions.
                    entry['groups'][group]={str(j):describe(bins==j,*arrays,rawref['mu_q95'],calref['mu_q95']) for j in range(10)}
                    entry['groups'][group]['all']=describe(np.ones(len(arrays[0]),dtype=bool),*arrays,rawref['mu_q95'],calref['mu_q95'])
            if not row['normal_image']:
                for j in ['all']+[str(k) for k in range(10)]:
                    a=entry['groups']['background'][j]['relative_delta_pp'];b=entry['groups']['defect'][j]['relative_delta_pp']
                    entry['paired_contrast'][j]=b-a if a is not None and b is not None else None
            rows.append(entry)
        report=dict(object=source['object'],shots=source['shots'],seed=source['seed'],source_analysis=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),conditioning='both arms use identical raw-mu bins from other-fold normal reference; not equal post-Q95 mu',images=rows,diagnosis=diagnosis(source['images']))
        target=args.output_root/f's{source["shots"]:02}_seed{source["seed"]}_{source["object"]}'/'conditional_analysis.json';target.parent.mkdir();target.write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
        print(f'CONDITIONAL_COMPLETE shot={source["shots"]} seed={source["seed"]} object={source["object"]}',flush=True)
    expected={(s,k,n) for s in (2,16) for k in range(3) for n in CLASSES}
    if identities!=expected:raise ValueError('full48 required')
    rng=np.random.default_rng(10212026);result=dict(scope='full8 ×2/16-shot ×3-seed; identical baseline raw-mu stratum membership',shots={},limits='baseline-mu matching, not equal calibrated mu; empty strata missing; descriptive post-hoc, fixed support/threshold; no new predictions')
    lines=['# 평균 응답 구간 및 클래스별 검출 변화 — 사후 재분석','','두 조건을 동일한 원점수 평균 구간과 대응 픽셀로 비교한다. 수치는 상대 추가 응답 변화(pp)이며 빈 구간은 missing이다.','']
    for shot in (2,16):
        groups={};contrasts={}
        for group in ('normal_image','background','defect'):
            groups[group]={}
            for j in ['all']+[str(k) for k in range(10)]:
                vals={};counts={};scaled={}
                for name in CLASSES:
                    seedvalues=[];seedscaled=[];pixelcount=0;imagecount=0
                    for seed in range(3):
                        d=next(d for d in reports if (d['shots'],d['seed'],d['object'])==(shot,seed,name));cells=[r['groups'][group][j] for r in d['images'] if group in r['groups'] and r['groups'][group][j]['count']>0]
                        v=mean([c['relative_delta_pp'] for c in cells]);w=mean([c['reference_scaled_delta_pp'] for c in cells]);pixelcount+=sum(c['count'] for c in cells);imagecount+=len(cells)
                        if v is not None:seedvalues.append(v)
                        if w is not None:seedscaled.append(w)
                    vals[name]=mean(seedvalues);scaled[name]=mean(seedscaled);counts[name]=dict(available_seeds=len(seedvalues),image_seed_cells=imagecount,sampled_pixels=pixelcount)
                groups[group][j]=dict(relative_delta=effect(vals,rng),reference_scaled_delta=effect(scaled,rng),counts=counts)
        for j in ['all']+[str(k) for k in range(10)]:
            vals={}
            for name in CLASSES:
                seeds=[]
                for seed in range(3):
                    d=next(d for d in reports if (d['shots'],d['seed'],d['object'])==(shot,seed,name));v=mean([r['paired_contrast'].get(j) for r in d['images'] if not r['normal_image']])
                    if v is not None:seeds.append(v)
                vals[name]=mean(seeds)
            contrasts[j]=effect(vals,rng)
        result['shots'][str(shot)]=dict(groups=groups,defect_minus_background_delta=contrasts)
        lines += [f'## {shot}-shot','','| Raw mu bin | Normal Δ relative excess | Defect Δ relative excess | Defect−background Δ | classes |','|---|---:|---:|---:|---:|']
        fmt=lambda v:f'{v:+.4f}' if v is not None else '—'
        for j in ['all']+[str(k) for k in range(10)]:
            lines.append(f'| {j} | {fmt(groups["normal_image"][j]["relative_delta"]["mean"])} | {fmt(groups["defect"][j]["relative_delta"]["mean"])} | {fmt(contrasts[j]["mean"])} | {contrasts[j]["classes"]} |')
    (args.output_root/'conditional_summary.json').write_text(json.dumps(result,indent=2)+'\n');(args.output_root/'CONDITIONAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    if args.export_root:
        for p in args.output_root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.md'):
                t=args.export_root/p.relative_to(args.output_root);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t);t.chmod(0o644)
    print('CONDITIONAL_SUMMARY_COMPLETE full48',flush=True)


if __name__=='__main__':main()
