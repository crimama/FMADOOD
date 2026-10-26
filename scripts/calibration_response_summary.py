#!/usr/bin/env python3
"""Summarize fixed-method excess/transition diagnostics, with cluster uncertainty."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import spearmanr

CLASSES=('can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts')


def effect(values,rng):
    pairs=[(name,float(value)) for name,value in values.items() if value is not None and np.isfinite(value)]
    if not pairs:return dict(mean=None,ci95=None,classes=0,per_class={})
    x=np.array([p[1] for p in pairs]);draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist() if len(x)>1 else None,
        classes=len(x),positive_classes=int((x>0).sum()),per_class=dict(pairs),
        without_fabric=float(np.mean([v for n,v in pairs if n!='fabric'])) if any(n!='fabric' for n,v in pairs) else None)


def safe_mean(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def image_effect_bootstrap(per_class,rng):
    """Each image effect already averages its three paired support seeds."""
    observed=[];samples=[]
    for name,values in per_class.items():
        values=np.array(values,dtype=float);values=values[np.isfinite(values)]
        if not len(values):continue
        observed.append(values.mean())
        samples.append(values[rng.integers(0,len(values),(2000,len(values)))].mean(1))
    return dict(mean=float(np.mean(observed)),ci95=np.quantile(np.mean(samples,axis=0),[.025,.975]).tolist(),
        classes=len(observed),condition='fixed support, thresholds, observed classes; paired images shared across seeds/arms') if observed else None


def region_effect(row,stage,size=None):
    regs=[r for r in row['regions'] if size is None or r['size_bin']==size]
    if not regs:return None
    # Interaction in per-region coverage, not a method selection experiment.
    return float(np.mean([(r['coverage'][f'q95_renyi/{stage}/0.01']-r['coverage'][f'q95_mean/{stage}/0.01'])-
                          (r['coverage'][f'none_renyi/{stage}/0.01']-r['coverage'][f'none_mean/{stage}/0.01']) for r in regs])*100)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--export-root',type=Path)
    args=parser.parse_args();paths=sorted(args.root.rglob('response_analysis.json'))
    ds=[json.loads(p.read_text()) for p in paths];idx={(d['shots'],d['seed'],d['object']):d for d in ds}
    expected={(s,k,n) for s in (2,16) for k in range(3) for n in CLASSES}
    if len(ds)!=48 or set(idx)!=expected:raise ValueError(f'full48 required, got {len(ds)}')
    for d in ds:
        if max(max(x.values()) for x in d['metric_parity_abs'].values())>3e-5:raise ValueError('endpoint parity invalid')
        if max(d['gap_identity_max_abs'],d['margin_identity_max_abs'])>2e-5:raise ValueError('identity invalid')
        if any(x[k] for arm in d['crossfit_count_mass_parity'].values() for x in arm.values() for k in ('tp','fp','positives','negatives','regions')):raise ValueError('crossfit count mismatch')
    rng=np.random.default_rng(10062026)
    result=dict(scope='AD2 public seen development full8, 2/16-shot × 3 support seeds; fixed method',
        validation=dict(completed=48,max_gap_error=max(d['gap_identity_max_abs'] for d in ds),max_margin_error=max(d['margin_identity_max_abs'] for d in ds)),
        estimand='equal image means within class; average paired support seeds then equal-class macro; region/pixel pooled metrics remain in source reports',shots={},source_paths=[str(p) for p in paths])
    lines=['# Rényi 추가 응답과 검출 변화 — 통합 분석','',
       '최종 설정 고정, 기존 캐시 재평가. 48개 실행의 endpoint 지표·교차검증 count 및 gap/margin 분해를 검증했다.', '',
       '표본 분포는 정상 최대8192/결함 최대4096픽셀. 픽셀을 독립 재표집하지 않는다. CQ 전 E=Rényi−평균의 성질과 CQ 후 검출 변화를 구분한다.', '',
       '| Shot | 결함−배경 상대 추가 응답 대비의 Q95−none 차이 | 클래스 bootstrap 95% | 정상 이미지 E/mu Q99의 Q95−none | 검출 coverage 상호작용 (%p) | 클래스 bootstrap 95% |',
       '|---|---:|---|---:|---:|---|']
    for shot in (2,16):
        perclass={};tail={};coverage={};image_h1={};image_h2={};size_effect={size:{} for size in ('small','medium','large')};association={}
        for name in CLASSES:
            runs=[idx[(shot,seed,name)] for seed in range(3)]
            by_seed=[{r['image']:r for r in d['images']} for d in runs]
            if any(set(by_seed[0])!=set(x) for x in by_seed[1:]):raise ValueError('query identity mismatch')
            h1=[];h2=[];normal_tail=[];sizes={s:[] for s in size_effect};associations=[]
            for image in sorted(by_seed[0]):
                rows=[by_seed[seed][image] for seed in range(3)]
                if any([(r['region_id'],r['area_pixels']) for r in rows[0]['regions']]!=[(r['region_id'],r['area_pixels']) for r in q['regions']] for q in rows[1:]):raise ValueError('region identity mismatch')
                if rows[0]['normal_image']:
                    normal_tail.append(np.mean([r['distribution']['q95/normal']['relative_excess']['q99']-r['distribution']['none/normal']['relative_excess']['q99'] for r in rows]))
                    continue
                contrasts=[]
                for row in rows:
                    value=[]
                    for norm in ('none','q95'):
                        a=row['distribution'][norm+'/defect']['relative_excess']['mean'];b=row['distribution'][norm+'/normal']['relative_excess']['mean']
                        value.append(a-b if a is not None and b is not None else None)
                    contrasts.append(value[1]-value[0] if None not in value else None)
                h1.append(safe_mean(contrasts));h2.append(safe_mean([region_effect(r,'post') for r in rows]))
                for size in sizes:
                    val=safe_mean([region_effect(r,'post',size) for r in rows])
                    if val is not None:sizes[size].append(val)
                for rid,region in enumerate(rows[0]['regions']):
                    cov=safe_mean([(r['regions'][rid]['coverage']['q95_renyi/post/0.01']-r['regions'][rid]['coverage']['q95_mean/post/0.01'])*100 for r in rows])
                    concentration=safe_mean([r['regions'][rid]['scores']['q95']['max_resolution_share'] for r in rows])
                    associations.append(dict(image=image,region_id=region['region_id'],log_area=float(np.log10(region['area_fraction'])),contrast=region['standardized_abs_contrast'],concentration=concentration,renyi_coverage_delta_pp=cov))
            perclass[name]=safe_mean(h1);tail[name]=safe_mean(normal_tail);coverage[name]=safe_mean(h2)
            image_h1[name]=[v for v in h1 if v is not None];image_h2[name]=[v for v in h2 if v is not None]
            for size in size_effect:size_effect[size][name]=safe_mean(sizes[size])
            association[name]={}
            for descriptor in ('log_area','contrast','concentration'):
                valid=[r for r in associations if r[descriptor] is not None and r['renyi_coverage_delta_pp'] is not None]
                if len(valid)>2:
                    x=np.array([r[descriptor] for r in valid]);y=np.array([r['renyi_coverage_delta_pp'] for r in valid]);rho=float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None
                else:rho=None
                association[name][descriptor]=dict(spearman_rho=rho,regions=len(valid),p_value=None,limit='descriptive association; no iid region p-value; seed-averaged same region')
        h1=effect(perclass,rng);h2=effect(coverage,rng)
        section=dict(relative_excess_contrast=h1,normal_relative_excess_q99_change=effect(tail,rng),coverage_interaction_post=h2,
            conditional_image_bootstrap=dict(relative_excess_contrast=image_effect_bootstrap(image_h1,rng),coverage_interaction_post=image_effect_bootstrap(image_h2,rng)),
            coverage_interaction_by_size={size:effect(values,rng) for size,values in size_effect.items()},descriptor_associations=association)
        # Direct margin terms and transitions are retained by class/image in JSON.
        section['per_class_stage_coverage_interaction']={stage:{n:safe_mean([region_effect(r,stage) for seed in range(3) for r in idx[(shot,seed,n)]['images'] if r['regions']]) for n in CLASSES} for stage in ('pre','post')}
        result['shots'][str(shot)]=section
        fmt=lambda ci: '['+', '.join(f'{v:+.4f}' for v in ci)+']' if ci else '—'
        lines.append(f'| {shot} | {h1["mean"]:+.5f} | {fmt(h1["ci95"])} | {section["normal_relative_excess_q99_change"]["mean"]:+.5f} | {h2["mean"]:+.3f} | {fmt(h2["ci95"])} |')
    lines += ['', '상대 추가 응답은 CQ 전 E/mu다. coverage 상호작용은 (Q95 Rényi−Q95 평균)−(none Rényi−none 평균), 정상 이미지 교차검증 목표 FPR1%의 영역 검출 비율 차이이다. 실제 FPR은 arm마다 달라 동일 FPR 효과가 아니다.', '',
       '클래스 구간은 8개 관측 클래스의 재표집 민감도이며, 이미지 구간은 support/threshold/class를 고정한 조건부 구간이다. 클래스별 반대 방향·fabric 제외 결과는 response_summary.json에 기록했다.', '',
       '크기별 분석은 <0.1%, 0.1–1%, >=1%의 고정 구간. 클래스가 비어 있으면 제외하고 해당 구간의 실제 클래스 수를 표시한다. 이를 전체 full8 지표와 동일한 범위로 해석하지 않는다.', '',
       '대비·면적·해상도 응답 집중도 연관은 기술 통계이다. 영역 독립 p값을 생성하지 않는다. gray contrast는 texture/색 정보를 완전히 설명하지 못한다.', '',
       '원본 response_analysis.json에는 정상 mu 분위수별 조건부 분포, 영역별 pre/post coverage, gain/loss, CQ·threshold·양자화 margin 분해가 포함된다. 완료 후 RESULTS.md에서 이 근거를 연결해 해석한다.','']
    (args.root/'response_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.root/'RESPONSE_RESULTS.md').write_text('\n'.join(lines))
    if args.export_root:
        for p in args.root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.yaml','.md'):
                target=args.export_root/p.relative_to(args.root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);target.chmod(0o644)
    print(f'RESPONSE_SUMMARY_COMPLETE full48 root={args.root}',flush=True)


if __name__=='__main__':main()
