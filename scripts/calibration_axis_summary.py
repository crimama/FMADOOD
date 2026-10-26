#!/usr/bin/env python3
"""Full-scope summary of predeclared calibration-axis contrasts."""
import argparse,json,shutil
from pathlib import Path
import numpy as np

CLASSES=('can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts')
MODES=('none','global','layer','resolution','separable','full')


def contrast(matrix,rng):
    # [class,seed] -> equal-seed then equal-class estimand.
    means=matrix.mean(1);draw=means[rng.integers(0,len(means),(10000,len(means)))].mean(1)
    return dict(mean_pp=float(means.mean()),ci95_pp=np.quantile(draw,[.025,.975]).tolist(),per_class_pp=dict(zip(CLASSES,means.tolist())),per_seed_macro_pp=matrix.mean(0).tolist(),without_fabric_pp=float(np.delete(means,1).mean()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--export-root',type=Path);args=parser.parse_args()
    paths=sorted(args.root.rglob('axis_analysis.json'));ds=[json.loads(p.read_text()) for p in paths];idx={(d['shots'],d['seed'],d['object']):d for d in ds}
    if len(ds)!=48 or set(idx)!={(s,k,n) for s in (2,16) for k in range(3) for n in CLASSES}:raise ValueError('full48 unique identities required')
    if any(max(max(v.values()) for v in d['endpoint_parity_abs'].values())>3e-5 or d['max_full_token_error']>2e-5 for d in ds):raise ValueError('invalid endpoint')
    result=dict(scope='AD2 public full8 2/16-shot seeds0/1/2; frozen diagnostic factor interventions',sources=[str(p) for p in paths],shots={},limits='class bootstrap robustness on eight observed classes; no method selection; axis contrasts keep the same per-bag global factor')
    rng=np.random.default_rng(21102026);lines=['# Q95 보정 축 분리 — 완료 결과','','AUPRO@FPR≤.05, 8클래스 후 3-seed 평균(%). global은 같은 bag 공통 척도를 유지한다.','','| Shot | 분모 | Mean | Rényi | Rényi−Mean (pp) |','|---|---|---:|---:|---:|']
    for shot in (2,16):
        values={f'{m}/{a}':np.array([[idx[(shot,k,n)]['metrics'][f'{m}/{a}']['seg_AUPRO_0_05']*100 for k in range(3)] for n in CLASSES]) for m in MODES for a in ('mean','renyi')}
        means={k:float(v.mean()) for k,v in values.items()};effects={}
        for agg in ('mean','renyi'):
            v=lambda m:values[m+'/'+agg]
            for key,matrix in [('layer_minus_global',v('layer')-v('global')),('resolution_minus_global',v('resolution')-v('global')),('axis_main_effect_nonadditivity',v('separable')-v('layer')-v('resolution')+v('global')),('full_minus_separable',v('full')-v('separable'))]:effects[agg+'/'+key]=contrast(matrix,rng)
        for m in MODES:
            effects[m+'/renyi_minus_mean']=contrast(values[m+'/renyi']-values[m+'/mean'],rng)
            lines.append(f'| {shot} | {m} | {means[m+"/mean"]:.3f} | {means[m+"/renyi"]:.3f} | {means[m+"/renyi"]-means[m+"/mean"]:+.3f} |')
        result['shots'][str(shot)]=dict(arm_means_pct=means,effects=effects)
    lines += ['','분모 계층/해상도 축의 분리는 정상 support log-Q95의 직교 평균 분해다. none→global 효과를 계층/해상도 효과와 혼동하지 않는다. full−separable은 두 축 residual의 진단이며 유일한 인과적 분해는 아니다. 클래스·시드별 효과와 fabric 제외 결과는 axis_summary.json에 기록한다.','']
    (args.root/'axis_summary.json').write_text(json.dumps(result,indent=2)+'\n');(args.root/'AXIS_RESULTS.md').write_text('\n'.join(lines))
    if args.export_root:
        for p in args.root.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.yaml','.md'):
                t=args.export_root/p.relative_to(args.root);t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t);t.chmod(0o644)
    print('AXIS_SUMMARY_COMPLETE full48',flush=True)


if __name__=='__main__':main()
