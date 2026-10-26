#!/usr/bin/env python3
"""Aggregate completed paired interventions; no pixel independence assumption."""
import argparse
import itertools
import json
from pathlib import Path
import shutil

import numpy as np


ARMS = ('raw_rank_raw_value', 'q95_rank_raw_value', 'raw_rank_q95_value',
        'q95_rank_q95_value', 'raw_mean', 'q95_mean')
CLASSES = ('can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts')


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def effect_summary(values, rng):
    # Input [class, seed]. Average repeated fits within class first.
    means = np.mean(values, axis=1)
    draws = means[rng.integers(0, len(means), (10000, len(means)))].mean(1)
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(means))))
    null = np.mean(signs*means[None], axis=1)
    return dict(mean_pp=float(means.mean()), class_bootstrap_ci95_pp=interval(draws),
                positive_classes=int((means > 0).sum()), class_effects_pp=means.tolist(),
                per_seed_mean_pp=np.mean(values, axis=0).tolist(),
                leave_one_class_out_pp=[float(np.delete(means, i).mean()) for i in range(len(means))],
                sign_flip_two_sided_p=float(np.mean(np.abs(null) >= abs(means.mean())-1e-12)))


def conditional_image_bootstrap(by_identity, shot, draws=2000):
    """Shared image resamples across all seeds/arms, fixed crossfit thresholds."""
    rng = np.random.default_rng(33029 + shot)
    sampled = np.zeros((draws, len(ARMS), 3))
    observed = np.zeros((len(ARMS), 3))
    for name in CLASSES:
        runs = [by_identity[(shot, seed, name)] for seed in range(3)]
        images = [{r['image']: r for r in d['images']} for d in runs]
        if any(set(images[0]) != set(x) for x in images[1:]):
            raise ValueError('query identity mismatch across support seeds')
        keys = sorted(images[0])
        rows = [images[0][key] for key in keys]
        counts = np.zeros((draws, len(keys)), dtype=np.int32)
        # Stratify normal/defect images; no resampling pixels or support seeds.
        for normal in (True, False):
            ids = [i for i, row in enumerate(rows) if row['normal_image'] == normal]
            counts[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)
        for seed in range(3):
            for a, arm in enumerate(ARMS):
                cells = [images[seed][key]['heldout_counts'][arm]['0.01'] for key in keys]
                matrix = np.asarray([[v[k] for k in ('fp', 'negatives', 'tp', 'positives', 'pro_mass', 'regions')]
                                     for v in cells], dtype=np.float64)
                totals = counts @ matrix
                ratios = totals[:, (0, 2, 4)] / totals[:, (1, 3, 5)]
                point = matrix.sum(0)
                sampled[:, a] += ratios/(len(CLASSES)*3)
                observed[a] += point[(0, 2, 4),]/point[(1, 3, 5),]/(len(CLASSES)*3)
    return dict(target_normal_calibration_fpr=.01, draws=draws,
        estimand='equal-class, equal-support-seed mean; fixed learned state and crossfit thresholds',
        interval_limit='conditional evaluation-image uncertainty; excludes threshold-estimator and support-population uncertainty',
        comparisons={label: {metric: dict(delta_pp=float((observed[b,j]-observed[a,j])*100),
                    ci95_pp=interval((sampled[:,b,j]-sampled[:,a,j])*100))
                    for j, metric in enumerate(('realized_fpr', 'tpr', 'pro'))}
            for label, a, b in [('final_vs_none_renyi', 0, 3), ('rank_only_vs_none_renyi', 0, 1),
                                ('value_only_vs_none_renyi', 0, 2), ('final_vs_q95_mean', 5, 3)]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--export-root', type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.rglob('statistical_analysis.json'))
    records = [json.loads(p.read_text()) for p in paths]
    identities = {(d['shots'], d['seed'], d['object']): d for d in records}
    expected = {(shot, seed, name) for shot in (2,16) for seed in range(3) for name in CLASSES}
    if len(records) != len(identities) or set(identities) != expected:
        raise ValueError(f'full48 required, got {len(records)} unique={len(identities)}')
    if any(max(d['metric_parity_abs'].values()) > 3e-5 or d['crossfit']['status'] != 'complete' for d in records):
        raise ValueError('incomplete or invalid replay')
    rng = np.random.default_rng(120026)
    report = dict(scope='AD2 public full8, 2/16-shot, seeds 0/1/2; post-hoc',
        inference_limit='8 observed benchmark classes, 3 support seeds; class bootstrap is robustness, not evidence of universal transfer',
        p_value_limit='exploratory unadjusted class sign-flip; requires class sign exchangeability under null',
        source_paths=[str(p) for p in paths], shots={})
    lines = ['# Q95 × Rényi 통계 분석 결과', '',
             'AD2 public의 사후 분석. 모든 조건은 동일 캐시와 support를 사용한다.', '',
             '## AUPRO@FPR≤5% (%, 클래스 평균 후 시드 평균)', '',
             '| Shot | none+평균 | none+Rényi | Q95+평균 | Q95+Rényi | 순위만 변경 | 크기만 변경 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for shot in (2,16):
        values = np.asarray([[[identities[(shot,seed,name)]['metrics'][arm]['seg_AUPRO_0_05']*100
                              for arm in ARMS] for seed in range(3)] for name in CLASSES])
        effects = dict(interaction=(values[:,:,3]-values[:,:,5])-(values[:,:,0]-values[:,:,4]),
                       rank_only=values[:,:,1]-values[:,:,0], value_only=values[:,:,2]-values[:,:,0],
                       full=values[:,:,3]-values[:,:,0],
                       joint_nonadditivity=values[:,:,3]-values[:,:,1]-values[:,:,2]+values[:,:,0])
        means = values.mean(axis=(0,1))
        result = dict(arm_means_pct=dict(zip(ARMS,means.tolist())),
                      effects={key:effect_summary(value,rng) for key,value in effects.items()},
                      conditional_image_bootstrap=conditional_image_bootstrap(identities,shot))
        # Equal-image summaries within class, equal-seed within class, then macro.
        tail = []
        for name in CLASSES:
            for seed in range(3):
                rows = [r for r in identities[(shot,seed,name)]['images']
                        if r['normal_image'] and len(r['resolutions'])==4]
                if not rows:
                    continue
                reduction = [np.asarray(r['sampled']['normal']['raw_log_quantile_sd'])-
                             np.asarray(r['sampled']['normal']['q95_log_quantile_sd']) for r in rows]
                tail.append(dict(object=name,seed=seed,images=len(rows),
                    log_quantile_sd_reduction=np.mean(reduction,axis=0).tolist(),
                    mean_top_resolution_change=float(np.mean([r['sampled']['normal']['top_resolution_change'] for r in rows]))))
        result['normal_image_sampled_tail_alignment'] = dict(quantiles=[.95,.99,.999],
            per_class_seed=tail, limit='paired pixel sample summaries; four-resolution normal images only; routed images retained in main metrics')
        report['shots'][str(shot)] = result
        lines.append('| '+str(shot)+' | '+' | '.join(f'{means[i]:.3f}' for i in (4,0,5,3,1,2))+' |')
    lines += ['', '## 짝지은 효과와 클래스 bootstrap 95% 구간', '',
              '| Shot | 효과 | 평균 (%p) | 95% 구간 | 양수 클래스 |', '|---|---|---:|---|---:|']
    for shot, result in report['shots'].items():
        for label, effect in result['effects'].items():
            lo,hi=effect['class_bootstrap_ci95_pp']
            lines.append(f'| {shot} | {label} | {effect["mean_pp"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {effect["positive_classes"]}/8 |')
    lines += ['', '클래스별·시드별 효과, 클래스 하나씩 제외한 평균, 탐색적 sign-flip p값,',
              '고정 교차검증 임계값 조건부 이미지 bootstrap 결과는 statistical_summary.json에 기록한다.',
              '이미지 bootstrap은 같은 이미지 추출을 모든 시드·조건에 공유한다. 픽셀을 독립 표본으로 취급하지 않는다.',
              '임계값 추정 불확실성과 새로운 support 모집단 불확실성은 해당 이미지 구간에 포함되지 않는다.',
              '순위/크기 분리는 분석용 개입이며, 비가산성은 지표 수준의 기술 통계다. 인과적 매개효과로 해석하지 않는다.',
              '표본 분위수는 이미지당 정상 최대 16384픽셀로 추정한다. Q99.9는 표본 꼬리의 불확실성이 크다.','']
    (args.root/'statistical_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.root/'STATISTICAL_RESULTS.md').write_text('\n'.join(lines))
    if args.export_root:
        # Compact artifacts only. Large per-image histograms stay remote.
        for path in args.root.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.yaml', '.md'}:
                target = args.export_root / path.relative_to(args.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    print(f'STATISTICAL_SUMMARY_COMPLETE full48 root={args.root}',flush=True)


if __name__ == '__main__':
    main()
