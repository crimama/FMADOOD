#!/usr/bin/env python3
"""Reaggregate Figure 6's frozen counts by acquisition. No threshold refit."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

CLASSES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'walnuts', 'wallplugs']
GROUPS = ['all', 'regular', 'shift', 'shift_named_only']
CONDITIONS = {'regular', 'overexposed', 'underexposed', 'shift_1', 'shift_2', 'shift_3', 'shift_4'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows, keys):
    return {k: sum(r[k] for r in rows) for k in keys}


def metrics(c):
    assert c['normal'] > 0 and c['defect'] > 0
    assert c['full_tp']-c['raw_tp'] == c['recovered']-c['lost']
    assert c['full_fp']-c['raw_fp'] == c['introduced_fp']-c['corrected_fp']
    return dict(raw_recall=100*c['raw_tp']/c['defect'], full_recall=100*c['full_tp']/c['defect'],
                raw_fpr=100*c['raw_fp']/c['normal'], full_fpr=100*c['full_fp']/c['normal'],
                recovered=100*c['recovered']/c['defect'], lost=100*c['lost']/c['defect'],
                recall_gain=100*(c['full_tp']-c['raw_tp'])/c['defect'],
                fpr_change=100*(c['full_fp']-c['raw_fp'])/c['normal'])


def plot(docs, report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    edges = np.asarray(docs[0]['histogram_edges'][:-1])
    centers, widths = (edges[1:]+edges[:-1])/2, np.diff(edges)
    maximum = 0
    for row, group in enumerate(['regular', 'shift']):
        stored = 'regular' if group == 'regular' else 'nonregular'
        for arm, prefix in enumerate(['raw', 'full']):
            ax = axes[row, arm]
            for label, color, name in [(0, '#737C87', 'Normal'), (1, '#B03A2E', 'Defect')]:
                hs = [np.asarray(d['histograms'][stored])[arm, label] for d in docs]
                # Undisplayed tails remain in the denominator, as in Figure 6.
                density = np.mean([h[:-1]/h.sum()/widths for h in hs], axis=0)
                ax.plot(centers, density, color=color, label=name, lw=1.5)
                ax.fill_between(centers, density, color=color, alpha=.12)
                maximum = max(maximum, density[(centers >= .1) & (centers <= 1.8)].max())
            m = report['groups'][group]['macro']
            ax.text(.98, .97, f'Recall {m[prefix+"_recall"]:.2f}%\nFPR {m[prefix+"_fpr"]:.3f}%',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
            ax.axvline(1, color='#333333', ls='--', lw=.8)
            ax.set(xlim=(.1, 1.8), xlabel='Score / frozen threshold')
            ax.set_title(f'{group.capitalize()} | '+['Raw L23', 'ScopeAD'][arm])
            if arm == 0:
                ax.set_ylabel('Pixel density (class macro)')
        ax = axes[row, 2]
        cs = report['groups'][group]['classes']
        y = np.arange(len(CLASSES))
        ax.barh(y, [cs[c]['metrics']['recovered'] for c in CLASSES], color='#287EA3', label='Recovered')
        ax.barh(y, [-cs[c]['metrics']['lost'] for c in CLASSES], color='#E8B34C', label='Lost')
        ax.plot([cs[c]['metrics']['recall_gain'] for c in CLASSES], y, 'o', color='#222222', ms=3, label='Net gain')
        ax.set_yticks(y, [c.replace('_', ' ') for c in CLASSES], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='#333333', lw=.6)
        loss_max = max(report['groups'][g]['classes'][c]['metrics']['lost']
                       for g in ['regular', 'shift'] for c in CLASSES)
        recovery_max = max(report['groups'][g]['classes'][c]['metrics']['recovered']
                           for g in ['regular', 'shift'] for c in CLASSES)
        ax.set(xlabel='Defect pixels (%)', xlim=(-loss_max*1.15, recovery_max*1.15))
        ax.set_title(f'{group.capitalize()} | Recovery / loss')
    for row in range(2):
        for col in range(2):
            axes[row, col].set_ylim(0, maximum*1.13)
    axes[0, 0].legend(loc='upper left', frameon=False, fontsize=8)
    fig.legend(*axes[0, 2].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.79, 0), ncol=3, frameon=False, fontsize=8)
    fig.suptitle('Public acquisition groups | 2-shot, seed 0 | Same pooled thresholds', fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, .95))
    for ext in ['pdf', 'png']:
        fig.savefig(output/f'hsa_regular_shift.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).parent/'hsa_evidence')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    docs = [json.loads((args.input_dir/f'{c}.json').read_text()) for c in CLASSES]
    assert len({d['threshold_source_sha256'] for d in docs}) == 1
    report = dict(split='MVTec AD 2 test_public (seen, post-hoc)', shots=2, seed=0,
                  units='Recall/FPR/recovery/loss: percent; gains/changes: percentage points',
                  threshold_protocol='Frozen class/arm pooled test-normal Q99, every 100th normal pixel per native image. Metrics use all native pixels. No condition refit.',
                  grouping={'regular': 'regular only', 'shift': 'all nonregular acquisitions, including over/underexposed',
                            'shift_named_only': 'shift_1 through shift_4 only; sensitivity check'},
                  script_sha256=sha(Path(__file__)), threshold_source_sha256=docs[0]['threshold_source_sha256'],
                  groups={g: dict(classes={}) for g in GROUPS}, sources={})
    csv_rows = []
    for cls, d in zip(CLASSES, docs):
        assert (d['object'], d['shots'], d['seed']) == (cls, 2, 0)
        assert len({r['path'] for r in d['images']}) == len(d['images'])
        keys = list(d['totals']['all'])
        assert counts(d['images'], keys) == d['totals']['all']
        identities, conditions, selected = defaultdict(set), Counter(), defaultdict(list)
        for r in d['images']:
            p = Path(r['path'])
            scene, condition = p.stem.split('_', 1)
            assert scene.isdigit() and condition in CONDITIONS, p
            identity = (p.parent.name, scene)
            assert identity not in identities[condition]
            identities[condition].add(identity)
            conditions[condition] += 1
            selected['all'].append(r)
            group = 'regular' if condition == 'regular' else 'shift'
            assert r['group'] == ('regular' if group == 'regular' else 'nonregular')
            selected[group].append(r)
            if condition.startswith('shift_'):
                selected['shift_named_only'].append(r)
            assert r['full_tp']-r['raw_tp'] == r['recovered']-r['lost']
            assert r['full_fp']-r['raw_fp'] == r['introduced_fp']-r['corrected_fp']
        assert all(ids == identities['regular'] for ids in identities.values())
        for group in GROUPS:
            rows = selected[group]
            c, m = counts(rows, keys), metrics(counts(rows, keys))
            if group in ['all', 'regular', 'shift']:
                stored = 'nonregular' if group == 'shift' else group
                assert c == d['totals'][stored]
                for arm in range(2):
                    assert sum(d['histograms'][stored][arm][0]) == c['normal']
                    assert sum(d['histograms'][stored][arm][1]) == c['defect']
            report['groups'][group]['classes'][cls] = dict(queries=len(rows), counts=c, metrics=m)
            csv_rows.append(dict(group=group, object=cls, queries=len(rows), **c,
                                 **{'metric_'+k: v for k, v in m.items()}))
        report['sources'][cls] = dict(path=str((args.input_dir/f'{cls}.json').resolve()),
            sha256=sha(args.input_dir/f'{cls}.json'), index=d['index'], index_sha256=d['index_sha256'],
            conditions=dict(conditions), scene_sets_paired=True, config=d['config'],
            thresholds={a: d['audit'][a]['threshold'] for a in ['Raw_L23_1120', 'ScopeAD_Full']})
    for group, b in report['groups'].items():
        cs = b['classes']
        b['macro'] = {k: mean(v['metrics'][k] for v in cs.values()) for k in m}
        b['queries'] = sum(c['queries'] for c in cs.values())
        b['improved_classes'] = sum(c['metrics']['recall_gain'] > 0 for c in cs.values())
        b['micro'] = metrics({k: sum(c['counts'][k] for c in cs.values()) for k in keys})
        csv_rows.append(dict(group=group, object='MACRO', queries=b['queries'],
                             **{'metric_'+k: v for k, v in b['macro'].items()}))
    assert report['groups']['all']['queries'] == 1084
    assert abs(report['groups']['all']['macro']['recall_gain']-9.715087) < .0001
    report['checks'] = dict(figure6_native_counts_reproduced=True, per_image_transition_identities=True,
                            histograms_cover_all_pixels=True, acquisition_scene_sets_paired=True,
                            same_pooled_thresholds_in_both_groups=True)
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in csv_rows for k in r)))
        writer.writeheader()
        writer.writerows(csv_rows)
    plot(docs, report, output)
    lines = ['# HSA: regular vs shift — 2026-10-26', '',
             'Matched Figure 6 reaggregation; MVTec AD 2 test_public, 2-shot, seed 0, eight classes. CPU only; no new fitting/inference. GT is used only for post-hoc evaluation.', '',
             'Hypothesis: full ScopeAD recovers more defects than Raw L23/1120 in both regular and changed acquisition conditions under the same pooled diagnostic threshold protocol. Higher recall is better; actual FPR must be inspected alongside it. Stop after all eight class audits and the pooled Figure 6 result reproduce.', '',
             'Primary shift = all nonregular acquisitions (overexposed, underexposed and available shift_1–4). No synthetic perturbations. Each condition has the same good/bad scene IDs as regular within its class. Class macros give each class equal weight; shift pools available conditions within a class.', '',
             report['threshold_protocol'], '',
             '| Group | Queries | Raw recall (%) | ScopeAD recall (%) | Gain (pp) | Raw FPR (%) | ScopeAD FPR (%) | Improved classes |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, b in report['groups'].items():
        m = b['macro']
        lines.append(f'| {group} | {b["queries"]} | {m["raw_recall"]:.5f} | {m["full_recall"]:.5f} | {m["recall_gain"]:+.5f} | {m["raw_fpr"]:.5f} | {m["full_fpr"]:.5f} | {b["improved_classes"]}/8 |')
    lines += ['', '## Primary class comparison', '',
              '| Class | Regular raw → full recall (%) | Gain (pp) | Shift raw → full recall (%) | Gain (pp) |',
              '|---|---:|---:|---:|---:|']
    for c in CLASSES:
        a, b = [report['groups'][g]['classes'][c]['metrics'] for g in ['regular', 'shift']]
        lines.append(f'| {c} | {a["raw_recall"]:.2f} → {a["full_recall"]:.2f} | {a["recall_gain"]:+.2f} | {b["raw_recall"]:.2f} → {b["full_recall"]:.2f} | {b["recall_gain"]:+.2f} |')
    lines += ['', '## Interpretation and limits', '',
              'Macro recall improves in both groups; walnuts regresses in both. This supports fewer missed defects for the full pipeline on this seen population, including changed acquisitions. It does not isolate module causation or establish improvement for every class.', '',
              'Regular and shift FPRs are not separately constrained to 1%: both use the same pooled class/arm thresholds. Cross-group recall differences alone do not measure robustness; within-group gains are not exact equal-FPR gains. Shift contains easier and harder acquisitions and need not have lower recall than regular. Do not call shift-minus-regular recall a performance loss without this context.', '',
              'No regular-only threshold recalibration, condition-specific threshold tuning, AUPRO reconstruction, private evaluation, or multi-seed significance is claimed. This is not an untouched shift transfer evaluation.', '',
              'Artifacts: summary.json (counts, hashes, configs, thresholds, audits); metrics.csv (class and macro metrics); hsa_regular_shift.pdf/png (distribution and recovery comparison).']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({g: dict(queries=b['queries'], improved=b['improved_classes'], **b['macro'])
                      for g, b in report['groups'].items()}, indent=2))


if __name__ == '__main__':
    main()
