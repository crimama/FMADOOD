"""Plot paired, observed query scores from the calibration diagnostic suite."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

CLASSES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
LAYERS = [5, 11, 23]
RESOLUTIONS = [448, 672, 896, 1120]
COLORS = ['#4393c3', '#e6a241', '#8a62ad']


def load_pairs(root):
    arms = {}
    for path in sorted(root.glob('gpu*/evaluation/ad2_*/**/run_manifest.json')):
        manifest = json.loads(path.read_text())
        label = path.relative_to(root).parts[2]
        for cls, meta in manifest['objects'].items():
            key = (label, cls)
            if key in arms:
                raise ValueError(f'Duplicate cell: {key}')
            arms[key] = (manifest, meta, path)
    pairs, audit = {}, []
    for shot in (2, 16):
        for cls in CLASSES:
            a, x, pa = arms[(f'ad2_none_s{shot:02d}', cls)]
            b, y, pb = arms[(f'ad2_tailscale_s{shot:02d}', cls)]
            for key in ['protocol', 'ablation_bank_feature_sha256', 'backbone',
                        'noise_guard', 'noise_guard_summary', 'bank_sizes',
                        'feature_layers', 'resolutions', 'prediction_postprocess']:
                if x[key] != y[key]:
                    raise ValueError(f'Unpaired {shot}/{cls}/{key}')
            assert a['code_sha256'] == b['code_sha256']
            assert x['prediction_postprocess'] == 'none'
            dx, dy = x['calibration_diagnostic'], y['calibration_diagnostic']
            assert dx['columns'] == dy['columns']
            assert dx['labels_used_only_after_prediction'] and dy['labels_used_only_after_prediction']
            identity = lambda d: [(q['query'], {k: (v['pixels'], v['sampled'])
                        for k, v in q['strata'].items()}) for q in d['queries']]
            assert identity(dx) == identity(dy)
            cols = dx['columns']
            raw = [cols.index(f'raw/r{r}/l{l}') for r in RESOLUTIONS for l in LAYERS]
            norm = [cols.index(f'normalized/r{r}/l{l}') for r in RESOLUTIONS for l in LAYERS]
            samples = {}
            for kind in ('normal', 'defect'):
                vx = np.asarray(dx['samples'][kind], dtype=np.float64)
                vy = np.asarray(dy['samples'][kind], dtype=np.float64)
                assert vx.shape == vy.shape and np.isfinite(vx).all() and np.isfinite(vy).all()
                # Exact equality also checks deterministic reservoir membership/order.
                np.testing.assert_array_equal(vx[:, raw], vy[:, raw])
                np.testing.assert_array_equal(vx[:, raw], vx[:, norm])
                samples[kind] = {'raw': vx[:, raw], 'calibrated': vy[:, norm],
                                 'fused_raw': vx[:, cols.index('fused')],
                                 'fused_calibrated': vy[:, cols.index('fused')]}
            pairs[shot, cls] = samples
            audit.append({'shot': shot, 'class': cls, 'raw_scores_exactly_equal': True,
                          'query_and_sampling_identity_matched': True,
                          'normal_samples': len(samples['normal']['raw']),
                          'defect_samples': len(samples['defect']['raw']),
                          'noise_guard_fired': x['noise_guard_summary']['fired'],
                          'none_manifest': str(pa.relative_to(root)),
                          'q95_manifest': str(pb.relative_to(root))})
    assert len(arms) == 32
    return pairs, audit


def boxes(ax, values, *, compact=False):
    box = ax.boxplot(values, positions=np.arange(12), widths=.66, whis=(5, 95),
                     showfliers=False, patch_artist=True, manage_ticks=False,
                     medianprops={'color': '#222222', 'linewidth': 1.0},
                     whiskerprops={'linewidth': .8}, capprops={'linewidth': .8})
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(COLORS[i % 3])
        patch.set_alpha(.72)
    for sep in (2.5, 5.5, 8.5):
        ax.axvline(sep, color='#dddddd', lw=.65, zorder=0)
    ax.set_xticks([1, 4, 7, 10], [str(r) for r in RESOLUTIONS])
    ax.set_xlim(-.7, 11.7)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.18)
    ax.tick_params(labelsize=8 if compact else 9)
    ax.set_xlabel('Input resolution', fontsize=8 if compact else 9)
    q = np.quantile(values, .95, axis=0)
    return float(q.max() / q.min())


def ecdf(ax, normal, defect):
    for label, values, color in [('Normal', normal, '#2874a6'), ('Defect', defect, '#c74a43')]:
        v = np.sort(values)
        ax.step(v, np.arange(1, len(v)+1)/len(v), where='post', label=label, color=color, lw=1.35)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('Actual fused score')
    ax.set_ylabel('Empirical CDF')
    ax.grid(alpha=.18)


def save(fig, output, stem):
    fig.savefig(output / f'{stem}.png', dpi=170, facecolor='white')
    fig.savefig(output / f'{stem}.pdf', facecolor='white')
    plt.close(fig)


def draw(pairs, output):
    handles = [Patch(facecolor=c, alpha=.72, label=f'Layer {l}') for l, c in zip(LAYERS, COLORS)]
    for shot in (2, 16):
        for kind in ('normal', 'defect'):
            fig, axes = plt.subplots(4, 4, figsize=(17, 11.8))
            for i, cls in enumerate(CLASSES):
                for j, stage in enumerate(('raw', 'calibrated')):
                    ax = axes[i//2, 2*(i%2)+j]
                    ratio = boxes(ax, pairs[shot, cls][kind][stage], compact=True)
                    ax.set_title(f'{cls} | {"Before" if j == 0 else "After q95"}\n'
                                 f'observation q95 max/min = {ratio:.2f}', fontsize=9)
                    ax.set_ylabel('Raw distance' if j == 0 else 'Calibrated distance', fontsize=8)
            fig.suptitle(f'{shot}-shot | {kind.capitalize()} pixels: observation score distributions', fontsize=15, y=.99)
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
            fig.text(.5, .012, 'Same sampled pixels before/after. Boxes: Q25–Q75; median line; whiskers: Q05–Q95. '
                     'Axes use actual stage-specific units; q95 max/min is scale invariant.\n'
                     'Image-capped diagnostic samples; these are not full-population pixel quantiles.', ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .045, 1, .935), h_pad=1.8, w_pad=1.4)
            save(fig, output, f'observation_{kind}_s{shot:02d}')
        fig, axes = plt.subplots(4, 4, figsize=(17, 11.4))
        for i, cls in enumerate(CLASSES):
            for j, stage in enumerate(('raw', 'calibrated')):
                ax = axes[i//2, 2*(i%2)+j]
                cell = pairs[shot, cls]
                ecdf(ax, cell['normal']['fused_'+stage], cell['defect']['fused_'+stage])
                ax.set_title(f'{cls} | {"Before" if j == 0 else "After q95"}', fontsize=10)
                ax.tick_params(labelsize=8)
        fig.suptitle(f'{shot}-shot | Actual fused-score distributions before/after calibration', fontsize=15)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .962), ncol=2, frameon=False)
        fig.text(.5, .01, 'Same sampled coordinates in both arms; normal/defect CDFs normalized separately. '
                 'X axes show actual stage-specific score units.\nImage-capped samples; all classes shown.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .935), h_pad=1.5)
        save(fig, output, f'fused_ecdf_s{shot:02d}')
        for cls in CLASSES:
            fig, axes = plt.subplots(3, 2, figsize=(10.5, 10))
            cell = pairs[shot, cls]
            for row, kind in enumerate(('normal', 'defect')):
                for col, stage in enumerate(('raw', 'calibrated')):
                    ratio = boxes(axes[row, col], cell[kind][stage])
                    axes[row, col].set_title(f'{kind.capitalize()} | {"Before" if col == 0 else "After q95"} '
                                             f'| q95 ratio {ratio:.2f}', fontsize=10)
                    axes[row, col].set_ylabel('Raw distance' if col == 0 else 'Calibrated distance')
            for col, stage in enumerate(('raw', 'calibrated')):
                ecdf(axes[2, col], cell['normal']['fused_'+stage], cell['defect']['fused_'+stage])
                axes[2, col].set_title('Actual fusion | '+('Before' if col == 0 else 'After q95'), fontsize=10)
            axes[2, 0].legend(frameon=False)
            fig.suptitle(f'{cls} | {shot}-shot | Paired score distributions', fontsize=15, y=.99)
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .965), ncol=3, frameon=False)
            fig.text(.5, .01, 'Same sampled pixels; boxes Q25–Q75, whiskers Q05–Q95. Actual stage-specific units.\n'
                     'Image-capped diagnostic sample; no evaluation-label use in calibration.', ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .045, 1, .935), h_pad=1.6)
            save(fig, output, f'detail_{cls}_s{shot:02d}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    pairs, audit = load_pairs(args.run_root)
    rows, quantiles = [], []
    for (shot, cls), samples in pairs.items():
        row = {'shot': shot, 'class': cls}
        for stage in ('raw', 'calibrated'):
            q = np.quantile(samples['normal'][stage], .95, axis=0)
            row[stage+'_normal_q95_ratio'] = float(q.max()/q.min())
        row['ratio_reduced'] = row['calibrated_normal_q95_ratio'] < row['raw_normal_q95_ratio']
        rows.append(row)
        for kind, stages in samples.items():
            for stage in ('raw', 'calibrated'):
                qs = np.quantile(stages[stage], [.05, .25, .5, .75, .95], axis=0)
                for i, (r, layer) in enumerate((r,l) for r in RESOLUTIONS for l in LAYERS):
                    quantiles.append(dict(shot=shot, class_name=cls, stratum=kind, stage=stage,
                         resolution=r, layer=layer, samples=len(stages[stage]),
                         **{f'q{int(p*100):02d}': float(qs[j,i]) for j,p in enumerate([.05,.25,.5,.75,.95])}))
    for name, table in [('normal_q95_spread.csv', rows), ('observation_quantiles.csv', quantiles)]:
        with (args.output_dir/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    (args.output_dir/'paired_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    draw(pairs, args.output_dir)
    print(json.dumps({'paired_cells': len(audit), 'normal_q95_spread_reduced': sum(r['ratio_reduced'] for r in rows),
                      'total_cells': len(rows), 'figure_pairs_png_pdf': 22, 'output': str(args.output_dir)}))


if __name__ == '__main__':
    main()
