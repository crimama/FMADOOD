#!/usr/bin/env python3
"""Plot measured HSA diagnostics; no fitting or threshold selection occurs here."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from figure_style import use_times_new_roman

HERE = Path(__file__).resolve().parent
DATA = HERE / 'hsa_evidence'
OUT = HERE.parent
CLASSES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'walnuts', 'wallplugs']
plt.rcParams.update({'font.size': 7.5,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
use_times_new_roman()
NORMAL, DEFECT = '#737C87', '#B03A2E'
TP, FP, FN, GT = '#287EA3', '#D34A41', '#E8B34C', '#33D4D9'


def fraction(d, key, population='defect', group='all'):
    c = d['totals'][group]
    return c[key] / c[population]


def spatial(axes, d, heading):
    with np.load(DATA / (d['object'] + '_example.npz')) as z:
        rgb, gt = z['rgb'], z['mask'].astype(bool)
        arrays = [None, z['raw'].astype(bool), z['full'].astype(bool)]
    im = d['selected_example']
    for a, (ax, pred) in enumerate(zip(axes, arrays)):
        ax.imshow(rgb)
        if pred is not None:
            overlay = np.zeros((*gt.shape, 4))
            from matplotlib.colors import to_rgba
            overlay[pred & gt] = to_rgba(TP, .85)
            overlay[pred & ~gt] = to_rgba(FP, .8)
            overlay[~pred & gt] = to_rgba(FN, .85)
            ax.imshow(overlay, interpolation='nearest')
        if gt.any() and (~gt).any():
            ax.contour(gt, levels=[.5], colors=[GT], linewidths=.55)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if a == 0:
            ax.set_title(heading + '\nQuery + GT', fontsize=7.5, loc='left')
        else:
            arm = ['raw', 'full'][a-1]
            name = ['Raw L23', 'ScopeAD'][a-1]
            ax.set_title(f'{name} | Recall {100*im[arm+"_tp"]/im["defect"]:.1f}%'
                         f' | FPR {100*im[arm+"_fp"]/im["normal"]:.2f}%', fontsize=7.1)


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    docs = [json.loads((DATA / f'{c}.json').read_text()) for c in CLASSES]
    assert len({d['threshold_source_sha256'] for d in docs}) == 1
    net = np.array([fraction(d, 'full_tp')-fraction(d, 'raw_tp') for d in docs])
    # Upper median by class gain, never the strongest gain; path-level selection
    # is independently fixed in the replay report before plotting.
    selected = docs[np.argsort(net, kind='stable')[len(docs)//2]]
    failure = docs[int(np.argmin(net))]
    fig = plt.figure(figsize=(3.35, 4.25))
    gs = fig.add_gridspec(2, 2, left=.15, right=.98, top=.82, bottom=.11,
                          hspace=.72, wspace=.2)
    edges = np.asarray(docs[0]['histogram_edges'][:-1])
    centers = (edges[:-1]+edges[1:])/2
    width = np.diff(edges)
    max_density = 0
    condition_metrics = {}
    for row, (group, name) in enumerate([('regular', 'Regular'), ('nonregular', 'Shift')]):
        metrics = {}
        for a, arm in enumerate(['raw', 'full']):
            ax = fig.add_subplot(gs[row, a])
            for label, color, label_name in [(0, NORMAL, 'Normal'), (1, DEFECT, 'Defect')]:
                hists = [np.asarray(d['histograms'][group])[a, label].astype(float) for d in docs]
                # Include out-of-view tails in each class denominator, then macro-average.
                density = np.mean([h[:-1]/h.sum()/width for h in hists], axis=0)
                ax.plot(centers, density, color=color, lw=1.25, label=label_name)
                ax.fill_between(centers, density, color=color, alpha=.12)
                max_density = max(max_density, density[(centers >= .1) & (centers <= 1.8)].max())
            recall = float(100*np.mean([fraction(d, arm+'_tp', group=group) for d in docs]))
            fpr = float(100*np.mean([fraction(d, arm+'_fp', 'normal', group=group) for d in docs]))
            metrics[arm] = dict(recall_percent=recall, fpr_percent=fpr)
            ax.axvline(1, color='#333333', ls='--', lw=.8)
            ax.text(.98, .96, f'Recall {recall:.2f}%\nFPR {fpr:.3f}%', transform=ax.transAxes,
                    ha='right', va='top', fontsize=6.6,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=.5))
            ax.set_xlim(.1, 1.8)
            if row == 1:
                ax.set_xlabel(r'Score / $t$')
            ax.set_xticks([.5, 1, 1.5])
            ax.set_title(f'({chr(97+2*row+a)}) '+['Raw L23', 'ScopeAD'][a],
                         loc='left', fontsize=7.5, fontweight='bold')
            if a == 0:
                ax.set_ylabel('Pixel density')
                if row == 0:
                    fig.legend(*ax.get_legend_handles_labels(), frameon=False,
                               loc='upper center', bbox_to_anchor=(.56, 1.015), ncol=2,
                               fontsize=7, handlelength=1.5, handletextpad=.4)
            else:
                ax.tick_params(labelleft=False)
        metrics['delta_recall_pp'] = metrics['full']['recall_percent']-metrics['raw']['recall_percent']
        condition_metrics[group] = metrics
        fig.text(.15, [.915, .465][row],
                 f'{name}  |  Recall +{metrics["delta_recall_pp"]:.2f} pp',
                 fontsize=8, fontweight='bold')
    for ax in fig.axes:
        ax.set_ylim(0, max_density*1.14)
    save(fig, 'fig_hsa_direct_evidence')

    # Preserve the pooled class-wise recovery/regression diagnostic in the appendix.
    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    recovered = 100*np.array([fraction(d, 'recovered') for d in docs])
    lost = 100*np.array([fraction(d, 'lost') for d in docs])
    y = np.arange(len(docs))
    ax.barh(y, recovered, color=TP, height=.6, label='Recovered defects')
    ax.barh(y, -lost, color=FN, height=.6, label='Lost defects')
    ax.plot(100*net, y, 'o', ms=3, color='#222222', label='Net change')
    ax.axvline(0, color='#333333', lw=.6)
    ax.set_yticks(y, [d['object'].replace('_', ' ') for d in docs], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel('Recovered (+) / lost (-) defects (%)', fontsize=7)
    ax.set_title('Recovery and regression (all conditions)', loc='left', fontsize=7.5, fontweight='bold')
    ax.legend(frameon=False, fontsize=6.5, loc='upper right', handlelength=1.2)
    save(fig, 'fig_hsa_recovery')
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.1), gridspec_kw={'wspace': .12})
    spatial(axes, failure, 'Failure: '+failure['object']+' / '+Path(failure['selected_example']['path']).stem)
    save(fig, 'fig_hsa_failure')
    summary = dict(primary_group='all', main_figure_groups=['regular', 'nonregular'],
                   condition_metrics=condition_metrics, representative_class=selected['object'],
                   representative_image=selected['selected_example'],
                   failure_class=failure['object'], failure_image=failure['selected_example'],
                   macro_recovered_percent=float(recovered.mean()), macro_lost_percent=float(lost.mean()),
                   macro_net_pp=float(100*net.mean()), improved_classes=int((net>0).sum()),
                   classes={d['object']: dict(delta_pp=100*net[i], recovered_percent=recovered[i], lost_percent=lost[i],
                       nonregular_delta_pp=100*(fraction(d,'full_tp',group='nonregular')-fraction(d,'raw_tp',group='nonregular')),
                       raw_fpr_percent=100*fraction(d,'raw_fp','normal'), full_fpr_percent=100*fraction(d,'full_fp','normal'))
                            for i,d in enumerate(docs)})
    (DATA / 'figure_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
