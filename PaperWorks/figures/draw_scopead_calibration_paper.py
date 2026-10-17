"""Compact manuscript figure from paired saved calibration diagnostics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from draw_scopead_calibration_distributions import load_pairs, boxes, LAYERS, COLORS

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'results/remote_runs/dsba3/scopead_calibration_diagnostic_20261015'
OUT = ROOT/'PaperWorks/figures'
pairs, audit = load_pairs(RUN)
rows = []
for (shot, cls), cell in pairs.items():
    spread = {}
    for stage in ('raw', 'calibrated'):
        q = np.quantile(cell['normal'][stage], .95, axis=0)
        spread[stage] = float(q.max()/q.min())
    rows.append(dict(shot=shot, class_name=cls, **spread))
ranked = sorted([r for r in rows if r['shot']==2], key=lambda r:(r['raw']-r['calibrated'],r['class_name']))
example = ranked[(len(ranked)-1)//2]
cls = example['class_name']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'pdf.fonttype':42})
fig, axes = plt.subplots(1,3,figsize=(7.3,2.65),gridspec_kw={'width_ratios':[1.12,1.12,1]})
for i, stage in enumerate(('raw','calibrated')):
    ax=axes[i]
    ratio=boxes(ax,pairs[2,cls]['normal'][stage],compact=True)
    ax.set_title(('(a) Before calibration' if i==0 else '(b) After calibration'),fontsize=9,pad=7)
    ax.set_ylabel('Raw distance' if i==0 else 'Calibrated distance',fontsize=8)
    ax.set_xlabel('Input resolution',fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(.97,.96,f'Tail spread: {ratio:.2f}',transform=ax.transAxes,ha='right',va='top',fontsize=7,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
ax=axes[2]
lim=max(max(r['raw'],r['calibrated']) for r in rows)+.13
ax.plot([1,lim],[1,lim],ls='--',color='#999999',lw=.8,zorder=0)
for shot,color,marker in [(2,'#2874a6','o'),(16,'#c87825','^')]:
    rs=[r for r in rows if r['shot']==shot]
    ax.scatter([r['raw'] for r in rs],[r['calibrated'] for r in rs],s=24,c=color,marker=marker,
               label=f'{shot}-shot',edgecolors='white',linewidths=.4,zorder=3)
ax.set_xlim(.98,lim); ax.set_ylim(.98,lim)
ax.set_xlabel('Tail spread before',fontsize=8); ax.set_ylabel('Tail spread after',fontsize=8)
ax.set_title('(c) All classes and shots',fontsize=9,pad=7)
ax.tick_params(labelsize=7); ax.grid(alpha=.15)
ax.text(.97,.84,'14/16 below diagonal',ha='right',transform=ax.transAxes,fontsize=7,
        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
ax.legend(loc='upper left',frameon=False,fontsize=7,handletextpad=.3,borderpad=.2)
fig.legend(handles=[Patch(facecolor=c,alpha=.72,label=f'Layer {l}') for l,c in zip(LAYERS,COLORS)],
           loc='lower left',bbox_to_anchor=(.065,.005),ncol=3,frameon=False,fontsize=7)
fig.subplots_adjust(left=.075,right=.985,bottom=.245,top=.87,wspace=.51)
stem=OUT/'fig_scopead_calibration_paper'
fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
fig.savefig(stem.with_suffix('.pdf'),facecolor='white')
plt.close(fig)
meta={'example':example,'selection':'2-shot lower median rank (4/8) by absolute reduction in normal Q95 max/min; ascending, class-name tie break',
      'panels_ab':'Same normal query sample; boxes Q25-Q75, whiskers Q05-Q95; actual stage-specific units',
      'panel_c':'All 8 classes x 2 shots; max/min of 12 observation normal-sample Q95s',
      'source':str(RUN.relative_to(ROOT)), 'paired_cells':len(audit),'rows':rows}
stem.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'figure':str(stem),'example':example},indent=2))
