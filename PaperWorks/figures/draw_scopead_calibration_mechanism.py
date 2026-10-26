"""Publication figure from verified, fixed-method calibration diagnostics.
Run: .venv/bin/python PaperWorks/figures/draw_scopead_calibration_mechanism.py
No new inference or selection. Error bars resample classes, never pixels.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'results/remote_runs/dsba3'
OUT=ROOT/'PaperWorks/cvpr-2026-author-kit-template/figures'
CLASSES=('can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts')
response_paths=sorted((RESULT/'scopead_response_trace_20261020').rglob('response_analysis.json'))
mechanism_paths=sorted((RESULT/'scopead_mechanism_20261020').rglob('calibration_mechanism_analysis.json'))
response={(d['shots'],d['seed'],d['object']):d for p in response_paths for d in [json.loads(p.read_text())]}
mechanism={(d['shots'],d['seed'],d['object']):d for p in mechanism_paths for d in [json.loads(p.read_text())]}
expected={(s,k,n) for s in (2,16) for k in range(3) for n in CLASSES}
assert len(response_paths)==len(mechanism_paths)==48 and set(response)==set(mechanism)==expected
stats_path=RESULT/'scopead_statistical_20261020/statistical_summary.json'
stats=json.loads(stats_path.read_text())
metadata=dict(scope='AD2 public full8, 2/16-shot, seeds 0/1/2; post-hoc fixed-method analysis',
    uncertainty='panel a/b: 95% percentile bootstrap over eight classes after paired image and seed averaging; not seed SD',
    aggregation='equal image means within each class/seed, then equal seed and class means; normal-image, background and defect populations explicitly distinct',
    labels='GT only after prediction; calibration factors fitted from normal supports; seen development split',
    panel_c='mean-one log-normal denominator factors, sigma .1/.2/.3; three draws paired between mean/Renyi per seed; not actual estimator uncertainty',
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in response_paths+mechanism_paths+[stats_path]],panels={})
rng=np.random.default_rng(210026)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7.5,'ytick.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(7.15,2.85),gridspec_kw={'width_ratios':[1.05,1.05,1.]},layout='constrained')
colors={2:'#2c6eac',16:'#d97924'}
for shot in (2,16):
    offset=-.14 if shot==2 else .14
    metadata['panels'].setdefault('a',{})[str(shot)]={}
    for i,(kind,label) in enumerate([('normal','Normal'),('background','Background'),('defect','Defect')]):
        effects=[]
        for name in CLASSES:
            seeds=[]
            for seed in range(3):
                rows=[r for r in response[(shot,seed,name)]['images'] if r['normal_image']==(kind=='normal')]
                population='defect' if kind=='defect' else 'normal'
                diffs=[(r['distribution']['q95/'+population]['relative_excess']['mean']-
                        r['distribution']['none/'+population]['relative_excess']['mean'])*100 for r in rows]
                seeds.append(float(np.mean(diffs)))
            effects.append(float(np.mean(seeds)))
        x=np.array(effects);mean=x.mean();draw=x[rng.integers(0,8,(10000,8))].mean(1);lo,hi=np.quantile(draw,[.025,.975])
        axes[0].scatter(x,i+offset+rng.uniform(-.025,.025,8),s=9,alpha=.28,color=colors[shot],edgecolors='none',zorder=2)
        axes[0].errorbar(mean,i+offset,xerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.4,capsize=2,lw=1,color=colors[shot],zorder=3)
        metadata['panels']['a'][str(shot)][kind]=dict(class_order=CLASSES,class_effects_pp=effects,mean_pp=float(mean),ci95_pp=[float(lo),float(hi)])
    metadata['panels'].setdefault('b',{})[str(shot)]={}
    for i,key in enumerate(('rank_only','value_only','full')):
        e=stats['shots'][str(shot)]['effects'][key];mean=e['mean_pp'];lo,hi=e['class_bootstrap_ci95_pp'];effects=e['class_effects_pp']
        axes[1].scatter(effects,i+offset+rng.uniform(-.025,.025,8),s=9,alpha=.28,color=colors[shot],edgecolors='none',zorder=2)
        axes[1].errorbar(mean,i+offset,xerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.4,capsize=2,lw=1,color=colors[shot],zorder=3)
        metadata['panels']['b'][str(shot)][key]=e
    metadata['panels'].setdefault('c',{})[str(shot)]={}
    for alpha,style in [('0','--'),('1','-')]:
        values=[0.]
        for sigma in (.1,.2,.3):
            values.append(float(np.mean([(mechanism[(shot,seed,n)]['metrics'][f'q95/noise{sigma}/rep{rep}/alpha{alpha}']['seg_AUPRO_0_05']-
                         mechanism[(shot,seed,n)]['metrics'][f'q95/alpha{alpha}']['seg_AUPRO_0_05'])*100
                        for seed in range(3) for n in CLASSES for rep in range(3)])))
        axes[2].plot([0,.1,.2,.3],values,style,color=colors[shot],marker='o',ms=3,lw=1.2)
        metadata['panels']['c'][str(shot)]['mean' if alpha=='0' else 'renyi']=dict(sigma=[0,.1,.2,.3],delta_aupro_pp=values)
for i,(labels,title) in enumerate([(['Normal','Background','Defect'],'(a) Relative excess response'),(['Rank only','Value only','Both'],'(b) Rank / value interventions')]):
    axes[i].set(yticks=np.arange(3),yticklabels=labels,title=title,ylim=(2.5,-.6))
    axes[i].axvline(0,color='.55',ls=':',lw=.8,zorder=1)
    axes[i].grid(axis='x',alpha=.13)
axes[0].set_xlabel(r'$\Delta(E/\mu)$: Q95 $-$ none (pp)')
axes[1].set_xlabel('$\\Delta$AUPRO vs none + Rényi (pp)')
axes[0].legend(handles=[Line2D([],[],marker='D',ms=3.5,color=colors[s],lw=1,label=f'{s}-shot') for s in (2,16)],loc='upper left',bbox_to_anchor=(0,-.25),ncol=2,frameon=False,handlelength=1.)
axes[2].set(title='(c) Calibration sensitivity',xlabel=r'Log-normal error $\sigma$',ylabel=r'$\Delta$AUPRO (pp)',xticks=[0,.1,.2,.3])
axes[2].axhline(0,color='.55',ls=':',lw=.8);axes[2].grid(alpha=.13)
axes[2].legend(handles=[Line2D([],[],color='.2',ls='--',lw=1.2,label='Mean'),Line2D([],[],color='.2',ls='-',lw=1.2,label='Rényi')],loc='lower left',frameon=False)
OUT.mkdir(parents=True,exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
for ext in ('pdf','png','svg'):fig.savefig(OUT/f'fig_scopead_calibration_mechanism.{ext}',dpi=240)
plt.close(fig)
(OUT/'source/calibration_mechanism.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Wrote verified 3-panel PDF/PNG/SVG and source metadata')
