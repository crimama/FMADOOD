"""Charts and map figures for the ScopeAD (Outline v18) deck. Run with the project .venv python.

Numbers are parsed from PaperWorks/outlines/Outline_v18.md (outline_tables.py). Map figures are recomposed
from the cached per-layer maps in PaperWorks/figures/cache_rsg4_mechanism: deep_r = (z11+z23)/2 and z5_r are
stored per resolution (bag mean), so the linear0125 map is mean_r(deep_r + 0.125 * z5_r) exactly for those cached
runs (FP32, support manifest of the cache). They are illustrations, not new experiment results.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from outline_tables import num, pair, table  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'charts'
OUT.mkdir(exist_ok=True)
CACHE = Path('/home/hun/Volume/RESEARCH/FSAD-TTA/PaperWorks/figures/cache_rsg4_mechanism')
RES = [448, 672, 896, 1120]
LAM = 0.125

for f in (HERE / 'fonts').glob('NanumSquare_ac*.ttf'):
    font_manager.fontManager.addfont(str(f))
FAMILY = font_manager.FontProperties(fname=str(HERE / 'fonts' / 'NanumSquare_acR.ttf')).get_name()

NAVY, INK, SUB = '#163866', '#242B35', '#5F6B7A'
DARK, LIGHT, PALE, GRID = '#5F6B7A', '#A9B1BC', '#D3D8DE', '#E6E9ED'
plt.rcParams.update({
    'font.family': FAMILY, 'font.size': 13, 'axes.edgecolor': LIGHT, 'axes.labelcolor': INK,
    'xtick.color': SUB, 'ytick.color': SUB, 'text.color': INK, 'axes.titlesize': 14,
    'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .8, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.dpi': 200, 'axes.unicode_minus': False, 'mathtext.fontset': 'dejavusans',
})
SHOTS = [1, 2, 4, 8, 16]


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=.04, facecolor='white')
    plt.close(fig)


def repel(values, gap):
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for _ in range(80):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if pos[b] - pos[a] < gap:
                mid = (pos[a] + pos[b]) / 2
                pos[a], pos[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return pos


def ours_style(name):
    if name.startswith('ScopeAD'):
        return dict(color=NAVY, lw=2.8, marker='o', ms=8, zorder=5)
    if name == 'SuperAD':
        return dict(color=DARK, lw=2.0, marker='s', ms=7, zorder=4)
    return dict(color=LIGHT, lw=1.5, marker='o', ms=5, zorder=3)


# ------------------------------------------------------------------ table 4
def table4():
    rows = table('**표 4.')
    head = rows[0][2:]
    names = [('ScopeAD' if h.startswith('Ours') else h.replace(' (224)†', '')) for h in head]
    data = {}
    shot = None
    for r in rows[1:]:
        shot = int(r[0]) if r[0] else shot
        data.setdefault(r[1], {})[shot] = dict(zip(names, map(num, r[2:])))
    return names, data


def main_curve(metric='AUPRO@0.05', name='t4_aupro', w=11.9, h=3.9, ylim=(8, 66), gap=3.0):
    names, data = table4()
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(SHOTS))
    series = {n: [data[metric][s][n] for s in SHOTS] for n in names}
    for n, ys in series.items():
        ax.plot(x, ys, **ours_style(n))
    ends = repel([ys[-1] for ys in series.values()], gap)
    for (n, ys), y in zip(series.items(), ends):
        strong = n in ('ScopeAD', 'SuperAD')
        lab = f'{n}  {ys[-1]:.2f}' if strong else n
        ax.text(x[-1] + .12, y, lab, va='center', fontsize=13 if strong else 12,
                color=NAVY if n == 'ScopeAD' else (DARK if strong else SUB), fontweight='bold' if strong else 'normal')
    best_other = [max(v for k, v in data[metric][s].items() if k != 'ScopeAD' and v is not None) for s in SHOTS]
    for xi, (a, b) in enumerate(zip(series['ScopeAD'], best_other)):
        ax.annotate(f'+{a - b:.2f}', (xi, a), xytext=(0, 9), textcoords='offset points', ha='center',
                    fontsize=12, color=NAVY, fontweight='bold')
    ax.set_xticks(x, [f'{s}-shot' for s in SHOTS])
    ax.set_xlim(-.25, len(SHOTS) - 1 + 1.35)
    ax.set_ylim(*ylim)
    ax.set_ylabel(f'{metric} (%)')
    ax.grid(axis='x', visible=False)
    ax.text(0, ylim[1] - (ylim[1] - ylim[0]) * .05, '숫자 = ScopeAD - 비교군 최고값 (pp)', fontsize=11, color=SUB)
    save(fig, name)


def aux_curves(w=11.9, h=3.5):
    names, data = table4()
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    for ax, metric in zip(axes, ['pAUROC@0.05', 'I-AUROC']):
        for n in names:
            ys = [data[metric][s][n] for s in SHOTS]
            ax.plot(range(5), ys, label=n, **ours_style(n))
        ax.set_xticks(range(5), [str(s) for s in SHOTS])
        ax.set_xlabel('shot', fontsize=11)
        ax.set_title(metric, fontsize=13)
        ax.grid(axis='x', visible=False)
        top = [data[metric][s]['ScopeAD'] for s in SHOTS]
        ax.annotate(f'{top[-1]:.2f}', (4, top[-1]), xytext=(6, 6), textcoords='offset points', fontsize=12,
                    color=NAVY, fontweight='bold')
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', ncol=8, bbox_to_anchor=(.5, 1.1), fontsize=11, columnspacing=1.0)
    fig.tight_layout(w_pad=2)
    save(fig, 't4_aux')


# ------------------------------------------------------------------ table 5 / 5b
def table5(w=11.9, h=3.9):
    rows = table('**표 5.')
    head = [('ScopeAD' if h_.startswith('Ours') else h_) for h_ in rows[0][2:]]
    data = {}
    for r in rows[1:]:
        data.setdefault(r[0], {})[int(r[1])] = dict(zip(head, map(pair, r[2:])))
    fig, axes = plt.subplots(1, 4, figsize=(w, h))
    for ax, (ds, k, metric) in zip(axes, [('MVTec AD', 0, 'I-AUROC'), ('MVTec AD', 1, 'P-AUROC'),
                                           ('VisA', 0, 'I-AUROC'), ('VisA', 1, 'P-AUROC')]):
        for n in head:
            ys = [data[ds][s][n][k] for s in SHOTS]
            if all(v is None for v in ys):
                continue
            xs = [i for i, v in enumerate(ys) if v is not None]
            ax.plot(xs, [ys[i] for i in xs], label=n, **{**ours_style(n), 'ms': 5 if n != 'ScopeAD' else 7})
        ax.set_xticks(range(5), [str(s) for s in SHOTS])
        ax.set_title(f'{ds} · {metric}', fontsize=13)
        ax.set_xlabel('shot', fontsize=11)
        ax.tick_params(labelsize=11)
        ax.grid(axis='x', visible=False)
        lo = min(v[k] for s in SHOTS for v in data[ds][s].values() if v[k] is not None)
        ax.set_ylim(max(lo - 1, 70 if k == 0 else 92), 100 if k == 0 else 99)
    h_, l_ = axes[2].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', ncol=8, bbox_to_anchor=(.5, 1.08), fontsize=11, columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    save(fig, 't5')


# ------------------------------------------------------------------ table 6 / 7
def private_dots(w=11.9, h=3.5):
    rows = table('**표 6.')
    keep = {'Ours (ScopeAD, 게이트 설정)†': 'Ours (게이트 설정)†', 'RoBiS': 'RoBiS (full)', 'SuperADD': 'SuperADD (full)',
            'SuperAD': 'SuperAD (full)', 'EfficientAD-S': 'EfficientAD-S (full)', 'PatchCore': 'PatchCore (full)'}
    picked, seen = [], set()
    for r in rows:
        if r[0] in keep and pair(r[1])[0] is not None and keep[r[0]] not in seen:
            picked.append((keep[r[0]], pair(r[1]), pair(r[3])))
            seen.add(keep[r[0]])
    order = list(keep.values())
    picked.sort(key=lambda t: order.index(t[0]))
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    y = np.arange(len(picked))[::-1]
    for ax, idx, title in ((axes[0], 1, 'AU-PRO@0.05 (연속 점수 맵)'), (axes[1], 2, 'SegF1 (고정 임계값 적용)')):
        for yi, row in zip(y, picked):
            priv, mix = row[idx]
            c = NAVY if yi == y[0] else DARK
            ax.plot([mix, priv], [yi, yi], color=PALE, lw=3, zorder=2, solid_capstyle='round')
            ax.scatter([priv], [yi], s=80, color=c, zorder=4)
            ax.scatter([mix], [yi], s=80, facecolor='white', edgecolor=c, lw=2, zorder=4)
            if yi == y[0]:
                ax.annotate(f'{priv:.2f} / {mix:.2f}', (max(priv, mix), yi), xytext=(10, 0), textcoords='offset points',
                            va='center', fontsize=12, color=NAVY, fontweight='bold')
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, 82)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('%', fontsize=11)
    axes[0].set_yticks(y, [p[0] for p in picked])
    axes[0].get_yticklabels()[0].set_color(NAVY)
    axes[0].get_yticklabels()[0].set_fontweight('bold')
    axes[0].scatter([], [], s=70, color=DARK, label='TESTpriv (일반 조명)')
    axes[0].scatter([], [], s=70, facecolor='white', edgecolor=DARK, lw=2, label='TESTpriv,mix (혼합 조명)')
    fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(.55, 1.08), fontsize=12)
    fig.tight_layout(w_pad=2)
    save(fig, 't6_private')


def corruption_baselines(w=11.9, h=3.2):
    rows = table('**표 7.')
    conds = rows[0][1:]
    data = {r[0]: [pair(c)[0] for c in r[1:]] for r in rows[1:] if pair(r[1])[0] is not None}
    marks = {'PatchCore': dict(color=LIGHT, marker='^', s=90), 'SuperAD': dict(color=DARK, marker='s', s=80),
             'FastRecon (project adaptation)†': dict(color=LIGHT, marker='D', s=60)}
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(conds))[::-1]
    for n, xs in data.items():
        ax.scatter(xs, y, label=n.replace(' (project adaptation)', ''), zorder=4, **marks.get(n, dict(color=LIGHT)))
    ax.set_yticks(y, [c.replace(' s3', '') for c in conds])
    ax.set_xlim(55, 90)
    ax.set_xlabel('AUPRO@0.05 (%)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.2), fontsize=12)
    ax.text(89.5, y[-1], 'ScopeAD(linear0125)\n아직 미실행', ha='right', va='center', fontsize=12, color=NAVY,
            fontweight='bold')
    save(fig, 't7_corruption')


# ------------------------------------------------------------------ ablation tables
def table8(w=11.9, h=3.35):
    rows = table('**표 8.')
    labels = {'Full (linear0125)': 'Full (ScopeAD)', '(b) w/o Score normalization': '(b) 정규화 생략',
              '(c) Equal-weight 3-layer mean': '(c) 3계층 동일 가중 평균', '(d) Layer concat (단일 bank)': '(d) 계층 concat 단일 bank',
              '(e) MAD 보정': '(e) MAD 보정', '(f) 448 단일 해상도': '(f) 448 단일 해상도'}
    items = [(labels[r[0]], num(r[4]), num(r[5]), r[6]) for r in rows[1:]]
    full = items[0][1]
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(items))[::-1]
    ax.axvline(full, color=NAVY, ls='--', lw=1.4, zorder=1)
    for yi, (lab, au, pa, d) in zip(y, items):
        strong = lab.startswith('Full')
        c = NAVY if strong else DARK
        ax.plot([50, au], [yi, yi], color=PALE, lw=2, zorder=1)
        ax.scatter([au], [yi], s=140 if strong else 90, color=c, zorder=4)
        txt = f'{au:.2f}' + ('' if strong else f'  ({d.replace("−", "-")})')
        ax.annotate(txt, (au, yi), xytext=(12, 0), textcoords='offset points', va='center', fontsize=13, color=c,
                    fontweight='bold' if strong else 'normal', bbox=dict(boxstyle='square,pad=.15', fc='white', ec='none'))
    ax.set_yticks(y, [i[0] for i in items])
    ax.get_yticklabels()[0].set_color(NAVY)
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.set_xlim(50, 62.5)
    ax.set_xlabel('AUPRO@0.05 (%)   ·   괄호 = Full 대비 (pp)')
    ax.grid(axis='y', visible=False)
    save(fig, 't8_ablation')


def table8a(w=11.9, h=3.6):
    rows = table('**표 8a.')
    cols = ['Full', 'No normalization', 'Equal-weight mean', 'Layer concat']
    data = {}
    for r in rows[1:]:
        data.setdefault(r[0], {})[int(r[1])] = [pair(c)[0] for c in r[2:]]
    marks = [dict(color=NAVY, lw=2.6, marker='o', ms=7, zorder=5), dict(color=DARK, lw=1.6, marker='s', ms=5, ls='--'),
             dict(color=LIGHT, lw=1.6, marker='^', ms=6), dict(color=LIGHT, lw=1.6, marker='D', ms=5, ls=':')]
    labels = ['Full (ScopeAD)', '정규화 생략', '3계층 동일 가중', '계층 concat 단일 bank']
    fig, axes = plt.subplots(1, 3, figsize=(w, h))
    for ax, (ds, metric) in zip(axes, [('AD2', 'AUPRO@0.05'), ('MVTec AD', 'I-AUROC'), ('VisA', 'I-AUROC')]):
        for j in range(4):
            ax.plot(range(5), [data[ds][s][j] for s in SHOTS], label=labels[j], **marks[j])
        ax.set_xticks(range(5), [str(s) for s in SHOTS])
        ax.set_title(f'{ds} · {metric}', fontsize=13)
        ax.set_xlabel('shot', fontsize=11)
        ax.grid(axis='x', visible=False)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.08), fontsize=12)
    fig.tight_layout(w_pad=1.5)
    save(fig, 't8a')


def table8b(w=11.9, h=3.3):
    rows = table('**표 8b.')
    lam = [0.125, 0.25, 0.5, 1, 2]
    a2 = [num(r[1]) for r in rows[1:]]
    a16 = [num(r[3]) for r in rows[1:]]
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(lam))
    ax.plot(x, a16, color=NAVY, lw=2.6, marker='o', ms=8, label='16-shot', zorder=5)
    ax.plot(x, a2, color=DARK, lw=2.0, marker='s', ms=7, ls='--', label='2-shot (λ 선택에 사용)', zorder=4)
    for xi, v in zip(x, a16):
        ax.annotate(f'{v:.2f}', (xi, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=12,
                    color=NAVY, fontweight='bold')
    for xi, v in zip(x, a2):
        ax.annotate(f'{v:.2f}', (xi, v), xytext=(0, -18), textcoords='offset points', ha='center', fontsize=11,
                    color=DARK)
    ax.set_xticks(x, ['λ = 0.125\n(ScopeAD)', '0.25', '0.5', '1\n(전체 가산)', '2'])
    ax.set_ylim(36, 64)
    ax.set_ylabel('AUPRO@0.05 (%)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower left', fontsize=12)
    save(fig, 't8b_lambda')


def table8cd(w=11.9, h=3.5):
    d8d = table('**표 8d.')
    d8c = table('**표 8c.')
    names = ['none', 'MAD', 'q95']
    marks = {'none': dict(color=DARK, marker='s', ms=6, lw=1.6, ls='--'), 'MAD': dict(color=LIGHT, marker='^', ms=6, lw=1.6),
             'q95': dict(color=NAVY, marker='o', ms=7, lw=2.6, zorder=5)}
    fig, axes = plt.subplots(1, 3, figsize=(w, h))
    ad2 = {n: [float(r[i + 1].split('/')[0]) for r in d8d[1:]] for i, n in enumerate(names)}
    for n in names:
        axes[0].plot(range(5), ad2[n], label=n if n != 'q95' else 'q95 (기본)', **marks[n])
    axes[0].set_title('AD2 · AUPRO@0.05', fontsize=13)
    for ax, ds in zip(axes[1:], ['MVTec AD', 'VisA']):
        rows = [r for r in d8c[1:] if r[0] == ds]
        for i, n in enumerate(names):
            ax.plot(range(5), [pair(r[i + 2])[0] for r in rows], **marks[n])
        ax.set_title(f'{ds} · I-AUROC', fontsize=13)
    for ax in axes:
        ax.set_xticks(range(5), [str(s) for s in SHOTS])
        ax.set_xlabel('shot', fontsize=11)
        ax.grid(axis='x', visible=False)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.08), fontsize=12)
    fig.tight_layout(w_pad=1.5)
    save(fig, 't8cd_norm')


def table10ab(w=11.9, h=3.4):
    t10a = table('**표 10a.')
    t10b = table('**표 10b.')
    fig, axes = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={'width_ratios': [1.25, 1]})
    labels = ['448', '672', '896', '1120', '4 해상도']
    x = np.arange(len(labels))
    for k, (shot, c, m, dx, off) in enumerate([('2-shot', DARK, 's', -.12, -16), ('16-shot', NAVY, 'o', .12, 10)]):
        vals = [pair(r[k + 1])[0] for r in t10a[1:]]
        axes[0].scatter(x + dx, vals, color=c, marker=m, s=[70] * 4 + [130], zorder=4, label=shot,
                        alpha=1.0)
        axes[0].annotate(f'{vals[-1]:.2f}', (x[-1] + dx, vals[-1]), xytext=(-14 if k == 0 else 14, off),
                         textcoords='offset points', ha='center', fontsize=12, color=c, fontweight='bold')
    axes[0].axvspan(3.5, 4.5, color='#F3F4F6', zorder=0)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(49, 62)
    axes[0].set_title('AD2 · AUPRO@0.05 (표 10a)', fontsize=13)
    axes[0].grid(axis='x', visible=False)
    axes[0].legend(loc='upper left', fontsize=12)
    groups = [(r[0], r[1], pair(r[2])[0], pair(r[3])[0]) for r in t10b[1:]]
    xs = np.arange(2)
    for j, ds in enumerate(['MVTec AD', 'VisA']):
        single = [g for g in groups if g[0] == ds and '1120' in g[1]][0]
        multi = [g for g in groups if g[0] == ds and '4' in g[1]][0]
        for s, (a, b) in enumerate([(single[2], multi[2]), (single[3], multi[3])]):
            xx = j * 2.4 + s
            axes[1].plot([xx, xx], [a, b], color=PALE, lw=3, zorder=1)
            axes[1].scatter([xx], [a], color=LIGHT, marker='s', s=80, zorder=3)
            axes[1].scatter([xx], [b], color=NAVY, s=100, zorder=4)
            axes[1].annotate(f'+{b - a:.2f}', (xx, b), xytext=(0, 8), textcoords='offset points', ha='center',
                             fontsize=11, color=NAVY, fontweight='bold')
    axes[1].set_xticks([0, 1, 2.4, 3.4], ['MVTec\n2-shot', 'MVTec\n16-shot', 'VisA\n2-shot', 'VisA\n16-shot'])
    axes[1].set_title('MVTec AD · VisA I-AUROC (표 10b)', fontsize=13)
    axes[1].scatter([], [], color=LIGHT, marker='s', s=70, label='1120 단일')
    axes[1].scatter([], [], color=NAVY, s=80, label='4 해상도')
    axes[1].legend(loc='lower right', fontsize=12)
    axes[1].grid(axis='x', visible=False)
    axes[1].set_ylim(85, 100.5)
    fig.tight_layout(w_pad=2.5)
    save(fig, 't10ab_resolution')


def table10_gate(w=11.9, h=3.0):
    rows = table('**표 10.')
    strata = ['Full8 (전체)', 'Regular', 'Shift 1']
    cfg = [('4-해상도 (토큰 256)', NAVY, 'o', 130), ('1120 단일 (토큰 256)', LIGHT, 's', 90),
           ('1120 단일 (토큰 1024, 용량 일치)', DARK, 'D', 80)]
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(3)
    for k, (r, (lab, c, m, s)) in enumerate(zip(rows[1:], cfg)):
        vals = [num(r[2]), pair(r[5])[0], pair(r[6])[0]]
        xs = x + (k - 1) * .18
        ax.scatter(xs, vals, color=c, marker=m, s=s, label=lab, zorder=4)
        if k == 0:
            for xi, v in zip(xs, vals):
                ax.annotate(f'{v:.2f}', (xi, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=12,
                            color=NAVY, fontweight='bold')
    ax.set_xticks(x, strata)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(55, 63)
    ax.set_ylabel('AUPRO@0.05 (%)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.22), fontsize=12)
    save(fig, 't10_gate')


def table11(w=11.9, h=3.3):
    rows = table('**표 11.')
    qs = [r for r in rows[1:] if num(r[0]) is not None and len(r[0]) < 8]
    q = [num(r[0]) for r in qs]
    a2 = [pair(r[1])[0] for r in qs]
    a16 = [pair(r[2])[0] for r in qs]
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(x, a16, color=NAVY, lw=2.6, marker='o', ms=8, label='16-shot', zorder=5)
    ax.plot(x, a2, color=DARK, lw=2.0, marker='s', ms=7, ls='--', label='2-shot', zorder=4)
    for xi, v in zip(x, a16):
        ax.annotate(f'{v:.2f}', (xi, v), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=11, color=NAVY)
    ax.set_xticks(x, [f'{v:g}' + ('\n(기본)' if v == .95 else '') for v in q])
    ax.set_xlabel('척도 분위수 q')
    ax.set_ylim(57.5, 60.5)
    ax.set_ylabel('AUPRO@0.05 (%)')
    ax.axvspan(.6, 5.4, color='#F3F4F6', zorder=0)
    ax.text(3, 57.75, 'q = 0.80 ~ 0.999: 변화 -0.22 ~ +0.04pp', ha='center', fontsize=12, color=DARK)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', fontsize=12)
    save(fig, 't11_q')


# ------------------------------------------------------------------ concept / illustrative
def ambiguity_concept(w=11.9, h=3.2):
    x = np.linspace(0, 3.2, 600)

    def bump(mu, sd, amp):
        return amp * np.exp(-.5 * ((x - mu) / sd) ** 2)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.fill_between(x, bump(.55, .18, 1.0), color=PALE, label='support와 비슷한 정상 패치')
    ax.fill_between(x, bump(1.25, .25, .45), color=LIGHT, alpha=.75, label='support에 없던 정상 변동 (조명·자세·반사)')
    ax.plot(x, bump(2.0, .38, .4), color=NAVY, lw=2.6, label='결함 패치')
    ax.axvspan(1.45, 2.0, color='#F3F4F6', zorder=0)
    ax.text(1.72, 1.03, '같은 "큰 거리"\n원인은 둘 중 무엇?', ha='center', va='top', fontsize=13, color=NAVY,
            fontweight='bold')
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_xlabel('가장 가까운 정상 패치와의 거리  →  클수록 이상')
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('개념도 (가상 분포, 실측 아님)', fontsize=12, color=SUB, loc='left', fontweight='normal')
    save(fig, 'ambiguity_concept')


def formula(w=11.9, h=2.4):
    fig = plt.figure(figsize=(w, h))
    fig.text(.02, .80, 'Scale Calibration', fontsize=15, fontweight='bold', color=NAVY)
    fig.text(.27, .80, r'$z_m = d_m \,/\, c_m$,    $c_m = Q_{0.95}(D_m^{\mathrm{LSO}})$', fontsize=19, color=INK)
    fig.text(.27, .62, '관측 m = (층, 해상도, bag)마다 bank를 두고, 그 bank의 정상 거리 상위 5% 값을 1로 맞춤', fontsize=13,
             color=SUB)
    fig.text(.02, .34, 'Deep-Anchored Weighting', fontsize=15, fontweight='bold', color=NAVY)
    fig.text(.27, .34, r'$S = z_d + \lambda\, z_5$,    $z_d = (z_{11}+z_{23})/2$,    $\lambda = 0.125$', fontsize=19,
             color=INK)
    fig.text(.27, .16, 'Deep 평균을 기준선으로 두고 Shallow는 작은 고정 가중치로만 더함 (게이트·학습 가중치 없음)', fontsize=13,
             color=SUB)
    save(fig, 'formula')


def fusion_example(w=11.9, h=2.9):
    rows = table('예시 계산')
    groups = [r[0] for r in rows[1:]]
    deep = [num(r[1]) for r in rows[1:]]
    z5 = [num(r[2]) for r in rows[1:]]
    ours = [num(r[4]) for r in rows[1:]]
    ours_txt = [r[4] for r in rows[1:]]
    full = [a + b for a, b in zip(deep, z5)]
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(groups))
    bw = .25
    for k, (vals, c, lab) in enumerate(((deep, PALE, 'Deep만 (z_d)'), (full, LIGHT, '전체 가산 (z_d + z5, λ = 1)'),
                                         (ours, NAVY, 'ScopeAD (z_d + 0.125·z5)'))):
        bars = ax.bar(x + (k - 1) * (bw + .02), vals, bw, color=c, label=lab)
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, v + .06, ours_txt[i] if k == 2 else f'{v:.2f}', ha='center',
                    fontsize=11, color=NAVY if k == 2 else INK, fontweight='bold' if k == 2 else 'normal')
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 6.2)
    ax.set_ylabel('결합 점수 (z 단위)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_title('설명용 예시 값 (Outline §3.4, 실측 아님)', fontsize=12, color=SUB, loc='right', fontweight='normal')
    save(fig, 'fusion_example')


def bank_schematic(w=11.9, h=3.0):
    """Concat single bank vs observation-wise banks (concept schematic)."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis('off')

    def box(x, y, bw, bh, text, edge=LIGHT, fill='white', color=INK, size=12, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle='round,pad=0,rounding_size=0.05', facecolor=fill,
                                    edgecolor=edge, lw=1.2))
        ax.text(x + bw / 2, y + bh / 2, text, ha='center', va='center', fontsize=size, color=color,
                fontweight='bold' if bold else 'normal')

    def arrow(x0, y0, x1, y1, color=DARK):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=color, lw=1.4))
    ax.text(.05, 3.05, '기존: 계층 특징을 이어 붙여 bank 하나', fontsize=13, color=DARK, fontweight='bold', va='top')
    for i, l in enumerate(['L5', 'L11', 'L23']):
        box(.1 + i * .62, 1.55, .56, .55, l, size=11)
    arrow(2.0, 1.82, 2.45, 1.82)
    box(2.5, 1.5, 1.5, .65, '[L5 | L11 | L23]\n단일 bank', size=11)
    arrow(4.05, 1.82, 4.5, 1.82)
    box(4.55, 1.5, 1.1, .65, '점수 1개', size=11)
    ax.text(.1, 1.05, '계층마다 다른 정상 변동이\n하나의 거리 안에 섞임', fontsize=11.5, color=SUB, va='top')
    ax.plot([6.05, 6.05], [.2, 3.0], color=GRID, lw=1.5)
    ax.text(6.35, 3.05, 'ScopeAD: 관측마다 bank를 따로 두고 단위를 맞춘 뒤 결합', fontsize=13, color=NAVY,
            fontweight='bold', va='top')
    for i, (l, scale) in enumerate([('L5', 'c5'), ('L11', 'c11'), ('L23', 'c23')]):
        y = 2.15 - i * .72
        box(6.35, y, .55, .5, l, size=11)
        arrow(6.95, y + .25, 7.2, y + .25)
        box(7.25, y, 1.45, .5, f'bank · 단위 {scale}', edge=NAVY, size=10.5)
        arrow(8.75, y + .25, 9.05, 1.45)
    box(9.1, 1.15, 1.2, .6, 'z = d / c\n공통 단위', edge=NAVY, fill='#EEF2F7', color=NAVY, size=10.5, bold=True)
    arrow(10.35, 1.45, 10.55, 1.45, NAVY)
    box(10.6, 1.15, 1.38, .6, 'z_d + 0.125·z5', edge=NAVY, fill='#EEF2F7', color=NAVY, size=10.5, bold=True)
    ax.text(6.4, .2, '(층, 해상도, bag) 48개 bank · 같은 저장 용량에서 단일 bank보다 AUPRO@0.05 +2.09pp (표 8)',
            fontsize=11.5, color=SUB)
    save(fig, 'bank_schematic')


# ------------------------------------------------------------------ recomposed maps (cache)
def _lin(d, lam, r=None):
    rs = [r] if r else RES
    return np.mean([d[f'r{x}_deep'].astype(np.float32) + lam * d[f'r{x}_normalized_l5'].astype(np.float32)
                    for x in rs], 0)


def _show(ax, img, title, vmax=2.0, gt=None, cmap='inferno', size=12):
    ax.imshow(img, cmap=cmap if img.ndim == 2 else None, vmin=0 if img.ndim == 2 else None,
              vmax=vmax if img.ndim == 2 else None)
    if gt is not None:
        ax.contour(gt > 0, levels=[.5], colors='#5CE1E6', linewidths=1.2)
    ax.set_title(title, fontsize=size, fontweight='normal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def task_maps():
    d = np.load(CACHE / 'ad_hazelnut__clean.npz')
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    _show(axes[0], d['rgb'], '입력: query 이미지\n(하늘색 = 정답 결함 윤곽)', gt=d['gt'], size=13)
    _show(axes[1], _lin(d, LAM), '출력: 결함 점수 맵\n(밝을수록 결함)', gt=d['gt'], size=13)
    fig.tight_layout(w_pad=2)
    save(fig, 'task')


def resolution_row():
    d = np.load(CACHE / 'ad_hazelnut__clean.npz')
    fig, axes = plt.subplots(1, 5, figsize=(11.9, 2.6))
    _show(axes[0], d['rgb'], 'query', gt=d['gt'])
    for ax, r in zip(axes[1:], RES):
        _show(ax, _lin(d, LAM, r), f'{r} px  ({r // 14}×{r // 14} 토큰)', gt=d['gt'])
    fig.tight_layout(w_pad=.6)
    save(fig, 'resolution_row')


def shallow_leak():
    good = np.load(CACHE / 'overview_walnuts__good.npz')
    bad = np.load(CACHE / 'overview_walnuts__defect.npz')
    fig, axes = plt.subplots(2, 4, figsize=(11.9, 5.0))
    cols = [('Deep만 (λ = 0)', 0.0), ('Shallow 전체 가산 (λ = 1)', 1.0), ('ScopeAD (λ = 0.125)', LAM)]
    for i, (d, lab) in enumerate([(good, '정상 query'), (bad, '결함 query')]):
        _show(axes[i, 0], d['rgb'], lab, gt=d['gt'] if i else None)
        for j, (t, lam) in enumerate(cols):
            m = _lin(d, lam)
            _show(axes[i, j + 1], m, f'{t}\n최댓값 {m.max():.2f}', gt=d['gt'] if i else None)
    fig.tight_layout(h_pad=1.0, w_pad=.6)
    save(fig, 'shallow_leak')


def fusion_maps():
    d = np.load(CACHE / 'ad_hazelnut__clean.npz')
    r = 1120
    deep = d[f'r{r}_deep'].astype(np.float32)
    z5 = d[f'r{r}_normalized_l5'].astype(np.float32)
    panels = [('Query + GT', d['rgb']), ('Deep 기준선 z_d', deep), ('Shallow z5 (전체)', z5),
              ('0.125 · z5 (가산분)', LAM * z5), ('ScopeAD  z_d + 0.125·z5', deep + LAM * z5),
              ('비교: z_d + z5 (λ = 1)', deep + z5)]
    fig, axes = plt.subplots(1, 6, figsize=(11.9, 2.55))
    for ax, (t, img) in zip(axes, panels):
        _show(ax, img, t, gt=d['gt'], size=11)
    fig.tight_layout(w_pad=.4)
    save(fig, 'fusion_maps')


ALL = [main_curve, aux_curves, table5, private_dots, corruption_baselines, table8, table8a, table8b, table8cd,
       table10ab, table10_gate, table11, ambiguity_concept, formula, fusion_example, bank_schematic, task_maps,
       resolution_row, shallow_leak, fusion_maps]

if __name__ == '__main__':
    names = sys.argv[1:]
    for fn in ALL:
        if not names or fn.__name__ in names:
            fn()
            print('ok', fn.__name__)
