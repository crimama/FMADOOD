"""Charts and figure crops for the SCARF v16.3 deck (run with the project .venv python).

All numbers come from PaperWorks/Outline_v16.3.md (tables 4–12). Figures are sized in slide inches so that
font sizes map 1:1 onto the slide.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'charts'
OUT.mkdir(exist_ok=True)
FIG = Path('/home/hun/Volume/RESEARCH/FSAD-TTA/PaperWorks/figures')

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
    'legend.frameon': False, 'savefig.dpi': 200, 'axes.unicode_minus': False,
})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=.04, facecolor='white')
    plt.close(fig)


def repel(values, gap):
    """Spread label y-positions so neighbours are at least `gap` apart (keeps order)."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for _ in range(50):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            if pos[b] - pos[a] < gap:
                mid = (pos[a] + pos[b]) / 2
                pos[a], pos[b] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    return pos


# ------------------------------------------------------------------ results
SHOTS = [1, 2, 4, 8, 16]
AUPRO = {  # table 4, AUPRO@0.05 means
    'SCARF': [54.23, 55.62, 58.36, 58.40, 59.63],
    'SuperAD': [49.96, 50.56, 53.68, 54.73, 55.16],
    'PatchCore': [36.78, 40.07, 44.41, 45.73, 49.16],
    'AnomalyDINO': [39.92, 42.36, 43.40, 44.28, 44.36],
    'SubspaceAD': [37.31, 38.28, 39.22, 39.38, 39.54],
    'RAD': [23.32, 26.85, 30.28, 32.96, 33.54],
    'RegAD': [12.73, 15.05, 17.40, 19.25, 20.28],
    'FastRecon': [17.96, 18.93, 20.00, 18.09, 17.31],
}


def style_of(name):
    if name == 'SCARF':
        return dict(color=NAVY, lw=2.8, marker='o', ms=8, zorder=5)
    if name == 'SuperAD':
        return dict(color=DARK, lw=2.0, marker='s', ms=7, zorder=4)
    return dict(color=LIGHT, lw=1.5, marker='o', ms=5, zorder=3)


def main_curve(w=11.9, h=3.95):
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(SHOTS))
    for name, ys in AUPRO.items():
        ax.plot(x, ys, **style_of(name))
    ends = repel([v[-1] for v in AUPRO.values()], 3.2)
    for (name, ys), y in zip(AUPRO.items(), ends):
        strong = name in ('SCARF', 'SuperAD')
        label = f'{name}  {ys[-1]:.2f}' if strong else name
        ax.text(x[-1] + .12, y, label, va='center', fontsize=13 if strong else 12,
                color=NAVY if name == 'SCARF' else (DARK if strong else SUB),
                fontweight='bold' if strong else 'normal')
    for xi, (a, b) in enumerate(zip(AUPRO['SCARF'], AUPRO['SuperAD'])):
        ax.annotate(f'+{a - b:.2f}', (xi, a), xytext=(0, 9), textcoords='offset points', ha='center',
                    fontsize=12, color=NAVY, fontweight='bold')
    ax.set_xticks(x, [f'{s}-shot' for s in SHOTS])
    ax.set_xlim(-.25, len(SHOTS) - 1 + 1.25)
    ax.set_ylim(8, 64)
    ax.set_ylabel('AUPRO@0.05 (%)')
    ax.grid(axis='x', visible=False)
    ax.text(0, 61.5, '숫자 = SCARF - SuperAD (pp)', fontsize=11, color=SUB)
    save(fig, 'main_curve')


def mvtec_visa(w=11.9, h=3.6):
    data = {
        ('MVTec AD', 'I-AUROC'): {'SCARF': [87.57, 91.93, 93.83, 94.69], 'SuperAD': [92.54, 95.43, 96.65, 97.39],
                                  'PatchCore': [88.12, 92.67, 95.06, 96.13], 'FastRecon': [88.84, 93.51, 95.61, 96.76]},
        ('MVTec AD', 'P-AUROC'): {'SCARF': [96.63, 97.31, 97.66, 97.89], 'SuperAD': [96.93, 97.36, 97.70, 97.95],
                                  'PatchCore': [96.22, 96.97, 97.37, 97.80], 'FastRecon': [96.60, 97.29, 97.87, 98.16]},
        ('VisA', 'I-AUROC'): {'SCARF': [85.29, 89.46, 91.77, 92.81, 93.10], 'SuperAD': [90.29, 92.20, 92.94, 94.50, 95.19],
                              'PatchCore': [86.09, 87.87, 89.52, 92.36, 93.64], 'FastRecon': [84.65, 87.67, 89.21, 91.53, 93.38]},
        ('VisA', 'P-AUROC'): {'SCARF': [96.73, 97.46, 97.94, 98.34, 98.56], 'SuperAD': [97.06, 97.68, 98.02, 98.29, 98.43],
                              'PatchCore': [95.56, 96.29, 96.92, 97.18, 97.30], 'FastRecon': [96.75, 97.43, 97.91, 98.11, 98.32]},
    }
    marks = {'SCARF': dict(color=NAVY, lw=2.6, marker='o', ms=7, zorder=5),
             'SuperAD': dict(color=DARK, lw=1.9, marker='s', ms=6, zorder=4),
             'PatchCore': dict(color=LIGHT, lw=1.5, marker='^', ms=6, zorder=3),
             'FastRecon': dict(color=LIGHT, lw=1.5, marker='D', ms=5, ls='--', zorder=3)}
    fig, axes = plt.subplots(1, 4, figsize=(w, h))
    for ax, ((ds, metric), series) in zip(axes, data.items()):
        for name, ys in series.items():
            ax.plot(range(len(ys)), ys, label=name, **marks[name])
        n = len(series['SCARF'])
        ax.set_xticks(range(n), [str(s) for s in SHOTS[:n]])
        ax.set_title(f'{ds} · {metric}', fontsize=13)
        ax.set_xlabel('shot', fontsize=11)
        ax.grid(axis='x', visible=False)
        ax.tick_params(labelsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.09), fontsize=12)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'mvtec_visa')


def private_dots(w=11.9, h=3.6):
    rows = [  # (label, AU-PRO priv/mix, SegF1 priv/mix)
        ('SCARF (16장)', (64.80, 62.49), (34.84, 30.68)),
        ('RoBiS (full)', (67.25, 59.65), (51.00, 46.52)),
        ('SuperADD (full)', (61.36, 59.07), (57.42, 54.35)),
        ('SuperAD (full)', (60.51, 58.37), (47.18, 42.51)),
        ('EfficientAD-S (full)', (30.8, 19.2), (15.4, 8.0)),
        ('PatchCore (full)', (28.8, 26.0), (3.7, 3.4)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(w, h), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, idx, title in ((axes[0], 1, 'AU-PRO@0.05 (연속 점수 맵)'), (axes[1], 2, 'SegF1 (고정 임계값 적용)')):
        for yi, row in zip(y, rows):
            priv, mix = row[idx]
            c = NAVY if yi == y[0] else DARK
            ax.plot([mix, priv], [yi, yi], color=PALE, lw=3, zorder=2, solid_capstyle='round')
            ax.scatter([priv], [yi], s=80, color=c, zorder=4)
            ax.scatter([mix], [yi], s=80, facecolor='white', edgecolor=c, lw=2, zorder=4)
            if yi == y[0]:
                ax.annotate(f'{priv:.2f} / {mix:.2f}', (max(priv, mix), yi), xytext=(10, 0),
                            textcoords='offset points', va='center', fontsize=12, color=NAVY, fontweight='bold')
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, 80)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('%', fontsize=11)
    axes[0].set_yticks(y, [r[0] for r in rows])
    axes[0].get_yticklabels()[0].set_color(NAVY)
    axes[0].get_yticklabels()[0].set_fontweight('bold')
    axes[0].scatter([], [], s=70, color=DARK, label='TESTpriv (일반 조명)')
    axes[0].scatter([], [], s=70, facecolor='white', edgecolor=DARK, lw=2, label='TESTpriv,mix (혼합 조명)')
    fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(.55, 1.08), fontsize=12)
    fig.tight_layout(w_pad=2)
    save(fig, 'private_dots')


def corruption_dots(w=11.9, h=3.4):
    conds = ['Clean', 'Brightness', 'Contrast', 'Defocus blur', 'Gaussian noise']
    data = {'SCARF': [85.36, 84.42, 83.15, 81.92, 69.01], 'SuperAD': [84.49, 83.68, 82.27, 82.17, 71.10],
            'PatchCore': [83.97, 83.43, 77.15, 81.58, 74.03], 'FastRecon †': [85.74, 77.53, 69.72, 82.35, 57.70]}
    marks = {'SCARF': dict(color=NAVY, marker='o', s=110, zorder=5),
             'SuperAD': dict(color=DARK, marker='s', s=70, zorder=4),
             'PatchCore': dict(color=LIGHT, marker='^', s=80, zorder=3),
             'FastRecon †': dict(color=LIGHT, marker='D', s=55, zorder=3)}
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(conds))[::-1]
    for name, xs in data.items():
        ax.scatter(xs, y, label=name, **marks[name])
    for yi, v in zip(y, data['SCARF']):
        ax.annotate(f'{v:.2f}', (v, yi), xytext=(0, 11), textcoords='offset points', ha='center',
                    fontsize=11, color=NAVY, fontweight='bold')
    ax.set_yticks(y, conds)
    ax.set_xlim(55, 90)
    ax.set_xlabel('AUPRO@0.05 (%)')
    ax.grid(axis='y', visible=False)
    ax.axhspan(-.5, .5, color='#F3F4F6', zorder=0)
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.2), fontsize=12)
    save(fig, 'corruption_dots')


# ------------------------------------------------------------------ ablation / analysis
def ablation_dots(w=11.9, h=3.55):
    rows = [('단순 전체 가산 (MAD 단위)', 51.16, '-7.51'), ('+ q95 척도 보정', 54.02, '-4.65'),
            ('+ 잔차 게이트 (MAD 단위)', 57.48, '-1.19'), ('SCARF = q95 보정 + 게이트', 59.15, '+0.47')]
    deep = 58.67
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(rows))[::-1]
    ax.axvline(deep, color=DARK, ls='--', lw=1.6, zorder=2)
    ax.text(deep, y[0] + .55, f'Deep-only {deep:.2f}', ha='center', fontsize=12, color=DARK)
    for yi, (label, v, delta) in zip(y, rows):
        strong = label.startswith('SCARF')
        c = NAVY if strong else DARK
        ax.plot([50, v], [yi, yi], color=PALE, lw=2, zorder=1)
        ax.scatter([v], [yi], s=140 if strong else 90, color=c, zorder=4)
        ax.annotate(f'{v:.2f}  ({delta})', (v, yi), xytext=(12, 0), textcoords='offset points', va='center',
                    fontsize=13, color=c, fontweight='bold' if strong else 'normal', zorder=6,
                    bbox=dict(boxstyle='square,pad=.15', fc='white', ec='none'))
    ax.set_yticks(y, [r[0] for r in rows])
    ax.get_yticklabels()[-1].set_color(NAVY)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xlim(50, 62.5)
    ax.set_ylim(-.5, len(rows) - .1)
    ax.set_xlabel('AUPRO@0.05 (%)   ·   괄호 = Deep-only 대비 (pp)')
    ax.grid(axis='y', visible=False)
    save(fig, 'ablation_dots')


def lowshot_dots(w=11.9, h=3.3):
    rows = [('MVTec AD 1-shot', 78.39, 79.50, 79.36, 72.68), ('MVTec AD 2-shot', 81.75, 82.21, 81.94, 75.75),
            ('MVTec AD 4-shot', 83.63, 84.10, 83.55, 79.32), ('VisA 1-shot', 72.59, 77.97, 75.69, 61.79)]
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(rows))[::-1]
    for yi, (_, g256, g1024, deep, add) in zip(y, rows):
        ax.plot([min(add, g256), max(g1024, deep)], [yi, yi], color=PALE, lw=2, zorder=1)
        ax.scatter([add], [yi], marker='X', s=90, color=LIGHT, zorder=3)
        ax.scatter([deep], [yi], marker='s', s=80, color=DARK, zorder=4)
        ax.scatter([g256], [yi], s=110, color=NAVY, zorder=5)
        ax.scatter([g1024], [yi], s=110, facecolor='white', edgecolor=NAVY, lw=2, zorder=5)
        ax.annotate(f'{g256 - deep:+.2f}', (g256, yi), xytext=(0, 11), textcoords='offset points',
                    ha='center', fontsize=11, color=NAVY, fontweight='bold')
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(60, 86)
    ax.set_ylim(-.6, len(rows) - .3)
    ax.set_xlabel('AUPRO@0.05 (%)   ·   숫자 = SCARF(cap256) - Deep-only (pp)')
    ax.grid(axis='y', visible=False)
    for kw, label in ((dict(marker='X', s=80, color=LIGHT), '전체 가산'), (dict(marker='s', s=70, color=DARK), 'Deep-only'),
                      (dict(s=90, color=NAVY), 'SCARF (cap256)'),
                      (dict(s=90, facecolor='white', edgecolor=NAVY, lw=2), 'SCARF (cap1024)')):
        ax.scatter([], [], label=label, **kw)
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(.45, 1.22), fontsize=12)
    save(fig, 'lowshot_dots')


def gate_bars(w=11.9, h=3.8):
    fig, axes = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={'width_ratios': [1, 1]})
    panels = [(axes[0], '결함 선택성 (전체 조건)', ['정상 픽셀', '결함 픽셀'], [1.15, 11.22], [LIGHT, NAVY], 13),
              (axes[1], '조명 변화 (정상 픽셀)', ['Regular', 'Shift 1'], [1.00, 1.34], [LIGHT, DARK], 13)]
    for ax, title, labels, vals, colors, top in panels:
        bars = ax.bar(labels, vals, color=colors, width=.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * .02, f'{v:.2f}%', ha='center', va='bottom',
                    fontsize=13, fontweight='bold', color=INK)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel('게이트가 열린 픽셀 (%)', fontsize=11)
        ax.grid(axis='x', visible=False)
    axes[0].text(.5, 7.2, '약 9.8배', ha='center', fontsize=13, color=NAVY, fontweight='bold')
    axes[1].text(.5, 3.2, '같은 장면 정상 쌍 기준 +0.27pp', ha='center', fontsize=12, color=DARK)
    fig.tight_layout(w_pad=3)
    save(fig, 'gate_bars')


def resolution_dots(w=11.9, h=3.35):
    strata = ['Full8 (전체)', 'Regular', 'Shift 1']
    cfg = [('4-해상도 (토큰 256)', [59.15, 61.26, 59.40], NAVY, 'o', 130),
           ('1120 단일 (토큰 256)', [56.96, 59.16, 56.89], LIGHT, 's', 90),
           ('1120 단일 (토큰 1024, 용량 일치)', [56.63, 59.20, 56.16], DARK, 'D', 80)]
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(strata))
    for k, (label, vals, c, m, s) in enumerate(cfg):
        xs = x + (k - 1) * .18
        ax.scatter(xs, vals, color=c, marker=m, s=s, label=label, zorder=4)
        if k == 0:
            for xi, v in zip(xs, vals):
                ax.annotate(f'{v:.2f}', (xi, v), xytext=(0, 10), textcoords='offset points', ha='center',
                            fontsize=12, color=NAVY, fontweight='bold')
    ax.set_xticks(x, strata)
    ax.set_xlim(-.5, len(strata) - .5)
    ax.set_ylim(55, 63)
    ax.set_ylabel('AUPRO@0.05 (%)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.22), fontsize=12)
    save(fig, 'resolution_dots')


def cost_bars(w=11.9, h=3.2):
    rows = [('DINOv2 특징 추출', 1516.7, 76.47), ('읽기 · 전처리 · 특징 정리', 238.3, 12.01),
            ('다해상도 최근접 검색', 92.7, 4.68), ('다해상도 공간 보간', 91.7, 4.62), ('CQ 후처리', 36.0, 1.82),
            ('척도 보정 · 잔차 결합 (제안)', 4.1, 0.21), ('기타', 3.9, 0.19)]
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(rows))[::-1]
    for yi, (label, ms, pct) in zip(y, rows):
        strong = '제안' in label
        ax.barh(yi, ms, color=NAVY if strong else (DARK if ms > 1000 else LIGHT), height=.62)
        ax.text(ms + 15, yi, f'{ms:,.1f} ms  ({pct:.2f}%)', va='center', fontsize=12,
                color=NAVY if strong else INK, fontweight='bold' if strong else 'normal')
    ax.set_yticks(y, [r[0] for r in rows])
    ax.get_yticklabels()[5].set_color(NAVY)
    ax.get_yticklabels()[5].set_fontweight('bold')
    ax.set_xlim(0, 2000)
    ax.set_xlabel('Query 1장당 시간 (ms), RTX 4090 · 전체 1,983.4 ms')
    ax.grid(axis='y', visible=False)
    save(fig, 'cost_bars')


# ------------------------------------------------------------------ concept / illustrative
def ambiguity_concept(w=11.9, h=3.2):
    x = np.linspace(0, 3.2, 600)

    def bump(mu, sd, amp):
        return amp * np.exp(-.5 * ((x - mu) / sd) ** 2)
    regular = bump(.55, .18, 1.0)
    shifted = bump(1.15, .25, .45)
    defect = bump(2.1, .35, .35)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.fill_between(x, regular, color=PALE, label='평소 조명의 정상 패치')
    ax.fill_between(x, shifted, color=LIGHT, alpha=.75, label='support에 없던 조명의 정상 패치')
    ax.plot(x, defect, color=NAVY, lw=2.6, label='결함 패치')
    ax.axvline(.85, color=DARK, ls=':', lw=1.8)
    ax.axvline(1.55, color=DARK, ls='--', lw=1.8)
    ax.text(.86, 1.05, 'support만 보면\n여기가 정상 경계', fontsize=12, color=DARK, va='top')
    ax.text(1.57, 1.05, '튀는 정상이 섞이면\n오경보 5%를 지키려고\n임계값이 여기까지 밀림', fontsize=12, color=DARK, va='top')
    ax.annotate('', xy=(1.53, .62), xytext=(.88, .62), arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    ax.text(2.45, .45, '임계값 아래로 내려간\n약한 결함은 놓침', fontsize=12, color=NAVY, fontweight='bold')
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_xlabel('최근접 거리 (결함 점수)  →  클수록 이상')
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=12)
    ax.set_title('개념도 (가상 분포, 실측 아님)', fontsize=12, color=SUB, loc='left', fontweight='normal')
    save(fig, 'ambiguity_concept')


def fusion_example(w=11.9, h=2.85):
    groups = ['정상 매끈한 면', '정상 질감 변화 면', '미세 스크래치', '대형 파손']
    deep = [.55, .65, .85, 2.40]
    add = [.95, 1.90, 3.65, 5.40]
    scarf = [.55, .65, 2.35, 4.10]
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(groups))
    bw = .25
    for k, (vals, c, label) in enumerate(((deep, PALE, 'Deep만 (z_d)'), (add, LIGHT, '전체 가산 (z_d + z5)'),
                                          (scarf, NAVY, 'SCARF (z_d + 초과분)'))):
        bars = ax.bar(x + (k - 1) * (bw + .02), vals, bw, color=c, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .06, f'{v:.2f}', ha='center', fontsize=11,
                    color=NAVY if k == 2 else INK, fontweight='bold' if k == 2 else 'normal')
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 6.2)
    ax.set_ylabel('결합 점수 (z 단위)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_title('설명용 예시 값 (Outline §3.4 표, 실측 아님)  ·  τ5 = 1.30', fontsize=12, color=SUB, loc='right',
                 fontweight='normal')
    save(fig, 'fusion_example')


def formula(w=11.9, h=2.35):
    fig = plt.figure(figsize=(w, h))
    fig.text(.03, .80, '① 척도 보정', fontsize=15, fontweight='bold', color=NAVY)
    fig.text(.20, .80, r'$z_m = d_m \,/\, c_m$,    $c_m = Q_{0.95}$ (정상 support 거리)', fontsize=19, color=INK)
    fig.text(.20, .62, '→ z = 1 은 "정상 상위 5% 경계". 층·해상도가 달라도 같은 눈금', fontsize=13, color=SUB)
    fig.text(.03, .34, '② 기여 제어', fontsize=15, fontweight='bold', color=NAVY)
    fig.text(.20, .34, r'$S = z_d + \max(0,\; z_5 - \tau_5)$,    $\tau_5 = Q_{0.99}\,/\,Q_{0.95}$ (Layer 5)',
             fontsize=19, color=INK)
    fig.text(.20, .16, '→ Deep 점수는 항상 쓰고, Shallow는 "정상 상위 1%"를 넘는 만큼만 더함', fontsize=13, color=SUB)
    save(fig, 'formula')


# ------------------------------------------------------------------ crops of existing paper figures
def crop_overview():
    """fig_scarf_overview without the stale 'Noise reference' and 'Resolution route' boxes (not in v16.3)."""
    im = Image.open(FIG / 'fig_scarf_overview.png').convert('RGB')
    a = im.crop((0, 0, 2245, 440))        # panel A up to the LSO histogram
    b = im.crop((440, 535, 2800, 1245))   # panel B fusion box + map aggregation
    w = max(a.width, b.width)
    out = Image.new('RGB', (w, a.height + b.height + 20), 'white')
    out.paste(a, (0, 0))
    out.paste(b, (0, a.height + 20))
    out.save(OUT / 'overview_crop.png')


def crop_resolution_row():
    im = Image.open(FIG / 'fig_rsg4_mech_resolution.png').convert('RGB')
    row = im.crop((150, 170, im.width - 60, 650))
    patch = row.crop((24, 380, 150, 412))          # dark background just above the stale 'rho' label
    row.paste(patch, (24, 425))
    row.save(OUT / 'resolution_row.png')


def crop_task():
    im = Image.open(FIG / 'fig_rsg4_mech_fusion.png').convert('RGB')
    q = im.crop((278, 208, 697, 631))
    s = im.crop((2117, 208, 2537, 631))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.1))
    for ax, img, title in ((axes[0], q, '입력: query 이미지\n(하늘색 = 정답 결함 윤곽)'), (axes[1], s, '출력: 결함 점수 맵\n(밝을수록 결함)')):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight='normal')
        ax.axis('off')
    fig.tight_layout(w_pad=2)
    save(fig, 'task')


def strip_captions():
    """Drop the English caption lines above paper figures (the slide note carries the caption)."""
    for name, top in (('fig_rsg4_mech_fusion_ad2', 88), ('fig_rsg4_mech_scale', 75),
                      ('fig_rsg4_mech_fusion', 150), ('fig_rsg4_mech_bagging', 70)):
        im = Image.open(FIG / f'{name}.png').convert('RGB')
        im.crop((0, top, im.width, im.height)).save(OUT / f'{name}_nocap.png')


ALL = [strip_captions, main_curve, mvtec_visa, private_dots, corruption_dots, ablation_dots, lowshot_dots, gate_bars,
       resolution_dots, cost_bars, ambiguity_concept, fusion_example, formula, crop_overview,
       crop_resolution_row, crop_task]

if __name__ == '__main__':
    names = sys.argv[1:]
    for fn in ALL:
        if not names or fn.__name__ in names:
            fn()
            print('ok', fn.__name__)
