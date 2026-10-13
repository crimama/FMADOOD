"""Introduction figures for the aligned problem -> motivation -> contribution story (project .venv python).

alignment: text follows Outline_v18 §1.1–1.3. lso_scales: real cached support LSO distances
(cache_rsg4_mechanism/overview_walnuts__support.npz; AD2 walnuts, 16-shot, r = 672, bag 0, 4,096 values / layer).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'charts'
CACHE = Path('/home/hun/Volume/RESEARCH/FSAD-TTA/PaperWorks/figures/cache_rsg4_mechanism')
for f in (HERE / 'fonts').glob('NanumSquare_ac*.ttf'):
    font_manager.fontManager.addfont(str(f))
FAMILY = font_manager.FontProperties(fname=str(HERE / 'fonts' / 'NanumSquare_acR.ttf')).get_name()
NAVY, INK, SUB, LIGHT, PALE, GRID = '#163866', '#242B35', '#5F6B7A', '#A9B1BC', '#D3D8DE', '#E6E9ED'
FILL = '#EEF2F7'
plt.rcParams.update({'font.family': FAMILY, 'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=200, facecolor='white', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def _box(ax, x, y, w, h, text, edge=LIGHT, fill='white', color=INK, size=11.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05', facecolor=fill,
                                edgecolor=edge, lw=1.2))
    ax.text(x + .14, y + h / 2, text, ha='left', va='center', fontsize=size, color=color, linespacing=1.3,
            fontweight='bold' if bold else 'normal')


def _arrow(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=1.5, mutation_scale=13))


def alignment():
    fig, ax = plt.subplots(figsize=(11.9, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.45)
    ax.axis('off')
    cols = [(.55, 3.25), (4.12, 3.72), (8.08, 3.88)]
    for (x, w), head in zip(cols, ['문제 정의', '동기 (근거)', '기여 · 방법 방향']):
        ax.text(x + .05, 5.4, head, fontsize=13.5, fontweight='bold', color=NAVY, va='top')
    _box(ax, .55, 4.48, 11.4, .48, '배경   정상 몇 장 + 동결 백본의 패치 매칭 — 큰 거리가 결함인지 support에 없던 정상 변동인지 모호함 '
         '(High-Score Ambiguity)', edge=PALE, fill='#F5F6F8', color=SUB, size=11)
    rows = [
        ('①', '정상 이미지가 적어, 미세 결함과\n정상 변동을 가를 단서가 부족', '다중 깊이·척도 관찰이 dense prediction을\n개선함 (DPT, S², FeatUp)',
         'Observation Expansion', '4 해상도 × 3 계층으로 관측 확장', '최선 단일 해상도 대비 +4.95 / +2.10pp'),
        ('②-a', '늘린 관측은 거리 척도가 다르고,\n한 bank로 합치면 정상 변동이 섞임', '결합 전에 점수 척도를 데이터 기준으로\n맞춰야 함 (Kim 2021, ViM 2022)',
         '관측별 bank + Scale Calibration', '정상 support LSO 거리의 q95를 1로', '단일 bank 대비 +2.09pp, MAD 대비 최대 +1.69pp'),
        ('②-b', 'Shallow 점수에는 결함뿐 아니라\n정상 질감 반응도 섞임', 'Deep은 안정적 기준선, Shallow는 민감하나\n정상 외관에도 반응 (Raghu 2021)',
         'Deep-Anchored Weighting', 'S = z_d + 0.125·z5', 'Shallow 전체 가산(λ = 1) 대비 +5.55pp'),
    ]
    y0, h, gap = 3.2, 1.0, .18
    for i, (tag, prob, mot, name, how, ev) in enumerate(rows):
        y = y0 - i * (h + gap)
        ax.text(.25, y + h / 2, tag, fontsize=14, fontweight='bold', color=NAVY, ha='center', va='center')
        _box(ax, cols[0][0], y, cols[0][1], h, prob, size=11)
        _arrow(ax, cols[0][0] + cols[0][1] + .03, cols[1][0] - .03, y + h / 2)
        _box(ax, cols[1][0], y, cols[1][1], h, mot, color=SUB, size=11)
        _arrow(ax, cols[1][0] + cols[1][1] + .03, cols[2][0] - .03, y + h / 2)
        _box(ax, cols[2][0], y, cols[2][1], h, '', edge=NAVY, fill=FILL)
        x = cols[2][0] + .14
        ax.text(x, y + h * .76, name, fontsize=11.5, fontweight='bold', color=NAVY, va='center')
        ax.text(x, y + h * .48, how, fontsize=11, color=INK, va='center')
        ax.text(x, y + h * .2, ev, fontsize=9.5, color=SUB, va='center')
    ax.plot([.1, 11.95], [.72, .72], color=GRID, lw=1.2)
    ax.text(.25, .35, '③', fontsize=14, fontweight='bold', color=NAVY, ha='center', va='center')
    ax.text(.55, .35, '검증   관측 확장(구성) 이득과 결합 이득을 통제 실험으로 분리  ·  AD2 16-shot 3-seed AUPRO@0.05 SuperAD 대비 +4.17pp',
            fontsize=12, color=INK, va='center')
    save(fig, 'alignment')


def lso_scales(w=11.9, h=3.3):
    d = np.load(CACHE / 'overview_walnuts__support.npz')
    layers = [('L5 (Shallow)', d['lso_l5'], '#C67A23'), ('L11', d['lso_l11'], '#7A93B5'), ('L23', d['lso_l23'], NAVY)]
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    for ax, norm, title in ((axes[0], False, '원시 정상 거리 d (LSO)'), (axes[1], True, 'q95로 나눈 뒤 z = d / c')):
        for name, v, c in layers:
            q = np.quantile(v, .95)
            x = v / q if norm else v
            ax.hist(x, bins=70, range=(0, 2.0 if norm else .5), histtype='step', lw=1.8, color=c, density=True,
                    label=f'{name}  q95 = {q:.3f}' if not norm else name)
            ax.axvline(1.0 if norm else q, color=c, ls='--', lw=1.3)
        ax.set_title(title, fontsize=13)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=11, frameon=False, loc='upper right')
    axes[0].set_xlabel('코사인 거리 (점선 = 계층별 q95)', fontsize=11)
    axes[1].set_xlabel('정규화 점수 (점선 = 공통 기준 1)', fontsize=11)
    fig.tight_layout(w_pad=3)
    save(fig, 'lso_scales')


if __name__ == '__main__':
    alignment()
    lso_scales()
    print('ok')
