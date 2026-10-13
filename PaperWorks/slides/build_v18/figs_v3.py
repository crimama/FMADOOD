"""Method diagrams replacing tables in the v18 deck (project .venv python).

Content follows PaperWorks/outlines/Outline_v18.md §3.1–3.2. Token grids are drawn on the cached hazelnut query
image; token selection in the bag diagram is an illustrative random draw (seeded), not the run's actual indices.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'charts'
CACHE = Path('/home/hun/Volume/RESEARCH/FSAD-TTA/PaperWorks/figures/cache_rsg4_mechanism')
for f in (HERE / 'fonts').glob('NanumSquare_ac*.ttf'):
    font_manager.fontManager.addfont(str(f))
FAMILY = font_manager.FontProperties(fname=str(HERE / 'fonts' / 'NanumSquare_acR.ttf')).get_name()
NAVY, INK, SUB, LIGHT, PALE, ORANGE = '#163866', '#242B35', '#5F6B7A', '#A9B1BC', '#D3D8DE', '#C67A23'
FILL_N, FILL_G = '#EEF2F7', '#F5F6F8'
plt.rcParams.update({'font.family': FAMILY, 'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False})


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, title, sub='', edge=NAVY, fill='white', tsize=12.5, ssize=10.5, tcolor=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06', facecolor=fill,
                                edgecolor=edge, lw=1.3))
    if sub:
        ax.text(x + w / 2, y + h * .66, title, ha='center', va='center', fontsize=tsize, fontweight='bold',
                color=tcolor or edge)
        ax.text(x + w / 2, y + h * .3, sub, ha='center', va='center', fontsize=ssize, color=INK, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=tsize, fontweight='bold',
                color=tcolor or edge)


def arrow(ax, x0, y0, x1, y1, color=SUB, dashed=False):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.4, linestyle='--' if dashed else '-',
                                mutation_scale=12))


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=200, facecolor='white', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def two_stage():
    fig, ax = canvas(11.9, 4.5, 12, 4.6)
    W, G = 1.82, .2
    xs = [.04 + i * (W + G) for i in range(6)]

    def lane(y, h, items, shaded):
        for i, (t, sb) in enumerate(items):
            box(ax, xs[i], y, W, h, t, sb, fill=FILL_N if i in shaded else 'white', ssize=10)
            if i:
                arrow(ax, xs[i - 1] + W + .01, y + h / 2, xs[i] - .01, y + h / 2)
    ax.text(.04, 4.55, '① 피팅 — support마다 한 번 (정상 이미지만, 레이블·학습 없음)', fontsize=13.5, fontweight='bold',
            color=NAVY, va='top')
    lane(2.95, 1.05, [('정상 support', 'N장 (1–16)'), ('특징 추출', '4 해상도 × 3 계층'), ('메모리 배깅', 'bag 4개\n× 이미지당 256토큰'),
                      ('48개 bank', 'B_m, m = (l, r, b)'), ('LSO 매칭', '자기 이미지는 빼고\n나머지와 비교'),
                      ('단위 c_m', 'Q0.95(LSO 거리)')], (3, 5))
    ax.text(.04, 2.42, '② 추론 — query마다 같은 규칙 (고정된 bank와 c_m 사용)', fontsize=13.5, fontweight='bold', color=NAVY,
            va='top')
    lane(.2, 1.25, [('query', 'x^q'), ('특징 추출', '4 해상도 × 3 계층'), ('1-NN 거리', '48개 bank마다\nd_m(x^q, p)'),
                    ('단위 정렬', 'z_m = d_m / c_m'), ('계층 결합', '(r, b)마다\nz_d + 0.125·z5'),
                    ('맵 · 점수', 'bag·해상도 평균 → A\n상위 1% 평균 → s')], (3, 4))

    def elbow(x0, x1, ylev, label):
        ax.plot([x0, x0, x1], [2.93, ylev, ylev], color=NAVY, lw=1.3, ls='--')
        arrow(ax, x1, ylev, x1, 1.47, NAVY, dashed=True)
        ax.text(x1 + .08, (ylev + 1.47) / 2, label, fontsize=11, color=NAVY, va='center')
    elbow(xs[3] + W * .35, xs[2] + W / 2, 1.85, 'bank B_m')
    elbow(xs[5] + W / 2, xs[3] + W / 2, 1.70, 'c_m')
    save(fig, 'two_stage')


COVER = {448: '3.1', 672: '2.1', 896: '1.6', 1120: '1.25'}  # Outline §3.2 table


def resolution_grids():
    d = np.load(CACHE / 'ad_hazelnut__clean.npz')
    img = d['rgb']
    h, w = img.shape[:2]
    # crop: a square of 1/4 of the short side around the crack rim (display only)
    side = min(h, w) // 4
    cy, cx = int(h * .30), int(w * .55)
    crop = img[cy - side // 2: cy + side // 2, cx - side // 2: cx + side // 2]
    fig, axes = plt.subplots(1, 5, figsize=(11.9, 3.0), gridspec_kw={'width_ratios': [1.15, 1, 1, 1, 1]})
    axes[0].imshow(img)
    axes[0].add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side, fill=False, edgecolor='#5CE1E6', lw=2))
    axes[0].set_title('query (확대 영역 = 짧은 변의 1/4)', fontsize=11.5)
    for ax, r in zip(axes[1:], [448, 672, 896, 1120]):
        n = r // 14 // 4  # tokens across the crop
        ax.imshow(crop, extent=(0, n, n, 0))
        for k in range(n + 1):
            ax.plot([k, k], [0, n], color='white', lw=.8, alpha=.85)
            ax.plot([0, n], [k, k], color='white', lw=.8, alpha=.85)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_title(f'{r} px · 토큰 {n} × {n}\n(토큰 1개 = 짧은 변의 약 {COVER[r]}%)', fontsize=11.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LIGHT)
    fig.tight_layout(w_pad=.8)
    save(fig, 'resolution_grids')


def bag_sampling():
    rng = np.random.default_rng(0)
    n = 80  # 1120 px: 80 tokens along the short side (square basis 6,400)
    fig = plt.figure(figsize=(11.9, 3.5))
    gs = fig.add_gridspec(1, 7, width_ratios=[1.25, .25, 1, 1, 1, 1, .1], wspace=.12)
    d = np.load(CACHE / 'ad_hazelnut__clean.npz')
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(d['rgb'])
    ax0.set_title('support 이미지 1장\n(1120 px → 80 × 80 토큰)', fontsize=11.5)
    ax0.axis('off')
    axa = fig.add_subplot(gs[1])
    axa.axis('off')
    axa.annotate('', xy=(1, .5), xytext=(0, .5), xycoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color=SUB,
                                                                                         lw=1.6, mutation_scale=14))
    for k in range(4):
        ax = fig.add_subplot(gs[2 + k])
        sel = np.zeros(n * n)
        sel[rng.choice(n * n, 256, replace=False)] = 1
        ax.imshow(np.ones((n, n, 3)) * .96, extent=(0, n, n, 0))
        ys, xs = np.divmod(np.flatnonzero(sel), n)
        ax.scatter(xs + .5, ys + .5, s=4.5, color=NAVY, marker='s', linewidths=0)
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(LIGHT)
        ax.set_title(f'bag {k + 1} (시드 {k + 1})\n256토큰 선택', fontsize=11.5)
    fig.text(.62, .02, '같은 bag에서는 L5·L11·L23이 같은 위치 토큰을 공유  ·  bag마다 bank와 단위 c_m을 따로 구함  ·  '
             '16-shot bank 1개 = 16 × 256 = 4,096토큰', ha='center', fontsize=11, color=SUB)
    save(fig, 'bag_sampling')


if __name__ == '__main__':
    two_stage()
    resolution_grids()
    bag_sampling()
    print('ok')
