"""Equation panels and the bank-construction schematic for the reinforced v18 deck (project .venv python).

Formulas are transcribed from PaperWorks/outlines/Outline_v18.md §3.1–3.5 (mathtext rendering only).
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'charts'
for f in (HERE / 'fonts').glob('NanumSquare_ac*.ttf'):
    font_manager.fontManager.addfont(str(f))
FAMILY = font_manager.FontProperties(fname=str(HERE / 'fonts' / 'NanumSquare_acR.ttf')).get_name()
NAVY, INK, SUB, LIGHT, PALE = '#163866', '#242B35', '#5F6B7A', '#A9B1BC', '#D3D8DE'
plt.rcParams.update({'font.family': FAMILY, 'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False})


def eq_panel(name, rows, w=11.9, row_h=.78, label_w=.21):
    """rows: (label, formula_mathtext, explanation[, tall]). Tall rows (sums, limits) get extra height."""
    hs = [(1.2 if (len(r) > 3 and r[3]) else row_h) for r in rows]
    H = sum(hs) + .05
    fig = plt.figure(figsize=(w, H))
    top = H
    for i, (r, h) in enumerate(zip(rows, hs)):
        label, formula, desc = r[:3]
        tall = len(r) > 3 and r[3]
        fy = (top - (.47 if tall else .3)) / H
        dy = (top - h + .16) / H
        fig.text(.01, (top - h / 2) / H, label, fontsize=14, fontweight='bold', color=NAVY, va='center')
        fig.text(label_w, fy, formula, fontsize=17, color=INK, va='center')
        fig.text(label_w, dy, desc, fontsize=12.5, color=SUB, va='center')
        if i:
            fig.add_artist(plt.Line2D([.01, .99], [top / H, top / H], color='#E6E9ED', lw=1))
        top -= h
    fig.savefig(OUT / f'{name}.png', dpi=200, facecolor='white', bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def equations():
    eq_panel('eq_bank', [
        ('컴포넌트', r'$m=(l,r,b),\ \ l\in\{5,11,23\},\ r\in\{448,672,896,1120\},\ b\in\{1,2,3,4\}$',
         '3 × 4 × 4 = 48개 관측 단위. 같은 support·backbone을 공유하며 통계적 독립성은 가정하지 않음'),
        ('1-NN 거리', r'$d_{l,r,b}(x^q,p)=\min_{u\in B_{l,r,b}}\,[\,1-f_{l,r}(x^q,p)^{\top}u\,]$',
         'query 위치 p의 L2 정규화 특징과 해당 bank 토큰 사이의 최소 코사인 거리', True),
    ])
    eq_panel('eq_scale', [
        ('공통 연산자', r'$\mathcal{T}_q[t]=t\,/\,Q_q(\mathcal{N}_S[t])$',
         '관측 통계량 t를 정상 support 기준 분포의 q-분위수로 나눔 (Support-Referenced Ratio Operator)'),
        ('① LSO 분포', r'$D_m^{\mathrm{LSO}}=\bigcup_{i=1}^{N}\{\min_{u\in B_m\setminus B_m^{(i)}}[1-v^{\top}u]\ :\ v\in B_m^{(i)}\}$',
         'support 이미지 i의 토큰을 나머지 이미지의 토큰과만 매칭 (16-shot: 거리 4,096개)', True),
        ('② 단위 척도', r'$c_m=Q_{0.95}(D_m^{\mathrm{LSO}})$', '정상 support 거리의 상위 5% 기준점을 그 bank의 단위로 사용'),
        ('③ query 거리', r'$d_m(x^q,p)=\min_{u\in B_m}[\,1-f_m(x^q,p)^{\top}u\,]$', '쿼리 특징과 bank의 1-NN 코사인 거리 (정상 reference와의 특징 차이)', True),
        ('④ 정규화 점수', r'$z_m(x^q,p)=\mathcal{T}_{0.95}[d_m]=d_m(x^q,p)\,/\,c_m$', '무차원 점수: z = 1이면 정상 상위 5% 경계, z = 2면 그 2배'),
    ], row_h=.74)
    eq_panel('eq_control', [
        ('① Deep 기준선', r'$z_{d,r,b}(x^q,p)=\frac{1}{2}\,[\,z_{11,r,b}(x^q,p)+z_{23,r,b}(x^q,p)\,]$', 'Layer 11·23 정규화 점수의 산술 평균'),
        ('② Shallow 가산', r'$S_{r,b}(x^q,p)=z_{d,r,b}(x^q,p)+\lambda\,z_{5,r,b}(x^q,p),\quad\lambda=0.125$',
         'Layer 5 정규화 점수에 고정 가중치를 곱해 더함 — 각 (해상도 r, bag b)에서 독립적으로 계산'),
    ], row_h=.86)
    eq_panel('eq_aggregate', [
        ('① Bag 평균', r'$S_r(x^q,p)=\frac{1}{4}\sum_{b=1}^{4}S_{r,b}(x^q,p)$', '해상도 r에서 4개 bag 점수를 같은 토큰 위치 p에서 평균', True),
        ('② 다해상도 평균', r'$A(x^q)=\frac{1}{|\mathcal{R}|}\sum_{r\in\mathcal{R}}\mathrm{Resize}_{H,W}[\,S_r(x^q)\,]$',
         '해상도별 맵을 원본 크기 (H, W)로 bilinear 보간한 뒤 동일 가중 평균 → 픽셀 결함 맵', True),
        ('③ 이미지 점수', r'$s(x^q)=\frac{1}{|\Omega_{1\%}|}\sum_{p\in\Omega_{1\%}}A(x^q,p)$',
         'Ω1% = 최종 맵의 상위 1% 픽셀. SuperAD·PatchCore의 I-AUROC 집계와 같은 정의', True),
    ], row_h=.84)


def bank_schematic(w=11.9, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.6)
    ax.axis('off')

    def box(x, y, bw, bh, text, edge=LIGHT, fill='white', color=INK, size=11, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle='round,pad=0,rounding_size=0.05', facecolor=fill,
                                    edgecolor=edge, lw=1.2))
        ax.text(x + bw / 2, y + bh / 2, text, ha='center', va='center', fontsize=size, color=color,
                fontweight='bold' if bold else 'normal')

    def arrow(x0, y0, x1, y1, color=SUB):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=color, lw=1.4))
    ax.text(.05, 3.5, '기존 패치 매칭: 특징을 통합해 bank 하나', fontsize=13, color=SUB, fontweight='bold', va='top')
    for i, l in enumerate(['L5', 'L11', 'L23']):
        box(.1 + i * .6, 2.0, .54, .5, l)
    arrow(1.95, 2.25, 2.3, 2.25)
    box(2.35, 1.95, 1.45, .6, '[L5 | L11 | L23]\n통합 bank 1개', size=10.5)
    arrow(3.85, 2.25, 4.2, 2.25)
    box(4.25, 1.95, 1.1, .6, '점수 1개', size=10.5)
    ax.text(.1, 1.55, '· 계층마다 다른 정상 변동이 한 거리 안에 섞임\n· 토큰 표본(coreset 등)도 하나', fontsize=11, color=SUB, va='top')
    ax.plot([5.75, 5.75], [.15, 3.45], color='#E6E9ED', lw=1.5)
    ax.text(6.0, 3.5, 'ScopeAD: 특징 × 해상도 × bag마다 독립 bank', fontsize=13, color=NAVY, fontweight='bold', va='top')
    res = ['448', '672', '896', '1120']
    x0, y0, cw, chh = 6.75, .95, .62, .5
    for j, r in enumerate(res):
        ax.text(x0 + j * cw + cw / 2, y0 + 3 * chh + .12, r, ha='center', fontsize=10.5, color=SUB)
    for i, l in enumerate(['L5', 'L11', 'L23']):
        ax.text(x0 - .1, y0 + (2 - i) * chh + chh / 2, l, ha='right', va='center', fontsize=10.5, color=SUB)
        for j in range(4):
            for k in range(4):  # 4 bags drawn as a small stack
                ax.add_patch(Rectangle((x0 + j * cw + .06 + k * .035, y0 + (2 - i) * chh + .07 + k * .03), .38, .3,
                                       facecolor='white' if k < 3 else '#EEF2F7', edgecolor=NAVY, lw=.8))
    ax.text(x0 + 2 * cw, y0 - .25, '각 칸 = bag 4개 (이미지당 256토큰 × 4 시드) → 3 × 4 × 4 = 48 bank', ha='center',
            fontsize=10.5, color=SUB)
    arrow(9.35, 1.7, 9.7, 1.7, NAVY)
    box(9.75, 1.95, 2.1, .62, 'bank별 단위 c_m\n= Q0.95(LSO 거리)', edge=NAVY, fill='#EEF2F7', color=NAVY, size=10.5, bold=True)
    box(9.75, .95, 2.1, .62, 'z = d / c_m 정렬 후\nz_d + 0.125·z5', edge=NAVY, fill='#EEF2F7', color=NAVY, size=10.5,
        bold=True)
    arrow(10.8, 1.93, 10.8, 1.6, NAVY)
    fig.savefig(OUT / 'bank_schematic.png', dpi=200, facecolor='white', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


if __name__ == '__main__':
    equations()
    bank_schematic()
    print('ok')
