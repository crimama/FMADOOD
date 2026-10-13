"""Build the SCARF (Outline v16.3) presentation deck in the DSBA template language.

Main deck = figures / charts / bullets (tables only for summary, setup, pending checks, conclusion);
exact numbers live in Appendix tables. Charts come from charts.py (run first with the project .venv).
usage: python3 build_scarf_deck.py out.pptx
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/.claude/skills/dsba-slide-deck/scripts'))
from PIL import Image  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402
from deck_style import (BOLD, FONT, INK, LOGO, NAVY, SUB, XBOLD, _textbox, figure, frame_slide,  # noqa: E402
                        new_deck, style_run, table, text_box)
from deck_ops import set_sections  # noqa: E402

FIG = Path('/home/hun/Volume/RESEARCH/FSAD-TTA/PaperWorks/figures')
CH = Path(__file__).resolve().parent / 'charts'
OUT = sys.argv[1]
SRC = '출처: PaperWorks/outlines/Outline_v16.3.md'

# ---------------------------------------------------------------- page plan
ORDER = ['cover', 'toc', 'summary',
         'bg', 'ambig', 'dilemma', 'axes',
         'flow', 'bank', 'scale', 'fusion', 'fusion_map',
         'setup', 'main', 'mvtec', 'private', 'shift', 'corrupt',
         'ablation', 'lowshot', 'gate', 'resolution', 'cost',
         'discussion', 'pending', 'conclusion',
         'appx_aupro', 'appx_metrics', 'appx_settings', 'appx_mvtec', 'appx_private', 'appx_corrupt',
         'appx_ablation', 'appx_cost', 'appx_spec']
MAIN_LAST = ORDER.index('conclusion') + 1
P = {k: i + 1 for i, k in enumerate(ORDER)}
A = {k: f'A-{i + 1}' for i, k in enumerate(ORDER[MAIN_LAST:])}


def page(key):
    return A.get(key, P[key])


S1, S2, S3, S4, S5 = ('01. 문제 정의', '02. 방법: SCARF', '03. 실험 결과', '04. Ablation · 분석',
                      '05. 논의와 향후 계획')
SA = 'Appendix A. 상세 표'

prs = new_deck()
slides = {}


def em(text):
    return sum(1.0 if ord(c) > 0x1100 else (0.3 if c == ' ' else 0.6) for c in text)


def lint(key, title, lead, detail):
    if em(title) * 28 / 72 > 10.6:
        print(f'WARN title wraps  p{page(key)}: {title}')
    if em(lead) * 16 / 72 > 11.4:
        print(f'WARN lead wraps   p{page(key)}: {lead}')
    for line in detail:
        if em(line) * 16 / 72 > 11.6:
            print(f'WARN bullet wraps p{page(key)}: {line} ({em(line) * 16 / 72:.1f} in)')


def slide(key, section, title, lead, detail=(), note='', notes=''):
    lint(key, title, lead, detail)
    s, top = frame_slide(prs, section=section, title=title, lead=lead, detail=detail,
                         note=note, notes=notes, page=page(key))
    slides[key] = s
    return s, top


def stacked(s, paths, top, bottom=6.72, gap=.1):
    """Several figures stacked vertically at one common width, centred in the body area."""
    ratios = [Image.open(p).size[1] / Image.open(p).size[0] for p in paths]
    w = min(11.94, (bottom - top - gap * (len(paths) - 1)) / sum(ratios))
    y = top
    for p, r in zip(paths, ratios):
        figure(s, p, y, bottom=y + w * r)
        y += w * r + gap


def boxed(s, x, y, w, h, label, text):
    """The single bordered box allowed by the design spec ('라벨: 내용')."""
    sh = _textbox(s, x, y, w, h, 'KeyBox')
    sh.line.color.rgb = RGBColor.from_string(NAVY)
    sh.line.width = Pt(1)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(.2)
    tf.margin_top = tf.margin_bottom = Inches(.12)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = label + '  '; style_run(r, 15, NAVY, BOLD, True)
    r = p.add_run(); r.text = text; style_run(r, 15, INK, FONT, False)


# ---------------------------------------------------------------- cover, toc, summary
def cover():
    blank = next(l for l in prs.slide_layouts if l.name == '빈 화면')
    s = prs.slides.add_slide(blank)
    s.shapes.add_picture(str(LOGO), Inches(.9), Inches(1.15), Inches(1.55), Inches(.81)).name = 'Logo'
    text_box(s, .9, 2.25, 11.5, .8, 'SCARF', 44, NAVY, XBOLD, True, name='Title')
    text_box(s, .9, 3.10, 11.5, .5, 'Support-Calibrated Asymmetric Residual Fusion', 24, NAVY, BOLD, True,
             name='Subtitle')
    text_box(s, .9, 3.65, 11.5, .5, '소수 정상 이미지 기반 산업 결함 국소화를 위한 척도 보정·비대칭 잔차 결합', 18,
             INK, FONT, name='Subtitle2')
    line = s.shapes.add_connector(1, Inches(.9), Inches(4.45), Inches(12.4), Inches(4.45))
    line.line.color.rgb = RGBColor.from_string(NAVY)
    line.line.width = Inches(.015)
    text_box(s, .9, 4.65, 11.5, .4, '연구 아웃라인 v16.3 발표  ·  2026-10-03', 16, SUB, FONT, name='Date')
    s.notes_slide.notes_text_frame.text = (
        '근거 문서: PaperWorks/outlines/Outline_v16.3.md (CVPR/WACV 제출용 아웃라인). 발표자 이름·소속 표기는 확인 후 추가.')
    slides['cover'] = s


def toc():
    rng = lambda a, b: f'{P[a]}–{P[b]}쪽'  # noqa: E731
    slide('toc', '목차', '발표 순서',
          '흐름 │ 문제 → 방법 → 주 결과 → 소거·분석 → 남은 검증 순으로 진행한다',
          [f'01 문제 정의: 왜 "큰 거리 = 결함"이 아닌가, 관측을 늘리면 무엇이 문제인가 ({rng("bg", "axes")})',
           f'02 방법 SCARF: 척도 보정과 비대칭 잔차 결합의 계산과 실제 맵 ({rng("flow", "fusion_map")})',
           f'03 실험 결과: few-shot 주 비교, 공식 private, 조명·합성 손상 변화 ({rng("setup", "corrupt")})',
           f'04 Ablation · 분석: 어느 장치가 이득을 만드는가, 비용은? ({rng("ablation", "cost")})',
           f'05 논의와 향후 계획: 한계와 아직 비어 있는 검증 ({rng("discussion", "conclusion")})',
           f'Appendix A: 결과 표 원본, 비교군 설정, 고정 사양 (A-1–{A["appx_spec"]})'])


def summary():
    s, top = slide('summary', '요약', '한 장 요약',
                   '결론 │ Shallow를 "초과분만" 더하는 규칙으로 AD2 저-FPR 국소화에서 가장 높은 성능을 얻었다')
    rows = [
        ['문제', '소수 정상 support에서 큰 거리는 결함인가?', '정상 변동도 큰 거리를 내 저-FPR 오경보를 유발',
         f'{P["ambig"]}·{P["dilemma"]}'],
        ['방법', '층·해상도 점수를 어떻게 합칠까?', '정상 q95로 단위 통일 + Shallow는 q99 초과분만 가산',
         f'{P["scale"]}·{P["fusion"]}'],
        ['주 결과', 'AD2 few-shot에서 경쟁력은?', '1–16 shot AUPRO@0.05 1위, 16-shot SuperAD +4.47pp', str(P['main'])],
        ['강건성', '조명·손상 변화에서는?', '조명 Shift 1–3·밝기·대비는 안정, Gaussian noise는 약함',
         f'{P["shift"]}·{P["corrupt"]}'],
        ['소거', '이득은 어디서 오나?', '단위 보정 + 게이트가 함께 있어야 Deep-only 초과 (+0.47pp)', str(P['ablation'])],
        ['한계', '아직 확인 안 된 것은?', '저샷 순이득 불안정, 선형 가중·Hard Gate·q/β 민감도 미완',
         f'{P["lowshot"]}·{P["pending"]}'],
    ]
    table(s, .70, top + .05, [1.4, 3.6, 5.74, 1.2], ['섹션', '질문', '결론', '쪽'], rows, size=13, bottom=6.2)


# ---------------------------------------------------------------- 01 problem
def bg():
    s, top = slide('bg', S1, '배경: 정상 몇 장으로 결함 위치 찾기',
                   '목적 │ 재학습 없이 정상 1–16장만으로, 오경보가 적은 구간에서 결함 영역을 짚는다',
                   ['방식: 동결 DINOv2 패치를 정상 패치 뱅크와 비교 → 가장 가까운 정상 패치와의 거리 = 결함 점수',
                    '주 지표 AUPRO@0.05: 오경보율 5% 이하에서 결함 영역을 얼마나 덮는가',
                    '왜 저-FPR인가: 공정에서 오경보는 라인 정지 비용 → 낮은 FPR 구간 성능이 실사용 성능'],
                   note=f'예시: MVTec AD hazelnut, SCARF 결함 맵 (그림 M3에서 발췌). {SRC} §1.1, 표 2',
                   notes='용어: Support = 제품별 정상 이미지 1–16장(라벨 없음), Query = 검사할 이미지, '
                         'Bank = support 패치 특징 저장소, pAUROC@0.05 = 같은 FPR 구간의 픽셀 단위 ROC 면적.')
    figure(s, CH / 'task.png', top + .05)


def ambig():
    s, top = slide('ambig', S1, '문제 ①: 큰 거리가 항상 결함은 아니다',
                   '원인 │ 정상 16장은 조명·자세·반사 변화를 다 담지 못해 처음 보는 정상도 멀게 나온다',
                   ['비유: 맑은 날 사진만 본 검사원은 흐린 날 찍힌 정상품도 "처음 보는 모습"이라 의심함',
                    '관측의 한계: 거리값 하나로는 결함인지, support에 없던 정상 변동인지 가릴 수 없음',
                    '저-FPR에 치명적: 임계값은 정상 점수의 상위 꼬리로 정해짐 → 튀는 정상이 임계값을 끌어올림'],
                   note=f'그림은 개념도(가상 분포)이며 실측이 아님. High-Score Ambiguity. {SRC} §1.1',
                   notes='요구 사항: 정상 수준 신호는 점수 누적을 막고, 정상 범위를 명백히 넘는 반응만 반영.')
    figure(s, CH / 'ambiguity_concept.png', top + .05)


def dilemma():
    s, top = slide('dilemma', S1, '문제 ②: 얕은 층을 그냥 더하면 손해',
                   '딜레마 │ 얕은 층은 미세 결함을 잡지만 정상 질감에도 반응해, 그대로 더하면 결함이 묻힌다',
                   ['Shallow(Layer 5): 컵 테두리·젤리 경계 같은 정상 구조에도 강하게 반응 (세 번째 열)',
                    '전체 가산: 정상 점수가 함께 올라 임계값 상승 → 이 결함의 적중 98% → 20% (네 번째 열)',
                    '근거 (AD2 16-shot): q95로 단위를 맞춘 뒤에도 전체 가산은 AUPRO@0.05 58.67 → 54.02%'],
                   note='그림: PaperWorks/figures/fig_rsg4_mech_fusion_ad2.png (fruit_jelly, 16-shot). 녹색 = 시각화용 1%-FPR 임계값 초과 픽셀 '
                        '(해당 클래스 정상 test 이미지로 설정한 진단값, 방법에는 미사용). 수치 근거: 표 8.')
    figure(s, CH / 'fig_rsg4_mech_fusion_ad2_nocap.png', top + .05)


def axes():
    s, top = slide('axes', S1, '해법의 틀: 단위와 기여량을 따로 푼다',
                   '핵심 │ 점수 결합을 "척도 보정"과 "기여 제어" 두 축으로 나누고 둘 다 정상 support 통계로 정한다',
                   ['척도 보정: 층·해상도마다 거리 눈금이 다름 → 정상 상위 5% 값으로 나눔 (cm·inch 환산처럼)',
                    '기여 제어: 단위를 맞춰도 Shallow 정상 반응은 남음 → 상위 1%를 넘는 만큼만 (초과 성과급처럼)',
                    f'미리보기: 척도 보정만 54.02%, 둘 다 59.15% — Deep-only 58.67%를 넘는 유일한 구성 ({P["ablation"]}쪽)'],
                   note=f'Q_q = 정상 support 거리 분포(Leave-Source-Out)의 q 분위수. {SRC} §1.2–1.3, §3.3–3.4')
    figure(s, CH / 'formula.png', top + .25, bottom=min(6.6, top + 2.9))


# ---------------------------------------------------------------- 02 method
def flow():
    s, top = slide('flow', S2, 'SCARF 전체 흐름',
                   '구조 │ 정상 support로 기준값을 한 번 계산하고(A), query마다 같은 규칙으로 결합한다(B)',
                   ['A 피팅 (1회): 층 3 × 해상도 4 × bag 4 = 48개 뱅크 → 정상끼리 매칭(LSO)해 c_m, τ5 계산',
                    'B 추론: z = d / c → S = z_d + 초과분 → bag·4개 해상도 평균 → CQ 후처리 → 최종 맵'],
                   note='그림: fig_scarf_overview에서 v16.3에 없는 Noise reference·Resolution route 상자를 잘라낸 것 (walnuts, r = 672 예시). '
                        f'고정 사양 {A["appx_spec"]}. {SRC} §3.1')
    figure(s, CH / 'overview_crop.png', top + .02)


def bank():
    s, top = slide('bank', S2, '관측 구성: 4개 해상도 × 4개 bag',
                   '설계 │ 형태(저해상도)와 미세 결함(고해상도)을 함께 보고, bag 평균으로 표본 우연성을 줄인다',
                   ['해상도: 448 px 토큰 1개 = 짧은 변 3.1% (형태) ↔ 1120 px = 1.25% (미세 결함) → 4개 모두 사용',
                    'Bag: 이미지당 256토큰을 다른 시드로 4번 추출 → bag 간 픽셀 표준편차 평균 0.038, 평균으로 완화'],
                   note='위: 같은 query의 해상도별 점수 맵 (fig_rsg4_mech_resolution 윗줄). 아래: 그림 M1, bag별 맵과 평균·표준편차 '
                        f'(hazelnut 1120 px). {SRC} §3.2')
    stacked(s, [CH / 'resolution_row.png', CH / 'fig_rsg4_mech_bagging_nocap.png'], top + .02)


def scale():
    s, top = slide('scale', S2, 'Scale Calibration: 거리 단위 통일',
                   '방법 │ 각 거리를 정상 support 거리의 95% 분위수 c_m으로 나눠 z = 1을 공통 경계로 만든다',
                   ['같은 거리, 다른 뜻: d = 0.178은 Layer 5 기준(0.123)은 넘고 Layer 23 기준(0.232)은 못 넘음',
                    '환산 후: z5 = 1.44 > 1, z23 = 0.76 < 1 → 두 층을 같은 눈금에서 비교 (오른쪽 아래)'],
                   note=f'c_m = Q0.95(D_m^LSO), z_m = d_m / c_m. 그림 M2: hazelnut 1120 px, bag 평균, d*는 예시값. {SRC} §3.3')
    figure(s, CH / 'fig_rsg4_mech_scale_nocap.png', top + .02)


def fusion():
    s, top = slide('fusion', S2, 'Asymmetric Residual Fusion',
                   '방법 │ S = z_d + max(0, z5 − τ5): 정상 범위 안의 Shallow 반응은 0, 넘은 부분만 더한다',
                   ['Deep만: 미세 스크래치(0.85)가 정상 질감(0.65)과 거의 구분되지 않음',
                    '전체 가산: 정상 질감 면까지 1.90으로 올라가 오경보 위험',
                    'SCARF: 정상 질감은 0.65 그대로, 스크래치만 2.35로 올라가 둘이 확실히 갈림'],
                   note=f'τ5 = Q0.99/Q0.95 (Layer 5 정상 거리) ≥ 1. 막대 값은 설명용 예시(실측 아님). {SRC} §3.4',
                   notes='선형합 z_d + λ·z5와 달리 정상 변동을 0으로 잘라내 가중치 탐색이 필요 없음. '
                         'z5 ≤ τ5 → S = z_d, z5 > τ5 → S = z_d + (z5 − τ5).')
    figure(s, CH / 'fusion_example.png', top + .05)


def fusion_map():
    s, top = slide('fusion_map', S2, '실제 맵에서 본 잔차 결합',
                   '확인 │ 정상 껍질에서 초과분은 0이고, 균열 테두리에서만 켜져 Deep 맵에 더해진다',
                   ['Shallow 전체: 균열 테두리와 함께 정상 껍질 윤곽에도 반응 (z5 > 1인 픽셀 8.8%)',
                    '초과분: τ5 = 1.44 이하를 잘라내면 0.05 넘는 픽셀이 4.9%로 줄고 균열 쪽에 남음',
                    'SCARF: Deep 기준선은 그대로 두고 균열 테두리 반응만 보태 결함 경계를 보강'],
                   note=f'그림 M3: hazelnut 1120 px, bag 평균, 같은 색 범위. 하늘색 윤곽 = GT(표시용). {SRC} §3.4')
    figure(s, CH / 'fig_rsg4_mech_fusion_nocap.png', top + .1)


# ---------------------------------------------------------------- 03 results
def setup():
    s, top = slide('setup', S3, '평가 설정',
                   '설정 │ 주 비교는 조명 변화가 섞인 MVTec AD 2 test_public 8클래스, 1–16 shot',
                   ['Support: regular 조명 train/good에서 선택 → Shift 1–3 조명은 support에 없던 변화',
                    'Seed: SCARF·SuperAD·PatchCore는 2·16-shot만 seed 0–2 평균, 1·4·8-shot은 seed 0 단일',
                    '지표: AUPRO@0.05(주), pAUROC@0.05, I-AUROC — 연속 점수 맵에서 계산, 임계값 불필요'],
                   note=f'{SRC} §4.1 표 1–3. AnomalyDINO·FastRecon·RAD·SubspaceAD·RegAD는 모든 shot seed 0–2 평균. '
                        f'비교군 설정 {A["appx_settings"]}.')
    rows = [
        ['train/good', '2,528', '정상, regular 조명 (support 선택 원천)'],
        ['test_public', '1,084 (정상 379 / 결함 705)', '여러 조명, 라벨 공개 → 개발 평가'],
        ['test_private', '2,045 (748 / 1,297)', 'regular 조명, GT는 서버 비공개'],
        ['test_private,mix', '2,045 (748 / 1,297)', 'seen/unseen 조명 무작위 구성 (별도 split)'],
    ]
    table(s, .70, top + .1, [2.6, 3.6, 5.74], ['Split', '이미지 수', '조명 · 라벨 범위'], rows, size=13, bottom=5.7)


def main_cmp():
    s, top = slide('main', S3, 'Few-shot 주 비교 (AD2 test_public)',
                   '결과 │ 1–16 shot 모두 AUPRO@0.05 1위, 16-shot에서 SuperAD 대비 +4.47pp (59.63 vs 55.16)',
                   ['격차 유지: 모든 shot에서 2위 SuperAD보다 3.67–5.06pp 높음 (차트 위 숫자)',
                    f'보조 지표: pAUROC@0.05 전 shot 1위, I-AUROC는 2-shot만 SuperAD가 앞섬 ({A["appx_metrics"]})'],
                   note='8-class macro 평균값. 2·16-shot은 seed 0–2 평균, SCARF·SuperAD·PatchCore의 1·4·8-shot은 seed 0 단일. '
                        f'각 방법은 자기 백본·입력 설정을 유지한 전체 파이프라인 비교. 표 원본 {A["appx_aupro"]}. {SRC} §4.2 표 4')
    figure(s, CH / 'main_curve.png', top + .05)


def mvtec():
    s, top = slide('mvtec', S3, 'MVTec AD · VisA few-shot',
                   '한계 │ P-AUROC는 상위권과 비슷하지만 I-AUROC는 SuperAD보다 낮다 — 강점은 AD2 저-FPR 국소화',
                   ['I-AUROC (1·3번째 패널): 두 데이터셋 모든 조건에서 SuperAD보다 1.2–5.0pp 낮음',
                    'P-AUROC (2·4번째 패널): 최고값과 0.33pp 이내, VisA 8·16-shot은 최고 (98.34, 98.56)'],
                   note='전체 FPR 구간 지표, seed 0, MVTec AD 15·VisA 12 클래스 macro, support hash 일치. FastRecon = DINOv2 기반 프로젝트 변형. '
                        f'16-shot은 VisA만 완료. 표 원본 {A["appx_mvtec"]}. {SRC} §4.2 표 5')
    figure(s, CH / 'mvtec_visa.png', top + .3)


def private():
    s, top = slide('private', S3, 'MVTec AD 2 공식 private 평가',
                   '결과 │ 정상 16장으로 AU-PRO 64.80 / 62.49% — full-data 방법과는 예산이 달라 순위를 매기지 않음',
                   ['SegF1: 34.84%로 SuperAD·RoBiS·SuperADD(47–57%)보다 낮음 → 고정 임계값 설계가 다음 과제',
                    '임계값: support를 뺀 train/good의 8번째 간격 샘플 픽셀 p95 × 1.421, private GT 미사용'],
                   note='● TESTpriv, ○ TESTpriv,mix. AU-PRO는 연속 점수 맵, SegF1은 임계값 적용. Full-data 행은 원 논문·리더보드 수치. '
                        f'ClassF1 포함 전체 표 {A["appx_private"]}. {SRC} §4.3.1 표 6')
    figure(s, CH / 'private_dots.png', top + .3)


def shift():
    s, top = slide('shift', S3, '조명 변화 (AD2 Shift 1–3)',
                   '결과 │ support에 없던 3가지 조명 모두에서 1–16 shot 전 구간 SCARF 곡선이 가장 위에 있다',
                   ['읽는 법: 패널 = 촬영 조건, x축 = 정상 support 수, y축 = AUPRO@0.05',
                    '16-shot seed 0: Regular 61.26% → Shift 1 59.40% (−1.86pp, 표 10)'],
                   note='범례 "Ours (RSG4)" = SCARF 내부 실행명. Regular·Shift 1은 8클래스, Shift 2·3은 fruit_jelly 제외 7클래스 macro. '
                        f'{SRC} §4.3.2 그림 2')
    figure(s, FIG / 'fig_acquisition_shift_by_shot.png', top + .15)


def corrupt():
    s, top = slide('corrupt', S3, '합성 손상 (MVTec AD corruption)',
                   '결과 │ 밝기·대비 변화에서는 최고지만, Gaussian noise에서는 PatchCore보다 5.02pp 낮다',
                   ['Contrast: Clean 대비 하락 SCARF −2.21pp ≈ SuperAD −2.22pp, PatchCore −6.82pp',
                    'Gaussian noise: SCARF −16.35pp로 PatchCore(−9.94)·SuperAD(−13.39)보다 크게 하락',
                    '가설 (확인 필요): 고해상도·Layer 5 관측이 고주파 노이즈에 반응해 손해를 봄'],
                   note='MVTec AD 15클래스, 16-shot, seed 0, severity 3, support는 깨끗한 이미지. † DINOv2 기반 프로젝트 변형. '
                        f'pAUROC 포함 표 원본 {A["appx_corrupt"]}. {SRC} §4.3.3 표 7')
    figure(s, CH / 'corruption_dots.png', top + .25)


# ---------------------------------------------------------------- 04 ablation
def ablation():
    s, top = slide('ablation', S4, '구성 요소 소거 (AD2 16-shot)',
                   '결과 │ 단위 보정과 게이트가 둘 다 있어야 Deep-only(점선)를 넘는다 (+0.47pp)',
                   ['읽는 법: 모든 구성이 같은 Deep 기준선을 쓰고, 위에서 아래로 Shallow 처리 장치를 추가',
                    '게이트 효과가 가장 큼: 같은 q95 단위에서 전체 가산 → 초과분만 가산으로 +5.13pp',
                    '교환 관계: pAUROC@0.05는 Deep-only보다 0.24pp 낮음 (77.11 vs 77.35) → 이득은 영역 단위'],
                   note='MVTec AD 2 test_public, 16-shot, seed 0, full8, CQ 후처리. CQ 없이도 SCARF 58.32 vs Deep-only 57.74. '
                        f'Linear λ*·Hard Gate는 진행 중({P["pending"]}쪽). 표 원본 {A["appx_ablation"]}. {SRC} §4.4.1 표 8')
    figure(s, CH / 'ablation_dots.png', top + .1)


def lowshot():
    s, top = slide('lowshot', S4, '저샷·다른 데이터셋 진단',
                   '한계 │ 전체 가산보다는 늘 낫지만, 1–2 shot에서는 Deep-only보다 낮은 경우가 있다',
                   ['안전장치는 일관: 4개 조건 모두 전체 가산(×)보다 4.31–10.80pp 높음',
                    '순이득은 조건 의존: Deep-only(■) 대비 −3.10 ~ +0.08pp, VisA 1-shot 손해가 가장 큼',
                    '가설: 정상 표본이 적어 τ5 추정이 흔들림 → bank 4배(○)면 4개 조건 모두 +0.46~+5.38pp'],
                   note='seed 0, 4개 해상도, bag 4, CQ. MVTec AD 15·VisA 12 클래스 macro, development/post-hoc 진단. cap1024 비용: '
                        f'fit 86.7 → 116.4 s/class, query 5.39 → 5.98 s/장 (A6000). {SRC} §4.4.1 표 8a, §4.5.2 표 12a')
    figure(s, CH / 'lowshot_dots.png', top + .3)


def gate():
    s, top = slide('gate', S4, '게이트 동작 확인',
                   '확인 │ 게이트는 결함 픽셀에서 정상 대비 9.8배 자주 열리고, 조명이 바뀌어도 정상 쪽은 거의 그대로',
                   ['왜 약 1%인가: 기준을 정상 거리 99% 분위수로 잡음 → 정상이면 100개 중 1개꼴로만 열림',
                    '역할의 범위: 결함 픽셀의 약 89%에서는 게이트가 닫혀 Deep 점수만 사용 → Shallow는 보조 신호'],
                   note='MVTec AD 2 test_public, 16-shot, seed 0, class macro. 활성 조건 z5 > τ5 ⇔ d5 > Q0.99(D5). '
                        f'1-shot LTO 행은 대조군 미완으로 제외. {SRC} §4.4.2 표 9')
    figure(s, CH / 'gate_bars.png', top + .1)


def resolution():
    s, top = slide('resolution', S4, '해상도 구성의 용량 통제 비교',
                   '결과 │ 토큰 수를 맞춰도 4-해상도가 단일 1120 px보다 +2.52pp — 이득은 "섞는 구성"에서 온다',
                   ['용량 통제: 1120 px 단일에 토큰을 4배(1024) 줘도 56.63%로 오히려 −0.33pp',
                    '비유: 고배율 돋보기 하나로 오래 보는 것보다 저·중·고배율을 함께 보는 편이 낫다'],
                   note='MVTec AD 2 test_public, 16-shot, seed 0, full8, DINOv2 ViT-L/14. E12 resolution control (historical fusion-base provenance). '
                        f'{SRC} §4.4.3 표 10')
    figure(s, CH / 'resolution_dots.png', top + .3)


def cost():
    s, top = slide('cost', S4, '계산 비용',
                   '비용 │ 제안 연산은 장당 4.1 ms(0.21%)뿐이고, 시간 대부분은 4개 해상도 특징 추출이다',
                   [f'비교: A6000 FP32에서 SCARF 5.75 s vs SuperAD 0.79 s/장, fit VRAM 10.43 vs 1.42 GiB ({A["appx_cost"]})',
                    '시사점: 속도 개선은 결합 규칙이 아니라 백본 실행 횟수(해상도 수)·정밀도에서 찾아야 함'],
                   note=f'로컬 RTX 4090, 16-shot, 클래스당 12 query (총 96). {SRC} §4.5.1 표 12')
    figure(s, CH / 'cost_bars.png', top + .1)


# ---------------------------------------------------------------- 05 discussion
def discussion():
    s, top = slide('discussion', S5, '논의: 관찰과 시사점',
                   '시사점 │ 저-FPR 국소화 주장은 지지되지만, 저샷·노이즈·고정 임계값은 다음 검증 대상이다',
                   ['클래스 의존: can +2.91, wallplugs +1.63pp에 이득 집중 → Shallow가 필요한 결함 유형 분석',
                    f'지표 교환: AUPRO +0.47 / pAUROC −0.24pp ({P["ablation"]}쪽) → 영역 단위 이득으로 명시',
                    f'저샷 불안정: 1–2 shot에서 Deep-only보다 낮은 조건 존재 ({P["lowshot"]}쪽) → τ5 추정 안정화',
                    f'고정 임계값: private SegF1 34.84% ({P["private"]}쪽) → 조명 변화별 임계값 보정',
                    f'노이즈 취약: Gaussian noise −16.35pp ({P["corrupt"]}쪽) → 고해상도·Layer 5 반응 확인'],
                   note=f'{SRC} §5.1–5.2 및 해당 표. 화살표 뒤 시사점은 초안.')
    boxed(s, .70, top + .35, 11.94, .9, '주장 범위',
          '고정 다층·다해상도 관측에서, 정상 support 통계만으로 Shallow를 안전하게 더해 저-FPR 영역 국소화를 높인다. '
          '이미지 판정·노이즈 강건성·고정 임계값 성능은 주장하지 않는다.')


def pending():
    s, top = slide('pending', S5, '남은 검증',
                   '계획 │ 선형 가중·Hard Gate 대조와 q·β 민감도가 채워져야 "튜닝 불필요" 주장이 완성된다',
                   note='진행 중 비교는 configs/benchmarks/scarf_v16_fusion_20261003/PROTOCOL.md 기준. '
                        f'{SRC} §4.4 표 8·11, §4.5.3, 부록 A.1–A.2')
    rows = [
        ['Linear weight λ*', '최적 선형 가중 z_d + λ·z5보다 나은가?', '진행 중: 2-shot에서 λ ∈ {0, …, 2} 선택 → 16-shot 비교',
         '≥ λ*: 비선형 차단이 필요\n≈ λ*: 라벨 없이 최적 수준 자동 달성'],
        ['Hard Gate', '초과량 차감이 이진 마스킹보다 나은가?', '진행 중', '우수 시: 경계 부근 연속 가산의 안정성'],
        ['q · β 민감도', '기본값 근방에서 성능이 안정한가?', '미측정 (표 11 비어 있음)', '안정 시: 데이터셋별 튜닝 불필요 근거'],
        ['정성 비교 (그림 3)', '실제 맵에서 게이트가 어떻게 작동하나?', '패널 사양만 확정', 'Regular · Shift 1 · Noise × 7열'],
    ]
    table(s, .70, top + .1, [2.4, 3.3, 3.2, 3.04], ['검증', '질문', '현재 상태', '결과에 따른 서술'], rows,
          size=12, bottom=6.2)


def conclusion():
    s, top = slide('conclusion', S5, '결론 및 향후 진행 방향',
                   '결론 │ 척도 보정과 기여 제어를 분리해 Shallow를 안전하게 더하고 AD2 저-FPR 국소화 1위를 얻었다',
                   note=f'{SRC} §6. 근거 쪽: 방법 {P["ablation"]}, 성능 {P["main"]}·{P["private"]}·{P["mvtec"]}, '
                        f'강건성 {P["shift"]}·{P["corrupt"]}, 효율 {P["cost"]}.')
    rows = [
        ['방법', '정상 support 분위수만으로 척도 보정 + 비대칭 잔차 결합', '선형 가중·Hard Gate 대조 미완', '대조 완료 후 표 8 확정'],
        ['성능', 'AD2 1–16 shot AUPRO@0.05 1위\n(16-shot SuperAD +4.47pp), private 64.80 / 62.49',
         'MVTec AD·VisA I-AUROC는 SuperAD보다 낮음', '주장을 저-FPR 국소화로 한정'],
        ['강건성', '조명 Shift 1–3, 밝기·대비 변화에 안정', 'Gaussian noise −16.35pp', '노이즈 원인(해상도·층) 분리 실험'],
        ['효율', '제안 연산은 전체의 0.21%', '전체는 SuperAD보다 약 7배 느림', '해상도 수·정밀도 조정 검토'],
    ]
    table(s, .70, top + .1, [1.4, 4.1, 3.3, 3.14], ['구분', 'Summary', '현재 문제', '향후 진행 방향'], rows,
          size=13, bottom=6.2)


# ---------------------------------------------------------------- appendix (exact numbers)
HEAD8 = ['Shot', 'PatchCore', 'AnomalyDINO', 'FastRecon', 'RAD', 'SubspaceAD', 'SuperAD', 'RegAD', 'SCARF']
W8 = [.74] + [1.4] * 8


def appx_aupro():
    s, top = slide('appx_aupro', SA, '주 비교 원본: AUPRO@0.05',
                   f'원본 │ {P["main"]}쪽 차트의 수치 (AD2 test_public, 8-class macro, %)',
                   note=f'* = seed 0 단일, 그 외 seed 0–2 mean ± sample SD. {SRC} §4.2 표 4')
    rows = [
        ['1', '36.78*', '39.92 ± 2.43', '17.96 ± 1.21', '23.32 ± 0.26', '37.31 ± 1.99', '49.96*', '12.73 ± 0.38', '54.23*'],
        ['2', '40.07 ± 2.43', '42.36 ± 1.24', '18.93 ± 1.94', '26.85 ± 1.11', '38.28 ± 0.70', '50.56 ± 2.37', '15.05 ± 0.30', '55.62 ± 2.70'],
        ['4', '44.41*', '43.40 ± 0.40', '20.00 ± 1.51', '30.28 ± 0.83', '39.22 ± 0.70', '53.68*', '17.40 ± 1.07', '58.36*'],
        ['8', '45.73*', '44.28 ± 0.59', '18.09 ± 0.45', '32.96 ± 1.60', '39.38 ± 0.42', '54.73*', '19.25 ± 0.64', '58.40*'],
        ['16', '49.16 ± 0.77', '44.36 ± 0.27', '17.31 ± 1.20', '33.54 ± 0.98', '39.54 ± 0.34', '55.16 ± 0.46', '20.28 ± 0.80', '59.63 ± 0.80'],
    ]
    table(s, .70, top + .1, W8, HEAD8, rows, size=11, bottom=5.2, strong_cols=(0, 8))


def appx_metrics():
    s, top = slide('appx_metrics', SA, '보조 지표: pAUROC@0.05 · I-AUROC',
                   '결과 │ pAUROC@0.05는 전 shot 1위, I-AUROC는 2-shot을 제외하고 1위',
                   note=f'AD2 test_public 8-class macro (%). * = seed 0 단일, 그 외 seed 0–2 mean ± SD. {SRC} §4.2 표 4')
    pa = [
        ['1', '64.38*', '72.28 ± 0.79', '62.22 ± 0.60', '62.89 ± 0.65', '70.47 ± 0.72', '72.18*', '57.76 ± 0.30', '72.50*'],
        ['2', '67.24 ± 0.93', '73.50 ± 0.42', '63.03 ± 0.66', '65.54 ± 0.18', '71.10 ± 0.34', '74.51 ± 0.33', '58.76 ± 0.49', '75.64 ± 0.53'],
        ['4', '68.21*', '74.59 ± 0.65', '63.15 ± 0.23', '67.14 ± 0.56', '71.72 ± 0.13', '75.69*', '60.36 ± 0.48', '76.54*'],
        ['8', '69.54*', '74.85 ± 0.34', '60.76 ± 0.77', '69.25 ± 0.79', '71.70 ± 0.18', '76.26*', '62.02 ± 0.32', '77.04*'],
        ['16', '70.87 ± 0.33', '75.10 ± 0.17', '60.64 ± 0.97', '70.03 ± 0.78', '71.80 ± 0.23', '76.54 ± 0.60', '63.46 ± 0.76', '77.23 ± 0.29'],
    ]
    ia = [
        ['1', '66.32*', '69.98 ± 2.10', '60.33 ± 0.63', '58.12 ± 0.77', '68.34 ± 2.64', '72.16*', '57.73 ± 1.82', '73.26*'],
        ['2', '68.58 ± 2.51', '72.59 ± 2.88', '61.77 ± 4.06', '56.01 ± 4.49', '70.79 ± 1.85', '74.11 ± 2.26', '60.56 ± 3.94', '73.79 ± 1.80'],
        ['4', '72.98*', '74.52 ± 0.22', '63.05 ± 2.52', '58.69 ± 2.39', '71.77 ± 1.32', '75.36*', '63.79 ± 2.34', '77.34*'],
        ['8', '70.32*', '74.69 ± 0.77', '59.09 ± 3.32', '63.89 ± 1.16', '72.34 ± 0.16', '74.60*', '64.34 ± 1.05', '76.86*'],
        ['16', '71.33 ± 0.93', '74.59 ± 1.47', '59.79 ± 1.70', '65.60 ± 1.64', '71.39 ± 0.07', '74.25 ± 0.36', '65.64 ± 1.34', '76.58 ± 0.24'],
    ]
    text_box(s, .70, top - .05, 6, .3, 'pAUROC@0.05', 13, NAVY, BOLD, True, name='TableLabel1')
    table(s, .70, top + .27, W8, HEAD8, pa, size=10, row_h=.3, strong_cols=(0, 8))
    y2 = top + .27 + .42 + .3 * 5 + .15
    text_box(s, .70, y2, 6, .3, 'I-AUROC', 13, NAVY, BOLD, True, name='TableLabel2')
    table(s, .70, y2 + .32, W8, HEAD8, ia, size=10, row_h=.3, strong_cols=(0, 8), name='SummaryTable2')


def appx_settings():
    s, top = slide('appx_settings', SA, 'AD2 비교군 설정',
                   '설정 │ 같은 support·query·평가기를 쓰되, 각 방법은 자기 백본·입력·layer 설정을 유지한다',
                   note='FastRecon·RAD는 논문 식을 따른 독립 구현. SuperAD는 공통 support 사용(원 논문은 전체 train으로 coreset 후 16장 선택). '
                        f'{SRC} §4.1 표 3')
    rows = [
        ['SCARF (ours)', 'Frozen; support bank · scale fitting', 'DINOv2 ViT-L/14; [5, 11, 23]', '448/672/896/1120; 4-scale bank'],
        ['SuperAD', 'Frozen; support reference bank', 'DINOv2 ViT-L/14; [5, 11, 17, 23]', '672'],
        ['PatchCore', 'Frozen; support memory / coreset', 'DINOv2 ViT-L/14; [5, 11, 17, 23]', '672'],
        ['AnomalyDINO', 'Frozen; support references', 'DINOv2 ViT-L/14; [23]', '672; sheet_metal 448'],
        ['FastRecon (paper)', 'No gradient; closed-form fit', 'ResNet-50', '368; 15% coreset; λ = 0.3'],
        ['RAD', 'Frozen; support retrieval bank', 'DINOv3 ViT-B/16; [3, 6, 9, 11]', '512 → crop 448; global bank'],
        ['SubspaceAD', 'Frozen; support PCA fit', 'DINOv2-reg ViT-G/14; [22–28]', '672'],
        ['RegAD', 'Official source-trained model', 'ResNet-18 STN', '224; 2-shot bottle checkpoint'],
    ]
    table(s, .70, top + .1, [2.3, 3.3, 3.3, 3.04], ['방법', '학습 · support fit', 'Backbone / layers', '입력 · 해상도'],
          rows, size=11, bottom=6.6)


def appx_mvtec():
    s, top = slide('appx_mvtec', SA, 'MVTec AD · VisA 원본',
                   f'원본 │ {P["mvtec"]}쪽 차트의 수치 (각 셀 I-AUROC / P-AUROC, 전체 FPR, %)',
                   note=f'seed 0, MVTec AD 15·VisA 12 클래스 macro, support hash 일치. FastRecon = DINOv2 기반 프로젝트 변형. {SRC} §4.2 표 5')
    rows = [
        ['MVTec AD · 1', '88.12 / 96.22', '88.84 / 96.60', '92.54 / 96.93', '87.57 / 96.63'],
        ['MVTec AD · 2', '92.67 / 96.97', '93.51 / 97.29', '95.43 / 97.36', '91.93 / 97.31'],
        ['MVTec AD · 4', '95.06 / 97.37', '95.61 / 97.87', '96.65 / 97.70', '93.83 / 97.66'],
        ['MVTec AD · 8', '96.13 / 97.80', '96.76 / 98.16', '97.39 / 97.95', '94.69 / 97.89'],
        ['VisA · 1', '86.09 / 95.56', '84.65 / 96.75', '90.29 / 97.06', '85.29 / 96.73'],
        ['VisA · 2', '87.87 / 96.29', '87.67 / 97.43', '92.20 / 97.68', '89.46 / 97.46'],
        ['VisA · 4', '89.52 / 96.92', '89.21 / 97.91', '92.94 / 98.02', '91.77 / 97.94'],
        ['VisA · 8', '92.36 / 97.18', '91.53 / 98.11', '94.50 / 98.29', '92.81 / 98.34'],
        ['VisA · 16', '93.64 / 97.30', '93.38 / 98.32', '95.19 / 98.43', '93.10 / 98.56'],
    ]
    table(s, .70, top + .05, [2.54, 2.35, 2.35, 2.35, 2.35], ['Dataset · Shot', 'PatchCore', 'FastRecon', 'SuperAD', 'SCARF'],
          rows, size=12, bottom=6.72, strong_cols=(0, 4))


def appx_private():
    s, top = slide('appx_private', SA, '공식 private 평가 전체 표',
                   '참고 │ Full-data 행은 정상 데이터 예산·백본·입력·후처리가 달라 few-shot과 직접 순위화하지 않는다',
                   note=f'각 셀 TESTpriv / TESTpriv,mix (%). Few-shot 비교군의 private 결과는 아직 없음. {SRC} §4.3.1 표 6')
    rows = [
        ['SCARF (16-shot)', '64.80 / 62.49', '76.38 / 77.52', '34.84 / 30.68'],
        ['PatchCore', '28.8 / 26.0', '8.0 / 9.9', '3.7 / 3.4'],
        ['RD', '26.4 / 25.0', '70.8 / 72.9', '18.1 / 16.7'],
        ['RD++', '27.1 / 25.2', '70.5 / 73.0', '19.2 / 16.7'],
        ['EfficientAD-S', '30.8 / 19.2', '76.8 / 77.8', '15.4 / 8.0'],
        ['MSFlow', '24.3 / 11.9', '51.1 / 72.0', '21.8 / 9.0'],
        ['SimpleNet', '21.1 / 13.7', '73.3 / 77.1', '17.7 / 8.7'],
        ['DSR', '20.3 / 17.4', '78.0 / 78.0', '10.9 / 9.5'],
        ['SuperAD', '60.51 / 58.37', '70.20 / 74.40', '47.18 / 42.51'],
        ['RoBiS', '67.25 / 59.65', '79.62 / 79.86', '51.00 / 46.52'],
        ['SuperADD', '61.36 / 59.07', '68.03 / 73.25', '57.42 / 54.35'],
    ]
    table(s, .70, top + .05, [3.24, 2.9, 2.9, 2.9], ['방법 (full-data는 2행부터)', 'AU-PRO@0.05', 'ClassF1', 'SegF1'],
          rows, size=11, bottom=6.72)


def appx_corrupt():
    s, top = slide('appx_corrupt', SA, '합성 손상 원본',
                   f'원본 │ {P["corrupt"]}쪽 차트의 수치 (각 셀 AUPRO@0.05 / pAUROC@0.05, %)',
                   note=f'MVTec AD 15클래스, 16-shot, seed 0, severity 3. † DINOv2 기반 프로젝트 변형. {SRC} §4.3.3 표 7')
    rows = [
        ['PatchCore', '83.97 / 88.80', '83.43 / 88.38', '77.15 / 85.27', '81.58 / 87.41', '74.03 / 83.66'],
        ['FastRecon †', '85.74 / 90.82', '77.53 / 86.35', '69.72 / 84.47', '82.35 / 88.90', '57.70 / 77.37'],
        ['SuperAD', '84.49 / 89.15', '83.68 / 88.47', '82.27 / 87.86', '82.17 / 87.74', '71.10 / 82.45'],
        ['SCARF', '85.36 / 88.92', '84.42 / 88.26', '83.15 / 87.44', '81.92 / 86.71', '69.01 / 80.58'],
    ]
    table(s, .70, top + .1, [1.94] + [2.0] * 5, ['방법', 'Clean', 'Brightness', 'Contrast', 'Defocus blur', 'Gaussian noise'],
          rows, size=12, bottom=4.6)


def appx_ablation():
    s, top = slide('appx_ablation', SA, '소거·진단 원본',
                   f'원본 │ {P["ablation"]}·{P["lowshot"]}쪽 차트의 수치 (AUPRO@0.05, %)',
                   note=f'위: AD2 16-shot seed 0 full8, CQ. 아래: seed 0, MVTec AD 15·VisA 12 클래스 macro. {SRC} 표 8, 8a')
    rows = [
        ['Deep-only', '—', '—', '—', '58.67', '77.35', '0.00'],
        ['단순 전체 가산', '전체', 'MAD', '—', '51.16', '73.02', '−7.51'],
        ['+ q95 척도 보정', '전체', 'q95', '—', '54.02', '74.04', '−4.65'],
        ['+ 잔차 게이트', '초과분', 'MAD', 'q99', '57.48', '76.16', '−1.19'],
        ['SCARF (둘 다)', '초과분', 'q95', 'q99', '59.15', '77.11', '+0.47'],
    ]
    table(s, .70, top + .05, [2.84, 1.4, 1.5, 1.4, 1.6, 1.6, 1.6],
          ['구성', 'Shallow L5', '척도 보정', '게이트', 'AUPRO@0.05', 'pAUROC@0.05', 'Δ vs Deep'], rows, size=11, row_h=.3)
    rows2 = [
        ['MVTec AD', '1', '78.39', '79.50', '79.36', '72.68'],
        ['MVTec AD', '2', '81.75', '82.21', '81.94', '75.75'],
        ['MVTec AD', '4', '83.63', '84.10', '83.55', '79.32'],
        ['VisA', '1', '72.59', '77.97', '75.69', '61.79'],
    ]
    table(s, .70, top + .05 + .42 + .3 * 5 + .3, [1.9, 1.0, 2.3, 2.3, 2.2, 2.24],
          ['Dataset', 'Shot', 'SCARF (cap256)', 'SCARF (cap1024)', 'Deep-only', '전체 가산'], rows2, size=11, row_h=.3,
          name='SummaryTable2')


def appx_cost():
    s, top = slide('appx_cost', SA, '비교군 계산 비용 (A6000)',
                   '참고 │ SCARF는 48개 컴포넌트 뱅크와 4개 해상도로 SuperAD보다 fit·query 모두 무겁다',
                   note=f'MVTec AD 2 full8 test_public, seed 0, 16-shot, frozen DINOv2 ViT-L/14, FP32, batch 1. {SRC} §4.5.1 표 12')
    rows = [
        ['SCARF', '196,608 (48 banks)', '87.25 ± 83.66', '5,746 ± 6,107', '5,521 / 18,600', '10.43 / 3.44'],
        ['Deep-only', '131,072 (32 banks)', '88.06 ± 100.45', '5,697 ± 7,263', '5,377 / 23,037', '7.68 / 3.03'],
        ['1120-only', '49,152 (12 banks)', '50.97 ± 53.57', '3,822 ± 4,768', '3,586 / 15,328', '6.86 / 2.65'],
        ['SuperAD', 'All tokens', '10.48 ± 2.17', '794 ± 154', '803 / 1,069', '1.42 / 1.42'],
    ]
    table(s, .70, top + .1, [1.7, 2.3, 2.0, 2.0, 2.0, 1.94],
          ['방법', 'Bank 벡터/클래스', 'Fit (s)', 'Query 평균 (ms)', 'Query 평균 / p95 (ms)', 'Peak VRAM fit / query (GiB)'],
          rows, size=11, bottom=5.0)


def appx_spec():
    s, top = slide('appx_spec', SA, 'SCARF 고정 사양',
                   '사양 │ 모든 값은 사전에 고정하며, 클래스·데이터셋별 튜닝을 하지 않는다',
                   note=f'{SRC} §3.5–3.6, 표 8 (CQ 후처리 적용 조건).')
    rows = [
        ['백본', 'DINOv2 ViT-L/14 (동결, 약 304M)', '패치 14 × 14'],
        ['층', 'Layer 5 (Shallow), Layer 11·23 (Deep)', 'Deep = 두 층 z 평균'],
        ['해상도', '448 / 672 / 896 / 1120 (짧은 변)', '모든 query에 고정'],
        ['Bag', '4개 × 이미지당 256토큰', 'bag 점수는 같은 토큰 격자에서 평균'],
        ['분위수 q / β', '0.95 / 0.99', '척도 보정 / 잔차 게이트'],
        ['맵 통합', '해상도별 맵을 원본 크기로 bilinear 보간 → 동일 가중 평균', '이미지 점수 = 최종 맵 최댓값'],
        ['후처리 (선택)', 'CQ: 상위 1% 연결 성분을 그 최댓값으로 채움', '표 8 결과에 적용'],
        ['재현성', 'seed 0·1·2, 고정 support manifest', '역전파·파라미터 갱신·테스트 라벨 없음'],
    ]
    table(s, .70, top + .1, [2.3, 5.6, 4.04], ['항목', '값', '비고'], rows, size=12, bottom=6.5)


for key in ORDER:
    globals()['main_cmp' if key == 'main' else key]()

groups = [('표지 · 요약', ['cover', 'toc', 'summary']),
          ('01 문제 정의', ['bg', 'ambig', 'dilemma', 'axes']),
          ('02 방법', ['flow', 'bank', 'scale', 'fusion', 'fusion_map']),
          ('03 실험 결과', ['setup', 'main', 'mvtec', 'private', 'shift', 'corrupt']),
          ('04 Ablation', ['ablation', 'lowshot', 'gate', 'resolution', 'cost']),
          ('05 논의', ['discussion', 'pending', 'conclusion']),
          ('Appendix A', ORDER[MAIN_LAST:])]
assert [k for _, ks in groups for k in ks] == ORDER
set_sections(prs, [(n, [slides[k] for k in ks]) for n, ks in groups])
prs.save(OUT)
print('saved', OUT, len(prs.slides), 'slides')
