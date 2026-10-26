"""Build a DSBA progress deck from the current paper figure and unmodified tables.

Run: PYTHONPATH=/tmp/scopead-slide-deps python3 PaperWorks/slides/build_20261019/build_deck.py
Requires python-pptx, PyMuPDF, Pillow and LuaLaTeX. Never changes the paper or old deck.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

import pymupdf as pdf
from PIL import Image
from pptx.util import Inches

SK = Path('/home/hun/.claude/skills/dsba-slide-deck')
sys.path.insert(0, str(SK / 'scripts'))
from deck_style import new_deck, frame_slide, figure, figure_side, table, text_box, NAVY
from deck_ops import set_sections, fix_table_ids

HERE = Path(__file__).resolve().parent
SLIDES = HERE.parent
PAPER = SLIDES.parent / 'cvpr-2026-author-kit-template'
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
OLD = SLIDES / 'ScopeAD_Rényi_Update_발표_20261018.pptx'
OUT = HERE / 'ScopeAD_Method_Experiments_20261019.pptx'
BK = SLIDES / 'backup' / 'ScopeAD_Rényi_Update_발표_20261018_재구성전백업_20261019.pptx'
if not BK.exists():
    shutil.copy2(OLD, BK)

doc = pdf.open(PAPER / 'main.pdf')
manifest = {'paper_sha256': hashlib.sha256((PAPER/'main.pdf').read_bytes()).hexdigest(),
            'previous_deck': str(OLD), 'backup': str(BK), 'assets': {}, 'slides': []}

def crop(name, page, rect):
    path = ASSETS / (name + '.png')
    doc[page-1].get_pixmap(matrix=pdf.Matrix(4,4), clip=pdf.Rect(rect)).save(path)
    manifest['assets'][name] = {'source': 'main.pdf', 'page': page, 'clip': rect}
    return path

figdoc = pdf.open(PAPER / 'figures/fig_scopead_overview.pdf')
fw, fh = figdoc[0].rect.width, figdoc[0].rect.height
def panel(name, rect):
    path = ASSETS / (name+'.png')
    box = pdf.Rect(rect[0]*fw, rect[1]*fh, rect[2]*fw, rect[3]*fh)
    figdoc[0].get_pixmap(matrix=pdf.Matrix(4,4), clip=box).save(path)
    manifest['assets'][name] = {'source': 'figures/fig_scopead_overview.pdf', 'fractional_clip': rect}
    return path

overview = panel('main_figure', (0,0,1,1))
fitting = panel('panel_a', (0.01,0.015,0.995,0.295))
matching = panel('panel_bc', (0.01,0.325,0.615,0.975))
rank = panel('panel_d', (0.62,0.325,0.995,0.99))
calib = PAPER / 'figures/fig_scale_calibration_lso.png'

# Direct crops retain every original number and emphasis. Overfull column tables
# are typeset independently from the exact source to avoid neighbouring-column ink.
tables = {}
for num,page,rect in [
    (1,6,(58,100,554,267)), (2,6,(58,305,554,414)),
    (5,7,(58,99,295,183)),
    (6,7,(58,235,295,285)), (8,7,(316,99,554,157)),
    (9,7,(345,196,519,239)), (10,10,(76,110,278,175)),
    (11,10,(82,224,272,290))]:
    tables[num] = crop('table_'+str(num),page,rect)

source = (PAPER/'sec/4_experiments.tex').read_text()
blocks = re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', source, re.S)
# Reuse the paper fonts and table body verbatim; no data is parsed/reformatted.
prefix = r'''\documentclass[10pt,letterpaper]{article}
\usepackage{kotex}
\usepackage[margin=18mm]{geometry}
\usepackage{booktabs,amssymb,amsmath,graphicx}
\setmainfont{TeX Gyre Termes}
\setmainhangulfont{Noto Serif CJK KR}
\pagestyle{empty}
\begin{document}
'''
for num in (3,4,7):
    block = blocks[num-1]
    body = re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',block,re.S).group()
    spacing = re.search(r'\\setlength\{\\tabcolsep\}\{[^}]+\}',block).group()
    tex = ASSETS / ('table_'+str(num)+'_isolated.tex')
    tex.write_text(prefix + r'\noindent\scriptsize' + spacing + '\n' + body + '\n'+r'\end{document}')
    subprocess.run(['lualatex','-interaction=nonstopmode','-halt-on-error',
                    '-output-directory='+str(ASSETS),str(tex)],stdout=subprocess.DEVNULL,check=True)
    isolated = pdf.open(tex.with_suffix('.pdf'))
    bounds = pdf.Rect()
    for b in isolated[0].get_text('blocks'):
        bounds |= pdf.Rect(b[:4])
    for drawing in isolated[0].get_drawings():
        bounds |= drawing['rect']
    bounds += (-3,-3,3,3)
    path = ASSETS / ('table_'+str(num)+'.png')
    isolated[0].get_pixmap(matrix=pdf.Matrix(5,5),clip=bounds).save(path)
    tables[num] = path
    manifest['assets']['table_'+str(num)] = {'source':'sec/4_experiments.tex',
        'table_number':num,'mode':'unmodified tabular, isolated typesetting',
        'tabular_sha256':hashlib.sha256(body.encode()).hexdigest()}

prs = new_deck(OLD)
sections = {}
def slide(title,lead,section='01. 방법론 변경과 동기',details=(),note='',notes=''):
    n=len(prs.slides)+1
    s,top=frame_slide(prs,section=section,title=title,lead=lead,detail=details,
        note=note,notes=notes,page=n)
    sections.setdefault(section,[]).append(s)
    manifest['slides'].append({'page':n,'title':title,'lead':lead,'source':note})
    return s,top

def picture(title,lead,path,section='01. 방법론 변경과 동기',details=(),note='',notes=''):
    s,top=slide(title,lead,section,details,note,notes)
    figure(s,path,top)
    return s

def beside(title,lead,path,lines,note='',notes=''):
    s,top=slide(title,lead,note=note,notes=notes)
    figure_side(s,path,top,lines,fig_w=7.65,size=16)
    return s

def result(num,title,lead,details=(),notes=''):
    return picture(title,lead,tables[num],section='02. 실험 결과',details=details,
        note=f'출처: main.pdf Table {num} | 표의 값·행·열·강조 유지; 세부 비교 조건은 발표자 노트 참조',
        notes=notes)

s,t=slide('ScopeAD: 방법론 변경과 실험 근거',
    '구성 │ 메인 그림의 변경 동기에서 출발해 논문 표로 효과와 한계를 확인한다',section='ScopeAD 연구 진행 보고',
    note='2026-10-19 | 기준: 현재 main.pdf / sec/3_method.tex / sec/4_experiments.tex')
table(s,.70,2.10,[2.4,5.15,4.35],['섹션','발표 흐름','핵심 질문'],[
    ['01. 방법론 변경과 동기','전체 구조 → 척도 보정 → 계층 결합 → 순위 집계','무엇이 바뀌었고, 왜 필요한가?'],
    ['02. 실험 결과','주 비교 → ablation → 메커니즘 → 비용·강건성','어떤 조건에서 무엇이 확인되었나?']],row_h=1.05)

picture('전체 구조: 정상 참조 적합과 쿼리 예측',
    '구조 │ 정상 서포트로 뱅크와 척도를 고정하고, 쿼리마다 해상도 점수의 순위를 정한다',overview,
    note='출처: main.pdf Figure 2 원본 | 출력은 CQ99 적용 전 맵',
    notes='메인 그림 원본을 그대로 사용. 예시 맵은 캐시에서 재구성한 설명용 맵이며 새로운 평가 결과가 아니다. 그림의 분위수 표기는 원본을 유지했다.')

s,t=slide('이전 ScopeAD에서 바뀐 결합 규칙',
    '변경 │ 독립 매칭과 계층 선형 결합을 유지하고, 해상도 평균을 순위 가중합으로 바꾼다',
    note='출처: 기존 v18 발표자료 및 현재 sec/3_method.tex; 변경점의 설명은 이번 발표용 종합')
table(s,.70,1.90,[2.2,3.4,3.5,2.8],['요소','이전 구성','현재 구성','수정 동기'],[
    ['관측 구성','3계층 × 4해상도 × 4배그','동일한 48개 뱅크','다양한 비교 단서 확보'],
    ['척도 보정','정상 상위 분위수로 나눔','정상 기준에 맞춘 점수 비교로 설명','원시 거리의 크기 편향 완화'],
    ['계층 결합','deep 평균 + 0.125 × shallow','같은 비대칭 선형 결합 유지','표층 단서의 비중 제한'],
    ['해상도 집계','해상도별 점수의 산술 평균','픽셀별 정렬 + Rényi 순위 가중합','일부 해상도의 강한 반응 보존'],
    ['사후처리','최종 맵 및 이미지 점수 산출','방법 끝에 CQ99를 짧게 명시','핵심 집계 효과와 구분']],row_h=.78)

picture('정상 참조: 쿼리를 보기 전에 비교 기준을 고정',
    '동기 │ 같은 이미지의 유사 패치가 정상 거리를 과소추정하는 문제를 줄인다',fitting,
    details=['독립 뱅크: 계층·해상도·배그별로 최근접 정상 패치를 따로 찾는다.',
             '척도 추정: 다중 샷은 이미지 하나를 제외하는 LSO, 단일 샷은 자기 토큰을 제외하는 LTO를 쓴다.'],
    note='출처: Figure 2(a), sec/3_method.tex (관측 확장, LSO/LTO); 단일 샷 LTO는 이미지 간 변동의 대리 추정')

beside('척도 보정: 정상 상단 기준을 맞춘 뒤 비교',
    '동기 │ 서로 다른 거리 단위가 해상도 점수의 순위를 좌우하지 않도록 한다',calib,
    ['정의: cₘ = Q₁₋ᵦ(Dₘ), zₘ = dₘ / cₘ',
     '기준: zₘ = 1은 각 관측의 정상 참조 상단에 해당한다.',
     '이유: 저오경보 구간에서 비교하므로 평균보다 상단 기준을 맞춘다.',
     '한계: 서포트 기준의 정렬이며, 쿼리 FPR을 보장하지 않는다.'],
    note='출처: main.pdf Figure 3, sec/3_method.tex Eq. (4–5); ε 하한 생략 표기',
    notes='상위 분위수 선택은 저-FPR 평가 구간과 연결한 설계 동기다. 보정 표본의 경험적 분위수 정렬은 정의상 성립하므로 독립 쿼리의 정렬이나 FPR 제어 증거로 해석하지 않는다. 예시 분포 그림이 새 held-out 검증을 대신하지 않는다.')

beside('계층 결합: 깊은 표현을 기준으로 표층 단서를 더함',
    '설계 │ 계층 결합은 기존과 같은 비대칭 선형 결합이다',matching,
    ['기준: z_d = (z₁₁ + z₂₃) / 2',
     '결합: S = z_d + 0.125 z₅',
     '동기: 정상 질감에도 반응할 수 있는 얕은 계층의 비중을 제한한다.',
     '해석: 기준에 잔여 단서를 더하는 형태이며, 새 게이트나 비선형 residual 모듈은 없다.'],
    note='출처: Figure 2(b–c), sec/3_method.tex Eq. (6–8); λ는 개발 비교에서 선택 후 고정')

beside('해상도 집계: 결함에 맞는 해상도의 반응을 보존',
    '동기 │ 한두 해상도에서만 강한 반응이 평균 과정에서 약해질 수 있다',rank,
    ['전처리: 배그 평균과 보간으로 맵의 위치를 맞춘다.',
     '정렬: 같은 픽셀의 네 해상도 점수를 오름차순으로 정렬한다.',
     '가중: 큰 점수의 순위에 더 큰 비중을 준다.',
     '주의: 큰 점수가 단독 잡음일 수도 있어, 척도 보정과 대조 실험이 필요하다.'],
    note='출처: Figure 2(d), sec/3_method.tex Eq. (9–10); 순위는 픽셀마다 달라짐',
    notes='상위 순위 가중은 다수 해상도의 합의를 보장하지 않는다. 한 해상도만 높아도 최상위 가중치가 적용된다. 따라서 단독 잡음 억제 또는 consensus 증명으로 설명하지 않는다.')

s,t=slide('Rényi 가중치: 순서 통계량에서 정한 고정 규칙',
    '근거 │ 지수 표본의 기대 순서 통계량을 정규화해 상위 순위의 비중을 정한다',
    details=['aₖ = Σⱼ₌₁ᵏ 1 / (R − j + 1),  wₖ = aₖ / Σₘ aₘ',
             'R = 4이면 Σaₖ = 4. 가중치는 해상도 이름이 아니라 점수 순위에 붙는다.'],
    note='출처: sec/3_method.tex Eq. (11–12); 분포 적합이나 최적성 정리를 주장하지 않음',
    notes='Rényi 표현의 수식은 iid 표준 지수 표본에 대한 정확한 결과다. 실제 해상도 점수가 iid 지수 분포를 따른다거나 EVT로 그 분포에 수렴한다는 주장은 하지 않는다. 이 방법에서는 고정된 증가 가중치의 설계 규칙으로 사용한다.')
table(s,.70,2.75,[2.7,2.9,3.1,3.2],['오름차순 순위','기대값 aₖ','정규화 가중치 wₖ','집계에서의 역할'],[
    ['1: 가장 작은 점수','1/4','1/16 = 0.0625','낮은 점수의 영향 축소'],
    ['2','7/12','7/48 ≈ 0.1458','중간 이하 반응 반영'],
    ['3','13/12','13/48 ≈ 0.2708','중간 이상 반응 반영'],
    ['4: 가장 큰 점수','25/12','25/48 ≈ 0.5208','강한 반응의 희석 완화']],row_h=.64)
text_box(s,.75,6.15,11.8,.45,'실제 점수의 지수 분포를 가정하지 않는다. 가중치의 최적성은 별도 입증이 필요하다.',16,NAVY)

s,t=slide('비음수 집계의 성질과 사후처리의 위치',
    '범위 │ 순위 집계의 안정성은 수식으로, 탐지 효과는 실험으로 확인한다',
    details=['성질: 비음수이고 합이 1인 가중치 → 출력은 입력 점수의 최솟값과 최댓값 사이에 있다.',
             '안정성: 입력 섭동의 최댓값보다 출력의 변화가 커지지 않는다.',
             '한계: 이 성질만으로 이상 신호와 정상 고점수를 구분할 수는 없다.'],
    note='출처: sec/3_method.tex Eq. (10) 및 사후처리 문단; CQ99의 기여는 CQ-off 실험과 분리')
text_box(s,.80,4.40,11.6,1.5,[
    '사후처리: 최종 단계에서 CQ99로 연결 영역 내부 점수를 보정한다.',
    '이미지 점수: 최종 맵의 상위 1% 픽셀 점수 평균을 사용한다.',
    '실험에서의 구분: 주 비교는 최종 파이프라인, 순수 집계 비교는 CQ99 제외 조건이다.'],18)

result(1,'주 비교: AD2의 모든 shot에서 세 지표 최고 평균',
    '결과 │ 3개 시드 평균 AUPRO는 1-shot 52.18%, 16-shot 60.74%다',
    notes='Table 1 전체 원본. AD2 test_public full8, seed 0/1/2 평균±표본 SD. 비교군의 입력 해상도는 설정 절 참조. 평균 순위를 보고하며 통계적 유의성 검정을 수행했다는 주장은 하지 않는다. CQ99 포함 최종 방법.')

result(2,'교차 데이터셋: VisA의 이미지 판별에서도 이득',
    '결과 │ VisA 4·8·16-shot I-AUROC가 최고이며, MVTec AD에서는 격차가 작다',
    details=['MVTec AD의 작은 격차는 배경·조명과 높은 기준선 성능에 연결해 해석하되, 인과로 단정하지 않는다.'],
    notes='Table 2 전체 원본. Seed 0 단일 실행. I-AUROC/P-AUROC이며 AD2의 AUPRO와 직접 비교하지 않는다. 배경의 청결도가 작은 향상 폭의 원인이라는 설명은 가설이며 배경 통제 실험으로 입증된 결론이 아니다.')

result(3,'종합 ablation: 각 구성 요소를 차례로 추가',
    '결과 │ 2-shot은 50.51 → 59.92%, 16-shot은 53.78 → 60.16%로 오른다',
    details=['E → F: Rényi 집계 추가 시 두 shot 모두 +0.59pp. 누적 비교는 추가 순서에 조건부인 효과다.'],
    notes='Table 3의 tabular 원문을 별도 렌더링. 원본 PDF에서는 표가 오른쪽 단에 침범한다. 값·열·행·강조 변경 없음. AD2 test_public, seed0, CQ99 적용 조건. CQ-off Table 6과 혼합하지 않는다.')

result(4,'용량 일치: 토큰 수 증가만으로 설명되지 않는 이득',
    '대조 │ 총 cap 1024를 맞춰도 다해상도 60.16%가 단일 해상도 57.14%보다 높다',
    details=['같은 총 토큰 cap에서 +3.02pp. 해상도 관측의 다양성이 제공하는 이득을 확인한다.'],
    notes='Table 4 tabular 원문 별도 렌더링. 원본 PDF에서 Table3의 우측 침범과 겹침. AD2 16-shot seed0. 이 표의 값과 비용 표(Table8)의 단일 해상도 값에는 0.01pp 수준 차이가 있어 동일 실행으로 단정하지 않는다.')

result(5,'독립 매칭: 계층마다 정상 참조를 따로 탐색',
    '결과 │ AD2에서 Layer Concat 대비 2-shot +7.21pp, 16-shot +2.09pp다',
    details=['이 표는 Rényi 도입 전의 독립 매칭 비교다. 최종 모델의 절대 성능과 구분해 읽는다.'],
    notes='Table 5 원본. Seed0. AD2 셀은 AUPRO/pAUROC, MVTec AD 및 VisA는 I-AUROC/P-AUROC. ScopeAD 행은 59.33/59.57의 이전 집계 설정이며 최종 Rényi 모델 행으로 바꾸지 않는다.')

picture('척도 진단: 관측마다 다른 정상 거리 분포',
    '확인 │ 정상 참조 상단을 맞추는 것과 쿼리에서 척도가 맞는 것은 구분해야 한다',calib,
    section='02. 실험 결과',
    details=['보정 표본에서 상단 기준이 일치하는 것은 정의상 성립한다.',
             '일반화 근거는 별도 쿼리의 분포·상위 해상도 점유율·결함 크기별 분석으로 확인해야 한다.'],
    note='출처: main.pdf Figure 3; 현재 원고에 수록된 분포 그림을 그대로 사용',
    notes='진단의 요구와 현재 보이는 근거를 구분한다. 원고의 14/16 조건 개선 및 1.6배 차이 수치는 독립 표/원시 근거가 이 슬라이드에 없으므로 단정하는 추가 수치로 싣지 않는다.')

result(6,'사후처리 제외: 집계 자체의 성능 차이',
    '결과 │ None 58.35% 대비 보정+Rényi 59.27%로 +0.92pp다',
    details=['보정+평균은 58.50%(+0.15pp), 같은 보정에서 Rényi로 바꾸면 추가 +0.77pp다.',
             'pAUROC는 76.71 → 76.64로 소폭 낮아진다. 이득은 지표별로 구분한다.'],
    notes='Table 6 원본. AD2 2-shot seed0 full8, CQ99 제외. +0.92pp를 scale calibration 단독의 효과로 명명하지 않는다. scale calibration과 aggregation을 동시에 바꾼 비교다.')

result(7,'상호작용: 보정이 순위 집계의 효과를 바꿈',
    '근거 │ 16-shot Rényi 효과는 보정 전 −0.01pp, 보정 후 +0.68pp다',
    details=['2-shot의 보정 후 Rényi 효과도 평균 대비 +0.75pp다.',
             '세 시드 평균±SD를 제시한다. 평균의 개선만으로 통계적 유의성을 주장하지 않는다.'],
    notes='Table 7 tabular 원문 별도 렌더링. 원본 PDF에서는 우측 본문과 겹친다. 값·열·행·강조 변경 없음. Full8 seed0/1/2. Table6은 seed0 CQ-off이고 Table7은 최종 평가 조건이므로 절대값을 교차 비교하지 않는다.')

result(11,'정규화 대안: 보정의 단독 효과를 확인',
    '한계 │ 상위 분위수는 MAD보다 높지만, none보다 항상 높은 것은 아니다',
    details=['이 표는 평균 집계에서의 보정 대안 비교다. 보정의 필요성은 순위 집계 상호작용과 함께 판단한다.'],
    notes='Appendix Table11 원본. 단일 시드 기존 평균 집계 설정. 원본의 q95 표기 및 수치를 그대로 유지했다. 개발 조건에서 선택한 설정이라는 사실을 감추지 않는다.')

result(10,'계층 가중치: 얕은 계층의 기여를 제한',
    '결과 │ 제시한 양의 가중치 중 0.125가 두 shot에서 가장 높다',
    details=['λ를 키울수록 성능이 낮아진다. 이 표에 없는 deep-only나 작은 λ 구간의 성능은 주장하지 않는다.'],
    notes='Appendix Table10 원본. λ=0.125는 2-shot 개발 결과에서 선택 후 고정. 이 표만으로 0.0625–0.25 범위의 0.3pp 이내 민감도를 주장할 수 없다.')

result(8,'비용: cap을 늘린 만큼 성능이 오르지는 않음',
    '결과 │ cap 256 → 1024에서 +0.17pp, 피크 메모리는 3.12 → 6.71 GiB다',
    details=['최종 cap256 설정: Query 1365ms, Peak 3.12GiB. 성능과 비용의 절충으로 선택한다.'],
    notes='Table8 원본. AD2 16-shot seed0 FP16. 측정된 후보 범위의 비교이며 전역 최적 또는 유일한 Pareto optimum을 주장하지 않는다.')

result(9,'환경 변화: 획득 조건별 성능 범위',
    '관측 │ 같은 7개 클래스에서 Shift 조건의 AUPRO는 58.74–60.83%다',
    details=['Regular 62.62% 대비 하락은 존재한다. 이 표에는 조건별 비교군이 없으므로 우월한 강건성으로 단정하지 않는다.'],
    notes='Table9 원본. AD2 16-shot seed0 동일 7-class macro. 주 비교 Table1의 8-class macro와 절대값을 직접 비교하지 않는다.')

s,t=slide('실험으로 확인한 범위와 남은 근거',
    '해석 │ 성능 차이와 보정·집계의 상호작용이 현재 핵심 근거다',section='02. 실험 결과',
    note='출처: main.pdf Tables 1–11; 확인 필요 항목은 발표용 종합이며 새 실험 결과가 아님')
table(s,.70,1.95,[2.15,5.3,4.45],['구분','현재 근거','해석 범위 / 추가 확인'],[
    ['최종 성능','AD2 전 shot 최고 평균; 교차 데이터셋 비교','평균 순위와 지표별 이득 보고'],
    ['집계 기여','CQ-off +0.92pp; 보정 후 Rényi 이득','CQ99 효과와 분리해서 설명'],
    ['척도 정렬','정상 참조 분포 + 보정×집계 상호작용','별도 정상 쿼리에서 꼬리 정렬 확인'],
    ['Rényi 선택','고정 순위 규칙 + Mean 대비 개선','최댓값·다른 양의 순위 가중 대조 보강'],
    ['외관·결함 구분','해상도별 반응과 배경에 대한 해석','결함 크기별 순위 진단으로 검증']],row_h=.76)

set_sections(prs,list(sections.items()))
fix_table_ids(prs)
prs.save(OUT)
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(OUT)
print('slides:',len(prs.slides))
print('backup:',BK)
