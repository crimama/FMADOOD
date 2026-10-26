from pathlib import Path
import sys, json, hashlib
from datetime import datetime
SK=Path('/home/hun/.claude/skills/dsba-slide-deck')
sys.path.insert(0,str(SK/'scripts'))
from deck_style import new_deck, frame_slide, table, figure, bullets, text_box
from deck_ops import fix_table_ids, set_sections
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/'cvpr-latex'
ROOT=PAPER.parents[1]
prs=new_deck()
records=[]
I='sec/1_intro.tex'; R='sec/2_related_work.tex'; M='sec/3_method.tex'; E='sec/4_experiments.tex'; A='sec/7_appendix.tex'
H='results/remote_runs/dsba3/scopead_hsa_overlap_20261024/audit_v2/comparison.md'
X='results/remote_runs/dsba3/scopead_axis_analysis_20261021/AXIS_ANALYSIS.md'
XR='results/remote_runs/dsba3/scopead_axis_analysis_20261021/REANALYSIS.md'
ME='results/remote_runs/dsba3/scopead_mechanism_20261020/RESULTS.md'
OLD='PaperWorks/slides/ScopeAD_Method_Experiments_20261019.pptx'
PRIVATE='results/scopead_renyi_private_submission_20261019/RESULTS.md'

def add(title, lead, *, section='01. 피드백과 수정 방향', detail=(), headers=None, rows=None, widths=None, y=None, row_h=.7, size=13, png=None, source='', notes='', footer=''):
    n=len(records)+1
    page=str(n) if n<=14 else 'A-'+str(n-14)
    s,top=frame_slide(prs,section=section,title=title,lead=lead,detail=detail,
        note=footer or source,notes='발표용 종합 초안. 원고 수정 내용과 해석을 구분함.\n출처: '+source+'\n'+notes,page=page)
    if headers:
        table(s,.70,y or max(2.05,top+.1),widths,headers,rows,row_h=row_h,size=size)
    if png:
        figure(s,PAPER/png,max(1.90,top+.04),bottom=6.60)
    records.append(dict(page=page,title=title,lead=lead,detail=list(detail),headers=headers,rows=rows,source=source,notes=notes))
    return s

add('10월 19일 피드백 대응 보고',
    '결론 │ 문제–설계–평가의 연결을 보강했고, HSA 주장은 근거별로 구분했습니다.',
    detail=['보고 기간: 2026.10.19 연구 미팅 → 2026.10.26 현재 원고·분석 기록',
            '핵심 수정: few-shot과 획득 조건 변화의 결합을 정의하고, 방법 전체의 대응 논리를 연결',
            '추가 근거: 조건별 평가·저-FPR 검출률·점수 중첩 진단으로 주장 범위를 점검',
            '발표 순서: 피드백 대응 → 원고·그림 수정 → 실험 근거 → 남은 확인 사항'],
    headers=['항목','이번 보고의 판단'],widths=[2.3,9.64],row_h=.80,y=3.45,
    rows=[['서술 반영','5개 메모에 대응하는 문제 정의·벤치마크 선정·통합 방법 설명 반영'],
          ['실험 해석','조건 변화 성능은 지지됨. HSA의 모든 의미가 해결됐다는 주장은 제한'],
          ['비교 기준','10/19 발표에 있던 방법·성과와 이후 추가한 설명·분석을 구분']],
    source=OLD+'; '+I+'; '+M+'; '+E+'; '+H,
    notes='사용자 제공 10/19 검토 메모 22·5·2·1·4를 기준으로 현재 파일을 비교했다. 모든 변경이 커밋된 상태는 아니므로 현재 작업본 기준이다. 원고나 기존 발표자료는 수정하지 않았다.')

add('다섯 메모를 네 가지 질문으로 묶었습니다',
    '대응 │ 과제 정의, 벤치마크 선정, 방법의 통합성, HSA 근거를 연결했습니다.',
    headers=['메모','질문','현재 반영 내용','근거·상태'],widths=[.8,3.05,5.25,2.84],row_h=.83,y=2.0,size=12,
    rows=[['4','few-shot + shift의 문제가 충분한가?','정상 관측 부족 → 미관측 변동 → 점수 중첩 → 검출 손실','서술 반영 / 3–4쪽'],
          ['1','왜 MVTec AD 2인가?','조명·획득 조건 변화와 저-FPR 평가를 타깃 과제로 명시','범위 명확화 / 5쪽'],
          ['22','왜 이 과제에 맞는 방법인가?','관측 확장·척도 정렬·통합의 역할을 문제와 대응','설계 연결 / 6–7쪽'],
          ['2','방법이 통합됐고 HSA를 줄이는가?','전체 파이프라인 설명과 서로 다른 HSA 진단 추가','근거 혼재 / 8–11쪽'],
          ['5','서론 끝이 궁극적 목표로 돌아오는가?','정상-only 설계와 few-shot·조건 변화 성능으로 마무리','서술 반영 / 12쪽']],
    source='사용자 제공 main.pdf 메모 4·1·22·2·5; '+I+'; '+R+'; '+M+'; '+E,
    notes='메모 번호와 인용은 사용자 메시지 기준. 원고의 현 페이지는 재조판에 따라 달라질 수 있어 섹션 파일을 대응 기준으로 삼았다.')

add('과제의 원인과 실패 양상을 분리했습니다',
    '메모 4 │ 적은 정상 관측과 새로운 획득 조건이 함께 점수 모호성을 만듭니다.',
    headers=['설명 단계','10/19 피드백이 지적한 빈틈','현재 원고의 연결'],widths=[2.05,4.2,5.69],row_h=.82,y=2.05,size=13,
    rows=[['조건 1: few-shot','정상 참조가 적다는 사실만으로는 실패 원인이 불명확','소수의 정상 support가 정상 외관 변동을 충분히 덮지 못함'],
          ['조건 2: 획득 변화','일반적인 distribution shift와 목표 조건의 관계가 약함','조명·시점·반사 등 support에서 보지 못한 정상 변동 발생'],
          ['실패: HSA','높은 점수의 중첩만 설명하면 과제와의 연결이 약함','미관측 정상도 최근접 거리가 커져 결함의 높은 점수와 혼재'],
          ['결과: 저-FPR 손실','모호성이 최종 검출에 왜 중요한지 설명 필요','오경보 억제에 필요한 임계값 상승이 결함 누락으로 이어짐']],
    source=I+' (문제 정의 및 High-Score Ambiguity 단락)',
    notes='기존 문구는 사용자 메모를 요약한 것이며 10/19 TeX 원문의 완전 복원은 아니다. 현재 연결은 원고의 문단을 발표용으로 재구성했다. HSA는 높은 점수의 원인을 거리 하나만으로 구분하기 어려운 현상으로 정의하며 모든 shift의 일반 이론으로 주장하지 않는다.')

add('실제 사례로 HSA의 의미를 보강했습니다',
    '그림 수정 │ 미관측 조명에서 정상과 결함이 함께 높은 점수를 얻는 상황을 보여줍니다.',
    section='02. 문제와 방법의 서술 보강',png='figures/fig_intro_ambiguity.png',
    source='figures/fig_intro_ambiguity.png; figures/source/README.md; '+I,
    footer='Figure 1 원본 재사용 · walnuts 실제 이미지 · 점수 분포와 임계값은 개념도이며 실측 통계가 아님',
    notes='원본 그림을 변형하지 않았다. support: train/good/129_regular, 266_regular. query: good/000_overexposed, bad/004_overexposed. GT contour는 표시용. 도식의 분포 겹침은 정량 HSA 개선 증거가 아니다. 원본 PNG의 소형 라벨은 발표 화면에서 확대 설명 필요.')

add('MVTec AD 2의 선정 이유를 한정했습니다',
    '메모 1 │ 정상 support를 제한한 상태에서 조명·획득 변화의 영향을 평가합니다.',
    section='02. 문제와 방법의 서술 보강',
    detail=['데이터셋 자체가 few-shot인 것은 아님: train/good에서 정상 support 수를 제한해 과제를 구성',
            '주요 표적: 학습에서 보지 못한 조명·획득 조건에서 저-FPR 국소화가 유지되는지 확인',
            '주장 범위: AD 2의 조건 변화 결과를 중심으로 설명하고, 모든 distribution shift로 일반화하지 않음'],
    headers=['평가 축','역할','해석 범위'],widths=[2.4,4.4,5.14],row_h=.85,y=3.3,
    rows=[['MVTec AD 2','few-shot × 획득 조건 변화 × 저-FPR','이번 논문의 핵심 검증 환경'],
          ['MVTec AD · VisA','기존 산업 이상 탐지 벤치마크와 비교','보완적 검증이며 모든 shift의 대체 근거는 아님'],
          ['AeBAD 등 관련 연구','획득 조건 변화를 다룬 선행연구의 위치 설명','현재 원고에서 실험 검증한 데이터셋으로 보지 않음']],
    source=R+' (획득 조건 변화와 저-FPR 국소화); '+E,
    notes='피드백의 luminance를 조명 변화로 명시하여 일반 shift 표현과 연결했다. AeBAD는 관련연구 맥락이며 이 자료에서 새 실험을 수행하지 않았다.')

add('각 설계의 역할을 하나의 흐름으로 연결했습니다',
    '메모 22·2 │ 관측을 늘리고, 비교 척도를 맞춘 뒤, 결함 단서가 희석되지 않게 통합합니다.',
    section='02. 문제와 방법의 서술 보강',
    headers=['문제','설계','과제에 대한 대응 논리','한계'],widths=[2.25,2.4,4.6,2.69],row_h=.90,y=2.05,size=12,
    rows=[['정상 참조의 관측 범위가 좁음','레이어·해상도별 독립 관측','서로 다른 공간·의미 수준에서 상보적 최근접 단서 확보','support 밖 변동을 학습하는 것은 아님'],
          ['관측마다 거리 단위가 다름','정상 참조 Q95 척도 정렬','정상-only 기준으로 관측 간 점수 비교 가능성을 높임','FPR 보장이나 shift 추정은 아님'],
          ['깊은 의미와 얕은 세부 정보의 차이','비대칭 계층 결합','깊은 층을 중심으로 얕은 층의 국소 단서를 보완','가중치의 범용 최적성은 미입증'],
          ['평균이 강한 결함 반응을 희석','Rényi 순위 집계','해상도별 반응을 정렬하고 높은 반응에 더 큰 비중','높은 정상 점수도 강조될 수 있음']],
    source=M+'; '+A,
    notes='이 표는 발표용 설계 논리 종합이다. 설계가 HSA를 언제나 제거한다는 인과적 보장은 없다. 현재 method: layer 5/11/23, resolution 448/672/896/1120, bags 4, Q95, shallow weight 0.125. 정상 support만 fitting하며 GT는 사후 진단에 사용한다.')

add('방법 그림도 통합 파이프라인으로 정리했습니다',
    '그림 수정 │ 정상 참조 적합부터 다중 관측·척도 정렬·순위 집계까지 연결합니다.',
    section='02. 문제와 방법의 서술 보강',png='figures/fig_scopead_overview.png',
    source='figures/fig_scopead_overview.png; figures/source/draw_overview.py; '+M,
    footer='Figure 2 원본 재사용 · 실제 walnuts 결함 픽셀 사례 · 표시된 최종 맵은 CQ99 적용 전',
    notes='정상 support fit → layer/resolution별 1-NN → Q95 → asymmetric layer fusion → ordered aggregation 순서로 설명한다. 패널(d)은 실제 결함 픽셀 row 380,col 133. 그림의 한 사례를 일반 성능 보장으로 해석하지 않는다. 원본 PNG의 소형 라벨은 발표 화면에서 확대 설명 필요.')

add('조건 변화에서의 성능과 임계값 손실을 분리했습니다',
    '결과 │ 조건 변화 AUPRO가 높고, 공통 임계값 사용에 따른 추가 손실이 작습니다.',
    section='03. 추가 근거와 주장 범위',
    detail=['변화 조건 AUPRO와 공통 임계값 손실을 함께 제시해, 단순 전체 평균의 빈틈을 보완'],
    headers=['방법','2-shot 변화 AUPRO','2-shot 손실','16-shot 변화 AUPRO','16-shot 손실'],
    widths=[2.6,2.65,1.9,2.85,1.94],row_h=.64,y=2.35,size=12,
    rows=[['PatchCore','42.30%','−0.16 pp','52.97%','−0.46 pp'],
          ['AnomalyDINO','39.52%','−1.83 pp','44.12%','−2.74 pp'],
          ['SubspaceAD','49.65%','−5.28 pp','52.37%','−6.52 pp'],
          ['SuperAD','49.96%','−0.33 pp','56.33%','−0.80 pp'],
          ['ScopeAD','56.60%','−0.05 pp','60.74%','−0.27 pp']],
    source=E+' (tab:condition); 비교 방법은 본문 표의 일부',
    notes='AD2 test_public full8, seed0/1/2 평균, AUPRO@0.05. 변화 AUPRO는 over/under/shift 조건별 AUPRO 평균. 손실은 모든 조건을 합쳐 공통 임계값을 sweep한 AUPRO에서 regular 포함 전체 조건별 AUPRO 평균을 뺀 값. 운영 환경의 고정 임계값 보장이 아니다. 전체 shot 손실 0.05–0.27pp이나 regular 대비 변화 조건 성능은 1.01–1.72pp 낮다. CQ99 포함 기존 메인 평가 결과를 재분석한 것으로 전부 신규 성능은 아니다.',
    footer='AD2 public full8 · 3 seeds · AUPRO@0.05 · 손실 = 공통 sweep − 조건별 평균 · CQ99 포함 평가')

add('저-FPR 검출률로 HSA 대응 근거를 추가했습니다',
    '결과 │ 원고의 사후 진단에서는 전체 관측 확장이 동일 FPR의 결함 검출률을 높였습니다.',
    section='03. 추가 근거와 주장 범위',
    detail=['FPR 1%에서 단일 깊은 층 대비 +9.71 pp / +10.26 pp; Q95 단독 개선과는 구분'],
    headers=['진단 구성','2-shot FPR 1%','2-shot FPR 5%','16-shot FPR 1%','16-shot FPR 5%'],
    widths=[3.1,2.2,2.2,2.22,2.22],row_h=.78,y=2.35,size=12,
    rows=[['Raw L23 · 1120','36.24%','56.88%','39.78%','60.65%'],
          ['Raw 계층 결합 · 1120','39.65%','60.71%','43.01%','62.81%'],
          ['Q95 계층 결합 · 1120','38.51%','60.68%','42.71%','62.53%'],
          ['전체 ScopeAD','45.95%','66.14%','50.04%','66.95%']],
    source=E+' (tab:hsa_direct_validation); run_hsa_eval.py; run_hsa_eval_s16.py',
    footer='원고 기재값 · full8 · seed 0 · CQ 미적용 사후 진단 · 정상 픽셀 부분표집 · 원 JSON/공식 평가와의 일치 추가 확인',
    notes='수치는 현재 원고 표와 대조했다. 원시 결과 JSON은 로컬에서 확인하지 못했다. 스크립트는 seed0 캐시를 사용하고 good 및 bad의 정상 배경을 모아 100개당 1개를 threshold 추정에 사용하며 arm/class별 percentile 99/95를 정한다. GT는 사후 사용. 공식 full-resolution 3seed AUPRO와 프로토콜이 다르다. bag averaging/interpolation parity도 확인 필요. 정상 모집단 전체에서 모든 HSA가 해소됐다는 결론으로 확대하면 안 된다.')

add('점수 중첩 진단은 별도의 한계를 드러냈습니다',
    '한계 │ 비슷한 고점수 쌍의 순위 회복은 확인되지 않아, HSA 해소 주장을 한정해야 합니다.',
    section='03. 추가 근거와 주장 범위',
    detail=['10/24 완료 분석: 1120 점수로 뽑은 정상–결함 유사 점수 쌍에서 결함의 순위 우위를 측정',
            '저-FPR recall과 선택 모집단·지표가 다름: 긍정 결과와 부정 결과를 함께 해석',
            '해석 제한: 기준 맵으로 고른 쌍이므로, 전체 모집단의 HSA 악화로 단정할 수 없음'],
    headers=['nonregular','1120 단독','4-scale 평균','4-scale Rényi','Rényi − 1120'],
    widths=[1.6,2.1,2.2,2.2,3.84],row_h=.82,y=3.2,size=13,
    rows=[['2-shot','82.79%','71.64%','72.09%','−10.70 pp\n95% CI [−17.91, −4.10]'],
          ['16-shot','82.22%','69.73%','69.98%','−12.24 pp\n95% CI [−21.37, −5.00]']],
    source=H,
    footer='full8 · 3 seeds · CQ off · 1120 정상 상위 5%, |log score gap| ≤ 0.02 매칭 · class bootstrap CI',
    notes='audit_v2 최신 재집계. Concordance: defect score>normal score이면1, tie0.5. image→class/seed→class macro. 기준 맵으로 선택한 조건부 쌍이며 최대 개수 매칭은 score gap 부호를 균형화하지 않는다. 기준 맵의 순위가 선택 단계에서 유리해질 수 있어 음의 차이를 전체 모집단 HSA 악화로 단정할 수 없다. Rényi−mean 차이는 2/16shot +0.45/+0.25pp, CI 모두0포함. 이 분석만으로 calibration이나 layer expansion 효과를 분리할 수 없다.')

add('Q95와 순위 집계의 역할을 분리해 해석했습니다',
    '해석 │ 척도 정렬과 큰 반응 보존은 뒷받침되지만, 특정 결합의 최적성은 입증되지 않았습니다.',
    section='03. 추가 근거와 주장 범위',
    headers=['추가 분석','관측','이번에 정리한 해석'],widths=[2.7,4.3,4.94],row_h=.98,y=2.1,size=12,
    rows=[['Q95 × Rényi 상호작용','2/16-shot +0.544 / +0.690 pp\n클래스 CI는 0 포함','상보성의 경향은 있으나 안정적인 상승 보장으로 표현하지 않음'],
          ['해상도별 척도 보정','global 대비 Rényi +0.551 / +0.822 pp\nfabric 제외 시 −0.110 / −0.098 pp','평균 이득은 클래스 의존적이며 보편적 축별 효과는 아님'],
          ['순위와 결함 크기 분리','rank-only +0.05 / −0.14 pp\nsize-only +0.46 / +0.70 pp','순위 자체보다 큰 결함 반응을 보존하는 효과와 연결해 설명'],
          ['집계·계층 가중치 민감도','선형 순위 가중과 차이 <0.1 pp\n얕은 층 λ=0.125가 모든 shot 최적은 아님','Rényi 이론적 최적성·고정 λ의 범용 최적성 주장 제한']],
    source=ME+'; '+X+'; '+XR+'; '+A,
    notes='첫 행 mechanism 10/20의 48/48 replay, 26 interventions, parity0 근거. 다음 행 10/21 axis 분석. 분석별 CQ, 설정, arm의 차이가 있어 숫자를 하나의 성능 표로 합치지 않았다. Q95 이후 상호작용 수치는 10/19 덱의 +0.75/+0.68과 프로토콜이 다르므로 새로 갱신된 동일 수치로 표현하지 않는다.')

add('서론부터 결론까지 같은 목표로 연결했습니다',
    '메모 5 │ 설명의 끝을 few-shot·조건 변화의 저-FPR 국소화 목표로 되돌렸습니다.',
    section='04. 대응 현황과 다음 확인',
    headers=['원고 위치','주요 수정','발표에서 전달할 의미'],widths=[2.2,5.2,4.54],row_h=.70,y=2.05,size=12,
    rows=[['서론','HSA의 원인과 검출 손실을 연결하고 기여·마무리 문단 보강','왜 이 문제를 다루고 무엇을 검증하는지 먼저 설명'],
          ['관련연구','획득 조건 변화·저-FPR 국소화 및 AD 2의 위치 명시','일반 shift 표현과 실제 조명·획득 평가의 범위를 일치'],
          ['방법·Figure 2','정상-only 관측 확장 → 척도 정렬 → 비대칭 결합 → 순위 집계','모듈 나열을 넘어 각 설계가 대응하는 문제를 연결'],
          ['실험·부록','조건별·공식 private 평가, low-FPR recall, 기전·민감도 분석 보강','강건성 결과와 HSA 진단의 지지 범위를 구분'],
          ['논의·한계','HSA의 직접 근거, 클래스별 효과, 배포 임계값의 한계를 명시','전체 성능 상승과 직접적인 모호성 해소를 구분'],
          ['결론·주장 점검','궁극적 목표로 연결하되 포괄적 HSA 해소 표현은 추가 정합화 필요','검출 개선과 모든 고점수 쌍의 구분은 같은 주장이 아님']],
    source=I+'; '+R+'; '+M+'; '+E+'; '+A+'; sec/5_discussion.tex; sec/6_conclusion.tex',
    notes='서론의 현재 마지막 문단은 정상-only 기준, query 적응 없음, 모든 shot AD2 저FPR 및 조건 변화 성능, 기존 AD/VisA 평가로 닫힌다. 결론에 대한 표의 마지막 행은 발표용 해석과 향후 문구 점검 사항이다. broad HSA mitigation 표현을 이번 작업에서 원고에 새로 수정하지는 않았다.')

add('서술 반영과 실증 해결 수준을 구분했습니다',
    '대응 현황 │ 설명은 반영됐고, 실험으로 뒷받침되는 주장 범위는 항목마다 다릅니다.',
    section='04. 대응 현황과 다음 확인',
    headers=['피드백','현재 대응 수준','남은 부분'],widths=[2.65,5.15,4.14],row_h=.84,y=2.05,size=12,
    rows=[['메모 4 · 과제 정의','few-shot 관측 부족과 획득 변화의 결합을 명시','전체 shift를 대표한다는 인상은 피하기'],
          ['메모 1 · AD 2 선정','조명·획득 조건 변화 및 support 제한의 이유 설명','외부 shift 일반화는 추가 검증 범위'],
          ['메모 22 · 과제 맞춤 설계','문제–관측–척도–통합의 역할을 연결','각 단계의 독립적 인과 효과는 제한적으로 해석'],
          ['메모 2 · 통합성·HSA 근거','통합 구조와 low-FPR 진단을 추가','유사 고점수 쌍 회복은 미확인; 진단 간 범위 구분'],
          ['메모 5 · 서론 마무리','few-shot·조건 변화·저-FPR 목표로 복귀','원고의 포괄적 HSA 표현과 분석 결과의 정합화']],
    source=I+'; '+R+'; '+M+'; '+E+'; '+H,
    notes='대응 완료를 과학적 문제 전체의 해결로 오해하지 않도록 반영 수준을 구분했다. 특히 memo2는 구조 설명의 반영과 경험적 HSA 입증을 나누어 보고한다.')

add('다음 확인은 주장과 평가 조건의 정합성입니다',
    '우선순위 │ HSA 표현, CQ 유무, 결과 출처를 맞춘 뒤 최종 주장 범위를 확정해야 합니다.',
    section='04. 대응 현황과 다음 확인',
    headers=['순서','확인 항목','판단 기준'],widths=[.8,4.2,6.94],row_h=.99,y=2.05,size=13,
    rows=[['1','HSA 주장의 범위','조건 변화 저-FPR 성능, low-FPR recall, 조건부 쌍 순위 회복을 별개 주장으로 기술'],
          ['2','원고 방법과 메인 결과의 CQ 조건','현재 방법에서 CQ 설명을 뺀 상태와 CQ99 포함 메인 평가를 일치시킬 방향 결정'],
          ['3','HSA 진단의 재현성','seed0 원 JSON 확보, 정상 픽셀 표집·집계 순서·공식 평가와의 일치 확인'],
          ['4','최신 private 결과의 출처 연결','원고의 10/24 DONE 기록과 공식 결과 파일·기존 제출 로그를 연결']],
    source=E+'; '+M+'; '+H+'; '+PRIVATE,
    notes='후속 확인 계획안이며 새 실험 실행·원고 변경을 수행하거나 승인받은 것으로 해석하지 않는다. 현 자료는 현재 파일과 분석 기록을 정리한 발표 초안이다. 공식 서버 최신 상태를 이번 작업에서 조회하지 않았다.')

add('10월 19일 성과와 이후 보강을 구분했습니다',
    '비교 기준 │ 핵심 방법과 대표 성능은 기존 성과이며, 이후에는 논증과 진단을 보강했습니다.',
    section='Appendix A. 비교 기준과 근거',
    headers=['구분','10/19 발표에 이미 포함','이후 현재 자료에서 확인한 보강'],widths=[2.1,4.6,5.24],row_h=.97,y=2.05,size=12,
    rows=[['핵심 방법','Q95 + 비대칭 계층 결합 + Rényi 집계\n48개 bank · CQ99','few-shot·shift 문제와 각 설계의 역할 연결\n현재 방법의 CQ 서술 제거는 결과 변경과 구분'],
          ['대표 성능','AD2 1-shot 52.18%, 16-shot 60.74%\nCQ on/off와 비용·ablation','조건별 성능 해석과 공통 임계값 손실 설명\n기존 수치를 이번 주 신규 상승으로 표시하지 않음'],
          ['기전 근거','Q95 이후 상호작용 +0.75/+0.68 pp\n다음 확인 질문','10/20 mechanism · 10/21 축별 분석\n10/24 matched-pair HSA 및 원고 recall 진단'],
          ['원고 전달','방법 설명과 기본 실험','서론 인과관계, AD2 선정 이유, 통합 설명\n실제 사례 Figure 1·overview와 주장 범위 점검']],
    source=OLD+' (당시 슬라이드 10, 17, 22); git 14b235c (2026-10-21); 현재 원고; '+ME+'; '+H,
    notes='10/17 git 상태를 10/19 원고의 정확한 대용으로 삼지 않았다. 사용자의 실제 10/19 슬라이드 및 피드백과 현재 작업본을 대응했다. 현재 파일의 모든 변경에 정확한 작성 일자를 부여할 수는 없다. 10/19 기존 정성 그림을 새 실험으로 분류하지 않았다.')

add('축별 보정의 평균 이득은 클래스 의존적입니다',
    '부록 근거 │ 해상도 척도 정렬이 평균 이득을 보였지만, fabric 제외 시 같은 결론이 유지되지 않습니다.',
    section='Appendix A. 비교 기준과 근거',
    headers=['Q95 보정 축','2-shot 평균','2-shot Rényi','16-shot 평균','16-shot Rényi'],
    widths=[3.1,2.2,2.2,2.22,2.22],row_h=.55,y=2.05,size=12,
    rows=[['없음','56.118','56.321','59.734','59.725'],['global','56.119','56.322','59.735','59.725'],
          ['layer','55.910','56.130','59.900','59.824'],['resolution','56.265','56.872','59.885','60.547'],
          ['separable','56.021','56.779','60.025','60.727'],['full','56.030','56.777','60.057','60.738']],
    source=X+'; '+XR,
    footer='AUPRO@0.05 (%) · 2/16-shot · 축별 사후 재분석 · full−separable 효과는 거의 없음',
    notes='resolution−global Rényi +0.551/+0.822pp. 95% class CI [-0.362,1.992]/[-0.404,2.795]로0포함. fabric 제외 -0.110/-0.098pp. layer-only 효과는 shot별 다르고 full−separable 차이는 작다. 이 분석을 신규 채택한 방법이나 일반 보장으로 표현하지 않는다.')

add('평가 조건과 결과 기록의 차이를 표시했습니다',
    '근거 점검 │ 원고의 최신 기록을 반영하되, CQ 조건과 공식 결과 출처는 추가로 맞춰야 합니다.',
    section='Appendix A. 비교 기준과 근거',
    headers=['항목','확인한 현재 상태','해석·확인 사항'],widths=[2.1,5.1,4.74],row_h=1.15,y=2.05,size=12,
    rows=[['CQ 조건','현재 Method에서 CQ 설명 제거\n메인 결과 설정은 component_quantile','문구 제거가 CQ-off 재평가를 뜻하지 않음\n메인 성능과 CQ-off 진단을 구분'],
          ['Private 상태','원고: 10/24 확인한 Official DONE 주석\nAUPRO private/mixed 66.53/64.40%','기존 로컬 로그의 10/23 SUBMITTING은 과거 상태\n공식 결과 파일과 원고 수치 연결 필요'],
          ['Private 지표 범위','ScopeAD Seg-F1 36.33/32.47%\nSuperAD Seg-F1 39.43/33.86%','AUPRO 우위를 모든 지표의 우위로 확장하지 않음'],
          ['HSA recall 출처','현재 원고 표와 seed0 진단 스크립트 확인\n원 JSON은 로컬에서 확인 못함','원고 기재값으로 보고하고\n공식 full3seed 평가와 동일시하지 않음']],
    source=M+'; '+E+' (tab:private_comparison); '+PRIVATE+'; configs/benchmarks/scopead_renyi_cq_20261018/',
    notes='원고 DONE submission id 6b770950-5c57-4120-87c1-6f5190f3d3df, checked2026-10-24. 이번 작업에서 공식 서버 라이브 상태를 검증하지 않았다. 원고가 과거 로그보다 최신이므로 pending만으로 현재 상태를 설명하면 안 된다.')

add('원래 메모와 원고 대응 위치를 보존했습니다',
    '참조 │ 다섯 피드백의 원문 취지와 현재 대응 위치를 발표 노트에 함께 남겼습니다.',
    section='Appendix A. 비교 기준과 근거',
    headers=['메모·당시 쪽','원문 취지','대응 위치'],widths=[1.7,6.0,4.24],row_h=.83,y=2.05,size=12,
    rows=[['22 · p3','왜 few-shot distribution shift에 특화됐고\n어떻게 high-score ambiguity를 해결하는가?','sec/3_method.tex\n본문 6–11쪽'],
          ['5 · p2','HSA 완화와 few-shot + shift 강건성이\n궁극적 목표라는 마무리가 필요','sec/1_intro.tex\n본문 12–13쪽'],
          ['2 · p3','방법이 통합된 인상과\n앞서 제시한 문제를 해결하는 근거가 필요','sec/3_method.tex; sec/4_experiments.tex\n본문 6–11쪽'],
          ['1 · p2','왜 few-shot에서 AD2를 우선 타깃으로 삼고\n조명 강건성과 일반 shift를 어떻게 연결하는가?','sec/2_related_work.tex\n본문 5쪽'],
          ['4 · p1','현재 HSA 정의가 few-shot + shift의\n문제를 충분히 설명하는가?','sec/1_intro.tex\n본문 3–4쪽']],
    source='사용자 제공 main.pdf 검토 메모 22·5·2·1·4; 현재 원고 섹션',
    notes='메모22: 지금 방법론 매커니즘 자체에 대한 전달력은 좋다고 느껴짐. 다만 전반적으로 이 방법이 왜 few-shot distribution shift에 특화된 방법이고, 어떻게 high-score ambiguity를 해결하는지에 대한 서술이 부족하다고 느껴짐.\n메모5: 궁극적으로 해결하고자 하는 것은 결국 High-score ambiguity를 완화하고, few-shot + distribution shift에서 강건한 성능을 내는 것이라는 마무리가 필요할 것으로 보임.\n메모2: 방법론이 좀 더 통합되어 있는 늬앙스가 필요. 그리고 이 방법론을 통해 앞서 제시한 문제들이 해결되고 있는지에 대한, 특히 high-score ambiguity가 해결되는지가 보여야 함.\n메모1: Few-shot인데 왜 이 데이터셋을 우선적으로 타겟으로 하는지. MVTecAD2의 luminance와 서론의 일반적 distribution shift 관계 및 강건성 언급 필요.\n메모4: 이걸로 충분히 Few-shot + distribution shift라는 task의 문제점을 온전히 설명하고 있는걸까 하는 의문.\n원문 메모와 기존 파일은 삭제·수정하지 않았다.')

fix_table_ids(prs)
set_sections(prs,[('01. 피드백과 수정 방향',list(prs.slides)[:3]),('02. 문제와 방법의 서술 보강',list(prs.slides)[3:7]),('03. 추가 근거와 주장 범위',list(prs.slides)[7:11]),('04. 대응 현황과 다음 확인',list(prs.slides)[11:14]),('Appendix A. 비교 기준과 근거',list(prs.slides)[14:])])
prs.save(HERE/'ScopeAD_Feedback_Response_20261026.pptx')
(HERE/'slides.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
md=['# 2026.10.26 연구 미팅: 10/19 피드백 대응','', '비교 기준: 2026.10.19 발표자료와 사용자 제공 원고 메모. 보고 대상: 2026.10.26 현재 작업본. 본문 14장, 부록 4장. 원고와 기존 발표자료는 수정하지 않음.','', '표현 원칙: 서술 반영, 성능 근거, HSA의 직접 진단, 남은 검증을 구분한다. 직접 구성한 해석은 발표용 종합 초안이다.','']
for r in records:
    md += ['## '+r['page']+'. '+r['title'],'',r['lead'],'']
    md += ['- '+x for x in r['detail']]
    if r['headers']:
        md += ['', '| '+' | '.join(r['headers'])+' |','| '+' | '.join(['---']*len(r['headers']))+' |']
        md += ['| '+' | '.join(x.replace('\n','<br>') for x in row)+' |' for row in r['rows']]
    md += ['', '**발표 노트:** '+r['notes'],'','**근거:** '+r['source'],'']
(HERE/'ScopeAD_Feedback_Response_20261026.md').write_text('\n'.join(md))
paths=[PAPER/f for f in [I,R,M,E,A,'sec/5_discussion.tex','sec/6_conclusion.tex','figures/fig_intro_ambiguity.png','figures/fig_scopead_overview.png','figures/source/README.md']]
paths += [ROOT/f for f in [H,X,XR,ME,PRIVATE,OLD]]
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
(HERE/'source_sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Built',len(prs.slides),'slides in',HERE)
