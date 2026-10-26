from pathlib import Path
import json,sys,shutil,hashlib
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_DATA_LABEL_POSITION, XL_TICK_MARK
SK=Path('/home/hun/.codex/skills/dsba-slide-deck')
sys.path.insert(0,str(SK/'scripts'))
from deck_style import new_deck,frame_slide,figure,table,text_box,bullets,NAVY,INK,FONT,BOLD
from deck_ops import fix_table_ids,set_sections
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'build_feedback_20261026'
PAPER=HERE.parents[1]/'cvpr-latex'
rows=json.loads((OLD/'slides.json').read_text())
original=Presentation(HERE.parent/'ScopeAD_Feedback_Response_20261026.pptx')
prs=new_deck(); log=[]
TITLES=[
'10월 19일 피드백 대응 보고','피드백의 핵심 쟁점과 대응 방향','Few-shot·획득 조건 변화의 문제 정의',
'실제 사례를 통한 HSA 개념 보강','MVTec AD 2 선정 근거','관측 확장과 점수 통합의 설계 논리',
'정상 참조 기반의 통합 파이프라인','획득 조건 변화 성능과 임계값 손실','저-FPR 결함 검출률의 단계별 비교',
'유사 고점수 쌍의 순위 회복 진단','Q95·순위 집계의 효과와 해석 범위','서론부터 결론까지의 논증 정리',
'피드백 반영 수준과 남은 검증','주장·평가 조건·출처의 정합성 점검','10월 19일 성과와 이후 보강 내용',
'축별 척도 보정의 상세 결과','CQ 조건·Private 결과·진단 출처','원문 피드백과 원고 대응 위치']
LEADS=[
'요약 │ 문제 정의·통합 설계·실험 해석의 보강과 HSA 주장 범위의 구분',
'대응 방향 │ 과제 정의, 벤치마크 선정, 방법 통합성, HSA 근거의 연결',
'메모 4 │ 정상 참조 부족과 미관측 획득 변화에 따른 고점수 모호성',
'그림 수정 │ 조명 변화에서 정상·결함의 높은 매칭 거리와 점수 중첩',
'메모 1 │ 정상 support 제한과 조명·획득 조건 변화의 결합 평가',
'메모 22·2 │ 관측 확대, 척도 정렬, 결함 반응 보존을 위한 단계별 설계',
'그림 수정 │ 정상 참조 적합부터 독립 매칭·계층 결합·순위 집계까지의 연결',
'결과 │ 높은 변화 조건 AUPRO와 작은 공통 임계값 손실',
'결과 │ FPR 1% 기준 전체 관측 확장의 결함 검출률 개선',
'한계 │ 조건부 고점수 쌍의 순위 회복 미확인과 모집단 해석의 제한',
'해석 │ 척도 정렬·큰 반응 보존의 기여와 특정 결합의 최적성 한계',
'메모 5 │ Few-shot·획득 조건 변화·저-FPR 국소화 목표로의 논증 연결',
'대응 현황 │ 서술 반영, 성능 근거, 직접 HSA 검증의 구분',
'다음 확인 │ HSA 표현, CQ 적용 조건, 결과 출처의 일치',
'비교 기준 │ 기존 핵심 방법·대표 성능과 이후 추가된 논증·진단의 구분',
'부록 근거 │ 해상도 척도 정렬의 평균 이득과 클래스 의존성',
'근거 점검 │ 현재 원고와 평가 설정·결과 기록 사이의 차이',
'참조 │ 메모 22·5·2·1·4의 취지와 현재 원고 위치의 대응']

def frame(i,footer=None):
    r=rows[i-1]
    section=('01. 피드백과 수정 방향' if i<=3 else '02. 문제와 방법의 서술 보강' if i<=7 else '03. 추가 근거와 주장 범위' if i<=11 else '04. 대응 현황과 다음 확인' if i<=14 else 'Appendix A. 비교 기준과 근거')
    note=original.slides[i-1].notes_slide.notes_text_frame.text
    note+='\n\nv2 편집: 제목·리드의 명사형 통일, 표의 설명·차트 전환. 원 수치와 해석 제한 유지.\n'
    if r.get('headers'):
        note+='이전 표의 전체 내용(근거 보존):\n'+' | '.join(r['headers'])+'\n'
        note+='\n'.join(' | '.join(row) for row in r['rows'])
    s,top=frame_slide(prs,section=section,title=TITLES[i-1],lead=LEADS[i-1],
          note=footer or r['source'],notes=note,page=r['page'])
    log.append({'page':r['page'],'title':TITLES[i-1],'lead':LEADS[i-1],'source':r['source'],'notes':note,'content':[]})
    return s

def blocks(s,items,start=2.0,step=1.08):
    for j,(head,body) in enumerate(items):
        y=start+j*step
        text_box(s,.70,y,11.94,.36,head,18,NAVY,BOLD,True,name='BodyHeading'+str(j))
        bullets(s,[body],size=16,y=y+.42,name='Explanation'+str(j))
        log[-1]['content'].append(head+' — '+body)

def chart(s,categories,series,x,y,w,h,maxval,*,horizontal=False,fmt='0.00',legend=True):
    d=CategoryChartData();d.categories=categories
    for name,values in series: d.add_series(name,values)
    typ=XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    c=s.shapes.add_chart(typ,Inches(x),Inches(y),Inches(w),Inches(h),d).chart
    c.has_title=False;c.has_legend=legend
    c.font.name=FONT;c.font.size=Pt(12)
    if legend:
        c.legend.position=XL_LEGEND_POSITION.BOTTOM;c.legend.include_in_layout=False
        c.legend.font.name=FONT;c.legend.font.size=Pt(11)
    colors=['8C9EAD',NAVY]
    for n,se in enumerate(c.series):
        se.format.fill.solid();se.format.fill.fore_color.rgb=RGBColor.from_string(colors[n%2])
        se.format.line.fill.background()
    ca,va=c.category_axis,c.value_axis
    ca.tick_labels.font.name=FONT;ca.tick_labels.font.size=Pt(11)
    va.tick_labels.font.name=FONT;va.tick_labels.font.size=Pt(10)
    ca.major_tick_mark=XL_TICK_MARK.NONE;ca.minor_tick_mark=XL_TICK_MARK.NONE
    va.major_tick_mark=XL_TICK_MARK.NONE;va.minor_tick_mark=XL_TICK_MARK.NONE
    va.minimum_scale=0;va.maximum_scale=maxval
    va.tick_labels.number_format='0'
    va.has_major_gridlines=True
    va.major_gridlines.format.line.color.rgb=RGBColor.from_string('E2E7ED')
    if horizontal: ca.reverse_order=True
    plot=c.plots[0];plot.gap_width=70;plot.has_data_labels=True
    dl=plot.data_labels;dl.position=XL_DATA_LABEL_POSITION.OUTSIDE_END
    dl.font.name=FONT;dl.font.size=Pt(10);dl.font.color.rgb=RGBColor.from_string(INK)
    dl.number_format=fmt
    log[-1]['content'].append({'categories':categories,'series':series})
    return c

s=frame(1, '보고 기간 2026.10.19–10.26 · 현재 원고·분석 기록 기준 · 본문 14장 / 부록 4장')
blocks(s,[('원고 서술의 보강','Few-shot과 획득 조건 변화의 결합, MVTec AD 2 선정 이유, 통합 방법의 대응 논리'),
          ('실험 근거의 추가','조건별 성능, 저-FPR 검출률, 유사 고점수 쌍 분석을 통한 주장 범위 점검'),
          ('이번 보고의 구분','10/19 기존 성과와 이후 보강 내용, 서술 반영과 실증 해결 수준의 구분')],step=1.30)

s=frame(2,'사용자 제공 main.pdf 메모 4·1·22·2·5 · 각 메모의 상세 대응 위치는 부록 A-4 참조')
blocks(s,[('문제 정의 · 메모 4','정상 관측 부족 → 미관측 정상 변동 → 높은 점수의 중첩 → 저-FPR 결함 누락'),
          ('벤치마크 선정 · 메모 1','일반적인 shift 표현에서 조명·획득 조건 변화라는 실제 검증 범위로 구체화'),
          ('통합 방법과 검증 · 메모 22·2','관측 확장·척도 정렬·통합의 역할 연결 및 HSA 진단 추가'),
          ('서론의 마무리 · 메모 5','Few-shot·조건 변화에서의 HSA 완화와 강건한 국소화라는 목표로 복귀')])

s=frame(3)
blocks(s,[('정상 참조의 희소성','소수의 정상 support에 담기 어려운 정상 외관의 변동 범위'),
          ('미관측 획득 조건','조명·시점·반사 변화로 인한 정상 패치의 최근접 거리 상승'),
          ('High-Score Ambiguity','미관측 정상과 실제 결함의 고점수 중첩 및 단일 거리값의 원인 구분 한계'),
          ('저-FPR 국소화 손실','정상 오경보 억제를 위한 임계값 상승과 동반되는 결함 영역 누락')])

s=frame(4,'Figure 1 원본 재사용 · walnuts 실제 이미지 · 분포와 임계값은 실측 통계가 아닌 개념도')
figure(s,PAPER/'figures/fig_intro_ambiguity.png',1.92,bottom=6.62)

s=frame(5)
blocks(s,[('Few-shot 과제의 구성','AD 2 자체의 few-shot 속성이 아닌, train/good 정상 support 수 제한을 통한 평가 설정'),
          ('조명·획득 변화의 명시적 검증','새로운 획득 조건에서의 저-FPR 국소화 성능과 공통 임계값 손실 평가'),
          ('벤치마크별 역할의 구분','핵심 검증은 AD 2, 보완적 비교는 MVTec AD·VisA, 관련연구 맥락은 AeBAD'),
          ('일반화 범위의 한정','관측한 조명·획득 조건에 대한 결과와 모든 distribution shift의 강건성 보장의 구분')])

s=frame(6)
blocks(s,[('관측 확장 · 정상 참조의 비교 범위 확대','레이어·해상도별 독립 매칭을 통한 국소 외관·전역 문맥의 상보적 단서 확보'),
          ('Q95 척도 정렬 · 관측 간 거리 단위 보정','정상 support 내부의 교차 매칭을 기준으로 한 점수 척도 정렬'),
          ('비대칭 계층 결합 · 의미 단서와 세부 단서의 조합','깊은 층을 주축으로 한 결합과 작은 가중치의 얕은 층 보완'),
          ('Rényi 순위 집계 · 강한 국소 반응의 보존','해상도별 점수 정렬과 상위 반응 가중을 통한 평균 희석 완화')])

s=frame(7,'Figure 2 원본 재사용 · 실제 walnuts 결함 픽셀 사례 · CQ99 적용 전 최종 맵')
figure(s,PAPER/'figures/fig_scopead_overview.png',1.90,bottom=6.62)

s=frame(8,'AD2 public full8 · 3 seeds · AUPRO@0.05 · CQ99 포함 · 오른쪽은 원고 음의 손실값의 절댓값')
text_box(s,.7,1.88,5.8,.4,'변화 조건 AUPRO (%)',17,NAVY,BOLD,True)
text_box(s,6.95,1.88,5.65,.4,'공통 임계값 추가 손실 크기 (pp)',17,NAVY,BOLD,True)
cats=['PatchCore','AnomalyDINO','SubspaceAD','SuperAD','ScopeAD']
chart(s,cats,[('2-shot',[42.30,39.52,49.65,49.96,56.60]),('16-shot',[52.97,44.12,52.37,56.33,60.74])],.7,2.35,6.0,3.55,70,horizontal=True)
chart(s,cats,[('2-shot',[.16,1.83,5.28,.33,.05]),('16-shot',[.46,2.74,6.52,.80,.27])],6.95,2.35,5.65,3.55,8,horizontal=True)
bullets(s,['해석 범위: 임계값 sweep 기반의 조건 간 일관성 평가이며, 배포 고정 임계값의 보장과 구분'],y=6.07)

s=frame(9,'원고 기재값 · full8 · seed 0 · CQ off 사후 진단 · 정상 픽셀 부분표집 · 원 JSON/평가 일치 확인 필요')
text_box(s,.7,1.85,11.9,.4,'Defect Recall @ FPR 1% (%)',17,NAVY,BOLD,True)
chart(s,['Raw L23\n1120','Raw 계층 결합\n1120','Q95 계층 결합\n1120','전체 ScopeAD'],
      [('2-shot',[36.24,39.65,38.51,45.95]),('16-shot',[39.78,43.01,42.71,50.04])],.9,2.28,11.5,3.45,60)
bullets(s,['단일 깊은 층 대비 +9.71 / +10.26 pp: 전체 관측 확장에 따른 저-FPR 검출률 개선',
           'Q95 단독 효과와의 구분 및 원시 결과·공식 평가 프로토콜의 추가 확인'],y=5.85)

s=frame(10,'full8 · 3 seeds · CQ off · 1120 정상 상위 5% · |log score gap| ≤ 0.02 매칭 · 상세 수치·CI는 발표 노트')
text_box(s,.7,1.85,11.94,.4,'Nonregular 조건의 정상–결함 쌍 concordance (%)',17,NAVY,BOLD,True)
chart(s,['1120 단독','4-scale 평균','4-scale Rényi'],
      [('2-shot',[82.79,71.64,72.09]),('16-shot',[82.22,69.73,69.98])],1.3,2.28,10.6,3.35,100)
bullets(s,['진단 결과: 단일 관측 대비 순위 회복 미확인, low-FPR recall과 다른 모집단·지표',
           '선택 한계: 기준 맵으로 뽑은 조건부 쌍에 해당하며, 전체 HSA 악화의 단독 근거로 사용 불가'],y=5.77)

s=frame(11,'Mechanism 10/20 · Axis analysis 10/21 · Appendix · 분석별 설정 차이와 클래스 CI의 0 포함에 유의')
blocks(s,[('척도 보정과 집계의 상보성','Q95 × Rényi 상호작용 +0.544 / +0.690 pp, 클래스 신뢰구간의 0 포함'),
          ('평균 이득의 클래스 의존성','해상도별 보정 +0.551 / +0.822 pp, fabric 제외 시 −0.110 / −0.098 pp'),
          ('강한 반응 보존의 기여','Rank-only 효과보다 큰 size-only 효과와의 연결 및 순위 자체의 우월성 해석 제한'),
          ('특정 가중치의 최적성 한계','선형 순위 가중과 <0.1 pp 차이, 모든 shot의 최적값으로 보기 어려운 λ=0.125')])

s=frame(12,'sec/1_intro.tex · sec/2_related_work.tex · sec/3_method.tex · sec/4_experiments.tex · 논의·결론·부록')
blocks(s,[('서론·관련연구 · 문제와 목표의 명료화','HSA의 원인과 저-FPR 손실 연결, AD 2 선정 이유 및 조명·획득 변화 범위 명시'),
          ('방법·그림 · 하나의 통합 구조','정상-only 관측 확장·척도 정렬·계층 결합·순위 집계의 역할 연결'),
          ('실험·부록 · 주장별 근거의 분리','조건별·private 평가, 저-FPR recall, 기전·민감도·직접 HSA 진단 보강'),
          ('논의·결론 · 궁극적 목표와 한계','Few-shot·조건 변화의 강건한 국소화라는 목표 정리와 HSA 표현의 정합화 필요')])

s=frame(13)
blocks(s,[('서술 반영 · 메모 4·1·22·5','과제 정의, AD 2 선정 근거, 설계 논리, 서론 마무리에 대한 설명 보강'),
          ('성능 근거 · 메모 2·22','조건 변화 AUPRO와 작은 임계값 손실, 사후 저-FPR 검출률 개선'),
          ('남은 검증 · 메모 2의 직접 HSA 근거','유사 고점수 쌍의 순위 회복 미확인 및 진단별 지지 범위의 구분')],step=1.30)

s=frame(14)
blocks(s,[('HSA 주장 범위의 정리','조건 변화 성능, 저-FPR recall, 고점수 쌍 순위 회복을 구분한 문구 점검'),
          ('방법 설명과 CQ 조건의 일치','현재 CQ 서술 제거 상태와 CQ99 포함 메인 결과 사이의 정합성 확인'),
          ('HSA 진단의 재현성 확인','Seed 0 원 JSON, 정상 픽셀 표집, 집계 순서, 공식 평가와의 일치 확인'),
          ('Private 결과의 출처 연결','원고의 10/24 DONE 기록과 공식 결과 파일·기존 제출 로그의 연결')])

s=frame(15,'10/19 발표자료와 현재 작업본의 비교 · 기존 성능을 이번 기간의 신규 성능 상승으로 해석하지 않도록 구분')
text_box(s,.70,2.05,5.55,.45,'10월 19일 기존 성과',20,NAVY,BOLD,True)
bullets(s,['Q95·비대칭 계층 결합·Rényi 집계',
           '48개 bank와 CQ99, CQ on/off·비용 분석',
           'AD2 1-shot 52.18%, 16-shot 60.74%',
           '기존 상호작용 분석과 후속 질문'],y=2.73,x=.70,w=5.6,hang=285750,name='Previous')
text_box(s,7.00,2.05,5.55,.45,'이후 보강 내용',20,NAVY,BOLD,True)
bullets(s,['Few-shot·획득 변화와 설계 논리의 연결',
           '10/20 기전·10/21 축별 보정 분석',
           '10/24 고점수 쌍·원고 recall 진단',
           '실제 사례 그림과 주장 범위 점검'],y=2.73,x=7.00,w=5.6,hang=285750,name='Added')
log[-1]['content'].append('기존 성과와 이후 보강 내용의 두 열 설명, 상세 원 수치는 노트 보존')

s=frame(16,'AUPRO@0.05 (%) · 2/16-shot · 사후 재분석 · 상세 출처와 클래스별 해석 제한은 발표 노트')
r=rows[15]
table(s,.70,2.12,[3.1,2.2,2.2,2.22,2.22],r['headers'],r['rows'],row_h=.52,size=12)
bullets(s,['Fabric 제외 시 해상도별 보정의 이득 반전: 평균 결과의 클래스 의존성'],y=6.02)

s=frame(17,'현재 Method·Experiments · 기존 Private 제출 로그 · 진단 스크립트 · 공식 서버의 실시간 조회 결과와 구분')
blocks(s,[('CQ 적용 조건의 차이','Method의 CQ 설명 제거와 component_quantile 포함 메인 평가 설정의 불일치'),
          ('Private 기록과 지표 범위','원고 10/24 DONE, AUPRO 66.53/64.40%; Seg-F1은 SuperAD보다 낮은 결과'),
          ('결과 출처의 추가 확인','과거 SUBMITTING 로그의 갱신, 공식 결과 파일 연결, HSA recall 원 JSON 확보')],step=1.30)

s=frame(18,'사용자 제공 main.pdf 검토 메모 22·5·2·1·4 · 원문 메모와 상세 설명은 발표 노트에 보존')
r=rows[17]
table(s,.70,2.05,[1.7,6.,4.24],r['headers'],r['rows'],row_h=.83,size=12)

fix_table_ids(prs)
set_sections(prs,[('01. 피드백과 수정 방향',list(prs.slides)[:3]),('02. 문제와 방법의 서술 보강',list(prs.slides)[3:7]),('03. 추가 근거와 주장 범위',list(prs.slides)[7:11]),('04. 대응 현황과 다음 확인',list(prs.slides)[11:14]),('Appendix A. 비교 기준과 근거',list(prs.slides)[14:])])
name='ScopeAD_Feedback_Response_20261026_v2'
prs.save(HERE/(name+'.pptx'))
(HERE/'slides.json').write_text(json.dumps(log,ensure_ascii=False,indent=2))
md=['# 10월 19일 피드백 대응 보고 — v2','', '제목·리드 명사형 통일. 본문 표 제거, 정량 비교의 편집 가능한 차트 전환. 상세 표 2개는 부록 유지. 본문 14장 + 부록 4장.','']
for r in log:
    md += ['## '+r['page']+'. '+r['title'],'',r['lead'],'']
    md += ['- '+(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False)) for x in r['content']]
    md += ['','**출처:** '+r['source'],'','**발표 노트:**\n'+r['notes'],'']
(HERE/(name+'.md')).write_text('\n'.join(md))
checks={'slides':len(prs.slides),'tables':sum(sh.has_table for s in prs.slides for sh in s.shapes),'charts':sum(sh.has_chart for s in prs.slides for sh in s.shapes),'titles_with_sentence_endings':[t for t in TITLES+LEADS if t.endswith(('다','다.'))]}
assert checks['tables']==2 and checks['charts']==4 and not checks['titles_with_sentence_endings']
(HERE/'structure_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
print(checks)
