# 1. Update 0_abstract.tex
abstract_text = r"""egin{abstract}
소수 정상 이미지만을 사용하는 few-shot 산업 이상 국소화에서는 촬영 조건 변화(조명 드리프트, 자세 변동 등)로 인해 정상 영역에서도 큰 거리가 나타나는 	extbf{High-Score Ambiguity}가 발생한다.
추가 정상 표본이 없는 제약 아래 미세 결함 단서를 확보하기 위해 다해상도와 다계층으로 	extbf{Observation Expansion}을 수행하면, 관측 조건마다 거리 척도가 달라지고 얕은 층의 정상 질감 반응이 점수에 섞여 low-FPR 오경보가 급증한다.
이를 해결하기 위해 본 논문은 	extbf{method}를 제안한다.
method는 관측 조건마다 독립 뱅크를 구축한 뒤, 정상 서포트의 기준 상위 분위수로 각 관측의 거리 단위를 공통 척도로 	extbf{Scale Calibration}한다.
이어 Deep 의미 기준선 위에 Shallow 층의 정상 질감 과민반응을 억제하도록 작은 고정 가중치로 	extbf{Contribution Control}(선형 가산)을 적용하고 다해상도 맵을 평균한다.
MVTec AD 2 16-shot에서 method는 3개 시드 평균 	extbf{60.06,% AUPRO}를 기록하여 SuperAD(55.89,%) 대비 	extbf{+4.17,pp}의 성능 도약을 달성했으며, 단일 뱅크 대비 +2.09,pp, 단일 저해상도 입력 대비 +7.56,pp의 뚜렷한 구성 및 결합 이득을 실증했다.
end{abstract}
"""

with open('PaperWorks/cvpr-2026-author-kit-template/sec/0_abstract.tex', 'w', encoding='utf-8') as f:
    f.write(abstract_text)

# 2. Update 1_intro.tex
with open('PaperWorks/cvpr-2026-author-kit-template/sec/1_intro.tex', 'r', encoding='utf-8') as f:
    intro = f.read()

intro = intro.replace(r'subsection{소수 정상 참조와 거리 해석의 모호성}', r'subsection{소수 정상 참조와 High-Score Ambiguity}')
intro = intro.replace(r'거리 해석의 모호성(High-Score Ambiguity)', r'High-Score Ambiguity')
intro = intro.replace(r'"큰 거리값"이라는 관측만으로는 그 원인이 실제 결함인지 서포트에 미포함된 정상 외관 변화인지 확정할 수 없다(High-Score Ambiguity)', r'큰 거리값이라는 관측만으로는 그 원인이 실제 결함인지 서포트에 미포함된 정상 외관 변화인지 확정할 수 없다(	extbf{High-Score Ambiguity})')
intro = intro.replace(r'subsection{고정 다층·다해상도 관측 결합의 과제}', r'subsection{Observation Expansion과 결합의 과제}')
intro = intro.replace(r'item 	extbf{Scale Calibration (거리 단위 정렬)}:', r'item 	extbf{Scale Calibration}:')
intro = intro.replace(r'item 	extbf{Contribution Control (Shallow 기여 제어)}:', r'item 	extbf{Contribution Control}:')
intro = intro.replace(r'item 	extbf{소수 표본 제약을 돌파하는 다차원 관측 확장}:', r'item 	extbf{Observation Expansion을 통한 표본 제약 극복}:')
intro = intro.replace(r'item 	extbf{관측별 뱅크 구축 및 척도 보정과 기여 제어의 분리}:', r'item 	extbf{Scale Calibration과 Contribution Control의 분리}:')

with open('PaperWorks/cvpr-2026-author-kit-template/sec/1_intro.tex', 'w', encoding='utf-8') as f:
    f.write(intro)

# 3. Update 3_method.tex
with open('PaperWorks/cvpr-2026-author-kit-template/sec/3_method.tex', 'r', encoding='utf-8') as f:
    method = f.read()

method = method.replace(r'subsection{Scale Calibration: 분위수 기반 거리 정규화}', r'subsection{Scale Calibration: 분위수 기반 거리 정렬}')
method = method.replace(r'subsection{Contribution Control: 고정 가중 Shallow 선형 가산}', r'subsection{Contribution Control: 고정 가중 Shallow 결합}')

with open('PaperWorks/cvpr-2026-author-kit-template/sec/3_method.tex', 'w', encoding='utf-8') as f:
    f.write(method)

# 4. Update 5_discussion.tex
with open('PaperWorks/cvpr-2026-author-kit-template/sec/5_discussion.tex', 'r', encoding='utf-8') as f:
    disc = f.read()

disc = disc.replace(r'
oindent	extbf{관측 확장과 관측별 뱅크 결합의 시너지.}', r'
oindent	extbf{Observation Expansion과 관측별 뱅크 결합의 시너지.}')
disc = disc.replace(r'
oindent	extbf{Shallow 가산과 정상 질감 제어.}', r'
oindent	extbf{Contribution Control을 통한 Shallow 질감 제어.}')

with open('PaperWorks/cvpr-2026-author-kit-template/sec/5_discussion.tex', 'w', encoding='utf-8') as f:
    f.write(disc)

print("Updated technical terms directly into English in sec/*.tex")
