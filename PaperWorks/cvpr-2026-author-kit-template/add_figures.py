# 1. Update sec/3_method.tex
with open('PaperWorks/cvpr-2026-author-kit-template/sec/3_method.tex', 'r', encoding='utf-8') as f:
    method = f.read()

# Add Fig M1 after 3.2 bank
fig_m1 = r"""
egin{figure}[t]
centering
includegraphics[width=linewidth]{figures/fig_rsg4_mech_bagging.pdf}
caption{	extbf{4-Bag 점수 맵 앙상블 및 분산 완화.} 실제 hazelnut 쿼리(1120px)의 4개 독립 Bag 점수 맵과 평균. 각 Bag은 이미지당 256개 토큰을 무작위 추출하여 뱅크를 구성한다. 4-Bag 평균은 표본화 분산을 효과적으로 완화한다.}
label{fig:mech-bagging}
end{figure}
"""

# Add Fig M2 after 3.3 scale
fig_m2 = r"""
egin{figure}[t]
centering
includegraphics[width=linewidth]{figures/fig_rsg4_mech_scale.pdf}
caption{	extbf{Scale Calibration을 통한 무차원 점수 정렬.} 상단: 레이어별 원시 거리 분포 및 기준 상위 분위수($q=0.95$) 단위 척도 $c_m$. 동일 원시 거리라도 레이어마다 정상 변동 척도가 다르다. 하단: 각 점수를 해당 컴포넌트의 $c_m$으로 나눈 정규화 맵($z_m = d_m / c_m$)으로 $z=1.0$을 공통 통계 기준으로 정렬한다.}
label{fig:mech-scale}
end{figure}
"""

if 'fig_rsg4_mech_bagging.pdf' not in method:
    method = method.replace(r'subsection{Scale Calibration: 분위수 기반 거리 정렬}', fig_m1 + '

' + r'subsection{Scale Calibration: 분위수 기반 거리 정렬}')
    method = method.replace(r'subsection{Contribution Control: 고정 가중 Shallow 결합}', fig_m2 + '

' + r'subsection{Contribution Control: 고정 가중 Shallow 결합}')

with open('PaperWorks/cvpr-2026-author-kit-template/sec/3_method.tex', 'w', encoding='utf-8') as f:
    f.write(method)

# 2. Update sec/4_experiments.tex
with open('PaperWorks/cvpr-2026-author-kit-template/sec/4_experiments.tex', 'r', encoding='utf-8') as f:
    exp = f.read()

fig_2 = r"""
egin{figure}[t]
centering
includegraphics[width=linewidth]{figures/fig_acquisition_shift_by_shot.pdf}
caption{	extbf{Support shot별 native acquisition strata의 AUPRO@0.05.} 패널: Regular, Shift 1, Shift 2, Shift 3. ScopeAD는 획득 조건 변화(조명 드리프트, 자세 변동 등) 하에서도 기존 방법 대비 안정적인 국소화 성능을 유지한다.}
label{fig:acquisition-shift}
end{figure}
"""

if 'fig_acquisition_shift_by_shot.pdf' not in exp:
    exp = exp.replace(r'egin{table}[t]' + '
' + r'centeringscriptsizesetlength{	abcolsep}{2pt}' + '
' + r'caption{MVTec AD 2 공식 private 서버 평가', fig_2 + '

' + r'egin{table}[t]' + '
' + r'centeringscriptsizesetlength{	abcolsep}{2pt}' + '
' + r'caption{MVTec AD 2 공식 private 서버 평가')

with open('PaperWorks/cvpr-2026-author-kit-template/sec/4_experiments.tex', 'w', encoding='utf-8') as f:
    f.write(exp)

print("Added Figure M1, Figure M2, and Figure 2 into sec/*.tex")
