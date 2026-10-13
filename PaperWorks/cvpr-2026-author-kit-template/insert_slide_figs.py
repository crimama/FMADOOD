# 1. Update sec/1_intro.tex
with open('PaperWorks/cvpr-2026-author-kit-template/sec/1_intro.tex', 'r', encoding='utf-8') as f:
    intro = f.read()

fig_intro = r"""
egin{figure}[t]
centering
includegraphics[width=linewidth]{figures/fig_intro_ambiguity.png}
caption{	extbf{Illustration of High-Score Ambiguity under few-shot industrial inspection.} A large nearest-neighbor distance (high anomaly score) can stem from either an actual defect or an unobserved normal variation (e.g., lighting drift, pose shift, or surface specular reflection) absent from the sparse normal support bank. Consequently, raw distance alone cannot definitively distinguish defects from benign environmental changes, inflating low-FPR false alarms.}
label{fig:intro-ambiguity}
end{figure}
"""

target_intro = r"subsection{Observation Expansion과 결합의 과제}"
if 'fig_intro_ambiguity.png' not in intro:
    intro = intro.replace(target_intro, fig_intro + "

" + target_intro)

with open('PaperWorks/cvpr-2026-author-kit-template/sec/1_intro.tex', 'w', encoding='utf-8') as f:
    f.write(intro)

# 2. Update sec/3_method.tex
with open('PaperWorks/cvpr-2026-author-kit-template/sec/3_method.tex', 'r', encoding='utf-8') as f:
    method = f.read()

fig_scale = r"""
egin{figure}[t]
centering
includegraphics[width=linewidth]{figures/fig_scale_calibration_lso.png}
caption{	extbf{Observation-wise distance scale discrepancy and quantile alignment.} Left: Raw Leave-Source-Out (LSO) normal distance distributions across different backbone layers (Layers 5, 11, 23) and resolutions exhibit substantial scale disparities. Right: By dividing each component distance by its empirical upper quantile ($c_m = Q_{0.95}(D_m^{	ext{LSO}})$), the critical threshold $z=1.0$ is unified across all 48 component banks into a dimensionless, comparable coordinate before fusion.}
label{fig:scale-calibration-lso}
end{figure}
"""

# Replace the previous fig_rsg4_mech_scale or add it
if 'fig_rsg4_mech_scale.pdf' in method:
    # replace fig_rsg4_mech_scale with the slide figure
    import re
    method = re.sub(
        r'\begin{figure}[t].*?fig_rsg4_mech_scale.pdf.*?\end{figure}',
        fig_scale.strip().replace('\', '\\'),
        method,
        flags=re.DOTALL
    )
elif 'fig_scale_calibration_lso.png' not in method:
    target_method = r"subsection{Contribution Control: 고정 가중 Shallow 결합}"
    method = method.replace(target_method, fig_scale + "

" + target_method)

with open('PaperWorks/cvpr-2026-author-kit-template/sec/3_method.tex', 'w', encoding='utf-8') as f:
    f.write(method)

print("Inserted Slide 3 figure and Slide 6 figure with English captions")
