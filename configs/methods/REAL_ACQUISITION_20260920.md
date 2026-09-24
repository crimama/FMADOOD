# Actual acquisition-condition diagnosis — 2026-09-20

Preregistered before new outcome evaluation. User authorized continuing the
real-shift audit/experiment after the synthetic framing gates failed.

## Data provenance

Official dataset paper III-B/C/E confirms public images and GT for all lighting
conditions; exposure time and extra lights were physically varied while scenes
were held fixed. GT can differ by lighting because defect visibility changes.
https://arxiv.org/html/2503.21622v1 (accessed2026-09-20).
Local README names this paper. Directory audit confirms regular,underexposed,
overexposed,shift_1 for every class; shift_2/3 absent for fruit_jelly; shift_4
only for vial. Good counts per condition: can12,fabric11,fruit_jelly5,rice7,
sheet_metal4,vial5,wallplugs10,walnuts10; bad15 everywhere. GT files match bad
counts. Parser and runtime check require matching (good/bad,scene_id) sets across
conditions. Per-condition GT is used, not the regular mask copied across scenes.
Public is already seen development, NOT a held-out final benchmark. Private
images have no local GT and are excluded; no server submissions are authorized.

## Fixed design and budget

32 full class evaluations =8classes ×1/16shot ×QR0/QR1. Same source seed0,
nested SD_support_1/16 manifests, frozen DINOv2-L layers11/23,448/672 scales,
one256-token/source bank, MAD normalization. QR1 changes only bounded global
query-reference translation. Do not add methods, tune severity, or modify banks.
Retain original full-public metrics for parity against the completed SD suite.

New evaluation-only mode `real_acquisition` scores ALL public images with no
synthetic transforms, grouped by acquisition token only AFTER prediction.
As before, threshold is each method's clean-normal q95, from16 fixed training
normals excluded from the entire16-support pool; sample up to4096 pixels/image.
Additional calibration normals are declared and never used to fit the method.
Keep threshold fixed across acquisition conditions. Save scalar per-image
normal FPR, defect TPR, score means, scene ID and condition, plus condition-wise
streaming AUPRO/pAUROC. No maps or weights. Condition accumulators are evaluator
state only. No labels or condition tokens are passed as fitting parameters.

Calibration sensitivity: recompute threshold leaving out each of16 calibration
images. Store16 thresholds and each query's FPR/TPR at their min/max. This is
a jackknife sensitivity range, NOT a confidence interval. Avoid independent-
pixel significance claims; effective public sample size is4–12 normal scenes
and15 anomalous scenes per class/condition.

## Analysis and stop rules

Primary condition `shift_1` exists in every class; comparison is against regular
within matching scene IDs, equal-image then equal-class means. Exposure changes
and shift_2/3/4 are secondary descriptive endpoints, with explicit missing cells.
Do not select the worst condition after seeing results.

H0/problem gate: QR0 at1shot shift_1 raises normal FPR by >=1pp macro and in
>=5/8 classes. H1/few-shot interaction: shift-minus-regular FPR increase is
greater at1 than16shot in>=5/8 classes and positive macro. H2/correction tradeoff:
QR1 reduces shift_1 FPR and defect TPR jointly in>=4/8 classes at1shot, and both
macro deltas are negative. Report condition ranking metrics separately: scalar
threshold changes are not evidence of destroyed representation. TPR condition
differences also reflect actual defect visibility/annotation differences.
Only if all gates pass is the proposed framing supported for this development
setting; no new method is automatically launched. Otherwise revise the framing.
If threshold sensitivity changes a gate conclusion, report it as fragile.

Existing local/dsba3 containers and assigned GPUs, same class ownership, generic
run/collect helpers. No push, provisioning, private GT fabrication, or artifact
cleanup. Runtime failures trigger diagnosis, not a negative research conclusion.
