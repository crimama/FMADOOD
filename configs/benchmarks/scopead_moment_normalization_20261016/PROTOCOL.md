# Moment-based normal-reference calibration
2026-10-16 post-hoc development comparison requested by user.
Hypothesis: estimating an observation scale from all normal source-LOO distances
may be more stable than a single Q95. Compare mean, RMS, standard deviation,
and centered Z-score against a fresh Q95 control.
Each observation is resolution x bag x layer; estimates use normal support only.
mean_scale: d / mean(D); rms_scale: d / sqrt(mean(D^2));
std_scale: d / std(D); zscore: (d - mean(D)) / std(D).
Variance uses ddof=0; all denominators use the existing 1e-4 numerical floor,
with floor activation recorded. No clipping, probability/FPR guarantee or
independent-token assumption. Source-LOO tokens are correlated.
Mean is the exponential scale MLE and RMS the half-normal scale MLE only under
those respective distribution assumptions; no distributional fit is asserted.
Z-score versus std_scale isolates centering; affine offsets may cancel in
linear fusion before postprocessing, so paired equivalence is an expected audit.
Common pipeline: AD2 test_public seen development, seed0, VOID support2/16,
DINOv2-L FP16, layers5/11/23, resolutions448/672/896/1120, 4 bags, cap256,
linear0125 via ablation_shallow_mode=all, CQoff, existing noise guard,
float16 streaming metric contract. No backbone adaptation.
Primary AUPRO@0.05; secondary pAUROC@0.05; higher is better.
Budget/stop: 5 arms x 2 shots x 8 classes =80 evaluations; 10800 sec per cell.
Four assigned existing dsba3 GPU UUIDs, existing hun_fsad_tta_012 container.
Concurrent PatchCore work is retained. Source release is staged immutably,
active checkout preserved. No dense maps or fitted weights.
Report all arms including failures, normal-only fitted moments, floor counts,
and paired support/query/GT/bank hashes before interpreting deltas.
Use existing experiments.py collector for canonical Markdown+CSV registration.
