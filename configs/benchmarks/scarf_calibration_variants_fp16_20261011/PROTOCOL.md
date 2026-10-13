# SCARF calibration variants (AD2 development, seed 0)

Motivation: the support q95 scales of the 12 (layer x resolution) components differ by only 1.8x on average (CV 0.18 AD2; 1.4x for deep layers), so q95 behaves as a mild reweighting; q95 matched no normalization on AD2, MVTec AD and VisA. Question: does a different calibration shape or unit improve the Full method (linear0125)?

Variants (only method_config changes relative to ad2_A_linear0125_tailscale): tail095 = score_normalization tail (support-CDF survival transform with an exponential tail beyond the q95 point); q050/q080/q090/q099/q0999 = tailscale with tail_quantile changed; localscale = tailscale plus local_reliability scale (k 8, shrinkage 8): the divisor is a per-query-token local estimate of the bank's normal looseness, computed from support only. Lambda 0.125 is frozen; its interplay with the new units is part of what each variant measures (a variant that changes units also changes the effective shallow weight).

Scope: AD2 public full8, shots 2 and 16, seed 0, FP16, 4 resolutions, cap256, same support manifests. 14 configs x 8 classes = 112 class runs. Baseline: ad2_A_linear0125_tailscale_s02/s16 (59.33/76.48, 59.57/77.46 AUPRO/pAUROC) from the calibration factorial under the same code.

Decision rule (fixed before results): a variant is a candidate when AUPRO is at least +0.5 pp above the q95 baseline at both shots and pAUROC is not lower by more than 0.3 pp at either shot. A candidate must then be confirmed on seeds 1-2 and on MVTec AD and VisA before adoption. These are post-hoc development comparisons on a seen population, not final evidence. Support data only; no labels in fitting; maps and weights off.

