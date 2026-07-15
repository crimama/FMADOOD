# MVTec AD1 density0 + guided-r8 hybrid 1/2/4/8-shot report

## 1. Negative Evidence Intake

The preceding 4-shot component study established two facts. Removing the
density term improved the five macro metrics, while applying guided-r8 to the
only output map improved localization but narrowly failed the image-AUROC
retention gate. The present run therefore tests the preregistered two-output
contract rather than retuning either component.

## 2. Motivation

The hypothesis is that image ranking and pixel boundary refinement have
different optimal score fields: retain the unfiltered density0 map for the
top-1% image score and use its label-free guided-r8 refinement only for pixel
metrics. Robustness is tested at 1, 2, 4, and 8 first-support shots.

## 3. Implementable Design

All non-target settings are fixed: frozen `dinov2_vitb14_reg`, layers
`[2,5,8,11]`, 448 input, layer-normalized mean fusion, three NF epochs, two
couplings, hidden multiplier one, and a static latent 1-NN bank. Density is
zero, DVT/TTE/context/register/morphology are absent, and guided filtering uses
grayscale guidance, half-native resolution, radius 8, and epsilon 0.01.

The hybrid metric record copies image-level fields from the raw evaluation and
pixel-level fields from the guided evaluation. Ground truth is used only by the
evaluator, never by filtering or score construction.

## 4. Evaluation Alignment

The target is the full classic MVTec AD test split over all 15 classes. Each
shot is compared with its same-shot density-0.25 static control. The requested
metrics are unweighted class macro means: i-AUROC, i-AUPRC, p-AUROC, p-AUPRC,
and p-AUPRO. Exact gates and controls were sealed in `preregistration.md` before
remote execution. Strict external VisionAD/SuperAD parity remains unavailable,
so the external status stays `BLOCKED_BASELINE`.

## 5. Code Modification / Creation

- `scripts/combine_mvtecad1_hybrid_metrics.py` creates an explicit, auditable
  two-source metric record without recomputing scores.
- `scripts/run_mvtecad1_density0_guided_hybrid_shot_sweep_remote.sh` assigns
  shots 1/4 to dsba5 GPU 0 and 2/8 to GPU 1, then removes dense maps.
- `tests/test_mvtecad1_hybrid_metrics.py` locks field ownership and input
  immutability.

## 6. Added Code Evaluation

Before launch, Python compilation, shell syntax checking, the 19 focused tests,
and `git diff --check` passed. Post-run verification again checks the focused
tests and the full local test suite. The combiner audit confirms exact equality
of every macro and per-class image field with raw metrics and every macro and
per-class pixel field with guided metrics.

## 7. Remote Execution and Audit

The four full evaluations completed in the existing dsba5 container on GPUs 0
and 1 with no traceback, CUDA error, or OOM. The prior 4-shot density0 artifact
and the new raw artifact have numeric maximum difference `0.0` over all 193
numeric metric fields. Every class uses exact ordered first-N supports. Initial
and final memories are identical at 1024, 2048, 4096, and 8192 patches for
shots 1, 2, 4, and 8 respectively. No `.npy` or `.npz` dense-map artifact
remains after cleanup.

## 8. Unified Metrics and Analysis

Values below are percentages; parenthesized values are percentage-point deltas
from the same-shot static density-0.25 control.

| Shot | i-AUROC | i-AUPRC | p-AUROC | p-AUPRC | p-AUPRO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 95.8838 (+0.2871) | 98.0134 (+0.1220) | 97.2334 (+0.4412) | 59.9343 (+4.7777) | 93.1167 (+0.4804) |
| 2 | 97.1562 (+0.1853) | 98.5743 (+0.0785) | 97.6024 (+0.4014) | 61.4666 (+4.4199) | 93.9942 (+0.4791) |
| 4 | 97.3680 (+0.1579) | 98.7876 (+0.0444) | 97.7426 (+0.3679) | 62.4378 (+4.3217) | 94.2379 (+0.4219) |
| 8 | 98.2304 (+0.0553) | 99.0896 (-0.0266) | 97.8892 (+0.3437) | 63.0672 (+4.0011) | 94.6499 (+0.4094) |

All four shots pass macro retention and the positive diagnostic gate. Pixel
AUPRC rises on every class at shots 1 and 2, and on 14/15 classes at shots 4
and 8; the only latter loss is zipper (`-0.399` and `-0.252` point). Pixel
AUROC improves on 14/15 classes at every shot. The worst requested-metric
class loss is transistor i-AUPRC at 1 shot (`-1.131` points), well inside the
5-point catastrophic threshold. All five absolute metrics improve
monotonically as shots increase.

## 9. Continuation Assessment and Conclusion

The robust continuation criterion passes at 4/4 shots, exceeding the required
3/4, with no catastrophic class loss. Separating image scoring from pixel
refinement resolves the earlier guided-r8 image-retention failure and preserves
the density0 image-ranking benefit while adding a consistent localization
gain. The internal verdict is `PROMISING_DIAGNOSTIC`; this is the preferred AD1
variant for the tested shot range. It is not yet a strict SOTA claim because
the external evaluator/training parity requirement remains
`BLOCKED_BASELINE`.

## Post-Conclusion Storage Cleanup

Source and refined dense maps were deleted on the remote host immediately after
metric generation. The compact local pullback contains only metrics,
manifests, logs, cleanup evidence, and a completion marker.
