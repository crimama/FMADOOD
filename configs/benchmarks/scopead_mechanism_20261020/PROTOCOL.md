# Q95 × Rényi mechanism analysis — 2026-10-20

## Hypotheses and revision rules

Post-hoc mechanism study on the already-seen AD2 public development population.
No new final method is selected and this is not an untouched test evaluation.

1. Q95 reduces resolution-dependent rank dominance. Compare paired sampled
   native-pixel rank occupancy for normal/defect populations, all three seeds.
   If dominance is not reduced consistently, do not claim general rank alignment.
2. Calibration interacts with aggregation. Evaluate none/Q95 × mean,
   0.5 mean + 0.5 Rényi, Rényi, max. Report the interaction
   (Q95 Rényi - Q95 mean) - (none Rényi - none mean), all eight classes and
   seed mean ± sample SD. Max is a diagnostic endpoint.
3. Calibration uncertainty can affect rank-weighted aggregation. Inject
   mean-one log-normal factors at sigma 0.1/0.2/0.3 into each bag/layer/resolution
   denominator; three paired draws fixed across a class's queries.
   Compare loss relative to unperturbed mean and Rényi, including seed 2.
   Synthetic sensitivity does not estimate actual support Q95 uncertainty.
   Do not attribute actual seed failures to estimation error from this alone.

## Matched conditions and budget

- AD2 test_public full8, shots 2/16, seeds 0/1/2: 48 class captures.
- Fixed manifests copied exactly from corresponding final runs: seed 0 VOID;
  seeds 1/2 E1. No new support sampling.
- FP16 DINOv2 ViT-L/14, [5,11,23], cap256, 4 bags, four resolutions,
  deep mean + 0.125 shallow, Q95, existing noise guard, CQ99 unchanged.
- Common pipeline owns fitting, evaluation, manifests and metrics.
  Primary AUPRO@0.05 (higher is better); secondary pAUROC@0.05. Also report
  TPR/PRO at FPR .001/.005/.01/.02/.05 from native-resolution histograms.
- 26 paired replays per class: 8 aggregation/calibration and 18 noise arms.
  All reuse captured nearest-neighbor distances.
- dsba3 existing hun_fsad_tta_012, GPUs 0–3 by UUID. No local GPU run.
- Per-cell timeout 21,600 s. Stop a cell on nonfinite values/replay mismatch.
  Runtime failure is not evidence against a method.

## Capture and validity

Capture raw and Q95-calibrated token grids per resolution/bag/layer in original
precision. Keep these optional analysis artifacts remotely. Replay at native
image resolution without a new backbone/nearest-neighbor pass.
Final Q95+Rényi+CQ replay must match every prediction within max abs 2e-5 and
class AUPRO within 3e-5 (fraction units). Otherwise invalidate the analysis.
Preserve each query's actual noise-guard resolution route.

Labels enter the common evaluator after prediction and this analysis after
the entire common run, never fitting/prediction. Per-layer summaries use feature
grids and nearest-sampled GT; tiny regions can disappear. Native-resolution
metrics and rank samples do not have that approximation.
Save configs, support/query manifests, logs, metrics, analysis JSON and RESULTS.md.

Actual held-out-normal estimator error, single-bank native AUPRO, oracle-axis
reanalysis and defect-size PRO are follow-up analyses, not completed here.
