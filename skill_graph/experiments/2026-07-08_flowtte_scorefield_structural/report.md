# FlowTTE Score-Field Structural Diagnostics

Date: 2026-07-08
Verdict: KILL_FOR_CLAIM / NO_CONTINUE

## Negative Evidence Intake

This branch is not a class-specific hyperparameter sweep. It tests whether a
single all-object score-field rule can reduce position/background driven false
positive fragmentation in the current DINOv3-H+ DVT Flow-LatentBank ranker.

Prior evidence already killed broad threshold-only, morphology-only, and
NF-removal explanations for the SuperADD F1 gap. This diagnostic therefore
keeps the backbone/support/NF/no-TTE setup fixed and changes only the score
field before thresholding.

Likely failure basin: post-hoc score shaping that improves one weak object but
harms strong categories or is matched by morphology. It is diagnostic only,
not a strict method-claim branch.

## Motivation

The current best FlowTTE branch is a strong continuous patch ranker:
`dinov3_vith16plus`, layers `[7,15,23,31]`, DVT position mean denoise
`alpha=1.0`, fixed 16-shot support, no-TTE, latent NN scoring.

Recorded reference metrics:

- H+ DVT raw FlowTTE: `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`
- H+ DVT with close/fill/erode morphology: `seg_F1=0.542316`
- reported SuperADD context: `seg_AUROC_0.05=0.839300`, `seg_F1=0.626113`

The gap is now mainly F1/spatial-field quality rather than AUROC ranking.

## Implementable Design

Run all eight public MVTec AD2 objects with the same H+ DVT no-TTE reference
and five class-agnostic variants:

- `baseline`: no score-field calibration
- `support_position_center`: subtract support leave-one-out positional mean
- `support_position_zscore`: z-score by support positional mean/std
- `foreground_energy`: suppress background-like patches by support feature
  energy prior
- `center_plus_foreground`: combine position centering and foreground prior

Dense anomaly maps are removed after fragmentation analysis.

## Evaluation Alignment

Target dataset: MVTec AD2 single-image.
Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
Objects: `can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal`.
Split: full `test_public/good,bad`.
Primary metrics: `seg_AUROC_0.05`, `seg_F1`; secondary diagnostics:
high-score component count, positive area, largest-component share, predicted
GT overlap, GT recall.

SuperAD/SuperADD values are context baselines unless same-condition artifacts
are available; no `KEEP` verdict is allowed from this diagnostic alone.

## Code Modification / Creation

Implemented before remote execution:

- `scripts/flow_tte_score_field.py`: support position calibration and support
  feature-energy foreground prior
- `scripts/flow_tte_components.py`: high-score connected-component summaries
- `scripts/flow_tte_score_field_analysis.py`: all-object fragmentation report
- `scripts/run_flow_tte_score_field_remote.sh`: all8 score-field variant runner

## Added Code Evaluation

Local checks before remote execution:

- `ruff check` on changed Python files: passed
- focused pytest for adapter, layer-wise, context, score-field tests: passed
- `basedpyright`: passed
- `py_compile` and `bash -n`: passed

## Remote Execution

Remote container: `hun_fsad_tta_012`.
Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`.
Remote result root:
`/workspace/results_remote/flowtte_scorefield_structural_all8_20260708_v4`.

Completed. Local pullback:
`results/remote_runs/dsba3/flowtte_scorefield_structural_all8_20260708_v4`.

Cleanup evidence: remote and local `anomaly_maps/` directory count is `0`.

## SuperAD Baseline and Unified Metrics

Strict same-condition SuperAD rerun is not part of this branch. Reference
contexts:

- recorded SuperAD-16: `0.765802/0.385534`
- reported SuperADD: `0.839300/0.626113`
- directly comparable internal H+ FlowTTE reference: `0.836739/0.527427`

## Results and Analysis

Mean all-eight results:

| Variant | AUROC_0.05 | F1 | dAUROC vs baseline | dF1 vs baseline |
|---|---:|---:|---:|---:|
| baseline | 0.836739 | 0.527427 | 0.000000 | 0.000000 |
| support_position_center | 0.785133 | 0.429653 | -0.051606 | -0.097775 |
| support_position_zscore | 0.701830 | 0.278144 | -0.134909 | -0.249284 |
| foreground_energy | 0.834598 | 0.526807 | -0.002141 | -0.000620 |
| center_plus_foreground | 0.808839 | 0.471613 | -0.027900 | -0.055815 |

Detailed artifacts:

- `scorefield_summary.tsv`
- `scorefield_per_object_delta.tsv`
- `scorefield_fragmentation_bad.tsv`

Interpretation:

- Support-position calibration is structurally wrong for the current H+ DVT
  FlowTTE score field. `support_position_center` loses F1 on all eight
  objects; `support_position_zscore` collapses the mean further.
- The support feature-energy foreground prior is near-neutral on the mean, not
  a robust improvement. It wins F1 on 5/8 objects, but the gains are small and
  offset by meaningful losses on `wallplugs`, `walnuts`, and `rice`.
- Combining position centering with foreground prior inherits the positional
  calibration damage.
- This rules out a simple support-stat score-field correction as the next
  structural answer. The issue is not just repeated positional bias that can be
  subtracted from support maps.

## Continuation Assessment

Strict method claim: no. The branch loses to its own H+ FlowTTE baseline and
does not approach reported SuperADD F1.

Continuation: no for this exact support-position/feature-energy score-field
family. The only non-catastrophic variant, `foreground_energy`, is effectively
baseline-tied and does not justify another class-agnostic score-field sweep.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE`.

Asset retained: fragmentation analysis code is useful for future diagnostics,
but the tested score-field transformations should not be promoted or tuned.
