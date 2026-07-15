# Shift-Factorized Latent Bank on MVTec AD2

Date: 2026-07-15
Verdict: RUNNING / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

This is not a positional-PPD, threshold, score-fusion, or passive descriptor
retune. It estimates a test-environment shared low-rank displacement from
support-1NN latent residuals. The main risk is anomaly absorption into the
estimated shift subspace; residual-norm trimming and low rank are the
pre-registered anti-absorption controls.

## Motivation

Under MVTec AD2 illumination shift, normal test patches can move systematically
away from few-shot support. The candidate tests whether batch-shared residual
directions can be removed without suppressing sparse anomalous residuals.

## Implementable Design

- Dataset: MVTec AD2 single-image, all 8 objects, full `test_public/good,bad`.
- Support: exact SuperAD-selected 16 paths per object.
- Frozen stack: DINOv2 ViT-L/14, layers 5/11/17/23, canonical resolutions,
  Flow 3 epochs/2 couplings/width 1, static latent bank, DVT position mean,
  density 0.25, RGB guided-r8, and close/fill/erode morphology.
- Candidate-only change: uncentered eigenspace of trimmed support-1NN latent
  residuals over the complete unlabeled test batch.
- Variants: rank1/trim20; rank2/trim10,20,30; rank4/trim20; rank8/trim20.
- Primary official metric: AU-PRO@0.05. Diagnostics: pixel partial-AUROC@0.05,
  oracle morphology F1, per-object no-harm, and retained projection energy.

## Evaluation Alignment

The candidate and baseline share support, backbone, preprocessing, object set,
split, scoring stack, and aggregation. Existing exact-support Basic default is
the immediate control. SuperAD and RN-FMLK remain required before a method
claim. This rank/trim sweep is diagnostic, not a claim-level promotion run.

## Code Modification / Creation

- `src/flow_tte/shift_projection.py`
- `src/flow_tte/pipeline.py`
- `scripts/flow_tte_mvtec_ad2_core.py`
- `scripts/run_flow_tte_mvtec_ad2.py`
- `scripts/run_sflb_ad2_variant.sh`
- `tests/test_shift_projection.py`

## Added Code Evaluation

Python compilation passed. Projection tests and existing FlowTTE tests pass:
`25 passed`. Remote fruit-jelly smoke fitted rank-2 projection at 59.9 seconds
and completed projected test scoring at 69.5 seconds.

## Remote Execution

- dsba3 `/workspace/results_remote/sflb_ad2_rank_trim_20260715_v1`:
  GPUs 0--3 run rank1/t20, rank2/t10, rank2/t20, rank2/t30.
- dsba5 `/workspace/results_remote/sflb_ad2_rank_trim_20260715_v1`:
  GPUs 0--1 run rank4/t20 and rank8/t20.
- Dense maps remain until AU-PRO@0.05 is computed and audited.

## SuperAD Baseline and Unified Metrics

Pending run completion and official AU-PRO evaluation. No superiority claim is
permitted from the repository's partial pixel AUROC field alone.

## Results and Analysis

Pending.

## Continuation Assessment

Continue only if shifted-object AU-PRO or partial AUROC improves without broad
regular-object/F1 collapse. Hard-stop if all ranks are absorbed by the baseline
or higher rank monotonically removes anomalous structure.

## Conclusion

RUNNING / CONTINUE_DIAGNOSTIC.

## Post-Conclusion Storage Cleanup

Pending official AU-PRO computation; remove all regenerable `anomaly_maps/`
after compact metrics and the final verdict are recorded.
