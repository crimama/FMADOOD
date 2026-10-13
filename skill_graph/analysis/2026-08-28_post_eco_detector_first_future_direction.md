# Post-ECO detector-first future direction

## Decision

The next executable research generation is `PSAR-D1`: a frozen-backbone,
shallow query-map detector trained with fixed shifted-assumed synthetic anomaly
pairs. It is **not** another ECO continuation and it does **not** begin with
LoRA. Before GPU launch, close five static contract gaps and independently
verify the preregistration; then run the fixed three-object metric checkpoint
without post-result revision.

ECO-O1 remains `INVALID_FAIL_CLOSED / END_ECO_FAMILY`. No fourth O1 attempt,
support-only rescue, detector bridge retrofitted onto ECO, or ECO-derived LoRA
is permitted. The absence of an O1 metric result is not a negative detector
result.

## Why detector-first, not adapter-first

Project evidence says frozen dense ViT features and exact retrieval remain
useful, whereas several LoRA/support-only lines failed because the purported
mechanism was missing, null-absorbed, inactive, or anomaly-destructive. A
larger trainable surface is therefore not the first missing ingredient. The
first unanswered question is whether paired synthetic supervision produces a
real low-FPR pixel-map gain beyond an absolute detector and relation-broken
controls.

External work supports both sides of this choice. Direct/frozen ViT anomaly
detectors are viable in
[AnomalyDINO](https://openaccess.thecvf.com/content/WACV2025/papers/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.pdf),
[VisualAD](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VisualAD_Language-Free_Zero-Shot_Anomaly_Detection_via_Vision_Transformer_CVPR_2026_paper.html),
and [AnomalyVFM](https://openaccess.thecvf.com/content/CVPR2026/html/Fucka_AnomalyVFM_--_Transforming_Vision_Foundation_Models_into_Zero-Shot_Anomaly_Detectors_CVPR_2026_paper.html).
Synthetic anomaly detector training is established by
[DeSTSeg](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html),
[SimpleNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf),
and [RealNet](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.pdf).
Thus PSAR-D1's novelty claim must remain narrow: it tests whether the paired
relation matters under the exact SuperAD-16 shifted few-shot setting.

## PSAR-D1 checkpoint contract

| Item | Frozen choice |
|---|---|
| Objects | `fabric` strong, `vial` weak/stress, `walnuts` control |
| References | exact SuperAD-selected P16 identities and order |
| Backbone | frozen DINOv2 ViT-L/14, layers 5/11/17/23 |
| Local map inputs | unsquared, unnormalized Euclidean 1-NN per layer |
| Detector | normalization-free width-16 `1x1/DW3x3/1x1` head |
| Learned arms | M0 paired, I0 absolute/no-pair, N0+ and N0- unpaired |
| Comparators | exact same-condition SuperAD-16 and RN-FMLK |
| Training | 576 updates, seed 20260828, no early stop or sweep |
| Primary | `seg_AUROC_0.05` |
| Guardrails | `seg_F1`, `AU-PRO@0.05`, map/active/defect coverage, normal FPR, runtime, memory, per-object no-harm |
| Result policy | `revision_budget=0`; all failures and successes enter the ledger |

No per-query min-max or result-dependent thresholding is allowed. Calibrate
arm-specific q95 thresholds only from leave-source-out normal support views,
and retain raw continuous maps for AUROC/F1/AU-PRO evaluation.

The schema must emit these literal fields: `seg_AUROC_0.05`, `seg_F1`,
`AU-PRO@0.05`, `map_coverage`, `normal_FPR`, `runtime`, `memory`,
`object_results`, `delta_vs_superad`, `delta_vs_rn_fmlk`, and
`delta_vs_hard_null`. For each higher-is-better metric, the hard-null delta is
M0 minus the better of N0+ and N0-. Existing internal F1/AUPRO aliases may be
kept only with an exact-equality assertion.

## Immediate execution order

1. Encode and test the five prelaunch definitions in the final method handoff:
   cyclic calibration donor, example-role P12/P16 semantics, fixed-donor
   composite claim, distinct B0/I0 identity roles, and exact runtime/source
   custody.
2. Implement the output schema and fail-closed aggregator first. This makes
   missing/nonfinite/constant maps, coverage failure, comparator absence, or
   threshold leakage terminal before model code can hide them.
3. Implement the core, runner, remote controller, and no-data/unit tests.
4. Perform a remote CUDA validity smoke that produces no TESTpublic result;
   seal source and preregistration only after it passes.
5. Run the immutable three-object checkpoint, aggregate once, and branch from
   the preregistered verdict table.

When remote execution opens, first revalidate only the project-owned fixed
container and assigned slots. A practical shard plan is dsba3 GPU0=`fabric`,
GPU1=`vial`, GPU2=`walnuts`, GPU3=`fabric` exact repeat; dsba2 GPU6 can run
same-condition B0/B1 parity or held-synthetic work. Never enumerate, stop,
modify, or reuse another user's processes, containers, files, or GPUs. Local
GPU/data/model execution remains forbidden.

## Result-dependent branches

- `INVALID`: repair only execution/contract defects identified before any
  readable metric; otherwise create a new generation. Do not reinterpret an
  invalid run.
- `VALID_BUT_NO_GAIN / CLOSE_FAMILY`: if both SuperAD and RN-FMLK dominate,
  hard nulls absorb M0, or F1/coverage collapses, close PSAR-D1. The next new
  generation is a frozen transform-equivariant P16 memory or subspace detector
  inspired by
  [SubspaceAD](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html),
  not a PSAR rescue.
- `PROMISING_DIAGNOSTIC`: at least one intended core-metric movement, no
  F1/coverage collapse, non-absorbed hard-null signal, and nonfatal
  strong/control no-harm authorize unchanged all-eight confirmation with
  seeds `20260828/29/30` and confidence intervals.
- `ALL_EIGHT_POSITIVE + CAPACITY_GAP`: only this branch opens a separately
  named PSAR-D2. It may test one late/post-readout detector-aware LoRA
  configuration against the identical LoRA-off head. No layer/rank/LR/loss or
  seed sweep is allowed.

## Stop condition

Stop this planning branch after the static prelaunch contract passes an
independent review. Stop PSAR-D1 scientifically on a valid negative checkpoint.
Do not spend another support-only diagnostic between PSAR-D1 and its metric
checkpoint.

Design-loop artifacts:
`.omx/specs/deep-research-autoresearch-post-eco-detector-bearing-generation/`.
