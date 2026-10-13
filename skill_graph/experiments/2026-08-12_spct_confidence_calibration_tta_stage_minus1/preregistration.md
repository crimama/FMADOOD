# SPCT Confidence, Calibration, Synthetic Gate, and Key-TTA Stage -1

Date: 2026-08-12
Protocol ID: `spct-confidence-calibration-tta-stage-minus1-20260812`
Status: `PRE-REGISTERED / NOT RUN`
Maximum verdict: `PROMISING_DIAGNOSTIC`

## 1. Custody and frozen inputs

Use only MVTec AD2 single-image `train/good` under
`/home/hunim/Volume/DATA/mvtec_ad_2`, for `can`, `fabric`, `vial`, and
`walnuts`. The paired Probe16 manifest and full-population manifest remain
byte-identical to SPCT, with SHA-256 values
`7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
and `58f6b249d82ac713c9a0051675244db5216ebbe995d18ecb123058c9499a592d`.
Full counts are `412/387/291/432` for can/fabric/vial/walnuts.

The exact four Probe16 folds, held sources, eight signed nuisance codes,
scratch/blur evaluation families, masks, and seeds are shared across budgets
and arms. A held source is absent from every fit graph, retrieval memory,
tangent, confidence state, conformal block, and synthetic gate. No
`test_public`, `test_private`, GT, condition token, or filename-derived label
may be read.

## 2. Frozen representation and SPCT state

Retain the completed SPCT representation and geometry without retuning:

- frozen DINOv2 ViT-L/14, resolution 672;
- layer-5 L2-normalized retrieval key and layer-11 L2-normalized value;
- adaptive `24 x 24` part grid;
- same-position cross-source graph `k_graph=8`;
- frozen retrieval `k=4` with SHA-path tie ordering;
- eight signed normal photometric displacements;
- root-plus-neighbours rank-4 local tangent;
- root/part scale from fit-only normal displacement energy.

All full-budget sources remain eligible for graph, retrieval, and tangent
fitting. Calibration and gate-query caps are execution bounds only and use
the first eligible sources under `(SHA256(relative_path), relative_path)`.

## 3. Support-LOO reliability state

For at most 64 fit sources per fold, score the clean view and all available
normal nuisance endpoint views while excluding that source from retrieval.
The state is fitted in source-image blocks, never as independent patch labels.

For each retrieved root/part, record:

- top-1 distance and normalized top-1/top-2 distance margin;
- clean-to-shift top-k Jaccard and winner stability;
- raw score `s_raw`, tangent score `s_tan`, and projection ratio
  `s_tan/(s_raw+1e-6)`;
- source-block normal exceedance under the fit-only q95 threshold.

The root/part reliability is the shrinkage estimate of stable normal routes:
root/part evidence, same-part evidence, and object-global evidence are combined
with fixed pseudo-counts `4`, `16`, and `32`. The activation threshold is the
largest deterministic confidence threshold whose source-block LOO normal
exceedance is at most `.05`; ties choose the higher threshold. No held outcome
can change this state.

## 4. Hierarchical conformal calibration

Calibration operates on support-LOO normal blocks. For a query patch score,
construct finite-sample upper-tail conformal probabilities at three levels:
retrieved-source/part, part, and object-global. Combine their counts with the
same `4/16/32` pseudo-count shrinkage and use `-log(p)` as the calibrated score.
The fixed alarm is `p <= .05`. Image/source is the exchangeability block;
patch-IID coverage is reported only as a deliberately invalid null.

## 5. Synthetic activation gate

At most 32 fit sources per fold generate CutPaste and chromatic-blotch pairs
from a seed domain disjoint from held scratch/blur. Graph, retrieval, tangent,
and LOO reliability are frozen first. A small L2-regularized logistic gate is
fitted on compact reliability features to predict whether tangent projection
improves the clean-to-synthetic separation relative to raw retrieval. It may
choose between `s_raw` and `s_tan`; it may not add a learned anomaly score,
change a feature, or modify calibration state.

The fit must record positive/negative target counts, activation rate,
objective, gradient norm, coefficient hash, and generator-family holdout.
Label-swap and deterministic mask-roll fits use identical rows and optimizer.

## 6. Constrained retrieval-key TTA

TTA is an isolated final lane. It updates only a temporary query layer-5 key;
layer-11 value, backbone, support memory, graph, tangent, gate, and calibration
remain frozen. Starting from the frozen key, run exactly three projected steps
toward the mean of its current same-part top-4 support keys:

```text
k_(t+1) = normalize(0.9 * k_0 + 0.1 * mean(top4(k_t)))
```

Retrieval is recomputed after each step. No query is inserted into memory and
no state persists between images. The hard null uses the same update energy
toward a deterministic part-shuffled support prototype. The unavailable clean
held key remains an oracle only.

## 7. Arms

| arm | purpose |
| --- | --- |
| `B0_retrieval_raw` | independent normalized raw retrieval residual |
| `B1_unconditional_tangent` | completed SPCT local orthogonal residual |
| `G0_loo_confidence_gate` | LOO reliability chooses B0 or B1 per patch |
| `GN0_activation_shuffle` | fixed activation-rate root/part shuffle |
| `GN1_distance_only_gate` | top-1 distance/margin gate without LOO stability |
| `C0_hierarchical_conformal` | G0 with source/part/global block conformal score |
| `CN0_global_conformal` | object-global block conformal only |
| `CN1_patch_iid_conformal` | deliberately invalid patch-IID calibration null |
| `S0_synthetic_activation_gate` | disjoint synthetic data trains activation only |
| `SN0_synthetic_label_swap` | same fit with activation targets reversed |
| `SN1_synthetic_mask_roll` | same fit after deterministic synthetic mask roll |
| `T0_key_tta` | three-step same-part retrieval-key-only refinement then G0 |
| `TN0_part_shuffle_key_tta` | matched-energy wrong-part key refinement |
| `O0_clean_key_retrieval` | unavailable clean-key oracle; diagnosis only |

All 14 arms run for both budgets and all four objects. `O0` is excluded from
threshold selection, model choice, continuation gates, and positive claims.

## 8. Metrics and gates

Primary diagnostic metric is held synthetic patch
`diagnostic_seg_pAUROC_0.05`, with AUPRC, local/broad family values,
support-derived fixed-threshold FPR/TPR, source-block conformal coverage,
activation rate, retrieval Jaccard, and source entropy. Use 10,000 fixed-seed
paired hierarchical bootstrap draws: object, held source, then code/family
averaging.

Validity requires exact manifests/folds/coverage, executable held-source
exclusion, finite and nonconstant scores, source-block LOO custody, disjoint
synthetic seed/family custody, no oracle contribution, TTA key-only mutation,
no cross-query state, and compact JSON-only artifacts.

Permissive independent continuation gates are:

- `G0`: pAUROC at least `B1-.005`, FPR at least `.02` below B1, no object FPR
  more than `.01` above B1, activation strictly between `.05` and `.95`, and
  superiority over at least one gate null by `.005`;
- `C0`: held-normal FPR at most `.06`, pAUROC loss at most `.01` versus G0,
  source-block coverage in `[.92,.98]`, and lower FPR than both calibration
  nulls;
- `S0`: nonzero finite gate gradient and activation, plus pAUROC `+.005` or
  FPR `-.02` versus G0, while beating both synthetic nulls and remaining
  nonnegative on scratch and blur;
- `T0`: pAUROC `+.01` versus the same pre-TTA G0 scorer, FPR increase at most
  `.01`, local and broad retention at least `.95`, and `+.005` over TN0.

Full-support evidence additionally requires paired improvement over support16
with positive bootstrap lower bound and at least three of four object wins for
the winning non-oracle arm. Weak early gates do not cancel later arms.

## 9. Execution

Each `(object,budget)` shard shares feature extraction across all arms. Queues
run sequentially per GPU and in parallel across servers:

- dsba3 host GPU3 / container `cuda:0`: `vial/support16`, `vial/full`,
  `walnuts/support16`, `walnuts/full`;
- dsba5 host GPU3 / container `cuda:0`: `fabric/support16`, `fabric/full`,
  `can/support16`, `can/full`.

The wrapper must verify GPU UUID mapping, at least 64 GiB host available RAM,
20 GiB free GPU memory, immutable code/manifest hashes, and exact ownership.
The CUDA transport backend remains bounded by `score_batch_rows=262144`.

## 10. Claim boundary

This run may yield only `PROMISING_DIAGNOSTIC`, `VALID_NO_SIGNAL`, or
`INVALID`. A SuperAD comparison or paper claim requires a separately frozen
Stage 0 public-test protocol using the paper-aligned support budget and all
eight objects. Full-train support results remain benchmark-incomparable.
