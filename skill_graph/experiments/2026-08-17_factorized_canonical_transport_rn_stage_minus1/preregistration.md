# Factorized Canonical Transport RN Stage -1 Preregistration

Date: 2026-08-17  
Protocol: `fact-rn-stage-minus1-20260817`  
Status: frozen before execution  
Claim class: mechanism diagnostic only; not benchmark-comparable

## 1. Research question and hypothesis

Can an offline intervention-learned, support-conditioned canonical representation reduce normal nuisance displacement without suppressing local or broad defect evidence? The candidate must change the feature geometry before nearest-neighbor scoring. It may not gate, fuse, or rerank raw RN scores.

## 2. Data custody and split

- Dataset: MVTec AD2 single-image.
- Held target objects: `can`, `vial`, `rice`, `walnuts`.
- For each target, offline learning may read only `train/good` from the other seven AD2 objects.
- Target conditioning uses the exact fixed SuperAD-16 support manifest only.
- Target evaluation uses `train/good` excluding those 16 sources plus `validation/good` where present.
- `test_public`, `test_private`, their mixed variants, labels, and ground truth are forbidden in every Stage -1 phase.
- Synthetic local and broad defects are generated only from held target normal images after all model and support-context parameters are frozen.

Every shard must emit hashes of the offline source set, target support set, checkpoint, code/config, and the actual path-access ledger. Query features may not update parameters, support context, or memory.

## 3. Architecture and training

The candidate `M0_FACT` contains:

1. Frozen DINOv2 ViT-L/14 patch descriptors.
2. A permutation-invariant support-set encoder.
3. Four probabilistic part-conditioned frames.
4. Two learned conditional affine-coupling blocks.
5. An explicit content/nuisance latent split.
6. A support-conditioned canonical transport learned offline from paired normal interventions.
7. One exact nearest-neighbor distance in canonical content space.

Losses are canonical content consistency, nuisance intervention prediction, coupling cycle reconstruction, support leave-one-out deployed-distance alignment, VICReg-style variance preservation, and local/broad synthetic non-contraction. Synthetic defects supervise only non-contraction; they are not anomaly-classification positives.

The offline model is frozen before target support is encoded. Target support only produces a static context. No query-time optimization or TTA is allowed.

## 4. Arms and controls

- `B0_raw_RN`: exact RN on the same frozen raw descriptors.
- `I0_identity`: identical FACT scoring pipeline with identity transport; implementation sanity control.
- `M0_FACT`: full learned candidate.
- `N0_intervention_shuffle`: intervention codes shuffled during offline learning.
- `N1_part_frame_shuffle`: support part/frame condition shuffled.
- `N2_no_noncontraction`: remove the non-contraction objective.
- `N3_support_PCA_projection`: support-only low-rank projection hard null.

`N3` is explicitly a closed-branch comparator and may never be reported as FACT-RN. Raw RN may be used for diagnostics and controls, but cannot enter `M0_FACT`'s final score.

## 5. Frozen implementation choices

- Targets: `can,vial,rice,walnuts`.
- Support budget: exact fixed 16 per object.
- Probabilistic frames: 4.
- Conditional coupling blocks: 2.
- Training is deterministic for a fixed seed and manifest.
- Offline source images may be deterministically capped per object for compute, but the cap and selected-path hash must be identical across applicable arms.
- Held-normal and synthetic examples are score-only inputs and never part of fit state.
- Artifacts are compact JSON/log/checkpoint hashes only; no feature banks or dense anomaly maps are retained.

## 6. Metrics

Primary mechanism metrics:

- held-normal exact-RN q95 and q99 distance change relative to `B0_raw_RN`;
- held nuisance displacement reduction under sealed intervention families;
- raw-RN clean neighbor top-1 and top-5 retention;
- local and broad synthetic defect contrast retention;
- finite/nonconstant patch-score coverage;
- margins from `M0_FACT` to all hard nulls.

Macro results average target objects equally. Per-object values remain mandatory.

## 7. Decision gates

Strict pass requires all of:

- q95 and q99 normal tails reduced by at least 10% macro and on at least 3/4 objects;
- held nuisance displacement reduced by at least 30%, and `M0` beats `N0` by at least 15%;
- clean raw-neighbor retention top-1 at least 0.85 and top-5 at least 0.95;
- local and broad defect retention at least 0.95 macro, with no object below 0.90;
- `M0` beats every hard null on macro and at least 3/4 objects;
- complete, finite, nonconstant outputs and a clean custody audit.

Soft continuation requires q99 tail reduction of at least 5% on 2/4 objects, macro defect retention at least 0.90, and `M0` above both intervention and frame shuffles.

Hard stop applies if a shuffled null matches `M0`, any object has defect retention below 0.85, clean top-1 retention is below 0.70, `N3` matches the full model, or any forbidden path/query update occurs.

## 8. Compute and execution plan

One immutable code/checkpoint snapshot is used for all four target shards. Local execution first performs a tiny smoke and checkpoint determinism test. The four held-object shards are then assigned to idle, UUID-verified GPU slots across local, dsba3, and dsba5. A shard is complete only after its compact summary, custody ledger, and completion marker pass validation. Aggregation fails closed until all four shards are available.

## 9. Interpretation boundary and next action

Passing Stage -1 means only that learned factorization has a distinct nuisance/defect-preserving mechanism. It does not establish anomaly-detection performance. Only a strict or soft continuation verdict authorizes a separately preregistered reduced TESTpublic experiment. Failure closes this FACT-RN formulation without tuning on public data.

