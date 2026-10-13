# ATAC-v1 local Stage -1 preregistration

## Status and scope

- Design source: `.omx/specs/deep-research-autoresearch-structural-evidence-redesign-v2/stage-02/method.md`
- Design verdict: `DESIGN CONDITIONAL-KEEP / FMAD_EXPERIMENT_READY`
- Execution scope: local diagnostic requested by the user; it is not a remote-protocol result.
- Claim ceiling before results: `PROMISING_DIAGNOSTIC` only.

## Frozen custody

- Dataset root: `/home/hun/Volume/DATA/mvtec_ad_2` on the host, mounted as
  `/home/hunim/Volume/DATA/mvtec_ad_2` in the local CUDA container.
- Public objects: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`,
  `wallplugs`, `walnuts`.
- Reference manifest:
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`.
- Required manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Exactly 16 listed `train/good` references per object; no replacement or
  support selection.
- Frozen backbone: `dinov2_vitl14`, layers `[5, 11, 17, 23]`, with ATAC at
  layer 23.
- Smaller edge: 672, except `sheet_metal=448`; existing resize/crop custody.
- Primary epsilon: `0.05`; support-only locality diagnostic: `0.10`.
- Haar boundary rule: separable orthonormal pair transform with an unpaired odd
  boundary sample carried unchanged in the low branch at each level. This is
  fixed before scoring because five objects have an odd level-one patch-grid
  axis after the common DINO crop; it introduces no pad, crop, or interpolation.
- No clamp, quantization, interpolation, JPEG, or color conversion after the
  antithetic RGB path is formed.
- No query fitting, fusion, learned head, region lift, per-image rank
  normalization, or label-selected choice.

## Candidate and fixed comparators

- Candidate: two-level orthonormal-Haar antithetic normalized-token central
  acceleration, scalar calibrated by pooled support median/MAD.
- Same-run comparator: exact final-layer normalized-token 1-NN RN-FMLK.
- Same-run comparator: exact four-layer normalized-token mean 1-NN SuperAD-16.
- Hard nulls: symmetric first derivative, one-sided cosine drift, patch Haar
  RMS, Laplacian energy, Sobel energy, deterministic phase-scrambled high band,
  one-patch spatial roll, fixed random nonlinear patch encoder, and RN-bin
  matched spatial shuffle.

## Ordered execution and stopping

1. Run analytical/unit and artifact-contract tests.
2. Run a no-label smoke that verifies support resolution, model loading,
   finite maps, RGB identity, no-op identity, and linear-encoder identity.
3. Run the smallest real public object only to validate memory, runtime, map
   ordering, and evaluator compatibility.
4. Continue to all eight objects only if no fatal identity, excursion,
   support-LOO, or artifact-integrity gate has already failed. A fatal gate is
   recorded without retuning.
5. Aggregate strict claim gates first, then the soft continuation gates.

## Frozen decision gates

- RGB antithetic identity infinity residual `<=1e-6`.
- No-op and linear-encoder token acceleration maxima `<=1e-6`.
- Every map finite and nonconstant.
- Per-image out-of-range RGB fraction `<=0.005` and score/excursion Spearman
  `<=0.30`.
- Median support map Spearman between epsilon 0.05 and 0.10 `>=0.90`; median
  scale ratio in `[0.5, 2.0]`.
- All 16 LOO scales finite and `max/min <=2`.
- Candidate macro `seg_AUROC_0.05` margin over the complete hard-null envelope
  `>=0.010`.
- Weighted RN-bin conditional margin over its matched shuffle `>=0.010`.
- Shifted-good FPR increase versus exact RN `<=0.020`.
- Macro candidate F1 loss versus exact RN `<=0.020`.
- Strict keep additionally requires positive all-eight same-condition deltas
  against both SuperAD and RN on the frozen metrics and scale safeguards.

## Local runtime

- GPU observed before execution: one NVIDIA RTX 4090, 24 GB, idle.
- Reused container: `hun_fsad_tta_local_stage1a0`.
- Cached DINOv2-L/14 checkpoint and hub repository are used offline.
- Raw/native anomaly-map intermediates are disposable after aggregation; JSON,
  compact diagnostics, commands, hashes, and reports are retained.
