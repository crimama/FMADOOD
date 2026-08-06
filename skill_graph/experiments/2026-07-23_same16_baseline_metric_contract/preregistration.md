# Same-16 Basic/SuperAD/RN-FMLK public-shadow baseline

Date frozen: 2026-07-23  
Status: `DESIGN_FROZEN / IMPLEMENTATION_VERIFICATION`

## 1. Motivation

The DINOv2-L Basic mechanism audit cannot support a public-shadow comparison
until Basic, SuperAD-16, and RN-FMLK are evaluated on the same 16 normal
references and the same original-resolution metric implementation. Historical
SuperAD/RN-FMLK results used a different reference set, while Stage M1 metrics
were computed on feature patch grids. Neither is a valid substitute.

## 2. Implementable design

- Dataset: MVTec AD2, all eight objects, full `test_public/good,bad`.
- Claim scope: public-shadow development evidence only; no TESTprivate claim.
- Shared support: exactly 16 paths per object from
  `superad16_dinov2_reference_paths.json`, SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Backbone: DINOv2 ViT-L/14, resolution 672 except `sheet_metal=448`.
- Basic: operational `minus_loo`, layers `[5,11,17,23]`, position-mean DVT,
  two coupling layers, width 1, 3 epochs, LR `2e-4`, clamp 1.9, tail 0.3,
  top ratio 0.05, log-det `1e-3`, distance 1.0 plus density 0.25, no RGB
  guide for the strict raw row.
- SuperAD-16: multi-layer normalized 1-NN mean over `[5,11,17,23]`.
- RN-FMLK: normalized final-layer 1-NN over layer 23.
- No TTA, query update, query transformation, object routing, or map switching.

## 3. Evaluation alignment

Primary machine metrics are decimals:

- `seg_AUROC_0.05`: sklearn standardized partial pixel AUROC at max FPR 0.05;
- `seg_F1_raw`: pooled original-resolution oracle best-threshold pixel F1;
- `seg_AUPRO_0.05`: connected-component PRO area normalized over FPR
  `[0,0.05]`;
- coverage: prediction/GT map counts, shape matches, nonconstant map count,
  anomalous region count, and non-finite score count.

`seg_F1` equals `seg_F1_raw` for this strict table. Guided filtering and
morphology are separate P1/P2 output ablations and cannot replace the raw row.
All three baselines must report the exact same support-manifest hash and all
eight objects. A missing object, reference mismatch, non-finite map, or metric
schema mismatch yields `BLOCKED_BASELINE`.

## 4. Execution and retention

Run only in the fixed dsba3 container `hun_fsad_tta_012`, after host GPUs 0--3,
dataset population, offline DINOv2 cache, free-space, and source-hash checks.
Dense maps are retained only until all three raw metrics and coverage audits
are written and independently aggregated, then deleted from remote and local
result roots. Compact metrics, manifests, hashes, logs, and cleanup evidence
remain.

## 5. Decision gate

This stage establishes a comparable baseline table; it does not select a new
method. Basic-vs-baseline deltas are descriptive public-shadow evidence.
Subsequent P0/P1/P2 and A6/fusion work may start only after the table is
complete or explicitly marked `BLOCKED_BASELINE` with an infrastructure
reason.
