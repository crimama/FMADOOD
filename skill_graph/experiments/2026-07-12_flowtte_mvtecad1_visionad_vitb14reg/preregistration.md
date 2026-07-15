# FlowTTE MVTec AD1 VisionAD-aligned shot sweep

## Motivation and claim boundary

Measure the current representative FlowTTE configuration on classic MVTec AD
under the user-requested VisionAD encoder and input contract. This is an
input/backbone-aligned comparison, not a VisionAD reproduction: FlowTTE fits its
normal-only Flow head while the DINOv2 encoder remains frozen. A numerical
superiority claim is blocked until a same-condition VisionAD artifact exists.

## Locked protocol

- Dataset: classic MVTec AD, all 15 categories, complete test split.
- Shots: 1, 2, 4, 8, 16; seed 1; seeded random supports without replacement.
- Encoder: `dinov2_vitb14_reg`, frozen and in eval mode.
- Input: square bicubic resize 448, center crop 392, ImageNet normalization.
- Feature contract: layers 2, 5, 8, 11; four-layer `visionad_mean_l2` fusion;
  patch 14; expected grid 28x28 and embedding width 768.
- Current-method structure: DVT `position_mean` alpha 1.0; fused patch-wise MLP
  flow; 3 epochs; 2 coupling blocks; hidden multiplier 1; LR 2e-4; clamp 1.9;
  `lambda_logdet=0.02`; brightness 0.8-1.2; no TTE
  (`expansion_budget=1.0`); latent distance plus density weight 0.25.
- GPUs: dsba5 GPU 0 and 1, shot-level parallelism.
- Cleanup: remove only generated anomaly maps after metrics and manifest exist.

## Evaluation contract

Primary metrics are class-macro `i_AUROC`, `i_AUPRC`, `p_AUROC`, `p_AUPRC`,
and `p_AUPRO`. Image scores use the mean top 1% of the full-resolution anomaly
map. Pixel rank metrics use a monotonic signed-log1p transform followed by a
per-category 65,536-bin linear uint16 histogram. This avoids the observed raw
float16 overflow while preserving global score order at histogram resolution.
AUPRO uses 8-connected regions and normalized integration through FPR 0.30.
Legacy metric names are retained as aliases.

The current evaluator upsamples predictions to original resolution and compares
against original masks. VisionAD's released evaluator applies its own mask/output
resize convention, so paper-only values are contextual rather than a strict
paired baseline.

## Stage gates

1. Unit contract: frozen encoder, 392/14 geometry, four 784x768 outputs.
2. CLI contract: all structural and evaluation settings serialized.
3. dsba5 preflight: resolve host/container; confirm GPUs 0,1, MVTecAD mount,
   cached DINOv2-R repo/checkpoint, and real 28x28/768 feature output.
4. Full sweep: all five shot manifests and metrics files present.
5. Aggregate and audit: check finite five-metric rows, all 15 category entries,
   cleanup evidence, logs, and no failed GPU worker.

## Verdict rules

- `ACCEPT`: all execution and artifact gates pass; report measured performance.
- `REJECT`: run completes but contract or metric validity fails.
- `BLOCKED_BASELINE`: measured result is valid, but SOTA superiority cannot be
  concluded without a same-condition VisionAD run.
- `BLOCKED_EXECUTION`: dsba5 identity/access/container/data cannot be resolved.
