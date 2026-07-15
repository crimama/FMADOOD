# FlowTTE MVTec AD1 VisionAD-aligned shot sweep

## Outcome

The complete all-15-category MVTec AD sweep finished on dsba5 for shots
1, 2, 4, 8, and 16. All five artifact and metric gates passed. The measured
result is valid for the locked FlowTTE protocol, but the final comparison
verdict is `BLOCKED_BASELINE`: this run aligns the frozen encoder and input
geometry with VisionAD, but is not a same-evaluator VisionAD reproduction.

## Aggregate results

All values are unweighted macro means over the 15 classic MVTec AD categories.

| Shots | i-AUROC | i-AUPRC | p-AUROC | p-AUPRC | p-AUPRO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 75.69 | 88.16 | 87.40 | 23.12 | 65.24 |
| 2 | 88.72 | 93.59 | 94.23 | 36.07 | 81.57 |
| 4 | 91.58 | 95.41 | 95.03 | 38.89 | 84.86 |
| 8 | 94.05 | **96.66** | 95.65 | 41.72 | 86.69 |
| 16 | 93.94 | 96.60 | **95.95** | **42.55** | **87.53** |

The largest gain occurs from 1 to 2 shots. Relative to 1 shot, 16 shots adds
18.25 points i-AUROC, 8.44 points i-AUPRC, 8.54 points p-AUROC, 19.43 points
p-AUPRC, and 22.29 points p-AUPRO. The small 8-to-16 image-level decrease
(-0.12 i-AUROC and -0.06 i-AUPRC) is within a single support draw and cannot
be treated as a stable scaling reversal.

## Stage 0 — claim and baseline contract

- Target claim: performance of the current representative FlowTTE setting on
  classic MVTec AD under VisionAD-aligned encoder/input settings.
- Strict superiority claim: blocked because no same-condition VisionAD artifact
  is available (`visionad_baseline_source=BLOCKED_BASELINE`).
- VisionAD alignment: frozen `dinov2_vitb14_reg`, resize 448, center crop 392,
  patch size 14, and 28x28 feature grid.
- Non-equivalence: FlowTTE fits a normal-only flow head for each category while
  VisionAD's quoted few-normal-shot setting performs no further dataset
  training. The evaluator geometry also differs.

## Stage 1 — protocol lock

- Dataset: complete classic MVTec AD test split, all 15 categories.
- Supports: seeded random normal training samples without replacement,
  seed 1, with every selected path serialized in each manifest.
- Shots: 1, 2, 4, 8, 16.
- Encoder: DINOv2-R ViT-B/14, frozen, evaluation mode, embedding width 768.
- Features: layers 2, 5, 8, 11; four-layer `visionad_mean_l2` fusion.
- Method: current Phase-3 FlowTTE configuration recorded in the preregistration
  and per-shot manifests.
- GPUs: dsba5 host GPUs 0 and 1 through fixed container
  `hun_fsad_tta_012`; shot-level parallel scheduling.

## Stage 2 — local contract verification

The runner was repaired to serialize the complete configuration, validate
448-to-392 and 392/14 geometry, force the encoder frozen, and expose the locked
method flags. Metric aliases use requested names. Image AP was corrected to
match standard tied-score average-precision semantics.

Final local verification:

- Full test suite: `407 passed` with five non-failing warnings.
- Ruff checks: passed.
- Python compilation and launcher shell syntax: passed.
- Real-backbone contract: input 3x392x392; four outputs of 784x768; all encoder
  parameters frozen; model in evaluation mode.

## Stage 3 — dsba5 preflight

- The supplied dsba5 preset was repaired and restricted to owner-only file
  permissions.
- The existing unrelated container was not reused because its workspace mount
  and Python dependencies were incompatible.
- The project image was built and the requested fixed container was created
  with exactly host GPUs 0 and 1 visible.
- All 15 dataset categories and their normal/test/ground-truth structure were
  checked before execution.
- The official DINOv2 repository and ViT-B/14-register checkpoint were cached,
  then a real forward pass confirmed the 28x28/768 contract.

## Stage 4 — metric validity correction

An initial pre-result run revealed that every raw anomaly-map pixel exceeded
the float16 finite range; storing raw scores as float16 would therefore have
made pixel ranking invalid. That incomplete run was stopped and removed before
being used as evidence.

The valid rerun uses a monotonic signed-log1p transform followed by a
per-category 65,536-bin linear uint16 histogram for p-AUROC and p-AUPRC. This
preserves ordering at histogram resolution without float16 overflow. A
high-score synthetic regression test verifies the corrected order and metric
behavior. p-AUPRO is computed from original float32 maps with 8-connected
regions and normalized integration through FPR 0.30.

The retained `pro_integration_limit=0.05` field is an unused legacy evaluator
configuration because legacy segmentation evaluation was disabled; the active
p-AUPRO contract is explicitly serialized as `pixel_PRO_max_fpr=0.3` and
`p_AUPRO_max_fpr=0.3`.

## Stage 5 — execution

The clean run root is
`flowtte_mvtecad1_visionad_vitb14reg_s1_2_4_8_16_20260712_v1`.
Shots 1/4/16 were assigned to one GPU worker and shots 2/8 to the other. Each
shot completed all 15 categories and wrote `metrics.json`,
`metrics_seed=1.json`, `run_manifest.json`, and cleanup evidence. The
controller ended with `[complete]` and no run process remains.

## Stage 6 — artifact audit

- Five of five shot summaries are finite and lie in [0, 1].
- Every shot has exactly 15 per-category metric rows and 15 object diagnostics.
- Every object has exactly the requested number of unique selected support
  paths, and the selection seed is serialized.
- Embedded manifest metrics exactly match the standalone metric files.
- Logs contain no traceback, CUDA out-of-memory, NaN, infinity, or error line.
- The aggregate summary exactly matches the five standalone metric files.

## Stage 7 — evaluator and comparison boundary

Predictions are upsampled to each original image resolution and compared with
the original mask. This differs from the released VisionAD evaluator's own
output/mask resize convention. Consequently, paper-only VisionAD numbers must
not be subtracted from this table as if they were paired measurements.

The table answers “how does the current FlowTTE setting perform with the same
frozen encoder family and input geometry?” It does not yet answer “does FlowTTE
beat VisionAD under an identical end-to-end protocol?”

## Stage 8 — cleanup and retention

Only regenerable anomaly-map directories were deleted after metrics and
manifests existed. Remote and local audits both find zero retained
`anomaly_maps` directories. Compact logs, manifests, selected-support
provenance, per-category metrics, and the aggregate summary remain available.

## Stage 9 — verdict and next gate

Final verdict: `BLOCKED_BASELINE`.

The execution is accepted as a valid candidate measurement. A strict SOTA
comparison requires running VisionAD code on the same five support selections,
same original-resolution evaluator, same five metrics, and preferably multiple
support seeds. Until that baseline exists, no superiority or statistical claim
is supported.

## Artifacts

- Preregistration: `preregistration.md`
- Aggregate summary:
  `results/remote_runs/dsba5/flowtte_mvtecad1_visionad_vitb14reg_s1_2_4_8_16_20260712_v1/shot_sweep_summary.json`
- Per-shot manifests and metrics: the `shot_1`, `shot_2`, `shot_4`, `shot_8`,
  and `shot_16` directories under that result root.
- Controller log:
  `results/remote_runs/dsba5/flowtte_mvtecad1_visionad_vitb14reg_s1_2_4_8_16_20260712_v1_controller.log`
