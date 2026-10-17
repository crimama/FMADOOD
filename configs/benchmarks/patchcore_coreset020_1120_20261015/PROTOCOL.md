# PatchCore coreset 0.2 — AD2 public/private

## Fixed comparison

User explicitly requests coreset ratio 0.2 on public and private. The first run
matches the existing official private baseline: 16-shot seed 0, all eight AD2
classes, fixed VOID_support_16 manifest. Only coreset_sampling_ratio changes
from 1.0 to 0.2; existing exact CPU greedy farthest-point sampler is retained.
DINOv2 ViT-L/14, layers 5/11/17/23, 1120 input, FP32, normalized kNN and other
preprocessing/scoring settings remain unchanged. No trainable parameters.
This is a project PatchCore variant, not an assertion of exact paper reproduction.

Hypothesis: restoring representative coreset selection changes accuracy and
reduces bank/query cost. No assumed accuracy direction. Primary public metric:
8-class macro AUPRO@0.05 (higher better); secondary pAUROC@0.05, I-AUROC and
fit/query cost. Public is previously seen development data, not sealed evaluation.
No performance-based stopping or ratio selection. A failure invalidates the
failed class execution and stops its lane; completed artifacts are retained.

## Private protocol

Inference covers test_private and test_private_mixed. Threshold is recomputed
for this configuration from sorted train/good excluding support, every eighth
normal image, class-wise normal pixel p95 multiplied by 1.421. Additional normal
images beyond 16-shot are used only for threshold calibration. Private labels
are never used. TIFF score maps and binary PNG masks are explicitly enabled
for the authorized official private evaluation; no fitted weights are saved.
Official AUPRO/ClassF1/SegF1 require server evaluation, not local public proxies.
If quota is exhausted, retain the checked submission archive pending upload.

## Execution

Reuse hun_fsad_tta_012 on dsba3; all four already exposed GPU UUIDs, two classes
per lane. No remote artifact deletion or container reallocation. Public uses
scripts/experiments.py run. Private uses the existing submission preparation
script. Retain logs remotely and pull only compact evidence by default.
After all private classes finish, check exact expected class/split filenames,
finite float16 TIFF arrays and binary PNG masks and build submission.tar.gz.
A local immutable snapshot is deployed with run_remote.sh update-code; no git
push is authorized. Record source/config hashes and actual launch command.

Additional public shot/seed scope is pending the user's optional clarification;
this queue starts only 16-shot seed 0 public and private, without waiting on it.
