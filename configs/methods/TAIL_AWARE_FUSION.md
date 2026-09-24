# Tail-aware bounded fusion: locked TF0--TF3 development comparison

Authorized implementation/execution: 2026-09-14, local GPU0 and dsba3 GPUs0--3.
Hypothesis: learning bounded layer fusion against sampled high-scoring normals
can retain R1 regional performance while recovering low-FPR pixel discrimination.
This is a posthoc hypothesis from the R/P suite, not an established cause.

## Method and brief overlap check

R1's frozen DINOv2 ViT-L14 layers11/23, independent per-layer NN references,
normal16 supports, 256 tokens/source/resolution and leave-source-out MAD scales
are unchanged. Resolutions448/672 remain equally weighted. Source bank sampling
and scales are exactly R1, not refitted to synthetic/query scores.
For each resolution, native-interpolated normalized distances a,b are fused as
w*a+(1-w)*b, where w=(1+rho*tanh(g))/2, rho=.5. The global gate has1 parameter;
the local gate is a shared2->16->1 GELU MLP (65 parameters), taking log1p(max(e,0)).
Zero output initialization gives uniform fusion; TF0 uses the exact parent R1
prediction path. Weight range [.25,.75] bounds score changes, NOT metric loss.
The gate cannot change a location where both layer distances are equal.

Existing tail_contrast.py already mines high-scoring normals and uses paired
synthetic/photometric contrast. This is NOT a new tail-ranking loss claim.
The tested distinction is updating only the fusion of fixed independent R1
distances, not an adapter/distance metric or shared-reference selector. External
paper-level novelty has not been established; this is a performance experiment.

## Ablations and fixed training protocol

|Stage|learn_fusion|hard_normal_ranking|local_fusion|params|
|---|---|---|---|---|
|TF0|false|false|false|0|
|TF1|true|false|false|1|
|TF2|true|true|false|1|
|TF3|true|true|true|65|

TF1--3:64 updates, AdamW lr=.01, weight_decay0, gradient clip5, margin .5
in MAD-normalized score units; paired contrast weight .1. No sweep or best-test
checkpoint selection. These previously unspecified numeric settings are fixed
before benchmark execution. TF0 config declares64 but executes zero updates.

Each source appears four times (two per synthetic family) in the64-step deck.
Each episode has clean, photometric-clean, synthetic, and identically shifted
synthetic views. Existing rectangle_cutpaste/scratch_noise are same-image only;
the original source is excluded from all episode reference banks. There is no
external donor. Inference uses all16 supports; training uses15-source banks.
This leave-source-out/all-support bank-size difference is a declared limitation.

Positive masks contain pixels that actually changed in BOTH paired anomaly
views. Up to128 changed native pixels are sampled uniformly within the episode
(area-proportional across regions within an image, NOT globally across images).
Only clean/shifted-clean views supply normal candidates, avoiding synthetic
background contamination. At most4096 native pixels per view are uniformly
sampled without replacement. TF2/3 mine the current top5% of this pooled sample
and draw up to128 negatives. TF1 uniformly samples up to128 from the same pool.
This is a sampled tail-ranking surrogate, NOT exact dataset-wide pAUROC loss.
Lrank=mean relu(.5-positive+negative). Lpair=mean[(S(a)-S(n))-(S(Ta)-S(Tn))]^2
on the SAME positive coordinates. L=Lrank+.1*Lpair. Tail selection is detached;
selected normal scores retain gradients. No evaluation GT enters fitting.

RNG streams for source/family/view generation, native sampling and negative
sampling are independent of gate initialization. TF1/2/3 persist the same64
episode identities, view/mask/coordinate hashes and transforms for audit. Eight
encoder calls/update (four views x two resolutions); no backbone gradients.
Fitting retains one episode of small sampled scalar distance tensors, not
dense native maps or image/feature caches across updates. Native samples use
bilinear grid_sample, tested against the common native interpolation.
Inference keeps only the current query/resolution's two scalar maps; the gate
is chunked at4096 pixels. No query updates, model downloads or map/checkpoint
outputs. Compact training losses, quantiles, gate weights and manifests persist.

## Evaluation and decision boundary

32 class runs: TF0->TF1->TF2->TF3, all eight AD2 classes, seed0, normal16,
no_mask_no_rotation, identical backbone weights and common float16 streaming
metric contract. TESTpublic is seen development, not a sealed final holdout.
Primary remains full8 macro AUPRO@.05; targeted secondary is pAUROC@.05;
F1 oracle remains diagnostic. Macro weights classes equally, not GPU lanes.
Require non-decreasing AUPRO versus fresh TF0 and increased pAUROC for a joint
win. Prior R1 anchors52.47033/75.99215, L3C pAUROC77.15604 are context only;
fresh TF0 is the comparator. Any AUPRO decrease is reported as a tradeoff.
No numerical improvement, significance or synthetic-to-real transfer promised.
TF3-TF2 changes capacity and conditional fusion together; disclose65 vs1 params.

Stop each lane on an execution error or14400s/class safety timeout; do not
automatically revise methods. Preserve completed outputs and failure evidence.
Default32 runs only; no multiseed, tuning sweep, or automatic second suite.
TF0/R1 exact parity, CPU/CUDA contracts and real cached-backbone smoke precede
launch. Synthetic-vs-real mismatch, gate saturation, concordant layer errors
and sampled-tail coverage are explicit failure possibilities.

## One-command launch and placement

Same class/device mapping as the preceding R/P suite:
local0:can,fabric; dsba3 GPU0:fruit_jelly,wallplugs; GPU1:rice,walnuts;
GPU2:sheet_metal; GPU3:vial. Existing containers only, one worker/device.
Remote source is an immutable local snapshot through run_remote.sh update-code,
not a git push. Do not mutate mounted source/configs while the suite runs.

```bash
source configs/remote/dsba3.env
export REMOTE_GPU_DEVICES=GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f,GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae,GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639,GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62
bash scripts/run_experiments.sh --backend both --container hun_fsad_tta_012 \
  --configs configs/methods/tail_aware_fusion_TF0.yaml configs/methods/tail_aware_fusion_TF1.yaml configs/methods/tail_aware_fusion_TF2.yaml configs/methods/tail_aware_fusion_TF3.yaml \
  --labels TF0 TF1 TF2 TF3 \
  --local-device GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b --local-objects can fabric \
  --remote-devices GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 \
  --remote-objects fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --output-root /results/tail_aware_fusion_20260914 \
  --log-dir results/experiment_launch/tail_aware_fusion_20260914 \
  --split test_public_seen_development --timeout 14400 --workers 1
```

Remote full logs remain remote. Recover only compact quantitative results,
configs/manifests; collect through scripts/experiments.py into the existing
results/comparisons.md and paired canonical CSV. No method-specific queue or
collector is introduced. Final collection must check64 matching episode hashes
per TF1/2/3 triplet, support/query/GT provenance, scales, update counts and
prediction validity before comparing metrics.
