# R / P: two independent method families (2026-09-14)

User authorized implementation and the full experiment suite on the existing
local GPU and all four dsba3 GPUs. This is a development experiment, not a
sealed evaluation. No expected improvement is represented as an observed result.

## Locked comparison

Eight configs: D3, L3C, R0, R1, R2, P0, P1, P2, each on all eight AD2 classes
(64 final class runs). Same normal16 support selection, seed0, query population,
no_mask_no_rotation shared preprocessing, frozen DINOv2 ViT-L/14 weights,
448+672 resolutions, streaming metric implementation. Primary full8 macro
AUPRO@0.05, secondary pAUROC@0.05. F1 oracle is diagnostic only. TESTpublic is
seen development. Run fresh D3 and L3C anchors; historical numbers are not
substitutes for the same-batch controls.

R0/D3/L3C use layer23; R1/R2/P0/P1/P2 explicitly use [11,23]. This is a
declared feature-condition ablation, not an unchanged-feature claim.
D3 and L3C retain their existing 64-update configs. R has no learned parameters.
P uses128 updates: normal16 x eight hidden tiles, each source/tile once.
P0/P1/P2 use identical episodes, synthetic generation, sampling, initialization
of shared parameters and feature-call budgets. P variants differ only in their
two declared switches. Comparing R against P is a method-level comparison,
not equal training compute or equal inference latency.

Each assigned device executes the supplied config sequence with one class
process at a time; devices execute in parallel. A class stays on one device
for all eight configs. No hyperparameter sweep, additional seed, new model
download or automatic method revision. Per-class timeout14400s is a safety
ceiling, not a runtime estimate. A lane stops on an error; other already
launched lanes may finish. An execution error is not a method-family verdict.

## R: reference matching

Implementation: src/fmad/methods/reference_matching.py.

- R0: raw frozen layer23 cosine nearest-neighbour, parity control for BCMA B0.
- R1: layers11/23 share sampled bank token identities, but search independently.
  Divide each layer distance by a fixed positive support-only leave-source-out
  MAD scale (1.4826*MAD, floor1e-4), then average layers/resolutions.
- R2: select one bank identity by late appearance +0.25 times ring-context
  similarity, then measure both layer distances against that same identity.
  Ring context is the normalized mean of the 3x3 neighbours excluding center.

Sampling is256 tokens/source/resolution, query chunks128. Support scales and
bank identities do not use queries/labels. Only sampled layer/context tokens
are retained; full support feature maps are released after each image.
R1-R0 tests the entire scaled multi-layer readout, not layer choice isolated
from normalization. R2-R1 tests the coupled context/shared-identity mechanism;
it does not individually identify the effects of its two parts.
R1/R2 intentionally share EXACT independent-NN leave-source-out scales:
R2 is not recalibrated after changing reference selection. This fixes the
layer units/weights for the paired comparison; calibration is not an estimate
of R2's deployed residual distribution. Tests enforce scale/bank equality.

Hypothesis: some late-only/global NN matches suppress actual defect evidence.
Failure: missing normal references, wrong context correspondence, or middle
features more sensitive to lighting than to defects. R2 losing to R1 would
not support the proposed correspondence mechanism under these conditions.

## P: blind-region normal prediction

Implementation: src/fmad/methods/masked_prediction.py.

All P variants hide a target tile and one-token guard in prepared image pixels
BEFORE the frozen encoder. Fill values and six RGB mean/std style features use
visible pixels only. A two-block cross-attention predictor (width128,4heads,
dropout0,8x8 pooled context) predicts layer11 features from masked layer23
context, position and visible style. There is no NN bank, observed-target skip
connection, trainable backbone, or unmasked teacher input to the predictor.
Targets are unit features scaled by sqrt(D), a fixed numerical rescaling.
Initialize predicted means and a pooled scalar variance from normal supports
only, accumulated as vector/scalar statistics without retaining feature maps.

- P0: learned mean with fixed pooled normal variance.
- P1: also predict a scalar variance factor per patch, bounded [0.1,10].
- P2: additionally apply a synthetic local margin with weight0.1, margin0.2.

Normal objective: mean_d 0.5*((z-mu)^2/variance + log variance).
Inference: mean_d((z-mu)^2/variance), not log likelihood or anomaly probability.
Synthetic objective: weighted mean ReLU(0.2+s_normal-s_synthetic) on modified
pixels' fractional patch coverage. Variance is detached in this loss: only
the mean prediction pathway receives its gradient. Normal NLL still trains
the variance and shared predictor. AdamW lr0.001, weight_decay0, clip_norm5.

Both clean and photometrically shifted normal views predict their OWN
unmasked features, not a canonical clean target. Existing rectangle_cutpaste
and scratch_noise are generated inside the selected native tile. After shared
resize/crop, synthetic pixels are clipped to the scored tile, preventing
interpolation spillover into visible context. Fractional masks are derived
from actually changed prepared pixels. Zero sampled synthetic coverage yields
a logged zero synthetic term, not fabricated labels. P0/P1 compute the same
synthetic targets for exposure-matched diagnostics but do not optimize them.
Input/augmentation episode hashes and coverage are scalar training-log fields.

Training views are streamed with backward per view/resolution. Inference uses
2x4 nonoverlapping scored tiles; guard masks can overlap but every patch is
scored exactly once. Per resolution: one observed +eight masked encoder calls.
Thus18 encoder calls/query vs two for R, not a claimed ninefold measured latency.
No cross-query or cross-step dense map retention; maps/checkpoints remain off.

Hypothesis: hiding the inspected region removes L2's direct observed-feature
copy pathway; predictive variance can distinguish normal ambiguity from
unexpected local change. This does not establish statistical identifiability.
Failure: unpredictable normal textures, normal16 memorization, anomalous
surroundings, variance absorbing defects, synthetic shortcuts, or insufficient
optimization. P1 must beat P0 on final metrics to support variance's utility;
P2 must beat P1 to support synthetic supervision on this architecture.
Training loss reduction alone is not evidence of detection improvement.

## Related-work scope

- UniVAD (CVPR2025): component-aware matching is known, not a new claim here.
  https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html
- I-JEPA (CVPR2023): masked-context feature prediction is known. Here encoders
  are frozen and the predictor is fitted on normal16; this is not I-JEPA pretraining.
  https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf
- Uninformed Students (CVPR2020): prediction discrepancy/uncertainty for AD is
  known. Our variance is conditional residual variance, not ensemble epistemic uncertainty.
  https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html

These distinctions are hypotheses to evaluate, not verified publication novelty.
Keep runtime/config/support/code provenance and compact quantitative outputs.
Collect with the existing generic experiments.py into the canonical comparison
Markdown and paired CSV, retaining full8 vs partial identity and history.

## Single Bash invocation and placement

Run from the project root after local validation and `scripts/run_remote.sh
update-code` under the sourced remote preset. The existing container is reused.
Do not update code while this command is active. The parent waits for all
lanes; launch it with nohup/setsid to outlive a terminal session. Reusing the
same output root without an appropriate resume is refused by experiments.py.

```bash
source configs/remote/dsba3.env
bash scripts/run_experiments.sh --backend both \
  --configs configs/methods/tail_contrast_D3.yaml \
    configs/methods/family_balanced_paired_L3C.yaml \
    configs/methods/reference_matching_R0.yaml \
    configs/methods/reference_matching_R1.yaml \
    configs/methods/reference_matching_R2.yaml \
    configs/methods/masked_prediction_P0.yaml \
    configs/methods/masked_prediction_P1.yaml \
    configs/methods/masked_prediction_P2.yaml \
  --labels D3 L3C R0 R1 R2 P0 P1 P2 \
  --local-device GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b \
  --local-objects can fabric \
  --remote-devices GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f \
    GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae \
    GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 \
    GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 \
  --remote-objects fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --data-root /data/mvtec_ad_2 \
  --remote-data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
  --output-root /results/reference_prediction_20260914 \
  --log-dir results/experiment_launch/reference_prediction_20260914 \
  --timeout 14400 --workers 1 --split test_public_seen_development
```

Fixed placement: localGPU0 can/fabric; dsba3GPU0 fruit_jelly/wallplugs;
dsba3GPU1 rice/walnuts; dsba3GPU2 sheet_metal; dsba3GPU3 vial.
Remote /results is the existing symlink to /workspace/results_local_compatible,
not a new mount. Local /results maps to results/local_runs/bcma_docker_20260911.
Quantitative outputs use local/lane_0 and remote/lane_0..3 underneath this run.
Training logs remain in original run directories; host launch logs contain
local progress and the remote execution-log pointer. Do not pull dense maps,
checkpoints or full remote logs by default.
