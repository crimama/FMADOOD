# L1 / L2 / L3 — method-first matched development comparisons

Date: 2026-09-12. User authorized implementing all three designs and running
them sequentially. They are independent candidates, NOT a cumulative stack.
No novelty, performance, or unseen-domain guarantee is claimed before results.

## Evidence and hypothesis

The latest matched K1/K3 evidence is indexed in `results/comparisons.md`.
K3 lowers its support preservation surrogate but does not improve full8
AUPRO or recover vial. This motivates three competing, unproven explanations:
operating-region objective mismatch, failure to distinguish nuisance from
defect correction, and synthetic-specific rather than transferable updates.
Keep historical results; do not treat lower training loss as method success.

## L1: low-FPR operating-region objective

`operating_region.py`, switch `operating_region`; off is exactly K1.
K1's frozen source-excluded synthesis, score model, preservation/controller,
four views, optimizer and inference remain fixed. Replace ranking only:

    N = concat(full clean scores, full shifted-clean scores)
    tau_u = quantile(N, 1-u), u in {0.01, 0.025, 0.05}
    L1_rank = mean_views mean_regions mean_u mean_pixels_in_region
              T * softplus((tau_u + margin - score) / T)

Use existing margin 0.05, fixed T=0.05, equal threshold and region weights.
T multiplication retains soft-hinge score units (no arbitrary 1/T loss gain).
Regions are 8-connected components of the KNOWN synthetic mask. Quantiles
have live gradients; no tail-biased negative subsampling and no new RNG draws.
This is a support-derived synthetic surrogate, not an exact AUPRO estimator
or query FPR guarantee. Sparse quantile gradients and synthetic-real mismatch
are failure cases. No extra parameters, backbone calls or inference branch.

## L2: same-shift normal restoration

`shift_restore.py`, switch `shift_restore`; off is exactly K1.
Keep K1 synthetic strength selection and the same four image views. A rank16
residual normal-feature restorer R replaces the optimized contextual metric
and adapted-nearest-neighbor score. The backbone remains frozen.

    z0=f(x), zt=f(Tx), za=f(Ax), zta=f(T(Ax))
    L2:  R(z0)->z0, R(zt)->zt, R(za)->z0, R(zta)->zt
    L2C: R(z0)->z0, R(zt)->z0, R(za)->z0, R(zta)->z0

All targets are detached. L2C and L2 share inputs, initialization, exposure,
loss weights, model and inference; only shifted targets change. Normal and
synthetic reconstruction terms have weight 1 each. Feature MSE is averaged
per normal view; synthetic foreground/background errors have equal mass,
using area-interpolated fractional synthetic-mask coverage on each token grid.
Average views and resolutions; no global shift subspace removal is assumed.

    inference score(y,p)=||f(y)_p-R(f(y))_p||_2^2

Resolution interpolation/mean and the evaluator stay unchanged. No paired
query normal is available or required. Trainable restoration parameters are
32768 for D=1024, rank16, versus K1's100384; the unused contextual metric is
not optimized. Frozen banks remain only for K1 synthetic teacher selection.
This is an explicit objective/score/parameter change, not a method-only loss
control. Copying real defects, synthetic-only restoration, and unseen shifts
producing normal residuals are key failure conditions. Teacher reconstruction
error is not itself evidence of real-defect detection.

## L3: exact cross-environment meta-update

`transfer_update.py`, switch `transfer_update`; off is exactly K1.
`cross_environment_meta:false` is the REQUIRED L3C pooled-ERM control.
Each step has distinct sources a=step%N, b=(step+1)%N, opposite synthetic
families, separately sampled photometric transforms, and a shared frozen bank
excluding BOTH source identities. At least3 supports are required. K1 teacher
strength selection uses that bank and caches thresholds by (source, excluded
sources). Episode descendants stay with their source. The same scheduling,
synthesis and random draws are used by L3C and L3.

    La, Lb = K1 rank + shift + contrast + detached_gain * preservation
    theta' = theta - 0.01 * grad_theta La(theta)
    L3  = (La(theta) + Lb(theta')) / 2
    L3C = (La(theta) + Lb(theta)) / 2

Both episode controller gains are evaluated at theta and held detached during
the unroll. Rank sampling RNG is replayed, not consumed a second time by the
virtual evaluation. The functional inner SGD is differentiated exactly,
including its Hessian-vector contribution. Real parameters and Adam moments
are not changed by it. One real AdamW update/step follows gradient clipping5.
The existing detached-score VJP replay is NOT used for this higher-order fit.
The scorer still uses detached NN argmin with differentiable selected values;
derivatives are local/piecewise and need not apply across reference switches.

Trainable parameters and inference match K1 (100384), but training uses two
sources/eight views per step, altered training-bank exclusion, and extra
scoring/second-order work. Do not attribute L3-K1 changes solely to meta-learning:
use L3-L3C. Virtual domains are training data, NOT a sealed held-out set.
Shared synthetic shortcuts, too little source diversity, unstable second-order
updates, or negligible virtual displacement can defeat the hypothesis.

## Focused related-work check

- [When AUC meets DRO, ICML2022](https://proceedings.mlr.press/v162/zhu22g.html):
  pAUC/DRO optimization is known; L1's regional synthetic surrogate is not that
  paper's exact estimator or a novel general pAUC optimizer.
- [DeSTSeg, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html):
  synthetic-corruption denoising to normal teacher features is known. L2C/L2
  isolates canonical versus same-shift targets in this few-shot scorer.
- [MLDG, AAAI2018](https://ojs.aaai.org/index.php/AAAI/article/view/11596):
  optimizing cross-domain performance after a virtual training step is known.
  L3 applies it to support/synthetic environments, without real-defect labels.

## Locked execution and interpretation

- Sequence: K1 -> L1 -> L2C -> L2 -> L3C -> L3; six configs, eight classes each.
- Seed0, same normal16 support selection/content, frozen DINOv2 ViT-L14 layer23,
 448+672,64 actual optimizer updates, no_mask_no_rotation, seen TESTpublic.
- Primary: full8 macro AUPRO@0.05; secondary pAUROC@0.05; oracle F1 diagnostic.
  Primary matched contrasts: L1-K1, L2-L2C, L3-L3C. Also report every candidate
  against K1 with declared method/parameter/training-exposure differences.
- Local existing fsad-tta-local-bcma, GPU0 UUID
  GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b, observed450W: can/fabric/fruit_jelly/
  rice/sheet_metal/walnuts. dsba3 existing hun_fsad_tta_012, physical GPU2 UUID
  GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639: vial/wallplugs. One worker/GPU;
  stages sequential within each queue. No global inter-server stage barrier.
- Budget48 class runs total;3600s/class upper bound; no additional seeds,
  sweep, automatic method revision or unbounded retries. Stop queue on errors.
- Reuse scripts/experiments.py run + existing CLI. No method-specific queue or
  ledger scripts. Local code tested before immutable local-snapshot update via
  scripts/run_remote.sh update-code; no git push/container provisioning.
- Save metrics,scalar logs,configs and provenance; maps and weights OFF.
  No cross-step dense accumulation. Remote logs stay remote; recover compact
  metrics/config/manifests only. Collect to the canonical Markdown/CSV ledger.
- Check disabled K1 parity INCLUDING weak synthetic selection, live normal
  gradients, exact functional meta-gradient/HVP, frozen backbone, matched
  control inputs, source exclusion, module registration and output policy.
- Promote only after valid full8 primary results. A matched-control gain without
  beating K1 is component evidence, not a new best method. Single seed on a
  seen set does not establish significance or paper-ready generalization.
