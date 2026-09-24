# J1 — hard-pair tail risk (2026-09-12)

## Evidence and single intervention

The completed H1/I1/I2/I3 comparison is in `results/comparisons.md`.
I2's ranking loss was zero at 306/512 steps and its mean support P/b fell,
but full8 AUPRO fell from H1 47.41448 to 46.95461 (x100). This does not prove
the cause: easy synthetic uplift and removal of hard-normal competition are
competing explanations. J1 retains H1's cross-location normal/defect competition
and changes only its empirical loss reduction, not the synthetic generator.

For each synthetic view v, use the exact H1 sampling and RNG order:

- P_v: up to 128 uniformly sampled synthetic-mask positions.
- N_v: H1's tail/random samples from clean, shifted-clean, and unmodified
  pixels of this synthetic view. Tail fraction remains 0.05.
- h_ij = relu(0.05 - s_v(P_i) + s(N_j)).
- k = ceil(0.25 * number_of_pairs).
- J1 ranking = mean over the two views of mean(TopK(h_v, k)).

H1 instead averages every sampled h_ij, including satisfied zero-loss pairs.
Selection is detached; gradients flow to both selected synthetic and normal
scores through the unchanged exact four-view VJP replay. The sampled pool is
still H1's pool: this is not exhaustive hard-example mining across images.
Fixed risk fraction 0.25, no sweep. `tail_risk: false` delegates exactly to H1;
`tail_risk_fraction: 1.0` gives the full-mean ranking within floating tolerance.
The latter is a unit-tested mathematical limit, not an additional GPU run.

All H1 preservation/controller, shift consistency, paired contrast, reference
search, backbone, adapter, fitting loop, inference and evaluator remain fixed.
No I1/I2/I3 components are combined. The same 100,384 parameters are trainable;
the backbone remains frozen. `src/fmad/methods/tail_risk.py` is the only new
method and `tail_risk_J1.yaml` differs from H1 only in name and its switch.

## Expected effects and limits

Hypothesis: prioritizing violating normal/defect pairs improves separation
where easy synthetic pairs dilute the average. It predicts a different gradient
allocation with no new feature extraction or additional image-level forward.
It does not guarantee that the fixed sampled positives contain realistic defects.

Log scalar sampled_pair_mean_loss, sampled_pair_tail_loss,
sampled_pair_active_fraction and tail_risk_fraction alongside existing losses
and P/b. Per-view pair storage is bounded by 128 x 768; no across-step dense
map, prediction, gradient, or reference-index accumulation is added.

Failure conditions: poorly labelled synthetic boundaries dominate the hard
tail; the harder objective damages normal preservation; support tails fail to
cover real shifted normals; harder ranking still does not transfer to defects.
The loss changes both selection and effective scale (at most 4x the full mean
for this retained fraction), so improvement alone would not isolate mining
from reweighting. A scale-matched control is a future ablation only if useful.
No exact pAUC/AUPRO equivalence, causal proof, novelty, or guaranteed gain claim.

## Brief prior-work check

- [OHEM, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html)
  selects hard examples in object detection. Hard-example selection is known;
  J1 is a bounded application to this support-only score-adaptation pipeline.
- [When AUC meets DRO, ICML 2022](https://proceedings.mlr.press/v162/zhu22g.html)
  connects partial-AUC surrogate optimization with CVaR/DRO formulations.
  J1's upper tail across sampled pair hinges is not that estimator: pools are
  tail-biased, positives are synthetic, and our primary metric is region AUPRO.

## Locked development comparison

- Rerun H1 then J1, seed0, normal16, 448+672, 64 steps,
  `test_public_seen_development`, same shared CLI and support selection.
- Primary full8 macro AUPRO@0.05; pAUROC secondary; F1 oracle diagnostic.
- Local RTX4090 GPU0: can, fabric, fruit_jelly, rice, sheet_metal, walnuts.
  dsba3 physical GPU2: vial, wallplugs. Two class workers per GPU. A given
  class's H1/J1 stays on the same GPU and environment. Other GPUs excluded.
- Budget: 16 class runs total, timeout 3600 seconds/class, no sweep/retry
  expansion. Use `scripts/experiments.py run` and collect to the existing ledger.
- Stop on execution errors. A primary-metric loss means no performance-gain
  claim; partial or single-class gains cannot override full8. One seed and a
  seen set do not establish significance or sealed-final performance.
- Verify support/query/GT identities, config/source/backbone/runtime and metric
  contract before comparing. Tests must prove disabled parity, gradient direction,
  scalar-only diagnostics, exact replay, and frozen-backbone/parameter parity.
- Reuse existing containers. Local reboot outside this task loaded 580.178.04;
  both new H1/J1 runs use this matched driver. No comparison to old H1 replaces
  the new control. Do not restart the host or recreate containers for this run.
- Dense maps/weights off. Remote logs and manifests stay remote; retrieve compact
  metrics only. Runtime errors invalidate execution, not the method family.
