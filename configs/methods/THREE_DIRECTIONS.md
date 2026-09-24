# I1 / I2 / I3 — independent H1 extensions (2026-09-12)

## Problem and evidence boundary

Completed H1/G1 comparison: seed0, normal16, 448+672, 8 seen-development
classes, mixed hardware matched within each class. Full8 AUPRO@0.05 x100:
G1 46.71464, H1 47.42062 (+0.70598pp). H1 improves vial +8.25242pp,
but improves only 3/8 classes; can -2.12269pp, fruit_jelly -0.95248pp.
The earlier local4-only result was incomplete, not a full-method verdict.
Vial final support P/b changes from 4.169 to 0.677. This is consistent with
better preservation, not proof that it caused the metric improvement.
D3 48.96953 remains a historical performance reference, not a current control.
See `results/comparisons.md` and its linked metric artifacts.

Three distinct questions, not three controller hyperparameters:

1. Does preserving every normal score obstruct useful tail correction?
2. Does absolute cross-location synthetic ranking optimize texture/score
   offsets instead of the local defect response under photometric shift?
3. Does adapting the nearest-reference identity along with its distance let
   the metric reduce anomaly scores through reference switching?

The first has indirect evidence from class tradeoffs. The second and third
are code-supported possibilities, not established root causes. H1 preserves
full-map MSE plus standardized score relations; `tail_ranking` compares
synthetic pixels with other locations; `BCMA.score` searches in the adapted
metric. These are the exact intervention boundaries below.

## Independent designs

All start from H1, not from the preceding candidate. No combination or sweep.
Let c,s,a,as denote clean, shifted-clean, synthetic, shifted-synthetic scores;
t,ts are frozen baseline scores on support clean/shift views. M is the known
synthetic mask, and b is the existing H1 support-derived preservation budget.
Controller remains lambda=stopgrad(clip(P/b,0.1,1.0)). Backbone frozen; the
same rank-16 adapter and contextual diagonal metric are the only parameters.

### I1 — selective preservation

`src/fmad/methods/selective_preservation.py`, config
`selective_preservation_I1.yaml`, switch `selective_preservation`.

B={p: t(p)<=quantile(t,0.95)} (include ties). Replace only preservation:

`P_B = mean_B((c-t)^2) + 0.25 * SmoothL1(z(c_B),z(t_B))`.

H1's ranking, shift/contrast losses, budget and inference remain unchanged.
The same subset is used for controller context and optimized preservation.
Selection is detached and uses normal support only. Predict lower false
positive tail without losing normal bulk geometry; record preserved_fraction.
Failure: normal tail contains crucial semantic boundaries, so releasing it
may suppress genuine defects or lose H1's vial recovery. Quantile ties/flat
teachers select all; no arbitrary tail is released. Extra cost: teacher
quantile/temporary mask, no trainable parameter or additional backbone pass.

### I2 — paired local margin

`src/fmad/methods/paired_margin.py`, config `paired_margin_I2.yaml`, switch
`paired_margin`. Replace only absolute cross-location ranking:

`m_p = 0.05 + stopgrad(min(|ts_p-t_p|,0.05))`

`L_rank = mean_{p in sampled M} relu(m_p-min(a_p-c_p,as_p-s_p))`.

Use matched positions and at most 128 uniformly sampled synthetic pixels;
the worse of the two views must show a local defect increment. H1's existing
preservation, shift consistency and contrast stay fixed. No confidence gate
or true-anomaly label. The teacher displacement is discarded after this step.
Predict improved defect-response stability, but absolute normal-tail false
positive suppression is no longer directly optimized by ranking. Failure:
local CutPaste/noise uplift may not transfer to real defects, or low uplift
may reflect an unrealistic synthesis. Extra cost negligible, no new forward
pass or parameter. Log paired_margin_mean and paired_uplift_mean.

### I3 — frozen-reference retrieval

`src/fmad/methods/anchored_retrieval.py`, config
`anchored_retrieval_I3.yaml`, switch `anchored_retrieval`.

`j*(x)=argmax_j f(x)^T b_j` in the original normalized frozen feature space.
Then evaluate H1's learned diagonal distance between `A(f(x))` and `A(b_j*)`.
Reference identity is detached; both selected-reference and query adapter
gradients remain. Same rule during fitting and inference. Baseline teacher
remains unadapted. H1's complete objective/controller is unchanged.
Predict fewer score reductions due to reference switching. Failure: the
frozen nearest neighbor is wrong under real shift; locking it prevents a
useful correspondence correction. Chunked original-space search replaces
adapted full-bank search; no NxBxD allocation, no saved indices or extra bank.
Same parameter count; timing is measured, not assumed faster.

## Brief prior-work sanity check (not a novelty audit)

- [CutPaste, CVPR 2021, author copy](https://arxiv.org/abs/2104.04015): learns
  from normal versus synthetic local modifications. This supports the proxy
  training principle, not I2's within-location worst-view margin or its
  transfer to actual defects.
- [Reverse Distillation, CVPR 2022, author copy](https://arxiv.org/abs/2201.10703):
  normal-only teacher/student discrepancy motivates preserving pretrained
  information. I1 changes which normal support scores are constrained;
  it is not a reproduction of reverse distillation's encoder/decoder design.
- [PatchCore, CVPR 2022, author copy](https://arxiv.org/abs/2106.08265): uses
  a representative bank of normal patch features. I3 retains the frozen
  reference selection while learning its distance, unlike a wholly frozen
  bank detector. A new paper-level contribution is not established.

Published venues were confirmed through CVF records; author-hosted arXiv
copies supplied abstracts because direct CVF pages returned HTTP 403. No
newness, state-of-the-art, or guaranteed gain claim follows from this check.

## Locked experiment and stop condition

- Re-run H1 control under the same new source snapshot; then I1, I2, I3.
  This is three separate one-component comparisons, not I1->I2->I3 stacking.
- Same shared CLI/data/support selector/backbone/evaluator; seed0, normal16,
  64 steps, 448+672, `test_public_seen_development`.
- Primary: full8 macro AUPRO@0.05. Secondary: pAUROC@0.05; oracle F1 is
  diagnostic. No sealed-final or variance claim from one seed.
- Before interpreting deltas, match support/query/GT hashes, evaluator,
  shared config, source and runtime within each class/method comparison.
- Local RTX4090: can, fabric, fruit_jelly, rice, sheet_metal, walnuts.
  dsba3 physical GPU 2 only: vial, wallplugs. Both containers use exactly one
  assigned UUID and two concurrent class workers. Each class's four methods
  stay on one device; macro is explicitly mixed-hardware.
- Budget 32 class runs (8 control + 24 candidates), one seed, no sweep;
  timeout 3600s per class, no unbudgeted tuning/retries. Runtime errors mean
  invalid execution, not method-family rejection.
- Maps/checkpoints off. Only one-step temporary maps/features; remote logs
  and provenance stay remote, compact metrics retrieved for the same ledger.
- Each switch off must reproduce H1. Tests check on/off parity, frozen
  backbone, parameter count, gradient direction, exact VJP replay, bounded
  margins, quantile ties, anchored index selection and common config parity.
- No full8 gain means no performance improvement claim. Partial/vial-only
  gains do not override the primary metric. Mechanism attribution additionally
  requires the predicted internal change; performance alone is not proof.
