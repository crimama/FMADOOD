# GuardedMetric: E1–E3 original-space metric ablation

Locked before execution on 2026-09-11. This is a post-hoc method revision of
TailContrast D3, evaluated on **seen development**, not a sealed holdout.
Hypothesis: a bounded, direction-selective metric can retain D3's improvements
while reducing the vial/fruit_jelly regressions. This is not an asserted result.

| Config | Original-space low-rank metric | Nuisance protection | Weak anomaly / hard positives |
|---|---|---|---|
| guarded_metric_E1.yaml | on | off | off |
| guarded_metric_E2.yaml | on | on | off |
| guarded_metric_E3.yaml | on | on | on |

## Scoring and the distance guarantee

Use the original normalized frozen features z and raw bank r. Let e=z-r,
P=qr(W), a=0.3*tanh(logits), a+=(a+abs(a))/2, a-=(abs(a)-a)/2. Score the full
bank with the minimum of:

`d(z,r) = 0.5 e^T [D(z) + P diag(a+) P^T - N diag(a-) N^T] e`.

D(z) is the original BCMA contextual diagonal in [0.8,1.2], computed from
unadapted features. E1 uses N=P, equivalent to a signed rank-16 correction.
There is no feature adapter or re-normalization after a trainable transform.
All variants satisfy `0.5*s_raw_NN <= s <= 1.5*s_raw_NN` (up to floating-point
roundoff), also after the fixed bilinear multi-resolution fusion. This is a
distance-distortion bound, **not** a ranking/performance guarantee.

W has 1024x16 parameters and logits has 16. Combined with BCMA this gives
84,016 trainable parameters, versus D3's 100,384. E1/E2/E3 parameter counts
match. W starts random; logits start zero, hence the raw NN scoring identity.
The backbone is frozen. Exact detached NN selection and differentiable selected
distances use the same gradient-replay fitter as D1–D3.

## E2: support-only evidence and protected contraction

Before the 64 optimizer steps, form uncentered second moments of aligned
normal-shift differences and synthetic-minus-clean differences. Reuse clean
support features, then extract one photometric view and both existing strong
synthetic families for each of the 16 sources at both resolutions. A separate
seed+100003 NumPy generator isolates this from the main bank/training RNG.
At most 128 tokens per evidence group are sampled. Defect groups are averaged
equally, independent of mask area. Token eligibility uses synthetic-mask grid
coverage >=0.25, falling back to any nonzero coverage for very small masks.

Q is the top rank-16 eigenspace of normal-shift second moments C_T. Its gates
are `g_i=(q_i^T C_T q_i)/(q_i^T(C_T+C_A)q_i+1e-8)`. Use
`N=Q diag(sqrt(g)) Q^T P`. Only the **negative** (distance-contracting)
low-rank term is restricted to these directions; the positive term remains
available to strengthen anomaly evidence. This makes contraction protection
explicit without changing the number of trainable parameters between stages.
Outside span(Q), E2 adds no contraction beyond the original bounded diagonal.
Zero shift evidence grants zero additional contraction. This is not a
confidence/correctness estimator or a guarantee about unseen real defects.

Only Q/g survive as small model buffers; covariance matrices are temporary.
This stage adds 96 frozen-backbone evidence calls per class (16x2x3), recorded
as method-fitting overhead, not hidden as baseline-equivalent compute. Both
families/resolutions cover every source. E2/E3 evidence is identical; E3 changes
only the subsequent synthetic training, not the evidence used for protection.

## E3: finite-budget weak synthetic discrimination

Use the same rectangle/scratch masks and main RNG as E1/E2, blending anomaly
images toward their clean source by strengths [0.35,0.65,1.0]. Strength index is
`(source+cycle)%3`, cycling across each source's visits. The same photometric
transform is applied to clean and anomalous images as in D3. Positive budget
stays 128 per view: half sampled from its lowest-score quartile and half
uniformly from the anomaly mask. Normal-tail sampling stays unchanged.
Weak synthesis and hard-positive sampling are **one coupled training component**;
this ablation does not isolate their individual effects.

Loss remains `L_rank + L_normal_pair + L_defect_contrast`. Baseline keep loss
is diagnostic only. In particular, these experiments do not isolate paired
consistency's effect with the D3 adapter; a rank-only D3 control is not included
in this 24-run authorization. Log losses, raw score drift, coefficient extrema,
strength, and evidence gate summaries. Never accumulate dense maps to disk.

## Locked comparison and stop conditions

- Existing D3 full8 reference: AUPRO@0.05 48.96953, pAUROC@0.05 77.88021 (x100).
  Reuse completed results; do not re-evaluate D3. Validate D0–D3 tiny-backbone
  histories/predictions unchanged after introducing method extension hooks.
- All new stages: normal random16, seed0, DINOv2 ViT-L/14 layer23, 448+672,
  bank256/source, full-bank NN, source-exclusion during fitting, 64 steps,
  lr0.001, margin0.05, no masking/rotation. Common support/query/evaluation
  pipeline stays fixed. Compare content hashes and original artifacts to D3;
  code hashes necessarily change with the newly registered method.
- Primary AUPRO@0.05; secondary partial pixel AUROC@0.05. F1 oracle is a
  diagnostic. All eight MVTec AD2 test_public classes are seen development.
- One sequential E1 -> E2 -> E3 run: 24 class runs, no seed/search repeats.
  <=3600 seconds per class; <=24h queue upper bound, expected roughly 1–2h
  based on previous timing, not a measured runtime for these new methods.
- Stop the queue on runtime error, nonfinite optimization/predictions or
  timeout. Preserve failed evidence; do not treat a crash as a method failure.
- Report macro and all class deltas. Improvement must be evaluated alongside
  vial/fruit_jelly regressions and preservation of wallplugs/can gains. No
  best-class switching, oracle deployment thresholds, or held-out claims.
- Existing container `fsad-tta-local-bcma`, same RTX4090 UUID
  `GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b`. Maps/weights off. No environment
  replacement, downloads, standalone pipeline or new per-method scripts.
- Use `scripts/experiments.py run` through `run.py -> fmad.cli -> pipeline`;
  host `collect --watch` updates `results/comparisons.md` and its existing CSV.

## Failure conditions and prior-work boundary

Pre-launch validation: 65 focused tests passed in the existing container,
including CUDA eigensolver/backward, explicit dense-distance/gradient agreement,
the raw-distance bounds, protected complement, exact gradient replay, evidence
RNG isolation, frozen backbone/state reset, CLI configuration, output policy,
streaming contracts, and generic queue/collector behavior. A separate before/
after tiny-backbone check confirmed byte-identical D0–D3 predictions and training
histories after the extension hooks. No real-data pilot/search was run.

E1 may cap both harmful drift and useful D3 gains. E2 may suppress too little
when strong synthetic defects dominate its gate, or suppress real defects that
share photometric shift directions. E3 can upweight ambiguous synthetic
boundaries and increase false positives. A favorable metric alone does not
prove the proposed mechanism or robustness to other shifts/seeds.

Feature modulation overlaps [ReConPatch, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.html).
Weak synthesis overlaps [GLASS, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08382.pdf).
Tail-focused ranking relates to [When AUC meets DRO, ICML 2022](https://proceedings.mlr.press/v162/zhu22g.html),
but is not exact AUPRO optimization. No standalone novelty claim is made for
these components. Prior-work sanity check was performed in the preceding D3
analysis; this execution tests the scoped structural revision.
