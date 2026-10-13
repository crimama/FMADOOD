# Outline v13 public comparison, 2026-09-29

Hypothesis: RSG4's AU-PRO lead over published few-shot methods persists across
1/2/4/8/16 normal support images and acquisition conditions. This is a
comparison of whole methods, not a method-only ablation. RSG4 code, settings,
and completed metric identities are frozen. Missing fixed-threshold F1 may
require a separate maps-only inference under a new result identity.

- Population: MVTec AD 2 full eight classes, `test_public`, previously seen in
  development. Query/mask split and streaming evaluator are shared. Query labels
  and acquisition labels enter evaluation only.
- Support: seed 0 uses `VOID_support_{1,2,4,8,16}.json`; seeds 1 and 2 use the
  existing E1 2/16-shot manifests, with nested 1/4/8-shot prefixes of the
  16-shot manifest. Verify the 2-shot prefixes before execution. No extra
  normal train images may enter fitting or threshold selection.
- Primary: macro AU-PRO@FPR 0.05, higher is better. Secondary: macro pixel
  pAUROC@0.05, I-AUROC using maximum of the final score map, and acquisition
  strata (regular/shift_1 full8; shift_2/3 matched full7). Seed summaries use
  mean and sample SD. Keep missing and failed cells distinct.
- First new comparator: AnomalyDINO, official repository
  `dammsi/AnomalyDINO` commit `b9d1c2648e3a5247437d4d953d907a8f3d994457`.
  The adapter follows the official final-layer DINOv2 patch features, unit
  normalization, FAISS 1-NN squared L2 / 2, bilinear resize and sigma-4
  Gaussian. Shared support manifests require no rotation or masking. The
  comparison uses DINOv2 ViT-L/14 at 672 pixels, rather than the official
  script's default ViT-S/14 at 448; the changed backbone/resolution is explicit.
  I-AUROC uses map maximum per the manuscript contract rather than the
  official script's top-1%-patch image statistic.
  The shared AD 2 dataset contract overrides the requested 672-pixel short
  edge for `sheet_metal` to 448; record the actual per-object resolution in
  each run manifest. This matches the existing shared comparator pipeline.
- Execution order: local single-class validity pilot; dsba3 seed-0 16-shot
  full8 and collection; remaining AnomalyDINO shots/seeds sequentially;
  remaining published methods after official-code and paper fidelity audits.
  Local RTX 4090 and four assigned dsba3 A6000 GPUs may process independent
  object cells. Use existing container and `scripts/experiments.py run/collect`.
- Budget: 15 AnomalyDINO configurations × 8 objects, then each accepted method
  at the same 15 × 8 scope. Stop a method on invalid provenance or material
  disagreement with its paper and repair before adding further cells.
- SegF1/ClassF1 require a single support-only threshold rule applied to all
  methods; leave these cells empty until defined and run. ReFP-AD is a
  full-data reference and must remain separate from few-shot ranking.
- Legacy FastRecon and RegAD modules are simplified project adaptations. Their
  existing results retain provenance but are not official paper replications.
  Do not expand their shot/seed rows until official code or a paper-faithful
  implementation is established.
