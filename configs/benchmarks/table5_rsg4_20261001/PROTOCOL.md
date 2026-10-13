# Table 5 RSG4 cross-dataset evaluation — 2026-10-01

- Hypothesis: RSG4 retains competitive image-level and full-range pixel AUROC on MVTec AD and VisA.
- Candidate: canonical RSG4 (`fusion_factorial` R1, tail-scale residual fusion, component quantile, query noise guard k=9 routing lowres); frozen DINOv2 ViT-L/14, layers 5/11/23, resolutions 448/672/896/1120, bank 256 tokens/source.
- Comparison: reuse E9 seed-0 support images per class and shot exactly; no fitting/prediction labels. Query split/evaluator remains the E9 test split. This is development evidence.
- Scope: full15 MVTec AD and full12 VisA, shots 1/2/4/8/16, seed 0 first. MVTec AD has E9 artifact-backed support draws for shots 1/2/4/8; the E9 MVTec 16-shot draw is absent from pulled artifacts, so run 16-shot only as a separately generated, explicitly unpaired support draw after paired scopes finish. VisA has E9 draws for all five shots. Metrics: `mean_image_auroc` and `seg_AUROC_1.00` (full FPR), class macro, ×100.
- Execution: dsba3 existing `hun_fsad_tta_012` container, only after checking assigned GPU availability and matching current RSG4 source snapshot. Start with MVTec AD seed0 1-shot pilot; outputs maps/weights off.
- Record per-class support identity/content hashes, query/mask fingerprints, code and config hashes. Collect through scripts/experiments.py and canonical ledger.

## Execution notes — 2026-10-01

- Runtime compatibility: the shared segmentation pipeline rejects `eval_clf`; image AUROC is still computed from query anomaly scores, so all Table 5 configs set `eval_clf: false`. The MVTec AD and VisA acquisition-diagnostic parser is AD2-specific, so `fusion_diagnostic` and `condition_strata` are disabled; neither contributes to the requested metrics.
- Local snapshot activation (not a git commit; no push): `aec3c2904503b02358391517ce7739245fa83f91ec310fea52bf649f4aec1229` was the first activation, followed by corrected config releases `c8ae2d23abe0d52c548a7c745d2981f85e51279029cc58810acac310b887bccc` (disable `eval_clf`) and `3b31cb177331305024c28e7e36479d3b84b0c1c5809b8445c2353c4c869921c9` (disable acquisition diagnostics). The latter is the source snapshot used for successful runs.
- Two failed pilot attempts are retained remotely: first rejected `eval_clf: true`; second rejected `fusion_diagnostic: true` on MVTec AD numeric filenames. Both failed before metrics were produced.
- Successful pilot path: `results_remote/table5_rsg4_20261001/mvtec_seed0_s01_pilot_v3`, one MVTec toothbrush class, GPU UUID `GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f`. Config SHA-256: `4782520e075b20ccd00723147d3312bb9a93cf3489e7350413d96823adecbf5b`.
- Pilot support, query and ground-truth hashes exactly match the corresponding E9 toothbrush run. Pilot metrics are I-AUROC 88.6111%, full-range pixel AUROC 96.3419%, AUPRO@0.05 68.5447%. Pilot is not included as a separate collected macro run; full15 queue is the reportable 1-shot scope.
- The generic `pull` over all remote results encounters pre-existing root-owned files. Pullback succeeds when `REMOTE_RESULTS_DIR` is narrowed to `/home/hunim/Volume/FMAD-OOD-remote/results_remote/table5_rsg4_20261001`.
- MVTec AD 16-shot support was generated only after the paired 1/2/4/8-shot runs. It uses the pipeline's deterministic policy (`sorted(train/good)` then `numpy.default_rng(0).choice(..., size=16, replace=False)`) and is explicitly unpaired with E9. Manifest SHA-256: `00d94f6c849030ae9636797e3ffab5f633ef8ffc79df64599327cfb5f9a11701`; config SHA-256: `a9750a274f7872c64bce993aa5b776e771e9cb6711029bf134c45113ef4cff94`.
