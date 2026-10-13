# Table 5 blank-cell completion - 2026-10-07

Authorized by the user on 2026-10-07: fill the empty cells of Outline v16.3 Table 5
(MVTec AD / VisA few-shot; I-AUROC / full-FPR pixel AUROC).

Scope (seed 0, paired E9 support manifests from table5_rsg4_20261001; MVTec AD
16-shot uses the deterministic seed-0 draw `mvtec_seed0_s16_generated.json`, the
same rule E9 applied with no manifest):
- New columns at the Table 4 1120 settings, copied from the Table 4 configs:
  AnomalyDINO (DINOv2 ViT-L/14 layer 23, 1120), RAD (DINOv3 ViT-B/16, 1280 resize
  -> 1120 crop, global multi-category bank over all declared classes), SubspaceAD
  (DINOv2-registers ViT-G/14, layers 22-28, 1120). MVTec AD full15 and VisA full12,
  shots 1/2/4/8/16.
- RegAD official (released 2-shot MVTec bottle STN checkpoint, 224) on VisA only,
  shots 1/2/4/8/16. MVTec AD is excluded: the released checkpoint was trained on
  the other MVTec AD categories, so MVTec AD targets would be seen categories.
- Blank MVTec AD 16-shot row for existing columns at their own E9 settings:
  PatchCore, SuperAD and the FastRecon project adaptation (effective input 448,
  copied from E9 8-shot configs), and SCARF (Table 5 config, FP32 fusion_factorial)
  with the top-1% mean image score now used in Tables 4/5.
- 39 configs, 444 class runs.

Code: isolated source = release 6d01d1873437e5cd2d38ce7b86fb50998b95794fc17904951d1a7aa9b85cfb6d
with the audited image-score overlay (scarf_image_score_20261007) plus one RAD
change: support_objects may be any declared unique object list whose manifest keys
match exactly (previously hardcoded to the eight AD2 objects). RAD scoring is
unchanged.

Runtimes follow the source runs: AnomalyDINO, RegAD, PatchCore, SuperAD, FastRecon
and SCARF use container python3 (3.10, torch 2.1.2+cu118); RAD and SubspaceAD use
the isolated Python 3.11 / torch 2.6.0+cu124 runtime with the local DINOv3
checkpoint, offline.

Image score: each comparator keeps its native image score (as in Table 4). Labels
and GT are evaluator-only. MVTec AD and VisA test sets are development data already
used for analysis. Maps and weights off. Primary reporting: class macro
image_AUROC and seg_AUROC_1.00 (x100); AUPRO@0.05 and pAUROC@0.05 also recorded.

Execution: dsba3 existing container, GPU UUIDs 0-3, disjoint class shards; per
GPU four sequential steps (MVTec py310, MVTec py311, VisA py310, VisA py311), each
through scripts/experiments.py run. Class timeout 21600 s. Remote logs retained.
