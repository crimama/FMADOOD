# Coherent matching development pilot — 2026-09-19

Recorded before evaluation. User authorized experiments on local and dsba3.

## Hypothesis and boundary

Independent appearance matches can explain a locally inconsistent composition.
CM restricts the center and four axis neighbors to one actual support window,
then measures the cost gap against independent choices from the same five
appearance-selected candidate windows. This is local shared correspondence,
not the previous LR sorted edge descriptor, global registration, a graph neural
network, or a claim that the backbone's geometry has been repaired.

Frozen DINOv2 ViT-L/14 layers 5/11/23, RSG comparator, 16 normal supports,
resolutions 448/672/896/1120, four 256-token/source banks, unchanged preprocessing,
support sampling, normalization, fusion, interpolation and CQ. Structural
matching uses layer 11, five center-appearance candidates, cross-shaped five
positions, replicate boundaries, fp16 window storage with fp32 costs. No
trainable parameters. Orientation is fixed: no rotation invariance is claimed.
Full normal feature windows are additional method state, not additional data.

Per bank/resolution, whole-source-excluded normal matches calibrate gap q95
(floor 1e-4). Penalty = .25 max(gap/q95-1,0); thus unstable normal correspondence
weakens the constraint at bank level, not a learned per-location reliability.
Only support data calibrate the method; test labels never enter fit/predict.

## Four fixed arms and evaluation budget

- CM_A: unchanged RSG comparator.
- CM_B: independent relaxation, zero constraint gap; exact baseline control.
- CM_C: shared-window correspondence gap with normal q95 calibration.
- CM_D: appearance-only nearest-distance empirical normal rank transported to
  the sorted normal CM_C penalty distribution. It matches the normal calibration
  marginal (up to ties), NOT the unseen query marginal; out-of-range values
  saturate. Consequently C>D alone does not prove a geometry mechanism.

One seed (0), all eight AD2 objects, test_public_seen_development: 32 evaluations.
This is repeatedly seen development, not untouched final evaluation. No weight,
neighborhood, threshold or candidate-count sweep. Metrics higher-is-better:
AUPRO@0.05 primary, pAUROC@0.05 safeguard, common float16 streaming contract.
Paired support manifest/config/backbone provenance must agree within each class.
Local owns can/fabric; dsba3 owns the remaining six classes on its existing four
exposed GPUs. Cross-host torch versions differ, but each class's four arms share
one environment. Maps and weights off; compact results and training logs kept.

Unit/synthetic diagnostic first: normal coherent window has zero gap; a spliced
window retains independently matchable features but has positive gap. This is
an operator check, NOT evidence of real-data normal invariance. Real normal
support source-excluded calibration is part of fit and is reported via metadata.

Promotion gate: C-A >=1 AUPRO percentage point, pAUROC >=-0.5pp; require at least
four improving classes and positive macro gain after removing the largest-gain
class. Also B-A <=1e-5 absolute metric tolerance. If gate fails, stop this
candidate with no default change. If it passes, inspect controls before planning
replication; the present execution budget is 32, no automatic larger sweep.
Execution failures are not negative method evidence; bounded same-config retry
is allowed after diagnosis. No commits/pushes or container replacement.

## Novelty sanity check and limits

UniVAD (CVPR2025) already combines component-aware patch matching and graph
component modeling; generic structural FSAD is not novel. Prior project R2
uses a pooled ring-context retrieval key and cross-layer shared identity; CG
guards translation-adapted reference choices. CM instead compares jointly
selected local windows with an explicit independent relaxation. This difference
is an experimental hypothesis, not an established literature novelty claim.
Primary page: https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html
Previously checked abstract; current re-fetch returned HTTP403. No new claim is
inferred from inaccessible full text. Broader nearest-neighbor/window matching
prior art remains to be assessed if performance merits further work.

## Execution validation (before pilot metrics)

- 63 coherent/reference tests passed; only a nonfatal pytest-cache permission warning.
- Real DINOv2 smoke: two normal can supports, separate third normal image,
  resolution448, bank32, one bag. A/B map maximum absolute difference0;
  C/D finite. C normal active180/2208 tokens, raw penalty sum58.59314,
  tau0.04861762; C/D maximum map increments0.35418832/0.03855413. This
  immediately confirms that normal-distribution calibration does not imply
  exact held-out query magnitude matching. No parameters were adjusted.
- Backbone state SHA256 `59a755e385188971c57f52a38773028b508722c804ce6c9279d1dc2ac14ccbe8`;
  repository SHA256 `8e82773220ebd38c121c4b5d0f5c57b2ee009c82cf76c26efe306513379d2a18`.
- Remote local-mode immutable snapshot activated:
  `2630cd59e4b92a2b256c3535f4ed476a4bf11829d4c85d25d99f168623dcc112`.
- Local artifacts: `results/local_runs/bcma_docker_20260911/coherent_matching_20260919/lane_0`.
  Remote artifacts: `/workspace/results_remote/coherent_matching_20260919/lane_0..3`.
  Launch: four `configs/methods/CM_{A,B,C,D}.yaml` in that order through
  `scripts/run_experiments.sh` / `scripts/experiments.py run`, per-object
  timeout3600, split `test_public_seen_development`; exact commands in status.json.
