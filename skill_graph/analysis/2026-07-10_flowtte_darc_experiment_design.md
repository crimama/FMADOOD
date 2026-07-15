# DARC Resolution and Correspondence Experiment Design

Date: 2026-07-10
Status: frozen for implementation

## Claim, Evidence, Boundary, Positioning

- Claim: `can`의 핵심 실패 원인은 672/patch-16에서의 sub-token 희석과
  위치를 버린 global nearest-neighbor이며, detached coarse geometry를
  native-resolution micro scorer로 전달하면 이 원인을 직접 완화할 수 있다.
- Evidence: `can` median defect bbox는 약 `15x13` native pixels이고 현재
  grid-compatible oracle F1은 672에서 `0.355`, 해상도를 높이면
  896/1120/1344/2048에서 `0.465/0.511/0.593/0.707`로 증가한다. 현재 score
  상위권은 anomaly보다 정상 구조에 집중되어 bad GT recall이 약 5.1%다.
- Boundary: AD2 public GT는 이미 설계에 영향을 주었으므로 shadow development
  결과일 뿐 최종 논문 claim set이 아니다. Private labels 또는 별도 untouched
  benchmark가 없으면 final claim gate는 열 수 없다.
- Positioning: novelty 후보는 high resolution, registration, reconstruction
  각각이 아니라 detached coarse-to-micro geometry transfer와 unchanged coarse
  semantic branch의 결합이다.

## Frozen Method

1. Coarse anchor: current 672 H+ `[7,15,23,31]`, layer-normalized mean, DVT
   position-mean `alpha=1.0`, MLP flow, static memory, raw continuous map.
2. Micro extraction: native `512x512` crops, stride `384`; paired resize `512`
   and `1024`; H+ layer 7 only; exact native token-center spacing 16 and 8 pixels.
3. Coordinate convention: realized integer resize dimensions, half-pixel affine
   maps, patch center `16*(index+0.5)-0.5`, right/bottom remainder crop only.
4. Geometry: frozen 672 layer-23 mutual-NN matches, orientation-preserving
   4-DOF similarity RANSAC; minimum 12 inliers, ratio 0.25, scale `[0.8,1.25]`,
   median error 24 resized pixels.
5. Ladder: `G0` global 1-NN, `L0` identity local 3x3, `L1` detached-aligned local
   3x3, `R1` per-support componentwise median plus cross-support geometric median.
6. Support: same query-ranked `K=min(5,n_valid)` supports for every rung; fewer
   than three valid supports gives deterministic coarse-only fallback.
7. Calibration: leave-one-image-out empirical upper tail with ties; fused map is
   `max(e_coarse, confidence * e_micro)`; fused threshold uses image-disjoint normals.

## Resource and Evaluator Contract

- `P16-random`: sort the normal-path pool, apply one NumPy `PCG64(seed)`
  permutation, and expose exactly its first 16 paths before any image decode or
  feature load for seeds 0/1/2. Consecutive blocks of four are each fold's
  calibration/query set and the other 12 are memory. No seventeenth normal image
  is visible to the method. This is the genuine few-shot diagnostic.
- `P16-superad`: use the exact same DINOv2 CLS coreset 16 paths as the matched
  SuperAD branch. This is paper-budget comparable but full-pool-selected.
- `M16-fullpool`: any other full-pool-to-16 selection diagnostic.
- `Pfull`: SuperADD-style all-normal access, never mixed with P16 tables.
- One evaluator consumes raw maps for all methods and reports all-test and bad-only
  pixel `AUROC_0.05`, AP, oracle max-F1, fixed normal-threshold F1, no morphology,
  and one identical fixed morphology profile.

### Frozen AD1 Synthetic Suite

- Development source is only the 15 MVTec AD1 `train/good` folders; neither
  `test/` nor `ground_truth/` is enumerated.
- Version `darc-line-cue-v1` contains `thin-w1-l32`, `thin-w2-l48`, and the
  no-harm control `broad-control-w16-l96`.
- Each cue is a deterministic non-antialiased 8-connected line. Its location and
  angle use a seed derived from the object, relative source path, P16 seed, and
  profile. The line is black on locally bright pixels and white otherwise; the
  exact rasterized pixels are the mask.
- A selected image is queried only in the fold where it is one of the four
  calibration paths, so its clean source is absent from that fold's memory.

### Frozen Shared Morphology and Baseline Source

- Raw continuous metrics and no-morphology results are primary. The fixed shared
  diagnostic is `shared-superadd-v1`: official SuperADD radius 26, 16 line
  angles, lower factor 0.8, contour fill, and elliptical erosion radius 1,
  applied identically to every method at the same threshold.
- Official SuperADD source is commit
  `44cf25144442fbbc1334ea59d1632327a4376d1a`. Its reported binary output remains
  a separate provenance row; it is not allowed to select the shared profile or
  replace raw-map comparison.

## Gates

- Gate 0: evaluator/protocol parity; no method-gap claim until map population,
  threshold, morphology, normal access, and reference paths are explicit.
- Gate 1: paired 16px vs 8px `G0`; require AP `+0.005` absolute and `+50%`
  relative or component recall `+0.10`, paired image-bootstrap lower bound `>0`,
  and control mean pAUROC loss `<=0.005`.
- Gate 2: `L1` held-out-normal p99.9 residual at least 20% below `L0`, positive
  AP/component-recall paired bound, and `R1` retaining at least 90% of synthetic
  thin-cue response.
- Gate 3: frozen AD2 shadow on can/fabric/vial/wallplugs/rice; mean oracle F1
  `+0.03`, mean pAUROC loss `<=0.002`, no class F1 loss over 0.02, and gains on
  can plus at least two gap classes.
- Gate 4: untouched/private only; matched FlowTTE and SuperAD, three P16 seeds and
  Pfull, stable fixed-threshold F1, RN-FMLK and factorial hard-null dominance.

## Execution Matrix

- GPU 0: protocol-parity anchors and resolution-low controls.
- GPU 1: resolution-high candidates and synthetic/AD1 mechanism tests.
- GPU 2: correspondence ladder or matched hard nulls after the resolution gate.
- Independent object chunks are scheduled without changing method constants.
- Raw maps are retained only through the common evaluation pass, then removed
  after compact metrics, manifests, bootstrap rows, and conclusions are stored.

## Stop Rules

- A failed resolution gate kills registration/reconstruction as a method family,
  though reusable coordinate/evaluator utilities remain.
- A failed L1 gate kills alignment novelty; a tie with operational component nulls
  reduces the result to engineering.
- Threshold instability suppresses deployable fixed-F1 claims.
- AD2-public retuning, unequal normal access, bad-only-only gains, or oracle
  post-processing invalidates a claim regardless of metric gain.
