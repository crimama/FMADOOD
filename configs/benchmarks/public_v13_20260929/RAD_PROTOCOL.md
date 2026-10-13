# RAD public-v13 preparation, 2026-09-29

## Sources and code status

- Paper: Xingwu Zhang et al., *Is Task-Specific Training Necessary for Anomaly
  Detection?* (Retrieval-based Anomaly Detection, RAD), accepted at ICML 2026;
  use arXiv v3 (9 June 2026) as the citable paper version:
  <https://arxiv.org/abs/2601.22763v3>. The multi-level retrieval method is in
  §3; the reported DINOv3/MVTec few-shot settings are in §5.3.
- Official implementation: <https://github.com/longkukuhi/RAD>, pinned at
  `59847f4f55082c31649d4e0a2dd0ec329b932633`.
- The pinned repository has no `LICENSE`, `LICENSE.md`, `COPYING`, or `NOTICE`,
  and GitHub reports no repository license. Although its README says that a
  root license file exists, it is absent from the tree. The RAD adapter in
  `src/fmad/methods/rad.py` is an independent implementation of the paper and
  published equations; it does not copy or vendor the repository's method
  code. This does not grant a license to reuse that source code.
- The required encoder is the native DINOv3 ViT-B/16 (86M parameter) model.
  The official model and checkpoint exist, but Meta/Hugging Face access is
  manually gated. The initial unauthenticated check could not fetch the
  checkpoint; the authorized account and pinned checkpoint used for the current
  execution are recorded below. The existing ViT-L/16 and ViT-H/16plus caches
  are not substitutes for the paper's ViT-B/16 setting.

## Method and shared-pipeline contract

RAD freezes DINOv3 ViT-B/16, stores final-normalized patch and CLS features
from block indices `[3, 6, 9, 11]`, retrieves up to 150 normal support images by
final-layer CLS cosine similarity, and computes local patch anomaly scores as
one minus the largest cosine similarity within a spatial neighborhood of
radius 1. The four layer score maps use equal weights. No labels or query
metadata enter fitting or prediction.

The paper and code evaluate original MVTec AD and use a multi-category support
bank. The released runner has no MVTec AD2 adapter or fixed few-shot manifest
support. For this v13 comparison, the same seed/shot manifest used by the
shared runner supplies all eight categories' normal examples. Since the
shared runner calls `fit()` with one category at a time, RAD verifies that
category's selected paths against the manifest and reads only the remaining
manifest-listed normal images to build the global bank. It does not enumerate
training directories. The query object remains unknown to the method, matching
RAD's multi-category retrieval contract.

The official setting uses `k_image=150`; the adapter clamps this to the global
bank size because the v13 bank contains only 8, 16, 32, 64, or 128 images for
1/2/4/8/16 shots. Thus every configured public-v13 run uses all available
support images as retrieval candidates. This is the exact top-150 rule applied
to a smaller bank, not an added tuning choice.

## Explicit AD2 and evaluator adaptations

- Dataset: MVTec AD 2 `test_public`, all eight objects, shared query/mask split
  and streaming evaluator. Preprocessing declaration remains
  `no_mask_no_rotation`.
- The official loader resizes every square MVTec AD image to 512×512 and takes
  a 448×448 center crop. The adapter retains that published transform so every
  stored/query image has a 28×28 patch grid. AD2 train images have different
  aspect ratios and orientations; the square resize can distort them and the
  center crop leaves margins unobserved. Scores from the crop are mapped back
  through the inverse resize/crop geometry, with zero scores on unobserved
  margins, so the shared evaluator receives a map matching the original query
  dimensions. This is a material source-method limitation on AD2.
- The official evaluation uses a 256×256 map, 5×5 Gaussian smoothing with
  sigma 1, and by default averages the top 1% map scores for image AUROC. The
  adapter keeps the map resize and smoothing, maps the result to the full
  query frame, and supplies the maximum final map score for v13's shared
  image-score contract. Image AUROC therefore follows the project contract,
  not the repository's default top-1%-mean option.
- The public-v13 primary endpoint remains shared macro AUPRO@FPR 0.05. Fitting
  is frozen feature extraction and memory construction; no task-specific
  gradient updates occur.

## Configurations and support provenance

Fifteen intended configurations are provided as `rad_seed{0,1,2}_s{01,02,04,08,16}.yaml`.
They name native `dinov3_vitb16`, block indices `[3,6,9,11]`, 448 crop size,
global support retrieval 150, radius 1, uniform weights, and the shared
`mvtec_ad2_float16_streaming_v1` evaluator.

- Seed 0: `configs/methods/VOID_support_{1,2,4,8,16}.json`.
- Seeds 1 and 2, 1/4/8 shots:
  `configs/benchmarks/public_v13_20260929/seed{1,2}_support_{1,4,8}.json`.
- Seeds 1 and 2, 2/16 shots:
  `configs/benchmarks/v7_evidence/E1_seeds/manifests/seed{1,2}_support_{2,16}.json`.

The 1/4/8 manifests for seeds 1 and 2 were verified against the corresponding
E1 seed-16 manifests as exact prefixes for every object. Each selected
manifest contains the same shot count for all eight classes. The RAD method
config and shared protocol fields intentionally point to the same manifest.

## Execution state (2026-09-30)

The `dinov3_vitb16` backbone and `rad` method are registered locally. The
backbone resolves the official `facebook/dinov3-vitb16-pretrain-lvd1689m`
model ID. The authorized Hugging Face account downloaded revision
`5931719e67bbdb9737e363e781fb0c67687896bc`; its `model.safetensors`
SHA-256 is `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
The local existing container uses the verified copy at
`/model-cache/hub/hf_dinov3_vitb16_5931719e`, selected with `DINOV3_MODEL_ID`.
The container's Transformers and tokenizers packages were updated to 4.57.6
and 0.22.2 for native `dinov3_vit` support; PyTorch is 2.6.0+cu124.

The 1-shot seed-0 `can` pilot and the other seven objects completed on the
local GPU. At user request, the local continuation was stopped after 17 more
completed objects; its active object was not counted. Thus 25 local
seed/shot/object cells are retained and will not be rerun.
The same checkpoint hash was verified in `dsba3` at
`/model-cache/hub/hf_dinov3_vitb16_5931719e`. Its existing Python 3.11
runtime can load the model with isolated Transformers 4.57.6 and tokenizers
0.22.2 under `/workspace/.local-runtime/rad_deps_20260930`, without changing
the packages used by concurrent experiments. The concurrent `dsba3` GPU-2
experiment uses a separate immutable source release. RAD uses an isolated
copy at `/workspace/.local-sync/rad_20260930_official_b16` with only the
locally validated DINOv3 registry and RAD import changes overlaid; the active
checkout and release were not modified. A seed-0 8-shot `fabric` RAD pilot
completed on GPU 0. The remaining cells were first partitioned over GPU 0,
GPU 1, and GPU 3. At user request, GPU 1 stopped after 22 completed cells;
its unfinished seed-2 tail was divided into disjoint continuations on GPU 1
and GPU 2. GPU 0 and GPU 3 completed their original shards. The GPU-2 RAD
continuation shared that device with the existing unrelated experiment; no
checkpoint, source release, or output directory is shared. No RSG4 method
files were changed for RAD.

## Completed public comparison

All 120 seed/shot/object cells (3 seeds × 5 shots × 8 classes) have one unique
completed `metrics.json`, `config.yaml`, and `run_manifest.json`. Twenty-five
cells were recovered from the stopped local runs and 95 from `dsba3`; the
stopped runners' stale `running` status files were not treated as completion
evidence. The shared collector registered the verified full-eight results in
`results/public_v13_20260929/rad_paper_full8/comparison.md` and the canonical
Markdown/CSV ledger. At 16 shots, macro AUPRO@0.05 is 33.54 ± 0.98%, mean ±
sample standard deviation across three seeds. The split is
`test_public_seen_development`, so this is development evidence rather than
an untouched final test. The detailed report preserves class-level results;
remote quantitative artifacts were pulled while original logs remain on
`dsba3`.
