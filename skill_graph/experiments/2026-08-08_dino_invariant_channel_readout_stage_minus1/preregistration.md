# DINO Invariant Channel Readout Stage -1 사전등록

Date: 2026-08-08  
Protocol: `dino-invariant-channel-readout-stage-minus1-20260808`  
Stage: support-only held-transform diagnostic; dsba5-only

## 1. Negative evidence intake and motivation

현재 DINOv3 representation은 DVT가 올바른 positional structure를 제거할 수 있을 만큼 유용하지만,
small local anomaly와 global photometric shift를 동시에 보존/분리하는지는 아직 불명확하다.
기존 paired canonicalization은 shifted-normal inflation을 약 `9.77%` 낮췄지만 defect retention
`0.8707`에서 실패했고, conditional normalization, FiLM/register, support augmentation, query orbit은
각각 query transport, learned canonicalization, memory expansion 또는 post-hoc score shaping이라는
닫힌 failure basin에 속한다.

이번 후보는 shift direction을 빼거나 query를 canonicalize하지 않는다. Normal support의 known
photometric pair에서 **shift-sensitive DINO channel을 처음부터 읽지 않는** fixed readout이 unseen
compound shift에서도 normal displacement를 줄이면서 local/broad defect contrast를 보존하는지 묻는다.
이는 passive feature surgery의 재포장이 될 위험이 있으므로, 실제 mask와 같은 cardinality의
score-shuffled hard null을 반드시 이겨야 한다.

## 2. Target, data, and access boundary

- `target_dataset=MVTec AD2 single-image`, dsba5 data root
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`; fixed 16 normal `train/good` support.
- support manifest:
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`,
  SHA256 `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- four fixed folds: each fold `12 fit / 4 held`, fold-disjoint. Mask fitting은 해당 fold의 12 clean
  support와 그 fit transforms만 사용한다.
- backbone: frozen DINOv3 ViT-H/16+, layers `[7,15,23,31]`, layer-wise token L2 normalization 후
  mean fusion `F`; native 672 resolution.
- Flow, DVT, query adaptation, score fusion, learned adapter/parameter update, `test_public`,
  `test_private`, GT, condition label, filename/split/metadata access는 모두 off/false다.
- synthetic local/broad defect mask는 held evaluation에서 contrast 계산에만 쓰며, mask selection,
  mask fitting, channel selection, memory/scoring에는 절대 쓰지 않는다.
- SuperAD/RN-FMLK/segmentation metric: `N/A`, `comparable=false`. 이는 image-only
  support-pair representation diagnostic이며 method or Table-1 claim이 아니다.

## 3. Frozen readout and three arms

Fit fold의 clean support `x_i`, patch `p`, channel `c`, fit photometric transform `T_t`에 대해
channel shift sensitivity를 다음처럼 고정한다.

\[
\delta_c=\operatorname{median}_{i,p,t}\left|F(T_t(x_i))_{p,c}-F(x_i)_{p,c}\right|.
\]

정확히 lower 50% channel만 선택한다.

\[
m_c=\mathbb{1}[\delta_c\leq\operatorname{median}(\delta)],\qquad
R_m(F)=\sqrt{C/\sum_c m_c}\,(m\odot F).
\]

동점은 stable channel index order로 정리하여 정확히 `C/2` channels가 남도록 한다. Scale factor는
mask dimension 감소 자체가 거리 크기를 바꾸는 효과를 제거하며, 후속 normalization/threshold/strength
search는 없다.

| Arm | Readout | Purpose |
|---|---|---|
| `I0_identity` | `F` 전체 channel | fixed baseline |
| `I1_invariant_mask` | 실제 `delta`의 lower 50% mask `m` | candidate |
| `I2_score_shuffled_mask` | fixed seed로 `delta`의 channel assignment만 shuffle한 lower 50% mask | hard null |

`I2`는 retained dimension, norm scale, sparsity regularization의 일반 효과를 통제한다. 모든 arm은
동일 fold의 clean 12-support memory에서 fixed raw `k=1` patch nearest-neighbor distance로 평가한다.
Query는 한 번만 forward하며 memory에 augmented view를 추가하지 않는다.

## 4. Fixed fit and held-transform protocol

Fit pair transforms는 각 축의 deterministic physical transform을 그대로 고정한다.

- exposure `±0.25 EV`;
- gamma `±0.10` log-step;
- contrast `±0.10` log-step;
- white balance `±0.08` log-step.

Held evaluation transforms는 fit에서 보지 않은 compound 두 개이며 amplitude는 `2x`다.

- `exposure_white_balance`;
- `gamma_contrast`.

각 held source에는 `clean`, held compound `U`, `U+D_local`, `U+D_broad`를 만든다. Defect generator,
seed, mask geometry는 sealed prior contract를 사용하며 per-object/arm/transform 선택 또는 sweep은
금지한다.

## 5. Metrics

Object별 fold median을 계산하고 object equal-weight median/macro를 함께 기록한다.

1. **representation invariance** — held clean pair displacement ratio:

\[
d_{arm}=\frac{\lVert R_{arm}(F(U))-R_{arm}(F(clean))\rVert}
{\lVert F(U)-F(clean)\rVert}.
\]

2. **scoring relevance** — clean 대비 held-shift 1-NN `q995` inflation 및 I0 대비 relative
reduction; I1-I2 reduction margin.
3. **anomaly preservation** — `U+D_local` 및 `U+D_broad`의 raw 1-NN defect-core minus
background contrast를 I0 대비 retention ratio로 기록한다.
4. **clean geometry retention** — clean pair-distance Spearman correlation 및 patch nearest-neighbor
identity retention.
5. **audit** — per-fold mask indices/cardinality, true/shuffled masks의 scale, finite/nonconstant
features/scores, exact support paths, transform IDs, access ledger.

## 6. Strict diagnostic gates

`PROMISING_FIXED_INVARIANT_READOUT`은 아래 모든 조건을 만족할 때에만 가능하다.

### A. Invariance specificity

- I1 held displacement ratio `<=0.85`;
- I1 ratio가 I2보다 적어도 `0.05` 작다.

위 값은 object-level fold median을 보고하며 object equal-weight aggregate에서 둘 다 만족해야 한다.

### B. Raw 1-NN relevance

각 object의 two held compounds를 함께 평가하여, 적어도 `3/4` objects에서 둘 다 만족해야 한다.

- I1 `q995` inflation relative reduction vs I0 `>=10%`;
- I1-I2 reduction margin `>=5 percentage points`.

### C. Anomaly and geometry preservation

- local 및 broad defect-contrast retention의 object macro가 각각 `>=0.95`;
- local 또는 broad retention이 어느 object에서도 `<0.90`이면 fail;
- clean pair-distance Spearman과 NN-identity retention이 각각 `>=0.90`.

### D. Validity

- exact support manifest/hash, all four fold coverage와 fold disjointness;
- true/shuffled mask cardinality가 정확히 같고 scale factor가 동일함;
- I1 mask는 fold-local 12 support의 fit pair로만 계산됨;
- finite/nonconstant values, deterministic transform/seed, access ledger가 유효함;
- query adaptation/parameter update/test/private/GT/metadata/filename-condition access는 모두 false.

## 7. Frozen verdicts and continuation

- `PROMISING_FIXED_INVARIANT_READOUT`: A--D strict gates 전부 통과. 이때만 작은 **trainable
  diagonal gate** 하나를 별도 사전등록하는 것이 가능하다.
- `CONTINUE_DIAGNOSTIC`: strict claim은 실패했지만, invariance direction A가 적어도 `2/4`
  objects에서 통과하고 anomaly retention C가 통과했으며 raw 1-NN relevance B만 미달. 다음은
  all-8 single confirmation 한 번으로 제한한다.
- `GENERIC_DIMENSIONALITY_EFFECT`: I1가 I2보다 invariant displacement나 q995 reduction에서
  구별되지 않는다.
- `ANOMALY_ABSORPTION`: shift/q995는 줄지만 local 또는 broad retention C가 실패한다.
- `NO_SELECTIVE_CHANNEL_INVARIANCE`: invariance A 또는 raw 1-NN relevance B가 실패하고 위
  continuation 조건도 만족하지 않는다.
- `INVALID_READOUT_DIAGNOSTIC`: custody, fold, numerical, mask scale/cardinality 또는 access
  failure.

`KEEP` 및 benchmark performance verdict는 없다. Continuation 외 결과는 trainable adapter,
Flow/DiT/Diffusion/Flow Matching replacement, DVT tuning 또는 channel-rank sweep을 열지 않는다.

## 8. Execution, outputs, and cleanup

- execution lane: dsba5 fixed container `hun_fsad_tta_012`, host GPU3 / in-container CUDA0 only.
- local은 implementation tests/dry-run만 담당하며 metric experiment는 실행하지 않는다.
- expected remote root:
  `/workspace/results_remote/dino_invariant_channel_readout_stage_minus1_20260808/`; pullback root:
  `results/remote_runs/dsba5/dino_invariant_channel_readout_stage_minus1_20260808/`.
- compact fold/object/aggregate JSON/TSV, run manifest, mask audit, access ledger, logs만 보존한다.
  `anomaly_maps/`는 생성하지 않으며, 생성 시 conclusion 후 제거하고 부재를 기록한다.

