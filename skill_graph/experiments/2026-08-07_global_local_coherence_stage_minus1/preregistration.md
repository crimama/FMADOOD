# Global/Local Coherence Stage -1a0 사전등록

## 목적과 범위

이 실험은 새 target-specific module의 성능을 주장하거나 학습하는 실험이 아니다. frozen
Baseline1-core 표현에서, 조명·색조처럼 공간적으로 coherent한 normal nuisance와 결함으로
유발되는 local/broad residual이 측정 가능하게 분리되는지를 support-only로 점검한다. 이
가정이 지지될 때에만 global-alignment / local-residual factorization module의 구현 후보를
열 수 있다.

닫힌 branch를 재시도하지 않는다. `shift_conditioned_patch_local_flow`, global rank
projection, semantic atlas metric은 각각의 사전등록된 구성에서 Stage -1 kill 판정을 받았다.
본 audit에는 학습 가능한 register, FiLM, projection, flow 재학습, query adaptation이 없다.

## 고정 protocol

- 대상: MVTec AD2 single-image, `fruit_jelly`, `walnuts`, `vial`, `fabric`.
- support: 고정 16장 normal support manifest (SHA-256은 runner가 강제 검증).
- 표현: DINOv3 ViT-H/16+ (layers 7/15/23/31), layer-normalized mean fusion, support position-mean
  DVT. DVT 뒤 frozen Baseline1 core Flow latent도 함께 기록하되, flow 파라미터는 update하지
  않는다.
- 접근: `train/good` support 외 파일, `test_public`, `test_private`, `ground_truth` 접근 금지.
- 점수: static patch 1-NN only. density, RGB/morphology, query adaptation off.
- 고정 transform 조건 (각 source마다 deterministic):
  - `S_global`: brightness, contrast, gamma, RGB gain의 전역 photometric 변화.
  - `S_smooth`: 저주파 multiplicative illumination field.
  - `D_local`: 중앙 이외 deterministic 작은 rectangular RGB defect.
  - `D_broad`: 넓은 rectangular RGB defect.
  - `SD_*`: 위 shift와 defect의 조합.

## 측정량

각 held source는 4-fold (12 fit / 4 held)로 평가한다.

1. **normal-shift tail**: clean과 shifted patch 1-NN score의 q995, relative inflation.
2. **coherence fraction**: 같은 source의 patch-level feature displacement \(\Delta_p\)에 대해
   \(\|mean_p\Delta_p\|^2 / mean_p\|\Delta_p\|^2\). 전역 nuisance가 local defect보다 높아야
   한다.
3. **spatial smoothness**: 인접 patch displacement cosine similarity. smooth/global nuisance가
   local defect보다 높아야 한다.
4. **defect contrast**: synthetic mask inside score mean - outside score mean. shift+defect에서
   defect-only 대비 보존 비율을 local과 broad에서 별도 기록한다.
5. **clean geometry**: clean patch bank의 raw-to-latent pair-distance Spearman 및 1-NN retention.

모든 수치는 object 및 fold/source 단위 원자료와 함께 저장한다. 임계값 탐색은 하지 않으며
동일 config의 두 execution lane(local GPU 0, dsba5 GPU 3) 결과 hash와 수치를 비교한다.

## 사전 판정 규칙

이 것은 후보 선택 gate이지 benchmark keep/kill gate가 아니다.

- `SUPPORTS_FACTORIZATION_HYPOTHESIS`: 네 object 중 적어도 세 object에서 global/smooth의
  coherence fraction과 spatial smoothness가 local defect보다 모두 높고, local 및 broad
  defect contrast retention의 object median이 양수이다. 두 lane은 동일한 input/result schema와
  condition hash를 가져야 하며, aggregate metric은 float32·상이한 GPU kernel의 수치 오차를
  고려해 absolute `1e-3` 또는 relative `1%` 이내여야 한다.
- `INCONCLUSIVE_OR_NOT_SUPPORTED`: 위 조건 중 하나라도 충족하지 못하거나, access/manifest/
  deterministic parity가 실패한 경우. 새 module 구현으로 진행하지 않는다.

위 결과가 양성이더라도 Flow/Flow Matching/Diffusion/DiT 어느 family도 자동 채택되지 않는다.
그 다음 단계에서 R1--R9 requirement, clean-geometry safety, held-normal tail 및 defect
retention gate를 만족하는 별도 N1 candidate를 사전등록한다.

## 실행 환경 및 산출물

- local: `hun_fsad_tta_image:stage1a0`, CUDA slot 0 = local host GPU 0.
- dsba5: 고정 `hun_fsad_tta_012`, CUDA slot 0 = dsba5 host GPU 3.
- 결과: `results/stage_minus1/global_local_coherence/{local,dsba5}/` 및 remote pull copy.
- 산출물: `run_manifest.json`, `access_ledger.json`, object summary, `progress.jsonl`,
  `checksums.sha256`, lane parity summary, 최종 `report.md`.
