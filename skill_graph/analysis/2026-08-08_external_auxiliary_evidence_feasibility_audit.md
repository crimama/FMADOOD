# External Auxiliary Evidence Feasibility Audit

Date: 2026-08-08

Verdict: `NO_METADATA / DESIGN_KILL_METADATA_ORACLE / NEXT_FROZEN_IMAGE_ONLY_PRIOR_FEASIBILITY`

## 1. 핵심 결론

few-shot AD under distribution shift에서 지금 필요한 것은 support/query score를 다시
가공하는 또 다른 internal signal이 아니라, nuisance와 defect를 식별하게 해 줄 **외부
증거원**이다. 이번 audit은 MVTec AD2 single-image 조건에서 그런 외부 증거원이 현재
repo/data에 deployable하게 존재하는지 확인했다.

결론은 세 갈래다.

1. 파일명, split directory, public condition token, private `regular/mixed` suffix는 사용할 수 없다.
   이는 데이터 부재가 아니라 `FORBIDDEN / DESIGN_KILL_METADATA_ORACLE`이다. benchmark가 이미
   답을 파일명에 노출하므로, 이를 쓰면 image-dependent oracle이 된다.
2. per-image acquisition metadata, camera/exposure/sensor sidecar, EXIF는 현재 확인되지 않았다.
   sampled PNG는 조건과 무관하게 `gamma=0.45455`, `dpi=(299.9994, 299.9994)`, EXIF length `0`이었다.
3. single-image task를 유지하는 한 paired acquisition은 금지다. 사용자가 single-image 조건을
   명시적으로 해제하지 않는 이상 paired normal capture나 multi-view/controlled-observation으로
   넘어가지 않는다.

따라서 다음에 열 수 있는 유일한 비중복 후보는 query별 benchmark 답을 주지 않는
**frozen general image-only prior**의 feasibility audit이다.

## 2. 감사 범위

- `target_dataset`: MVTec AD2 single-image
- protocol default remote root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- local evidence root in this session: `/home/hun/Volume/DATA/mvtec_ad_2`
- object set: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`
- method family: deployable support/evidence identification, not a candidate detector run
- SuperAD metric contract: `seg_AUROC_0.05`, `seg_F1`, map/claim coverage when a method claim exists

이번 감사는 method result가 아니므로 same-condition SuperAD comparison은 `N/A`다. code/GPU run도
없으며, Flow Matching, Diffusion, DiT를 승격하지 않는다.

## 3. 데이터 관찰

로컬 데이터 루트에서 split과 count를 확인했다.

| object | train | validation | test_public | test_private | test_private_mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| can | 412 | 46 | 252 | 321 | 321 |
| fabric | 387 | 43 | 246 | 314 | 314 |
| fruit_jelly | 263 | 37 | 140 | 255 | 255 |
| rice | 313 | 35 | 222 | 277 | 277 |
| sheet_metal | 137 | 19 | 204 | 142 | 142 |
| vial | 291 | 41 | 245 | 276 | 276 |
| wallplugs | 293 | 33 | 240 | 232 | 232 |
| walnuts | 432 | 48 | 240 | 228 | 228 |

비이미지 파일은 dataset-level `readme.txt`, `license.txt`뿐이었다. per-image
`manifest/CSV/JSON/XML/camera/exposure/sensor` sidecar는 발견되지 않았다.

파일명은 다음처럼 condition을 직접 노출한다.

- public bad: `000_regular.png`, `000_overexposed.png`, `000_underexposed.png`,
  `000_shift_1.png`, `000_shift_2.png`, `000_shift_3.png`
- private: `000_regular.png`
- private mixed: `000_mixed.png`
- train/validation good: `*_regular.png`

이 token은 acquisition metadata가 아니라 benchmark label channel이다. 특히 public split에서는
normal shift condition과 bad/regular pairing이 파일명으로 노출된다. 이를 사용하면 target-specific
module의 evidence가 아니라 benchmark protocol을 이용한 oracle routing이 된다.

PNG metadata 샘플 audit:

| sample | PNG info | EXIF |
| --- | --- | --- |
| `can/train/good/000_regular.png` | `dpi=(299.9994,299.9994)`, `gamma=0.45455` | empty |
| `can/test_public/bad/000_overexposed.png` | same | empty |
| `can/test_public/bad/000_shift_1.png` | same | empty |
| `can/test_private/000_regular.png` | same | empty |
| `can/test_private_mixed/000_mixed.png` | same | empty |
| `fabric/test_public/bad/000_shift_2.png` | same | empty |

따라서 deployable metadata branch는 열리지 않는다.

## 4. 기존 internal signal branch와의 경계

다음은 외부 증거원이 아니므로 이번 audit의 해법이 아니다.

- CLS/register/global task code: shift-conditioned Flow에서 held shifted-normal inflation을 악화
- augmentation label 또는 synthetic transform id: benchmark query의 실제 acquisition state가 아님
- paired loss: normal alignment는 일부 되지만 defect retention 실패
- support banks / transformed-bank union / patchwise-min: 작은 coverage signal과 object heterogeneity,
  `NO_CONTINUE`
- query persistence / nuisance orbit: rejected closed branch, post-hoc score-map shaping
- density/object foreground priors: calibration 또는 segmentation prior이지 nuisance-defect 식별 변수 아님

이는 “외부 증거원이 없으니 internal signal을 더 튜닝하자”가 아니라, internal signal family가 이미
식별 불가능성 근처에서 반복적으로 실패했다는 negative evidence다.

## 5. 허용되는 다음 후보

유일하게 다음 audit로 열 수 있는 후보는 **frozen general image-only prior**다.

허용 조건:

- query별 정답, condition token, split token, paired capture를 사용하지 않는다.
- prior는 일반 이미지 지식에서 고정되어 있어야 하며 MVTec AD2 public/private test에 맞춰 튜닝하지 않는다.
- named prior, weights, license, input/output contract가 명시되어야 한다.
- anomaly-preservation assay를 사전등록해야 한다.
- SuperAD 비교는 method claim이 생긴 뒤에만 같은 split/reference/metric으로 한다.

예시 후보군은 가능성 감사 대상일 뿐이며 아직 method로 승격하지 않는다.

- frozen segmentation/foreground prior
- frozen illumination or intrinsic-image prior
- frozen visual foundation prior that emits image-only uncertainty or structure cues
- frozen diffusion prior only if it has a defined reconstruction/residual output and defect-preservation assay

## 6. 명시적 금지선

- filename/split/suffix based condition inference: `FORBIDDEN / DESIGN_KILL_METADATA_ORACLE`
- public `regular/overexposed/underexposed/shift_*` routing: `FORBIDDEN`
- private `regular/mixed` suffix routing: `FORBIDDEN`
- validation-good as shift metadata: protocol-changing support expansion, not external evidence
- paired normal capture: single-image condition violation unless user explicitly reauthorizes task switch
- Flow Matching/Diffusion/DiT promotion: evidence source 없이 parameterization만 바꾸는 retread

## 7. Evidence / Inference / Unknown

Evidence:

- local root exists at `/home/hun/Volume/DATA/mvtec_ad_2`; protocol remote default path was not mounted
  in this session.
- all eight public objects are present locally.
- only dataset-level `readme.txt` and `license.txt` were found as non-image files.
- sampled PNG info and EXIF contained no useful per-image acquisition metadata.
- filenames expose benchmark condition and split suffixes.

Inference:

- filename/split/suffix use would be image-dependent oracle routing rather than deployable metadata.
- current internal support/query signals are not acceptable substitutes for external evidence because their
  failure modes are already documented in closed branches.

Unknown:

- whether an external non-benchmark acquisition log exists outside the current repo/data tree.
- whether a named frozen general prior can produce a useful deployable signal without defect absorption.
- whether such a prior's license and weights permit the planned research use.

## 8. Next Frozen Prior Feasibility Contract

Before any implementation, the next report must name exactly one prior and record:

- prior name and version
- weights source and local availability
- license and research-use compatibility
- input preprocessing and output tensor/map schema
- whether output is image-only and independent of AD2 filename/split/test condition
- frozen use policy: no MVTec AD2 test tuning, no prompt/threshold selection on public labels
- anomaly-preservation assay: local and broad synthetic defects must not be suppressed
- intended integration point: scoring feature, mask prior, calibration feature, or diagnostic-only
- hard-stop: if prior output correlates mainly with filename/split/condition proxy or absorbs defect maps,
  mark `DESIGN_KILL` before GPU benchmark

This is the next non-overlapping step. It is not yet a Flow Matching, Diffusion, DiT, or Normalizing Flow
architecture claim.

