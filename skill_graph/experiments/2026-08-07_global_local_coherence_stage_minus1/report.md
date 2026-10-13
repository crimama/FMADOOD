# Global/Local Coherence Stage -1a0 결과

## Verdict

`INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE / NO_PUBLIC_ACCESS`

새 target-specific module의 구현은 열지 않는다. 이 실험은 performance benchmark가 아니라
factorisation 가정의 support-only audit이며, preregistration의 네 object 중 세 object 조건을
충족하지 못했다.

## 실행·custody

- 대상: MVTec AD2 single-image `fruit_jelly`, `walnuts`, `vial`, `fabric`, 고정 normal
  support 16장.
- lane: local GPU 0과 dsba5 host GPU 3 (container CUDA slot 0)을 같은 DINOv3 checkpoint,
  support manifest, runner hash로 병렬 실행했다.
- runner SHA-256: `2e4bfefeabb22906e1b0bb24cf59a7a68fcb4d5f0e129bbb01c0a8b58952a6aa`.
- support manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- 두 lane 모두 `test_public_accessed=false`, `test_private_accessed=false`,
  `ground_truth_accessed=false`를 기록했다. local/remote access ledger는 byte-equivalent
  schema와 content hash를 가졌다.
- 두 `checksums.sha256` manifest는 재검증을 통과했다. local controller log는 최초에 결과
  root 안에 있어 manifest 생성 뒤 append되어 불일치가 한 건 있었고, 수치 산출물은 변경하지
  않은 채 root 밖으로 이동시킨 뒤 checksum manifest만 재생성했다.

## 결과 (dsba5 aggregate; local은 모두 tolerance 내 재현)

| object | global coherence | local/broad defect coherence | global smoothness | local/broad defect smoothness | local retention | broad retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | 0.2747 | 0.0553 | 0.7150 | 0.7715 | 0.9027 | 0.8385 |
| walnuts | 0.2121 | 0.0689 | 0.5860 | 0.7256 | 0.9557 | 0.9838 |
| vial | 0.1618 | 0.1005 | 0.6785 | 0.8326 | 0.9289 | 0.8182 |
| fabric | 0.1397 | 0.0512 | 0.6201 | 0.5530 | 0.9782 | 0.9715 |

모든 object에서 global/smooth perturbation의 **coherence fraction**은 defect 조건보다 높았고,
shift+defect contrast retention도 양수였다. 그러나 preregistered spatial-smoothness 비교는
`fabric` 한 object에서만 global > defect였다. 따라서 preregistration의 통과 조건을 만족한
object는 `1/4`이고, 정해진 `>=3/4` gate를 통과하지 못했다.

## 재현성

모든 비교 metric에서 local–dsba5 차이는 absolute `6.3e-7` 이하였고, 사전등록한 absolute
`1e-3` 또는 relative `1%` tolerance를 통과했다. 자세한 비교는
`results/stage_minus1/global_local_coherence/aggregate_v2/verdict.json`에 저장했다.

## 해석과 다음 제약

이 결과는 “전역 shift가 더 coherent한 displacement를 만든다”는 부분적 관측은 제공하지만,
현재 smoothness statistic이 factorisation evidence로 충분하다는 주장을 지지하지 않는다.
특히 local rectangular defect의 다수 patch가 변하지 않아 인접-patch 평균 cosine을 높일 수
있다. 이는 post-hoc threshold 조정으로 고치지 않는다.

다음 단계가 필요하다면, **새 module이 아니라** 별도의 preregistration으로 (a) 변화량이
유의한 patch만 조건부로 평가하는 smoothness statistic, (b) mask를 모르는 실제 inference에서
사용 가능한 mask-free proxy, (c) local/broad defect 보존과 clean geometry gate를 함께
정의하는 measurement audit부터 다시 시작한다. Flow, Flow Matching, Diffusion, DiT candidate의
구현·public evaluation은 이 결과로는 허가되지 않는다.

## 산출물

- Design: `preregistration.md`
- Local: `results/stage_minus1/global_local_coherence/local_v2/`
- dsba5 pull: `results/remote_runs/dsba5/global_local_coherence_20260807_v2/mechanism/`
- Parity/gate: `results/stage_minus1/global_local_coherence/aggregate_v2/verdict.json`
