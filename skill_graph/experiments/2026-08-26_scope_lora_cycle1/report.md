# SCOPE-LoRA progressive Cycle 1 report

Date: 2026-08-26  
Protocol: `scope-lora-cycle1-support-intervention-20260826`  
Status: `COMPLETED_P0 / P1_NOT_AUTHORIZED`  
Verdict: `INSUFFICIENT_PROJECTOR_SIGNAL / CONTINUE_ONE_DIAGNOSTIC`

## Outcome

The four-object remote P0 completed with valid compact artifacts.  The rank-4
projector produced a real positive normal-correction signal: all structural
checks passed, aggregate median held explained energy was `0.387690`, and the
median PT-P normal-q99 reduction was `+2.6555%`.  The frozen gate nevertheless
failed because the whole anomaly-residual commutation check failed on `can`
and `vial`.  P1 oracle-routed LoRA was therefore not launched.

This is a useful bounded failure rather than a null result.  It separates two
facts: a small intervention span can remove part of the held normal shift, but
the frozen backbone does not preserve the held synthetic anomaly residual
under every shift.  Cycle 2 may change only the anomaly-retention diagnostic
and protection contract; it may not retroactively relabel Cycle 1 as passing.

## Remote execution

- Successful run:
  `scope_lora_cycle1_dsba3_20260826T1442Z_v3`.
- Fixed container: `hun_fsad_tta_012`.
- GPU/object map: GPU0 `can`, GPU1 `fabric`, GPU2 `vial`, GPU3 `walnuts`.
- Immutable snapshot SHA-256:
  `b8b57cd4d4e2fb368cf000988f2d4d14db880455c3b83d0a1e6cb5b7dc046f55`.
- Host and container GPU UUID-map SHA-256 both matched
  `a27a8597ba2d435fc05d059d2f9f28bdb323881bda68ad9bd9574c145492595d`.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Aggregate artifact SHA-256:
  `af62f5c580606db0530ab35cae807e1f6e5dc251fb145c6c2ac43e3cb59cb26c`.

The successful preflight passed 22 focused remote tests, offline DINOv2 cache
loading, an actual CUDA/bf16 backbone/LoRA forward, zero-init no-op validation,
GPU UUID mapping, disk availability, and support custody.  After pull, the
remote snapshot was removed, no `anomaly_maps/` directory existed, and host
GPUs 0--3 were idle at `1 MiB`, `0%` with no compute process.

Two launch-path failures preceded the successful run and contain no
scientific result.  v1 consumed the controller script and source archive from
the same stdin, creating an empty snapshot; v2 reached CUDA model-smoke but
predeclared `torchvision::nms` before loading the container's native operator.
The snapshot stream is now a separate hash-verified transfer, controller
failure trapping preserves the real exit status, CUDA paths force the native
torchvision operator, and both failures are covered by regression contracts.
The locally started P0 was interrupted immediately when local execution was
stopped and produced no retained result; all result-bearing computation above
is from dsba3.

## P0 results

| object | held EV median | PT-P q99 reduction | min anomaly cosine | anomaly norm range | structural |
| --- | ---: | ---: | ---: | ---: | --- |
| can | `0.488714` | `+5.1852%` | `0.732538` | `[0.961920, 1.000941]` | pass |
| fabric | `0.293944` | `+0.1258%` | `0.939957` | `[0.918632, 0.998451]` | pass |
| vial | `0.481436` | `+11.8942%` | `0.608433` | `[0.648301, 0.988063]` | pass |
| walnuts | `0.069075` | `+0.0839%` | `0.942866` | `[0.983918, 1.006259]` | pass |

Aggregate facts:

- valid objects: `4/4`; valid held cells: `12/12`;
- zero-init no-op maximum absolute error: `0` on every object;
- projection/complement error: at most `1.754e-15`;
- rank-4 median explained energy: `0.387690`, above the frozen `0.20` floor;
- PT-P median normal-q99 reduction: `+2.6555%`;
- every object's median EV was positive and every object had a positive median
  q99 reduction;
- whole anomaly gate: fail, driven by `can` and `vial`;
- `benchmark_comparable=false`; TESTpublic/private/GT were never opened.

The signal is heterogeneous.  `can` and `vial` have the strongest combination
of rank-4 energy and normal-tail reduction.  `walnuts` has little rank-4
energy for exposure/noise but high energy for spatial illumination, so a
single global rank increase would mix two different failures and is not the
next authorized repair.

## Isolated failure

For the P0 anomaly audit, the same paired clean correction `delta_P` is added
to shifted anomalous and shifted clean tokens before differencing:

`(Z(T(A)) + delta_P) - (Z(T(x)) + delta_P) = Z(T(A)) - Z(T(x))`.

Consequently the failed cosine is independent of the projector intervention.
It measures whether the frozen representation commutes with the held anomaly
under the shift, not whether PT-P erases anomaly evidence.  Gaussian shift is
the largest failure: anomaly cosine is `0.7325` for `can` and `0.6084` with
norm ratio `0.6483` for `vial`.

Cycle 1 must remain failed because this was the preregistered gate.  The next
single question is whether the positive clean-tail correction preserves the
task-relevant masked anomaly score/contrast even when raw residual vectors do
not commute.  If a separately preregistered score-level audit fails, P1 stays
closed and the Cycle-3 repair must address the representation/protection path.
If it passes, it can license P1 without rewriting the Cycle-1 outcome.

## Custody and stop

All rows are exact support-16 plus support-disjoint `train/good` descendants
and deterministic synthetic transforms.  Forbidden-access ledgers are empty;
there is no public/private/test/GT/query-memory result.  No hyperparameter,
rank, seed, or threshold sweep was run.  P1 stopped exactly at the negative P0
gate.

