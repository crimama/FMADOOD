# SIAI-S1 J0 oracle Stage -1 report

Date: 2026-08-27  
Protocol: `siai-s1-j0-true-action-oracle-20260827`  
Status: `COMPLETED / SUPPORT-ONLY CAN SENTINEL`  
Verdict: `INSUFFICIENT_SIAI_SIGNAL / END_SIAI_S1`

## Outcome

The explicit signed-exposure inverse produced a real but insufficient signal.
It reconstructed paired normal pixels `81.24%` better than the wrong-sign
control and improved feature-correction cosine over the stronger learned-free
null by `+0.07628`.  It also preserved exact-RN topology and slightly reduced
shifted-normal q99.  However, its feature correction cosine was only `0.55529`
and its anomaly-residual cosine was `0.79451`, below the preregistered `0.90`
requirements.  The immediate-stop rule therefore closes SIAI-S1; J2 and a
coefficient estimator are not licensed.

The paired-clean ceiling passed essentially perfectly (correction cosine/norm
`1.0/1.0`, anomaly cosine/norm `0.99970/1.0`, RN rank/neighbor/scale
`1.0/1.0/1.0`).  This rules out the readout and metric plumbing as the main
failure.  The most concrete mechanism is exposure-sign asymmetry: the two
positive-EV held bases saturated `29.47%` and `30.20%` of pixels and retained
pixel MAE `6.68` and `6.84` after inversion, whereas both negative-EV bases had
zero saturation and pixel MAE only `0.349` and `0.350`.  A quantized inverse
cannot recover information destroyed by positive-exposure clipping.

## Gate result

| measure | observed | requirement | result |
|---|---:|---:|---|
| correction cosine | `0.555286` | `>=0.90` | fail |
| correction norm ratio | `0.803209` | `[0.80,1.25]` | pass |
| anomaly cosine | `0.794509` | `>=0.90` | fail |
| anomaly norm ratio | `1.001242` | `[0.80,1.25]` | pass |
| RN rank correlation | `0.988590` | `>=0.90` | pass |
| RN neighbor retention | `0.936523` | `>=0.90` | pass |
| RN scale ratio | `1.013850` | `[0.80,1.25]` | pass |
| normal q99 reduction | `+0.005741` | `>=-0.005` | pass |
| correction hard-null margin | `+0.076280` | `>=+0.01` | pass |
| pixel-MAE gain over wrong sign | `+0.812371` | `>=+0.10` | pass |
| exact repeat | max diff `0.0`, same signature | `<=1e-7`, exact hash | pass |

The static-mean correction cosine was `0.47901`; wrong sign was `-0.56591`.
The true action therefore used the supplied intervention coordinate, but the
remaining clipping/quantization error was large enough that the frozen DINO
representation did not reach canonical clean geometry.  Anomaly cosine barely
improved over shifted frozen RN (`0.79165 -> 0.79451`), so the method did not
solve anomaly preservation even though RN rank and q99 were healthy.

## Remote execution and custody

- Scientific run:
  `siai_s1_j0_oracle_stage_minus1_dsba3_20260827_siai03_v1`.
- Host/container/GPU: dsba3, fixed project container
  `hun_fsad_tta_012`, GPU0 only.
- Remote preflight passed `20/20` focused tests, exact source/controller hash
  audit, and offline DINOv2 loading.
- Repeat scientific maximum absolute difference: `0.0`; signature SHA-256:
  `e7bb8442b42b47343f342fef432210e1e188c08e7566cbf7e83851ce45d4d78c`.
- Immutable snapshot SHA-256:
  `1e8c34f1bf431e311980dc533331bf21b1a9b7f2d5a07f8ebe524c7d94a8149f`.
- Aggregate SHA-256:
  `7ac67c0328df7b20344ebaa4e8bf202729649fdf16e8ae73c54ff818a852694e`.
- Source manifest SHA-256:
  `47810d45c3db6c03e6f73a66785de91e0a0d53f897f288fdb4a61d8a5c46d4c1`.
- `support_only=true`, `benchmark_comparable=false`; public/private/test/GT
  and query-memory paths remained sealed.  Only compact JSON/log evidence was
  retained; the exact immutable snapshot was removed after pull.
- Only the selected GPU0 UUID/memory/utilization/compute UUID, fixed project
  container, and this run's own paths were accessed.  No other user's
  container, process, file, volume, GPU, or job was enumerated or touched.

Two earlier attempts were result-free operational stops.  `siai01_v1` stopped
in remote unit tests, and `siai02_v1` stopped because the controller created a
runner output directory before the runner's fresh-directory check.  Neither
opened experiment images or produced scientific metrics; both defects were
fixed and the source/controller identities were refrozen before the completed
run.

## Continuation boundary

Do not run SIAI-S1 J2, fit a coefficient estimator, weaken the `0.90` feature
gates, remove positive-EV cases, or tune exposure magnitude after this result.
The valid positive signal is narrower: explicit intervention coordinates beat
both static and wrong-sign controls, while irreversible clipping dominates the
remaining error.  Any continuation must be separately named and exploratory,
must diagnose the sign/saturation mechanism before training a backbone, and
must retain the same support-only custody, held-G2 anomalies, exact RN, hard
nulls, and no-public/test boundary.
