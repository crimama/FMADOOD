# SIAI-D1 J0 sign/saturation diagnostic Stage -1 report

Date: 2026-08-27  
Protocol: `siai-d1-j0-sign-diagnostic-20260827`  
Status: `COMPLETED / POSTHOC EXPLORATORY / SUPPORT-ONLY`  
Verdict: `NO_REVERSIBLE_SIGN_SIGNAL`

## Outcome

The sign split exposed a strong, exactly repeatable mechanism signal, but not a
complete anomaly-preserving inverse.  On the two negative-EV bases, where the
forward action clipped no pixels, the explicit inverse reached correction
cosine/norm `0.90997/0.96159`, RN rank/neighbor/scale
`0.99859/0.99609/1.00142`, hard-null margin `+0.26792`, and pixel-MAE gain over
wrong sign `98.56%`.  The corresponding positive-EV correction cosine was only
`0.18301`; its two bases clipped `29.47%` and `30.20%` of pixels.

`SATURATION_ASYMMETRY` therefore passed.  Median reconstructed pixel MAE was
`0.34973` for negative EV versus `6.76342` for positive EV, a `94.83%` lower
error, while negative-EV correction cosine exceeded positive-EV by `+0.72696`.
This confirms that clipping explains most of the normal-feature correction
failure seen in the combined SIAI-S1 result.

The decisive remaining failure was anomaly preservation.  Negative-EV anomaly
residual cosine was only `0.80972` against the preregistered `0.90`, even though
all other `NEGATIVE_EV_REVERSIBLE_SIGNAL` components passed.  The paired-clean
ceilings passed for both signs, so the readout, RN, and metric plumbing were
capable of reporting the required geometry.  Consequently clipping is not the
only limitation: even the nearly reversible negative-EV image action does not
commute sufficiently with the synthetic anomaly residual in frozen DINO
feature space.

## Gate result

| measure | negative EV | requirement | result |
|---|---:|---:|---|
| correction cosine | `0.909973` | `>=0.90` | pass |
| correction norm ratio | `0.961589` | `[0.80,1.25]` | pass |
| anomaly cosine | `0.809715` | `>=0.90` | fail |
| anomaly norm ratio | `0.999497` | `[0.80,1.25]` | pass |
| RN rank correlation | `0.998590` | `>=0.90` | pass |
| RN neighbor retention | `0.996094` | `>=0.90` | pass |
| RN scale ratio | `1.001422` | `[0.80,1.25]` | pass |
| correction hard-null margin | `+0.267925` | `>=+0.01` | pass |
| pixel-MAE gain over wrong sign | `+0.985643` | `>=+0.10` | pass |
| negative/positive pixel-MAE ratio | `0.05171` | `<=0.20` | pass |
| negative-positive correction cosine | `+0.726965` | `>=+0.10` | pass |
| exact repeat | max diff `0.0`, same signature | `<=1e-7`, exact hash | pass |

The aggregate gates were
`SATURATION_ASYMMETRY=true`, `NEGATIVE_EV_REVERSIBLE_SIGNAL=false`,
`repeatability=true`, and `structural_custody=true`.  Because the positive
diagnostic verdict required both scientific gates, the frozen verdict is
`NO_REVERSIBLE_SIGN_SIGNAL`.

## Remote execution and custody

- Completed run:
  `siai_d1_j0_sign_diagnostic_stage_minus1_dsba3_20260827_d1sign01_v2`.
- Host/container/GPU: dsba3, fixed project container
  `hun_fsad_tta_012`, GPU0 only.
- Remote preflight passed `17/17` focused tests, exact source/controller hash
  audit, and offline DINOv2 loading.
- Repeat maximum absolute difference: `0.0`; scientific-signature SHA-256:
  `7b4da4230ab33097afb8bab3aa3cffc517c4ff3ecc80d27854c7734946a1e001`.
- Immutable snapshot SHA-256:
  `0df86617365c7e9df01095275a8da9988dea597ef46be30e7a512431f2fd248a`.
- Aggregate SHA-256:
  `a35ca4e1ddbf80ce048246ed7bc97995566daade2b7e5ce5bb7ccefa94e402ef`.
- Source-manifest SHA-256:
  `1d83713d8f325a992c45ed73db95ed972681e361c494d6b814abee31dc195642`.
- `support_only=true`, `benchmark_comparable=false`; public/private/test/GT
  and query-memory paths remained sealed.  Only compact JSON/log evidence was
  retained, and the exact immutable snapshot was removed after pull.
- Only selected GPU0's index/UUID/memory/utilization/compute UUID, the fixed
  project container, and this run's exact paths were accessed.  No other
  user's container, process, file, volume, GPU, or job was enumerated or
  touched.  GPU0 returned to `1 MiB / 0%` after completion.

The preceding `v1` launch passed preflight but stopped before support loading,
feature extraction, or scientific output because the runner confused the
inherited SIAI-S1 module with its inner runtime object.  The sole reference
repair was recorded as a pre-result amendment, identities were refrozen, and
the exact failed snapshot was removed before `v2`.

## Continuation boundary

Stop explicit image-space exposure inversion.  Do not remove or downweight
positive-EV cases, weaken the anomaly threshold, tune exposure magnitude, add
a clipping-aware inverse, or promote this posthoc result to benchmark evidence.
The useful signal is narrower and architectural: a nuisance intervention can
canonicalize normal frozen features when information is retained, but anomaly
residuals do not commute with that canonicalization.  Any continuation must be
separately named and must test anomaly-preserving feature commutation directly;
it may not revive the closed synthetic QKV-LoRA, family-routing, CRFR, score
bypass, or explicit pixel-inverse branches.
