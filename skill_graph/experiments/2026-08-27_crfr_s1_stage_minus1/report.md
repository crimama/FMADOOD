# CRFR-S1 Stage -1 report

Date: 2026-08-27  
Protocol: `crfr-s1-cross-recomposed-factorized-representation-20260827`  
Status: `COMPLETED / SUPPORT-ONLY FOUR-GPU PANEL`  
Verdict: `INSUFFICIENT_CRFR_SIGNAL / END_SUPPORT_ONLY_SYNTHETIC_REPRESENTATION_BRANCH`

## Outcome

CRFR-S1 was deterministic and numerically trainable, but it did not identify a
causal content/nuisance factorization.  The candidate passed `0/12` causal
factor cells, `2/12` anomaly-isolation cells, `1/12` synthetic-contribution
cells, and `0/12` positive hard-null cells.  The preregistered immediate-stop
rule therefore applies.

The decisive diagnostic failure was nuisance-code non-identifiability.  In every held
cell, shuffling the nuisance code changed reconstruction error by essentially
zero: the relative M1 gain over `N1_SHUFFLED_CODE` ranged only from about
`-3.4e-5` to `+3.6e-5`; the maximum absolute cross-error change was
`6.99e-7`.  Zeroing the code hurt by at least 10% in `6/12` cells, so a
code-dependent offset could matter, but the code's held sample identity did
not.  The retained evidence cannot distinguish nuisance-code collapse from a
decoder that ignored sample-specific code variation; both are consistent with
the residual content/coordinate shortcut.

Global candidate correction cosine was `0.0585479`, correction norm ratio was
`0.3957309`, and candidate-minus-best-hard-null margin was `-0.3956439`.
Shifted-normal q99 reduction was positive (`+0.0460873`), but no held cell beat
its strongest hard null.  Median anomaly cosine/norm were `0.8740680/0.9883847`.
Anomaly retention was strong on `fabric/walnuts` and weak on `can/vial`, while
generator-ID balanced accuracy exceeded the `0.35` ceiling on
`can/fabric/vial` (`0.4375/0.6250/0.6875`).  Thus synthetic generator identity
remained decodable instead of being cleanly separated from the deployable
content correction.

The useful positive results are bounded but real: all four objects optimized
without divergence; can repeated with scientific-metric maximum absolute
difference `0.0` and an identical model-state SHA-256; anomaly magnitude was
globally preserved; and q99 improved on `can/vial`.  These establish a reliable
experimental substrate, not a viable representation method.

## Remote execution and custody

- Immutable scientific run:
  `crfr_s1_stage_minus1_dsba3_20260827Tcrfr01Z_v3`.
- Host/container: dsba3, fixed project container `hun_fsad_tta_012`.
- GPU map: `can/fabric/vial/walnuts -> 0/1/2/3`; can was repeated on GPU 0
  before GPUs 1--3 were opened.
- Remote preflight passed 20 focused tests, exact source-hash checks, offline
  DINOv2 loading, and an actual CUDA model/gradient smoke.
- Repeatability: scientific metrics maximum absolute difference `0.0`; M1
  state SHA-256
  `76bf3e092335502bc10b963e742e0c51bb98d4d2b0d45b0beba98f1880d45e07`.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Immutable snapshot SHA-256:
  `7d6c503fedd69ef17d5fb985f28f34709893110aad58a2dbd5f629b96f6cea4c`.
- Aggregate SHA-256:
  `686edb11500e7b22ccb1cc5c20f02b42f9ea5384bd8fdd263d6f68efd9010979`.
- Exact source hashes were runner
  `7112820affc861271cb5037600ac6a8ef259b885efb223026454d0d09dc02d9b`,
  core `0fa6aef7635b02406aa1d1b6e79cafd31570cd3595dcec9288d562940bfc2c3c`,
  aggregator
  `2fedd7c2dfe5575ced0459c2a016b2699e3945760eb05fad5c9d747ff5490b99`,
  and frozen readout
  `25dc7c78903294cf7787345f9e67a6aef612aae2c857a5cfb2061f9450a3b8cc`.
- All four object summaries, `12/12` cells, exact nine arms, three trained
  arms, 144 updates per arm, and traces at `1/36/72/108/144` were validated by
  the remote aggregator.
- `support_only=true`, `benchmark_comparable=false`; public/private/test/GT
  and query-memory paths remained sealed.  No dense maps were retained.
- GPUs 0--3 returned idle after aggregation.  The compact remote result was
  retained and only this exact run's immutable snapshot was removed.
- Only the fixed project container, selected GPU UUIDs, and this run's own
  files were inspected or modified.  No other user's container, process,
  file, volume, or job was enumerated or touched.

Two earlier attempts were result-free operational stops.  `v1` exposed an
incorrect unit-test expectation for the preregistered inactive residual loss;
`v2` exposed lazy DINO wrapper initialization during model smoke.  Both were
fixed before any result-bearing training and neither contributed scientific
metrics.

## Frozen design

- Frozen DINOv2 ViT-L/14 features at the immutable TARCD readout.
- Residual 64-wide content head, 8-D nuisance head, and 64-wide crossed
  recomposer; no LoRA, router, retrieval memory, or second score stream.
- Fit quadruplets `x,T(x),A(x),T(A(x))`, exact 144-row schedule, AdamW
  `1e-4`, seed `20260826`.
- Candidate `M1_CRFR_CROSS`; same-view `N2_NO_CROSS`; independently trained
  `N3_CLEAN_ONLY`; zero-code, shuffled-code, static-mean, identity, frozen-RN,
  and paired-teacher held controls.
- Deployment discarded nuisance/recomposer heads and used only the content
  representation with unchanged exact 1-NN RN scoring.

## Aggregate result

| measure | observed | progressive requirement | result |
|---|---:|---:|---|
| structural/custody | true | true | pass |
| repeatability | exact | `<=1e-7` metrics + exact state | pass |
| causal-factor cells | `0/12` | `>=6/12` | fail |
| anomaly-isolation cells | `2/12` | `>=9/12` | fail |
| synthetic contribution | `1/12` | `>=3/12` | fail |
| positive hard-null cells | `0/12` | `>=3/12` with two objects | fail |
| global hard-null margin | `-0.3956439` | `>=-0.094` | fail |
| global q99 reduction | `+0.0460873` | `>=0` | pass |
| strict/progressive gate | false / false | true | fail |

### Object medians

| object | correction cosine | correction norm | hard-null margin | q99 reduction | anomaly cosine | anomaly cells | generator BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| can | `0.0223518` | `0.3783291` | `-0.3402069` | `+0.0956924` | `0.7912269` | `0/3` | `0.4375` |
| fabric | `0.0947440` | `0.4574821` | `-0.4510809` | `-0.0035179` | `0.9624970` | `2/3` | `0.6250` |
| vial | `0.2523149` | `0.3663804` | `-0.4719560` | `+0.1829881` | `0.7766499` | `0/3` | `0.6875` |
| walnuts | `-0.0282022` | `0.4131326` | `-0.1869957` | `-0.0046554` | `0.9569092` | `0/3` | `0.2500` |

## Mechanism analysis

1. The nuisance code was not causally identified.  `M1` and
   `N1_SHUFFLED_CODE` reconstruction errors were almost identical in all
   `12/12` cells.  Correct code pairing therefore carried no measurable held
   value.
2. The result is consistent with an architectural shortcut.  The recomposer
   retained a direct residual content path and fixed coordinates, while the
   content head was simultaneously trained toward canonical features.  A
   code-dependent mean offset could reduce reconstruction loss without using
   the instance's actual nuisance.  Because held code variance was not stored,
   this remains a mechanism inference rather than a uniquely proven cause.
3. Crossed factorization did not beat the same-view control robustly.  Relative
   gains over `N2_NO_CROSS` changed sign across objects/families, and no cell
   jointly passed the null gains plus correction direction/scale gate.
4. Synthetic supervision remained entangled with generator identity.  Three
   objects exceeded the generator-decoding ceiling, while only one cell showed
   strict M1-over-N3 synthetic contribution.
5. Lower training losses did not yield a useful canonical correction.  The
   global correction cosine fell far below Protected-PCGrad (`0.3664`) and
   Oracle-Family (`0.4341`), and every hard-null margin was negative despite
   positive q99 reduction on two objects.

## Continuation boundary

Do not tune CRFR head width, code dimension, decoder depth, loss weights,
learning rate, seed, generator/shift strength, threshold, or training length.
Do not add a contrastive term to the same residual recomposer and call it a
continuation: that would reopen the explicitly closed support-only synthetic
representation branch after observing its held failures.

Any next experiment must be separately named and must make nuisance action
structurally identifiable rather than hoping an unconstrained decoder uses a
latent code.  It must retain exact RN, the same support-disjoint held G2 and
hard-null controls, support-only custody, and public/test fail-closed gates.
No benchmark or public-performance claim is licensed by this result.
