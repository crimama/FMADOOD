# TARCD-LoRA-S1 Stage -1 Report

Date: 2026-08-26  
Protocol: `tarcd-lora-s1-stage-minus1-support-adapted-20260826`  
Status: `COMPLETED / SINGLE_SEED_SUPPORT_ADAPTED_PANEL`  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Scope and custody

This experiment tests whether target-support-fitted backbone PEFT can learn a
shift correction from paired shifted-assumed augmentation and synthetic
anomalies while retaining anomaly and exact-RN geometry. It is a support-only
representation mechanism diagnostic, not a benchmark comparison.

- Seed: exactly one, `20260826`; no post-result seed replication or sweep.
- Objects: `can`, `fabric`, `vial`, and `walnuts`.
- Fit custody per object: the exact 16 `train/good` paths in the SuperAD-16
  manifest, SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Evaluation custody per object: four support-disjoint `train/good` bases whose
  complete deterministic descendant trees were sealed from fitting.
- Fit views: clean target support plus deterministic shifted and synthetic
  descendants. The held generator was `G2_structural_scratch`; the three held
  shifts were exposure, Gaussian RMS-matched control, and spatial illumination.
- `TESTpublic`, private test, ground truth, query images, query-memory updates,
  and public authorization remained forbidden. SuperAD metrics are `N/A` and
  `benchmark_comparable=false`.

## Frozen mechanism

The student adds rank-4, alpha-8 fused-qkv LoRA to DINOv2 ViT-L/14 blocks
8--11. Every trained arm runs exactly 144 updates. The frozen teacher anchors
clean and anomaly representations over paired quadruplets
`x, T(x), A(x), T(A(x))`.

- `M0_CANONICAL_KD`: canonical teacher distillation plus identity anchoring.
- `M1_TARCD_FULL`: `M0` plus exact-RN relation/scale preservation and the
  anti-absorption hinge `relu(|cos(delta_shift,r_anom)|-0.20)`.
- Hard controls: shuffled quadruplets, random norm-matched LoRA, and static
  mean correction; zero-init identity and frozen-teacher controls establish
  no-op behavior. A paired-teacher arm is an unattainable oracle.

A held cell requires at least 5% shifted-normal q99 reduction, at least +0.01
preservation-guarded gain over every hard control, anomaly cosine/norm
retention, normal and anomaly RN topology retention, and candidate correction
cosine at least 0.70 with norm ratio in `[0.50,1.50]`. All three cells plus
generator-confound gates are required for an object pass.

## Implementation and local verification

Added a separately named LoRA core, runner, fail-closed aggregator, dsba3
four-GPU controller, focused tests, preregistration, and this report. Frozen
ATAC and CRLP protocols were not modified or reopened.

- `py_compile`: passed for the core, runner, and aggregator.
- Focused pytest with the project environment: `18 passed in 1.21s`; the same
  suite also passed under system Python (`18 passed`). The project-environment
  run exercised actual torch model paths.
- Remote wrapper: `bash -n` passed and `selftest=passed`.
- Deterministic CPU synthetic smoke passed with finite, nonconstant maps and
  all forbidden-access flags false.
- Model smoke verified M1 relation/margin execution, zero-init identity,
  detached-prefix/frozen-gradient boundaries, and aligned, anti-aligned, and
  orthogonal margin controls.
- Frozen-config SHA-256:
  `707be77dbc28dfe7a837d468339e5987b35cd0e1b8eb7e2a08c0963d55ec44fa`.

Three immutable preflight attempts stopped before any scientific gate or
result root was opened:

1. `tarcd_lora_s1_stage_minus1_dsba3_20260825T161336Z_v1` found that the fixed
   container does not include pytest; preflight was changed to direct smoke
   checks rather than installing or changing the container.
2. `tarcd_lora_s1_stage_minus1_dsba3_20260825T161420Z_v2` exposed CUDA
   nondeterministic backward in `adaptive_avg_pool2d`.
3. `tarcd_lora_s1_stage_minus1_dsba3_20260825T161530Z_v3` exposed an invalid
   integer-grid assumption for the `can` token grid (`48 x 104`).

The pooling path was replaced with exact deterministic adaptive-bin row/column
means, whose forward values were checked against adaptive average pooling for
arbitrary grids, before launching a new snapshot.

## Remote execution and integrity

Successful immutable run:
`tarcd_lora_s1_stage_minus1_dsba3_20260825T161712Z_v4`.

- Fixed container: `hun_fsad_tta_012`; CUDA model smoke passed on GPU0.
- GPU0 `can` sentinel passed validity and authorized expansion using validity
  only, not scientific success.
- Panel allocation: GPU0 `can`, GPU1 `fabric`, GPU2 `vial`, GPU3 `walnuts`.
- Start/completion: `2026-08-25T16:17:42Z` to
  `2026-08-25T17:19:03Z`; status `complete_support_only`, stopped after
  `panel_support_only`.
- Snapshot archive SHA-256:
  `81f7a9098f0740c9e19f9de0d6eac02190e189847676940541cc19c0271acb2a`.
- Compact hash-ledger SHA-256:
  `2cea8940694ed2979e79e0d7a1dcbaa18e119052b3213d80865bd65818fec051`.
- Pulled result:
  `results/remote_runs/dsba3/tarcd_lora_s1_stage_minus1_dsba3_20260825T161712Z_v4`.
- All 25 remote-ledger entries plus the two local launch-audit files were
  retained; every recorded hash passed. The aggregator independently
  reconstructed fit and held pathsets from the exact manifest and data root.
- Independent local reaggregation was byte-identical to the remote panel
  aggregate, SHA-256
  `bbc392e0e1908f467371670e20fe44772834560f35d97abdf5b6a1d3e9187433`.

## Object results

All four objects and all 12 held cells were valid and finite/nonconstant, but
no held cell or object passed the mechanism gate.

| object | GPU | valid cells | pass cells | median M1 q99 reduction | generator-ID BA | max low-order BA | mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|
| can | 0 | 3/3 | 0/3 | 6.47% | 0.2917 | 0.2500 | fail |
| fabric | 1 | 3/3 | 0/3 | 0.29% | 0.3958 | 0.3750 | fail |
| vial | 2 | 3/3 | 0/3 | 9.62% | 0.2500 | 0.3125 | fail |
| walnuts | 3 | 3/3 | 0/3 | 0.23% | 0.2500 | 0.3125 | fail |

The median of object-level candidate q99 reductions was `3.381%`, below the
5% cell floor. Fabric also exceeded the generator-ID ceiling of `0.35`; all
reported low-order confound probes stayed below `0.60`. Every object had
median hard-null margin `-1.0`, because no candidate cell satisfied the full
preservation guard needed to define a guarded positive gain.

## Gate analysis

Across the 12 held cells, M1 passed the isolated components as follows:

| component | passing cells |
|---|---:|
| shifted-normal q99 reduction `>=5%` | 5/12 |
| normal and anomaly RN topology | 6/12 |
| anomaly cosine and norm retention | 6/12 |
| paired-teacher correction cosine and norm | 0/12 |
| complete preservation guard | 0/12 |
| complete cell gate including hard-null margin | 0/12 |

The universal fatal failure was the correction direction. M1 correction
cosine had median `0.0667` and maximum `0.5125`, never reaching the frozen
`0.70` floor; median correction norm ratio was `0.3435`, also below the lower
bound. The model could sometimes lower normal RN tails or preserve anomaly
geometry, but it did not approximate the causal paired shift correction.

The auxiliary objective did not improve on canonical distillation. M1 q99
reduction was lower than M0 in all `12/12` cells, with median difference
`-0.01142` (about -1.14 percentage points). Median control behavior was:

| arm | q99 reduction | correction cosine | correction norm | anomaly cosine | normal/anomaly RN rank |
|---|---:|---:|---:|---:|---:|
| M0 canonical KD | 3.90% | 0.2713 | 0.5586 | 0.9108 | 0.9061 / 0.9112 |
| M1 TARCD full | 0.32% | 0.0667 | 0.3435 | 0.9119 | 0.8705 / 0.8770 |
| N1 shuffled quadruplet | 23.55% | 0.1014 | 2.2703 | 0.8457 | 0.6608 / 0.6727 |
| N3 static mean | 3.15% | 0.5095 | 0.7216 | 0.9166 | 0.9869 / 0.9873 |
| paired-teacher oracle | 4.02% | 1.0000 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |

The shuffled control demonstrates why raw q99 reduction is insufficient: it
produced the largest median reduction while substantially damaging anomaly and
RN geometry and missing the correction direction. Conversely, the paired
oracle satisfies correction, anomaly, and topology retention by construction,
showing that the gate is attainable even though q99 improvement is
shift/object dependent.

## Repeatability audit

The sentinel and panel repeated `can` from the same immutable snapshot. The
M1 adapter state, all M1 held-cell metrics, M1 training summary, and candidate
object medians were byte-identical. Zero-init identity and random-matched LoRA
states were also identical.

However, the whole-control repeat was not byte-identical: M0 adapter hashes
changed from `f2c8faac...` to `99c6387d...`, and N1 from `db74b2a1...` to
`0617092f...`, with small associated numeric differences. This is an
additional audit failure under the preregistered nondeterminism clause. It
does not rescue or reverse the candidate result—the candidate itself repeated
exactly and already failed every held correction gate—but deterministic
training quadruplets or cached features must be established before any new
LoRA protocol is trusted.

## Interpretation and continuation boundary

The broad research thesis is retained: shifted-assumed augmentation and
synthetic anomalies can both be useful training signals. What this result
rejects is the specific TARCD-LoRA-S1 construction in which teacher-anchored
fused-qkv LoRA, RN relation loss, and a local orthogonality hinge are expected
to discover a paired shift-correction direction.

The evidence is consistent with over-normalization or representation rotation:
M1 sometimes reduces nuisance score while preserving an anomaly summary, but
its correction does not point toward the paired teacher target and it is worse
than simpler canonical KD on normal-tail reduction. Therefore no additional
seed, rank, loss-weight, threshold, generator-strength, or learning-rate sweep
is authorized for this protocol, and public/test data must remain closed.

A future experiment must be separately named and preregistered. The most
direct mechanism change is an explicitly correction-supervised two-head or
factorized representation: one head must predict the paired teacher shift
correction, while a separate anomaly head preserves the residual under
cross-recomposition. Its first support-only gate should require deterministic
repeat training, held-generator invariance, and paired-correction agreement
before any public benchmark branch.

## Conclusion and cleanup

TARCD-LoRA-S1 is valid as an experiment but fails its scientific claim:
mechanism objects `0/4`, cells `0/12`, median hard-null margin `-1.0`, verdict
`KILL_FOR_CLAIM / NO_CONTINUE`. No public/test/private/GT path was opened.
The pulled compact root contains zero `anomaly_maps/` directories, and dsba3
GPUs 0--3 returned to 1 MiB allocation and 0% utilization after completion.
