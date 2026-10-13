# CRLP-Predict Stage -1 Report

Date: 2026-08-25  
Protocol: `crlp-predict-stage-minus1-support-only-20260825`  
Status: `COMPLETED / SUPPORT_ONLY_ALL8`  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Scope and custody

This experiment tests whether source-normal shifted-assumed augmentation and
source-derived synthetic anomalies contain a separable residual direction that
can support a learned shift correction without erasing anomalies. It is a
mechanism diagnostic, not a benchmark comparison.

- Dataset interface: MVTec AD2 `train/good` only and deterministic descendants.
- Frozen exact-RN memory: SuperAD16 manifest SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Representation: DINOv2 ViT-L/14 layer 11, 672 canvas, pooled `16 x 16`.
- Split: leave one object, one shift family, and one synthetic-generator family
  out together; every descendant of a held base remains sealed from fitting.
- `TESTpublic`, private test, ground truth, query images, query memory updates,
  and target synthetic fitting remained forbidden. SuperAD metrics are `N/A`
  and `benchmark_comparable=false`.

## Frozen mechanism

Gate A fits a rank-64 linear direction probe after exact geometric-mean norm
matching of shift and anomaly residuals. It must beat family-label, cyclic
source-base, matched-random, magnitude, base/support-spectrum, and
residual-spectrum controls while suppressing generator identity.

Gate B fits a source-only rank-64-to-rank-16 ridge map from shifted features to
paired clean shift residuals. It must reduce shifted-normal exact-RN q99 by at
least 5%, preserve synthetic-anomaly direction/norm and exact-RN
neighbor/distance topology, and beat no-op, target-shuffle, base-pair-shuffle,
magnitude-only, random-basis, and static-projection controls. Paired clean
shift subtraction is reported only as an unattainable oracle.

## Implementation and local verification

Added an isolated core, runner, fail-closed aggregator, dsba3 four-GPU wrapper,
focused tests, preregistration, and this report. ATAC-v1 and ATAC-SB-v2 remain
separate named protocols and were not reopened.

- `py_compile`: passed for core, runner, and aggregator.
- Ruff format plus critical `E,F,I,B,UP,RUF`: passed.
- Final focused pytest: `18 passed in 1.06s`.
- Remote wrapper: `bash -n` and `selftest=passed`.
- Deterministic synthetic smoke: byte-identical across two runs, SHA-256
  `be57af84cac3fe279db2e6ae4ca83631626f2969abfdd2acbdd9ffb02a57abef`;
  finite, nonconstant, support-only, probe BA `1.0`, and all forbidden flags
  false.
- Frozen-config SHA-256:
  `fac01b8f231704b2279d0e57ca9c0b1aefcbcad311bb0561a77b569fb9a1dab9`.

The first immutable attempt,
`crlp_predict_stage_minus1_dsba3_20260825T121632Z_v1`, stopped before writing a
sentinel result with `invalid projection dimensions`: the one-dimensional
magnitude control exposed an unnecessary `feature_dim >= 2` utility check.
No all-eight expansion occurred. The utility was corrected to admit the valid
one-dimensional projection and a direct regression test was added before a
new snapshot was launched.

## Remote execution and integrity

Successful immutable run:
`crlp_predict_stage_minus1_dsba3_20260825T122151Z_v2`.

- Fixed container: `hun_fsad_tta_012`.
- GPU0 `can` validity sentinel passed `12/12` valid cells and alone authorized
  the support-only expansion; scientific success was not used for selection.
- All-eight allocation: GPU0 `can -> fruit_jelly`, GPU1 `fabric -> rice`, GPU2
  `sheet_metal -> vial`, GPU3 `wallplugs -> walnuts`.
- Start/completion: `2026-08-25T12:21:56Z` to
  `2026-08-25T12:34:46Z`; stopped after `all8_support_only`.
- Snapshot SHA-256:
  `a6a2d75f50eeb4cc76fcf208e2bf8f5a71dde4e4c744ac9410ad1ab80f511c7e`.
- Compact manifest SHA-256:
  `a0ff9e9355ef7837f30ed3b3d9afe802d72bf6c9e594350d43f21834071ec83a`.
- Pulled result:
  `results/remote_runs/dsba3/crlp_predict_stage_minus1_dsba3_20260825T122151Z_v2`.
- Every pulled compact-file hash passed. Independent local reaggregation was
  byte-identical to the remote aggregate, SHA-256
  `cf52235e3f03326cbd72cc4969fea3234b4847b371597131255d2d3ba7c9f4fa`.

## Object results

All objects and all 96 held cells were numerically valid. No object passed
Gate A, Gate B, or their conjunction.

| object | valid cells | Gate-A cells | Gate-B cells | generator-ID BA | A-to-B Spearman | mechanism |
|---|---:|---:|---:|---:|---:|---:|
| can | 12/12 | 0/12 | 0/12 | 0.8100 | 0.0000 | fail |
| fabric | 12/12 | 0/12 | 0/12 | 0.8274 | 0.0341 | fail |
| fruit_jelly | 12/12 | 0/12 | 0/12 | 0.7778 | 0.0000 | fail |
| rice | 12/12 | 0/12 | 0/12 | 0.7680 | 0.0000 | fail |
| sheet_metal | 12/12 | 0/12 | 0/12 | 0.7656 | 0.0000 | fail |
| vial | 12/12 | 0/12 | 0/12 | 0.8503 | 0.0000 | fail |
| wallplugs | 12/12 | 0/12 | 0/12 | 0.8740 | 0.3745 | fail |
| walnuts | 12/12 | 0/12 | 0/12 | 0.8004 | 0.0437 | fail |

Aggregate counts are valid `8/8`, Gate A `0/8`, Gate B `0/8`, association
`1/8`, and mechanism `0/8`.

## Gate-A analysis: apparent separation is mostly shortcut signal

Across 96 cells, candidate direction BA was min/median/max
`0.1863/0.5957/0.9818`; AUROC was `0.1284/0.6893/1.0000`. Only 26 cells met
BA `>=0.70`, 40 met AUROC `>=0.75`, one beat every hard null by `>=0.05`, and
only four kept every low-order confound at BA `<=0.60`; no cell met the full
conjunction.

The maximum confound BA was `0.5341/0.9764/1.0000`. The winning hard null was
magnitude-only in 48 cells, residual-spectrum-only in 39, cyclic base-pair in
7, and family-label shuffle in 2. The best apparent candidate cell (`can`,
Gaussian-control shift, material blotch) reached BA/AUROC `0.9091/0.9557` and
hard-null margin `+0.1727`, but residual-spectrum BA was `0.7364`, above the
frozen confound ceiling. Conversely, the cleanest low-confound near-case
(`vial`, spatial illumination, structural scratch) had confound BA `0.5795`
but candidate BA/AUROC only `0.5909/0.5945` and margin `+0.0114`.

Generator identity was directly recoverable at BA `0.7656--0.8740` on every
held object versus chance `0.25` and the `0.35` ceiling. The synthetic families
therefore carry strong implementation signatures that cannot be interpreted as
a generator-invariant anomaly direction.

## Gate-B analysis: learned correction is neither effective nor unique

The learned candidate's shifted-normal q99 reduction was min/median/max
`-1.1359/-0.04375/+0.04947`; zero of 96 cells met the frozen 5% improvement.
Only 12 cells passed the full anomaly/RN preservation guard, and zero achieved
the required `+0.01` guarded-gain margin over matched controls. Candidate
anomaly cosine met `>=0.90` in 51/96 and norm ratio stayed in `[0.8,1.25]` in
71/96, but exact-RN neighbor retention passed only 24/96 normal and 19/96
anomaly cells. Rank passed 40/96 normal and 44/96 anomaly cells; distance scale
was less problematic at 88/96 and 89/96.

The best raw learned reduction was `4.9468%` on `walnuts` exposure, still below
the gate; it failed preservation, while the magnitude-only control reached a
larger guarded `6.4281%`. The candidate guarded-gain maximum was `1.3098%`, but
no cell beat its same-cell hard comparator.

The paired-clean-shift oracle separates mechanism feasibility from estimator
failure: it passed preservation in 93/96 cells and reached q99 reduction
`>=5%` in 40/96, versus 12/96 and 0/96 for CRLP. Thus subtractive correction is
not intrinsically impossible, but this source-only low-rank predictor cannot
estimate a safe, useful shift residual under the frozen cross-object/family
generalization contract.

## Interpretation and continuation boundary

The broad hypothesis—use both shifted-assumed augmentation and synthetic
anomalies during source learning—is not logically refuted. What is refuted is
this specific decomposition: residual-direction probing plus a shift-only
low-rank predictor. Gate A shows that most separability comes from magnitude,
residual spectrum, or generator identity; Gate B shows that the learned
correction usually worsens the normal RN tail and never adds value beyond
matched controls.

CRLP-Predict-S1 is closed. Do not tune its ranks, ridge, thresholds, shift
strengths, generator strengths, or folds, and do not open public/test data. A
separately named successor would first need a generator-invariant paired
quadruplet representation over `x, T(x), A(x), T(A(x))`, with explicit
cross-recomposition/interaction consistency and source-only RN geometry
constraints. Its first gate must prove both that generator identity is near
chance and that a learned shift estimate approaches the paired-shift oracle
without matched-control equivalence; otherwise retain exact RN/no correction.

## Cleanup

Every forbidden-access flag is false in the sentinel, all-eight shards, and
aggregates. Local and remote compact roots contain zero `anomaly_maps/`
directories. After completion dsba3 GPUs 0--3 were all back to 1 MiB and 0%
utilization.
