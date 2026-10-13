# Source/Part-Conditioned Local Transport Full-Shot Stage -1 Report

Date: 2026-08-11  
Status: `COMPLETE / SUPPORT_ONLY DIAGNOSTIC`  
Verdict: `VALID_DIAGNOSTIC / RETRIEVAL_SIGNAL / TANGENT_SIGNAL / FULLSHOT_SIGNAL / SYNTHETIC_NO_SIGNAL`

## Scope

Paired Probe16 support-only synthetic diagnostic over `can`, `fabric`, `vial`,
and `walnuts`, comparing `support16` with `full_train_good` support. This is not a
benchmark result: `benchmark_comparable=false`; SuperAD, RN-FMLK, test, and
ground-truth metrics are `N/A`.

## Frozen method

See [preregistration.md](preregistration.md): frozen DINOv2 layer-5 retrieval,
layer-11 values, same-position cross-source graph, rank-4 local normal tangent,
and orthogonal-residual score. Synthetic margin arms are fit-only second-gate
diagnostics.

## Execution and results

All eight `(budget, object)` shards completed and passed custody, coverage,
finite/nonconstant-score, source-exclusion, and compact-artifact validation. The
full manifest is sealed at
`58f6b249d82ac713c9a0051675244db5216ebbe995d18ecb123058c9499a592d`.
Every eligible fit support participates in graph construction, retrieval, and
tangent fitting; only calibration queries and synthetic-margin queries are
bounded at 64 and 32 SHA-selected fit sources per fold.

| budget | macro R1 pAUROC | local | broad | normal FPR | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| support16 | 0.689831 | 0.666983 | 0.712680 | 0.058085 | retrieval/tangent signal |
| full_train_good | 0.769953 | 0.725394 | 0.814513 | 0.113487 | full-shot/tangent signal |

The paired full-shot gain was `+0.080122` pAUROC with a 10,000-replicate
hierarchical-bootstrap 95% CI `[+0.039551, +0.117081]`; all four objects won.
The local and broad family gains were `+0.058412` and `+0.101832`. This gain
comes with a held-normal FPR increase of `+0.055402`, so threshold calibration
must be repaired before any deployment or benchmark claim.

| object | R1 support16 | R1 full | full - 16 | full R1 - R0 | full R1 - N3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| can | 0.620858 | 0.720399 | +0.099541 | +0.077840 | +0.058287 |
| fabric | 0.638216 | 0.723643 | +0.085427 | -0.001250 | -0.000494 |
| vial | 0.782429 | 0.804646 | +0.022217 | +0.120840 | +0.108880 |
| walnuts | 0.717822 | 0.831125 | +0.113303 | +0.001182 | +0.001551 |

## Mechanism conclusion

Full-support R1 beat source shuffle by `+0.163114` (95% CI
`[+0.100758,+0.228462]`) and part shuffle by `+0.258039` (95% CI
`[+0.211773,+0.304253]`), with four of four object wins. Correct source/part
correspondence is therefore the strongest and most consistent causal factor.

The local tangent is real but heterogeneous. Full-support R1 beat the global
tangent by `+0.033025` (95% CI `[+0.000560,+0.068702]`) and the tangent shuffle
by `+0.042056` (95% CI `[+0.000334,+0.087711]`). The effect is concentrated in
`can` and `vial`; `fabric` and `walnuts` improve almost entirely from increased
retrieval coverage. The primary continuation should therefore be a
retrieval-confidence-gated local tangent, not unconditional projection.

The PCA32 nonnegative synthetic-margin head was exactly inactive:
`R2-R1=0`, `R2-N4=0` on every object and both budgets. This rejects the current
additive margin formulation, not the broader use of synthetic anomalies.
Synthetic examples should next supervise tangent activation, retrieval
confidence, or an orthogonal-residual rejection boundary rather than an
additive diagonal score head.

Clean-key retrieval O0 remained `+0.047015` above R1 at full support, and the
forbidden same-source O1 remained `+0.202715` above R1. Full-support oracle-gap
recovery reached `24.45%` on `can` and `47.04%` on `vial`, but was negligible on
`fabric` and `walnuts`. O0/O1 were excluded from every verdict.

## Execution note

The original CPU v1 attempt produced no accepted shards. CUDA v2 completed
seven shards, but `can/full` exceeded the 24 GiB dsba5 device when a repeated
full-tensor finite check allocated a 1.79 GiB boolean mask. The immutable v3
retry changed execution only: CPU fold prevalidation, source-chunk fallback,
and CUDA cache release between precompute/folds. Resident memory fell from
23.08 GiB at failure to 12.85 GiB, and the retry completed. The three already
valid dsba5 shards were imported byte-for-byte with explicit retry provenance.

Final compact result:
`results/remote_runs/merged/spct_stage_minus1_transport_fullshot_cuda_v3_can_retry/summary.json`.

## Required interpretation

O0/O1 are forbidden oracle ceilings. Their gap/recovery figures explain the
remaining retrieval insufficiency but did not select a model, threshold, or
positive claim. This result supports continued development of support-only
source/part retrieval plus confidence-gated local transport. TTA remains a
separate later lane after support-only calibration is repaired.
