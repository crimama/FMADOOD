# Protected Residual-Consensus RN Stage -1 Preregistration

Date: 2026-08-16
Protocol: `prc-rn-stage-minus1-20260816`
Target: MVTec AD2 single-image TESTpublic, reduced-object diagnostic

## Negative Evidence Intake

- Exact full-support normalized RN is the immutable baseline.
- Tangent subtraction/quotient, additive geometry penalties, and CF-GC-RN local
  offset/neighborhood selection failed against RN1.
- Global spatial-offset consensus is closed as a retread of the failed spatial
  conditioner and CF-GC-RN branches.
- This experiment instead tests a feature-residual nuisance hypothesis.  The
  hypothesis may select a different member of the already frozen RN top-5, but
  it may not modify query features, memory, parameters, or the selected
  candidate's normalized RN distance.

## Claim, Evidence, Boundary, Positioning

- Claim: reliable query patches share an image-level feature residual caused by
  nuisance, while anomalous patches are inconsistent with that residual.
- Evidence sought: a robust residual code improves candidate selection over
  exact RN1 and cannot be reproduced by code/pair/confidence nulls.
- Boundary: reduced-object full-support diagnostic only; no SuperAD-16 or
  deployable claim.
- Positioning: protected test-time correspondence inference, not gradient TTA
  and not flow/transport score correction.

## Frozen Method

For query patch `p` and frozen exact-RN top-5 candidate `k`, let
`r[p,k] = q[p] - m[p,k]` and let `d[p,k]` be the original normalized RN
distance.  Select the deterministic lowest-distance half of top-1 patches,
estimate `c` by a coordinate-wise median of their `r[p,0]`, and compute a
robust explained-energy ratio.  If the ratio is below `0.10`, use the identity
path and reproduce exact RN1.  Otherwise select
`k*(p)=argmin_k ||r[p,k]-c||^2` and emit exactly `d[p,k*(p)]`.

No score subtraction, quotient, fusion, rescaling, memory write, parameter
update, or query-feature update is allowed.  Ties use deterministic rank order.

## Arms

- `B0_exact_RN1`: exact full-support normalized RN top-1.
- `M1_protected_residual_consensus`: robust code plus confidence fallback.
- `C0_always_on`: same robust code with fallback disabled.
- `N0_zero_code`: exact identity/no-op control.
- `N1_code_dimension_shuffle`: norm-preserving deterministic permutation of
  code dimensions.
- `N2_patch_pair_shuffle`: code estimated from deterministic mismatched
  query/support top-1 patch pairs.
- `N3_untrimmed_mean`: all-patch arithmetic-mean code.
- `N4_confidence_shuffle`: deterministic shuffle of the B0 ranks used to form
  the robust inlier half.

## Data And Comparators

- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `rice`, `vial`.
- Support: full `train/good` (`rice=313`, `vial=291`).
- Evaluation: full `test_public/good,bad` (`rice=42/90`, `vial=35/105`).
- Backbone/preprocessing/evaluator: identical to the same-run exact RN1 branch.
- SuperAD-16: `N/A`, `comparable=false`; this is a different-budget reduced
  diagnostic.
- Primary metrics: `seg_AUROC_0.05`, `seg_F1`, `seg_AUPRO_0.05`, complete map
  coverage, and normal FPR.

## Gates

Strict claim promotion is unavailable for this reduced panel.  A
`PROMISING_DIAGNOSTIC` requires all of:

- macro `M1-B0 seg_AUROC_0.05 >= +0.002`;
- `M1-B0 seg_F1 >= -0.005` and `seg_AUPRO_0.05 >= -0.005`;
- complete finite/nonconstant map coverage;
- `M1` beats every `N1`--`N4` null on pAUROC;
- exact score equality to B0 on every fallback query;
- no memory/parameter/query-feature update and no TESTprivate access.

`CONTINUE_DIAGNOSTIC` requires either macro pAUROC improvement or conditional
bad-vs-good separation AUC at least `0.55`, with no metric worse than `-0.005`
and at least one causal null margin.  Hard-stop this formulation if M1 does not
beat B0, is matched by all hard nulls, or inflates normal FPR by more than
`0.02`.

## Execution

Run one immutable shard per object on idle project GPU3 resources after local
unit/contract tests, remote dataset/hash/GPU UUID preflight, and same-run RN1
verification.  Persist compact JSON evidence only; remove regenerable dense
maps after the conclusion.
