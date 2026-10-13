# Transport-Quotient RN TESTpublic Preregistration

Date: 2026-08-13
Protocol ID: `transport-quotient-rn-testpublic-20260813`
Status: `PRE-REGISTERED / NOT RUN`
Maximum verdict: `PROMISING_DIAGNOSTIC`

## Boundary

All-eight MVTec AD2 single-image `test_public` diagnostic with full per-object `train/good` support. This is not SuperAD16-comparable because the reference budget is full support, not the same 16-shot support manifest.

Dense maps remain RAM-only. Ground truth may be loaded only after all arm maps for an object are generated. No gating, cap, query adaptation, memory update, parameter update, `test_private`, or filename-condition routing is allowed.

## Arms

- `B0_RN1`: exact normalized final-layer RN rank-1 distance.
- `B1_exact_anchor_orbit`: primary exact-anchor transport quotient score.
- `M0_topk_orbit`: top-k orbit comparator from the same core score family.
- `N0_shuffled_tangent`: shuffled-tangent hard null.
- `C0_no_tangent`: top-k metric comparator with RN term and residual but no tangent quotient.

The unified score contract is RN candidate distance plus orthogonal residual plus support-variance-regularized tangent-coordinate penalty. Tangent evidence is not a gate or cap.

## Metrics

Primary aggregate metric: macro `seg_AUROC_0.05`.
Secondary metrics: macro `seg_F1`, macro `seg_AUPRO_0.05`, and coverage.

The aggregate reports `M0-B0`, `M0-B1`, `M0-N0`, `M0-C0`, `B1-B0`, and `C0-B0` contrasts for mechanism diagnosis. A strict public-shadow signal requires top-k orbit improvement over RN1, exact-anchor orbit, shuffled tangent, and no-tangent comparators on primary pAUROC with complete coverage. Because the primary runner arm is `B1_exact_anchor_orbit`, report interpretation must separate runner primary from aggregate top-k mechanism contrasts.

## Pre-result Rule

If any shard fails schema, support/test path-set, RAM-only map audit, full support diagnostic, or no-gating score component custody, no partial object result may be used as positive evidence.
