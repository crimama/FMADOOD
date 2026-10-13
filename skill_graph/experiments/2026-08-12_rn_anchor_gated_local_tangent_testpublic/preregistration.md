# RN-Anchor Gated Local Tangent TESTpublic Preregistration

Date: 2026-08-12  
Protocol ID: `rn-fmlk-exact-anchor-gated-tangent-testpublic-20260812`  
Status: `PRE-REGISTERED / NOT RUN`  
Maximum verdict: `PROMISING_DIAGNOSTIC`

## 1. Boundary and custody

This is a TESTpublic-only public-shadow experiment over all eight MVTec AD2
single-image objects. The support pool is the full `train/good` population for
each object. No `test_private`, no query-time memory update, no parameter
update, no support update, no filename condition token, and no hidden split
access is allowed.

Public GT may be opened only after every arm's dense score map for the current
query batch has been generated and staged in RAM. Dense maps are not persisted;
the retained evidence is compact JSON/TSV only.

`A0_RNFMLK_final_layer_1nn_exact_anchor` is a mandatory hard null and must be present for every object. It is
not a convenience baseline: every positive claim is measured against the exact
RN final-layer normalized nearest-neighbor anchor and distance.

## 2. Frozen inputs

| object | full `train/good` count | TESTpublic count |
| --- | ---: | ---: |
| can | 412 | 162 |
| fabric | 387 | 156 |
| fruit_jelly | 263 | 80 |
| rice | 313 | 132 |
| sheet_metal | 137 | 114 |
| vial | 291 | 140 |
| wallplugs | 293 | 150 |
| walnuts | 432 | 150 |

The aggregator expects shard schema
`rn-gated-tangent-testpublic-object-v1`, protocol
`rn-fmlk-exact-anchor-gated-tangent-testpublic-20260812`, exact all-eight object
coverage, exact support/test-public path-set hashes, and per-arm RAM-only map
audits.

## 3. Arms

| arm | definition |
| --- | --- |
| `A0_RNFMLK_final_layer_1nn_exact_anchor` | mandatory hard null: normalized final-layer RN 1NN distance with exact anchor |
| `A1_RN_anchor_unconditional_local_tangent` | same RN anchor/distance with unconditional local tangent admission diagnostics |
| `A2_support_confidence_gated_tangent` | support-LOO confidence admits tangent correction only where reliable |
| `A3_capped_gated_tangent` | primary arm: exact RN distance with bounded multiplicative support-LOO gate |
| `N0_part_shuffled_tangent_control` | matched wrong-part gate/tangent hard null |
| `O0_oracle` | optional diagnostic oracle; excluded from metrics, gates, ranking, and claims |

The primary `A3` score preserves exact RN anchor and RN distance identity. The
only allowed correction is:

```text
d_rn * (1 - 0.25 * gate)
```

where `gate` is fit from support leave-one-out evidence only. Tangent score may
control admission, but it must not enter score magnitude. Each shard must record
`rn_anchor_identity_sha256` and `rn_distance_identity_sha256`; both hashes must
match across all objects.

## 4. Metrics and gates

Primary metric: macro `seg_AUROC_0.05`.

Secondary metrics: macro `seg_F1`, macro `seg_AUPRO_0.05`, and coverage.

The aggregate reports mandatory contrasts `A3-A0` and `A3-N0`, plus object
wins/no-harm counts. Strict claim requires `A3 > A0` on all three claim metrics:
`seg_AUROC_0.05`, `seg_F1`, and `seg_AUPRO_0.05`. `A3` must also beat the
part-shuffle null on primary pAUROC and preserve complete coverage.

Continuation requires all of:

- `A1_RN_anchor_unconditional_local_tangent` improves over the prior SPCT reference on primary
  pAUROC;
- `A2_support_confidence_gated_tangent` closes the RN gap at least as well as `A1`;
- `A3_capped_gated_tangent` closes the RN gap at least as well as `A1`;
- neither `A2` nor `A3` has any object pAUROC harm worse than `0.02` versus
  `A0_RNFMLK_final_layer_1nn_exact_anchor`.

## 5. Negative-evidence intake

This launch is not a score-fusion retread. Prior SPCT confidence, conformal,
synthetic-gate, and key-TTA evidence found no deployable gate under the previous
objective; full support improved raw/tangent retrieval but still left RN-FMLK as
the hard comparator. Therefore this protocol pins RN anchor identity and admits
only a bounded multiplicative gate fit from support LOO evidence. Arbitrary
linear or rank fusion of RN and tangent scores is forbidden.

SuperAD16 is budget-incomparable context unless a separate run uses the same
support budget and support manifest. A full-support run is diagnostic only and
cannot claim SuperAD16 dominance.

## 6. Pre-result rule

This preregistration is written before any RN-gated tangent TESTpublic result is
aggregated. If the final aggregate is invalid, no partial object result may be
used as positive evidence.
