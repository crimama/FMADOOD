# Transport-Quotient RN TESTpublic Report

Date: 2026-08-13 to 2026-08-14
Status: `COMPLETED / VALID_NO_SIGNAL`
Verdict ceiling: `PROMISING_DIAGNOSTIC`
Research decision: `KILL_FOR_CLAIM / NO_CONTINUE`

## Scope

This report records the completed all-eight TESTpublic transport-quotient RN
diagnostic. The support budget is the full per-object `train/good` population,
so the result is not directly comparable to SuperAD-16.

## Frozen Contract

- Dataset: MVTec AD2 single-image `test_public`.
- Support: full per-object `train/good` population.
- Runner primary arm: `B1_exact_anchor_orbit`.
- Required emitted arms: `B0_RN1`, `B1_exact_anchor_orbit`, `M0_topk_orbit`, `N0_shuffled_tangent`, `C0_no_tangent`.
- Score components: RN candidate distance, orthogonal residual, support-variance-regularized tangent-coordinate penalty.
- Forbidden: gating, cap, query adaptation, memory update, parameter update, persisted dense maps, `test_private`.

## Aggregation

Aggregator: `scripts/aggregate_transport_quotient_rn_testpublic.py`.

- Valid merged artifact:
  `results/remote_runs/transport_quotient_rn_testpublic/merged/transport_quotient_rn_testpublic_v6_chunked_20260814_reaggregate/summary.json`.
- Aggregate SHA-256:
  `832070deb4ec7b035642f01dd7c8eb6bb2998a2f93d906374f6f2abb5ae1700b`.
- Protocol verdict: `VALID_NO_SIGNAL`.
- Coverage: all eight objects and all five arms, `coverage=1.0`.

## Execution Readiness Update (2026-08-14)

The method and arm contract above are unchanged. The runner implementation was
restructured to remove execution-only bottlenecks before the first benchmark
snapshot:

- pass 1 keeps one normalized final-native support bank resident per object and
  reuses it for every query RN top-k search;
- pass 2 fits one deduplicated union cache of compact root statistics and drops
  the nuisance displacement tensor immediately afterward;
- pass 3 keeps support values and compact tangent statistics resident in one
  torch scorer shared by all queries and all B1/M0/N0/C0 arms;
- N0 retains its original query-seeded permutation and candidate provenance;
- resident RN and quotient states are released at their phase boundaries,
  including exceptional exits.

Focused verification passed 42 tests, lint, Python compilation, type checking,
remote-shell syntax, and CPU/CUDA parity. A bounded synthetic CUDA benchmark
(36,864 support rows, eight queries) preserved exact RN indices and distances
and reduced wall time from 0.5041 s to 0.0674 s (7.47x, including the one-time
upload). This is execution evidence only, not a TESTpublic result. The benchmark
status remains `PRE-REGISTERED / NOT RUN` until new immutable remote shards are
produced and aggregated.

## Remote Execution

- Failed execution-only run: `transport_quotient_rn_testpublic_v5_resident_20260814`.
- Active replacement: `transport_quotient_rn_testpublic_v6_chunked_20260814`.
- Immutable snapshot root:
  `/workspace/isolated_runs/transport_quotient_rn_testpublic/transport_quotient_rn_testpublic_v6_chunked_20260814/repo`.
- Remote result root:
  `/workspace/results_remote/transport_quotient_rn_testpublic/transport_quotient_rn_testpublic_v6_chunked_20260814`.
- dsba3 host GPU 3: `can -> fabric -> walnuts`.
- dsba5 host GPU 3: `fruit_jelly -> rice -> sheet_metal -> vial -> wallplugs`.
- Both queues passed manifest/data/RAM/free-CUDA-memory and host-container GPU
  UUID preflight before submission.
- Superseded run `transport_quotient_rn_testpublic_v4` was stopped after
  preserving its completed `fruit_jelly` and `rice` shards; it is not eligible
  for mixing with the v6 aggregate because its execution snapshot differs.
- v5 failed before producing any shard: the all-root torch gather attempted a
  52.73 GiB/33.04 GiB temporary on `can`/`fruit_jelly`. The v6 snapshot uses
  paired root/part gather in bounded root chunks, preallocated output, and no
  identity-remap duplication. Method scores and arm definitions are unchanged.

This protocol performs frozen feature extraction, support graph construction,
normal tangent-statistic fitting, and scoring. It does not perform gradient
training or query/test-time parameter updates.

## Final Results

Macro pixel metrics are decimals. `B0_RN1` is the matched full-support exact-RN
hard baseline.

| Arm | pAUROC@0.05 | F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| `B0_RN1` | **0.749400** | **0.336069** | **0.435772** |
| `M0_topk_orbit` | 0.652313 | 0.184042 | 0.289897 |
| `N0_shuffled_tangent` | 0.650009 | 0.184727 | 0.285249 |
| `C0_no_tangent` | 0.639220 | 0.176678 | 0.232629 |
| `B1_exact_anchor_orbit` | 0.626976 | 0.153130 | 0.268266 |

Per-object pAUROC@0.05:

| Object | B0 RN1 | B1 exact orbit | M0 top-k orbit | N0 shuffle | C0 no tangent |
|---|---:|---:|---:|---:|---:|
| can | 0.589333 | 0.528182 | 0.520597 | 0.512829 | 0.556863 |
| fabric | 0.711887 | 0.593506 | 0.616929 | 0.617870 | 0.603824 |
| fruit_jelly | 0.754181 | 0.666124 | 0.716067 | 0.716460 | 0.735666 |
| rice | 0.914328 | 0.728265 | 0.732005 | 0.724912 | 0.763647 |
| sheet_metal | 0.800723 | 0.606511 | 0.689429 | 0.693159 | 0.614623 |
| vial | 0.656234 | 0.641882 | 0.648896 | 0.647326 | 0.637867 |
| wallplugs | 0.680401 | 0.548054 | 0.566554 | 0.559206 | 0.506968 |
| walnuts | 0.888117 | 0.703286 | 0.728024 | 0.728313 | 0.694303 |

Key macro deltas:

- `M0-B0`: `-0.097088/-0.152026/-0.145875` on pAUROC/F1/AU-PRO.
- `B1-B0`: `-0.122424/-0.182939/-0.167506`.
- `M0-B1`: `+0.025336/+0.030913/+0.021631`.
- `M0-N0`: `+0.002303/-0.000684/+0.004648`.
- `M0-C0`: `+0.013092/+0.007365/+0.057268`.

`B0_RN1` beat `M0_topk_orbit` on pAUROC for all eight objects. M0 improved
over the weaker exact-anchor B1 and no-tangent C0 comparators, but was nearly
indistinguishable from the shuffled-tangent null and lost heavily to the raw RN
baseline. The aggregate continuation gate therefore failed.

## Interpretation

The useful component is the frozen exact RN correspondence itself, not the
current transport quotient. The orbit/tangent score adds a strong ranking
penalty that suppresses defect evidence faster than it removes nuisance. The
near-zero `M0-N0` margin also means the fitted local tangent orientation is not
the causal source of M0's residual strength.

This closes the present formulation of RN-conditioned local transport for a
performance claim. It does not disprove flow-matching or normal-transport ideas
in general; it says that adding the current support-derived tangent quotient to
an already strong RN comparison is harmful. Any later transport work must
preserve the RN score as an invariant lower-level signal and demonstrate a
non-shuffle-specific incremental term before public evaluation.

## Validity, Provenance, and Cleanup

- All eight shards report `validity=true`, complete good/bad TESTpublic
  coverage, `query_adaptation=false`, `memory_updated=false`,
  `parameter_updates_at_score=false`, and no `test_private` access.
- Full-support manifest SHA-256:
  `ac084290bd40d171f12e876b45c3baf9b2269010cc2e62a0e91d7c4b80badd53`.
- Frozen runner SHA-256:
  `0d431ea48312c064ac04de62749da51276f5908668055401625a858b5bb3d203`.
- The frozen snapshot aggregator SHA-256 was
  `b2721864847b7b25f6772fb4db8b2a55ccbbd54eff49a6f671877334b7e5a44c`.
  It rejected valid compact runner artifacts because it expected redundant
  top-level support/test path lists. The original invalid aggregate was kept.
- A fail-closed compatibility patch reconstructs support identity from the
  pinned manifest and TESTpublic identity from arm-consistent
  `per_image_audit`. Patched aggregator SHA-256:
  `89cf020975fcbe5263aa10583fb611a64a5c215579b405641abb9b1220049e90`.
  Focused verification passed `5` tests, Ruff, compilation, and diff checks.
- The merged compact checksum manifest verifies successfully. No dense map,
  tensor, image, NumPy, or pickle artifact is retained; the runner also reports
  `dense_maps_persisted=false` for every shard.
