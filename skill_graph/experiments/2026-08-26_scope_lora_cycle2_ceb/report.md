# SCOPE-LoRA progressive Cycle 2 report

Date: 2026-08-27  
Protocol: `scope-lora-cycle2-calibrated-excess-bypass-20260826`  
Status: `COMPLETED / SUPPORT-ONLY FOUR-GPU PANEL`  
Verdict: `INSUFFICIENT_PROTECTED_SCORE_SIGNAL / END_OUTER_LOOP`

## Outcome

Cycle 2 reproduced the useful rank-4 projected-teacher signal without a
structural or custody failure.  The candidate reduced the shifted-normal q99
tail by `0.0265548389`, and every object passed the preregistered synthetic
no-harm comparison.  The calibrated excess bypass itself added no anomaly
ranking value: global `PT_CEB` and `PT_P` pAUROC were both `0.1444675213`, so
their margin was exactly `0.0`.

The spatial-specificity and static-null-value gates therefore failed together.
This is not the single narrow failure required for a Cycle-3 repair.  The
preregistered CEB outer loop is closed; lowering q99 to q95 or tuning a bypass
threshold in this protocol is not authorized.

## Remote execution

- Immutable run:
  `scope_lora_cycle2_ceb_dsba3_20260826T1640Z_v1`.
- Host/container: dsba3, fixed project container `hun_fsad_tta_012`.
- GPU mapping: `can/fabric/vial/walnuts -> 0/1/2/3`.
- Preflight: `24/24` focused remote tests passed; offline DINO cache and an
  actual CUDA DINOv2/zero-init LoRA model smoke passed.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Snapshot SHA-256:
  `0985104975250ea7ab2f3ac2fe80544db4d2844be790c1d65a6173b3b2f8181b`.
- All four object summaries, all twelve cells, and all 64 spatial-null
  replicas per cell were complete.  No public/private/test/GT/query-memory
  access occurred and no dense anomaly maps were retained.

## Gate result

| gate | result | evidence |
|---|---|---|
| structural/custody | pass | all exact errors `<=1e-7`; hashes and split counts matched |
| normal utility | pass | global q99 reduction `0.0265548389`; held-normal activation `0.0078125` |
| synthetic protection | pass | every object and global median within the frozen no-harm envelope |
| spatial specificity | fail | `PT_CEB-PT_P=0.0`; spatial-null q95 `0.0`; strict superiority absent |
| static-null value | fail | pAUROC `0.1444675` vs N3 `0.1440568`, q99 `0.0265548` vs N3 `0.0289189`; no `+0.01` strict margin |

Object-level q99 reduction was strongly heterogeneous: `can 0.05185`,
`fabric 0.00126`, `vial 0.11894`, and `walnuts 0.00084`.  This is a positive
normal-correction signal, but it is not evidence for a query-specific bypass.

## Diagnosis

The CEB activation condition is only

`S_frozen - S_PT > tau_family`.

The held G2 anomaly mask never constructs the bypass; it only partitions the
already produced activation for evaluation.  In the completed panel,
activation was nearly absent inside G2 and appeared only sparsely outside.
Consequently the bypass restored nuisance-tail differences rather than
anomaly-aligned evidence.  The max-of-four calibration-image q99 threshold
made that starvation stronger, but threshold conservatism alone cannot explain
the simultaneous failure to beat the spatial shuffle and privileged static
mean.

## Continuation decision

The following facts are retained for a separately named LoRA-centered loop:

1. paired shift intervention contains a real normal-tail correction signal;
2. the correction is highly family/object dependent;
3. a shared adapter or score-only bypass has not beaten the shift-family static
   template;
4. the next useful test is an oracle family-routed LoRA upper bound, not a CEB
   threshold sweep.

No benchmark/public claim is licensed by this result.
