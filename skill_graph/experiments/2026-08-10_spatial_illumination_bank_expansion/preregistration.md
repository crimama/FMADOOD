# Spatial-Illumination Memory-Bank Expansion Preregistration

Date: 2026-08-10

Protocol ID: `spatial-illumination-bank-expansion-all8-20260810`

Tier: mechanism diagnostic with a conditional promotion gate. Not a
SuperAD-comparable method claim.

## Negative Evidence Intake

`synthetic-shift-bank-expansion-all8-20260810` expanded the normal memory bank
with globally exposure-shifted support views and produced macro
`seg_AUROC_0.05 +.00036`, `seg_F1 +.00054`, `seg_AUPRO_0.05 +.00016` — roughly
`1/28` of its own strict gate. A public-split condition decomposition then
explained why, and that explanation is the entire motivation for this protocol.

Normal false-positive rate at the global oracle threshold, macro over eight
objects:

| condition group | content | normal FPR | vs regular |
| --- | --- | ---: | ---: |
| `regular` | reference | .00126 | — |
| `photometric` (`_overexposed`, `_underexposed`) | global exposure | .00140 | `+11%` |
| `_shift_1/2/3` | additional light sources | **.00284** | **`+125%`** |

The evaluator labels `_shift_*` as `geometric`. That label is wrong and is not
used here. Phase correlation against the matching `regular` image gives
alignment offsets of `(0,0)` to `±1` pixel — the objects do not move. What
changes is the *spatial structure* of illumination. Spatial coefficient of
variation of the multiplicative field `img / regular`:

| object | overexposed | underexposed | shift_1 | shift_2 | shift_3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| can | .048 | .055 | .274 | .144 | .240 |
| fabric | .016 | .017 | .050 | .133 | .116 |
| vial | .061 | .042 | .097 | .117 | .044 |
| walnuts | .023 | .027 | .241 | .087 | .193 |

The previous experiment therefore augmented along the axis that was already
benign (`+11%`, near-uniform, CV `.016--.061`) and left the damaging axis
(`+125%`, spatially structured, CV `.044--.274`) uncovered. Its augmentation
*model* was wrong, not the augmentation idea.

Also respected and not reopened: Stage -1c shared-direction projection removal,
foreground/background priors, the `k=5` scorer, DVT parameter sweeps, per-image
rank normalization, and identify-then-suppress on normal data.

**Adjacency declared.** Stage -1a1's conditional-smoothness gate was `0/4`. That
protocol attempted to *identify* a smooth nuisance field from data. This one
*synthesises* one from fixed parameters and never fits a field to any image. The
identify-versus-synthesise distinction established by
`normal-tail-structure-predictability-all8-20260810` — the nuisance is absent
from `train/good` and so cannot be learned from it — is what separates them. If
this protocol fails, Stage -1a1 is not revived and the smooth-field family
closes for synthesis as well.

## Research Question

Does expanding the static normal memory bank with **spatially structured**
synthetic illumination views reduce the `shift_*`-group normal false-positive
inflation and improve low-false-positive-rate performance, beyond the
unaugmented baseline, a spatial-frequency hard null, the already-measured global
exposure arm, and real same-condition normals?

## Dataset and Frozen Baseline Contract

Identical to `synthetic-shift-bank-expansion-all8-20260810`:

- `target_dataset=MVTec AD2 single-image`; objects
  `can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts`; full
  `test_public/good,bad`
- support: frozen DINOv3-H+ Baseline1-core support-16, SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- backbone DINOv3-H+ `[7,15,23,31]`, `layer_norm_mean`, resolution 672 except
  `sheet_metal=448`; DVT `position_mean`; density weight 0; `raw_1nn`;
  `rgb-guide none`; `threshold-calibration none`; `binary-postprocess none`
- forbidden: query-time update, condition/filename metadata **as a scoring or
  selection input**, `test_private`, `test_private_mixed`, fitting on
  `test_public`
- SuperAD fields `N/A`; `comparable=false`.

**Condition labels are used for post-hoc reporting stratification only**, via the
existing `condition_attribution` block, exactly as in the 2026-08-08 condition
attribution protocol. They never enter scoring, bank construction, augmentation
design, or arm selection.

## Flow Is Held Fixed

As before and for the same attribution reason: the Flow is fitted on the
**original 16 support views only**, with identical seed and epochs, and
augmented views are pushed through the **already-fitted, frozen** Flow to
populate the bank. Arms must report identical Flow parameter checksums, both
before and after bank population. This is a hard validity gate.

## Fixed Augmentation

Synthetic, parametric, deterministic. No field is fitted to any image.

In log space, a multiplicative illumination field over normalized coordinates
`u = x/W`, `v = y/H`:

```text
g_m(u,v) = cos(pi * kx_m * u) * cos(pi * ky_m * v)
I_m      = clamp( I * exp( a_m * (g_m - mean(g_m)) ),  0, 255 )
```

Seven views per support image, including identity:

| index | `(kx, ky)` | role |
| ---: | --- | --- |
| 0 | identity, `a = 0` | original view |
| 1 | `(1, 0)` | horizontal gradient |
| 2 | `(0, 1)` | vertical gradient |
| 3 | `(1, 1)` | diagonal gradient |
| 4 | `(2, 0)` | horizontal two-lobe |
| 5 | `(0, 2)` | vertical two-lobe |
| 6 | `(2, 2)` | centre-surround |

`a_m` is set per mode, before any test access, so that the spatial coefficient
of variation of the realized multiplicative field equals a **fixed target of
`0.15`** within `10%` relative. `0.15` is the approximate median of the observed
`shift_*` CV range `.044--.274` and is fixed here; it may not be swept.

`16 x 7 = 112` bank views per object — identical bank size to the previous
experiment's `E1`, which makes the two directly comparable.

## Arms

| Arm | Bank contents | Bank size | Purpose |
| --- | --- | ---: | --- |
| `F0_baseline` | 16 original views | 16 | frozen anchor (reuse `E0_baseline`) |
| `F1_spatial` | 16 x 7 spatial illumination views | 112 | candidate |
| `F2_highfreq_null` | 16 x 7 same-CV high-frequency views | 112 | spatial-frequency hard null |
| `E1_global` | 16 x 7 global exposure views | 112 | frozen anchor, reused from the previous protocol |
| `S128_real` | 128 real `train/good` normals | 128 | frozen anchor, reused from A1 |

`F2` uses the same seven-view structure and the same CV target `0.15`, but with
`(kx, ky)` scaled by `16` (i.e. `(16,0)`, `(0,16)`, `(16,16)`, `(32,0)`,
`(0,32)`, `(32,32)`). Equal magnitude, wrong spatial scale. This isolates
"illumination-like low-frequency structure" from "any spatially varying
perturbation of the same size".

Reused anchors are subject to custody verification and are rerun if custody
fails.

### Amendment 2026-08-10 — arm classes (made before any `F1`/`F2` data was observed)

The predecessor protocol `synthetic-shift-bank-expansion-all8-20260810` returned
`INVALID` because it gated the reused `S128_real` anchor as if it were a
contract arm. `S128_real` is an A1 real-normal ladder arm: its Flow is fitted on
128 images and its artifacts carry no `bank_size`,
`flow_fit_source_view_count`, or `flow_parameter_checksum` fields. It cannot
satisfy a Flow-fixed or bank-size gate by construction.

Two arm classes are therefore declared here:

- **Contract arms** — `F0_baseline`, `F1_spatial`, `F2_highfreq_null`,
  `E1_global`. Generated under this contract with the Flow fitted on the
  original 16 views. Subject to every validity gate.
- **External comparator** — `S128_real`. Subject only to support-manifest
  custody, object coverage, evaluator identity, and access-ledger checks. Exempt
  from the Flow-fixed and bank-size gates, which are meaningless for it.

This amendment was made after the predecessor's verdict and **before any
`F1_spatial` or `F2_highfreq_null` endpoint was observed** — both runs were still
executing on dsba3. It changes which arms a validity gate applies to. It does not
change any endpoint gate, threshold, metric, or verdict rule.

The predecessor also failed its parity gate because no duplicated arm was run. An
`F1_spatial` parity duplicate on the local RTX 4090 was launched while the dsba3
arms were still executing, so validity gate 9 has the replication it requires.

## Primary Metrics

- Primary mechanism endpoint: `shift_*`-group `good_pixel_fpr_at_global_oracle`,
  macro over objects, from `condition_attribution`.
- Primary performance endpoint: macro `seg_AUROC_0.05`.
- Co-primary preservation: macro `seg_F1` and macro `seg_AUPRO_0.05`.
- Secondary: per-condition-group defect TPR and component recall,
  `normal_image_pooled_fpr_at_oracle`, `gt_component_recall_at_oracle`,
  `pooled_pixel_ap_float16`, `pooled_pixel_auroc_float16`, per-object versions.

The mechanism endpoint is stated first deliberately. If the `shift_*` FPR does
not move, the augmentation did not do the thing it was designed to do, and any
performance change is unexplained.

## Validity Gates

All must pass before any endpoint is read:

1. all eight objects complete in every arm; full map/GT coverage; finite,
   nonconstant maps; no retained `anomaly_maps/` after conclusion;
2. exact support manifest custody; identical preprocessing, resolution, scorer,
   DVT mode, and evaluator across arms, asserted from the emitted run config;
3. Flow fixed: identical Flow parameter checksum across the **contract arms**
   `F0_baseline`, `F1_spatial`, `F2_highfreq_null`, `E1_global`, and unchanged
   before versus after bank population;
4. bank sizes exactly `16 / 112 / 112 / 112` for the contract arms, and
   `F1`/`F2` contain the 16 identity views as a subset;
5. realized spatial CV of every `F1` and `F2` field is within `10%` of the
   `0.15` target, and `F2`'s mode frequencies are exactly `16x` `F1`'s;
6. `support_brightness_range` and `support_transforms` byte-identical across all
   arms and recorded;
7. condition labels appear only in reporting, never in scoring, bank
   construction, or arm selection — asserted from the access ledger;
8. no query update, no private access, no fitting on `test_public`;
9. duplicated local/remote arm parity: maximum absolute difference of shared
   continuous metrics `<= 1e-3`.

## Mechanism Gate

`F1 - F0` macro `shift_*` `good_pixel_fpr_at_global_oracle` `<= -20%` relative.
If this fails, the verdict is `AUGMENTATION_DID_NOT_COVER_THE_SHIFT / NO_CONTINUE`
and the spatial-illumination synthesis family closes. A `20%` reduction is
required because the measured inflation is `+125%`; anything smaller does not
materially close that gap.

## Hard-Null Gate

Both must pass, else `HARD_NULL_ABSORBED` and the family closes:

1. `F1 - F2` macro `seg_AUROC_0.05` `>= +0.005`;
2. `F1 - F2` macro `shift_*` `good_pixel_fpr_at_global_oracle` `<= 0`.

## Model-Correction Gate

`F1 - E1_global` macro `seg_AUROC_0.05` `>= +0.005`. This is the protocol's
central claim — that the previous null result was caused by the wrong
augmentation model rather than by the approach. If it fails, the verdict is
`AUGMENTATION_MODEL_NOT_THE_CAUSE / NO_CONTINUE`, and the bank-expansion family
closes regardless of other gates.

## Real-Normal Gate

`F1 - S128_real` macro `seg_AUROC_0.05` `>= +0.005`, else
`BANK_SIZE_EFFECT_ONLY / NO_CONTINUE`.

## Anti-Absorption Gate

1. no object loses more than `0.03` `seg_F1` versus `F0`;
2. no object loses more than `0.03` `seg_AUPRO_0.05` versus `F0`;
3. macro `pooled_pixel_ap_float16` `>= F0 - 0.005`.

Violation yields `ANOMALY_ABSORPTION / NO_CONTINUE`.

## Strict Promotion Gate

All must pass:

1. `F1 - F0` macro `seg_AUROC_0.05` `>= +0.010`;
2. `F1 - F0` macro `seg_F1` `>= -0.005` and macro `seg_AUPRO_0.05` `>= -0.005`;
3. mechanism, hard-null, model-correction, real-normal, and anti-absorption
   gates all pass.

Gate order: validity → mechanism → hard-null → model-correction → real-normal →
anti-absorption → strict → continuation. Stop at the first failure and do not
compute later gates.

## Continuation Gate

If strict fails but mechanism, hard-null, model-correction, real-normal, and
anti-absorption all pass and `F1 - F0` macro `seg_AUROC_0.05` `>= +0.005`, the
verdict is `CONTINUE_DIAGNOSTIC`, licensing exactly one preregistered follow-up.
Otherwise `KILL_FOR_CLAIM / NO_CONTINUE`.

## Anti-Overfitting and Stop Conditions

- No sweep of the CV target, mode set, mode count, frequency multiplier, bank
  size, `k`, DVT parameters, thresholds, or per-object policy.
- Refitting the Flow on the expanded bank remains out of scope.
- A positive result confined to the three high-normal-FPR objects must be
  reported as a subgroup observation and must not become a per-object policy.
- A promoted result licenses only a separately preregistered paired-evaluator
  run under the Main Table contract.

## Execution and Cleanup

- Terra implements the parametric illumination field, the CV calibration, the
  high-frequency null, and the aggregator with all gates, reusing the
  `synthetic_shift_bank` machinery and its frozen-Flow seam.
- `F1` runs on dsba3 container `hun_fsad_tta_012`, host GPU 2.
- `F2` hard null runs on dsba3 container `hun_fsad_tta_012`, host GPU 3.
- `F0`, `E1_global`, `S128_real` are reused after custody verification.
- One arm is duplicated on the local RTX 4090 for parity.
- Preserve compact JSON/log/summary/parity artifacts only; verify absence of
  `anomaly_maps/` after the conclusion.

## Reporting

The report must state the verdict, every gate block with per-object numbers, the
Flow-checksum evidence, the realized CV per mode, and the `shift_*`-group FPR
change. It must state plainly whether the spatial model outperformed the global
exposure model, since that is this protocol's central question.
