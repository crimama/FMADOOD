# TTE Collapse Decomposition (`flow_tte_budget125`)

Date: 2026-08-05
Verdict: `HISTORICAL_COLLAPSE_REPRODUCED / MECHANISM_ATTRIBUTED_TO_DEFECT_CONTAMINATION / NO_METHOD_CLAIM` (see Addendum 2; the original gate-failure verdict below was superseded once the DVT confound was found)

Run ID: `tte_collapse_decomposition_20260805_v1` (dsba5, host GPU 3, container
`hun_fsad_tta_012`). Local pull:
`results/remote_runs/dsba5/tte_collapse_decomposition_20260805_v1`.

## Summary

The preregistered replication gate failed. `A1_replicate` did not reproduce the
historical `fabric` collapse under Baseline1-core, so **no mechanism
decomposition of the recorded `-0.027955` may be read from this run**. Gates 2
through 6 are therefore not evaluated.

Two results survive the gate failure and are reported as descriptive only:
the historical macro number is a one-object artifact, and naive reservoir
expansion is not harmful under the current Baseline1-core representation.

## Pre-run reanalysis of the historical artifact

Recomputed from
`skill_graph/experiments/2026-07-07_flowtte_component_ablation/object_deltas.tsv`
with no new compute:

| Object | ΔpAUROC | ΔF1 | share of macro ΔpAUROC |
|---|---:|---:|---:|
| fabric | -0.177735 | -0.197444 | **79.5%** |
| sheet_metal | -0.022984 | -0.005916 | 10.3% |
| fruit_jelly | -0.018670 | +0.000920 | 8.3% |
| vial | -0.013731 | -0.032954 | 6.1% |
| rice | -0.010187 | -0.000980 | 4.6% |
| walnuts | -0.002216 | +0.004509 | 1.0% |
| wallplugs | +0.001431 | +0.087068 | -0.6% |
| can | +0.020448 | +0.004965 | -9.1% |

Macro mean reproduces the recorded `-0.027956 / -0.017479`. Macro median is
`-0.011959 / -0.000030`; `4/8` objects have non-negative ΔF1; excluding
`fabric` the arm reads `-0.006558` pAUROC and `+0.008230` F1.

"Static no-TTE memory is essential" was generalized from a single object.

## Implementation reading

`ReservoirMemory.absorb` (`src/flow_tte/memory.py:170`) fills open slots from
the arrival-ordered prefix (`candidates[:fill]`), then switches to reservoir
replacement. The only selection gate is `nll <= density_threshold` at the
support 0.90 quantile. `stream_test_images`
(`scripts/flow_tte_mvtec_ad2_core.py:2513`) sorts by
`(path.name, anomaly_type)`, and `"bad" < "good"`, so the stream opens
`000_overexposed(bad), 000_overexposed(good), 000_regular(bad), ...`.
`ScoreCalibration` stays frozen at the clean support-LOO fit while the bank
grows. The historical arm therefore tested one uncontrolled absorption rule,
not test-time adaptation as a class.

## Execution

Baseline1-core frozen: `dinov3_vith16plus`, layers `[7,15,23,31]`, native 672,
`layer_norm_mean`, DVT `position_mean` alpha 1.0, 2-layer patch-local MLP
coupling flow, raw Euclidean latent 1-NN, support-LOO standardization,
`density_weight=0.0`. Support manifest SHA-256
`9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
Objects `fabric` (156 images), `wallplugs` (150), `vial` (140), `can` (162),
full `test_public`. Thirteen arms, patch-grid metrics.

## Gate 1 — replication: FAILED

| Arm | fabric | wallplugs | vial | can |
|---|---:|---:|---:|---:|
| `A1_replicate` ΔpAUROC | **+0.0086** | +0.0087 | -0.0074 | +0.0100 |
| `A1_replicate` ΔF1 | **+0.0225** | +0.0319 | -0.0091 | +0.0007 |

The gate required a negative `fabric` ΔpAUROC. The observed value is positive,
and the historical `fabric` effect was `-0.177735`. The sign is opposite and
the magnitude differs by more than an order of magnitude. The run is valid as
an execution but cannot be used to attribute the historical number.

Configuration differences between this run and the 2026-07-07 arm, any of
which could carry the discrepancy: backbone `dinov3_vitl16` -> `dinov3_vith16plus`,
layers `[5,11,17,23]` -> `[7,15,23,31]`, `lambda_logdet` `1e-3` -> `2e-2`,
`density_weight` `0.25` -> `0.0`, original-resolution evaluator -> patch grid.

The leading hypothesis is the backbone. `fabric` is the object where the
H+ upgrade gained most (`+0.069797` AUROC, `+0.176069` F1 in
`2026-07-08_flowtte_hplus_backbone_only`). A representation that separates
normal texture poorly makes absorbing test patches far more damaging. If that
is correct, the collapse was a property of DINOv3-L on `fabric` and did not
survive the backbone upgrade Baseline1 already adopted. This is a hypothesis,
not a measurement; the run that would test it is a single `A1_replicate` arm at
`dinov3_vitl16` / `[5,11,17,23]` / `lambda_logdet=1e-3` / `density_weight=0.25`.

## Design defect found: the `M2` arms are inert

`A5_two_pass_recal` and `A7_ceiling` returned metrics bit-identical to
`A4_two_pass` and `A6_two_pass_clean` on every object, despite a genuinely
different calibration (`fabric` σ `4.536837` vs `4.306583`).

The cause is structural, not a bug. `ScoreCalibration.normalize_distance` is an
affine map with positive scale, and both pAUROC and best-F1 are invariant to
monotone affine rescaling. Calibration can only be measured against a **fixed**
threshold, never an oracle one. Any future test of `M2` must carry a
support-derived fixed-threshold F1; the preregistration was wrong to gate `M2`
on oracle metrics.

## Descriptive observations (not gated, no claim)

Macro over the four objects, versus `A0_static`:

| Arm | macro ΔpAUROC | macro ΔF1 | obj ≥0 (pA / F1) |
|---|---:|---:|---|
| `A1_replicate` | +0.00497 | +0.01149 | 3/4, 3/4 |
| `A2_oracle_clean` | +0.00603 | +0.01158 | 3/4, 3/4 |
| `A4_two_pass` | +0.00262 | +0.01066 | 3/4, 3/4 |
| `A6_two_pass_clean` | +0.00715 | +0.00984 | 3/4, 3/4 |
| `A8_cap105` | +0.00312 | +0.00669 | 3/4, 3/4 |
| `A9_cap200` | +0.00416 | +0.01446 | 2/4, 3/4 |
| `A10-A12` order seeds | +0.00194..+0.00431 | +0.00833..+0.01088 | 3/4 |

Memory provenance:

| Object | absorbed rows | contaminated | contamination | defect self-hit |
|---|---:|---:|---:|---:|
| fabric | 8400 | 3 | 0.04% | 20.1% |
| wallplugs | 8400 | 10 | 0.12% | 26.1% |
| vial | 9576 | 58 | 0.61% | 31.7% |
| can | 15288 | 1 | 0.01% | 26.7% |

- **`M1` is near-absent under this configuration.** Ground-truth defect patches
  reach the bank at `0.01%`-`0.61%`, and `A2_oracle_clean` (contamination
  removed by construction) differs from `A1_replicate` by only `+0.0011` macro
  pAUROC. The NLL gate at the support 0.90 quantile already rejects almost
  every defect patch when the representation is strong.
- **`M3` exists as a phenomenon but does not carry metric harm here.** Online
  stream drift is `0.19`-`0.62` and two-pass drops it to `0.01`-`0.13`, so the
  protocol change removes non-stationarity as intended, yet both protocols land
  within `0.003` macro pAUROC of each other.
- **`vial` is the one object that degrades**, monotonically with capacity
  (`A8_cap105` `-0.0019`, `A1` `-0.0074`, `A9_cap200` `-0.0123` pAUROC). It also
  carries the highest contamination (`0.61%`) and the only negative static
  shifted-good inflation. It is the natural single-object target if this line
  is continued.
- The `20%`-`53%` defect self-hit rates are **not** contamination. The absorbed
  rows those defect patches match are almost entirely normal test patches, so
  the nearest neighbour is migrating toward test-side normal appearance, which
  is the `M4` behaviour the branch was supposed to test.

None of this is a method claim. The arms that look best use ground truth, the
metric grid is not the frozen contract, four objects is not all eight, and the
gate that would have licensed reading these numbers failed.

## Access ledger

TESTpublic images read; TESTpublic ground truth read (oracle arms, flagged
`uses_labels`); TESTprivate and TESTprivate_mixed untouched; no threshold
export; no method promotion; Baseline1 unchanged.

## Decision

`REPLICATION_GATE_FAILED`. The historical `-0.027955` is not attributable from
this run. What the run does establish is narrower and still useful: the recorded
verdict rests `79.5%` on one object, and under the Baseline1-core
representation that the project actually uses, naive reservoir expansion is not
the destructive operation the 2026-07-07 arm implied.

The test-time branch is therefore neither reopened nor closed. Reopening
requires the `dinov3_vitl16` replication arm to establish whether the collapse
is representation-specific, plus a fixed-threshold metric so `M2` is
measurable at all.

---

## Addendum — DINOv3-L/16 replication arm (2026-08-05)

Run ID: `tte_collapse_l16_replication_20260805_v1`. Local pull:
`results/remote_runs/dsba5/tte_collapse_l16_replication_20260805_v1`.

The backbone hypothesis stated above was tested directly. Configuration matched
the 2026-07-07 arm on every axis this runner controls: `dinov3_vitl16`, layers
`[5,11,17,23]`, `lambda_logdet=1e-3`, `density_weight=0.25`, identical support
manifest, `fabric`, full `test_public`. Runner gained CLI overrides and a
density-penalty scoring path for this purpose.

| Arm | L/16 ΔpAUROC | L/16 ΔF1 | H+ ΔpAUROC | H+ ΔF1 |
|---|---:|---:|---:|---:|
| `A1_replicate` | **+0.0053** | +0.0158 | +0.0086 | +0.0225 |
| `A2_oracle_clean` | +0.0059 | +0.0163 | +0.0085 | +0.0215 |
| `A4_two_pass` | +0.0014 | +0.0084 | +0.0059 | +0.0170 |
| `A9_cap200` | +0.0104 | +0.0254 | +0.0124 | +0.0366 |

**The backbone hypothesis is rejected.** `fabric` ΔpAUROC is `+0.0053` under the
historical backbone, still positive and still nowhere near `-0.177735`. The
weaker representation reproduces the same qualitative behaviour as H+: expansion
is mildly helpful, contamination stays at `0.10%`, and capacity dose is
monotonically positive rather than destructive. Absolute static performance does
drop as expected (`A0_static` pAUROC `0.8544` vs H+ `0.8981`), confirming the
backbone swap took effect.

**Secondary correction to the `M2` defect note.** With `density_weight=0.25`,
`A5_two_pass_recal` no longer equals `A4_two_pass` (`+0.0009` vs `+0.0014`
pAUROC). Recalibration is inert only when the score is a single affine distance
term; adding a density component makes the distance/density ratio observable.
The effect is nonetheless sub-material (`-0.0005` pAUROC, `-0.0014` F1), and a
fixed-threshold metric is still required to measure `M2` properly.

**Standing limitation, now the leading explanation.** This runner reimplements
the scoring path over cached latents. It does not execute the production
pipeline that the 2026-07-07 arm used (`run_object_fused` with RGB guided-r8
refinement, score-field calibration, and close/fill/erode morphology). `fabric`
is the most postprocessing-sensitive object on record — morphology alone moved
its F1 by `+0.067922` in `2026-07-18_default_shift_sequence`. A small raw-score
change amplified by that chain is now the most plausible route to `-0.177735`,
and it is untested.

**Revised next step.** The true replication is the production runner
`scripts/run_flow_tte_mvtec_ad2.py` on `fabric` at `--expansion-budget 1.0`
versus `1.25` under the historical configuration, evaluated at original
resolution with the guided/morphology chain intact. Until that runs, the
historical `-0.027955` remains unattributed and the test-time branch stays
suspended rather than reopened or closed.

---

## Addendum 2 — production replication and mechanism attribution (2026-08-05)

Runs: `tte_collapse_l16_nodvt_20260805_v1` (diagnostic, DVT off) and
`tte_collapse_production_replication_20260805_v1` (production pipeline,
`out_b125` / `out_b100`). Both on dsba5 host GPU 3.

### The confound was DVT, not the backbone

The 2026-07-07 `run_manifest.json` was recovered intact, including per-object
absolute metrics (`fabric seg_AUROC=0.577383`, `seg_F1=0.127903`). Key
diagnosis: that manifest has ~40 keys and contains **no** `dvt_denoise_mode`,
`rgb_guide`, or `binary_postprocess` entry, whereas the 2026-07-13 manifest has
87 keys including all three. Those features did not exist on 2026-07-07, so the
historical arm ran **without DVT and without any postprocessing**. Every
diagnostic arm reported above had DVT on, inherited from Baseline1.

Rerunning the diagnostic with `--dvt-mode none` flips the sign on `fabric`:

| Arm | DVT on | DVT off |
|---|---:|---:|
| `A1_replicate` ΔpAUROC | +0.0053 | **-0.0625** |
| `A2_oracle_clean` ΔpAUROC | +0.0059 | **+0.0234** |
| `A9_cap200` ΔpAUROC | +0.0104 | -0.0259 |
| contamination | 0.04% | 0.27% |
| defect self-hit | 20.1% | 52.4% |

This also retracts the earlier "postprocessing amplification" hypothesis, which
was rejected by manifest comparison before any run, and corrects the earlier
"`M1` is near-absent" statement, which held only with DVT enabled.

### Production replication succeeds

`scripts/run_flow_tte_mvtec_ad2.py` on `fabric` at the manifest configuration
(`dinov3_vitl16`, `[5,11,17,23]`, `lambda_logdet=1e-3`, `density_weight=0.25`,
`--dvt-denoise-mode none --rgb-guide none --binary-postprocess none`,
original-resolution evaluator):

| Quantity | Historical | This run | Gap |
|---|---:|---:|---:|
| `fabric` seg_AUROC @1.25 | 0.577383 | 0.567874 | 0.009509 |
| `fabric` seg_AUROC @1.00 | 0.755118 (implied) | 0.749907 | 0.005211 |
| **Δseg_AUROC** | **-0.177735** | **-0.182033** | 0.004298 |
| **Δseg_F1** | **-0.197444** | **-0.215377** | 0.017933 |
| ΔAU-PRO@0.05 | not recorded | -0.067644 | - |

The collapse reproduces. The residual gap is consistent with evaluator and
library drift between the 2026-07-07 container and the current one.

### Mechanism attribution

- **`M1` defect self-confirmation is the dominant mechanism.** Removing only
  ground-truth defect patches from absorption candidates moves `fabric` from
  `-0.0625` to `+0.0234`, a swing of `+0.0859` that more than reverses the
  loss. Preregistered gate 2 required recovering half the loss; the observed
  recovery exceeds it. Caveat: gate 2 was preregistered against the
  Baseline1-core configuration, and this is the corrected historical
  configuration, so this is a post-hoc application of the gate, not a clean
  preregistered pass.
- **`M3` non-stationarity is secondary.** `A4_two_pass` recovers only to
  `-0.0076`.
- **The patch-grid metric under-reads the effect roughly threefold**
  (`-0.0625` vs production `-0.1820`) while preserving sign and ordering under
  matched configuration. Diagnostic-grid deltas are directionally usable and
  quantitatively not.
- **Causal chain:** no DVT leaves position artifacts in the features, which
  lowers defect-patch NLL, which lets defect patches pass the support 0.90
  quantile gate, which puts them in the bank permanently, which suppresses every
  later occurrence of the same defect. DVT closes the chain at the first link.

### Operational note

The frozen support manifest stores absolute paths under
`/home/hunim/Volume/DATA/mvtec_ad_2`. `load_support_manifest` in the diagnostic
runners rebases by path tail; `select_support_paths_from_json` in the production
runner requires exact equality and fails closed. The production run was launched
against a rebased copy (`support_rebased.json`, all 128 paths verified to exist)
while the original manifest and its SHA-256 were left untouched.

### Revised decision

The historical `-0.027955` is now attributed. It is not evidence that test-time
memory expansion is harmful as a class. It is evidence that **unfiltered
absorption is harmful when the representation admits defect patches through the
NLL gate**, a condition that held on 2026-07-07 (no DVT) and does not hold under
Baseline1 (DVT on, contamination `0.04%`, expansion mildly positive).

This still authorizes no method claim: the production replication covers one
object, the mechanism arms use ground truth, and `A7_ceiling` was evaluated on
`fabric` alone rather than the preregistered `3/4` objects. What it does license
is reopening the test-time branch for design work, with an explicit requirement
that any absorption policy carry a defect-rejection mechanism rather than the
bare NLL gate.

### Storage

Both `anomaly_maps` directories (3.0 GB each) were removed after metrics were
pulled and verified. Compact metrics, manifests, and logs are retained under
`results/remote_runs/dsba5/tte_collapse_production_replication_20260805_v1`.
