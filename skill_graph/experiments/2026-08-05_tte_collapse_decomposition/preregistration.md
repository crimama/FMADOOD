# TTE Collapse Decomposition (`flow_tte_budget125`)

Date: 2026-08-05
Status: `DIAGNOSTIC_ONLY / NO_METHOD_PROMOTION / NO_PRIVATE_ACCESS`

## Negative Evidence Intake

The 2026-07-07 component ablation recorded `flow_tte_budget125` at
`-0.027955` pAUROC and `-0.017479` F1 versus `flow_noctx_base` and concluded
"static no-TTE memory is essential". Every experiment since has frozen
`expansion_budget=1.0`, so that single arm is the sole empirical basis for
closing the test-time branch of an project whose stated target is FSAD-**TTA**.

Two pre-run observations make the verdict unsafe to inherit.

**Object concentration.** Re-reading `object_deltas.tsv` from the completed
run, the macro loss is not broad:

| Object | ΔpAUROC | ΔF1 | share of macro ΔpAUROC |
|---|---:|---:|---:|
| fabric | -0.177735 | -0.197444 | **79.5%** |
| sheet_metal | -0.022984 | -0.005916 | 10.3% |
| fruit_jelly | -0.018670 | +0.000920 | 8.3% |
| vial | -0.013731 | -0.032954 | 6.1% |
| rice | -0.010187 | -0.000980 | 4.6% |
| walnuts | -0.002216 | +0.004509 | 1.0% |
| wallplugs | +0.001431 | **+0.087068** | -0.6% |
| can | +0.020448 | +0.004965 | -9.1% |

Macro mean reproduces the recorded `-0.027956 / -0.017479`. Macro **median**
is `-0.011959 / -0.000030`, `4/8` objects have non-negative ΔF1, and excluding
`fabric` alone the arm reads `-0.006558` pAUROC and **`+0.008230`** F1. The
recorded verdict is a one-object result presented as a component law.

**Implementation confound.** `ReservoirMemory.absorb` fills every open slot
from the arrival-ordered prefix (`candidates[:fill]`), then switches to
reservoir replacement, and the selection gate is only `nll <= density_threshold`
at the support 0.90 quantile. `stream_test_images` sorts by
`(path.name, anomaly_type)`, and `"bad" < "good"`, so the stream begins
`000_overexposed(bad), 000_overexposed(good), 000_regular(bad), ...`. Roughly
half of the images that fill the adaptation budget are defective, and nothing
prevents a defect patch that clears the NLL gate from becoming a permanent bank
row. `ScoreCalibration` meanwhile stays frozen at the clean support LOO fit
while the bank grows.

The arm therefore never tested "test-time adaptation". It tested one specific
uncontrolled absorption rule.

## Motivation

Split the single pooled `-0.027955` into separable mechanisms so the test-time
branch is either closed for a stated reason or reopened with a specification.

- `M1` defect self-confirmation: absorbed defect patches suppress later
  occurrences of the same defect.
- `M2` calibration invalidation: bank grows, support-LOO mean/std stays frozen.
- `M3` stream non-stationarity: early images are scored against a small bank
  and late images against a contaminated one, which breaks the single global
  threshold that pooled oracle F1 depends on.
- `M4` shift adaptation: the intended benefit of absorbing genuine normal
  patches from acquisition conditions the support never saw.

## Implementable Design

Baseline1-core is frozen throughout: `dinov3_vith16plus`, layers
`[7,15,23,31]`, native 672, `layer_norm_mean`, DVT `position_mean` alpha 1.0,
2-layer patch-local MLP coupling flow (3 epochs, lr 2e-4, clamp 1.9,
`lambda_logdet=2e-2`), raw Euclidean latent 1-NN, support-LOO standardization,
`density_weight=0.0`. Support is the frozen SuperAD-16 manifest,
SHA-256 `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`,
with the deterministic per-index brightness factor in `[0.8, 1.2]`.

Only the memory-expansion protocol varies.

| Arm | Budget | Candidates | Protocol | Recalibrate | Isolates |
|---|---:|---|---|---|---|
| `A0_static` | 1.00 | - | static | no | reference |
| `A1_replicate` | 1.25 | all | online | no | historical arm |
| `A2_oracle_clean` | 1.25 | GT-normal | online | no | `M1` |
| `A3_good_images` | 1.25 | good images | online | no | `M1` (image level) |
| `A4_two_pass` | 1.25 | all | two-pass | no | `M3` |
| `A5_two_pass_recal` | 1.25 | all | two-pass | yes | `M2`+`M3` |
| `A6_two_pass_clean` | 1.25 | GT-normal | two-pass | no | `M1`+`M3` |
| `A7_ceiling` | 1.25 | GT-normal | two-pass | yes | `M4` ceiling |
| `A8_cap105` | 1.05 | all | two-pass | no | capacity dose |
| `A9_cap200` | 2.00 | all | two-pass | no | capacity dose |
| `A10-A12_order_s{101,202,303}` | 1.25 | all | online | no | `M3` order variance |

Online arms score an image against the bank state preceding it, exactly as
`FlowTTE.score_then_expand` does. Two-pass arms absorb the whole stream first
and rescore everything against the final bank, which removes `M3` while leaving
`M1` intact. Arms `A2`, `A3`, `A6`, `A7` read ground truth and are explicitly
diagnostic upper bounds, never deployable policies.

Objects are `fabric` (79.5% of the macro loss), `wallplugs` (the largest F1
gain), `vial` (F1 loss without AUROC collapse), and `can` (AUROC gain). Full
`test_public` `good`+`bad` per object.

## Evaluation Alignment

Metrics are computed on the DINOv3 patch grid, not at original resolution, so
**absolute values are not comparable with any recorded anchor**. Only
within-run arm deltas against `A0_static` are interpretable, and every consumer
must state that. `partial_auroc` reproduces
`sklearn.metrics.roc_auc_score(max_fpr=0.05)` and `best_f1` reproduces the
`precision_recall_curve` maximum; both are unit-tested against scikit-learn.

Mechanism observables per arm: absorbed rows, contaminated rows
(GT-defect provenance tracked through an identically seeded parallel
reservoir), `self_hit_rate` (share of defect patches whose final 1-NN is an
absorbed test row), calibration mean/std, shifted-good inflation,
bad-good separation, and stream drift (correlation of per-image normal-score
mean with stream position).

## Access Ledger

- TESTpublic images: **read** (the historical arm was measured there).
- TESTpublic ground truth: **read**, oracle arms only, flagged `uses_labels`.
- TESTprivate / TESTprivate_mixed: **no access**.
- Threshold export, method promotion, shot scaling: **not authorized**.

## Decision Gates

Frozen before reading results.

1. **Replication.** `A1_replicate` must reproduce a negative ΔpAUROC on
   `fabric` and a non-negative ΔF1 on `wallplugs`. Otherwise the run is invalid
   and no mechanism claim may be read.
2. **Contamination credit** (`M1`). Credited if `A2 - A1` recovers at least
   half the `A1` ΔpAUROC loss on `fabric`.
3. **Non-stationarity credit** (`M3`). Credited if `A4 - A1` recovers at least
   half the `A1` ΔF1 loss, or if the order-seed spread `A10..A12` exceeds the
   `A1` effect magnitude.
4. **Calibration credit** (`M2`). Credited if `A5 - A4` is material
   (`>= 0.005` pAUROC or `>= 0.010` F1).
5. **Reopen gate** (`M4`). The test-time branch may be reopened for design work
   only if `A7_ceiling` beats `A0_static` on both pAUROC and F1 for at least
   `3/4` objects. A positive ceiling is necessary, not sufficient: it uses
   ground truth, so it bounds what a label-free policy could reach and proves
   nothing about achievability.
6. **Close gate.** If `A7_ceiling` fails to beat `A0_static`, the branch is
   closed permanently with a mechanism reason and no further TTE work is
   authorized.

Under every outcome this run produces no method claim, no promotion, and no
change to Baseline1.
