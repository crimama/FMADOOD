# FlowTTE Paired-Loss Diagnostic Report

## Status

`COMPLETE / KILL_FOR_CLAIM / RETAIN_BASELINE1 / NO_PUBLIC_ACCESS`

## Decision

The strongest retained patch-local DINOv3-H+/16 + DVT MLP Flow is now named
`Baseline1` and remains the reference. Direct paired canonicalization is not
promoted: its best arm reduced
held shifted-normal inflation by `9.77%`, but retained only `0.8707` of the
synthetic defect contrast.  The preregistered requirements were at least
`10%` shift reduction, at least `0.97` macro defect retention, and at least
`0.90` retention for every object.

The all-eight continuation was therefore not authorized.  No public, private,
test, or ground-truth path was enumerated or opened.

The canonical full configuration is recorded in
`skill_graph/methods/baseline1.md`. The paired-loss v4 row is specifically the
`Baseline1-core` scoring view with `density_weight=0.0`.

## Frozen experiment

- Baseline: patch-local DINOv3-H+/16 + DVT MLP Flow with raw latent 1-NN
  scoring.
- Support: the fixed 16 normal images per object, manifest SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- Objects: `fruit_jelly`, `walnuts`, `vial`, and `fabric`.
- Compute: fixed container `hun_fsad_tta_012` on dsba5 host GPU 3
  (`cuda:0` inside the container).
- Scope: support-only photometric pairs; no query adaptation, Transformer,
  register, or FiLM branch.
- Primary candidates: raw-pair loss with NLL weights `1.0` (V2) and `0.1`
  (V3).
- Controls: baseline V0, augmentation-NLL-only V1, normalized-pair C1, and
  random-pair C2.

## Aggregate results

| Arm | Shift reduction | Improved objects | Defect retention | Full gate |
|---|---:|---:|---:|---:|
| Baseline1 (`V0_default` artifact ID) | `0.00%` | `0/4` | `1.0000` | reference |
| V1 augmentation NLL | `-108.06%` | `0/4` | `2.1315` | fail |
| V2 raw pair + NLL 1.0 | `2.84%` | `2/4` | `0.8401` | fail |
| V3 raw pair + NLL 0.1 | `9.77%` | `4/4` | `0.8707` | fail |
| C1 normalized pair | `2.19%` | `2/4` | `0.9654` | fail |
| C2 random pair | `4.20%` | `2/4` | `0.4424` | fail |

V1 is not an improvement despite its defect-retention ratio exceeding one.  It
expanded the full latent geometry: shifted-normal inflation more than doubled,
leave-one-out drift became large, and neighborhood retention fell to roughly
`0.36--0.47`.

## Best candidate: V3

| Object | Shift reduction | Defect retention | Abs. LOO drift | Neighbor retention |
|---|---:|---:|---:|---:|
| fruit_jelly | `17.21%` | `0.7372` | `0.00482` | `0.9787` |
| walnuts | `0.06%` | `1.1280` | `0.00459` | `0.9858` |
| vial | `21.22%` | `0.6683` | `0.00629` | `0.9743` |
| fabric | `0.60%` | `0.9491` | `0.00030` | `0.9822` |

V3 preserved global rank, scale, and neighbors, so its failure is not ordinary
collapse.  Instead, the dimensions that make photometrically shifted patches
agree overlap with dimensions carrying defect evidence.  The two objects with
meaningful shift correction, `fruit_jelly` and `vial`, are exactly the objects
with the strongest anomaly attenuation.

## Control interpretation

- Lowering the NLL weight from V2 to V3 exposed useful shift-correction
  capacity, but a single global scalar trade-off did not separate nuisance and
  defect directions.
- C1 preserved geometry better than raw-pair training but delivered only
  `2.19%` macro shift reduction and still reduced `vial` defect contrast to
  `0.8241`.
- C2 superficially reduced shift on some objects while destroying defect
  contrast (`0.4424` macro) and increasing leave-one-out drift.  This confirms
  that distance reduction alone can be produced by destructive alignment.
- V1 shows that optimizing augmentation NLL alone is mismatched to the raw
  latent 1-NN inference geometry.

## Scientific conclusion and next direction

The high-potential path is to retain the Baseline1 geometry and use
this experiment's evidence to design a selective loss, not to continue tuning
V3's scalar coefficient.  The next candidate should estimate a low-rank global
nuisance subspace from support augmentations and restrict correction to that
subspace, while freezing or margin-protecting the orthogonal local-defect
residual.  If that separation cannot be demonstrated support-only, the learned
shift signal should be used for calibration, support routing, or conditional
DVT strength rather than directly warping every patch latent dimension.

## Adjudication correction

The frozen remote aggregator initially emitted `EARLY_VALIDITY_KILL` because it
incorrectly made every deliberately destructive control's numerical validity
an experiment-wide sentinel.  Post-run review corrected the adjudicator so
only V0 baseline validity is global; each candidate/control validity remains
inside its own full gate.  No model, checkpoint, score, or metric was changed.
The preserved original verdict and the independently regenerated corrected
verdict make the correction auditable.  The corrected conclusion is
`KILL_FOR_CLAIM`, and both candidate gates remain false.

## Artifact custody

- Valid run: `results/remote_runs/dsba5/flowtte_paired_loss_20260804_v4`.
- Original remote verdict: `protocol_verdict.json`.
- Corrected independent verdict: `corrected_aggregate/verdict.json`.
- All 20 compact-artifact entries passed `sha256sum -c` after pullback.
- Four object summaries and four checkpoints are present.
- The access ledger records all public/private/test/ground-truth enumeration
  and open flags as `false` and query adaptation as `false`.
- The immutable remote source snapshot was removed after verified pullback.
- dsba5 GPU 3 returned to `1 MiB` use and `0%` utilization.

Earlier v1--v3 launch attempts were invalidated before summary production and
their isolated snapshots were removed.  V4 is the only valid evidentiary run.

## Verification

- Focused unit/integration tests cover the paired loss, runner, aggregator,
  corrected validity semantics, and remote controller.
- Ruff, Python compilation, shell syntax, compact checksums, access ledger,
  and corrected verdict were checked after the run.
- Broad repository tests were not used as a completion gate because the shared
  worktree contains unrelated pre-existing changes; the experiment-specific
  test surface is isolated and passing.
