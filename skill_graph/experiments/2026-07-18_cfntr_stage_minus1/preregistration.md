# CF-NTR Stage -1 Support-Only Identifiability Preflight

Date: 2026-07-18
Maximum verdict: `PROMISING_DIAGNOSTIC`

## Negative Evidence Intake

The exact NODyR parameterization is closed: it was absorbed by its state-angle
hard null and remained acquisition-sensitive. A second object-selected
TESTpublic diagnostic would also reuse public evidence because the proposed
sentinel objects and risks were selected from prior TESTpublic results.

This preflight therefore reads no TESTpublic image or label. It tests one new,
bounded question on the exact 16 normal references of all eight objects:
whether a physical nuisance basis learned from controlled support interventions
is identifiable, stable, more specific than matched generic low-rank nulls, and
able to remove held-out synthetic acquisition drift without absorbing sealed
synthetic local defects.

Generic residual projection is the nearest hard-null precedent. Physical
intervention provenance and image-level validation are mandatory. A positive
result is synthetic mechanism evidence only, never an AD2 performance claim.

## Claim, Evidence, Boundary, Positioning

- Claim under test: a stable rank-4 photometric tangent can be identified from
  normal support images and can separate held-out global acquisition changes
  from localized feature violations.
- Required evidence: physical-vs-PCA/random shift-energy capture, image-level
  bootstrap basis stability, held-out normal drift reduction, sealed synthetic
  defect retention, and complete parity/coverage.
- Boundary: exact SuperAD-16 references only; no TESTpublic/private access and
  no same-run SuperAD/RN-FMLK claim comparison.
- Positioning: normal-only plus synthetic-stress identifiability preflight.

## Frozen Scope

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: all eight public object names, but support `train/good` references only
- references: exact recorded SuperAD-selected 16 paths per object
- folds: SHA256-sorted paths, four folds, 12 fit / 4 held out; every reference is
  evaluated once
- backbone: DINOv2 ViT-L/14, layers `[5,11,17,23]`, canonical per-object
  resolution (`672`, except `sheet_metal=448`)
- features: per-layer row L2 normalization; all layer diagnostics are aggregated
  with equal object/layer weight
- query/test access: none
- rank: exactly `4`; no rank, strength, transform, or threshold sweep

## Physical Intervention Basis

All transforms operate on RGB float values in `[0,1]`, clip once, and return
uint8 RGB. Central-difference axes are frozen before execution:

- log exposure: `±0.25 EV`
- gamma: log exponent step `±0.10`
- contrast about image RGB mean: log multiplier step `±0.10`
- reciprocal red/blue white balance: log gain step `±0.08`

For each fit image, layer, and axis, subtract negative-step features from
positive-step features, spatially average the normalized feature delta, and
retain one image-level observation. The physical basis is the rank-4 eigenspace
of those observations. Transformed features never enter a normal memory bank.

Held-out validation uses the same four axes at exactly twice the training
amplitude plus two fixed pairwise compounds: exposure+white-balance and
gamma+contrast, with signs derived by SHA256 of object/path/transform.

## Capacity-Matched Nulls

- `clean_pca_r4`: rank-4 eigenspace of clean fit-image global feature variation.
- `random_r4_q95`: 16 deterministic SHA256-seeded Haar rank-4 bases; report the
  95th-percentile hard-null envelope, never a selected random seed.
- `raw`: no nuisance removal.

All bases use the identical projection/scoring equations. Patch rows are never
treated as independent statistical samples.

## Sealed Synthetic Defect Stress

Each held-out support image receives four deterministic local defects whose
location/shape/source are derived only from SHA256(object,path,kind):

1. CutPaste from a non-overlapping normal region.
2. Elliptical chromatic blotch.
3. Local Gaussian blur.
4. Thin high-frequency scratch.

Known masks are retained for evaluation. Each defect is evaluated on the clean
image and on one deterministically assigned held-out synthetic acquisition
shift. These masks are synthetic stress-test labels, not real AD2 evidence.

## Diagnostics

1. Shift-energy capture:
   `E = 1 - ||(I-UU^T) delta||^2 / ||delta||^2` on held-out validation shifts.
2. Basis stability:
   `S = ||U_full^T U_boot||_F^2 / 4`, using 200 image-level stratified support
   bootstraps per object; patch bootstrap is forbidden.
3. Normal drift reduction:
   held-out normal feature-residual p95 change, comparing projected and raw
   residuals under the sealed validation shifts.
4. Defect retention:
   feature-grid pixel `AUROC@0.05`, best F1, and anomaly-mask residual-energy
   retention for clean and shifted synthetic defects. Native synthetic masks
   are rasterized to the DINO feature grid with area interpolation and a
   frozen any-overlap (`>0`) positive rule; no native-resolution score-map
   interpolation is used in this representation-level diagnostic. Energy
   retention is the object-level ratio of summed physical-arm energy to
   summed raw-arm energy inside these masks, not a mean of per-map ratios.
5. Uncertainty:
   2,000 object-stratified image-level bootstrap replicates for macro confidence
   bounds. All random streams are SHA256-derived and recorded.

## Pre-registered Gates

All values use unrounded float64 summaries.

### G1 — physical specificity

- macro `E_physical >= 0.60`
- 95% lower confidence bound of
  `E_physical - max(E_clean_pca, q95(E_random)) > 0.10`

### G2 — basis stability

- median `S >= 0.90`
- every object's 5th-percentile `S >= 0.75`

### G3 — held-out normal drift

- macro drift reduction `>= 0.65`
- 95% lower confidence bound `>= 0.50`
- every object drift reduction `>= 0.40`
- physical exceeds the best PCA/random drift reduction by `>= 0.10` macro

### G4 — synthetic defect retention

- physical-minus-raw macro pixel pAUROC delta `>= -0.005`
- physical-minus-raw macro F1 delta `>= -0.010`
- no object below `-0.020/-0.030` pAUROC/F1
- anomaly-mask residual-energy retention `>= 0.90` macro
- clean and shifted synthetic-defect tables must both satisfy coverage

### G5 — audit

- 100% finite, nonconstant map/score coverage
- exact support/fold/transform/defect parity across physical and null arms
- runner, core, config, support manifest, and result SHA256 values retained
- no public image, filename condition, public label, or ground truth read

## Verdict

- All G1--G5 pass: `PROMISING_DIAGNOSTIC`. This permits exactly one frozen,
  all-eight, full-population TESTpublic shadow with the same rank/transforms and
  same-run raw/SuperAD/RN-FMLK/hard-null controls.
- G1 or G2 fails: `KILL_FOR_CLAIM / NO_CONTINUE`; the physical tangent is not
  identifiable beyond generic low rank.
- G1/G2 pass but G3 or G4 fails: `KILL_FOR_CLAIM / NO_CONTINUE`; the identified
  nuisance is unusable because drift remains or defect evidence is absorbed.
- G5 fails: invalid run; only an exact-config rerun is allowed.

Even a complete pass does not establish actual AD2 acquisition invariance or
real-defect retention. Independent real confirmation requires external labeled
development data or private benchmark feedback, neither of which is locally
available.

## Remote and Storage Contract

- dsba3 fixed project container, host GPUs `0,1,2,3` only after an idle/process
  check; two objects per GPU
- isolated source snapshot and unique run root; never overwrite shared source
- retain only compact feature-level statistics, synthetic masks/metrics, hashes,
  manifests, and logs
- delete regenerable transformed images, dense feature caches, and score maps
  after the verdict is recorded and verified locally
