# FLRS Stage -1 Support-Only Preregistration

Date: 2026-07-18

## Negative Evidence Intake

The frozen default fails because acquisition and defect evidence overlap in an
object/layer/position-dependent residual. Density, routing, hard locality,
projection, and cross-layer prediction are closed. FLRS is allowed only as a
standalone same-layer spatial-relation measurement; it may not be fused into or
used to switch the default score.

## Target and access contract

- Dataset root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Population: exactly 16 `train/good` paths for each of all eight objects from
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`.
- Expected manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- Forbidden: `test_public`, `test_private`, `ground_truth`, query update,
  query transformation, TTA, support routing, projection, density, fusion.
- Backbone: frozen DINOv2 ViT-L/14, layers `[5,11,17,23]`, short edge 672,
  except `sheet_metal` at 448.

## Candidate

At each layer, L2-normalize the patch grid and compute exactly:

1. radius-1 eight-neighbor cosine mean;
2. radius-1 cosine population standard deviation;
3. radius-2 eight-neighbor cosine mean;
4. absolute horizontal-versus-vertical radius-1 cosine anisotropy.

For each image-disjoint 12-train/4-held-out fold, fit global per-layer median
and scaled MAD over training-support patches. Candidate patch energy is mean
squared robust z over four statistics and four layers. The MAD floor is fixed
at `1e-3`; no clipping, covariance, position model, or selected layer exists.

## Stimuli

- clean held-out support;
- four signed single-axis validation transforms and two signed compound
  transforms at frozen 2x amplitude;
- four deterministic synthetic defect families: CutPaste, chromatic blotch,
  Gaussian blur, and scratch;
- each defect under one deterministic assigned validation shift.

All source folds, signs, defect seeds, and assigned shifts are SHA-derived from
source-relative paths and a fixed protocol ID. No metric can select a retry.
If a sealed synthetic acquisition is a pixel-level no-op, up to 32 SHA-derived
seeds are tried in fixed order; the first acquisition-visible stimulus is used,
without extracting features or consulting any score or label metric.

## Hard nulls and controls

- `raw_state`: identical median/MAD estimator on normalized feature channels;
- `random_neighbor`: identical relation statistics using fixed distant offsets;
- `spatial_permutation`: identical local relation statistics after a fixed
  deterministic permutation of grid positions, with scores inverted to source
  coordinates before mask evaluation;
- `rgb_gradient`: grayscale Sobel magnitude standardized from fold-train clean
  supports;
- `radius1_only`: candidate ablation without radius-2 information.

RN-FMLK/full-memory DINO kNN is deferred only because this stage cannot support
a public method claim. It becomes mandatory with same-run H+ and SuperAD-16 if
and only if Stage -1 promotes a public shadow.

## Metrics and gates

For each variant/object report finite/nonconstant coverage, clean and shifted
image p95, mean paired positive p95 inflation, clean-defect and shifted-defect
pixel pAUROC@0.05/F1, and per-defect-family values.

Strict promotion requires all:

1. FLRS macro positive shift inflation is at least 50% below `raw_state`, and
   every object is at least 25% below;
2. FLRS defect pAUROC/F1 are no worse than `raw_state` by `0.005/0.010` for
   both clean-defect and shifted-defect populations;
3. FLRS exceeds the per-object maximum of all non-ablation nulls by
   `0.005/0.005` on defect pAUROC/F1;
4. object-bootstrap lower bounds for macro shift reduction and both null
   margins are positive;
5. every expected map is finite, nonconstant, and aligned to its mask;
6. support path parity and forbidden-path audit pass.

Before execution, the remaining aggregation conventions are frozen as follows:
the hard-null envelope is the maximum of `raw_state`, `random_neighbor`,
`spatial_permutation`, and `rgb_gradient` (the `radius1_only` ablation is
excluded); macro shift reduction is the ratio of object-mean inflations;
object bootstrap uses 2,000 SHA-seeded resamples of the eight objects; null
superiority is required for every object, acquisition population, and metric.
Soft no-collapse means no object/acquisition falls below raw by more than
`0.020` pAUROC or `0.030` F1. The synthetic stimuli and their signs reuse the
frozen CF-NTR Stage -1 stimulus protocol, while spatial permutation alone uses
the FLRS protocol seed.

Soft continuation requires macro shift reduction at least 25%, defect
noninferiority, positive null margin on at least four objects, and no object
collapse. It authorizes one new-cycle revision only, never a public claim.

Any validity/leakage failure, universal null absorption, defect collapse, or no
shift reduction gives `KILL_FOR_CLAIM / NO_CONTINUE` for exact FLRS. A strict
pass gives `PROMISING_DIAGNOSTIC` and authorizes one frozen full-eight public
shadow. Public labels may not change this signature.

## Remote and cleanup

Run in fixed dsba3 container `hun_fsad_tta_012` on host GPUs `0,1,2,3`, two
objects per GPU. Pull compact JSON/TSV/log/manifests. Do not persist input
variants, dense features, or `anomaly_maps/`; verify zero regenerable dense
output directories after conclusion.
