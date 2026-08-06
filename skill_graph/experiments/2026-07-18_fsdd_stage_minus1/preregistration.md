# FSDD-LODF Stage -1 Support-Only Preregistration

Date: 2026-07-18
Executable protocol: `fsdd-lodf-stage-minus1-support-only-20260719-r1`

## Claim boundary

This is a support-only synthetic identifiability assay.  It cannot establish a
public method improvement.  TESTpublic/private, ground truth, TTA, query
transformation/update, routing, map switching, and fusion are forbidden.

## Population

- exact support manifest
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`;
- SHA-256 `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`;
- all eight MVTec AD2 objects, SHA-derived four image folds, 12 train / 4 held out;
- frozen DINOv2 ViT-L/14, layers `[5,11,17,23]`, 672 pixels except
  `sheet_metal=448`.

## Candidate

FSDD learns a hidden-free patch classifier from the factorial cells clean,
shift-only, defect-only, and shift+defect.  The full arm uses balanced
classification plus fixed paired clean/shift and defect/shifted-defect
consistency.  The initial assay uses fixed deterministic training with no
early stopping or metric-selected retry.  Evaluation crosses held-out support
sources with both a leave-one-defect-family-out head and a
leave-one-shift-axis-out head.  The excluded axis at 1x and 2x is primary;
compounds are secondary.  Two fixed SHA training seeds must independently
satisfy directional gates.

The executable head is a hidden-free linear 1x1 scorer over fixed signed
channel pools from the four layers, trained for exactly 400 updates.  If Stage
-1 passes, the deployable object head is retrained once on all 16 supports,
all four defect families, and all four shift axes with the preregistered seed;
its q=0.995 threshold is computed only from its all-support clean training
maps.  Cross-validation metrics never select its seed, updates, or threshold.

Each source/cell contributes a fixed eight sampled patches per available
class.  Short boundary interiors are sampled with deterministic replacement so
all equal-capacity learned controls receive the same row budget.  Treatment
and nonfactorial supervised-only use byte-identical C/S/D/SD rows and differ
only in the frozen paired loss.  The treatment penalty is the sum of
`(C-S)^2`, `(D-SD)^2`, and the factorial interaction
`((D-C)-(SD-S))^2`; thus a shared shift logit offset cannot satisfy the loss.

## Controls

Equal-capacity nonfactorial/supervised-only, defect-only, deterministic
stimulus-label shuffle, toroidal mask shuffle, same-feature/capacity
boundary-only, RGB seam-only, RGB texture-only, and raw-state energy.  Defect interiors excluding
the synthesized boundary are reported separately.  All learned arms share
features, source folds, seeds, optimizer, and update budget.

`raw_state` and nonfactorial are operational baselines.  The hard-null
envelope is recomputed from label-shuffle, mask-shuffle, boundary-only,
RGB-seam, and RGB-texture arms within every bootstrap replicate; defect-only is a mechanism
ablation and is not silently folded into either comparison group.

The `rgb_texture` null is absolute four-neighbour grayscale Laplacian energy
(`cv2.Laplacian`, `ksize=1`) resized to the identical DINO patch grid.  It uses
the same defect image and metric mask as every learned arm, and its threshold is
the q=0.995 quantile of training-source clean maps.  This fixed null prevents a
high-frequency raster pattern from being credited as semantic discrimination.

## Metrics

- train-clean 0.995 threshold, held-out clean FPR, and positive shifted-normal
  FPR excess;
- held-out-family clean and shifted defect pixel pAUROC@0.05 and
  train-threshold fixed F1 are co-primary; oracle best F1 is diagnostic only;
- per-object and per-family values, finite/nonconstant/aligned coverage;
- source-macro summaries and 2,000 hierarchical bootstraps over object, source,
  LODF/LOSF head, and seed, with hard-null envelopes recomputed per replicate.

Synthetic generation uses a fixed metric-blind retry order.  A candidate must
change pixels and produce a nonconstant feature delta before it can enter any
arm; the same accepted stimulus is shared by every treatment/control.  No
score, label metric, or downstream loss can select a retry.

## Strict gate

All validity/access/LODF audits and map coverage must pass.  Clean FPR must be
at most 0.02 for every object.  Shift FPR excess must improve at least 30%
over both operational baselines with positive bootstrap LCB and no object harm
over 0.01.  Clean and shifted held-out-family pAUROC/F1 must be noninferior to
raw and supervised-only by 0.005/0.010.  Treatment must exceed the hard-null
envelope by 0.005 pAUROC and 0.010 F1 in both acquisitions with positive LCBs.
All four families must be raw-noninferior and at least three must have positive
dual-metric null margins.  Shifted-defect drop must be no more than 0.010
pAUROC and 0.020 F1 and smaller than supervised-only.
In addition to the all-family gate, the high-frequency-ribbon-only macro and
the ribbon-excluded three-family macro must independently pass every shift,
retention, hard-null, and shifted-defect-drop directional gate under both
fixed seeds.

Soft continuation requires 20% shift-excess improvement, raw defect
noninferiority, positive hard-null margins on at least two families and four
objects, and no family collapse.  It authorizes only one fixed seed-stability
repeat.  Every directional gate must pass under both fixed training seeds.
Otherwise verdict is `KILL_FOR_CLAIM / NO_CONTINUE`.

The unrevised design scored 0.71 in independent adversarial review.  The
nested LOSF, linear head, final all-16 rule, fixed-F1, two-seed, boundary null,
and feature-visibility corrections above raised the executable design score to
0.83.  Only this revised version is authorized to run.

## Pre-metric protocol amendment r1 (2026-07-19)

Attempts v1--v3 produced no object summary and no accepted method metric.  v3
showed that a thin raster scratch cannot have a positive after the frozen 3x3
patch-grid erosion on every support resolution.  The construct is therefore
renamed, before metric access, to `high_frequency_ribbon`; it is **not** a
thin-scratch proxy and no scratch-generalization claim is allowed.  Its center
line is supplied by the sealed scratch generator, thickness is exactly
`ceil(4*max(H/grid_h,W/grid_w))`, and fill is a one-pixel 0/255 checkerboard in
all three RGB channels.  Retry order and all downstream masks remain shared
across arms.

An independent review scored the first ribbon amendment 0.79 and prohibited
launch because the existing Sobel seam null could miss the checkerboard
interior.  Protocol r1 adds the same-image Laplacian `rgb_texture` hard null
and the ribbon-only/ribbon-excluded dual gate defined above.  Source constants,
the exact support manifest, seeds, head optimization, thresholds, metrics, and
all other gates are unchanged.  A new empty result root is mandatory; v1--v3
cannot be resumed or aggregated.
