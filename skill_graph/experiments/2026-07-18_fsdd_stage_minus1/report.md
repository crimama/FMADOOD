# FSDD-LODF Stage -1 Report

Date: 2026-07-19
Verdict: `KILL_FOR_CLAIM / NO_PUBLIC_PROMOTION`

The first remote attempt `fsdd_stage_minus1_20260719_c2_v1` was invalidated
before any object summary or method metric: the patch mask remained flat after
rasterization, so the two-dimensional toroidal mask-shuffle null raised a shape
error.  The partial shards were stopped, the failing log was retained locally,
and both remote result/snapshot roots were removed.  No scientific verdict is
assigned to v1.

The mask was explicitly reshaped to the frozen DINO grid and a regression test
now exercises erosion, boundary separation, and mass-preserving toroidal
shuffle.  Attempt v2 reached the next validity check but produced no summary:
thin scratches on one `can` and one `vial` support had fewer than two
nonconstant patches *after erosion*.  That incorrectly coupled the generation
visibility population to the boundary-excluded metric population.

The preregistered nonconstant visibility check now uses every patch overlapping
the native defect, while pAUROC/F1 still excludes the synthesized boundary and
uses only eroded interior positives plus outside-mask negatives.  A regression
test proves the full-overlap stimulus can pass while a one-patch eroded subset
cannot.  v2 partial shards were stopped and removed with zero accepted method
metrics.  Local verification reached `18 passed`, Ruff clean, and shell/Python
syntax clean.  The empty-root rerun `fsdd_stage_minus1_20260719_c2_v3`
nevertheless failed the same pre-metric interior-validity condition: a
one-patch-scale thin scratch cannot retain an interior after the frozen 3x3
patch-grid erosion.  All shards were stopped with zero summaries; partial logs
were retained as invalid-attempt evidence and the remote result/snapshot roots
were removed.

A pre-metric amendment changed the construct, not the method result: the fourth
family is now explicitly `high_frequency_ribbon`, with no thin-scratch claim.
Its fixed four-patch-span 0/255 checkerboard ribbon creates a measurable patch
interior, but an independent adversarial review scored the initial amendment
only `0.79`, below the `0.80` launch threshold.  It identified a texture
shortcut not covered by the Sobel seam null, so no v4 run was launched.

Executable protocol r1 now adds a same-image absolute Laplacian
`rgb_texture` hard null, includes it in every hard-null envelope, and requires
both ribbon-only and ribbon-excluded three-family macros to pass every
directional gate under both seeds.  The family name, exact thickness/fill rule,
new protocol ID, and mandatory empty run root are frozen in the amended
preregistration.  Focused validation is `22 passed`.  Independent re-scoring
returned `0.84` (`launch_authorized=true`) after verifying that the Laplacian
null responds on all 38 eroded-interior patches of an actual ribbon fixture and
that either subset failure blocks the strict gate.  Authorization is limited to
a new support-only Stage -1 root.  No public data is authorized unless every
strict support gate passes.

## Completed r1 result

The authorized empty-root run `fsdd_stage_minus1_20260719_c2_v4` completed all
eight objects and both frozen seeds on dsba3 GPUs 0--3.  It produced 4,096
atomic source × LODF × LOSF × seed records.  All 16 object/seed summaries,
compact artifacts, and pullback checksums verified; TESTpublic/private,
ground-truth, TTA, query update, and query transform access remained false.
The remote source snapshot was removed after the verified pullback.

G1--G6 passed, both seeds agreed on every all-family directional gate, and the
macro treatment was substantially stronger than the same-cell nonfactorial
control: clean pAUROC@0.05 `0.856416` versus `0.825570` (`+0.030846`) and fixed
F1 `0.545078` versus `0.501497` (`+0.043580`).  Shifted values were
`0.855609/0.544427` versus `0.824512/0.498379`.  Shift FPR excess was reduced
`54.20%` versus nonfactorial and `39.81%` versus raw state, with positive 2.5%
bootstrap LCBs.  All-family hard-null margins were `+0.150635` pAUROC and
`+0.263711` F1 on clean defects, and `+0.150656/+0.265144` under held-out
shifts.

Strict promotion nevertheless failed G7 under both seeds.  On
`high_frequency_ribbon`, the frozen Laplacian null achieved pAUROC `1.000` and
fixed F1 `0.858046`; treatment reached only `0.967612/0.749566` (seed 0) and
`0.961114/0.740914` (seed 1).  The negative hard-null margins prove that the
fourth construct is primarily a raster-texture shortcut, not learned
shift/defect factorization.  `fruit_jelly` and `vial` produced the largest
treatment-minus-texture deficits.

The ribbon-excluded three-family macro did show a reproducible factorial
benefit: clean pAUROC improvement over nonfactorial was `+0.042571/+0.042930`
for seeds 0/1, with hard-null margins `+0.164865/+0.157198`.  It missed only
the strict relative shifted-defect pAUROC-drop comparison: treatment versus
nonfactorial was `0.000693` versus `0.000149` at seed 0 and `0.000638` versus
`0.000634` at seed 1.  F1 drop was better than nonfactorial under both seeds.

The aggregate soft gate passed, but an exact fixed repeat cannot resolve either
blocking fact: the texture arm and its perfect ribbon ranking are deterministic,
and both training seeds already agree on the failure.  Therefore the optional
one-repeat authorization is not consumed.  Public evaluation and an all-16
deployable head remain forbidden.  The transferable Cycle-3 seed is restricted
to the three non-ribbon families and must explicitly repair worst-axis defect
drop before any public comparison.

The runner emits one record per source × held-out defect family × held-out
shift axis × seed.  Thresholds are q=0.995 of training-source clean maps;
excluded-axis x1/x2 normal FPR and clean/x1/x2 defect metrics remain atomic
through the object→source→LODF/LOSF→seed bootstrap.  The deployable all-16
head is intentionally not fitted until the strict support gate passes.
