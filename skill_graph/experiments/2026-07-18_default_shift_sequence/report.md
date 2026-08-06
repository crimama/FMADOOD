# Default FlowTTE distribution-shift sequential experiment

## Scope and verdict status

This is an all-eight MVTec AD2 `TESTpublic` shadow diagnostic of the frozen
H+ default. It uses the exact SuperAD-selected 16 normal references per object,
dsba3 GPUs 0--3, no TTA, no query transformation or memory update, and no
`TESTprivate`. All four preregistered stages are executed sequentially even
when an earlier gate fails. The maximum possible disposition is
`PROMISING_DIAGNOSTIC`; this report does not establish a method claim.

Run ID: `default_shift_sequence_20260718_v1`.

## Frozen operational default

- DINOv3-H+/16 layers `[7, 15, 23, 31]`, native 672 input except 448 for
  `sheet_metal`, `layer_norm_mean` fusion, no context feature.
- DVT position-mean correction at alpha 1, identity-only NF training for three
  epochs with two coupling layers, static expansion budget 1, support-LOO
  standardization, latent nearest-neighbor distance plus density NLL at 0.25.
- RGB guided-r8 refinement followed by the existing close/fill/erode
  morphology only in the post-processing arms.
- Exact support manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.

## Stage 1 — failure attribution

| Arm | pAUROC@0.05 | F1 | Shifted-good inflation | Bad-good separation |
|---|---:|---:|---:|---:|
| A0 distance only | 0.837518 | 0.523902 | 0.153952 | 0.165893 |
| A1 + density 0.25 | 0.836739 | 0.527428 | 0.157898 | 0.196794 |
| A2 + guided-r8 | 0.837528 | 0.546785 | 0.156850 | 0.196051 |
| A3 + morphology | 0.837528 | 0.553859 | 0.156850 | 0.196051 |

Density changed pAUROC/F1 by `-0.000779/+0.003526`. It increased bad-good
separation by `+0.030901`, but shifted-good inflation also increased by
`+0.003945` rather than falling. The effect is therefore a broader score
redistribution, not removal of acquisition shift, and neither primary metric
clears the frozen `+0.005` materiality gate.

Guided refinement changed A1 pAUROC/F1 by `+0.000789/+0.019357`, while
shifted-good inflation fell only `0.001047` and separation fell slightly. Its
main benefit is boundary/shape refinement after scoring, not distribution-
shift correction. The macro gain also hides a `-0.036604` AUROC loss on
`can`, violating the no-harm gate.

Morphology leaves ranking scores unchanged and raises pooled F1 by `+0.007075`,
but the gain is concentrated in `fabric` (`+0.067922` object F1). It lowers
condition-macro F1 on overexposed, regular, shift-2, shift-4, and underexposed
inputs; within `wallplugs`, regular/overexposed/underexposed losses are all
approximately `-0.11` or worse. A pooled thresholded-mask gain therefore
cannot be interpreted as robust correction of distribution shift.

Stage 1 conclusion: the default's principal risk is object- and acquisition-
condition heterogeneity hidden by pooled metrics. Density raises anomaly
separation while slightly worsening normal shift tails; guided and morphology
recover segmentation shape for some objects but create large local harms.

## Stage 2 — acquisition-routed fixed support bank

| Arm | pAUROC@0.05 | F1 | Shifted-good inflation | Bad-good separation |
|---|---:|---:|---:|---:|
| Identity | 0.836739 | 0.527427 | 0.157898 | 0.196794 |
| Union | 0.838529 | 0.527378 | 0.144423 | 0.199818 |
| Patchwise minimum | 0.838529 | 0.527378 | 0.144423 | 0.199818 |
| Routed single bank | 0.837341 | 0.527137 | 0.153619 | 0.195269 |

The first launch produced no evaluation metric and is invalid. `fabric` had
zero correct leave-one-support-view routes, so the 5th percentile of correct
route margins did not exist. Before reading any Stage 2 outcome metric, the
runner was changed to fail closed to the identity bank whenever this occurs.
The partial result and isolated source snapshot were deleted, a regression test
was added, and the full stage was restarted from an empty result root. This is
also direct mechanism evidence: for at least one object, the proposed early-
layer acquisition state is not identifiable even on its own transformed
support views.

A second launch was also invalidated before a final summary: accumulating all
four Stage 2 arms while retaining Stage 1 maps for Stage 3 exhausted remote
storage. The scientific computation and evaluator were unchanged; lifecycle
was corrected to evaluate/validate one object, retain its compact JSON, and
delete only that object's Stage 2 dense maps before the next object. A focused
regression test establishes equality between compact-object aggregation and
the original direct summarization. The disk-full partial result and snapshot
were removed before the clean streaming rerun.

The corrected streaming run completed 8/8 objects. Routed versus identity
changed pAUROC/F1 by `+0.000602/-0.000291`, reduced shifted-good inflation by
only `2.71%` relative, and reduced bad-good separation by `0.001525`. It passed
coverage and object no-harm but failed the performance, 15% tail-reduction,
and separation gates: `NO_CONTINUE`.

The union and patchwise-minimum arms are exactly identical, as expected from
nearest-neighbor minimization over a concatenated bank. They improve pAUROC by
`+0.001790`, reduce shifted-good inflation by `8.53%`, and increase separation
by `+0.003024`, but F1 changes by `-0.000049`. The pAUROC gain is dominated by
`can` (`+0.013173`); every other object is within approximately `+0.00133`,
and `sheet_metal` loses `0.005930` F1. This is weak heterogeneous support-bank
coverage, not a robust distribution-shift solution.

The routing mechanism explains why the routed arm loses even that weak union
signal. Across transformed support, leave-one-view route accuracy ranges from
0% (`fabric`) and 6.25% (`walnuts`) to 76.79% (`rice`); `fabric` must fall back
to identity for all 156 queries. Aggregating known acquisition conditions,
the nearest prototype matches identity on only 29.35% of regular views,
exposure-positive on 45.11% of overexposed views, and exposure-negative on
22.83% of underexposed views. Gamma and exposure prototypes are strongly
confounded, and white-balance-negative is never selected after fallback.

Stage 2 conclusion: transformed support contains a small nuisance-coverage
benefit, but the earliest-layer global mean is not an identifiable acquisition
state. Hard image-level bank selection discards useful local alternatives and
slightly erodes anomaly separation. This branch should not be tuned further;
any future use of support diversity must avoid claiming that the present
prototype route recovers the physical shift axis.

## Stage 3 — label-free guided fallback

| Arm | pAUROC@0.05 | F1 | Shifted-good inflation | Bad-good separation |
|---|---:|---:|---:|---:|
| Always raw | 0.836739 | 0.527428 | 0.157898 | 0.196794 |
| Always guided | 0.837528 | 0.546785 | 0.156850 | 0.196051 |
| Frozen fallback | 0.838376 | 0.542432 | 0.148115 | 0.174803 |

Fallback versus always-guided changes pAUROC/F1 by
`+0.000848/-0.004352`, below the direct `+0.002/+0.005` gate. It reduces the
`can` guided AUROC harm from `-0.036604` to `-0.021810` versus raw, but does
not cross the frozen `-0.02` no-harm boundary. Guided and fallback harm sets
therefore both contain `can`; neither continuation route passes.

The mechanism failure is more important than the small macro deltas. The
fallback lowers shifted-good inflation by `0.008736` versus always-guided, but
also lowers bad-good separation by `0.021247` (about 10.8%). Its gate checks
spatial rank, top-region mean ratio, and peak displacement; it does not make
raw and guided score scales exchangeable across images. The gate accepts bad
images at roughly 75% but good images at roughly 58%, so guided scores are
mixed into the bad distribution more often. This asymmetric cross-image score
mixing suppresses separation even when each selected map is spatially similar.

No image fails the Spearman threshold; rejected images are driven by peak
displacement (260 failures) and top-region ratio (133 failures, with overlap).
Thus the label-free gate does detect spatial instability, but spatial
agreement is not sufficient evidence that guided refinement preserves global
anomaly calibration or removes distribution shift. Verdict: `NO_CONTINUE`.

## Stage 4 — local soft nuisance shrinkage

| Arm | pAUROC@0.05 | F1 | Shifted-good inflation | Bad-good separation |
|---|---:|---:|---:|---:|
| Identity | 0.825469 | 0.488300 | 0.159126 | 0.354567 |
| Global hard rank-4 | 0.819189 | 0.475765 | 0.129810 | 0.334284 |
| Local soft rank-2 | 0.825946 | 0.487000 | 0.149657 | 0.356365 |

These are matched compact-orbit metrics and must not be compared directly to
the full-population Stage 1--3 absolute values.

Global hard projection reduces normal-shift inflation by `18.42%`, but changes
pAUROC/F1 by `-0.006280/-0.012535` and separation by `-0.020283`. It removes
defect evidence together with nuisance response and is not promotable.

Local-soft changes pAUROC/F1 by `+0.000477/-0.001300`, increases separation by
`+0.001798`, has no object-level harm, and completes every frozen orbit. It
therefore passes validity, no-harm, separation, and macro-noninferiority, but
normal-shift inflation falls only `5.95%`, far below the frozen 25% gate.
Object-level reduction is concentrated in `can` (`19.49%` with `+0.010051`
pAUROC); most other objects are below 5%, and `wallplugs` inflation worsens by
`0.95%`.

Tangent diagnostics show why soft local removal is safe but weak. Mean global
rank-4 energy coverage by layer is `0.673/0.316/0.269/0.393`, whereas quadrant
local rank-2 coverage is only `0.322/0.169/0.197/0.238`. `sheet_metal` has only
`0.139` mean local coverage and `rice` `0.192`. The three physical support axes
do not span most patch-conditioned deeper-layer drift. Correspondingly,
local-soft reduces inflation most on shift-1/shift-3 conditions, but barely
changes underexposure or shift-2.

Stage 4 conclusion: hard nuisance subtraction has enough strength to alter
shift response but is not defect-specific; local 25% shrinkage preserves
anomaly evidence but cannot capture the heterogeneous drift. This is an
identifiability/capacity trade-off, not a coefficient-tuning failure. Verdict:
`NO_CONTINUE`.

## Integrated conclusion

All four stages fail their continuation gates. Verdict:
`TESTPUBLIC_SHADOW_ONLY / NEGATIVE_OR_MIXED_CAUSAL_SIGNAL / KEEP_FROZEN_DEFAULT`.
No candidate is promoted and no TTA path is reopened.

The sequential interventions isolate four structural problems:

1. **Score separation is not shift removal.** Density raises bad-good
   separation but also raises shifted-normal tails. A stronger anomaly score
   can look better at a pooled threshold while becoming less acquisition
   stable.
2. **Support diversity is not an identifiable acquisition state.** Union
   banks contain a small local coverage benefit, but the early-layer global
   prototype confounds exposure, gamma, white balance, object identity, and
   spatial content. Hard single-bank routing loses the union benefit.
3. **Spatial agreement is not score exchangeability.** Guided fallback detects
   peak instability, but mixing raw and guided maps per image changes the score
   distribution asymmetrically between good and bad images and collapses
   separation.
4. **The nuisance/defect subspaces overlap.** Strong global projection removes
   both; conservative local shrinkage preserves defects but covers too little
   deeper-layer drift. The effect varies sharply by object and condition.

The common root is that acquisition shift is not represented as one shared,
low-rank, image-global factor independent of defect evidence. It is
object-, layer-, position-, and transformation-dependent. Pooled macro gains
therefore hide condition harms, while more aggressive correction suppresses
the anomaly signal it is supposed to preserve.

The operational recommendation is to keep the frozen identity-support A1
continuous score as the primary control. Guided/morphology may remain a
secondary segmentation presentation arm, but not evidence of distribution-
shift robustness; its condition-level harms must be reported. Do not tune the
tested router, fallback thresholds, or projection strength on this public
shadow. A future branch should first pass a support-only, held-out mechanism
assay that jointly proves nuisance-axis identifiability, cross-branch score
calibration, and synthetic/local defect retention before any new public run.

## Artifact and verification record

- Stage 1 compact pull checksum verified; 8/8 objects and 4/4 GPU groups
  completed.
- Stage 1 dense maps were retained remotely only until the preregistered Stage
  3 fallback was evaluated; Stage 3 then verified and removed them.
- The optimized Stage 1 evaluator uses an exact float16 65,536-bin histogram;
  tests establish agreement with the original exact rank computation.
- Focused local regression suite after the Stage 2 edge-case and streaming
  amendments: `19 passed`; focused Ruff checks passed.
- Stage 2, Stage 3, and Stage 4 compact pulls passed their SHA-256 manifests.
- Final remote cleanup verified all four stage completion markers, zero dense
  artifact directories, removal of both isolated source snapshots, 57 GiB
  free, and idle dsba3 GPUs 0--3.
