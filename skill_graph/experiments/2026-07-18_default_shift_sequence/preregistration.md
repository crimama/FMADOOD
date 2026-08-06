# Default FlowTTE Distribution-Shift Sequential Experiment

Date frozen: 2026-07-18, before Stage 1 metrics

## Claim boundary

This is a sequential MVTec AD2 `test_public` shadow diagnostic. It may identify
a continuation mechanism, but it cannot support `KEEP`: the public population
has already been inspected and no fresh same-run paper-aligned SuperAD/RN-FMLK
confirmation population is available. `TESTprivate` is forbidden. Query
adaptation, query memory updates, and TTA are forbidden in every stage.

## Frozen target and default

- target: MVTec AD2 single-image at
  `/home/hunim/Volume/DATA/mvtec_ad_2`;
- population: all eight objects and full `test_public/good,bad`;
- support: the exact fixed SuperAD-selected 16 paths used by the retained H+
  reference, identity views unless a Stage 2 support bank explicitly says
  otherwise;
- default: DINOv3-H+/16 `[7,15,23,31]`, native 672 except sheet-metal 448,
  layer-normalized mean fusion, DVT position mean alpha 1, three-epoch/two-
  coupling-layer fused NF, static latent memory, LOO standardization, latent
  NN distance plus density weight 0.25, guided-r8, then line-17/16-angle
  close-fill-erode;
- primary metrics: decimal pixel `seg_AUROC_0.05`, best-threshold `seg_F1`, and
  complete finite/nonconstant map coverage;
- condition diagnostics: per-image pixel-score p95, condition mean p95,
  shifted-good inflation from regular, bad-good p95 separation, and
  object-condition pAUROC/F1;
- execution: dsba3 fixed container `hun_fsad_tta_012`, host GPUs 0--3.

Historical H+ `0.836739/0.527427`, guided+morph `0.837528/0.553859`, and
SuperAD/SuperADD context are context only. `comparable=false` for a method
claim throughout this sequence.

## Negative-evidence intake

The sequence does not retune TTE, hard-local matching, position centering,
unconditional density, transition-direction residuals, or the killed global
hard projection. It first attributes shift inside the retained default. The
new Stage 2 evidence source uses an early/global acquisition estimate only to
select one fixed normal support bank; it cannot modify the query, update
memory, or select patches. Stage 3 is a safety bypass for an already-retained
postprocessor. Stage 4 is a bounded soft/local contrast to the killed global
hard projection and has no promotion path in this sequence.

## Stage 1 — A0--A3 default-chain attribution

One deterministic NF fit recipe is rerun with identical support and seed for:

- A0: LOO-standardized latent NN distance, density weight 0;
- A1: A0 plus the operational density penalty at weight 0.25;
- A2: guided-r8 applied to A1, no binary morphology;
- A3: A2 thresholded at A2's object-global raw best threshold, then frozen
  close-fill-erode morphology.

A0/A1 are separate deterministic executions because the production API does
not emit score components. Training inputs and hyperparameters must match and
support manifests must be identical. A2/A3 must reuse A1 maps byte-for-byte.

Attribution thresholds, used only to describe the failure location:

- material metric change: absolute macro delta at least 0.005;
- material normal-tail change: absolute mean shifted-good inflation delta at
  least 10% relative to the preceding arm;
- object harm: pAUROC below -0.02 or F1 below -0.03 versus the preceding arm;
- validity: exact full-population coverage, finite/nonconstant maps, and exact
  A1 source identity for A2/A3.

Stage 1 dense maps may remain remotely only until frozen Stage 3 evaluation;
this temporary exception must be recorded and then cleaned.

## Stage 2 — acquisition-routed fixed support bank

This stage is frozen regardless of Stage 1 outcome so sequential public
inspection cannot choose its structure.

- Fit the default NF on identity support only.
- Construct fixed support latent banks from identity and six photometric
  support transforms: exposure `+/-0.25 EV`, gamma `+/-0.10` in log exponent,
  and reciprocal red/blue white balance `+/-0.08`.
- Estimate one image-level acquisition state from the mean normalized earliest
  H+ layer. Select exactly one bank by nearest support prototype.
- If the nearest/second-nearest cosine-distance margin is below the 5th
  percentile of correctly routed leave-one-support-view margins, fall back to
  identity.
- The selected bank is fixed for the whole image. No patchwise bank choice,
  bank union, query transformation, GT/split/filename condition, TTA, or query
  memory update is allowed.

Controls: identity A1, union-bank distance, patchwise minimum bank distance,
and estimated single-bank routing. Primary continuation gate: routing improves
macro pAUROC or F1 by at least 0.005, reduces shifted-good inflation by at
least 15%, does not reduce macro bad-good separation, and has no object below
`-0.02/-0.03`. Otherwise `NO_CONTINUE` for routing.

Operational amendment after the first Stage 2 launch, before any Stage 2
metric was produced: `fabric` had zero correctly routed leave-one-support-view
examples, so the frozen 5th-percentile threshold was mathematically undefined.
For this pre-existing edge case only, zero correct examples now fails closed:
the confidence threshold is the maximum finite float and every query uses the
identity bank. This preserves the preregistered fallback direction, uses no
query labels or outcome metric, and records zero correct routes as a routing
identifiability failure rather than terminating the sequence. The invalid
first launch is excluded and Stage 2 is restarted from an empty result root.

Second operational amendment, before any valid Stage 2 summary was produced:
the all-object dense-map accumulation exhausted the remote filesystem while
Stage 1 maps remained sealed for Stage 3. Stage 2 now evaluates and validates
each completed object's four frozen arms immediately, retains its compact JSON
summary, and deletes only that object's Stage 2 dense maps before launching the
next object on the same GPU. The final summary aggregates those object JSON
records with the unchanged macro functions and gates. No score, metric,
population, routing decision, or scientific threshold changes. The disk-full
partial launch is excluded and restarted from an empty Stage 2 result root.

During the valid streaming launch, slow CPU evaluators briefly overlapped later
GPU objects and `df` reported less than 1 GiB free. No write, coverage, or
checksum failure occurred: all eight compact summaries completed, all Stage 2
dense maps were removed, and the final aggregate passed. That completed launch
is therefore retained. The local controller was subsequently hardened to use
GPU 0--1 and GPU 2--3 waves on any future rerun, but this scheduling safeguard
was not present in, and does not affect, the retained Stage 2 result.

## Stage 3 — label-free guided fallback

Using the exact Stage 1 A1 and A2 maps, choose A2 per image only when all frozen
conditions hold; otherwise use A1:

- Spearman correlation of raw/guided pixel ranks at least 0.90;
- guided/raw top-1% mean score ratio at least 0.90;
- raw/guided maximum-score location displacement at most 10% of the image
  diagonal.

No GT, split name, object identity, condition suffix, or metric may enter the
gate. Controls are always raw A1 and always guided A2. Continuation requires
macro pAUROC/F1 improvement of at least `0.002/0.005` over always-guided or
elimination of every `-0.02/-0.03` object harm without lowering either macro
metric. Otherwise stop this gate family.

## Stage 4 — local soft nuisance correction

This is a diagnostic contrast, not a rescue of CF-NTR. Using support-only
central differences for the same six photometric axes:

- fit separate rank-2 tangent bases for each feature layer and each 2x2 spatial
  quadrant after subtracting the quadrant mean delta;
- shrink only 25% of the projected query residual relative to the identity
  support position mean;
- retain the default fused NF scorer and identity memory;
- compare identity default, global hard rank-4 projection, and local soft
  rank-2 shrinkage on a frozen compact orbit of the first two good and first
  two bad source identities per object across every available acquisition
  condition.

Continuation requires at least 25% shifted-good inflation reduction, no loss
of bad-good separation, macro pAUROC/F1 deltas at least `-0.005/-0.010`, and
no object below `-0.02/-0.03`. Global hard projection is a negative control
and cannot be promoted.

## Sequential verdict rules

Each stage receives its own validity and continuation verdict before the next
stage is executed. Because the user requested all four attempts, a valid hard
negative does not cancel later preregistered stages; it only prevents that
stage's mechanism from being reused or tuned. The final report ranks mechanisms
by default preservation, shift-tail reduction, anomaly separation, object
no-harm, and complexity. Maximum verdict is `PROMISING_DIAGNOSTIC`.
