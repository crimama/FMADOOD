# GB-CMF Stage -1A Preregistration

Date: 2026-07-19  
Status: `COMPLETED_KILL_FOR_CLAIM / NO_PUBLIC_ACCESS / NO_STAGE_MINUS1B`

## Question

Can changing only the default H+ flow geometry create joint feasibility for
shifted-normal tail control and shifted-defect margin retention, while the
deployed score remains exactly hard latent 1NN distance plus `0.25` frozen-NLL
penalty?

## Frozen scope

- exact SuperAD-16 train/good manifest SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`;
- DINOv3-H+/16 layers `[7,15,23,31]`, layer-normalized mean, fold-train-only
  DVT position-mean alpha 1;
- 12 SHA-ordered train sources and four held sources per case;
- held family and axis are excluded from training;
- five arms: default, matched nonfactorial flow, GB-CMF, affine score null,
  nonlinear residual-head null;
- 300 aligned updates per learned arm, shared deterministic row schedule;
- exact-hard held scoring and full train-memory proxy audit;
- no TTA, query update/transform, routing, map fusion, public/private/test or
  ground-truth enumeration.

The four immutable cases are rice/0 fold 0 cutpaste×exposure on GPU 0; can/1
fold 1 chromatic-blotch×gamma on GPU 1; rice/1 fold 2 Gaussian-blur×contrast
on GPU 2; and can/0 fold 3 cutpaste×white-balance on GPU 3. Five arms are run
sequentially within each GPU, for exactly 20 arm fits.

## Strict decision

Every held source must satisfy `g_tail<=0` and `g_drop<=0`. GB-CMF must improve
both constraints over default and nonfactorial, pass all geometry and
soft/hard-proxy guards, retain component and total-score variance, beat the
maximum affine/nonlinear null by `0.005` pAUROC@0.05 and `0.010` fixed F1, and
remain raw-noninferior within `0.005/0.010`. Both seed directions must agree;
map and access coverage must be 100%.

A scientific failure is retained with named kill codes and no retry/tuning. A
pass is still diagnostic only: it requires a new post-result audit before any
Stage -1B transition. Public access and default-outperformance claims remain
forbidden.

The complete formula, geometry limits, custody rules, and assignments are in
`.omx/specs/deep-research-autoresearch-waff-infeasible-structural-redesign/stage-05/execution-manifest.md`.

The valid v3 execution completed all 20 fits and failed the scientific gate.
The frozen result and post-result structural diagnosis are in `report.md`.
