# SOVM Negative-Evidence Intake and Design Rationale

Date: 2026-08-11
Status: `DESIGN READY / NOT RUN`

## 1. Why this is not a closed-branch retread

The completed SO-CFM experiment used a learned field to transport the support
bank and then applied position-aligned endpoint distance. It lost to the exact
discrete orbit bank and its anomaly guard never activated. SOVM freezes the
normal field and never uses it to alter support or query features. It scores
the disagreement between two vector quantities along an affine feature path:

1. the geometric velocity from a clean support anchor toward the observation;
2. the normal nuisance velocity predicted by the support-trained field.

The killed query-orbit persistence branch generated multiple transformed query
views, inverse-aligned their score maps, and aggregated persistence. SOVM does
not transform the query, does not inverse-align maps, and does not fuse maps
from query views. Enumeration is over a frozen support-derived condition basis
inside one score definition.

The killed score-aligned rank-4 residual branch projected nearest-neighbour
residuals into a support-estimated subspace. SOVM instead evaluates a
time-conditioned vector field at intermediate path states. Raw distance,
orbit-bank distance, endpoint correction, and constant/shuffled fields remain
mandatory hard controls so that a distance repackaging cannot pass.

## 2. Main failure basins

The experiment is designed to distinguish the following outcomes.

- **Field misspecification**: even held normal paths disagree with the field.
- **Distance duplicate**: SOVM ranks patches exactly like raw or orbit distance.
- **Condition collapse**: one low-energy nuisance code wins nearly everywhere.
- **Oracle-only success**: the true code or unavailable clean query source is
  required.
- **Synthetic shortcut**: the score detects generator texture or magnitude but
  not disjoint local and broad anomaly families.
- **Anomaly absorption**: condition enumeration explains defect motion as a
  normal nuisance.
- **Inactive synthetic supervision**: the Stage G ranking loss has zero active
  pairs before fitting, repeating the old guard failure.

## 3. Literature-to-design translation

Flow Mismatching defines an affine path from a source sample to the target and
scores squared disagreement between the learned normal velocity and the path's
geometric velocity, aggregated over paths and times. SOVM maps its Gaussian
source samples to spatially aligned support anchors and maps its unconditional
normal flow to a signed nuisance-conditional DINO feature flow.

The translation is deliberately conservative:

- squared velocity residual is the only primary mismatch term;
- time weights are fixed to `t^2`;
- `t=1` is omitted;
- scored times `0.25/0.75` are exactly the two finite-difference midpoint times
  supervised by the three support trajectory samples;
- path aggregation is a minimum, asking whether any normal support path
  explains the observation;
- the field is trained on actual DINO endpoints produced by image transforms,
  not arbitrary latent endpoints;
- all condition energies are normalized with fit-support statistics before
  condition enumeration.

Theoretical results for Gaussian-to-image flow do not prove the DINO support
adaptation. This stage is therefore a falsification assay, not a transfer of
the paper's performance claim.

Primary references:

- Flow Matching: https://arxiv.org/abs/2210.02747
- Flow Mismatching: https://arxiv.org/abs/2605.23070
- High-dimensional reverse-flow caution: https://arxiv.org/abs/2508.05461
- What Should Not Be Contrastive: https://arxiv.org/abs/2008.05659

## 4. Why synthetic anomaly training is sequential

Stage M first tests whether a normal-only field supplies anomaly-aligned
mismatch evidence. If it does not beat constant/shuffled fields and endpoint
distance, synthetic supervision would merely train a new synthetic anomaly
classifier around a dead evidence source.

Only a surviving Stage M authorizes Stage G. Stage G uses CutPaste and
chromatic blotch in a score-aligned ranking loss, while thin scratch and broad
Gaussian blur remain disjoint held families. Stage G evaluation also uses a
new unopened SHA seed domain rather than the Stage M masks, preventing the
conditional launch from reusing its gate observations. The old endpoint-
distance hinge is not reused. Causal value requires held-family improvement
over a same-step normal-only continued-training control and a spatial
mask-shuffle training null.

## 5. Future benchmark boundary

No TESTpublic run is part of this protocol. A future all-eight benchmark is
licensed only after `PROMISING_DIAGNOSTIC` and requires a separate
preregistration using:

- `configs/baselines/same16_superad_ad2.yaml`;
- `configs/baselines/same16_rn_fmlk_ad2.yaml`;
- the exact same fixed reference manifest;
- full `test_public/good,bad`;
- `seg_AUROC_0.05`, `seg_F1`, map coverage, RN-FMLK margin, and hard-null
  margin.

Historical same-16 artifacts are context only until their exact split,
preprocessing, reference budget, and evaluator parity are revalidated.
