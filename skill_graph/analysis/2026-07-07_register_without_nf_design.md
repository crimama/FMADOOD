# Register Usage Without Normalizing Flow

Date: 2026-07-07
Status: design note
Scope: MVTec AD2 single-image, FlowTTE/Flow-LatentBank follow-up

## Starting Evidence

Recent all-eight MVTec AD2 component ablation shows:

- `CLS soft w10`: `seg_AUROC_0.05=0.805427`, `seg_F1=0.447118`
- no-context base: `0.797743`, `0.437800`
- `register soft w5`: `0.798287`, `0.438193`
- `register-conditioned NF`: `0.796554`, `0.436152`
- `CLS top-M4`: `0.797035`, `0.433893`
- `register NF + CLS top-M4`: `0.793241`, `0.426666`

Reduced morphology audit on `fabric,can,wallplugs,vial` showed a different
signal: register-conditioned NF reduced good false-positive area and component
count and improved bad precision, even though all-eight mean metrics did not
improve.

Implication: register is not currently a good direct retrieval key or primary
score source. Its plausible use is nuisance control, patch purification, score
calibration, or morphology control.

## Negative Evidence Intake

Avoid these retreads:

- register token as anomaly localization map;
- hard top-M memory routing by register similarity;
- direct score fusion with register similarity;
- another context-weight sweep without a new mechanism.

The branch should test whether register removes global/context contamination
from patch evidence, not whether another scalar weight can tune a killed
retrieval branch.

## Candidate A: Register-Residual Patch Features

Claim: register tokens carry image-level nuisance/style/context components that
should be removed from patch tokens before patch-level nearest-neighbor scoring.

Base pipeline:

```text
patch p_i
-> static support patch memory
-> nearest distance d_i
-> anomaly map
```

Proposed NF-free residualization:

```text
support:
  fit W from image register g_j to patch features p_{j,k}
  p_{j,k} ~= W g_j + b

support/test:
  r_i = LN(p_i - alpha * W g_parent)
  score_i = min ||r_i - r_bank||
```

Use ridge regression or low-rank CCA/PCA directions fit on support normals
only. The safer version removes only the top-k patch dimensions most
predictable from register context.

Controls:

- `alpha=0`: raw patch kNN base.
- random global vector residualization.
- CLS residualization.
- register residualization.
- CLS+register residualization.

Keep signal:

- mean `seg_AUROC_0.05` and `seg_F1` improve over NF-free raw patch kNN;
- good false-positive area/components decrease;
- patch-score/background shortcut overlap decreases.

Hard stop:

- residualization improves good-map cleanliness but loses bad recall/F1;
- random global residualization matches register residualization.

## Candidate B: Register-Conditioned Score Calibration

Claim: register is better as score normalization context than as a memory
selector. Previous hard top-M selection likely discarded useful patch memory.

Keep all support patches for nearest-neighbor retrieval:

```text
d_i = min_m ||p_i - m||
```

Use register only to estimate the expected normal distance distribution:

```text
w_j = softmax(cos(g_test, g_j) / tau)
mu_g, sigma_g = weighted leave-one-out support distance stats
score_i = (d_i - mu_g) / sigma_g
```

This differs from top-M retrieval: the memory is not restricted. Register only
calibrates score scale for current image context.

Controls:

- global calibration over all supports.
- CLS calibration.
- register calibration.
- CLS+register calibration.
- random support weights.

Keep signal:

- lower good positive area/component count;
- stable or improved `seg_F1`;
- no collapse on strong objects such as `rice/walnuts`.

Hard stop:

- gains are only threshold/F1 movement without AUROC improvement or map
  cleanliness;
- CLS calibration dominates register calibration on both metric and morphology.

## Candidate C: Register Shortcut Suppression

Claim: some patch tokens act as global/background shortcut carriers. Register
can identify patches overly aligned with global context and suppress their
contribution to anomaly maps.

Diagnostic score:

```text
shortcut_i = cos(LN(p_i), LN(g_parent))
```

or multi-register variant:

```text
shortcut_i = max_r cos(LN(p_i), LN(R_r))
```

Use this only as a suppression mask, not as an anomaly score:

```text
score_i' = score_i * (1 - gamma * q_i)
```

where `q_i` is high only for patches whose shortcut score is in the top normal
support percentile. The goal is to remove globally aligned background false
positives, not to re-rank true anomalies directly.

Controls:

- patch norm suppression.
- CLS shortcut suppression.
- register shortcut suppression.
- random vector suppression.

Keep signal:

- reduced good false-positive area/components;
- lower overlap between anomaly maps and high shortcut maps;
- no loss in bad recall on `fabric/vial`.

Hard stop:

- suppression removes true defect regions;
- patch norm control matches register.

## Candidate D: Register-Diverse Support Coreset

Claim: reference images should cover both patch-level normal variation and
global context/style variation. Register is useful for support-set coverage,
not test-time scoring.

Selection objective:

```text
candidate_gain =
  patch_coreset_gain
  + beta * register_context_coverage_gain
```

This is secondary because support-selection-only methods are weak as method
claims. It is useful as a diagnostic to test whether register adds context
coverage missed by CLS.

Controls:

- DINOv3 CLS coreset.
- register-only coreset.
- patch+register joint coreset.
- random 16-shot.

Keep signal:

- improves both raw patch kNN and CLS soft context variants;
- support diversity improves without overfitting weak objects only.

Hard stop:

- gains disappear under fixed SuperAD-16 reference set;
- random coreset matches the joint criterion.

## Candidate E: Register-Guided Morphology Prior

Claim: register can identify image-level context where the base map produces
fragmented false positives. This is a post-threshold cleanup use, not a primary
score use.

Fit support-only leave-one-out normal map statistics:

```text
for each support image:
  score support patches against other support patches
  threshold by normal percentile
  record component count, area, largest-component share
```

At test time, use register-nearest support normal-map statistics to set a
conservative component filter.

Controls:

- global morphology prior.
- CLS-nearest morphology prior.
- register-nearest morphology prior.
- fixed component-size threshold.

Keep signal:

- F1 improves without AUROC degradation;
- reduced good component count and positive area;
- no large loss in bad recall.

Hard stop:

- improvement is threshold-only and does not preserve AUROC/map coverage;
- fixed morphology matches register-conditioned morphology.

## Recommended Order

1. Register-conditioned score calibration.
   - Lowest implementation risk.
   - Directly addresses why top-M failed.
   - Uses register as context, not localization evidence.

2. Register-residual patch features.
   - Strongest structural hypothesis.
   - Directly tests whether register removes global nuisance from patch tokens.

3. Register shortcut suppression.
   - Good diagnostic for LazyStrike-style background shortcut failure.
   - Needs careful recall-preservation gate.

4. Register-guided morphology prior.
   - Useful only if the target is F1/false-positive cleanup.
   - Not a core anomaly evidence mechanism.

5. Register-diverse support coreset.
   - Diagnostic only; do not promote as a main method claim.

## Small Preflight Design

Run on reduced objects first:

```text
objects = fabric, can, wallplugs, vial
backbone = dinov3_vitl16
support = same fixed DINOv3 no-context 16-shot reference paths
base = NF-free raw patch kNN
metrics = seg_AUROC_0.05, seg_F1, good positive area, component count
```

Variants:

```text
raw_patch_knn
cls_score_calibration
register_score_calibration
register_residual_alpha_0.25
register_residual_alpha_0.50
register_shortcut_suppression
```

If none beats raw patch kNN on both mean F1 and good-map morphology, stop the
NF-free register branch. If one variant improves reduced morphology without
hurting AUROC, scale to all eight objects.

## Current Preferred Direction

The most defensible next method shape is:

```text
DINOv3 patch token -> static patch memory distance -> anomaly map
DINOv3 register token -> score calibration / nuisance residualization
```

Register should remain outside the direct localization evidence path.
