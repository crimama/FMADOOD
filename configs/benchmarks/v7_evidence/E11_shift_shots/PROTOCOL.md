# E11: MVTec AD 2 acquisition shift by support shot

Registered 2026-09-25. This is a follow-up to E10 and a development-set analysis,
not an untouched final evaluation. The acquisition strata and their labels have
already been inspected in earlier experiments. Freeze this comparison before
inspecting E11 results; any later method or plot selection is post-hoc.

## Hypothesis and comparison

At each support size, compare RSG3 with SuperAD (DINOv3), SuperAD (DINOv2),
and PatchCore on the same normal support images and `mvtec_ad2/test_public`
queries. The primary question is whether RSG3 retains the highest absolute
`shift_1` segmentation AUPRO@0.05 as support shrinks. A separate question is
whether its change from regular to shift is smaller. An absolute lead does not
establish a smaller shift drop.

These are end-to-end method comparisons. Each method retains its E10 native
backbone, feature layers, fitting and prediction settings: RSG3, SuperAD
(DINOv2), and PatchCore use DINOv2 ViT-L/14; SuperAD (DINOv3) uses DINOv3
ViT-L/16. The DINOv3 result therefore also changes backbone. Preprocessing
(`no_mask_no_rotation`), image size, seed 0, query set, masks, and per-stratum
evaluator follow E10. Ground-truth masks and condition labels are evaluator-only;
fitting receives normal training images only. No anomaly maps or fitted weights
are requested.

## Cells and support pairing

Run 1, 2, 4, and 8 shots with `e11_{method}_s{01,02,04,08}.yaml` for each of
the four methods. The intended E11 source snapshot is `a9b9e36` plus the
recorded E11 config snapshot; E10's remote 16-shot source snapshot was
`794529c`. Reuse an E10 16-shot anchor only after an explicit source-diff
impact audit establishes that the changed conditional pooled guard and
collector paths cannot affect that method's fitting, prediction, or metric
evaluation, and after checking the completed cell's config, split, evaluator
contract, support identity/content, query identity/content, and ground-truth
fingerprints. Check the RSG3 16-shot C0 reproduction against the stored
void-instrumentation cells and verify metric fingerprints across snapshots.
If source equivalence or any fingerprint match remains uncertain, rerun that
16-shot anchor under the E11 source snapshot and shared support manifest.
The planned curve has 20 method-shot points: 16 new lower-shot points (128
class runs) and four 16-shot anchors (32 class runs, reused only when audited).

Every method at shot count `N` uses the same
`configs/methods/VOID_support_{N}.json`. These eight-class manifests are exact
prefixes of `VOID_support_16.json` within each class (1 ⊂ 2 ⊂ 4 ⊂ 8 ⊂ 16).
Both `protocol.shots` and the manifest path change together; all other YAML
values are copied from the method's E10 config. Record the exact config and
manifest digests, selected support identities, and query/ground-truth
fingerprints in the run artifacts. Do not splice Table 1 shot rows into this
curve: their support choices differ.

## Metrics and display

- **Primary panel:** shot count 1, 2, 4, 8, 16 versus mean class
  `shift_1` AUPRO@0.05 (higher is better), with all eight classes and four
  methods. Show absolute values, including the E10 16-shot anchors.
- **Companion panel:** at each shot, `shift_1 − regular` AUPRO in percentage
  points for the same eight classes. Calculate each class's paired difference
  first, then average across classes. More negative values mean a larger drop.
- **Supplement:** analogous absolute curves for `shift_2` and `shift_3` on
  their respective seven-class scopes. For each shift-drop curve, restrict
  regular to exactly the classes present in that shift before subtraction;
  never subtract an eight-class regular macro from a seven-class shift macro.
- Report class-level values, scope, and the secondary segmentation
  pAUROC@0.05. The F1 oracle can remain diagnostic and does not support a
  deployment threshold claim.

Check matching query sets and metric contracts across shots and methods before
ranking. Interpret a higher absolute curve as an absolute performance lead
only at the observed shots. Claim greater robustness to decreasing support or
acquisition shift only if the appropriate paired contrasts support it; report
crossings and uncertainty instead of imposing a monotonic trend. E10's
predeclared 16-shot margin rule remains specific to E10 and is not silently
extended to the whole E11 curve.

Use the existing `scripts/experiments.py run` and `collect` workflow. A failed
cell is an execution failure, not a zero metric; complete or explicitly mark
missing cells before drawing a full curve. Stop after these four methods and
five shots unless a new research decision registers an extension.
