# Two-Band Protected-Residual DVT Preregistration

Date: 2026-08-09

Protocol ID: `two-band-protected-residual-dvt-all8-20260809`

Tier: mechanism diagnostic; not a method or SuperAD claim

## Negative Evidence Intake

The previous fixed low-pass DVT repaired small-defect top-1 coverage from `-.011532` to `+.001132`
against raw, but retained only `67.48%/71.71%` of current DVT's medium AP/top-5 gain. Correct low-pass
position beat an equal-energy shuffled null by `+.056823/+.063081`. This permits exactly one structural
continuation: isolate the validated low-frequency positional correction while carrying the query's
high-frequency residual through an untouched branch.

This is not an alpha, sigma, radius, score-fusion, threshold, scorer, or support-admission retune. It can
still fail through generic dimensionality, hard-null absorption, or anomaly absorption; the controls below
are mandatory.

## Dataset and Frozen Baseline Contract

- `target_dataset=MVTec AD2 single-image`
- local root: `/home/hun/Volume/DATA/mvtec_ad_2`
- remote root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal`
- population: full `test_public/good,bad`
- support: exact fixed SuperAD-selected 16 normal images per object
- backbone: DINOv3-H+ layers `[7,15,23,31]`, `layer_norm_mean`, resolution 672 except
  `sheet_metal=448`
- Flow/scorer: existing Baseline1 Phase-3 contract, 3 epochs, density weight 0, `raw_1nn`
- query-time update, condition metadata, filename/split signal, private split access: forbidden
- existing raw/current/low-pass artifacts are diagnostic anchors only. SuperAD fields remain `N/A` and
  `comparable=false`; no `KEEP` verdict is possible in this stage.

## Fixed Two-Band Transform

Let `G` be the already validated spatial Gaussian operator with `sigma=2.0`, `truncate=3.0`, reflect
padding. For feature map `x` and support positional artifact `a=mu_p-g`, define

```text
L(x) = G(x)
H(x) = x - G(x)
R(x; b) = concat(L(x) - b, H(x)) / sqrt(2)
```

The `1/sqrt(2)` factor is fixed for every two-band arm and is not tuned. The high-frequency branch never
receives DVT subtraction. The transform is fit from support only and is applied identically to support and
query before the existing Flow.

## Arms

| Arm | Low-band correction `b` | Purpose |
| --- | --- | --- |
| `B0_band_identity` | `0` | dimension/band-split hard baseline |
| `B1_band_position` | `G(a)` | candidate: correct low-frequency positional correction |
| `B2_band_position_shuffled` | `roll(G(a), H//2, W//2)` | equal-energy positional hard null |

Frozen external anchors: `raw`, `position_current`, and `position_lowpass` from the 2026-08-08 all-8
diagnostic. No additional arm or parameter sweep is allowed.

## Primary Metrics

- Small (`<=1%`): pixel AP, top-1% defect coverage, top-1% component hit.
- Medium (`1--5%`): pixel AP and top-5% defect coverage.
- Secondary: full AUROC, `seg_AUROC_0.05`, `seg_F1`, `seg_AUPRO_0.05`.
- All scale metrics use bad-image-only paired rows and the existing float16/tie-inclusive evaluator.

## Validity Gates

All must pass:

1. all eight objects, full map/GT coverage, finite nonconstant scores, no retained `anomaly_maps/`;
2. exact support basenames, evaluation rows, GT counts, preprocessing, Flow, and scorer custody across B0--B2;
3. output feature dimension exactly `2C` and fixed scale exactly `1/sqrt(2)` for every arm;
4. B1/B2 correction energy relative error `<=1e-6`, B2 roll nonzero;
5. B0 correction energy exactly zero, and high-branch transform identical across B0--B2 within `1e-6`;
6. no query update, no private/metadata/condition input, local/remote relevant-metric parity `<=1e-3`.

## Strict Diagnostic Gate

All must pass:

1. `B1-B0` medium AP and top-5 coverage each preserve at least 75% of the frozen
   `position_current-raw` gains (`+.075706/+.073207`);
2. `B1-B0` small top-1 coverage `>=0` and small AP `>=0`;
3. B1 small top-1 coverage is at least `+.0075` above frozen current DVT;
4. `B1-B2` medium AP and top-5 coverage each `>=+.01`;
5. `B1-B2` small top-1 coverage `>=+.005`;
6. `B1-B0` macro pAUROC and F1 are nonnegative;
7. no object has B1 medium AP delta `<-.03`, small top-1 delta `<-.02`, or standard F1 delta `<-.03`.

Passing yields `PROMISING_TWO_BAND_DVT_DIAGNOSTIC`, not `KEEP`.

## Continuation and Hard-Stop Gates

One more diagnostic is allowed only if:

- B1 preserves at least 60% of both frozen current medium gains relative to B0;
- `B1-B0` small top-1 coverage is nonnegative;
- B1 beats B2 on both medium AP and top-5 coverage by at least `+.005`;
- no validity or catastrophic no-harm failure occurs.

Then verdict is `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. Otherwise:

- B0 matches B1 and B2: `GENERIC_BAND_SPLIT / NO_CONTINUE`;
- B2 matches or beats B1 on both medium endpoints: `HARD_NULL_ABSORBED / NO_CONTINUE`;
- small/standard no-harm collapses: `ANOMALY_ABSORPTION / NO_CONTINUE`;
- any custody/numerical/access failure: `INVALID_TWO_BAND_DVT_DIAGNOSTIC`;
- all other failures: `KILL_FOR_CLAIM / NO_CONTINUE`.

## Execution and Storage

Implement only the three fixed modes, focused transform/aggregator tests, an arm-selectable runner, and a
compact all-8 summary. Run independent arms on local and dsba5 GPUs when idle, with at least one duplicated
arm for parity. Pull compact artifacts, record the frozen verdict, delete remote immutable snapshots and all
regenerable `anomaly_maps/` trees.
