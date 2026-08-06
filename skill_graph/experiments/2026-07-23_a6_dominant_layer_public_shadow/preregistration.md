# A6 and Dominant-L17 Public-Shadow Confirmation

Date frozen: 2026-07-23  
Protocol state: `DESIGN_FROZEN / NOT_RUN`  
Maximum verdict: `SUPPORTED_ON_PUBLIC_SHADOW`

## Motivation and boundary

The 2026-07-19 support-only mechanism audit identified A6—no DVT, Flow,
density weight zero—as the least shift-sensitive tested Flow arm. It also
found that fixed four-layer identity-feature fusion lost to the simultaneous
single-layer envelope, with layer 17 macro-dominant. This follow-up asks only
whether those two frozen signs transfer to original-resolution MVTec AD2
`test_public`.

The public split has already been inspected. Results are development
public-shadow evidence only. No layer, weight, threshold, scorer, object,
retry, or arm may be selected after the run. `test_private`, TTA, query
transformation/update, memory update, routing, and support expansion are
forbidden.

## Shared contract

- all eight MVTec AD2 objects;
- exact SuperAD-selected 16 normal references, seed 0;
- support-manifest SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`;
- frozen DINOv2 ViT-L/14, canonical resolution 672 except
  `sheet_metal=448`;
- identity-only support views, static memory, LOO standardization off;
- two coupling layers, hidden multiplier 1, three epochs, learning rate
  `2e-4`, clamp 1.9, tail weight 0.3, top ratio 0.05, log-det weight `1e-3`;
- no RGB guide, threshold calibration, or morphology;
- original-resolution standardized pixel pAUROC@0.05, raw oracle F1, and
  connected-component AU-PRO@0.05;
- complete map coverage, finite/nonconstant predictions, independent compact
  aggregation, and dense-map cleanup.

## Frozen arms

| Arm | Layers | DVT | Transform | Density | Purpose |
|---|---|---|---|---:|---|
| Basic/A0 | 5,11,17,23 normalized mean | position mean | Flow | 0.25 | operational anchor |
| A6 | 5,11,17,23 normalized mean | none | Flow | 0 | simplified candidate |
| A7 | 5,11,17,23 normalized mean | none | identity | 0 | Flow hard null |
| A6-L17 | 17 only | none | Flow | 0 | frozen dominant-layer repair |
| L17 | 17 only | none | identity | 0 | layer/Flow interaction control |

L17 is fixed from the completed support-only M1 assay before this public run.
No other single layer, layer subset, learned weight, late score fusion, or
object-specific selection is allowed. This prevents retreading the previously
failed layer-wise Flow and routing families.

## Contrasts and gates

All contrasts use 2,000 deterministic object bootstraps, seed 0. Material
effects are `0.005` pAUROC, `0.010` raw F1, or `0.010` AU-PRO. A supported
metric additionally requires a positive 95% bootstrap lower bound, the same
sign on at least six of eight objects, and no object collapse. Collapse is
below `-0.02` pAUROC/AU-PRO or `-0.03` raw F1.

1. **A6 simplification:** A6 minus Basic must have macro pAUROC and AU-PRO
   no worse than `-0.005`, raw F1 at least `+0.010`, and no collapse.
2. **Flow without DVT:** A6 minus A7 must support at least one primary metric
   under the material/sign/interval/no-collapse rule.
3. **Dominant-layer repair:** A6-L17 minus A6 must support at least one
   primary metric under the same rule. Failure kills this repair and retains
   A6; it cannot trigger another layer search.
4. **Interaction diagnostic:** L17 minus A7 is reported but cannot itself
   promote A6-L17.

Flow and dominant-layer promotion additionally require no collapse on any of
the three primary metrics, not only on the metric supplying positive evidence.
A6 can be promoted only when both its simplification gate and its A6-over-A7
Flow hard-null gate pass. A6-L17 can be selected only after A6 promotion and
its own repair gate.

The post-run adjudicator compares the selected detector with both same-run
SuperAD and RN-FMLK. Shot scaling is authorized only if all three macro
metrics are positive against both baselines, every metric wins on at least
five of eight objects, and no object crosses a collapse bound. Pulling the
evidence is not authorization; the runner's `authorize-shot-scaling` mode must
subsequently validate the local authorization marker.

Same-condition SuperAD-16 and RN-FMLK comparisons remain withheld unless the
separate exact-reference baseline contract completes. Neither a public-shadow
win nor an oracle F1 result authorizes tuning or a final superiority claim.

## Execution and custody

Use the fixed dsba3 container `hun_fsad_tta_012`, host GPUs 0--3, and keep all
arms for an object on the same GPU. Use a new empty result root and an isolated
source snapshot. Run Basic/SuperAD/RN-FMLK first, then the four additional
mechanism arms for each object. Aggregate only after all contracts pass; delete
dense maps only after metric coverage and compact artifacts are present.
