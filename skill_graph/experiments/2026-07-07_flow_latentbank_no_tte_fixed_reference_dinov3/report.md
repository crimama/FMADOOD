# Flow-LatentBank DINOv3 fixed-reference diagnostic

Date: 2026-07-07
Protocol: `fmad-experiment-protocol`
Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## Negative Evidence Intake

This experiment follows the previous DINOv3 positive gradient but isolates whether that gain came from DINOv3 support selection or from the DINOv3 feature backbone itself.

Known failure boundary remains: Flow-LatentBank is still a passive latent-memory method, and no RN-FMLK/hard-null dominance gate is present. Therefore this cannot produce a paper-level `KEEP` even if it exceeds recorded SuperAD-16 context.

## Motivation

Question: if DINOv3 uses the exact DINOv2 SuperAD-16 reference image set, does it keep the DINOv3 gain, or did the previous DINOv3 run benefit mostly from a different DINOv3 CLS coreset?

Claim: if fixed-reference DINOv3 stays close to DINOv3 coreset and far above DINOv2 same-reference, the measured gain is mostly backbone/feature quality, not support-selection luck.

## Implementable Design

Target dataset: MVTec AD2 single-image.
Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
Objects: all eight public objects.
Split: full `test_public/good,bad`.
Candidate reference policy: same 16 image paths selected by the previous DINOv2 SuperAD-16 coreset manifest.

Fixed reference manifest:
`skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`

Shared config: `score_mode=latent_distance`, `expansion_budget=1.0`, `density_weight=0.25`, `flow_epochs=3`, `coupling_layers=2`, `feature_fusion=layer_norm_mean`, support transforms `identity`.

## Evaluation Alignment

The candidate shares dataset, split, object set, evaluator, support budget, and exact reference image set with the DINOv2 SuperAD-16 reference set. It deliberately changes the feature backbone to `dinov3_vitl16`, so strict SuperAD Table 1 comparability remains false.

SuperADD remains reported context only, not a same-evaluator rerun.

## Code Modification / Creation

Added `fixed_json=<path>` support selection in `scripts/flow_tte_support.py`, with manifest-aware path membership checks. Updated `scripts/run_flow_tte_mvtec_ad2.py` so fixed-reference runs record `reference_budget_matched=true`.

Tests added in `tests/test_mvtec_classic_adapter.py` for fixed manifest ordering and rejection of paths outside train/good.

## Added Code Evaluation

Local verification:

- `python3 -m pytest tests/test_mvtec_classic_adapter.py tests/test_dinov3_backbone.py -q`: 17 passed.
- `python3 -m py_compile scripts/dinov3_backbone.py scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_support.py scripts/flow_tte_mvtec_ad2_core.py`: passed.
- Remote preflight verified all 8 objects found exactly 16 fixed paths in full train/good.

`ruff` and `basedpyright` are not installed in the local Python environment.

## Remote Execution

Remote host: dsba3.
Container: `hun_fsad_tta_012`.
Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`.

Object chunks:

| CUDA slot | Objects |
| ---: | --- |
| 0 | `can,fabric,fruit_jelly` |
| 1 | `rice,vial,wallplugs` |
| 2 | `walnuts,sheet_metal` |

Remote root:
`/workspace/results_remote/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_fixed_dinov2ref_notte_dw025_20260707_v1`

Local pullback:
`results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_fixed_dinov2ref_notte_dw025_20260707_v1`

## SuperAD Baseline and Unified Metrics

SuperAD-16 source:
`/home/hun/Volume/RESEARCH/FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`

| Method / baseline | Reference set | Backbone | Mean AUROC_0.05 | Mean F1 | Delta vs fixed AUROC | Delta vs fixed F1 | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Flow-LatentBank no-TTE fixed-ref | DINOv2 SuperAD-16 refs | dinov3_vitl16 | 0.800727 | 0.437437 | 0.000000 | 0.000000 | candidate |
| Flow-LatentBank no-TTE DINOv3 coreset | DINOv3 CLS coreset | dinov3_vitl16 | 0.797743 | 0.437800 | -0.002983 | +0.000363 | support-selection control |
| Flow-LatentBank no-TTE DINOv2 | DINOv2 SuperAD-16 refs | dinov2_vitl14 | 0.761782 | 0.350391 | -0.038945 | -0.087046 | backbone control |
| SuperAD-16 recorded | DINOv2 SuperAD-16 refs | dinov2_vitl14 | 0.765802 | 0.385534 | -0.034925 | -0.051902 | same evaluator, different method/backbone |
| SuperADD reported | paper Table 1 | not rerun | 0.839300 | 0.626112 | +0.038573 | +0.188676 | reported context |


Object-level AUROC/F1:

| Object | Fixed-ref DINOv3 | DINOv3 coreset | DINOv2 same ref | SuperAD-16 | SuperADD reported |
| --- | ---: | ---: | ---: | ---: | ---: |
| can | 0.666953/0.006763 | 0.676314/0.002963 | 0.647997/0.001661 | 0.586950/0.001553 | 0.516100/0.000000 |
| fabric | 0.779529/0.336694 | 0.755118/0.325347 | 0.696520/0.275325 | 0.687853/0.275239 | 0.842000/0.937400 |
| fruit_jelly | 0.813731/0.536979 | 0.813073/0.541089 | 0.741353/0.321203 | 0.797842/0.453155 | 0.811900/0.546800 |
| rice | 0.944476/0.692469 | 0.946106/0.695460 | 0.915980/0.656086 | 0.928397/0.665867 | 0.961100/0.733100 |
| vial | 0.708740/0.397371 | 0.705036/0.394468 | 0.692449/0.361069 | 0.692650/0.368248 | 0.825900/0.647700 |
| wallplugs | 0.849001/0.411664 | 0.852729/0.418772 | 0.756382/0.105793 | 0.775900/0.199382 | 0.925300/0.791600 |
| walnuts | 0.878573/0.690281 | 0.867691/0.696749 | 0.886771/0.735338 | 0.883834/0.718787 | 0.900300/0.756900 |
| sheet_metal | 0.764809/0.427273 | 0.765878/0.427553 | 0.756804/0.346652 | 0.772990/0.402043 | 0.931800/0.595400 |

## Results and Analysis

Fixed-reference DINOv3 scored `0.800727` AUROC and `0.437437` F1.

Against DINOv3 CLS coreset, fixed-reference changed by `+0.002983` AUROC and `-0.000363` F1. This is effectively a tie: same-reference slightly helps AUROC, while DINOv3 coreset slightly helps F1.

Against DINOv2 on the same DINOv2 reference set, fixed-reference DINOv3 improves by `+0.038945` AUROC and `+0.087046` F1. This supports the interpretation that the main gain is DINOv3 feature quality rather than support selection.

Against recorded SuperAD-16 context, fixed-reference DINOv3 is higher by `+0.034925` AUROC and `+0.051902` F1, but this is not a strict method claim because the backbone and method differ and RN-FMLK/hard-null gates are absent.

Against reported SuperADD, fixed-reference DINOv3 remains lower by `-0.038573` AUROC and `-0.188676` F1.

## Continuation Assessment

Strict method claim now: no.

Small next diagnostic justified: yes, but not another support-selection sweep. The support-control result says DINOv3 is the useful lever; the remaining gap to SuperADD is likely from SuperADD-style preprocessing/postprocessing/scoring rather than reference-set choice.

## Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

This experiment separates backbone from support selection. DINOv3 retains its improvement when forced to use the exact DINOv2 SuperAD-16 reference image set, so the prior DINOv3 gain is not mainly a coreset artifact. However, the method remains below reported SuperADD, especially in F1, so it is not claim-ready.

Next constrained experiment: keep the fixed DINOv2 SuperAD-16 reference set and test one SuperADD-aligned localization component at a time, starting with preprocessing/masking or post-threshold morphology, not another support policy.

## Post-Conclusion Storage Cleanup

The remote run used `--cleanup-maps`; all chunk manifests recorded `cleanup_anomaly_maps=true`.

Cleanup evidence:

- Local `anomaly_maps/` count under the pulled result root: `0`.
- Remote `anomaly_maps/` count under the completed result root: `0`.
