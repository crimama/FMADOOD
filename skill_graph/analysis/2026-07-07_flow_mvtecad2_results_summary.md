# Flow-family MVTec AD2 Results Summary

Date: 2026-07-07

Scope: Flow-related MVTec AD2 experiments recorded in this workspace.

Primary metrics:

- `AUROC` below means pixel-level `seg_AUROC_0.05`.
- `F1` means best-threshold pixel segmentation F1.
- Values are decimals, not percent.

Baseline handling:

- `SuperAD-16` is the recorded same-evaluator baseline:
  `/home/hun/Volume/RESEARCH/FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`.
- `SuperADD` is the reported TESTpublic Table 1 context used in prior reports, not a same-evaluator rerun.
- Reduced `can,rice` rows compare against `can,rice` subset baseline context only. They are not strict claims because shot/support protocols differ.

## Main Comparison Table

| Method | Scope | Shot/support | Backbone | Score/expansion | AUROC | F1 | Delta AUROC vs SuperAD | Delta F1 vs SuperAD | Delta AUROC vs SuperADD | Delta F1 vs SuperADD |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy FlowTTE-LatentBank | can,rice | 4/first | dinov2_vitl14 | latent_distance*, exp=1.25, denW=0.25* | 0.765772 | 0.315589 | +0.008098 | -0.018121 | +0.027172 | -0.050961 |
| FlowTTE-LatentBank | can,rice | 4/first | dinov2_vitl14 | latent_distance, exp=1.25, denW=0.25 | 0.765758 | 0.315586 | +0.008084 | -0.018124 | +0.027158 | -0.050964 |
| FlowTTE-LatentBank | can,rice | 4/first | dinov2_vitl14_reg | latent_distance, exp=1.25, denW=0.25 | 0.771840 | 0.333811 | +0.014166 | +0.000101 | +0.033240 | -0.032739 |
| FlowTTE-NFScore static NLL | can,rice | 4/first | dinov2_vitl14 | nf_nll, exp=1.0, denW=0.0 | 0.674336 | 0.127093 | -0.083338 | -0.206617 | -0.064264 | -0.239457 |
| Flow-LatentBank no-TTE | all8 | 4/first | dinov2_vitl14_reg | latent_distance, exp=1.0, denW=0.25 | 0.733798 | 0.297527 | -0.032004 | -0.088007 | -0.105502 | -0.328585 |
| FlowTTE-LatentBank TTE | all8 | 4/first | dinov2_vitl14_reg | latent_distance, exp=1.25, denW=0.25 | 0.712458 | 0.266227 | -0.053344 | -0.119307 | -0.126842 | -0.359885 |
| FlowTTE-LatentBank SuperAD-budget | all8 | 16/DINO CLS coreset | dinov2_vitl14 | latent_distance, exp=1.25, denW=0.0 | 0.714929 | 0.303336 | -0.050873 | -0.082198 | -0.124371 | -0.322776 |

`*`: legacy manifest was incomplete for some fields; values are inferred from the run command/report and should be treated as historical context.

## Baseline Means

| Baseline | Scope | AUROC | F1 | Notes |
| --- | --- | ---: | ---: | --- |
| SuperAD-16 | can,rice subset | 0.757674 | 0.333710 | same evaluator, subset context only |
| SuperADD reported | can,rice subset | 0.738600 | 0.366550 | reported-context only |
| SuperAD-16 | all8 | 0.765802 | 0.385534 | same evaluator baseline |
| SuperADD reported | all8 | 0.839300 | 0.626113 | reported TESTpublic context |

## All8 Object-Level Comparison

| Object | no-TTE 4shot AUROC/F1 | TTE 4shot AUROC/F1 | TTE 16shot AUROC/F1 | SuperAD-16 AUROC/F1 | SuperADD reported AUROC/F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| can | 0.576989/0.000609 | 0.619259/0.001459 | 0.631038/0.005413 | 0.586950/0.001553 | 0.516100/0.000000 |
| fabric | 0.687677/0.243356 | 0.595938/0.157578 | 0.574536/0.142368 | 0.687853/0.275239 | 0.842000/0.937400 |
| fruit_jelly | 0.716786/0.289808 | 0.616367/0.163486 | 0.630797/0.195635 | 0.797842/0.453155 | 0.811900/0.546800 |
| rice | 0.926475/0.667107 | 0.924423/0.666288 | 0.889664/0.598177 | 0.928397/0.665867 | 0.961100/0.733100 |
| vial | 0.687931/0.337461 | 0.654681/0.279502 | 0.653310/0.280517 | 0.692650/0.368248 | 0.825900/0.647700 |
| wallplugs | 0.725154/0.072587 | 0.809637/0.177903 | 0.772769/0.174315 | 0.775900/0.199382 | 0.925300/0.791600 |
| walnuts | 0.829177/0.488845 | 0.790068/0.434725 | 0.885514/0.741347 | 0.883834/0.718787 | 0.900300/0.756900 |
| sheet_metal | 0.720194/0.280446 | 0.689287/0.248875 | 0.681807/0.288919 | 0.772990/0.402043 | 0.931800/0.595400 |

## Key Settings

| Family | NF training | Feature/backbone | Support | Score | Expansion |
| --- | --- | --- | --- | --- | --- |
| FlowTTE-LatentBank reduced | 3 epochs, 2 coupling layers, hidden x1, lr 2e-4, clamp 1.9, tailW 0.3, tail top 0.05, logdet 1e-3 | DINOv2 ViT-L/14 or ViT-L/14-reg, layers 5/11/17/23, layer_norm_mean | 4 first train/good images, identity | latent distance + density penalty, distanceW 1.0, densityW 0.25 | TTE reservoir, budget 1.25 |
| FlowTTE-NFScore static NLL | same quick NF config | DINOv2 ViT-L/14, layers 5/11/17/23 | 4 first train/good images, identity | raw NF NLL, distanceW 0.0, densityW 0.0 | no expansion, budget 1.0 |
| Flow-LatentBank no-TTE all8 | same quick NF config | DINOv2 ViT-L/14-reg, layers 5/11/17/23, layer_norm_mean | 4 first train/good images, identity | latent distance + density penalty, densityW 0.25 | no expansion, budget 1.0 |
| FlowTTE-LatentBank all8 4shot | same quick NF config | DINOv2 ViT-L/14-reg, layers 5/11/17/23, layer_norm_mean | 4 first train/good images, identity | latent distance + density penalty, densityW 0.25 | TTE reservoir, budget 1.25 |
| FlowTTE-LatentBank all8 16shot | same quick NF config | DINOv2 ViT-L/14, layers 5/11/17/23, layer_norm_mean | 16 DINO CLS greedy coreset images, identity | latent distance only, densityW 0.0 | TTE reservoir, budget 1.25 |

## Interpretation

1. `FlowTTE-NFScore static NLL` is clearly weak. Raw NLL scoring drops hard on reduced `can,rice` (`0.674336/0.127093`) and should not be scaled without calibration or adapter anchoring.
2. Register backbone helps the reduced latent-bank run: `dinov2_vitl14_reg` improves mean `can,rice` from `0.765758/0.315586` to `0.771840/0.333811`, but the gain is mostly from `rice`.
3. Removing TTE is the strongest Flow-family signal so far. On all8 4-shot, `Flow-LatentBank no-TTE` beats same-condition TTE by `+0.021340` AUROC and `+0.031300` F1, winning 6/8 objects on both metrics.
4. None of the Flow-family AD2 rows beat the all8 baselines. The best all8 Flow AUROC is no-TTE 4-shot at `0.733798`, still below SuperAD-16 by `-0.032004` and below reported SuperADD by `-0.105502`. The best all8 Flow F1 is 16-shot TTE at `0.303336`, still below SuperAD-16 by `-0.082198` and SuperADD by `-0.322776`.
5. The dominant failure mode remains localization F1 collapse and expansion-driven ranking degradation. Current TTE expansion helps `can`/`wallplugs` in the 4-shot all8 ablation but hurts the other six objects enough to lower the mean.

## Verdict

Current Flow-family AD2 status: no strict method claim.

Most useful retained branch: `Flow-LatentBank no-TTE` as the static latent-memory baseline.

Next constrained direction: only revisit TTE if the expansion is gated or class-conditional and is required to beat the static no-TTE all8 baseline while improving `can`/`wallplugs` without damaging the six objects where no-TTE already wins.
