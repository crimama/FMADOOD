# FlowTTE CLS+Register w5 Non-Fixed Reference Diagnostic

Scope: MVTec AD2 single-image TESTpublic all 8 public objects, 16-shot, DINOv3 ViT-L/16 patch features, no-TTE Flow-LatentBank, `context_source=cls_register`, `context_weight=5.0`, `density_weight=0.25`. This run removes the fixed SuperAD-16 reference JSON and uses `support_selection=dinov3_cls_greedy_coreset`, seed 0.

## Motivation

Question: whether the CLS+register soft context penalty still helps when the candidate does not reuse the fixed DINOv2 SuperAD-16 reference set. This isolates the reference-policy effect from the earlier fixed-reference context ablation.

Negative-evidence intake: this remains a passive feature/memory diagnostic, so it is not promoted as a strict method claim. It tests a small reference-policy ablation of the register-conditioned retrieval idea, not a new paper-level result.

## Design

- Target: MVTec AD2 single-image, full `test_public/good,bad` for `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`, `walnuts`, `sheet_metal`.
- Reference pool: all `train/good`; selected 16 per object with DINOv3 CLS greedy coreset seed 0.
- Backbone/features: DINOv3 ViT-L/16, layer mean `[5, 11, 17, 23]`, `layer_norm_mean`.
- Flow/scoring: 3 epochs, 2 coupling layers, lr `2e-4`, latent distance + `0.25` density penalty, context soft penalty with CLS+register weight 5.
- Comparability: same shot budget as SuperAD-16, but not the same reference image set. Strict SuperAD paper-aligned claim is therefore false.

## Mean Results

| Method | Reference policy | AUROC_0.05 | F1 | ΔAUROC vs target | ΔF1 vs target | Comparability |
|---|---|---:|---:|---:|---:|---|
| Flow-LatentBank CLS+register w5 | DINOv3 CLS greedy coreset seed0 | 0.800546 | 0.440939 | 0.000000 | 0.000000 | target |
| No-context DINOv3 | see method | 0.797743 | 0.437800 | 0.002803 | 0.003139 | same non-fixed reference policy |
| CLS+register w5 fixed-ref | see method | 0.804600 | 0.441841 | -0.004054 | -0.000902 | different reference image set |
| No-context fixed-ref | see method | 0.800727 | 0.437437 | -0.000180 | 0.003503 | different reference image set |
| Recorded SuperAD-16 | see method | 0.765802 | 0.385534 | 0.034744 | 0.055405 | context only; DINOv2 + fixed SuperAD ref |
| Reported SuperADD | reported only | 0.839300 | 0.626112 | -0.038754 | -0.185173 | context only |

## Per-Object Delta vs Non-Fixed No-Context

| Object | AUROC | F1 | ΔAUROC vs non-fixed no-context | ΔF1 vs non-fixed no-context |
|---|---:|---:|---:|---:|
| can | 0.673556 | 0.002654 | -0.002758 | -0.000309 |
| fabric | 0.777584 | 0.349861 | 0.022466 | 0.024514 |
| fruit_jelly | 0.813555 | 0.541275 | 0.000482 | 0.000186 |
| rice | 0.946091 | 0.695560 | -0.000016 | 0.000101 |
| vial | 0.707013 | 0.395139 | 0.001977 | 0.000672 |
| wallplugs | 0.851399 | 0.415932 | -0.001330 | -0.002841 |
| walnuts | 0.868256 | 0.698328 | 0.000565 | 0.001579 |
| sheet_metal | 0.766916 | 0.428767 | 0.001038 | 0.001214 |

## Interpretation

Against the directly comparable non-fixed DINOv3 no-context baseline, CLS+register w5 improves mean AUROC by 0.002803 and mean F1 by 0.003139. The gain is mainly from `fabric` (+0.022466 AUROC, +0.024514 F1), while `wallplugs` drops slightly.

Compared with the fixed-reference CLS+register w5 run, this non-fixed reference policy is lower by -0.004054 AUROC and -0.000902 F1. That suggests the earlier fixed-reference setting was not solely responsible for the context gain, but the exact selected reference set still affects the mean.

## Artifacts

- Local run root: `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_ctx_clsreg_w5_coreset_notte_dw025_20260707_v1`
- Remote run root: `/home/hunim/Volume/FMAD-OOD-remote/results_remote/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_ctx_clsreg_w5_coreset_notte_dw025_20260707_v1`
- Consolidated metrics: `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_ctx_clsreg_w5_coreset_notte_dw025_20260707_v1/metrics.json`
- Consolidated manifest: `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_ctx_clsreg_w5_coreset_notte_dw025_20260707_v1/run_manifest.json`
- Summary: `summary.json`
- Tables: `method_summary.tsv`, `per_object_delta.tsv`

## Verification

- Remote execution completed on dsba3 container `hun_fsad_tta_012`, CUDA slots 0,1,2.
- Three chunk `metrics.json` files were produced and merged into all-8 metrics.
- Three `cleanup_evidence.txt` files were produced; no `anomaly_maps/` directory remained remotely or after pull.
- Final remote GPU memory returned to 1 MiB on slots 0,1,2.

## Verdict

`CONTINUE_DIAGNOSTIC`: the non-fixed CLS+register w5 condition beats the directly comparable non-fixed no-context baseline, but it is not strict SuperAD-16 comparable because the reference image set differs. Next bounded diagnostic should compare `CLS w5` vs `CLS+register w5` under this same non-fixed DINOv3 coreset policy, since fixed-reference ablation previously showed `CLS w5` as the best context source.
