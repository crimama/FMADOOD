# FlowTTE Register Context Ablation

Scope: MVTec AD2 TESTpublic all 8 public objects, 16-shot fixed SuperAD-16 DINOv2 reference paths, DINOv3 ViT-L/16 patch features, no-TTE Flow-LatentBank, latent distance + 0.25 density penalty.

## Mean Results

| Method | Context | Weight | AUROC_0.05 | F1 | ΔAUROC vs no-context | ΔF1 vs no-context | Wins AUROC/F1 vs no-context |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flow-LatentBank DINOv3 fixed-ref no context | none | 0.0 | 0.800727 | 0.437437 | 0.000000 | 0.000000 | 0/0 |
| Context CLS w1 | cls | 1.0 | 0.801802 | 0.438730 | 0.001075 | 0.001293 | 4/4 |
| Context register w1 | register | 1.0 | 0.800963 | 0.437695 | 0.000237 | 0.000259 | 4/4 |
| Context CLS+register w1 | cls_register | 1.0 | 0.801538 | 0.438395 | 0.000811 | 0.000958 | 4/4 |
| Context CLS w5 | cls | 5.0 | 0.805848 | 0.443443 | 0.005121 | 0.006006 | 4/4 |
| Context register w5 | register | 5.0 | 0.801799 | 0.438466 | 0.001073 | 0.001029 | 4/5 |
| Context CLS+register w5 | cls_register | 5.0 | 0.804600 | 0.441841 | 0.003873 | 0.004405 | 4/4 |

## External Comparisons

Recorded SuperAD-16: AUROC_0.05=0.765802, F1=0.385534.
Reported SuperADD Table 1 context: AUROC_0.05=0.839300, F1=0.626112; same evaluator is not guaranteed, so use only as contextual comparison.

Best AUROC in this ablation: Context CLS w5 (0.805848, Δ vs no-context 0.005121).
Best F1 in this ablation: Context CLS w5 (0.443443, Δ vs no-context 0.006006).

## Artifacts

- `summary.json`
- `method_summary.tsv`
- `comparison_rows.tsv`
- Pulled run roots under `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_fixedref_ctx_*_20260707_v1`

## Verification

- Local: `python3 -m py_compile ...` passed.
- Local: `python3 -m pytest tests/test_flow_tte.py tests/test_mvtec_classic_adapter.py -q` passed with 22 tests.
- Remote: container compile passed; context scoring smoke passed.
- Remote: all six runs completed with three chunk `metrics.json` files each; final GPU memory returned to 1 MiB on slots 0,1,2.
