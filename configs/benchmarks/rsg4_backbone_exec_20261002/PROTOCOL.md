# RSG4 backbone execution and resolution screen

Date: 2026-10-02. Development/post-hoc evaluation only. Existing dsba3 GPU 0, 1, and 2 allocation; no remote launch is part of this protocol setup.

## Hypothesis and fixed conditions

DINOv2-L/14 attention execution and compilation may reduce the FP16 backbone cost while preserving the layer features RSG4 consumes. Removing resolution 896 may reduce cost after the selected execution path is checked. These are inference/runtime and resolution ablations; no backbone parameters are trained.

All configs derive from the prior `rsg4_cost_opt_20261002` FP16 full-query candidate. Keep seed 0, fixed support manifests, normal-only 16-shot AD2 or 1-shot MVTec/VisA support, DINOv2-L/14, layers [5,11,23], token cap 1024, four bags, FP16 autocast, model-device FP32 output features, query chunk 512, shared calibration/bag-distance options, fusion, postprocessing, noise guard, preprocessing and evaluator fixed. The full scale list is [448,672,896,1120]; the single ablation list is [448,672,1120]. The strict flash arm changes attention backend only. The compile arm changes compile mode only and retains the default backend. All comparison configs set the backend and compile mode explicitly.

Default attention is the current no-compile DINO path; report its resolved backend (including xFormers/SDPA choice). `sdpa_flash` is strict flash-only SDPA with no fallback. Compile uses default PyTorch Inductor, fullgraph=true and dynamic=false on the bound `get_intermediate_layers` callable; report lazy cold compile time separately from warmed inference. Compile failure is an execution result, not a reason to silently fall back. Record PyTorch/CUDA/GPU, resolved attention backend, compile status, first-call/cold and warm timings, feature shape/dtype, and config/support hashes.

## Stage 1: isolated backbone screen

On actual dataset images, measure the baseline, strict-flash and compiled-default variants on AD2 fabric, AD2 sheet_metal and VisA pcb1. Use identical decoded/preprocessed inputs and requested layers/scales within each class; record the input manifest/hash and repeat count. Do not use labels for input selection, fitting or predictions. Run one class at a time on its assigned existing GPU, with no GPU/resource changes. Sheet_metal is a profile-only stress case; do not automatically schedule its full-query evaluation. Compare per-image warmed latency and memory to baseline, while showing compile cold-start cost separately. A runtime error, unsupported flash kernel or nonfinite feature is recorded as execution failure.

After reviewing the isolated timings and finite feature outputs, choose at most one useful finite candidate for quality validation. Run matched full-query baseline and chosen candidate on AD2 fabric (16-shot), MVTec AD cable (1-shot) and VisA pcb1 (1-shot), with one worker per GPU and identical support/query/evaluation data. If no candidate is useful, still establish a fresh full-query baseline for these three scopes; validate strict flash only when the isolated feature output differs materially or needs full-query numerical validation. No full-query sheet_metal run is planned by default.

Primary quality metric is AUPRO@0.05, higher is better; pAUROC@0.05 and oracle F1 are secondary diagnostics. Use provisional macro decrease <=0.2 percentage points as the retention threshold and investigate any class decrease >1 pp. These small seen-development scopes do not establish statistical equivalence. Label all metrics and decisions post-hoc. The previous FP16 report is context, not a substitute for a matched remeasurement.

## Stage 2: remove resolution 896

Only after Stage 1 evidence selects an execution config, compare its full four-scale form against the paired no896 form on the same three full-query classes and exact data conditions. Existing configs include default/no-compile and compiled-default no896 variants; if strict flash is selected, create its paired no896 config by changing only the resolution list. If no execution candidate is selected, compare baseline against baseline-no896. Apply the same provisional AUPRO rule and report per-class changes, latency, fit/query peak memory, and cold compile overhead where relevant. Stop an arm on OOM, nonfinite output, support/config mismatch, or exception; record the failure and do not expand GPU allocation.

Labels and ground truth are used only by the existing evaluator. The support manifests contain normal training images. Test/test_public data are seen development/post-hoc populations, not untouched final evaluation. Maps and fitted weights remain off. Preserve logs and compact metrics/config/manifest provenance through the existing experiment runner and collector; do not update the cumulative ledger until verified runs are collected.

## Config index

Each scope has `baseline`, `flash`, `compile`, `no896`, and `compile_no896` YAMLs:

- `mvtec_ad2_fabric_s16`
- `mvtec_ad2_sheet_metal_s16`
- `mvtec_cable_s01`
- `visa_pcb1_s01`

Only fabric, cable and pcb1 are in the default full-query stages. Sheet_metal is limited to isolated backbone timing unless the Stage 1 evidence and available execution budget justify an explicit later decision.
